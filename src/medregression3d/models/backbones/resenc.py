import warnings

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist

from medregression3d.training.trainer import BaseModel
from medregression3d.models.heads.regression import RegressionHead
from medregression3d.models.heads.ordinal_regression import OrdinalRegressionHead, OrdinalRegressionHead_MLP


def get_first_valid_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of the specified keys found: {keys}")



def compute_stages_and_strides(patch_size, n_stages=None, min_feature_map_size=4):
    """Decide encoder depth and per-stage, per-axis strides from the patch size.

    Stage 1 is always the stem (stride 1). A later stage downsamples an axis by 2
    only while that axis is even AND would stay >= ``min_feature_map_size``; this
    guarantees every downsampling divides cleanly, so no input padding is needed.

    Args:
        patch_size: spatial patch size, e.g. ``[160, 160, 160]``.
        n_stages: if ``None`` -> AUTO: stem ``[1, 1, 1]`` then uniform ``[2, 2, 2]``
            downsampling, adding stages until the most-constrained axis can no
            longer be halved. If an int -> exactly ``n_stages - 1`` downsampling
            stages, with the stride on any exhausted axis dropping to 1 so the
            rest keep going (e.g. ``[2, 2, 1]`` once the z-axis runs out).
        min_feature_map_size: smallest size an axis may be halved down to.

    Returns:
        ``(n_stages, strides)`` where ``strides`` has length ``n_stages`` and each
        entry is a per-axis list of ints; ``strides[0]`` is always all ones.
    """
    sizes = [int(s) for s in patch_size]
    dim = len(sizes)

    def can_halve(s):
        return s % 2 == 0 and s // 2 >= min_feature_map_size

    strides = [[1] * dim]  # stem stage never downsamples

    if n_stages is None:
        # AUTO: isotropic [2, 2, 2], limited by the most-constrained axis.
        while all(can_halve(s) for s in sizes):
            strides.append([2] * dim)
            sizes = [s // 2 for s in sizes]
        n_stages = len(strides)
    else:
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1, got {n_stages}")
        for _ in range(n_stages - 1):
            stride = [2 if can_halve(s) else 1 for s in sizes]
            sizes = [s // st for s, st in zip(sizes, stride)]
            strides.append(stride)
        if any(all(st == 1 for st in stage) for stage in strides[1:]):
            warnings.warn(
                f"n_stages={n_stages} exceeds what patch_size={list(patch_size)} "
                f"can downsample; some stages keep stride 1 (no downsampling). "
                f"Strides: {strides}"
            )

    return n_stages, strides


# Original nnSSL ResEnc-L topology. Must stay fixed to load pretrained weights.
_RESENC_L_STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_RESENC_L_FEATURES = [32, 64, 128, 256, 320, 320]
_RESENC_L_BLOCKS = [1, 3, 4, 6, 6, 6]


class ResEncoder(Module):
    def __init__(
        self,
        **hypparams,
    ):
        super(ResEncoder, self).__init__()

        patch_size = hypparams["input_shape"]
        n_stages_cfg = hypparams.get("n_stages", None)
        pretrained = hypparams.get("pretrained", False)

        if pretrained:
            # The nnSSL ResEnc-L checkpoint only matches the original 6-stage
            # architecture; refuse to silently build something incompatible.
            if n_stages_cfg is not None and n_stages_cfg != 6:
                raise ValueError(
                    f"pretrained=True requires the original 6-stage ResEnc-L "
                    f"architecture, but n_stages={n_stages_cfg} was requested. Set "
                    f"n_stages to 6 or null, or set pretrained=False to use a "
                    f"custom number of stages."
                )
            n_stages = 6
            strides = _RESENC_L_STRIDES
            features_per_stage = _RESENC_L_FEATURES
            n_blocks_per_stage = _RESENC_L_BLOCKS
        else:
            n_stages, strides = compute_stages_and_strides(patch_size, n_stages_cfg)
            # Channels: double from 32, capped at 320 (nnU-Net ResEnc default).
            features_per_stage = [min(32 * 2 ** i, 320) for i in range(n_stages)]
            # Blocks: reuse the ResEnc-L pattern, truncated or extended to depth.
            if n_stages <= len(_RESENC_L_BLOCKS):
                n_blocks_per_stage = _RESENC_L_BLOCKS[:n_stages]
            else:
                n_blocks_per_stage = _RESENC_L_BLOCKS + [6] * (
                    n_stages - len(_RESENC_L_BLOCKS)
                )

        dim = len(strides[0])
        kernel_sizes = [[3] * dim for _ in range(n_stages)]

        # Resolved geometry, stored for logging / reproducibility (esp. so the
        # actual depth is recoverable when n_stages was auto-resolved from null).
        self.n_stages = n_stages
        self.strides = strides
        self.features_per_stage = features_per_stage

        # Last-stage channel count; consumed by the regression / ordinal heads.
        self.output_channels = features_per_stage[-1]

        # Echo resolved geometry to stdout so it lands in the training .log
        # (model_architecture.txt holds the same info; this keeps it inline).
        print(
            f"[ResEncoder] resolved geometry: n_stages={n_stages} | "
            f"strides={strides} | features_per_stage={features_per_stage} | "
            f"embed_dim={self.output_channels}",
            flush=True,
        )

        self.res_unet = ResidualEncoderUNet(
            hypparams["input_channels"],
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=[1] * (n_stages - 1),
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            num_classes=hypparams["num_classes"],
        )
        self.res_unet.encoder.return_skips = False

        if hypparams["pretrained"]:
            self.res_unet = load_pretrained_weights(
                self.res_unet,
                hypparams["chpt_path"],
            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for n, param in self.res_unet.named_parameters():
                    param.requires_grad = False

    def forward(self, x):

        x = self.res_unet.encoder(x).mean(dim=[2, 3, 4])

        return x


class ResEncoder_Regressor(BaseModel):
    """ResEncoder backbone with a plain regression head.

    Use with ``task: 'Regression'``. By default the head emits a single scalar
    per sample (output shape ``[B]``); set ``num_outputs`` in the model config
    for multi-output regression.
    """

    def __init__(self, **hypparams):
        super().__init__(**hypparams)

        self.encoder = ResEncoder(**hypparams)

        self.reg_head = RegressionHead(
            embed_dim=self.encoder.output_channels,
            num_outputs=hypparams.get("num_outputs", 1),
            dropout=hypparams.get("regression_head_dropout", 0.1),
            patch_aggregation_method=hypparams.get("token_aggregation_method", "avg"),
            cls_token_available=False,
        )

        # Optionally restore reg_head weights from a checkpoint that was saved
        # with the same head shape.
        if hypparams.get("pretrained", False):
            ckpt = torch.load(hypparams["chpt_path"], map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            for name, param in state_dict.items():
                if name.startswith("reg_head") and name in self.state_dict():
                    if self.state_dict()[name].shape == param.shape:
                        self.state_dict()[name].copy_(param)

    def forward(self, x):
        x = self.encoder(x)
        return self.reg_head(x)


class ResEncoder_OrdinalRegressor(BaseModel):
    def __init__(self, **hypparams):
        super().__init__(**hypparams)

        self.encoder = ResEncoder(**hypparams)

        # Number of ordinal thresholds is (num_classes - 1)
        self.reg_head = OrdinalRegressionHead(
            embed_dim=self.encoder.output_channels,
            num_classes=hypparams["num_classes"],
            dropout=hypparams.get("regression_head_dropout", 0.1),
            patch_aggregation_method=hypparams.get("token_aggregation_method", "avg"),
            cls_token_available=False,
        )

        # Only load reg_head if weights are available
        if hypparams.get("pretrained", False):
            ckpt = torch.load(hypparams["chpt_path"], map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            for name, param in state_dict.items():
                if name.startswith("reg_head") and name in self.state_dict():
                    if self.state_dict()[name].shape == param.shape:
                        self.state_dict()[name].copy_(param)

    def forward(self, x):
        x = self.encoder(x)
        logits, probas = self.reg_head(x)
        return logits, probas


class ResEncoder_OrdinalRegressor_MLP(BaseModel):
    def __init__(self, **hypparams):
        super().__init__(**hypparams)

        self.encoder = ResEncoder(**hypparams)

        # Number of ordinal thresholds is (num_classes - 1)
        self.reg_head = OrdinalRegressionHead_MLP(
            embed_dim=self.encoder.output_channels,
            num_classes=hypparams["num_classes"],
            dropout=hypparams.get("regression_head_dropout", 0.1),
            patch_aggregation_method=hypparams.get("token_aggregation_method", "avg"),
            cls_token_available=False,
        )

    def forward(self, x):
        x = self.encoder(x)
        logits, probas = self.reg_head(x)
        return logits, probas


def load_pretrained_weights(
    resenc_model,
    pretrained_weights_file,
):
    if dist.is_initialized():
        saved_model = torch.load(
            pretrained_weights_file,
            map_location=torch.device("cuda", dist.get_rank()),
            weights_only=False,
        )
    else:
        saved_model = torch.load(pretrained_weights_file, weights_only=False)
    if 'network_weights' in saved_model:
        pretrained_dict = saved_model['network_weights']
    elif 'state_dict' in saved_model:
        pretrained_dict = saved_model['state_dict']
    else:
        raise KeyError("No compatible weight dictionary ('network_weights' or 'state_dict') found in checkpoint")


    if isinstance(resenc_model, DDP):
        mod = resenc_model.module
    else:
        mod = resenc_model
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    in_conv_weights_model = get_first_valid_key(model_dict, [
        "encoder.stem.convs.0.all_modules.0.weight",
        "encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight"
    ])

    in_conv_weights_pretrained = get_first_valid_key(pretrained_dict, [
        "encoder.stem.convs.0.all_modules.0.weight",
        "encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight"
    ])


    in_channels_model = in_conv_weights_model.shape[1]
    in_channels_pretrained = in_conv_weights_pretrained.shape[1]

    if in_channels_model != in_channels_pretrained:
        assert in_channels_pretrained == 1, (
            f"The input channels do not match. Pretrained model: {in_channels_pretrained}; your network: "
            f"your network: {in_channels_model}"
        )

        repeated_weight_tensor = in_conv_weights_pretrained.repeat(
            1, in_channels_model, 1, 1, 1) / in_channels_model
        target_data_ptr = in_conv_weights_pretrained.data_ptr()
        for key, weights in pretrained_dict.items():
            if weights.data_ptr() == target_data_ptr:
                pretrained_dict[key] = repeated_weight_tensor

        # SPECIAL CASE HARDCODE INCOMING
        # Normally, these keys have the same data_ptr that points to the weights that are to be replicated:
        # - encoder.stem.convs.0.conv.weight
        # - encoder.stem.convs.0.all_modules.0.weight
        # - decoder.encoder.stem.convs.0.conv.weight
        # - decoder.encoder.stem.convs.0.all_modules.0.weight
        # But this is not the case for 'VariableSparkMAETrainer_BS8', where we replace modules from the original
        # encoder architecture, so that the following two point to a different tensor:
        # - encoder.stem.convs.0.conv.weight
        # - decoder.encoder.stem.convs.0.conv.weight
        # resulting in a shape mismatch for the two missing keys in the check below.
        # It is important to note, that the weights being trained are located at 'all_modules.0.weight', so we
        # have to use those as the source of replication
        if "VariableSparkMAETrainer" in pretrained_weights_file:
            pretrained_dict["encoder.stem.convs.0.conv.weight"] = repeated_weight_tensor
            pretrained_dict["decoder.encoder.stem.convs.0.conv.weight"] = (
                repeated_weight_tensor
            )

        print(
            f"Your network has {in_channels_model} input channels. To accommodate for this, the single input "
            f"channel of the pretrained model is repeated {in_channels_model} times."
        )

    skip_strings_in_pretrained = [".seg_layers."]
    skip_strings_in_pretrained.extend(["decoder.stages", "decoder.transpconvs"])

    final_pretrained_dict = {}
    for key, v in pretrained_dict.items():
        if key in model_dict and all(
            [i not in key for i in skip_strings_in_pretrained]
        ):
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model "
                f"does not seem to be compatible with your network."
            )
            final_pretrained_dict[key] = v

    model_dict.update(final_pretrained_dict)

    mod.load_state_dict(model_dict)

    return mod