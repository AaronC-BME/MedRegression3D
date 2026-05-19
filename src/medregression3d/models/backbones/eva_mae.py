import os.path

from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    PatchEmbed,
)
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist
from einops import rearrange
from peft import get_peft_model, LoraConfig, TaskType
from base_model import BaseModel
from models.classification_head import ClassificationHead
from models.ordinal_regression_head import OrdinalRegressionHead
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile


class EvaEncoder(Module):
    def __init__(
        self,
        embed_dim,
        patch_embed_size,
        input_shape,
        depth,
        num_heads,
        num_reg_tokens,
        use_rot_pos_emb,
        use_abs_pos_emb,
        mlp_ratio,
        drop_path_rate,
        drop_path_scale,
        patch_drop_rate,
        proj_drop_rate,
        attn_drop_rate,
        rope_kwargs,
        chpt_path,
        **hypparams,
    ):
        super(EvaEncoder, self).__init__()

        self.eva = Eva(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ref_feat_shape=tuple(
                [
                    i // ds for i, ds in zip(input_shape, patch_embed_size)
                ]  # input_shape=patch size, patch_embed_size=(8,8,8)
            ),
            num_reg_tokens=num_reg_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_kwargs=rope_kwargs,
            # class_token=True,
        )
        self.down_projection = PatchEmbed(
            patch_embed_size, hypparams["input_channels"], embed_dim
        )

    def forward(self, x):
        x = self.down_projection(x)
        B, C, W, H, D = x.shape
        x = rearrange(x, "b c w h d -> b (h w d) c")
        x, _ = self.eva(x)

        return x


class Eva_MAE(BaseModel):
    def __init__(
        self,
        embed_dim,
        patch_embed_size,
        input_shape,
        depth,
        num_heads,
        num_reg_tokens,
        use_rot_pos_emb,
        use_abs_pos_emb,
        mlp_ratio,
        drop_path_rate,
        drop_path_scale,
        patch_drop_rate,
        proj_drop_rate,
        attn_drop_rate,
        rope_kwargs,
        chpt_path,
        **hypparams,
    ):
        super(Eva_MAE, self).__init__(**hypparams)

        self.eva_encoder = EvaEncoder(
            embed_dim,
            patch_embed_size,
            input_shape,
            depth,
            num_heads,
            num_reg_tokens,
            use_rot_pos_emb,
            use_abs_pos_emb,
            mlp_ratio,
            drop_path_rate,
            drop_path_scale,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            rope_kwargs,
            chpt_path,
            **hypparams,
        )

        if self.pretrained:
            self.eva_encoder = load_pretrained_weights(
                self.eva_encoder,
                chpt_path,
                handle_input_shape_mismatch=hypparams[
                    "pretraining_input_shape_mismatch"
                ],
                load_cls_token=hypparams["load_cls_token"],
                input_shape=input_shape,
                patch_embed_size=patch_embed_size

            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for n, param in self.eva_encoder.named_parameters():

                    if not hypparams["load_cls_token"] and "cls_token" in n:
                        param.requires_grad = True  # make cls_token trainable if it's not loaded from pretrained weights
                    else:
                        param.requires_grad = False

            elif hypparams["finetune_method"] == "lora":
                # Apply LoRA to attention layers

                lora_config = LoraConfig(
                    # task_type=TaskType.IMAGE_CLASSIFICATION,
                    r=8,  # LoRA rank
                    lora_alpha=32,  # Scaling factor
                    lora_dropout=0.1,
                    target_modules=["attn.qkv", "attn.proj"],
                )

                self.eva_encoder.eva = get_peft_model(self.eva_encoder.eva, lora_config)

                # Freeze all layers except LoRA-adapted ones
                for param in self.eva_encoder.parameters():
                    param.requires_grad = False

                for name, param in self.eva_encoder.eva.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True

            else:
                raise NotImplementedError

        self.cls_head = ClassificationHead(
            embed_dim,
            hypparams["num_classes"],
            dropout=hypparams["classification_head_dropout"],
            patch_aggregation_method=hypparams["token_aggregation_method"],
            cls_token_available=hypparams["cls_token_available"],
        )

    def forward(self, x):
        x = self.eva_encoder(x)
        x = self.cls_head(x)

        return x

class Eva_MAE_OrdinalRegressor(BaseModel):
    def __init__(
        self,
        embed_dim,
        patch_embed_size,
        input_shape,
        depth,
        num_heads,
        num_reg_tokens,
        use_rot_pos_emb,
        use_abs_pos_emb,
        mlp_ratio,
        drop_path_rate,
        drop_path_scale,
        patch_drop_rate,
        proj_drop_rate,
        attn_drop_rate,
        rope_kwargs,
        chpt_path,
        **hypparams,
    ):
        super(Eva_MAE_OrdinalRegressor, self).__init__(**hypparams)

        self.eva_encoder = EvaEncoder(
            embed_dim,
            patch_embed_size,
            input_shape,
            depth,
            num_heads,
            num_reg_tokens,
            use_rot_pos_emb,
            use_abs_pos_emb,
            mlp_ratio,
            drop_path_rate,
            drop_path_scale,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            rope_kwargs,
            chpt_path,
            **hypparams,
        )

        if self.pretrained:
            self.eva_encoder = load_pretrained_weights(
                self.eva_encoder,
                chpt_path,
                handle_input_shape_mismatch=hypparams[
                    "pretraining_input_shape_mismatch"
                ],
                load_cls_token=hypparams["load_cls_token"],
                input_shape=input_shape,
                patch_embed_size=patch_embed_size

            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for n, param in self.eva_encoder.named_parameters():

                    if not hypparams["load_cls_token"] and "cls_token" in n:
                        param.requires_grad = True  # make cls_token trainable if it's not loaded from pretrained weights
                    else:
                        param.requires_grad = False

            elif hypparams["finetune_method"] == "lora":
                # Apply LoRA to attention layers

                lora_config = LoraConfig(
                    # task_type=TaskType.IMAGE_CLASSIFICATION,
                    r=8,  # LoRA rank
                    lora_alpha=32,  # Scaling factor
                    lora_dropout=0.1,
                    target_modules=["attn.qkv", "attn.proj"],
                )

                self.eva_encoder.eva = get_peft_model(self.eva_encoder.eva, lora_config)

                # Freeze all layers except LoRA-adapted ones
                for param in self.eva_encoder.parameters():
                    param.requires_grad = False

                for name, param in self.eva_encoder.eva.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True

            else:
                raise NotImplementedError
            
        with torch.no_grad():
            dummy_input = torch.randn(1, hypparams["input_channels"], *input_shape).to("cuda")
            dummy_features = self.eva_encoder(dummy_input)
            actual_embed_dim = dummy_features.shape[-1]

        self.reg_head = OrdinalRegressionHead(
            embed_dim=actual_embed_dim,
            num_classes=hypparams["num_classes"],
            dropout=hypparams.get("classification_head_dropout", 0.1),
            patch_aggregation_method=hypparams.get("token_aggregation_method", "avg"),
            cls_token_available=False,
        )

    def forward(self, x):
        x = self.eva_encoder(x)
        logits, probas = self.reg_head(x)

        return logits, probas


def load_pretrained_weights(
    eva_model,
    pretrained_weights_file,
    handle_input_shape_mismatch="interpolate",
    load_cls_token=True,
    verbose=True,
    input_shape=None,
    patch_embed_size=None

):

    # Load weights
    if dist.is_initialized():
        saved_model = torch.load(
            pretrained_weights_file,
            map_location=torch.device("cuda", dist.get_rank()),
            weights_only=False,
        )
    else:
        saved_model = torch.load(pretrained_weights_file, weights_only=False)

    pretrained_dict = saved_model["network_weights"]
    pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items()}

    if isinstance(eva_model, DDP):
        mod = eva_model.module
        mod.to('cuda')
    else:
        mod = eva_model
        mod.to('cuda')

    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    # Initialize model state dictionary
    model_dict = mod.state_dict()

    in_conv_weights_model: torch.Tensor = model_dict[
        "down_projection.proj.weight"
    ]
    in_conv_weights_pretrained: torch.Tensor = pretrained_dict[
        "down_projection.proj.weight"
    ]

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
                # print(key)
                pretrained_dict[key] = repeated_weight_tensor


    if isfile(join(os.sep.join(pretrained_weights_file.split(os.sep)[:-2]), 'plans.json')):
        pretrained_config = load_json(join(os.sep.join(pretrained_weights_file.split(os.sep)[:-2]), 'plans.json'))['configurations']['3d_fullres']
        pretrained_input_image_patch_size = pretrained_config['patch_size']
    else:
        print('############ no plans file found. Assuming pretraining patch size [160 160 160]###############')
        pretrained_input_image_patch_size = [160, 160, 160]




    # adjust pos_embed if necessary
    handle_pos_embed_resize(pretrained_dict=pretrained_dict,
                            model_dict=model_dict,
                            mode=handle_input_shape_mismatch,
                            input_shape=input_shape,  # Only needed for trilinear
                            pretrained_input_patch_size=pretrained_input_image_patch_size,
                            patch_embed_size=patch_embed_size
                            )




    # Filter out unnecessary keys based on match_encoder_only flag
    skip_strings_in_pretrained = [".seg_layers."]
    skip_strings_in_pretrained.append(".decoder.")
    skip_strings_in_pretrained.append("up_projection")

    if not load_cls_token or 'eva.cls_token' not in pretrained_dict:
        skip_strings_in_pretrained.append("eva.cls_token")


    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be "
                f"compatible with your network."
            )
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model "
                f"does not seem to be compatible with your network."
            )

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict.keys()
        and all([i not in k for i in skip_strings_in_pretrained])
    }

    model_dict.update(pretrained_dict)

    print(
        "################### Loading pretrained weights from file ",
        pretrained_weights_file,
        "###################",
    )
    if verbose:
        print(
            "Below is the list of overlapping blocks in pretrained model and nnUNet architecture:"
        )
        for key, value in pretrained_dict.items():
            print(key, "shape", value.shape)
        print("################### Done ###################")

    mod.load_state_dict(model_dict)

    return mod

def interpolate_patch_embed_1d(patch_embed, target_len, mode="linear"):
    """Resizes patch embeddings using interpolation."""
    return F.interpolate(
        patch_embed.permute(0, 2, 1),  # [B, C, Tokens]
        size=target_len,
        mode=mode,
        align_corners=False,
    ).permute(0, 2, 1)  # [B, Tokens, C]

def interpolate_patch_embed_3d(patch_embed, in_shape, out_shape):
    """Resizes patch embeddings using 3D trilinear interpolation."""
    patch_embed = patch_embed.permute(0, 2, 1)
    patch_embed = rearrange(patch_embed, "B C (x y z) -> B C x y z", **in_shape)
    patch_embed = F.interpolate(patch_embed, size=list(out_shape.values()), mode="trilinear", align_corners=False)
    patch_embed = rearrange(patch_embed, "B C x y z -> B C (x y z)", **out_shape)
    return patch_embed.permute(0, 2, 1)


def handle_pos_embed_resize(pretrained_dict, model_dict, mode, input_shape=None, pretrained_input_patch_size=None, patch_embed_size=None):
    pretrained_pos_embed = pretrained_dict["eva.pos_embed"]
    model_pos_embed = model_dict["eva.pos_embed"]
    model_pos_embed_shape = model_pos_embed.shape

    # for key, value in pretrained_dict.items():
    #     print(f"{key}: {value.shape}")

    has_cls_token = "eva.cls_token" in pretrained_dict


    if has_cls_token:
        cls_pos_embed = pretrained_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed[:, 1:, :]
    else:
        if  "eva.cls_token" in model_dict.keys():
            cls_pos_embed = model_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed

    if mode == "interpolate":
        resized_patch_pos_embed = interpolate_patch_embed_1d(patch_pos_embed, target_len=model_pos_embed_shape[1] - int(has_cls_token))

    elif mode == "interpolate_trilinear":
        # Calculate input/output 3D shapes
        in_shape = dict(zip("xyz", [int(d / p) for d, p in zip(pretrained_input_patch_size, patch_embed_size)]))
        out_shape = dict(zip("xyz", [int(d / p) for d, p in zip(input_shape, patch_embed_size)]))
        resized_patch_pos_embed = interpolate_patch_embed_3d(patch_pos_embed, in_shape, out_shape)

    else:
        raise NotImplementedError(f"Unknown resize mode: {mode}")
    if "eva.cls_token" in model_dict.keys():
        resized_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=1)
    else:
        resized_pos_embed = resized_patch_pos_embed
    pretrained_dict["eva.pos_embed"] = resized_pos_embed
