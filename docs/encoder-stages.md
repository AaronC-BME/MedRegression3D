# Encoder stages & patch size

The `ResEncoder` backbone (a `ResidualEncoderUNet` encoder + global average pool)
downsamples the input through a series of **stages**. How many stages it uses, and
the stride on each axis, are derived from your `data.patch_size` and the
`model.n_stages` config field. This page explains that logic.

## Background: stages, strides, and the factor of 2

A stage is a stride-2 convolution block (the first stage — the *stem* — uses
stride 1 and does no downsampling). With *S* stages there are *S − 1*
downsampling steps, each halving every spatial axis. So the total downsampling
factor is `2^(S-1)`.

The original nnSSL ResEnc-L topology has **6 stages** → 5 halvings → factor **32**.
That's where the "patch size must be a multiple of 32" rule comes from: each axis
has to survive 5 clean halvings.

Because the model is **encoder-only** and ends in a global average pool
(`x.mean(dim=[2,3,4])`), the spatial size of the bottleneck doesn't need to be
1³ — any positive size works. The head sees a fixed `embed_dim` vector regardless.

## The `model.n_stages` field

```yaml
model:
  input_shape: ${data.patch_size}
  n_stages: null     # or an integer
```

### `null` → AUTO

The depth is inferred from the patch size:

- Stage 1 is the stem (`[1,1,1]`).
- Each subsequent stage uses a uniform `[2,2,2]` stride.
- Stages are added until the **most-constrained axis can no longer be halved** —
  i.e. until some axis is odd or would drop below the minimum feature-map size (4).

So a patch downsamples as deep as its smallest / least-divisible axis allows, and
every stage is isotropic.

### integer → FIXED

You request exactly `n_stages` stages (so `n_stages − 1` downsampling steps). On
each downsampling step, an axis gets stride 2 **only if it can still be halved**;
otherwise its stride drops to 1. This is what produces anisotropic strides like
`[2,2,1]`: once an axis runs out of factors of 2, the other axes keep going.

If you request more stages than the patch can support, the extra stages end up
all-stride-1 (no downsampling) and a warning is logged.

## The divisibility rule

To get **N downsampling steps on an axis, that axis must be divisible by 2 N times.**
Equivalently, *N* clean halvings require the axis to be a multiple of `2^N`
(and large enough to stay ≥ 4).

This is why, for example, **224 stops at 6 stages**: `224 = 2^5 × 7`, so it can be
halved only 5 times (`224→112→56→28→14→7`, then 7 is odd). To reach a 7th stage you
need an axis divisible by `2^6 = 64` (e.g. 256).

## Channel widths

For from-scratch builds the per-stage channel count is:

```
features_per_stage[i] = min(32 * 2**i, 320)
```

i.e. double from 32, capped at 320 → `[32, 64, 128, 256, 320, 320, ...]`. The head's
input width (`embed_dim`) is the **last** stage's channel count, so shallower models
feed a narrower head (e.g. a 4-stage model → `embed_dim=256`, a 3-stage → `128`).

## No input padding needed

Because an axis is only ever halved when it is even, the spatial dimensions stay
integer-valued through the whole encoder — division is always exact. The adaptive
strides make input padding unnecessary; the stride schedule itself guarantees the
patch flows cleanly to the bottleneck.

## Pretrained weights restriction

The nnSSL ResEnc-L checkpoint only matches the original **6-stage** topology
(`features_per_stage=[32,64,128,256,320,320]`). Therefore:

- With `model.pretrained: True`, `n_stages` **must** be `6` or `null`. Any other
  value raises a `ValueError` at construction time with an explanatory message.
- Custom depths (`n_stages` ≠ 6) are only available with `pretrained: False`
  (training from scratch).

## Where the resolved geometry is recorded

You don't have to guess what AUTO resolved to:

1. **Training log** — `ResEncoder` prints a line at build time, e.g.:
   ```
   [ResEncoder] resolved geometry: n_stages=6 | strides=[[1,1,1],[2,2,2],...] | features_per_stage=[...] | embed_dim=320
   ```
2. **`Configs/model_architecture.txt`** — written into each run's output dir, with
   the resolved `n_stages`, `strides`, `features_per_stage`, `embed_dim`, parameter
   counts, and the full module tree. (The saved `config.yaml` records what you
   *requested* — e.g. `n_stages: null` — so this file is the source of truth for
   what was actually built.)

## Worked examples

| `patch_size`     | `n_stages` | resolved | strides | bottleneck | `embed_dim` |
|------------------|-----------|----------|---------|-----------|-------------|
| `[160,160,160]`  | `null`    | 6        | stem + 5×`[2,2,2]`                          | 5³    | 320 |
| `[160,160,80]`   | `null`    | 5        | stem + 4×`[2,2,2]`                          | 10×10×5 | 320 |
| `[160,160,80]`   | `6`       | 6        | stem + 4×`[2,2,2]` + `[2,2,1]`              | 5³    | 320 |
| `[20,20,20]`     | `null`    | 3        | stem + 2×`[2,2,2]`                          | 5³    | 128 |
| `[224,224,224]`  | `null`    | 6        | stem + 5×`[2,2,2]`                          | 7³    | 320 |
| `[256,256,256]`  | `null`    | 7        | stem + 6×`[2,2,2]`                          | 4³    | 320 |
| `[256,256,64]`   | `7`       | 7        | stem + 4×`[2,2,2]` + 2×`[2,2,1]`            | 4³    | 320 |

The implementation lives in `compute_stages_and_strides()` in
[`src/medregression3d/models/backbones/resenc.py`](../src/medregression3d/models/backbones/resenc.py).
