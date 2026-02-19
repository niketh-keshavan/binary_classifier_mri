"""
model.py — SA-Net–style encoder for binary classification.

This file is built in layers of increasing abstraction:

    SE Block  →  ResSE Block  →  Encoder  →  BinaryClassifier

Read from top to bottom to understand each building block.
"""

import torch
import torch.nn as nn

import config


# ═══════════════════════════════════════════════
#  STEP 3 — Squeeze-and-Excitation (SE) Block
# ═══════════════════════════════════════════════
#
#  Paper: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
#
#  Idea:
#     Not all feature channels are equally useful for a given input.
#     The SE block learns a per-channel weighting vector s ∈ [0, 1]^C
#     and rescales the feature map:  x_out = x * s
#
#  How it works (3 stages):
#
#  ┌──────────────┐
#  │  Input (C,H,W)  │
#  └──────┬───────┘
#         │
#  1. SQUEEZE: Global Average Pool → (C, 1, 1)
#         │     Compress spatial dims into a channel descriptor.
#         ▼
#  2. EXCITATION: FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid
#         │     Two fully-connected layers learn inter-channel
#         │     relationships.  Reduction ratio r shrinks the
#         │     bottleneck to save parameters.
#         ▼
#  3. SCALE: element-wise multiply input × channel weights
#         │
#  ┌──────┴───────┐
#  │  Output (C,H,W) │
#  └──────────────┘

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    reduction : int
        Bottleneck reduction ratio r.  Default = 4.
        A channel count of 48 becomes a bottleneck of 48/4 = 12.

    Why reduction?
    ──────────────
    Without it the FC layers would have C² parameters — expensive.
    The bottleneck compresses to C/r params then expands back, acting
    like a small auto-encoder that captures channel correlations.
    """

    def __init__(self, channels: int,
                 reduction: int = config.SE_REDUCTION):
        super().__init__()

        mid = max(channels // reduction, 1)   # at least 1

        # Squeeze: spatial → 1×1 per channel
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation: learn channel weights
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),   # compress
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),   # expand back
            nn.Sigmoid(),                            # bound to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        returns : (B, C, H, W) — same shape, but channel-reweighted.
        """
        B, C, _, _ = x.shape

        # 1. Squeeze — global average over spatial dims
        s = self.squeeze(x)          # (B, C, 1, 1)
        s = s.view(B, C)             # flatten to (B, C)

        # 2. Excitation — learn importance per channel
        s = self.excitation(s)       # (B, C)  values in [0, 1]
        s = s.view(B, C, 1, 1)      # reshape for broadcasting

        # 3. Scale — reweight original features
        return x * s


# ═══════════════════════════════════════════════
#  STEP 4 — Residual SE (ResSE) Block
# ═══════════════════════════════════════════════
#
#  Combines a standard residual block with SE channel attention.
#
#  Why residual / skip connections?
#  ────────────────────────────────
#  Deep networks suffer from vanishing gradients.  ResNet's trick is
#  to learn a *residual* f(x) and add it to the input:
#
#       output = x + f(x)
#
#  If f(x) ≈ 0, gradients flow straight through the identity path,
#  making very deep networks trainable.
#
#  Why Instance Norm instead of Batch Norm?
#  ────────────────────────────────────────
#  Batch Norm computes statistics across the batch dimension.
#  With batch_size = 1 (common in medical imaging), those statistics
#  are meaningless.  Instance Norm computes them per-sample, per-channel,
#  so it works perfectly even with a single image.
#
#  Block diagram:
#
#        x ─────────────────────┐ (skip / identity)
#        │                      │
#   Conv 3×3 → InstanceNorm → ReLU
#        │                      │
#   Conv 3×3 → InstanceNorm     │
#        │                      │
#      SE Block                 │
#        │                      │
#        + ◄────────────────────┘  (additive skip)
#        │
#       ReLU
#        │
#      output

class ResSEBlock(nn.Module):
    """
    Residual block with embedded Squeeze-and-Excitation.

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    stride : int
        Stride of the *first* convolution.
        stride=2 halves spatial dimensions (spatial reduction).
        stride=1 keeps dimensions unchanged.

    When in_ch ≠ out_ch (or stride ≠ 1) the skip connection uses a
    1×1 convolution to match dimensions — otherwise the addition
    x + f(x) would fail on shape mismatch.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()

        # --- Main path ---
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_ch, affine=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_ch, affine=True)

        # --- SE attention ---
        self.se = SEBlock(out_ch)

        # --- Skip path ---
        # If spatial dims or channel count change, we need a projection
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)          # maybe projected

        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.se(out)               # channel recalibration

        out = out + identity             # ← additive skip
        out = self.relu(out)
        return out


# ═══════════════════════════════════════════════
#  STEP 5 — Encoder + Classification Head
# ═══════════════════════════════════════════════
#
#  The encoder progressively:
#     • Halves spatial resolution   (stride-2 convolutions)
#     • Doubles feature width       (24 → 48 → 96 → 192 → 384)
#
#  After the final encoder stage we do Global Average Pooling to
#  collapse the spatial dimensions, then a single Linear layer
#  produces a scalar logit for binary classification.
#
#  Feature map progression (for 128×128 input):
#
#     Input:   4 × 128 × 128     (4 modalities)
#     Stem:   24 × 128 × 128     (initial conv, no downscale)
#     Stage1: 48 ×  64 ×  64     (stride-2)
#     Stage2: 96 ×  32 ×  32     (stride-2)
#     Stage3: 192 × 16 ×  16     (stride-2)
#     Stage4: 384 ×  8 ×  8      (stride-2)
#     GAP:    384                 (global average pool)
#     Output: 1                   (sigmoid → probability)

class Encoder(nn.Module):
    """SA-Net–style CNN encoder using ResSE blocks."""

    def __init__(self,
                 in_channels: int = config.NUM_MODALITIES,
                 channel_list: list = None):
        super().__init__()

        if channel_list is None:
            channel_list = config.ENCODER_CHANNELS  # [24,48,96,192,384]

        # --- Stem: lift input modalities to initial feature width ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channel_list[0], kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(channel_list[0], affine=True),
            nn.ReLU(inplace=True),
        )

        # --- Encoder stages ---
        stages = []
        for i in range(1, len(channel_list)):
            # Each stage: one ResSE block with stride=2 (downscale),
            # optionally followed by a second block at stride=1 for
            # extra capacity.  We keep it to one block per stage for
            # simplicity — easy to add more later.
            stages.append(
                ResSEBlock(channel_list[i - 1], channel_list[i], stride=2)
            )
        self.stages = nn.ModuleList(stages)

        # --- Global Average Pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C_in, H, W)
        returns : (B, final_channels) — a flat feature vector per sample.
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x)             # (B, 384, 1, 1)
        x = x.view(x.size(0), -1)   # (B, 384)
        return x


class BinaryClassifier(nn.Module):
    """
    Full model = Encoder + linear classification head.

    Outputs a *logit* (raw score before sigmoid).
    We apply sigmoid inside the loss function (BCEWithLogitsLoss)
    for numerical stability — never apply sigmoid before the loss.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        final_ch = config.ENCODER_CHANNELS[-1]  # 384
        self.head = nn.Linear(final_ch, 1)       # single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 4, H, W)
        returns : (B, 1) — logit
        """
        features = self.encoder(x)   # (B, 384)
        logit = self.head(features)   # (B, 1)
        return logit
