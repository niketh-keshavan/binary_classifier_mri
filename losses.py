"""
losses.py — Loss functions for binary classification.

Two options:

1. **BCE (Binary Cross-Entropy)**
   Standard loss for binary problems.  Uses logits (pre-sigmoid) for
   numerical stability.

2. **Focal Loss**  (Lin et al., 2017)
   Designed for class-imbalanced detection.  Down-weights easy examples
   and focuses training on hard ones near the decision boundary.

   Standard BCE:   L = -log(p)
   Focal Loss:     L = -(1 - p)^γ · log(p)

   The modulating factor (1-p)^γ is key:
   • When p → 1 (easy, correct prediction):  (1-p)^γ → 0  → loss shrinks
   • When p → 0 (hard, wrong prediction):     (1-p)^γ → 1  → loss unchanged

   γ (gamma) controls how aggressively easy examples are down-weighted.
   γ = 0  →  standard BCE (no modulation)
   γ = 2  →  the default in the paper, strong down-weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

    Parameters
    ----------
    gamma : float
        Focusing / modulating exponent.  Higher = more focus on hard
        examples.  Paper default = 2.
    reduction : str
        'mean' or 'sum' or 'none'.

    Input
    -----
    logits : (B, 1)  — raw model output (before sigmoid)
    targets : (B, 1) — binary labels {0, 1}

    Math (step by step for one sample)
    ──────────
    1.  p = sigmoid(logit)              ← predicted probability
    2.  bce = -[y·log(p) + (1-y)·log(1-p)]   ← standard BCE per element
    3.  p_t = p   if y=1,  else (1-p)  ← probability of the TRUE class
    4.  focal_weight = (1 - p_t)^γ     ← down-weight easy examples
    5.  loss = focal_weight * bce       ← modulated loss
    """

    def __init__(self, gamma: float = config.FOCAL_GAMMA,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        # Step 1+2: BCE with logits (numerically stable version)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Step 3: probability of the TRUE class
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)

        # Step 4: focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Step 5: modulated loss
        loss = focal_weight * bce

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn() -> nn.Module:
    """
    Factory that returns the loss function based on config.LOSS_TYPE.

    BCEWithLogitsLoss = sigmoid + BCE combined for stability.
    FocalLoss = our custom implementation above.
    """
    if config.LOSS_TYPE == "bce":
        print("Using: BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    elif config.LOSS_TYPE == "focal":
        print(f"Using: FocalLoss (gamma={config.FOCAL_GAMMA})")
        return FocalLoss(gamma=config.FOCAL_GAMMA)
    else:
        raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")
