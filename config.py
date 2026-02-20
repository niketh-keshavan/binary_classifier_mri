"""
config.py — Central configuration for the binary classifier.

Every tunable hyper-parameter lives here so nothing is hard-coded
elsewhere. Read through each comment to understand what it controls.
"""

import torch

# ──────────────────────────────────────────────
# 1. DATA
# ──────────────────────────────────────────────
# Number of MRI modalities stacked as input channels.
# This LGG dataset has 3 channels: pre-contrast, FLAIR, post-contrast.
# (BraTS uses 4: T1, T1ce, T2, FLAIR — change to 4 if using BraTS.)
NUM_MODALITIES = 3

# Original images are 256×256.  We resize to this for faster training.
# Set to 256 to train at full resolution.
PATCH_SIZE = 128

# ──────────────────────────────────────────────
# 2. AUGMENTATION
# ──────────────────────────────────────────────
# Probability of applying each random flip (horizontal / vertical).
FLIP_PROB = 0.5

# Range for random contrast adjustment factor.
CONTRAST_RANGE = (0.9, 1.1)

# ──────────────────────────────────────────────
# 3. MODEL (SA-Net encoder style)
# ──────────────────────────────────────────────
# Initial feature width coming out of the first convolution.
INIT_FEATURES = 24

# Channel progression through the encoder stages.
# Each stage doubles the features while halving spatial dims.
# 24 → 48 → 96 → 192 → 384
ENCODER_CHANNELS = [24, 48, 96, 192, 384]

# Squeeze-and-Excitation reduction ratio.
SE_REDUCTION = 4

# ──────────────────────────────────────────────
# 4. TRAINING
# ──────────────────────────────────────────────
BATCH_SIZE = 1          # instance-norm is designed for batch_size=1
LEARNING_RATE = 3e-3    # Adam starting LR
LR_DECAY_FACTOR = 0.3   # multiply LR by this on plateau
LR_PATIENCE = 5         # epochs to wait before decaying

NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# 5. LOSS
# ──────────────────────────────────────────────
# "bce" or "focal"
LOSS_TYPE = "focal"

# Focal-loss gamma (modulating exponent).
FOCAL_GAMMA = 2.0

# ──────────────────────────────────────────────
# 6. PATHS  (adjust to your system)
# ──────────────────────────────────────────────
DATA_DIR = "data/kaggle_3m"   # LGG segmentation dataset root
CHECKPOINT_DIR = "checkpoints/"
VAL_SPLIT = 0.2               # fraction of patients held for validation

# ──────────────────────────────────────────────
# 7. CROSS-VALIDATION
# ──────────────────────────────────────────────
NUM_FOLDS = 5                  # k for k-fold patient-level CV
CV_CHECKPOINT_DIR = "checkpoints/cv/"  # per-fold model saves
