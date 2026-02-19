"""
dataset.py — Data pipeline for the LGG MRI Segmentation dataset.

DATASET STRUCTURE
─────────────────
    data/kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif        ← MRI slice (256x256, 3-ch)
            TCGA_CS_4941_19960909_1_mask.tif   ← binary mask  (256x256, 1-ch)
            ...
        TCGA_CS_4942_19970222/
            ...

Each .tif image has 3 channels:
    Channel 0 = pre-contrast sequence
    Channel 1 = FLAIR sequence
    Channel 2 = post-contrast sequence

Masks are binary:  0 = background,  255 = FLAIR abnormality.

BINARY CLASSIFICATION LABEL
────────────────────────────
Since our model is a *classifier* (not a segmenter), we derive a
single binary label per slice from the mask:

    label = 1  if mask contains ANY tumor pixels   (max > 0)
    label = 0  otherwise

PIPELINE OVERVIEW
─────────────────
    .tif file  ->  load (PIL)  ->  resize to PATCH_SIZE  ->  normalize
    each channel independently  ->  augment (train only)  ->  tensor

PATIENT-LEVEL SPLIT
───────────────────
We split by *patient*, NOT by slice.  If slices from the same patient
appear in both train and val, the model memorizes patient-specific
anatomy and gives inflated val scores.  Patient-level split prevents this.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config


# =============================================
#  1. SCANNING THE DATASET
# =============================================

def discover_samples(data_dir: str = config.DATA_DIR):
    """
    Walk the dataset folder and return a list of (image_path, mask_path)
    tuples, one per slice.

    Pairing logic: for every file that does NOT contain '_mask', check
    if a corresponding '_mask.tif' exists.

    Returns
    -------
    samples : list of (str, str)
        Each element is (image_path, mask_path).
    patient_ids : list of str
        Patient ID for each sample (used for patient-level splitting).
    """
    samples = []
    patient_ids = []

    # Each sub-folder is one patient
    patient_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for patient in tqdm(patient_dirs, desc="  Scanning patients", leave=False):
        pdir = os.path.join(data_dir, patient)
        # Find all non-mask .tif files
        image_files = sorted(glob.glob(os.path.join(pdir, "*.tif")))
        image_files = [f for f in image_files if "_mask" not in f]

        for img_path in image_files:
            # Derive mask path:  .../_10.tif  ->  .../_10_mask.tif
            base, ext = os.path.splitext(img_path)
            mask_path = base + "_mask" + ext

            if os.path.exists(mask_path):
                samples.append((img_path, mask_path))
                patient_ids.append(patient)

    return samples, patient_ids


# =============================================
#  2. NORMALIZATION
# =============================================

def normalize_modality(channel: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalization for a single 2D channel.

    Why per-channel?
    ────────────────
    Each MRI sequence (pre-contrast, FLAIR, post-contrast) has a
    completely different intensity distribution.  Normalizing them
    together would let the brightest modality dominate.

    The epsilon prevents division-by-zero on uniform patches
    (e.g., pure background).
    """
    mean = channel.mean()
    std = channel.std()
    eps = 1e-8
    return (channel - mean) / (std + eps)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Apply per-channel normalization.  image shape: (C, H, W)."""
    return np.stack(
        [normalize_modality(image[c]) for c in range(image.shape[0])],
        axis=0,
    )


# =============================================
#  3. AUGMENTATION
# =============================================

def random_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Random horizontal and/or vertical flip.  Shape: (C, H, W)."""
    if np.random.rand() < p:
        image = np.flip(image, axis=2).copy()   # horizontal
    if np.random.rand() < p:
        image = np.flip(image, axis=1).copy()   # vertical
    return image


def random_contrast(image: np.ndarray,
                    low: float = 0.9,
                    high: float = 1.1) -> np.ndarray:
    """Multiply by a random factor in [low, high] to vary contrast."""
    factor = np.random.uniform(low, high)
    return image * factor


def augment(image: np.ndarray) -> np.ndarray:
    """Full augmentation pipeline (training only)."""
    image = random_flip(image, p=config.FLIP_PROB)
    low, high = config.CONTRAST_RANGE
    image = random_contrast(image, low, high)
    return image


# =============================================
#  4. PYTORCH DATASET
# =============================================

class LGGDataset(Dataset):
    """
    PyTorch Dataset for LGG MRI slices.

    Parameters
    ----------
    samples : list of (image_path, mask_path)
    patch_size : int
        Images are resized to (patch_size, patch_size).
    train : bool
        If True, apply stochastic augmentation.

    __getitem__ returns
    -------------------
    image : FloatTensor, shape (3, patch_size, patch_size)
    label : FloatTensor, shape (1,)   -- 0.0 or 1.0
    """

    def __init__(self, samples: list, patch_size: int = config.PATCH_SIZE,
                 train: bool = True):
        self.samples = samples
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # -- Load from disk --
        # PIL Image -> numpy.  Image is (H, W, 3), mask is (H, W).
        image = np.array(
            Image.open(img_path).resize(
                (self.patch_size, self.patch_size), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        mask = np.array(
            Image.open(mask_path).resize(
                (self.patch_size, self.patch_size), Image.NEAREST
            ),
            dtype=np.float32,
        )

        # -- Derive binary label from mask --
        # If ANY pixel in the mask is > 0 -> tumor present -> label = 1
        label = 1.0 if mask.max() > 0 else 0.0

        # -- Rearrange to (C, H, W) --
        # PIL loads as (H, W, 3) but PyTorch expects channel-first
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)      # (3, H, W)
        else:
            image = image[np.newaxis, :, :]        # (1, H, W) grayscale

        # -- Per-channel normalization --
        image = normalize_image(image)

        # -- Augment (training only) --
        if self.train:
            image = augment(image)

        # -- To tensors --
        image = torch.from_numpy(image.copy()).float()
        label = torch.tensor([label], dtype=torch.float32)

        return image, label


# =============================================
#  5. PATIENT-LEVEL TRAIN / VAL SPLIT
# =============================================

def patient_split(samples: list,
                  patient_ids: list,
                  val_fraction: float = config.VAL_SPLIT,
                  seed: int = 42):
    """
    Split samples into train / val by PATIENT -- not by slice.

    Why patient-level?
    ------------------
    Adjacent MRI slices from the same patient look very similar.
    A random slice-level split would leak near-duplicate information
    into validation, giving falsely high scores.

    Returns
    -------
    train_samples, val_samples : lists of (img_path, mask_path)
    """
    rng = np.random.RandomState(seed)
    unique_patients = sorted(set(patient_ids))
    rng.shuffle(unique_patients)

    n_val = max(1, int(len(unique_patients) * val_fraction))
    val_patients = set(unique_patients[:n_val])

    train_samples = []
    val_samples = []
    for sample, pid in zip(samples, patient_ids):
        if pid in val_patients:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, val_samples


def get_dataloaders(data_dir: str = config.DATA_DIR,
                    val_fraction: float = config.VAL_SPLIT):
    """
    Full pipeline: discover -> split -> DataLoaders.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    # -- Discover all (image, mask) pairs --
    samples, patient_ids = discover_samples(data_dir)
    print(f"Found {len(samples)} slices from "
          f"{len(set(patient_ids))} patients in {data_dir}")

    # -- Count class balance --
    n_pos = sum(1 for _, m in tqdm(samples, desc="  Counting labels", leave=False)
                if np.array(Image.open(m)).max() > 0)
    n_neg = len(samples) - n_pos
    print(f"  Positive (tumor):  {n_pos}  ({n_pos/len(samples):.1%})")
    print(f"  Negative (clean):  {n_neg}  ({n_neg/len(samples):.1%})")

    # -- Patient-level split --
    train_samples, val_samples = patient_split(
        samples, patient_ids, val_fraction
    )
    print(f"  Train slices: {len(train_samples)}")
    print(f"  Val   slices: {len(val_samples)}")

    # -- Build DataLoaders --
    train_ds = LGGDataset(train_samples, train=True)
    val_ds = LGGDataset(val_samples, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,       # set >0 on Linux for speed
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader
