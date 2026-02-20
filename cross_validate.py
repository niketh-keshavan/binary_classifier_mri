"""
cross_validate.py — Patient-level K-Fold Cross-Validation.

Usage
-----
    python cross_validate.py
    python cross_validate.py --folds 5 --epochs 20

WHY CROSS-VALIDATION?
─────────────────────
A single train/val split depends on *which* patients land in each set.
K-fold CV rotates through K different splits so every patient is
validated exactly once.  The averaged metrics are a much more reliable
estimate of real-world performance.

PATIENT-LEVEL FOLDS
───────────────────
We assign folds at the patient level (not slice level) to prevent
data leakage — adjacent slices from the same patient are highly
correlated and must stay in the same fold.

PIPELINE
────────
For each fold k ∈ {1, …, K}:
  1. Patients in fold k → validation set
  2. All other patients → training set
  3. Train a fresh model from scratch
  4. Evaluate → accuracy, Dice/F1
  5. Save the fold's checkpoint

After all folds: print per-fold results + mean ± std.
"""

import os
import argparse
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import discover_samples, LGGDataset
from model import BinaryClassifier
from losses import get_loss_fn
from train import fit
from evaluate import evaluate_model


# ═════════════════════════════════════════════
#  PATIENT-LEVEL K-FOLD SPLITTER
# ═════════════════════════════════════════════

def patient_kfold_splits(samples: list,
                         patient_ids: list,
                         num_folds: int = config.NUM_FOLDS,
                         seed: int = 42):
    """
    Yield (train_samples, val_samples) for each of K folds,
    split at the patient level.

    Parameters
    ----------
    samples     : list of (image_path, mask_path)
    patient_ids : list of str, one per sample
    num_folds   : int, number of folds K
    seed        : int, for reproducible shuffling

    Yields
    ------
    fold_idx        : int (0-based)
    train_samples   : list of (img_path, mask_path)
    val_samples     : list of (img_path, mask_path)
    """
    rng = np.random.RandomState(seed)
    unique_patients = sorted(set(patient_ids))
    rng.shuffle(unique_patients)

    # Assign each patient to a fold
    # e.g., 110 patients, 5 folds → folds of size ~22
    fold_assignment = {}
    for i, patient in enumerate(unique_patients):
        fold_assignment[patient] = i % num_folds

    for fold in range(num_folds):
        val_patients = {p for p, f in fold_assignment.items() if f == fold}

        train_samples = []
        val_samples = []
        for sample, pid in zip(samples, patient_ids):
            if pid in val_patients:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

        yield fold, train_samples, val_samples


# ═════════════════════════════════════════════
#  BUILD DATALOADERS FOR A FOLD
# ═════════════════════════════════════════════

def build_fold_loaders(train_samples, val_samples):
    """Create DataLoaders for one fold."""
    train_ds = LGGDataset(train_samples, train=True)
    val_ds = LGGDataset(val_samples, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
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


# ═════════════════════════════════════════════
#  MAIN CV LOOP
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Patient-level K-Fold Cross-Validation"
    )
    parser.add_argument("--folds", type=int, default=config.NUM_FOLDS,
                        help=f"Number of folds (default: {config.NUM_FOLDS})")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=f"Epochs per fold (default: {config.NUM_EPOCHS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fold assignment")
    args = parser.parse_args()

    device = config.DEVICE
    num_folds = args.folds
    num_epochs = args.epochs
    ckpt_dir = config.CV_CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 1. Discover dataset ────────────────────
    print("Discovering dataset ...")
    samples, patient_ids = discover_samples(config.DATA_DIR)
    n_patients = len(set(patient_ids))
    print(f"Found {len(samples)} slices from {n_patients} patients")
    print(f"Running {num_folds}-fold patient-level cross-validation "
          f"({num_epochs} epochs/fold)\n")

    # ── 2. Storage for per-fold results ────────
    fold_results = []

    # ── 3. Iterate over folds ──────────────────
    fold_pbar = tqdm(
        patient_kfold_splits(samples, patient_ids, num_folds, args.seed),
        total=num_folds,
        desc="CV Folds",
        unit="fold",
        position=0,
    )

    for fold, train_samples, val_samples in fold_pbar:
        fold_pbar.set_postfix(fold=f"{fold + 1}/{num_folds}")
        tqdm.write("\n" + "=" * 60)
        tqdm.write(f"  FOLD {fold + 1} / {num_folds}")
        tqdm.write(f"  Train: {len(train_samples)} slices  |  "
                   f"Val: {len(val_samples)} slices")
        tqdm.write("=" * 60)

        # ── Build loaders ──────────────────────
        train_loader, val_loader = build_fold_loaders(
            train_samples, val_samples
        )

        # ── Fresh model each fold ──────────────
        model = BinaryClassifier()
        loss_fn = get_loss_fn()

        # ── Train ──────────────────────────────
        history = fit(
            model, train_loader, val_loader, loss_fn,
            num_epochs=num_epochs,
            device=device,
        )

        # ── Evaluate ───────────────────────────
        tqdm.write(f"\n  Evaluating fold {fold + 1} ...")
        metrics = evaluate_model(model, val_loader, device=device)

        fold_results.append({
            "fold": fold + 1,
            "accuracy": metrics["accuracy"],
            "dice": metrics["dice"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        })

        # ── Save fold checkpoint ───────────────
        ckpt_path = os.path.join(ckpt_dir, f"classifier_fold{fold + 1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        tqdm.write(f"  Checkpoint saved: {ckpt_path}")

    # ═════════════════════════════════════════
    #  SUMMARY
    # ═════════════════════════════════════════
    accs = [r["accuracy"] for r in fold_results]
    dices = [r["dice"] for r in fold_results]
    train_losses = [r["final_train_loss"] for r in fold_results]
    val_losses = [r["final_val_loss"] for r in fold_results]

    print("\n\n" + "=" * 65)
    print("       CROSS-VALIDATION SUMMARY")
    print("=" * 65)
    print(f"{'Fold':>6}  {'Accuracy':>10}  {'Dice/F1':>10}  "
          f"{'Train Loss':>11}  {'Val Loss':>10}")
    print("-" * 65)
    for r in fold_results:
        print(f"  {r['fold']:>3}   {r['accuracy']:>10.4f}  {r['dice']:>10.4f}  "
              f"{r['final_train_loss']:>11.4f}  {r['final_val_loss']:>10.4f}")
    print("-" * 65)
    print(f"  Mean  {np.mean(accs):>10.4f}  {np.mean(dices):>10.4f}  "
          f"{np.mean(train_losses):>11.4f}  {np.mean(val_losses):>10.4f}")
    print(f"  Std   {np.std(accs):>10.4f}  {np.std(dices):>10.4f}  "
          f"{np.std(train_losses):>11.4f}  {np.std(val_losses):>10.4f}")
    print("=" * 65)

    # ── Identify best fold ─────────────────────
    best_idx = int(np.argmax(dices))
    best = fold_results[best_idx]
    print(f"\nBest fold: {best['fold']} "
          f"(Dice={best['dice']:.4f}, Acc={best['accuracy']:.4f})")
    print(f"Best checkpoint: {ckpt_dir}/classifier_fold{best['fold']}.pt")
    print("\nDone!")


if __name__ == "__main__":
    main()
