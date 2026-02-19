"""
evaluate.py — Evaluation metrics for binary classification.

Three metrics are computed:

1. **Confusion Matrix**
   A 2×2 table counting:
       TN (True Neg)   FP (False Pos)
       FN (False Neg)   TP (True Pos)
   Tells you exactly WHERE the model goes wrong.

2. **Accuracy**
   acc = (TP + TN) / (TP + TN + FP + FN)
   Simple but misleading under class imbalance — if 95% of patches
   are negative, always predicting "no tumor" gives 95% accuracy.

3. **Dice Similarity Coefficient (DSC)**
   DSC = 2·TP / (2·TP + FP + FN)
   Measures spatial overlap between prediction and ground truth.
   Ranges from 0 (no overlap) to 1 (perfect).
   Unlike accuracy, it ignores true negatives — ideal for medical
   imaging where the region of interest is small.

   Connection to F1 score: DSC is mathematically identical to the
   F1 score (harmonic mean of precision and recall).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

import config


@torch.no_grad()
def collect_predictions(model: nn.Module,
                        loader,
                        device: str = config.DEVICE,
                        threshold: float = 0.5):
    """
    Run the model on an entire DataLoader and collect predictions.

    Parameters
    ----------
    threshold : float
        Probability above this → predict class 1.

    Returns
    -------
    all_preds  : np.ndarray, shape (N,) — predicted labels {0, 1}
    all_labels : np.ndarray, shape (N,) — true labels {0, 1}
    all_probs  : np.ndarray, shape (N,) — predicted probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)        # (B, C, H, W)
        logits = model(images)            # (B, 1)

        probs = torch.sigmoid(logits)     # convert logit → probability
        preds = (probs >= threshold).float()

        all_probs.append(probs.cpu().numpy().flatten())
        all_preds.append(preds.cpu().numpy().flatten())
        all_labels.append(labels.numpy().flatten())

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs))


def compute_metrics(preds: np.ndarray,
                    labels: np.ndarray) -> dict:
    """
    Compute confusion matrix, accuracy, and Dice coefficient.

    Parameters
    ----------
    preds  : np.ndarray of {0, 1}
    labels : np.ndarray of {0, 1}

    Returns
    -------
    dict with keys:
        'confusion_matrix' : 2×2 np.ndarray
        'accuracy'         : float
        'dice'             : float
    """
    # --- Confusion matrix via scikit-learn ---
    cm = sk_confusion_matrix(labels, preds, labels=[0, 1])
    # cm layout:
    #   [[TN, FP],
    #    [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    # --- Accuracy ---
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    # --- Dice / F1  ---
    # DSC = 2·TP / (2·TP + FP + FN)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "dice": dice,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


def evaluate_model(model: nn.Module,
                   loader,
                   device: str = config.DEVICE) -> dict:
    """
    End-to-end evaluation: collect predictions → compute metrics → print.
    """
    model = model.to(device)
    preds, labels, probs = collect_predictions(model, loader, device)
    metrics = compute_metrics(preds, labels)

    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 40)
    print("       EVALUATION RESULTS")
    print("=" * 40)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"  Actual Neg  [ {cm[0,0]:4d}   {cm[0,1]:4d} ]")
    print(f"  Actual Pos  [ {cm[1,0]:4d}   {cm[1,1]:4d} ]")
    print(f"\n  TP={metrics['tp']}  TN={metrics['tn']}  "
          f"FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"\n  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Dice/F1  : {metrics['dice']:.4f}")
    print("=" * 40 + "\n")

    return metrics
