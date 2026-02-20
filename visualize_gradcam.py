"""
visualize_gradcam.py — Generate Grad-CAM overlay images.

Usage
-----
    python visualize_gradcam.py

    # Custom checkpoint and output directory:
    python visualize_gradcam.py --checkpoint checkpoints/classifier.pt --output gradcam_outputs --num_samples 10

This script:
  1. Loads the trained model from a checkpoint.
  2. Picks random samples from the validation set.
  3. Generates Grad-CAM heatmaps showing WHERE the model is looking.
  4. Saves side-by-side images: Original | Grad-CAM | Overlay.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import config
from dataset import get_dataloaders, discover_samples, patient_split
from dataset import LGGDataset
from model import BinaryClassifier
from gradcam import GradCAM


def load_model(checkpoint_path: str, device: str) -> BinaryClassifier:
    """Load trained model from checkpoint."""
    model = BinaryClassifier()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model


def unnormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized (C, H, W) tensor back to displayable (H, W, 3).
    Since we z-normalized per channel, we just rescale to [0, 1].
    """
    img = image_tensor.cpu().numpy()                 # (C, H, W)
    img = img.transpose(1, 2, 0)                     # (H, W, C)
    # Clip and rescale each channel to [0, 1]
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            img[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            img[:, :, c] = 0
    return img


def create_overlay(image: np.ndarray, heatmap: np.ndarray,
                   alpha: float = 0.4) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap onto the original image.

    Parameters
    ----------
    image   : (H, W, 3), values in [0, 1]
    heatmap : (H, W), values in [0, 1]
    alpha   : blending factor (0 = all image, 1 = all heatmap)

    Returns
    -------
    overlay : (H, W, 3), values in [0, 1]
    """
    # Apply 'jet' colormap to heatmap
    colored_heatmap = cm.jet(heatmap)[:, :, :3]     # (H, W, 3), drop alpha

    # Blend
    overlay = (1 - alpha) * image + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def save_gradcam_figure(image: np.ndarray, heatmap: np.ndarray,
                        label: float, pred_prob: float,
                        save_path: str):
    """
    Save a 3-panel figure: Original | Grad-CAM Heatmap | Overlay.
    """
    overlay = create_overlay(image, heatmap, alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Original MRI
    axes[0].imshow(image)
    axes[0].set_title("MRI Slice", fontsize=12)
    axes[0].axis("off")

    # Panel 2: Grad-CAM heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    # Title with prediction info
    label_str = "TUMOR" if label > 0.5 else "NORMAL"
    pred_str = "TUMOR" if pred_prob > 0.5 else "NORMAL"
    correct = "✓" if label_str == pred_str else "✗"
    fig.suptitle(
        f"Ground Truth: {label_str}  |  "
        f"Prediction: {pred_str} ({pred_prob:.2%})  {correct}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "classifier.pt"),
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="gradcam_outputs",
                        help="Directory to save output images")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    device = config.DEVICE
    os.makedirs(args.output, exist_ok=True)

    # ── 1. Load model ──────────────────────────
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    # ── 2. Set up Grad-CAM ─────────────────────
    # Target: last encoder stage (highest-level spatial features)
    target_layer = model.encoder.stages[-1]
    grad_cam = GradCAM(model, target_layer)
    print(f"Grad-CAM target layer: encoder.stages[-1] "
          f"({config.ENCODER_CHANNELS[-1]} channels)")

    # ── 3. Load validation data ────────────────
    print("Loading validation data ...")
    samples, patient_ids = discover_samples(config.DATA_DIR)
    _, val_samples = patient_split(samples, patient_ids, config.VAL_SPLIT)

    # Build dataset WITHOUT augmentation
    val_ds = LGGDataset(val_samples, train=False)

    # Pick random indices
    rng = np.random.RandomState(args.seed)
    n = min(args.num_samples, len(val_ds))
    indices = rng.choice(len(val_ds), size=n, replace=False)

    # ── 4. Generate heatmaps ───────────────────
    print(f"\nGenerating Grad-CAM for {n} samples ...\n")
    for i, idx in enumerate(tqdm(indices, desc="Grad-CAM", unit="img")):
        image_tensor, label_tensor = val_ds[idx]
        label = label_tensor.item()

        # Forward for prediction probability
        input_batch = image_tensor.unsqueeze(0).to(device)
        input_batch.requires_grad_(True)

        with torch.enable_grad():
            heatmap = grad_cam(input_batch)

        # Get prediction probability
        with torch.no_grad():
            logit = model(input_batch)
            pred_prob = torch.sigmoid(logit).item()

        # Convert image for display
        display_img = unnormalize_image(image_tensor)

        # Save figure
        save_path = os.path.join(args.output, f"gradcam_{i:03d}.png")
        save_gradcam_figure(display_img, heatmap, label, pred_prob, save_path)

    grad_cam.remove_hooks()

    print(f"\nDone! {n} Grad-CAM images saved to {args.output}/")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
