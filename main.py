"""
main.py -- Run the full pipeline from data -> train -> evaluate.

Usage
-----
    python main.py

This script:
  1. Loads the LGG MRI dataset from data/kaggle_3m/
  2. Splits by patient into train / val DataLoaders
  3. Builds the BinaryClassifier model
  4. Selects the loss function (BCE or Focal)
  5. Trains the model
  6. Evaluates on the validation set
  7. Saves the trained weights

Each step maps to one of the modules we built:
    config.py   -- hyper-parameters
    dataset.py  -- data pipeline (LGG .tif loader)
    model.py    -- SE, ResSE, Encoder, Classifier
    losses.py   -- BCE / Focal Loss
    train.py    -- training loop
    evaluate.py -- metrics
"""

import os
import torch

import config
from dataset import get_dataloaders
from model import BinaryClassifier
from losses import get_loss_fn
from train import fit
from evaluate import evaluate_model


def main():
    # ------------------------------------------
    # 1. DATA  (LGG MRI .tif images)
    # ------------------------------------------
    print("Step 1: Loading LGG MRI dataset ...")
    train_loader, val_loader = get_dataloaders(
        data_dir=config.DATA_DIR,
        val_fraction=config.VAL_SPLIT,
    )

    # ──────────────────────────────────────────
    # 2. MODEL
    # ──────────────────────────────────────────
    print("\nStep 2: Building model ...")
    model = BinaryClassifier()

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # ──────────────────────────────────────────
    # 3. LOSS FUNCTION
    # ──────────────────────────────────────────
    print("\nStep 3: Selecting loss function ...")
    loss_fn = get_loss_fn()

    # ──────────────────────────────────────────
    # 4. TRAIN
    # ──────────────────────────────────────────
    print("\nStep 4: Training ...")
    history = fit(model, train_loader, val_loader, loss_fn,
                  num_epochs=config.NUM_EPOCHS,
                  device=config.DEVICE)

    # ──────────────────────────────────────────
    # 5. EVALUATE
    # ──────────────────────────────────────────
    print("\nStep 5: Evaluating on validation set ...")
    metrics = evaluate_model(model, val_loader, device=config.DEVICE)

    # ──────────────────────────────────────────
    # 6. SAVE CHECKPOINT
    # ──────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
