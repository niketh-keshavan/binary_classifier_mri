"""
train.py — Training and validation loops.

KEY CONCEPTS
────────────
1. **Adam optimizer**
   Adaptive learning rate optimizer that maintains per-parameter
   momentum.  The starting LR (0.003) is relatively high — the
   scheduler will lower it when needed.

2. **ReduceLROnPlateau scheduler**
   Monitors validation loss.  If it doesn't improve for `patience`
   epochs, multiply the LR by `factor` (0.3).  This prevents the
   model from overshooting minima late in training.

3. **Training loop anatomy**
   For each epoch:
     a. TRAIN phase  — forward → loss → backward → optimizer step
     b. VAL phase    — forward → loss (no gradients, no weight update)
     c. Scheduler step on val_loss
     d. Print metrics

4. **Why torch.no_grad() during validation?**
   We don't need gradients for evaluation.  Disabling them saves
   memory and speeds up the forward pass.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

import config


def train_one_epoch(model: nn.Module,
                    loader,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: str) -> float:
    """
    Run one full pass over the training set.

    Returns
    -------
    avg_loss : float
        Mean loss over all batches.
    """
    model.train()          # enable dropout / stochastic layers
    running_loss = 0.0
    n_batches = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)      # (B, C, H, W)
        labels = labels.to(device)      # (B, 1)

        # ---- Forward pass ----
        logits = model(images)           # (B, 1)
        loss = loss_fn(logits, labels)

        # ---- Backward pass ----
        optimizer.zero_grad()            # clear old gradients
        loss.backward()                  # compute new gradients
        optimizer.step()                 # update weights

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()    # ← no gradient computation
def validate(model: nn.Module,
             loader,
             loss_fn: nn.Module,
             device: str) -> float:
    """
    Run one full pass over the validation set.

    Returns
    -------
    avg_loss : float
    """
    model.eval()           # disable dropout / stochastic layers
    running_loss = 0.0
    n_batches = 0

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def fit(model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        num_epochs: int = config.NUM_EPOCHS,
        device: str = config.DEVICE):
    """
    Full training procedure.

    Steps
    ─────
    1. Move model to device (GPU if available).
    2. Create Adam optimizer (lr = 0.003).
    3. Create ReduceLROnPlateau scheduler.
    4. Loop over epochs: train → validate → schedule → log.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss'
    """
    model = model.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE)

    # --- LR Scheduler ---
    # mode='min'  → we want val_loss to decrease
    # patience=5  → wait 5 epochs of no improvement before reducing
    # factor=0.3  → new_lr = old_lr * 0.3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_DECAY_FACTOR,
        patience=config.LR_PATIENCE,
        verbose=True,
    )

    history = {"train_loss": [], "val_loss": []}

    print(f"\nTraining on {device} for {num_epochs} epochs\n" + "=" * 50)

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        train_loss = train_one_epoch(model, train_loader,
                                     loss_fn, optimizer, device)

        # ---- Validate ----
        val_loss = validate(model, val_loader, loss_fn, device)

        # ---- LR schedule step ----
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # ---- Log ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:3d}/{num_epochs}  |  "
              f"train_loss: {train_loss:.4f}  |  "
              f"val_loss: {val_loss:.4f}  |  "
              f"lr: {current_lr:.6f}")

    print("=" * 50 + "\nTraining complete.")
    return history
