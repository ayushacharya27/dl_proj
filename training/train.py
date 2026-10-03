import os
import random
import numpy as np
import torch
import torch.nn as nn

from models.baseline import BaselineModel
from data.dataset import create_dataloaders
from data.graph import build_knn_graph


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
CHECKPOINT_PATH = "checkpoints/baseline_best.pt"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, edge_index, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_pollution, _, Y in loader:
        X_pollution = X_pollution.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        optimizer.zero_grad()
        prediction = model(X_pollution, edge_index)
        loss = criterion(prediction, Y)
        loss.backward()

        # Prevent unstable gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = X_pollution.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, edge_index, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for X_pollution, _, Y in loader:
        X_pollution = X_pollution.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        prediction = model(X_pollution, edge_index)
        loss = criterion(prediction, Y)

        batch_size = X_pollution.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# Main
# ============================================================

def main():
    set_seed(42)
    os.makedirs("checkpoints", exist_ok=True)
    device = get_device()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------
    edges, _ = build_knn_graph(k=4)
    edge_index = torch.tensor(edges.T, dtype=torch.long, device=device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = BaselineModel(
        pollution_dim=6,
        embedding_dim=64,
        gat_heads=4,
        temporal_heads=4,
        temporal_layers=2,
        forecast_horizon=24,
        num_stations=12,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------------
    # Loss, Optimizer & Scheduler
    # --------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, edge_index, device
        )
        val_loss = validate(
            model, val_loader, criterion, edge_index, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                CHECKPOINT_PATH
            )
            print(f"  ✓ Saved best model (val loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()