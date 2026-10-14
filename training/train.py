import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.dataset import create_dataloaders
from data.graph import build_knn_graph
from models.md_stgat import MDSTGAT


# ---------------- Configuration ----------------

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "md_stgat_best.pt"

# ---------------- Reproducibility ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ---------------- Data ----------------

train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=2,
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# ---------------- Real station graph ----------------

graph_result = build_knn_graph(k=4)

graph_data = (
    graph_result[0]
    if isinstance(graph_result, tuple)
    else graph_result
)

edge_index = torch.as_tensor(
    graph_data,
    dtype=torch.long,
)

# The current graph builder returns [48, 2].
if edge_index.ndim != 2:
    raise ValueError(
        f"Unexpected graph dimensions: {tuple(edge_index.shape)}"
    )

if edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
    edge_index = edge_index.t().contiguous()
elif edge_index.shape[0] == 2:
    edge_index = edge_index.contiguous()
else:
    raise ValueError(
        f"Expected edge list [E,2] or [2,E], got "
        f"{tuple(edge_index.shape)}"
    )

if edge_index.min() < 0 or edge_index.max() >= 12:
    raise ValueError("Graph contains invalid station indices.")

edge_index = edge_index.to(device)

print("Graph shape:", tuple(edge_index.shape))

# ---------------- Model ----------------

model = MDSTGAT(
    pollution_dim=6,
    weather_dim=7,
    hidden_dim=64,
    num_stations=12,
    forecast_horizon=24,
    gat_heads=4,
    transformer_heads=4,
    transformer_layers=2,
    dropout=0.1,
).to(device)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters()
        if p.requires_grad),
)

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

# ---------------- One epoch ----------------

def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for pollution, weather, target in loader:
        pollution = pollution.to(device)
        weather = weather.to(device)
        target = target.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            forecast, _ = model(
                pollution,
                weather,
                edge_index,
            )

            loss = criterion(forecast, target)

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_size = pollution.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# ---------------- Training loop ----------------

best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    scheduler.step(val_loss)

    learning_rate = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f} | "
        f"LR: {learning_rate:.2e}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": best_val_loss,
                "seed": SEED,
            },
            CHECKPOINT_PATH,
        )

        print(f"  Saved best checkpoint: {CHECKPOINT_PATH}")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

print("\nTraining complete.")
print(f"Best validation MSE: {best_val_loss:.6f}")
print(f"Best checkpoint: {CHECKPOINT_PATH}")
print("Test loader prepared but not evaluated.")