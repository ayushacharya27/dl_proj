import torch
import torch.nn as nn

from data.dataset import create_dataloaders
from data.graph import build_knn_graph
from models.md_stgat import MDSTGAT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load data
train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32,
    num_workers=0,
)

# Inspect the first batch
pollution, weather, target = next(iter(train_loader))

for name, tensor in [
    ("Pollution", pollution),
    ("Weather", weather),
    ("Target", target),
]:
    finite = torch.isfinite(tensor)
    print(f"\n{name}:")
    print("  Shape:", tuple(tensor.shape))
    print("  Non-finite values:", (~finite).sum().item())

    if finite.any():
        valid = tensor[finite]
        print("  Min:", valid.min().item())
        print("  Max:", valid.max().item())

# Check the graph
result = build_knn_graph(k=4)
edges = result[0] if isinstance(result, tuple) else result
edge_index = torch.as_tensor(edges, dtype=torch.long)

if edge_index.ndim != 2:
    raise ValueError(f"Invalid graph dimensions: {tuple(edge_index.shape)}")

if edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
    edge_index = edge_index.t().contiguous()

if edge_index.shape[0] != 2:
    raise ValueError(f"Invalid edge_index shape: {tuple(edge_index.shape)}")

edge_index = edge_index.to(device)

# Move data to device
pollution = pollution.to(device)
weather = weather.to(device)
target = target.to(device)

# Initialize model
model = MDSTGAT().to(device)
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# Check inputs before the forward pass
if not all(
    torch.isfinite(x).all()
    for x in (pollution, weather, target)
):
    print("\nSTOP: The batch contains NaN or infinite values.")
    raise SystemExit(1)

# Forward pass
forecast, attention = model(pollution, weather, edge_index)

for name, tensor in [
    ("Forecast", forecast),
    ("Attention", attention),
]:
    print(f"\n{name}:")
    print("  Shape:", tuple(tensor.shape))
    print("  All finite:", torch.isfinite(tensor).all().item())

    finite = torch.isfinite(tensor)
    if finite.any():
        valid = tensor[finite]
        print("  Min:", valid.min().item())
        print("  Max:", valid.max().item())

loss = criterion(forecast, target)
print("\nInitial loss:", loss.item())
print("Loss finite:", torch.isfinite(loss).item())

if not torch.isfinite(loss):
    print("STOP: Non-finite loss before backpropagation.")
    raise SystemExit(1)

# Check gradients
optimizer.zero_grad(set_to_none=True)
loss.backward()

bad_grads = [
    name
    for name, param in model.named_parameters()
    if param.grad is not None
    and not torch.isfinite(param.grad).all()
]

print("\nParameters with non-finite gradients:", bad_grads)

if bad_grads:
    print("STOP: Non-finite gradients detected.")
    raise SystemExit(1)

grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

print("Gradient norm before clipping:", grad_norm.item())

optimizer.step()

bad_params = [
    name
    for name, param in model.named_parameters()
    if not torch.isfinite(param).all()
]

print("Parameters non-finite after optimizer step:", bad_params)

print("\nDiagnostic complete.")