
import numpy as np
import torch
from pathlib import Path

from data.dataset import create_dataloaders
from data.graph import build_knn_graph
from models.md_stgat import MDSTGAT


CHECKPOINT_PATH = "checkpoints/md_stgat_best.pt"
OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    _, _, test_loader = create_dataloaders(
        batch_size=32,
        num_workers=2,
    )

    # Build the same station graph used during training.
    result = build_knn_graph(k=4)
    graph_data = result[0] if isinstance(result, tuple) else result

    edge_index = torch.as_tensor(
        graph_data, dtype=torch.long
    )

    if edge_index.ndim != 2:
        raise ValueError(
            f"Unexpected graph shape: {tuple(edge_index.shape)}"
        )

    if edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    elif edge_index.shape[0] == 2:
        edge_index = edge_index.contiguous()
    else:
        raise ValueError(
            f"Expected [E,2] or [2,E], got {tuple(edge_index.shape)}"
        )

    edge_index = edge_index.to(device)

    # Load the full MD-STGAT architecture and best checkpoint.
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

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation MSE: {checkpoint['val_mse']:.6f}")

    predictions = []
    targets = []

    with torch.inference_mode():
        for X_pollution, X_weather, Y in test_loader:
            X_pollution = X_pollution.to(device)
            X_weather = X_weather.to(device)

            forecast, _ = model(
                X_pollution,
                X_weather,
                edge_index,
            )

            predictions.append(forecast.cpu().numpy())
            targets.append(Y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values.")

    # Inverse-transform the PM2.5 target using the existing
    # training-fitted pollution scaler.
    import joblib

    scaler = joblib.load("data/processed/pollution_scaler.pkl")
    pm25_index = 0  # PM2.5 is the first pollution feature.

    mean = scaler.mean_[pm25_index]
    scale = scaler.scale_[pm25_index]

    predictions = predictions * scale + mean
    targets = targets * scale + mean

    errors = predictions - targets

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n===== MD-STGAT TEST RESULTS =====")
    print(f"MAE  : {mae:.4f} µg/m³")
    print(f"RMSE : {rmse:.4f} µg/m³")
    print(f"R²   : {r2:.4f}")
    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)

    # Horizon-wise metrics.
    horizon_rows = []

    for h in range(predictions.shape[1]):
        e = predictions[:, h, :] - targets[:, h, :]
        horizon_rows.append([
            h + 1,
            np.mean(np.abs(e)),
            np.sqrt(np.mean(e ** 2)),
        ])

    horizon_path = OUTPUT_DIR / "md_stgat_horizon_metrics.csv"
    np.savetxt(
        horizon_path,
        horizon_rows,
        delimiter=",",
        header="hour,MAE,RMSE",
        comments="",
        fmt=["%d", "%.6f", "%.6f"],
    )

    # Station-wise metrics.
    station_names = [
        "Aotizhongxin", "Changping", "Dingling", "Dongsi",
        "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan",
        "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
    ]

    station_rows = []

    for i, station in enumerate(station_names):
        e = predictions[:, :, i] - targets[:, :, i]
        station_rows.append([
            station,
            np.mean(np.abs(e)),
            np.sqrt(np.mean(e ** 2)),
        ])

    import csv

    station_path = OUTPUT_DIR / "md_stgat_station_metrics.csv"
    with open(station_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "MAE", "RMSE"])
        writer.writerows(station_rows)

    np.savez_compressed(
        OUTPUT_DIR / "md_stgat_predictions.npz",
        predictions=predictions,
        targets=targets,
    )

    print("\nSaved:")
    print(horizon_path)
    print(station_path)
    print(OUTPUT_DIR / "md_stgat_predictions.npz")


if __name__ == "__main__":
    main()
