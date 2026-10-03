import numpy as np
import torch
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.baseline import BaselineModel
from data.dataset import create_dataloaders
from data.graph import build_knn_graph


CHECKPOINT_PATH = "checkpoints/baseline_best.pt"
BATCH_SIZE = 32


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def main():
    device = get_device()

    # ========================================================
    # Test loader
    # ========================================================
    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    # ========================================================
    # Graph
    # ========================================================
    edges, _ = build_knn_graph(k=4)
    edge_index = torch.tensor(edges.T, dtype=torch.long, device=device)

    # ========================================================
    # Model
    # ========================================================
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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation MSE: {checkpoint['val_loss']:.6f}")

    # ========================================================
    # Generate predictions
    # ========================================================
    predictions = []
    targets = []

    with torch.no_grad():
        for X_pollution, _, Y in test_loader:
            X_pollution = X_pollution.to(device, non_blocking=True)

            prediction = model(X_pollution, edge_index)

            predictions.append(prediction.cpu().numpy())
            targets.append(Y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    print("\nPrediction shape:", predictions.shape)
    print("Target shape:", targets.shape)

    # ========================================================
    # Inverse scaling
    # ========================================================
    pollution_scaler = joblib.load("data/processed/pollution_scaler.pkl")

    # PM2.5 is feature index 0.
    pm25_mean = pollution_scaler.mean_[0]
    pm25_std = pollution_scaler.scale_[0]

    predictions_real = predictions * pm25_std + pm25_mean
    targets_real = targets * pm25_std + pm25_mean

    # ========================================================
    # Overall metrics
    # ========================================================
    pred_flat = predictions_real.reshape(-1)
    target_flat = targets_real.reshape(-1)

    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
    r2 = r2_score(target_flat, pred_flat)

    print("\n========================================")
    print("OVERALL TEST METRICS")
    print("========================================")
    print(f"MAE  : {mae:.4f} µg/m³")
    print(f"RMSE : {rmse:.4f} µg/m³")
    print(f"R²   : {r2:.4f}")

    # ========================================================
    # Horizon-wise metrics
    # ========================================================
    print("\n========================================")
    print("HORIZON-WISE METRICS")
    print("========================================")
    print(f"{'Hour':>6} {'MAE':>12} {'RMSE':>12}")

    horizon_results = []
    for h in range(24):
        pred_h = predictions_real[:, h, :].reshape(-1)
        target_h = targets_real[:, h, :].reshape(-1)

        mae_h = mean_absolute_error(target_h, pred_h)
        rmse_h = np.sqrt(mean_squared_error(target_h, pred_h))

        horizon_results.append([h + 1, mae_h, rmse_h])
        print(f"{h + 1:>6} {mae_h:>12.4f} {rmse_h:>12.4f}")

    # ========================================================
    # Station-wise metrics
    # ========================================================
    station_names = [
        "Aotizhongxin", "Changping", "Dingling", "Dongsi",
        "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan",
        "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"
    ]

    print("\n========================================")
    print("STATION-WISE METRICS")
    print("========================================")
    print(f"{'Station':<20}{'MAE':>12}{'RMSE':>12}")

    station_results = []
    for station_idx, station in enumerate(station_names):
        pred_station = predictions_real[:, :, station_idx].reshape(-1)
        target_station = targets_real[:, :, station_idx].reshape(-1)

        mae_station = mean_absolute_error(target_station, pred_station)
        rmse_station = np.sqrt(mean_squared_error(target_station, pred_station))

        station_results.append([station, mae_station, rmse_station])
        print(f"{station:<20}{mae_station:>12.4f}{rmse_station:>12.4f}")

    # ========================================================
    # Save predictions
    # ========================================================
    np.savez_compressed(
        "evaluation/baseline_predictions.npz",
        predictions=predictions_real,
        targets=targets_real
    )

    np.savetxt(
        "evaluation/horizon_metrics.csv",
        np.asarray(horizon_results),
        delimiter=",",
        header="hour,mae,rmse",
        comments=""
    )

    print(
        "\nSaved:"
        "\n  evaluation/baseline_predictions.npz"
        "\n  evaluation/horizon_metrics.csv"
    )


if __name__ == "__main__":
    main()