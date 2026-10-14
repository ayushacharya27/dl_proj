"""
Generate two presentation-ready MD-STGAT result figures from actual evaluation CSVs.

Run from project root:
    python -m evaluation.visualize_md_stgat_results

Inputs:
    evaluation/md_stgat_station_metrics.csv
    evaluation/md_stgat_horizon_metrics.csv

Outputs:
    figures/md_stgat_rmse_by_station.png
    figures/md_stgat_key_findings.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
EVAL_DIR = ROOT / "evaluation"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

station_path = EVAL_DIR / "md_stgat_station_metrics.csv"
horizon_path = EVAL_DIR / "md_stgat_horizon_metrics.csv"

if not station_path.exists():
    raise FileNotFoundError(f"Missing results file: {station_path}")
if not horizon_path.exists():
    raise FileNotFoundError(f"Missing results file: {horizon_path}")

stations = pd.read_csv(station_path)
horizon = pd.read_csv(horizon_path)

required_station_cols = {"station", "MAE", "RMSE"}
required_horizon_cols = {"hour", "MAE", "RMSE"}
if not required_station_cols.issubset(stations.columns):
    raise ValueError(
        f"{station_path} must contain columns {sorted(required_station_cols)}"
    )
if not required_horizon_cols.issubset(horizon.columns):
    raise ValueError(
        f"{horizon_path} must contain columns {sorted(required_horizon_cols)}"
    )

stations = stations.sort_values("RMSE", ascending=True).reset_index(drop=True)
horizon = horizon.sort_values("hour")

# ---------------- Figure 1: RMSE by station ----------------
fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=200)
bars = ax.barh(stations["station"], stations["RMSE"])
ax.invert_yaxis()
ax.set_title(
    "MD-STGAT RMSE by Station",
    fontsize=16,
    fontweight="bold",
    pad=14,
)
ax.set_xlabel("Test-set RMSE (µg/m³)", fontsize=11)
ax.set_ylabel("Monitoring station", fontsize=11)
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

for bar, value in zip(bars, stations["RMSE"]):
    ax.text(
        value + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        ha="left",
        fontsize=9,
    )

ax.set_xlim(0, stations["RMSE"].max() * 1.16)
fig.text(
    0.5, 0.015,
    "Test-set results; lower RMSE indicates lower prediction error.",
    ha="center", fontsize=9, color="dimgray"
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
path1 = OUT_DIR / "md_stgat_rmse_by_station.png"
fig.savefig(path1, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- Figure 2: What results tell us ----------------
best = stations.iloc[0]
worst = stations.iloc[-1]
rmse_gap = float(worst["RMSE"] - best["RMSE"])
rmse_ratio = rmse_gap / float(best["RMSE"]) * 100

first = horizon.loc[horizon["hour"] == 1]
last = horizon.loc[horizon["hour"] == 24]
if first.empty or last.empty:
    raise ValueError("Horizon metrics must include hours 1 and 24.")

h1_rmse = float(first.iloc[0]["RMSE"])
h24_rmse = float(last.iloc[0]["RMSE"])
h1_mae = float(first.iloc[0]["MAE"])
h24_mae = float(last.iloc[0]["MAE"])
horizon_rmse_pct = (h24_rmse - h1_rmse) / h1_rmse * 100

fig = plt.figure(figsize=(11, 6.2), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.35, wspace=0.25)

fig.suptitle(
    "What the MD-STGAT Test Results Tell Us",
    fontsize=16, fontweight="bold", y=0.97
)

# Best/worst station cards
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis("off")
ax1.text(0.02, 0.85, "LOWEST STATION ERROR", fontsize=10, fontweight="bold",
         transform=ax1.transAxes, color="#245b38")
ax1.text(0.02, 0.53, str(best["station"]), fontsize=18, fontweight="bold",
         transform=ax1.transAxes)
ax1.text(0.02, 0.22, f'RMSE: {best["RMSE"]:.2f} µg/m³',
         fontsize=13, transform=ax1.transAxes)

ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")
ax2.text(0.02, 0.85, "HIGHEST STATION ERROR", fontsize=10, fontweight="bold",
         transform=ax2.transAxes, color="#8a3333")
ax2.text(0.02, 0.53, str(worst["station"]), fontsize=18, fontweight="bold",
         transform=ax2.transAxes)
ax2.text(0.02, 0.22, f'RMSE: {worst["RMSE"]:.2f} µg/m³',
         fontsize=13, transform=ax2.transAxes)

# Horizon RMSE plot
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(horizon["hour"], horizon["RMSE"], marker="o", linewidth=2, label="RMSE")
ax3.set_title("Error increases with forecast horizon", fontsize=12, fontweight="bold")
ax3.set_xlabel("Forecast horizon (hours)")
ax3.set_ylabel("RMSE (µg/m³)")
ax3.set_xticks(range(1, 25, 2))
ax3.grid(True, alpha=0.25)
ax3.legend(loc="upper left")

summary = (
    f"Station RMSE gap: {rmse_gap:.2f} µg/m³ ({rmse_ratio:.1f}% above the best station)\n"
    f"1-hour → 24-hour RMSE: {h1_rmse:.2f} → {h24_rmse:.2f} µg/m³ "
    f"(+{horizon_rmse_pct:.1f}%)\n"
    f"1-hour → 24-hour MAE: {h1_mae:.2f} → {h24_mae:.2f} µg/m³"
)
fig.text(
    0.5, 0.015, summary, ha="center", va="bottom", fontsize=9.2,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f6fb", edgecolor="#9bb8e8")
)

fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.92])
path2 = OUT_DIR / "md_stgat_key_findings.png"
fig.savefig(path2, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("Created two figures:")
print(f"1. {path1}")
print(f"2. {path2}")
print("\nKey values calculated from the CSV files:")
print(f"Best station: {best['station']} (RMSE {best['RMSE']:.4f} µg/m³)")
print(f"Worst station: {worst['station']} (RMSE {worst['RMSE']:.4f} µg/m³)")
print(f"Station RMSE gap: {rmse_gap:.4f} µg/m³ ({rmse_ratio:.2f}%)")
print(f"1h RMSE: {h1_rmse:.4f} | 24h RMSE: {h24_rmse:.4f}")
print(f"1h-to-24h RMSE increase: {horizon_rmse_pct:.2f}%")
