# MD-STGAT

## Multi-Modal Decoupled Spatio-Temporal Graph Attention Network for Multi-Station PM2.5 Forecasting

**Authors**
- **Ayush Acharya** — 23BPS1078
- **Aduru Guhya** — 23BPS1009

MD-STGAT forecasts PM2.5 concentrations for 12 Beijing air-quality monitoring stations using historical pollution, meteorological, spatial, and temporal information. This README reflects the implementation and experiments completed as of **14 October 2026**, with remaining research work clearly separated from implemented features.

---

## 1. Current Status

### Implemented and tested

- Loaded and validated the Beijing Multi-Site Air Quality dataset for 12 stations.
- Aligned station observations by timestamp and inspected missing values.
- Created chronological train/validation/test splits (70% / 15% / 15%).
- Implemented temporal forward-fill for short gaps, same-timestamp spatial inverse-distance imputation, and station-level training-set median fallback.
- Encoded wind direction as sine/cosine and fixed an initial NaN issue by including `wd_sin` and `wd_cos` in the numeric imputation feature list.
- Fitted scalers on training data and generated 72-hour input windows with a 24-hour forecast target.
- Implemented PyTorch datasets and dataloaders.
- Implemented independent pollution and weather encoders and independent spatial GAT modules.
- Implemented weather-query/pollution-key-value temporal cross-attention.
- Implemented a temporal Transformer and direct 24-hour forecast head.
- Trained and evaluated a pollution-only baseline.
- Trained and evaluated the full MD-STGAT model.
- Generated overall, horizon-wise, and station-wise evaluation metrics.
- Created scripts for dataset, model-verification, and result visualizations.

### Current limitations

- The current spatial graph is static and based on geographic proximity; it is not a dynamic wind-aware graph.
- The implemented training loss is MSE. The proposed graph-Laplacian spatial regularizer is **not yet integrated into training**.
- STL or STL-inspired temporal decomposition is **not yet integrated into the full model**.
- A persistence/last-value benchmark and controlled component ablations remain to be completed.
- The full model showed overfitting: its best validation checkpoint was from epoch 1. Further tuning and repeated runs are needed.
- Station coordinates should be verified against authoritative dataset metadata before being presented as precise geographic facts in a paper.

---

## 2. Dataset

The project uses the UCI Beijing Multi-Site Air Quality dataset, containing hourly measurements from 12 stations.

Stations:

1. Aotizhongxin
2. Changping
3. Dingling
4. Dongsi
5. Guanyuan
6. Gucheng
7. Huairou
8. Nongzhanguan
9. Shunyi
10. Tiantan
11. Wanliu
12. Wanshouxigong

Each station has 35,064 hourly records, for **420,768 station-time rows** across all stations.

### Input features

**Pollution (6 features):** `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`.

**Weather (7 features):** `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `wd_sin`, `wd_cos`.

### Window configuration

- Historical input window: **72 hours**
- Forecast horizon: **24 hours**
- Stations: **12**

Expected tensors:

```text
Pollution input: [B, 72, 12, 6]
Weather input:   [B, 72, 12, 7]
Target:          [B, 24, 12]
```

### Chronological split

| Split | Start | End | Window count |
|---|---|---|---:|
| Train | 2013-03-01 00:00 | 2015-12-18 15:00 | 24,449 |
| Validation | 2015-12-18 16:00 | 2016-07-24 19:00 | 5,165 |
| Test | 2016-07-24 20:00 | 2017-02-28 23:00 | 5,165 |

Windows are built independently within each split. The preprocessing pipeline splits chronologically before fitting scalers, to help avoid leakage from future data.

---

## 3. Preprocessing

Main file: `data/prepare_dataset.py`

Processing steps:

1. Encode compass wind direction as `wd_sin` and `wd_cos`.
2. Split the dataset chronologically into train, validation, and test sets.
3. Forward-fill recent past values for gaps up to six hours.
4. Impute remaining missing numeric values using inverse-distance weighted observations from up to four nearest stations at the same timestamp.
5. Use station-level medians computed from the training split as a fallback for remaining missing values.
6. Fit scalers on training data and transform the splits.
7. Create separate 72-hour input windows and 24-hour targets within each split.

After adding `wd_sin` and `wd_cos` to `ALL_NUMERIC`, the data diagnostic confirmed zero non-finite values in pollution, weather, and target tensors. A forward/backward diagnostic also confirmed finite model outputs, loss, gradients, and parameters after an optimizer step.

**Implementation note:** wind sine and cosine currently pass through the general numeric imputation pipeline. A more rigorous refinement would impute them together as a circular vector and normalize the vector after imputation.

Generated files:

```text
data/processed/
├── train.npz
├── val.npz
├── test.npz
├── pollution_scaler.pkl
└── weather_scaler.pkl
```

Observed dataset shapes:

```text
Train pollution:      (24449, 72, 12, 6)
Train weather:        (24449, 72, 12, 7)
Train target:         (24449, 24, 12)

Validation pollution: (5165, 72, 12, 6)
Validation weather:   (5165, 72, 12, 7)
Validation target:    (5165, 24, 12)

Test pollution:      (5165, 72, 12, 6)
Test weather:        (5165, 72, 12, 7)
Test target:         (5165, 24, 12)
```

---

## 4. Station Graph

Files: `data/graph.py`, `data/station_metadata.py`

The current graph is a **static physical-proximity graph**:

1. Calculate pairwise station distances using the Haversine formula.
2. Connect each station to its four nearest neighbors.
3. Use the resulting 48 directed edges for spatial GAT operations.

The graph builder returns an edge list with shape `[48, 2]`, which is transposed to PyTorch Geometric's expected `edge_index` shape `[2, 48]`.

Both modality-specific GATs use the same graph topology, but have independent trainable parameters.

---

## 5. Model Architecture

Full model file: `models/md_stgat.py`

```text
Pollution input [B,72,12,6]             Weather input [B,72,12,7]
          |                                          |
  Pollution Encoder                          Weather Encoder
      6 -> 64                                     7 -> 64
          |                                          |
    Pollution GAT                              Weather GAT
          |                                          |
          +------- Pollution: Key / Value ------------+
                        Cross-Modal Attention
                        Weather: Query
                                 |
                         Fused representation
                                 |
                       Temporal Transformer
                                 |
                      Select final timestep
                                 |
                           Forecast Head
                                 |
                       Output [B,24,12]
```

### 5.1 Pollution Encoder

File: `models/encoders.py`

- Linear projection: 6 → 64
- LayerNorm
- GELU
- Dropout

Output: `[B, T, N, 64]`.

### 5.2 Weather Encoder

File: `models/weather_encoder.py`

- Linear projection: 7 → 64
- LayerNorm
- GELU
- Dropout

Output: `[B, T, N, 64]`.

### 5.3 Spatial GAT

File: `models/gat.py`

- Input dimension: 64
- Hidden dimension: 64
- Attention heads: 4
- Dropout: 0.1
- Separate GAT instances for pollution and weather

The implementation batches the graph snapshots across batch and time dimensions instead of processing every graph snapshot in a separate Python loop.

### 5.4 Cross-Modal Temporal Attention

File: `models/cross_attention.py`

- Weather features provide the Query (Q).
- Historical pollution features provide the Key (K) and Value (V).
- For each station, weather queries attend over the 72-hour pollution history.
- Residual connections, layer normalization, and a feed-forward block are used after attention.

Input tensors:

```text
Pollution: [B, 72, 12, 64]
Weather:   [B, 72, 12, 64]
```

Outputs:

```text
Fused features: [B, 72, 12, 64]
Attention:      [B, 12, 72, 72]
```

The earlier one-token attention version was replaced because a `[1, 1]` attention matrix cannot select meaningfully across the historical sequence.

### 5.5 Temporal Transformer

File: `models/temporal.py`

- Hidden dimension: 64
- Attention heads: 4
- Encoder layers: 2
- Feed-forward dimension: 256
- Dropout: 0.1
- Learnable positional embedding for the 72-hour input sequence

Temporal self-attention is applied independently to each station's fused sequence. Output shape: `[B, 72, 12, 64]`.

### 5.6 Forecast head

The final historical representation is selected for each station, giving `[B, 12, 64]`. The forecast head applies `Linear(64 → 64)`, GELU, dropout, and `Linear(64 → 24)`, then permutes the result to `[B, 24, 12]`.

### Model verification

The model's forward pass was successfully tested on CUDA with the real 48-edge graph:

```text
Pollution input: (2, 72, 12, 6)
Weather input:   (2, 72, 12, 7)
Graph:           (2, 48)
Forecast:        (2, 24, 12)
Attention:       (2, 12, 72, 72)
Trainable params: 170,456
```

The full model has **170,456 trainable parameters**. The pollution-only baseline has **115,416 trainable parameters**.

---

## 6. Training

### Training configuration

The baseline and full model used the following main settings:

| Setting | Value |
|---|---|
| Batch size | 32 |
| Maximum epochs | 30 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Prediction loss | MSE |
| Gradient clipping | 1.0 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2 |
| Early stopping patience | 5 epochs |
| Random seed for full-model training script | 42 |

### Baseline

Training script: `training/train.py`

Best checkpoint: `checkpoints/baseline_best.pt`

- Best validation epoch: 6
- Best validation MSE: 0.627643

### Full MD-STGAT

Training script: `training/train_md_stgat.py`

Best checkpoint: `checkpoints/md_stgat_best.pt`

- Best validation epoch: 1
- Best validation MSE: 0.669713
- Early stopping triggered at epoch 6 after validation performance failed to improve.
- Training MSE continued to decrease while validation MSE increased, indicating overfitting.

---

## 7. Evaluation Results

### Overall test-set metrics

| Metric | Baseline | Full MD-STGAT | Comparison |
|---|---:|---:|---|
| MAE (µg/m³) | 45.1195 | 46.2771 | MD-STGAT is 2.57% higher (worse) |
| RMSE (µg/m³) | 74.3686 | 72.4479 | MD-STGAT is 2.58% lower (better) |
| R² | 0.3599 | 0.3925 | MD-STGAT is higher by 0.0326 |

The full model has lower RMSE and higher R² than the recorded baseline, but a slightly higher MAE. Results therefore show a mixed improvement rather than universal improvement across metrics.

**Fair-comparison check:** before using these as final research claims, verify both evaluations used identical processed test targets, the same scaler artifacts, and the same inverse-transform procedure.

### Full MD-STGAT: selected horizon-wise test metrics

| Forecast horizon | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| 1 hour | 35.96 | 57.94 |
| 6 hours | 40.67 | 66.03 |
| 12 hours | 46.23 | 72.81 |
| 18 hours | 50.87 | 77.67 |
| 24 hours | 55.21 | 81.65 |

Errors generally increase at longer horizons.

### Full MD-STGAT: station RMSE

| Station | RMSE (µg/m³) |
|---|---:|
| Dingling | 59.9309 |
| Huairou | 60.2416 |
| Changping | 62.1034 |
| Wanliu | 69.3945 |
| Aotizhongxin | 71.7996 |
| Shunyi | 73.2768 |
| Tiantan | 74.2051 |
| Guanyuan | 74.6933 |
| Nongzhanguan | 77.6236 |
| Dongsi | 79.9818 |
| Wanshouxigong | 80.2357 |
| Gucheng | 81.3679 |

Dingling has the lowest station RMSE among these results; Gucheng has the highest. These are observations from the metrics and do not, by themselves, explain why stations differ.

### Evaluation output files

Baseline:

```text
evaluation/baseline_predictions.npz
evaluation/horizon_metrics.csv
```

Full MD-STGAT:

```text
evaluation/md_stgat_predictions.npz
evaluation/md_stgat_horizon_metrics.csv
evaluation/md_stgat_station_metrics.csv
```

The `.npz` prediction archive stores predictions and targets; CSV files store the corresponding horizon-wise and station-wise metrics.

---

## 8. Visualizations

The project includes scripts to generate figures for data inspection, implementation verification, and result analysis.

### Dataset visualizations

Script: `visualize_md_stgat_data.py`

Intended outputs:

```text
figures/ppt_missing_values.png
figures/ppt_station_network.png
```

The missing-value chart visualizes the raw dataset's missing-data profile. The station-network chart visualizes the physical-proximity graph.

Run from the project root:

```bash
python visualize_md_stgat_data.py
```

### Result visualizations

Script: `evaluation/visualize_md_stgat_results.py`

Outputs:

```text
figures/md_stgat_rmse_by_station.png
figures/md_stgat_key_findings.png
```

These figures use the saved station-wise and horizon-wise CSV results to show RMSE by station and summarize the best/worst stations and change in error with forecast horizon.

Run:

```bash
python -m evaluation.visualize_md_stgat_results
```

A separate horizon plot can be generated with `evaluation/plot_horizon_metrics.py` if that file is present in the project.

---

## 9. How to Run

Run all commands from the project root (`~/dl_proj`). The project's virtual environment directory is named `dl_proj`.

### Activate the environment

```bash
cd ~/dl_proj
source dl_proj/bin/activate
```

### Rebuild the processed dataset

```bash
python -m data.prepare_dataset
```

### Check dataset batches

```bash
python -m data.dataset
```

Expected batch shapes:

```text
Pollution: [32, 72, 12, 6]
Weather:   [32, 72, 12, 7]
Target:    [32, 24, 12]
```

### Test model components

```bash
python -m models.encoders
python -m models.weather_encoder
python -m models.gat
python -m models.cross_attention
python -m models.temporal
python -m models.baseline
python -m models.md_stgat
```

### Train models

Baseline:

```bash
python -m training.train
```

Full MD-STGAT:

```bash
python -m training.train_md_stgat
```

### Evaluate models

Baseline:

```bash
python -m evaluation.evaluate
```

Full MD-STGAT:

```bash
python -m evaluation.evaluate_md_stgat
```

### Run diagnostics

If `debug_nan.py` is present in the project root, it can check input tensors, forecasts, loss, gradients, and parameters for non-finite values:

```bash
python debug_nan.py
```

---

## 10. Project Structure

```text
MD-STGAT/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── graph.py
│   ├── station_metadata.py
│   ├── imputation.py
│   ├── prepare_dataset.py
│   └── dataset.py
├── models/
│   ├── encoders.py
│   ├── weather_encoder.py
│   ├── gat.py
│   ├── cross_attention.py
│   ├── temporal.py
│   ├── baseline.py
│   └── md_stgat.py
├── losses/
│   ├── prediction_loss.py
│   └── physics_loss.py
├── training/
│   ├── train.py
│   ├── train_md_stgat.py
│   ├── validate.py
│   └── early_stopping.py
├── evaluation/
│   ├── evaluate.py
│   ├── evaluate_md_stgat.py
│   ├── metrics.py
│   ├── baselines.py
│   ├── ablation.py
│   ├── visualize_md_stgat_results.py
│   └── plot_horizon_metrics.py
├── figures/
├── checkpoints/
├── visualize_md_stgat_data.py
├── debug_nan.py
├── requirements.txt
└── README.md
```

Some listed research-support modules may exist as placeholders or remain to be completed; the implemented and tested parts are described in the status sections above.

---

## 11. Remaining Work

Recommended order:

1. **Verify evaluation fairness:** confirm the baseline and full MD-STGAT results use identical test windows, target/scaler artifacts, and inverse transformations.
2. **Tune overfitting:** test smaller hidden dimensions and stronger regularization, using validation performance for model selection.
3. **Add a persistence baseline:** compare against predicting future PM2.5 from the most recent observed value.
4. **Run ablations:** separately remove or share the weather GAT, cross-attention, and other proposed components to measure their contributions.
5. **Implement spatial regularization:** integrate a graph-Laplacian smoothness term into the training loss and test its effect. This should be described as spatial smoothness regularization, not automatically as mass conservation.
6. **Implement temporal decomposition:** add the planned trend/residual decomposition. Call it STL-inspired unless actual STL is used.
7. **Improve wind-direction imputation:** process sine/cosine components as a circular vector and normalize the result.
8. **Inspect actual-versus-predicted sequences and attention maps:** use these to understand error patterns, not to claim a causal explanation by themselves.
9. **Repeat training with multiple seeds:** report variability before drawing strong conclusions.
10. **Verify station metadata and document all experimental settings** before preparing a final paper.

---

## 12. Current Summary

The project now has a functioning preprocessing pipeline, a verified baseline, a full MD-STGAT model with dual modality encoders and GATs, weather-conditioned temporal cross-attention, a temporal Transformer, direct 24-hour predictions, and test-set evaluation artifacts.

The current full-model test results are **MAE 46.2771 µg/m³, RMSE 72.4479 µg/m³, and R² 0.3925**. Relative to the recorded baseline, RMSE and R² are better while MAE is worse. The model also shows validation overfitting, and the comparison must be rechecked against identical preprocessing artifacts before making final research claims.

The physics-guided regularizer, temporal decomposition, persistence comparison, and ablation studies remain future work. No performance improvement from those planned components should be claimed until they are implemented and evaluated.
