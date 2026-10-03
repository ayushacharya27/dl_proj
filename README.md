# MD-STGAT

**Multi-Modal Decoupled Spatio-Temporal Graph Attention Network for Multi-Station PM2.5 Forecasting**

This project forecasts PM2.5 concentrations across **12 Beijing air-quality monitoring stations** using pollution, meteorological, spatial, and temporal information.

The implementation is being developed incrementally so that each research component can be tested independently before the complete model is trained. Because debugging a giant neural network all at once is apparently considered character building.

---

## 1. Current Project Status

### Completed

- Raw Beijing Multi-Site Air Quality data loading and validation
- 12-station alignment and timestamp handling
- Missing-value analysis
- Leakage-aware chronological train/validation/test split
- Temporal + same-timestamp spatial imputation pipeline
- Train-only feature scaling
- Wind-direction encoding using sine/cosine
- 72-hour input window
- 24-hour multi-horizon prediction target
- PyTorch dataset and dataloaders
- Pollution encoder
- Weather encoder
- Spatial GAT for pollution
- Spatial GAT for weather
- Temporal Transformer
- Corrected temporal cross-modal attention
- Baseline model
- Baseline training
- Baseline evaluation
- Horizon-wise and station-wise metrics
- Full MD-STGAT architecture file

### Full-model status

The complete architecture has been implemented and the next immediate task is to verify its forward pass using the real station graph, followed by training and evaluation.

---

# 2. Project Architecture

The current intended pipeline is:

```text
                Pollution Features
                       │
                       ▼
               Pollution Encoder
                       │
                       ▼
                 Pollution GAT
                       │
                       │
                       ├──────────────┐
                       │              │
                       │         Key / Value
                       │              │
                       ▼              ▼
                  Cross-Modal Attention
                       ▲
                       │ Query
                       │
                       │
                 Weather GAT
                       ▲
                       │
                Weather Encoder
                       ▲
                       │
                Weather Features

                       │
                       ▼
             Fused Representation
                       │
                       ▼
            Temporal Transformer
                       │
                       ▼
                 Forecast Head
                       │
                       ▼
              24-hour PM2.5 forecast
                 for 12 stations
```

Cross-attention is currently designed as:

```text
Weather = Query
Pollution = Key
Pollution = Value
```

The attention operates over the **72-hour pollution history for each station**.

---

# 3. Dataset

The project uses the **UCI Beijing Multi-Site Air Quality Dataset** containing hourly observations from 12 monitoring stations.

### Stations

```text
Aotizhongxin
Changping
Dingling
Dongsi
Guanyuan
Gucheng
Huairou
Nongzhanguan
Shunyi
Tiantan
Wanliu
Wanshouxigong
```

Each station has **35,064 hourly records**.

Total combined observations:

```text
420,768 station-time rows
```

### Raw features

```text
No
year
month
day
hour
PM2.5
PM10
SO2
NO2
CO
O3
TEMP
PRES
DEWP
RAIN
wd
WSPM
station
```

---

# 4. Model Inputs

### Pollution modality

```text
PM2.5
PM10
SO2
NO2
CO
O3
```

Input size:

```text
6 features
```

### Weather modality

```text
TEMP
PRES
DEWP
RAIN
WSPM
wind-direction sin
wind-direction cos
```

Input size:

```text
7 features
```

### Sequence configuration

```text
Historical window = 72 hours
Forecast horizon   = 24 hours
Stations           = 12
```

Expected tensors:

```text
Pollution:
[B, 72, 12, 6]

Weather:
[B, 72, 12, 7]

Target:
[B, 24, 12]
```

---

# 5. Data Preprocessing

The preprocessing pipeline is in:

```text
data/prepare_dataset.py
```

The important rule is:

> The chronological split is performed before fitting imputers/scalers so that future information does not leak into training.

### Split

```text
Train = 70%
Validation = 15%
Test = 15%
```

Current split:

```text
Train:
2013-03-01 00:00
→ 2015-12-18 15:00

Validation:
2015-12-18 16:00
→ 2016-07-24 19:00

Test:
2016-07-24 20:00
→ 2017-02-28 23:00
```

### Missing values

Current preprocessing strategy:

1. Causal temporal forward-fill for short gaps
2. Same-timestamp spatial inverse-distance interpolation using neighboring stations
3. Train-set median fallback
4. Preserve chronological separation between train/validation/test

All three processed splits currently contain zero missing values after preprocessing.

### Scaling

Standardization is fitted **only on the training data**.

Saved scalers:

```text
data/processed/pollution_scaler.pkl
data/processed/weather_scaler.pkl
```

---

# 6. Generated Dataset Files

After preprocessing, the following files are generated:

```text
data/processed/
├── train.npz
├── val.npz
├── test.npz
├── pollution_scaler.pkl
└── weather_scaler.pkl
```

Current shapes:

```text
Train pollution = (24449, 72, 12, 6)
Train weather   = (24449, 72, 12, 7)
Train target    = (24449, 24, 12)

Validation pollution = (5165, 72, 12, 6)
Validation weather   = (5165, 72, 12, 7)
Validation target    = (5165, 24, 12)

Test pollution = (5165, 72, 12, 6)
Test weather   = (5165, 72, 12, 7)
Test target    = (5165, 24, 12)
```

---

# 7. Graph Construction

Graph code:

```text
data/graph.py
data/station_metadata.py
```

The current graph is a static physical-proximity graph.

Method:

```text
Haversine distance
        ↓
4 nearest neighbors per station
        ↓
Directed station graph
```

With 12 stations:

```text
12 × 4 = 48 directed edges
```

The coordinates in `station_metadata.py` should be verified against the authoritative dataset metadata before being used as exact geographic values in a paper.

The current graph represents **physical proximity**, not wind direction.

---

# 8. Data Loading

Dataset implementation:

```text
data/dataset.py
```

Example:

```python
from data.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32,
    num_workers=2
)
```

Expected batch:

```text
Pollution: torch.Size([32, 72, 12, 6])
Weather:   torch.Size([32, 72, 12, 7])
Target:    torch.Size([32, 24, 12])
```

---

# 9. Model Components

## 9.1 Pollution Encoder

File:

```text
models/encoders.py
```

Current structure:

```text
Linear(6 → 64)
LayerNorm
GELU
Dropout
```

Output:

```text
[B, T, N, 64]
```

---

## 9.2 Weather Encoder

File:

```text
models/weather_encoder.py
```

Current structure mirrors the pollution encoder:

```text
Linear(7 → 64)
LayerNorm
GELU
Dropout
```

Output:

```text
[B, T, N, 64]
```

---

## 9.3 Spatial GAT

File:

```text
models/gat.py
```

Current configuration:

```text
Input dimension = 64
Hidden dimension = 64
Attention heads = 4
Dropout = 0.1
```

The implementation batches all `B × T` graph snapshots rather than running a Python loop over every graph.

Two independent GAT blocks are now used:

```text
Pollution GAT
Weather GAT
```

This preserves the decoupled modality design.

---

## 9.4 Cross-Modal Attention

File:

```text
models/cross_attention.py
```

Current design:

```text
Weather → Query
Pollution → Key
Pollution → Value
```

For every station, the weather representation can attend across the 72-hour pollution history.

Input:

```text
Pollution = [B, 72, 12, 64]
Weather   = [B, 72, 12, 64]
```

Output:

```text
Fused = [B, 72, 12, 64]
```

Attention representation:

```text
[B, 12, 72, 72]
```

Current tested configuration:

```text
Parameters = 49,984
```

The previous one-token implementation was discarded because a `[1,1]` attention matrix does not provide meaningful attention over history.

---

## 9.5 Temporal Transformer

File:

```text
models/temporal.py
```

Current configuration:

```text
Hidden dimension = 64
Attention heads = 4
Transformer layers = 2
Feed-forward dimension = 256
Dropout = 0.1
```

Temporal attention is applied independently to each station.

Input:

```text
[B, 72, 12, 64]
```

Output:

```text
[B, 72, 12, 64]
```

---

## 9.6 Baseline Model

File:

```text
models/baseline.py
```

Current baseline:

```text
Pollution Encoder
        ↓
Pollution GAT
        ↓
Temporal Transformer
        ↓
Forecast Head
```

Parameter count:

```text
115,416
```

The baseline was trained successfully on the RTX 3050 Laptop GPU.

---

## 9.7 Full MD-STGAT

File:

```text
models/md_stgat.py
```

Current pipeline:

```text
Pollution Encoder
        ↓
Pollution GAT
        ↓
        P ───────┐
                 │
                 │ Key / Value
                 ▼
            Cross Attention
                 ▲
                 │ Query
                 │
        W ───────┘
        ↑
Weather GAT
        ↑
Weather Encoder

        ↓
Temporal Transformer
        ↓
Forecast Head
        ↓
[B, 24, 12]
```

The full-model forward pass must now be checked with the real graph before training.

---

# 10. Baseline Training

Training file:

```text
training/train.py
```

Current configuration:

```text
Batch size = 32
Maximum epochs = 30
Optimizer = AdamW
Learning rate = 1e-3
Weight decay = 1e-4
Gradient clipping = 1.0
Scheduler = ReduceLROnPlateau
Early stopping patience = 5
```

Best baseline checkpoint:

```text
checkpoints/baseline_best.pt
```

Best validation epoch:

```text
Epoch 6
Validation MSE = 0.627643
```

Training was stopped early after validation performance stopped improving.

---

# 11. Baseline Results

Evaluation file:

```text
evaluation/evaluate.py
```

Test results for the current baseline:

```text
MAE  = 45.1195 µg/m³
RMSE = 74.3686 µg/m³
R²   = 0.3599
```

The model's error increases with the forecasting horizon, which is expected for a 24-hour recursive-information problem, although this model performs direct multi-horizon prediction.

### Horizon RMSE

```text
H01  30.49
H02  36.97
H03  43.43
H04  49.08
H05  54.02
H06  58.49
H07  62.38
H08  65.92
H09  68.90
H10  71.70
H11  74.22
H12  76.49
H13  78.86
H14  80.88
H15  82.77
H16  83.73
H17  85.17
H18  86.71
H19  87.80
H20  89.02
H21  89.59
H22  91.10
H23  91.93
H24  92.29
```

Saved evaluation files:

```text
evaluation/baseline_predictions.npz
evaluation/horizon_metrics.csv
```

---

# 12. How To Run The Project

Run commands from the project root:

```bash
cd ~/dl_proj
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

If your environment has a different name, activate that environment instead.

---

## Step 1: Check the environment

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

The current setup successfully used the NVIDIA RTX 3050 Laptop GPU.

---

## Step 2: Prepare the dataset

Run:

```bash
python -m data.prepare_dataset
```

This should generate:

```text
data/processed/train.npz
data/processed/val.npz
data/processed/test.npz
data/processed/pollution_scaler.pkl
data/processed/weather_scaler.pkl
```

---

## Step 3: Test the dataset loader

```bash
python -m data.dataset
```

Verify:

```text
Pollution batch = [32,72,12,6]
Weather batch   = [32,72,12,7]
Target batch    = [32,24,12]
```

---

## Step 4: Test individual model components

Pollution encoder:

```bash
python -m models.encoders
```

Weather encoder:

```bash
python -m models.weather_encoder
```

Spatial GAT:

```bash
python -m models.gat
```

Cross-modal attention:

```bash
python -m models.cross_attention
```

Temporal Transformer:

```bash
python -m models.temporal
```

Baseline:

```bash
python -m models.baseline
```

Full MD-STGAT:

```bash
python -m models.md_stgat
```

For the full model, confirm:

```text
Pollution input: [4,72,12,6]
Weather input:   [4,72,12,7]
Forecast:        [4,24,12]
Attention:       [4,12,72,72]
```

Use the actual station graph from `data/graph.py` for the final test rather than a random graph.

---

# 13. Current Directory Structure

```text
MD-STGAT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── preprocessing.py
│   ├── graph.py
│   ├── station_metadata.py
│   ├── imputation.py
│   ├── prepare_dataset.py
│   └── dataset.py
│
├── models/
│   ├── encoders.py
│   ├── weather_encoder.py
│   ├── gat.py
│   ├── cross_attention.py
│   ├── temporal.py
│   ├── baseline.py
│   └── md_stgat.py
│
├── losses/
│   ├── prediction_loss.py
│   └── physics_loss.py
│
├── training/
│   ├── train.py
│   ├── validate.py
│   └── early_stopping.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   ├── baselines.py
│   └── ablation.py
│
├── configs/
│   └── config.yaml
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_graph_analysis.ipynb
│   └── 03_results.ipynb
│
├── checkpoints/
│
├── requirements.txt
│
└── README.md
```

---

# 14. Next Steps

The remaining work should be done in this order.

## Phase 1: Validate the full architecture

Run:

```bash
python -m models.md_stgat
```

Make sure the full model produces:

```text
Forecast: [B,24,12]
Attention: [B,12,72,72]
```

Also verify GPU memory usage.

---

## Phase 2: Train MD-STGAT

Create or update:

```text
training/train_md_stgat.py
```

Use the same data splits and training protocol as the baseline so the comparison is fair.

Start with:

```text
Loss = MSE
Optimizer = AdamW
Early stopping
Gradient clipping
Learning-rate scheduler
```

Do not change several training variables at once when comparing models.

---

## Phase 3: Evaluate MD-STGAT

Run evaluation using the test set and inverse-transform PM2.5 values.

Report:

```text
Overall MAE
Overall RMSE
R²
Horizon-wise MAE
Horizon-wise RMSE
Station-wise MAE
Station-wise RMSE
```

Compare against the baseline.

Important:

> Do not claim that MD-STGAT improves performance until the actual test results show it.

---

## Phase 4: Add the spatial regularization term

Planned loss:

```text
Total Loss =
Prediction Loss
+
λ × Graph Laplacian Smoothness Loss
```

The planned spatial term is based on graph smoothness:

```text
yᵀ L y
```

where `L` is the graph Laplacian.

This encourages spatially connected stations to have compatible predictions.

It should be described as a **spatial smoothness regularizer**, not as a generic "mass conservation" law.

---

## Phase 5: Build the ablation study

The research contribution needs controlled experiments.

Minimum useful ablations:

```text
A. Pollution Encoder + GAT + Transformer
B. A + Weather Encoder/GAT
C. B + Cross Attention
D. C + Temporal Decomposition
E. D + Spatial Regularization
F. Full MD-STGAT
```

This allows each proposed component to be evaluated separately.

---

## Phase 6: Add temporal decomposition

The intended idea is:

```text
Input sequence
      ↓
Trend component
      +
Residual component
      ↓
Temporal modeling
```

The implementation should be described accurately.

If a moving-average style decomposition is used, call it:

```text
STL-inspired decomposition
```

Do not call it full STL unless the actual STL procedure is implemented.

---

## Phase 7: Establish stronger baselines

Before writing performance claims, include simple reference models.

At minimum:

```text
Persistence / last-value baseline
Historical mean baseline
Current neural baseline
Full MD-STGAT
```

Additional published-model comparisons can be added later depending on the research scope.

---

## Phase 8: Visualization and analysis

Generate:

```text
Actual vs predicted PM2.5
Prediction error by horizon
Prediction error by station
Attention heatmaps
Training/validation loss curves
Station graph visualization
```

For cross-attention:

```text
X-axis = historical pollution timestep
Y-axis = weather query timestep
```

This can help analyze whether the model is actually learning meaningful temporal relationships rather than merely producing numbers with neural-network confidence.

---

# 15. Research Checklist Before Final Results

Before considering the model paper-ready, verify:

```text
[ ] Dataset source documented
[ ] Station metadata verified
[ ] Chronological split documented
[ ] No train/test preprocessing leakage
[ ] Missing-data procedure documented
[ ] Graph construction documented
[ ] Baseline implemented
[ ] Full MD-STGAT trained
[ ] Test metrics computed after training
[ ] Persistence baseline included
[ ] Ablation experiments completed
[ ] Hyperparameters documented
[ ] Random seeds documented
[ ] Multiple runs / variability reported
[ ] Attention visualizations generated
[ ] Error analysis completed
```

---

# 16. Immediate Command Sequence

The next practical sequence is:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Verify preprocessing artifacts
python -m data.prepare_dataset

# 3. Verify dataset
python -m data.dataset

# 4. Verify individual components
python -m models.encoders
python -m models.weather_encoder
python -m models.gat
python -m models.cross_attention
python -m models.temporal

# 5. Verify complete model
python -m models.md_stgat
```

After the full forward pass succeeds:

```text
FULL MODEL FORWARD PASS
        ↓
MD-STGAT TRAINING
        ↓
TEST EVALUATION
        ↓
BASELINE COMPARISON
        ↓
SPATIAL REGULARIZATION
        ↓
ABLATIONS
        ↓
FINAL ANALYSIS
```

---

# 17. Current Scientific State

At the current stage, the project has a functioning data pipeline, a validated baseline, and the main multimodal architecture components.

The baseline establishes a reference point:

```text
MAE  = 45.1195 µg/m³
RMSE = 74.3686 µg/m³
R²   = 0.3599
```

The next meaningful milestone is **training the complete MD-STGAT under the same experimental conditions and comparing it against this baseline**.

No performance improvement should be assumed in advance. Neural networks, like politicians and startup pitches, are extremely enthusiastic about their potential and occasionally less impressive when measured.
