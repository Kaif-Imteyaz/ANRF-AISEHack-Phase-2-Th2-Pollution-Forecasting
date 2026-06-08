# ANRF AISEHack 2026 - Theme 2: PM2.5 Forecasting over India

**Team:** Colab Survivors | Jamia Hamdard, New Delhi
**Leaderboard Score:** 0.8675 | **Baseline:** 0.7780
**Competition:** ANRF AISEHack 2026, IIIT Hyderabad

---

## What This Does

Forecasts PM2.5 concentration across the Indian subcontinent for the next 16 hours using only 10 hours of historical data. No future meteorological inputs are used at inference. The model predicts a full 140×124 spatial grid at 25 km resolution.

The dataset and competition are provided by ANRF (Anusandhan National Research Foundation). We use their WRF-Chem simulation data under the terms of the ANRF Open License.

---

## Results

| Metric | Baseline | Our Model |
|--------|----------|-----------|
| Leaderboard Score | 0.7780 | **0.8675** |
| Val RMSE (normalised) | ~0.48 | **0.2802** |
| Val RMSE (µg/m³) | ~25.1 | **14.6** |
| MAE (µg/m³) | --- | **6.66** |
| MAPE (%) | --- | **47.25** |
| R² | --- | **0.908** |
| Pearson r | --- | **0.954** |
| Skill Score (peak, h+9) | --- | **44.6%** |
| Parameters | --- | 1,550,290 |

All four US EPA PM2.5 model performance benchmarks (MFB, MFE, NMB, NME) are met at the Goal threshold.

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│              (B, 10, 26, 140, 124)                              │
│         10 hours × 26 channels × 140×124 grid                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │     FRAME ENCODER     │
                │  ┌─────────────────┐  │
                │  │ Stage 1         │  │──── skip features ──┐
                │  │ ResConv block   │  │    (B,10, 64,74,66) │
                │  │ ChannelAttn     │  │                     │
                │  │ stride-2 conv   │  │                     │
                │  └────────┬────────┘  │                     │
                │  ┌────────▼────────┐  │                     │
                │  │ Stage 2         │  │                     │
                │  │ ResConv block   │  │                     │
                │  │ ChannelAttn     │  │                     │
                │  │ stride-2 conv   │  │                     │
                │  └────────┬────────┘  │                     │
                └───────────┼───────────┘                     │
                            │ (B, 10, 128, 37, 33)            │
                ┌───────────▼───────────┐                     │
                │   TEMPORAL TRANSLATOR │                     │
                │  ┌─────────────────┐  │                     │
                │  │ Gated Self-Attn │  │                     │
                │  │   (8 heads ×2)  │  │                     │
                │  └────────┬────────┘  │                     │
                │  ┌────────▼────────┐  │                     │
                │  │ Cross-Attention │  │                     │
                │  │   (8 heads ×2)  │  │                     │
                │  │ 16 learnable    │  │                     │
                │  │ query vectors   │  │                     │
                │  │ (one per hour)  │  │                     │
                │  └────────┬────────┘  │                     │
                └───────────┼───────────┘                     │
                            │ (B, 16, 128, 37, 33)            │
                ┌───────────▼───────────┐                     │
                │     FRAME DECODER     │                     │
                │  ┌─────────────────┐  │                     │
                │  │ TransposedConv1 │◄─┼─────────────────────┘
                │  │  + skip fusion  │  │   skip connection
                │  └────────┬────────┘  │
                │  ┌────────▼────────┐  │
                │  │ TransposedConv2 │  │
                │  └────────┬────────┘  │
                │  ┌────────▼────────┐  │
                │  │  Dual 1×1 heads │  │
                │  │  main + aux     │  │
                │  └────────┬────────┘  │
                └───────────┼───────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                        OUTPUT                                   │
│              (B, 16, 140, 124)                                  │
│         16 future hours × 140×124 PM2.5 grid                   │
└─────────────────────────────────────────────────────────────────┘

  ŷ = h_main(x)  +  0.3 × h_aux(x)  +  0.1 × C_persist
```

**Parameters:** 1,550,290
**Training:** 80 epochs, best checkpoint at epoch 73

---

## Loss Function

Six-term composite loss totalling 1.00:

| Term | Weight | Purpose |
|------|--------|---------|
| Global SMAPE | 0.30 | Matches primary evaluation metric |
| Episode SMAPE | 0.25 | Grid points above 90th percentile (83.13 µg/m³), 3× penalty for underprediction |
| Episode Correlation | 0.20 | Spatial pattern accuracy at high-concentration locations |
| SSIM | 0.10 | Prevents spatial blurring |
| Temporal Gradient | 0.08 | Frame-to-frame coherence across 16-hour horizon |
| Huber (delta=1.0) | 0.07 | Stable pixel-level regression |

Episode-specific terms (SMAPE + Correlation) carry **0.45 of total weight**.

---

## Physics-Informed Input Features

26 channels per timestep. 10 are derived from physical transformations:

- Wind-advected PM2.5 (Lagrangian transport approximation)
- Wind divergence (central differences)
- Wind speed, sin/cos of wind direction
- Persistence channel (last observed PM2.5 frame)
- Diurnal encoding (sin/cos of hour-of-day)
- Latitude and longitude grids

Remaining 16 channels: raw PM2.5, 8 meteorological variables, 7 emission tracers.

---

## Training Configuration

```python
Optimiser    : AdamW (lr=2e-4, weight_decay=1e-4)
Batch size   : 4 (gradient accumulation x4 = effective 16)
LR schedule  : 5-epoch warmup + cosine annealing (T0=30, Tmult=2)
EMA decay    : 0.9995
Epochs       : up to 80, patience 20
TTA          : 4-fold spatial (hflip, vflip, both, original)
```

---

## What Did Not Work

| Attempt | Failure | Fix |
|---------|---------|-----|
| Larger model (BASE_CH 64→96) | No improvement on 2352 training samples | Kept 64 channels |
| Tighter restart (T0=20) | Training destabilised at convergence epoch | Kept T0=30 |
| Full DEC_16 holdout | Val RMSE stuck at 1.14 | Winter PM2.5 is 3–5× higher, needs joint training |
| Spatial self-attention | Memory infeasible at 37×33 grid | Replaced with ChannelAttention |
| Huber loss only | Normalised RMSE >0.35 (+0.07 degradation) | Kept composite loss |

---

## Model Checkpoint

Best checkpoint (`best_p2.pth`) available on Kaggle:
[https://www.kaggle.com/datasets/kaifimtz/colab-survivors-pm2-5-phase-2-best-checkpoint](https://www.kaggle.com/datasets/kaifimtz/colab-survivors-pm2-5-phase-2-best-checkpoint)

---

## Kaggle Notebook

Inference notebook used for submission:
[https://www.kaggle.com/code/kaifimtz/preds-npy](https://www.kaggle.com/code/kaifimtz/preds-npy)

---

## Data

Training and evaluation data (WRF-Chem PM2.5 simulations, 2016–2017) are provided by ANRF as part of AISEHack 2026 Theme 2. Data access is subject to ANRF competition terms.

---

## GenAI Disclosure

Claude (Anthropic) was used as an AI coding and writing assistant during development.
[[Declaration](https://drive.google.com/file/d/14jao__bfAqc1WVS0-dwLOu4HGCIYejbx/view?usp=sharing)]

---

## License
Developed as part of ANRF AISEHack 2026 (Theme 2).
Copyright © 2026 The Authors

Licensed under the **ANRF Open License**. See [LICENSE](./LICENSE) for full terms. 
This license is compatible with the MIT license under the global interoperability clause.
