# ANRF AISEHack 2026 | Theme 2: PM2.5 Forecasting over India

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
| Parameters | --- | 1,550,290 |

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
                │  │ Stage 1         │  │
                │  │ ResConv block   │  │──── skip features ──┐
                │  │ ChannelAttn     │  │    (B,10, 64,74,66) │
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
                │                       │                     │
                │  10 history steps     │                     │
                │  ┌─────────────────┐  │                     │
                │  │ Gated Self-Attn │  │                     │
                │  │   (8 heads ×2)  │  │                     │
                │  └────────┬────────┘  │                     │
                │           │           │                     │
                │  ┌────────▼────────┐  │                     │
                │  │ Cross-Attention │  │                     │
                │  │   (8 heads ×2)  │  │                     │
                │  │                 │  │                     │
                │  │ 16 learnable    │  │                     │
                │  │ query vectors   │  │                     │
                │  │ (one per hour)  │  │                     │
                │  └────────┬────────┘  │                     │
                └───────────┼───────────┘                     │
                            │ (B, 16, 128, 37, 33)            │
                ┌───────────▼───────────┐                     │
                │     FRAME DECODER     │                     │
                │  ┌─────────────────┐  │                     │
                │  │ TransposedConv1 │  │                     │
                │  │    upsample     │◄─┼─────────────────────┘
                │  │  + skip fusion  │  │   skip connection
                │  └────────┬────────┘  │
                │  ┌────────▼────────┐  │
                │  │ TransposedConv2 │  │
                │  │    upsample     │  │
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
LR schedule  : 5-epoch warmup + cosine annealing (T0=40, Tmult=2)
EMA decay    : 0.9995
Epochs       : up to 80, patience 20
TTA          : 16 passes (4 seasonal encodings x 4 spatial transforms)
```

---

## Model Checkpoint

Best checkpoint (`best_p2.pth`) is available on Kaggle:  
[https://www.kaggle.com/datasets/kaifimtz/colab-survivors-pm2-5-phase-2-best-checkpoint](https://www.kaggle.com/datasets/kaifimtz/colab-survivors-pm2-5-phase-2-best-checkpoint)

---

## Kaggle Notebook

Inference notebook used for submission:  
[https://www.kaggle.com/code/kaifimtz/preds-npy](https://www.kaggle.com/code/kaifimtz/preds-npy)

---

## What Did Not Work

| Attempt | Failure | Fix |
|---------|---------|-----|
| Larger model (BASE_CH 64→96) | No improvement, stopped at epoch 10 | Kept 64 channels |
| T0=30 cosine restart | Training destabilised at convergence epoch | Changed to T0=40 |
| Full DEC_16 holdout | Val RMSE stuck at 1.14 | Winter PM2.5 is 3-5x higher, needs joint training |
| Spatial self-attention | Memory infeasible at 37x33 grid | Replaced with ChannelAttention |


---

## Data

Training and evaluation data (WRF-Chem PM2.5 simulations, 2016-2017) are provided by ANRF as part of AISEHack 2026 Theme 2. Data access is subject to ANRF competition terms.

---

## GenAI Disclosure

Claude (Anthropic) was used as an AI coding and writing assistant during development.
[[link here](https://drive.google.com/file/d/14jao__bfAqc1WVS0-dwLOu4HGCIYejbx/view?usp=sharing)]

---

## License

ANRF Open License | Copyright © 2026 Colab Survivors  
See [LICENSE](./LICENSE) for full terms.

This license is compatible with the MIT license under the global interoperability clause.
