# ANRF AISEHack 2026 | Theme 2 | Pollution-Forecasting-IITD
Scientific ML models that can accurately forecast short-term PM2.5 concentration fields over India.

# Project tittle : India in the Haze - Country-Level PM2.5 Concentration Forecasting

[![License: ANRF Open](https://img.shields.io/badge/License-ANRF%20Open-teal.svg)](./LICENSE)
![Score](https://img.shields.io/badge/Score-0.8675-green)

---

## Overview

Phase 2 solution for ANRF AISEHack 2026 Theme 2. Forecasts PM2.5 concentrations across India (140×124 grid, 25 km resolution) predicting 16 future hours from a strictly 10-hour historical lookback with no future meteorological inputs.

**Leaderboard score: 0.8675 | Baseline: 0.7780**

---


## Model Architecture

```
Input  (B, 10, 26, 140, 124)
            |
      FrameEncoder
      ResConv + ChannelAttention + stride-2 × 2
            |
      TemporalTranslator
      Self-attention on history → Cross-attention to 16 future slots
            |
      FrameDecoder
      TransposedConv upsampling + auxiliary head
            |
Output (B, 16, 140, 124)
```

**Parameters:** 1,550,290 | **GPU:**  T4 14.6 GB | **Training:** ~4.97 hrs (80 epochs)

---

## Key Contributions

**Phase 2 Composite Loss** directly optimises all three evaluation metrics:
- Global SMAPE (0.30) + Asymmetric Episode SMAPE (0.35) + Episode Correlation (0.20) + SSIM (0.10) + Huber (0.05)
- Underprediction of PM2.5 spikes penalised 3× harder than overprediction

**Physics-informed features** (26 channels): PM2.5 history, wind advection warp, persistence, met variables, emissions, divergence, diurnal encoding, lat/lon

**Inference:** 4 seasonal encodings × 4 spatial TTA = 16 passes per sample with persistence blending

---

## Installation

```bash
pip install torch numpy scipy statsmodels
```

---

## Training

```bash
python phase2_v1.py   
```

Set `ROOT` path at top of each script to your data directory.

---

Set three paths at top of `inference.py`: `CHECKPOINT_PATH`, `DATA_ROOT`, `OUTPUT_PATH`. The checkpoint stores all normaliser statistics internally — no other training files needed.

---

## Results

| Metric | Baseline | Our Model |
|---|---|---|
| Leaderboard Score | 0.7780 | **0.8675** |
| Val RMSE (norm) | ~0.48 | **0.2802** |
| Val RMSE (µg/m³) | ~25.1 | **14.6** |

---

## Model Checkpoint

[https://www.kaggle.com/datasets/kaifimtz/colab-survivors-pm2-5-phase-2-best-checkpoint]

Only the top submission checkpoint is provided: `best_p2.pth`

---

## Kaggle Notebook

[https://www.kaggle.com/code/kaifimtz/preds-npy]

---

## What Did Not Work

1. **Larger model (BASE_CH 64→96):** No improvement; stopped at epoch 10.
2. **100 epochs with large model:** Degraded after epoch 35 due to lr restart (T_0=30). Fixed by T_0=40.
3. **Full DEC_16 holdout:** Val stuck at 1.14 — winter PM2.5 is 3-5× higher than training seasons.
4. **SpatialAttention:** 28 GB OOM on T4. Replaced with ChannelAttention.

---

## GenAI Disclosure

Claude (Anthropic) used as AI coding assistant. Full prompt log: [[link here](https://drive.google.com/file/d/14jao__bfAqc1WVS0-dwLOu4HGCIYejbx/view?usp=sharing)]

---

## License

ANRF Open License - Copyright © 2026 Colab Survivors. See LICENSE.

---

**Team: Colab Survivors | ANRF AISEHack 2026, IIIT Hyderabad**
