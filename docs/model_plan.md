# Model Plan

This document tracks the modeling roadmap: what's done, what's next, and the plan for the wind-aware graph model.

## Status

| Stage | Status |
| ----- | ------ |
| Data collection (OpenAQ + Open-Meteo) | ✅ Done |
| Data validation | ✅ Done |
| Dataset profiling | ✅ Done |
| Timestamp normalization | ✅ Done |
| Dataset merging | ✅ Done |
| Dataset trimming | ✅ Done |
| Feature engineering | ✅ Done |
| Dataset split (train/val/test) | ✅ Done |
| Reusable `BaseModel` class | ✅ Done |
| Persistence baseline | ✅ Done |
| Linear Regression | ✅ Done |
| Random Forest | Done |
| XGBoost | Done |
| LSTM | ⬜ Not started |
| Transformer | ⬜ Not started |
| Graph construction (wind-weighted station graph) | ⬜ Not started |
| GAT-GRU (wind-aware spatio-temporal model) | ⬜ Not started |
| Explainability | ⬜ Not started |

## Modeling Strategy

Models are evaluated in increasing order of complexity so that each one has to beat a clear, cheaper baseline before it's worth the added cost:

1. **Persistence baseline** — `PM2.5(t+1) = PM2.5(t)`. The minimum bar every model must clear.
2. **Classical ML** (Linear Regression → Random Forest → XGBoost) — tabular models using the engineered features in `MODEL_FEATURE_COLUMNS` (`scripts/config.py`), with no notion of sequence or spatial structure.
3. **Sequence models** (LSTM → Transformer) — capture temporal dependencies within a single station's history that lag/rolling features can only approximate.
4. **Wind-aware GAT-GRU** — the target architecture. Stations become graph nodes; edges are weighted by wind direction and speed between station pairs, so the model can learn how pollution is transported between locations rather than treating each station in isolation.

All models share the same evaluation contract via `models/base_model.py`: load the test split, predict, and report MAE, RMSE, and R2 to per-model `metrics.csv`, `summary.csv`, and `predictions.csv` files. Tuned baselines use validation-only selection before final test evaluation.

## Next Steps

- LSTM
- Transformer
- Graph construction from station geography + wind field
- Wind-aware GAT-GRU
