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
| Validation feature ablation | Done |
| Rolling-origin validation | Done |
| LSTM sequence dataset validation | Done |
| First station-specific LSTM baseline | Done |
| Transformer | ⬜ Not started |
| Graph construction (wind-weighted station graph) | ⬜ Not started |
| GAT-GRU (wind-aware spatio-temporal model) | ⬜ Not started |
| Explainability | ⬜ Not started |

## Modeling Strategy

Models are evaluated in increasing order of complexity so that each one has to beat a clear, cheaper baseline before it's worth the added cost:

1. **Persistence baseline** — `PM2.5(t+1) = PM2.5(t)`. The minimum bar every model must clear.
2. **Classical ML** (Linear Regression → Random Forest → XGBoost) — tabular models using the engineered features in `MODEL_FEATURE_COLUMNS` (`scripts/config.py`), with no notion of sequence or spatial structure.
3. **Sequence models** (LSTM → Transformer) — capture temporal dependencies within a single station's history that lag/rolling features can only approximate.
4. **Wind-aware GAT-GRU** — the target architecture. Stations become graph nodes; edges are weighted by wind direction and speed between station pairs, so the model can learn how pollution is transported between locations rather than treating each station in isolation. Since meteorological wind direction is the direction FROM which wind blows, future directed A-to-B edge alignment should use `transport_direction = (wind_direction + 180) % 360` before comparing with source-to-target bearings.

All models share the same evaluation contract via `models/base_model.py`: load the test split, predict, and report MAE, RMSE, and R2 to per-model `metrics.csv`, `summary.csv`, and `predictions.csv` files. Tuned baselines use validation-only selection before final test evaluation.

## LSTM Baseline Result

The first LSTM baseline is complete as a validation-only experiment. It
uses `data/processed/featured/`, 24-hour sequence-native windows, 11
input features, and station-specific PyTorch LSTMs. It does not use
handcrafted lag/rolling columns and does not evaluate the final test
split.

Validation result:

```text
Native LSTM validation sequences: 22,657
Native LSTM pooled RMSE: 15.036
Native LSTM pooled MAE: 9.619
Native LSTM pooled R2: 0.834

Matched validation rows: 22,477
Persistence pooled RMSE: 11.848
Frozen RF pooled RMSE: 12.233
LSTM pooled RMSE: 15.021
```

The first LSTM does not yet justify moving directly to a more complex
Transformer. Persistence and RF remain stronger on validation. Future
sequence work should diagnose whether the issue is station-specific data
sparsity, overfitting, architecture simplicity, or whether one-hour
forecasting is already dominated by current PM2.5 persistence.

## Next Steps

- Diagnose the LSTM failure mode before increasing sequence complexity:
  compare per-station errors, inspect small-station behavior, and consider
  a simpler sequence baseline or pooled model.
- Wind/spatial interaction design, using validation evidence only and
  avoiding premature feature-set changes based on the already-observed
  test split
- Transformer
- Graph construction from station geography + wind field
- Wind-aware GAT-GRU
