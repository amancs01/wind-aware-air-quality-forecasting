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
| Residual station-specific LSTM baseline | Done |
| Transformer | ⬜ Not started |
| Graph design audit | Done |
| Graph construction (wind-weighted station graph) | In progress |
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

## Residual LSTM Baseline Result

The persistence-anchored residual LSTM keeps the same architecture and
training setup as the direct LSTM, but predicts the one-hour PM2.5
change:

```text
delta_pm25 = PM2.5(t+1) - PM2.5(t)
prediction = PM2.5(t) + predicted_delta
```

Validation result:

```text
Native residual LSTM validation sequences: 22,657
Native residual LSTM pooled RMSE: 10.583
Native residual LSTM pooled MAE: 6.577
Native residual LSTM pooled R2: 0.918

Matched validation rows: 22,477
Direct LSTM pooled RMSE: 15.021
Residual LSTM pooled RMSE: 10.588
Persistence pooled RMSE: 11.848
Frozen RF pooled RMSE: 12.233
```

Residual learning materially improves the station-specific LSTM and
clears Persistence and RF on pooled validation. It should be treated as
the current temporal baseline for future graph-aware work.

## Graph Design Decision

Graph implementation should not proceed from the current
`station_mapping.csv` because it is keyed by human station name and drops
duplicate PM2.5 sensors. The graph identity key must be the canonical
featured `dataset_name` with `pm25_sensor_id` retained.

Policy:

```text
Canonical graph registry: 56 sensor-qualified featured datasets
First supervised graph node set: 51 train+validation model-usable nodes
Static candidates: directed expansion of the symmetric KNN union
Wind source for A->B: source node A
```

Dynamic wind edge equation:

```text
transport_direction_A(t) = (wind_direction_A(t) + 180) % 360
alignment_AB(t) = max(0, cos(angle_difference(transport_direction_A,
                                              bearing_A_to_B)))
speed_factor_A(t) = wind_speed_A(t) / (wind_speed_A(t) + 5)
distance_factor_AB = exp(-distance_AB / lambda_d)
raw_weight_AB(t) = candidate_AB * alignment_AB(t) *
                   speed_factor_A(t) * distance_factor_AB
```

Before implementing dynamic edges, regenerate the mapping, distance
matrix, bearing matrix, and static candidate edge list from the corrected
sensor-qualified node registry. See `docs/graph_design_audit.md`.

## Graph Timeline Protocol

The original global 2021-2026 70/15/15 split is not suitable for graph
training because train and validation graph windows contain too few
simultaneously supervised nodes. The old global test split has also been
inspected for graph coverage, so it is no longer a pristine graph-model
test split. No graph-model performance has been inspected.

Graph timeline redesign evaluated two deployment/data-availability
policies without training a model:

```text
Policy A, all-node common era:
51 nodes, 2026-05-10 16:00 to 2026-07-11 22:00, 62.3 days
train/validation/test usable windows: 971 / 200 / 201
nodes with train/validation targets: 45/51 and 28/51

Policy B, core-network era:
41 nodes, 2025-11-26 15:00 to 2026-07-11 22:00, 227.3 days
train/validation/test usable windows: 3,691 / 718 / 795
nodes with train/validation targets: 39/41 and 37/41
```

Policy B is the recommended graph-specific candidate protocol, pending
review, because it preserves a large fixed spatial cohort and avoids the
tiny all-node common era. GAT/GAT-GRU training should remain blocked
until the graph-specific split and cohort protocol are frozen.

Policy-B per-node training support audit:

```text
41 context nodes
train target distribution: min 0, Q1 991, median 1,790, Q3 2,657, max 3,391
zero-train nodes: 2
nodes with 1-99 train targets: 0
nodes with 100-499 train targets: 4
nodes with >=500 train targets: 35
```

Recommended refinement: keep all 41 nodes as graph/context nodes, but
freeze a 39-node supervised forecast/evaluation cohort by excluding the
two zero-train nodes from supervised loss and metrics. This choice is
based on train availability only, not validation or test performance.

The Policy-B graph dataset protocol is now frozen in
`data/processed/graph/policy_b/`:

```text
era: 2025-11-26 15:00 to 2026-07-11 22:00
context nodes: 41
supervised forecast/evaluation nodes: 39
context directed edges: 204
train/validation/test windows: 3,691 / 718 / 795
train/validation/test supervised targets: 73,662 / 4,766 / 14,542
```

Scaling remains a loader responsibility: fit input and residual-target
scalers on train windows only using saved masks, then apply unchanged
scalers to validation/test.

## Next Steps

- Move toward graph construction and wind-aware station interaction,
  using residual LSTM as the temporal baseline to beat.
- Wind/spatial interaction design, using validation evidence only and
  avoiding premature feature-set changes based on the already-observed
  test split
- Implement and validate the graph dataset loader against the frozen
  Policy-B artifacts before training GAT/GAT-GRU.
- Defer Transformer work until graph design or residual sequence
  diagnostics justify it.
- Wind-aware GAT-GRU only after protocol review.
