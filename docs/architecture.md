# Project Architecture

This document describes the end-to-end pipeline, from raw data collection to model evaluation. Each stage corresponds to one or more numbered scripts in `scripts/`, run in order via `scripts/run_pipeline.py`.

## Pipeline Overview

```
Station Discovery (00)
        ↓
Weather Download (01)  +  Air Quality Download (02)
        ↓
Validation (03)
        ↓
Profiling (04)
        ↓
Timestamp Normalization (05)
        ↓
Merge Weather + Air Quality (06)
        ↓
Trim Leading Missing PM2.5 Records (07)
        ↓
Validate Merged Timestamps (07b)
        ↓
Feature Engineering (08)
        ↓
Dataset Preparation (09)
        ↓
Train / Validation / Test Split (10)
        ↓
Verify Split (11)
        ↓
Persistence Baseline (12)
        ↓
Validation-Based Linear Model Selection (13a)
        ↓
Machine Learning Models (13 — Ridge, 15 — Random Forest, 16 — XGBoost)
        ↓
Feature Correlation Analysis (14)
        ↓
[Planned] Graph Construction → GAT-GRU Model → Explainability → Dashboard
```

## Stage Details

| Stage | Script(s) | Purpose |
| ----- | --------- | ------- |
| Station Discovery | `00_discover_stations.py` | Query OpenAQ for stations within the study radius and record their PM2.5/PM1 sensor IDs |
| Data Collection | `01_download_weather.py`, `02_download_air_quality.py` | Download per-station hourly weather (Open-Meteo) and PM2.5 measurements (OpenAQ), resumable and retry-safe |
| Validation | `03_validate_data.py` | Check for missing values, duplicate timestamps, and malformed files |
| Profiling | `04_profile_dataset.py` | Summarize coverage per station (row counts, date ranges, completeness) |
| Preprocessing | `05_preprocess_data.py`, `06_analyze_merge_data.py`, `07_trim_data.py`, `07b_validate_timestamps.py` | Normalize timestamps, merge weather with air quality, trim leading rows with no PM2.5 label, and re-validate the merged output |
| Feature Engineering | `08_feature_engineering.py` | Add lag features, rolling statistics, cyclical time encodings, and physical wind vector (u/v) components |
| Dataset Preparation | `09_prepare_dataset.py` | Build the final modeling table, including the one-hour-ahead `target_pm2_5` label |
| Splitting | `10_split_dataset.py`, `11_verify_split.py` | Chronological train / validation / test split per station, with integrity checks |
| Baseline & Models | `12_persistence_baseline.py`, `13a_tune_ridge.py`, `13_linear_regression.py`, `15a_tune_random_forest.py`, `15_random_forest.py`, `16a_tune_xgboost.py`, `16_xgboost.py` | Select Ridge, Random Forest, and XGBoost configurations on validation data, then evaluate frozen production baselines using the shared fair evaluation frame |
| Analysis | `14_feature_correlation.py` | Compute feature statistics and correlations against the forecasting target |

## Design Principles

- **Reusable infrastructure**: `downloaders/base_downloader.py` and `clients/http_client.py` centralize retry, backoff, and file-skip logic so both downloaders share the same reliability guarantees.
- **Reusable modeling**: `models/base_model.py` standardizes dataset loading, evaluation, and result export so every forecasting model (persistence, linear regression, and future models) reports metrics consistently.
- **Idempotent downloads**: Both downloaders skip files that already exist, so the pipeline can be safely re-run without re-fetching existing data.
- **Central configuration**: All paths, constants, and the model feature list live in `scripts/config.py`, avoiding hardcoded values scattered across scripts.

## Baseline Evaluation Protocol

Persistence and Ridge are compared on the same reusable evaluation frame.
For each test split, `BaseModel.prepare_evaluation_frame()` keeps only
rows where every `MODEL_FEATURE_COLUMNS` value and `target_pm2_5` are
present. This makes the naive persistence equation fair against Ridge:

```text
prediction(t + 1) = pm2_5(t)
```

The persistence prediction itself is unchanged; only its scored rows are
restricted to the Ridge-valid benchmark rows.

Each model writes:

- `metrics.csv`: per-station metrics plus original row count,
  evaluated row count, removed row count, and evaluation coverage.
- `predictions.csv`: station, original source index, timestamp, target,
  and prediction for row-level comparison.
- `summary.csv`: macro mean MAE/RMSE/R2, macro median R2, pooled
  MAE/RMSE/R2, and positive/negative R2 station counts.

Macro metrics average station-level metrics equally. Pooled metrics are
computed across all evaluated prediction rows. The simple mean of
per-station R2 values is not a global R2 and can be dominated by
low-variance stations.

Linear-model selection is handled separately by `13a_tune_ridge.py`.
That script uses train and validation splits only, writes
`validation_tuning.csv`, and selects one global linear configuration by
pooled validation RMSE before final test evaluation. The current frozen
linear baseline is unscaled `Ridge(alpha=1000.0)`.

Random Forest selection is handled separately by
`15a_tune_random_forest.py`. It uses the same train/validation split,
same `MODEL_FEATURE_COLUMNS`, same required-feature evaluation frame,
and pooled validation RMSE criterion. The current frozen Random Forest
baseline is `n_estimators=100`, `max_depth=10`,
`min_samples_leaf=10`, `max_features=1.0`, `random_state=42`, and
`n_jobs=1`. The tuning script is deliberate analysis work; the normal
pipeline runs only the frozen `15_random_forest.py` baseline.

XGBoost selection is handled separately by `16a_tune_xgboost.py`. It
uses train data for fitting and validation data for early stopping and
configuration selection; test data is not loaded during tuning. The
current frozen XGBoost baseline uses `learning_rate=0.1`,
`max_depth=3`, `min_child_weight=5`, `subsample=0.8`,
`colsample_bytree=0.8`, `reg_alpha=0.0`, `reg_lambda=1.0`,
`n_estimators=1000`, `early_stopping_rounds=50`, `tree_method="hist"`,
`random_state=42`, and `n_jobs=1`.

## Wind Feature Semantics

`wind_direction` follows the meteorological convention: it is the
direction FROM which wind blows, measured clockwise from north.
`wind_u` is the physical eastward component and `wind_v` is the physical
northward component:

```text
wind_u = -wind_speed * sin(wind_direction)
wind_v = -wind_speed * cos(wind_direction)
```

The current downloader does not override Open-Meteo's wind-speed unit,
so `wind_speed`, `wind_u`, and `wind_v` are in km/h.

## Planned Extensions

- **Graph construction**: Build a station graph where edges are weighted by wind direction/speed between station pairs. For directed A-to-B transport alignment, compare the A-to-B bearing with `transport_direction = (wind_direction + 180) % 360`, not raw meteorological wind direction.
- **GAT-GRU model**: A Graph Attention Network combined with a GRU to capture both spatial (wind-driven) and temporal dependencies.
- **Explainability**: Extract attention weights and feature attributions to explain individual forecasts.
- **Serving & dashboard**: Expose trained models through a FastAPI service, visualized in a React dashboard.

## Canonical Hourly AQ Migration

The modeling pipeline now uses OpenAQ `/hours` data, not raw
`/measurements`, for PM2.5 alignment. Raw `/measurements` remains an
archival/research layer, but canonical modeling rows use
`data/processed/air_quality_hourly/`.

Canonical PM2.5 timestamp semantics:

```text
timestamp t = OpenAQ one-hour PM2.5 interval ending at local clock time t
```

For example, a PM2.5 period `06:00 -> 07:00` is merged with the weather
row at `07:00`. This replaces the old downstream timestamp flooring,
which could collapse distinct OpenAQ intervals onto the same hour.

The merge remains weather-left: every weather hour is preserved, and
missing PM2.5 observations remain explicit `NaN` values.
