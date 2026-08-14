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
Machine Learning Models (13 — Linear Regression, ...)
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
| Feature Engineering | `08_feature_engineering.py` | Add lag features, rolling statistics, cyclical time encodings, and wind vector (u/v) components |
| Dataset Preparation | `09_prepare_dataset.py` | Build the final modeling table, including the one-hour-ahead `target_pm2_5` label |
| Splitting | `10_split_dataset.py`, `11_verify_split.py` | Chronological train / validation / test split per station, with integrity checks |
| Baseline & Models | `12_persistence_baseline.py`, `13_linear_regression.py` | Evaluate a naive persistence baseline and a linear regression model using shared MAE / RMSE / R² metrics |
| Analysis | `14_feature_correlation.py` | Compute feature statistics and correlations against the forecasting target |

## Design Principles

- **Reusable infrastructure**: `downloaders/base_downloader.py` and `clients/http_client.py` centralize retry, backoff, and file-skip logic so both downloaders share the same reliability guarantees.
- **Reusable modeling**: `models/base_model.py` standardizes dataset loading, evaluation, and result export so every forecasting model (persistence, linear regression, and future models) reports metrics consistently.
- **Idempotent downloads**: Both downloaders skip files that already exist, so the pipeline can be safely re-run without re-fetching existing data.
- **Central configuration**: All paths, constants, and the model feature list live in `scripts/config.py`, avoiding hardcoded values scattered across scripts.

## Planned Extensions

- **Graph construction**: Build a station graph where edges are weighted by wind direction/speed between station pairs.
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
