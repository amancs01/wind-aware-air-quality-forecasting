# Preprocessing Plan

This document describes the preprocessing pipeline that turns raw downloaded data into ML-ready datasets, and the timestamp-handling logic that makes merging possible.

## Pipeline

```
Raw Data
    ↓
Validation
    ↓
Profiling
    ↓
Merge (weather + air quality)
    ↓
Timestamp Alignment
    ↓
Trim Leading Missing Labels
    ↓
Feature Engineering
    - Time features
    - Lag features
    - Rolling features
    ↓
Train / Validation / Test Split
```

Corresponding scripts: `03_validate_data.py` → `04_profile_dataset.py` → `05_preprocess_data.py` → `06_analyze_merge_data.py` → `07_trim_data.py` / `07b_validate_timestamps.py` → `08_feature_engineering.py` → `09_prepare_dataset.py` → `10_split_dataset.py` / `11_verify_split.py`.

## Timestamp Alignment

Open-Meteo provides hourly observations on the exact hour. OpenAQ measurements are timestamped approximately every hour but with minute-level offsets, and the two sources use different timestamp formats — direct string matching fails.

During preprocessing (`05_preprocess_data.py`), timestamps from both sources are converted to proper `datetime` objects, normalized to a common format, and aligned before the merge step so that each row pairs a weather reading with the nearest corresponding air-quality reading.

## Trimming

Because most stations only have consistent PM2.5 coverage starting in late 2025 (see `docs/data_sources.md`), the merged dataset has a long leading stretch where weather data exists but PM2.5 does not. `07_trim_data.py` removes this leading stretch so training data isn't dominated by unlabeled rows, while later, sparser gaps are intentionally retained for future imputation work.

## Note on Scaling

Feature scaling/normalization is not yet implemented in the pipeline. It will be added once tree-based and sequence models are introduced (see `docs/model_plan.md`), since linear regression and the persistence baseline don't strictly require it.