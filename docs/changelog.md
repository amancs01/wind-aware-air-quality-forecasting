# Changelog

Notable changes to the project, grouped by milestone. Ongoing "how/why" narrative lives in `development_log.md`.

## Milestone: Fair Baseline Evaluation

### Changed
- Added a reusable model evaluation frame requiring
  `MODEL_FEATURE_COLUMNS + target_pm2_5`.
- Persistence and Ridge now score identical station/source-index/
  timestamp/target rows.
- Persistence still predicts current PM2.5; only its scored rows changed
  to match Ridge-valid benchmark rows.
- `BaseModel` now exports per-station coverage metrics, row-level
  predictions, and aggregate summary metrics.
- Summary reporting now distinguishes macro mean metrics, macro median
  R2, pooled metrics, and positive/negative R2 dataset counts.
- Ridge remains `Ridge(alpha=10.0)`; no tuning or feature-set change was
  introduced.

### Verified
- Persistence prediction rows: 26,236 across 51 stations.
- Ridge prediction rows: 26,236 across 51 stations.
- Row identity match between baselines: exact station/source-index/
  timestamp/target match with zero mismatches.
- Fair persistence summary: macro MAE 5.830, macro RMSE 8.815, macro
  mean R2 0.692, pooled MAE 6.005, pooled RMSE 12.083, pooled R2 0.820.
- Fair Ridge summary: macro MAE 9.446, macro RMSE 12.075, macro mean R2
  -127.128, macro median R2 0.570, pooled MAE 9.487, pooled RMSE
  14.202, pooled R2 0.751.

## Milestone: Machine Learning Pipeline

### Added
- Persistence baseline forecasting model for PM2.5 prediction.
- Reusable `BaseModel` class to remove duplicate code across ML models.
- Shared evaluation utilities for MAE, RMSE, and R².
- Per-model results directories under `results/`.
- Automatic metrics export (`metrics.csv`) for each model.
- Linear Regression model.

### Changed
- Persistence model refactored to inherit from `BaseModel`.
- Dataset loading, evaluation, metrics saving, and summary reporting moved into `BaseModel`.
- Dataset preparation updated to generate `target_pm2_5` (one-hour-ahead PM2.5) as the label.
- Empty-dataset handling improved: stations with insufficient PM2.5 observations are now skipped rather than erroring.

### Verified
- Train / validation / test dataset generation.
- Persistence baseline execution across all usable stations.
- Evaluation metrics generated successfully for each station.

## Milestone: Data Engineering Pipeline Complete

### Added
- Merge pipeline (weather + air quality).
- Dataset trimming (remove leading rows with no PM2.5 label).
- Feature engineering pipeline.
- Dataset preparation step.

### Feature Engineering
- Time features (hour, day, month, weekday).
- Lag features.
- Rolling statistics.
- Wind vector (u/v) features.
- Cyclical time encoding.

### Improvements
- Automatic removal of incomplete rows.
- ML-ready datasets generated per station.

## Milestone: Initial Data Collection

### Added
- Initial project structure.
- Weather downloader.
- Configuration system (`config.py`).
- Shared utilities (`utils.py`).
- `api.py` abstraction for OpenAQ requests.
- OpenAQ station discovery script.
- OpenAQ sensor discovery.
- First air quality downloader prototype.
- Automatic retry mechanism for failed API requests.
- Exponential backoff for transient network failures.
- Support for downloading current-year weather data.
- Dataset validation and profiling.
- Timestamp alignment between OpenAQ and Open-Meteo.

### Changed
- Weather downloader now uses `pathlib` instead of manual path strings.
- Weather downloader creates directories through reusable utilities.
- Air quality downloader switched from yearly requests to monthly partitioning.
- Improved filename sanitization for Windows compatibility.

### Fixed
- `.gitignore` encoding issue caused by UTF-16 LE.
- Raw datasets being accidentally tracked by Git.
- Windows path problems caused by invalid station name characters.
- Downloader repeatedly re-downloading existing weather files.

### Refactored
- Introduced `BaseDownloader` abstraction.
- Refactored the weather downloader and air quality downloader to use it.
- Centralized utilities into `utils.py`.
- Added a reusable HTTP client (`clients/http_client.py`).

## Milestone: Canonical Hourly AQ Downstream Migration

### Changed
- Profiling now uses canonical hourly OpenAQ `/hours` PM2.5 data instead
  of raw `/measurements` counts.
- `station_coverage.csv` now includes `dataset_name`, `station`, and
  `sensor_id` so duplicate station names do not collapse distinct PM2.5
  sensors.
- `DataMerger` now merges weather with canonical hourly AQ from
  `data/processed/air_quality_hourly/`.
- AQ timestamp flooring was removed from the merge path.
- `run_pipeline.py` now documents the canonical hourly preprocessing
  order and treats raw `/measurements` as an optional archival source.

### Validated
- 56 canonical AQ datasets were merged with weather-left cardinality.
- Every merged dataset preserved weather row count and had zero duplicate
  timestamps.
- Regenerated trimmed data had zero invalid hourly gaps.
- Temporal-feature validation reported 100% timestamp correctness for
  `lag_1`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `rolling_3`,
  `rolling_6`, and `rolling_24`.

### Baselines
- Regenerated Persistence baseline: MAE 5.774, RMSE 8.891, R2 0.699.
- Regenerated Ridge baseline: MAE 9.446, RMSE 12.075, R2 -127.128.
- These metrics are the new canonical-hourly baselines and are not
  directly comparable with older pre-migration metrics.
