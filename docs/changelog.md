# Changelog

Notable changes to the project, grouped by milestone. Ongoing "how/why" narrative lives in `development_log.md`.

## Milestone: Random Forest Baseline

### Added
- Added validation-only Random Forest tuning via
  `15a_tune_random_forest.py` and
  `analysis/random_forest_validation.py`.
- Added production Random Forest evaluation via `15_random_forest.py`
  and `models/random_forest.py`.
- Added frozen Random Forest configuration constants in `config.py`.

### Changed
- `run_pipeline.py` now runs the frozen Random Forest baseline after the
  frozen Ridge baseline.
- Random Forest uses the unchanged `MODEL_FEATURE_COLUMNS`, station-
  specific training, and the shared fair evaluation frame.

### Verified
- No existing Random Forest implementation was present before this
  milestone.
- Validation tuning used train/validation only and selected by pooled
  validation RMSE.
- Unbounded `max_depth=None` candidates exhausted local memory, so the
  feasible grid used `n_estimators=100`, `n_jobs=1`, `max_depth` in
  `[10, 20]`, `min_samples_leaf` in `[1, 5, 10]`, and `max_features` in
  `[1.0, "sqrt"]`.
- Selected Random Forest: `max_depth=10`, `min_samples_leaf=10`,
  `max_features=1.0`, `n_estimators=100`, `random_state=42`,
  `n_jobs=1`.
- Validation selected RF: 25,689 rows, macro RMSE 11.901, macro median
  R2 0.800, pooled RMSE 12.450, pooled R2 0.886.
- Validation selected RF beat persistence by station RMSE on 27
  datasets; persistence won on 24.
- Final held-out test RF: 26,236 rows, macro MAE 6.735, macro RMSE
  9.695, macro mean R2 -78.855, macro median R2 0.704, pooled MAE
  6.556, pooled RMSE 11.652, pooled R2 0.833.
- Test row identity matched persistence and Ridge exactly on station,
  source index, timestamp, and target.
- Random Forest improved pooled test RMSE over persistence by 0.430
  (3.56%), while persistence still won station-level test RMSE on 40 of
  51 datasets.

## Milestone: Validation-Selected Linear Baseline

### Added
- Added `13a_tune_ridge.py` for train-to-validation linear model
  selection without loading the test split.
- Added reusable validation tuning support for persistence,
  `LinearRegression()`, unscaled Ridge, and train-fitted
  `StandardScaler() + Ridge`.
- Added `LINEAR_BASELINE_ALPHA = 1000.0` as the frozen production linear
  baseline configuration.

### Changed
- Production `LinearRegressionModel` now uses unscaled
  `Ridge(alpha=1000.0)`, selected by pooled validation RMSE.
- The original fixed `Ridge(alpha=10.0)` result is retained as
  historical untuned baseline context.

### Verified
- Validation candidate rows: 25,689 across 51 stations.
- Validation persistence: macro RMSE 11.536, macro median R2 0.769,
  pooled RMSE 12.300, pooled R2 0.889.
- Best linear validation candidate: unscaled `Ridge(alpha=1000.0)`,
  macro RMSE 12.430, macro median R2 0.775, pooled RMSE 13.225,
  pooled R2 0.871.
- Selected Ridge beat persistence by validation RMSE on 17 stations;
  persistence won on 34 stations.
- Frozen final test selected Ridge: 26,236 rows, macro MAE 7.437,
  macro RMSE 10.076, macro mean R2 -29.748, macro median R2 0.702,
  pooled MAE 7.629, pooled RMSE 12.591, pooled R2 0.805.
- Test persistence remained stronger overall: pooled RMSE 12.083 and
  pooled R2 0.820.

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
