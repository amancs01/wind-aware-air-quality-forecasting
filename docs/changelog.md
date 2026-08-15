# Changelog

Notable changes to the project, grouped by milestone. Ongoing "how/why" narrative lives in `development_log.md`.

## Milestone: First LSTM Baseline

### Added
- Added PyTorch-based station-specific LSTM baseline via
  `20_lstm_baseline.py` and `analysis/lstm_baseline.py`.
- Added `torch>=2.0.0` to `requirements.txt`.
- Wrote outputs under `results/lstm/`: validation station metrics,
  native validation summary, validation predictions, matched
  LSTM/Persistence/RF predictions, matched summary, station win counts,
  training history, skipped stations, and a Markdown report.

### Method
- Used only `data/processed/featured/` and the validated
  sequence-native design.
- Used 24 consecutive hourly input rows to predict PM2.5 exactly one
  hour after the final input timestamp.
- Used input columns: `pm2_5`, `hour_sin`, `hour_cos`, `month_sin`,
  `month_cos`, `temperature`, `humidity`, `pressure`, `dew_point`,
  `wind_u`, and `wind_v`.
- Did not use handcrafted lag or rolling columns.
- Trained station-specific LSTMs with input size 11, hidden size 64, one
  LSTM layer, a linear prediction head, Adam at learning rate 0.001, MSE
  loss, batch size 64, max 50 epochs, early stopping patience 5, and
  random seed 42.
- Fit input and target scalers on training sequences only and applied
  them unchanged to validation.
- Used train and validation only; the final test split was not
  evaluated.
- Compared LSTM with Persistence and frozen Random Forest on identical
  validation target timestamps.

### Findings
- Runtime was PyTorch 2.13.0+cpu on CPU; CUDA was unavailable.
- Trained 51 stations and skipped 5 stations.
- Native LSTM validation cohort had 22,657 sequences.
- Native LSTM metrics: macro MAE 10.895, macro RMSE 15.112, macro
  median R2 0.713, pooled MAE 9.619, pooled RMSE 15.036, pooled R2
  0.834.
- Matched comparison used 22,477 validation rows.
- On matched rows, Persistence had pooled RMSE 11.848, frozen Random
  Forest had pooled RMSE 12.233, and LSTM had pooled RMSE 15.021.
- LSTM beat Persistence on 10/51 stations and Random Forest on 7/51
  stations; it beat both on 4 stations.
- Best epoch median was 15; 9 stations had best epoch <= 5, 2 had best
  epoch >= 40, and 1 reached epoch 50.
- Conclusion: the first LSTM baseline is technically valid but does not
  yet add useful temporal signal beyond Persistence or RF.

## Milestone: LSTM Sequence Dataset Validation

### Added
- Added validation-only LSTM sequence dataset design via
  `19_lstm_sequence_dataset_validation.py` and
  `analysis/lstm_sequence_dataset_validation.py`.
- Wrote sequence validation outputs under
  `results/lstm_sequence_validation/`, including source assessment,
  station split counts, aggregate rejection counts, accepted-window proof
  checks, sparse-station flags, input-design comparison, and a Markdown
  report.

### Method
- Used `data/processed/featured/` for sequence construction because it
  preserves the hourly timeline.
- Did not construct sequence windows from `data/processed/prepared/`,
  because prepared rows drop missing current/target PM2.5 and are not
  guaranteed to be consecutive hours.
- Validated 24-hour input windows with a one-hour-ahead PM2.5 target.
- Enforced exact hourly input timestamps, target timestamp exactly one
  hour after the final input timestamp, no missing input/target values,
  chronological split membership, and no split-boundary crossing.
- Compared full `MODEL_FEATURE_COLUMNS` against a sequence-native input
  design before recommending the first LSTM baseline inputs.

### Findings
- Featured data had zero invalid hourly gaps across 56 stations.
- Prepared data had row gaps in 51 stations, with 5,051 invalid adjacent
  hourly gaps and a largest adjacent prepared-row gap of 11,636 hours.
- Recommended input columns are current PM2.5, cyclical time, weather,
  and physical wind components:
  `pm2_5`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`,
  `temperature`, `humidity`, `pressure`, `dew_point`, `wind_u`,
  `wind_v`.
- The recommended sequence-native design produced 146,497 accepted
  sequences: 101,168 train, 22,657 validation, and 22,672 test.
- Full `MODEL_FEATURE_COLUMNS` produced 119,307 accepted sequences,
  mainly because lag/rolling PM2.5 summaries reject more windows after
  missing PM2.5 periods.
- Accepted-window proof checks had zero invalid input lengths, zero
  invalid hourly gaps, zero invalid target gaps, and zero split
  membership violations.
- Seven stations were flagged as too sparse for the recommended sequence
  design.

## Milestone: Rolling-Origin Validation

### Added
- Added `18_rolling_origin_validation.py` and
  `analysis/rolling_origin_validation.py`.
- Wrote rolling-origin outputs under `results/rolling_origin/`:
  fold summaries, station metrics, model comparisons, RF station win
  buckets, fold-frame row counts, and distribution-shift diagnostics.

### Method
- Used only each prepared station dataset's first 85% development
  portion.
- Did not load or evaluate the existing final 15% test split.
- Used three expanding-window folds:
  fold 1 train 0-55%, validate 55-65%; fold 2 train 0-65%, validate
  65-75%; fold 3 train 0-75%, validate 75-85%.
- Evaluated frozen Persistence, `Ridge(alpha=1000)`, and the frozen
  Random Forest configuration without retuning or changing
  `MODEL_FEATURE_COLUMNS`.
- Used the same full-feature-valid rows within each fold for all three
  models.

### Findings
- Each fold evaluated 51 datasets; two tiny prepared datasets were
  skipped for insufficient full-feature-valid rows.
- Random Forest beat Persistence by pooled RMSE in folds 1 and 2, but
  lost in fold 3.
- RF station wins vs Persistence were 31/51, 34/51, and 27/51 across
  folds 1-3.
- RF win consistency across stations: 10 stations won 3/3 folds,
  24 won 2/3, 14 won 1/3, and 3 won 0/3.
- Ridge beat Persistence by pooled RMSE in folds 1 and 2, but lost
  clearly in fold 3.
- Distribution-shift diagnostics highlighted large validation target
  mean shifts, including Phora Durbar Kathman in folds 2-3 and
  Sundarighat's fold-dependent shifts.

## Milestone: Validation Feature Ablation

### Added
- Added validation-only feature ablation analysis via
  `17_feature_ablation.py` and `analysis/feature_ablation.py`.
- Wrote ablation outputs under `results/feature_analysis/ablation/`:
  summary, station metrics, incremental comparisons, row identity, and
  wind station comparison files.

### Method
- Used only train and validation splits; the test split was not loaded.
- Used the frozen Random Forest configuration:
  `n_estimators=100`, `max_depth=10`, `min_samples_leaf=10`,
  `max_features=1.0`, `random_state=42`, `n_jobs=1`.
- Kept `MODEL_FEATURE_COLUMNS` unchanged.
- Used one fixed full-feature-valid row mask for every ablation variant:
  51 datasets, 115,725 training rows, and 25,689 validation rows.

### Findings
- Persistence remained the strongest pooled validation RMSE benchmark at
  12.300.
- Current-PM-only Random Forest had pooled validation RMSE 13.453, so a
  nonlinear calibration of current PM2.5 alone did not beat persistence.
- Adding PM2.5 history to current PM2.5 gave a small pooled RMSE
  improvement of 0.046.
- Adding cyclical time features gave the largest incremental pooled
  RMSE improvement: 0.908.
- Adding non-wind weather after PM history and time gave almost no
  pooled RMSE improvement: 0.002, although pooled MAE improved by 0.184.
- Adding wind without other weather improved pooled RMSE by 0.073.
- Adding wind after PM history, time, and non-wind weather improved
  pooled RMSE by 0.048 and pooled MAE by 0.033.
- Conditional wind benefit was not station-wide: wind improved RMSE on
  24 stations and worsened it on 27; median station RMSE effect was
  -0.011.

## Milestone: XGBoost Baseline

### Added
- Added XGBoost dependency declaration in `requirements.txt`.
- Added validation-only XGBoost tuning via `16a_tune_xgboost.py` and
  `analysis/xgboost_validation.py`.
- Added production XGBoost evaluation via `16_xgboost.py` and
  `models/xgboost_model.py`.
- Added frozen XGBoost configuration constants in `config.py`.

### Changed
- `run_pipeline.py` now runs the frozen XGBoost baseline after Random
  Forest.
- XGBoost uses the unchanged `MODEL_FEATURE_COLUMNS`, station-specific
  training, validation-only early stopping, and the shared fair
  evaluation frame.

### Verified
- Environment: Python 3.12.0 and XGBoost 3.4.0.
- XGBoost validation tuning used train/validation only and selected by
  pooled validation RMSE.
- Selected XGBoost: `learning_rate=0.1`, `max_depth=3`,
  `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`,
  `reg_alpha=0.0`, `reg_lambda=1.0`, `n_estimators=1000`,
  `early_stopping_rounds=50`, `tree_method="hist"`, `random_state=42`,
  and `n_jobs=1`.
- Selected XGBoost validation: 25,689 rows, macro RMSE 11.938, macro
  median R2 0.787, pooled RMSE 12.700, pooled R2 0.881.
- Selected XGBoost best iteration across validation stations: min 19,
  median 67, mean 88.78, max 261; no station hit the 1000-tree upper
  bound.
- Final held-out test XGBoost: 26,236 rows, macro MAE 7.336, macro RMSE
  10.257, macro mean R2 -443.742, macro median R2 0.701, pooled MAE
  7.184, pooled RMSE 12.043, pooled R2 0.821.
- Test row identity matched persistence, Ridge, and Random Forest
  exactly on station, source index, timestamp, and target.
- XGBoost improved pooled test RMSE over persistence by 0.039 (0.33%),
  but was worse than Random Forest by pooled RMSE, pooled MAE, and pooled
  R2.

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

## Milestone: Wind Component Semantics Correction

### Changed
- Corrected `wind_u` and `wind_v` feature engineering to use
  meteorological wind-direction semantics: `wind_direction` is the
  direction FROM which wind blows, clockwise from north.
- `wind_u` now represents the physical eastward component and `wind_v`
  the physical northward component.
- Wind speed and derived components remain in km/h; raw weather data was
  not redownloaded or converted to m/s.
- Updated documentation to distinguish meteorological FROM direction
  from pollution transport TO direction:
  `transport_direction = (wind_direction + 180) % 360`.

### Validated
- Synthetic cardinal sanity cases match the expected physical component
  signs.
- Regenerated featured data reconstructs wind speed from
  `sqrt(wind_u^2 + wind_v^2)` within floating-point tolerance.
- Ridge, Random Forest, and XGBoost hyperparameters were not retuned.

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
