# Wind-Aware Air Quality Forecasting --- Project Handover

**Purpose:** Canonical handover for continuing the project in a fresh
ChatGPT Project/chat without relying on the very long historical
conversation.

**Repository snapshots reviewed** -
`wind-aware-air-quality-forecasting-main (2).zip` -
`wind-aware-air-quality-forecasting-Nirika-work (1).zip`

**Important:** This document distinguishes the code that actually exists
in the uploaded branch snapshots from ideas that were only discussed.

------------------------------------------------------------------------

## 1. Project Overview

This is a final-year major project for **one-hour-ahead PM2.5
forecasting in the Kathmandu Valley**.

The long-term research direction is a **wind-aware spatio-temporal
forecasting system**, eventually using a graph-based architecture such
as **GAT-GRU**, where monitoring stations are graph nodes and wind
influences inter-station relationships.

The current `main` branch is focused on:

1.  data collection,
2.  validation and profiling,
3.  weather--air-quality merging,
4.  feature engineering,
5.  one-hour-ahead target construction,
6.  chronological train/validation/test splitting,
7.  classical baseline models,
8.  feature-correlation analysis.

The `Nirika-work` branch extends this with graph-construction work.

------------------------------------------------------------------------

## 2. Research Objective

Predict:

`PM2.5(t + 1 hour)`

using current/historical PM2.5, temporal variables, weather variables,
and eventually spatial/wind relationships between stations.

The persistence baseline is:

`PM2.5(t + 1) = PM2.5(t)`

Every more sophisticated model should be compared against this baseline
using the same metrics.

------------------------------------------------------------------------

## 3. Data Sources

### OpenAQ

Used for PM2.5 air-quality observations.

Important findings already recorded in the repository:

-   OpenAQ hierarchy is Location → Sensor → Measurements.
-   Measurements are retrieved per sensor.
-   Monthly downloads are more reliable than large yearly requests.
-   Pagination is required.
-   Coverage differs substantially by station.
-   Many GD Labs stations have relatively recent PM2.5 history.
-   Embassy Kathmandu has substantially longer coverage.

### Open-Meteo

Used for hourly meteorological variables.

Configured weather variables currently include:

-   temperature
-   relative humidity
-   dew point
-   surface pressure
-   wind speed
-   wind direction
-   precipitation

------------------------------------------------------------------------

## 4. Branch Structure

### `main`

This is the core data-engineering and baseline-modeling branch.

Important numbered stages currently present:

``` text
00_discover_stations.py
01_download_weather.py
02_download_air_quality.py
03_validate_data.py
04_profile_dataset.py
05_preprocess_data.py
06_analyze_merge_data.py
07_trim_data.py
07b_validate_timestamps.py
08_feature_engineering.py
09_prepare_dataset.py
10_split_dataset.py
11_verify_split.py
12_persistence_baseline.py
13_linear_regression.py
14_feature_correlation.py
```

`run_pipeline.py` executes them in this order.

### `Nirika-work`

Contains the core pipeline plus graph-specific work under:

``` text
scripts/graph/
├── 01_station_mapping.py
├── 02_distance_matrix.py
├── 03_bearing_matrix.py
├── 04_static_graph.py
├── 05_dynamic_edge_weights.py
├── 06_graph_snapshots.py
└── 07_sliding_windows.py
```

It also contains generated station mapping metadata.

Do **not** casually merge the graph branch into `main`. Review the graph
assumptions, input directories, station filtering, timestamp alignment,
and config changes first.

------------------------------------------------------------------------

## 5. Current `main` Pipeline

The repository currently implements:

``` text
Station Discovery
        ↓
Weather + Air Quality Download
        ↓
Raw Validation
        ↓
Profiling
        ↓
Timestamp Normalization
        ↓
Weather + Air Quality Merge
        ↓
Trim Leading Missing PM2.5
        ↓
Timestamp Validation
        ↓
Feature Engineering
        ↓
Dataset Preparation / Target Creation
        ↓
Train / Validation / Test Split
        ↓
Split Verification
        ↓
Persistence Baseline
        ↓
Ridge Regression Baseline
        ↓
Feature Correlation Analysis
```

This is the **actual current code order**. Do not silently reorder it.

------------------------------------------------------------------------

## 6. Merge and Trim Behavior

### Merge

`DataMerger` loads weather and air-quality data per station and
performs:

``` python
pd.merge(
    weather_df,
    air_df,
    on="timestamp",
    how="left",
)
```

Therefore the weather dataframe is the left side of the merge.

Air-quality timestamps come from the canonical OpenAQ `/hours` layer.
The model timestamp is `datetime_to_local`, interpreted as the local
one-hour interval end. The merge path no longer floors or rounds AQ
timestamps. Duplicate human-readable station names are preserved with
sensor-qualified dataset names where needed.

### Trim

`DataTrimmer` finds the first non-null `pm2_5` value and removes only
the leading period before that point.

It does **not** remove every later missing PM2.5 row.

This is important for understanding lag/rolling behavior.

------------------------------------------------------------------------

## 7. Timestamp-Gap Discovery

A dedicated timestamp validator was added at stage `07b`.

It calculates:

-   valid one-hour gaps,
-   invalid gaps,
-   largest gap in hours,
-   valid-gap percentage,
-   median gap,
-   per-station invalid rows,
-   gap distributions.

The investigation found that many stations are mostly hourly but can
contain large discontinuities. Examples observed during development
included gaps of days, weeks, or longer.

Therefore:

> Never assume `.shift(n)` means exactly `n` hours merely because the
> dataframe is sorted.

This is one of the most important findings of the project.

------------------------------------------------------------------------

## 8. Resolved Lag/Rolling Timestamp Semantics

The earlier concern that row-based lag/rolling operations might cross
non-hourly timestamp gaps has been resolved by the canonical hourly AQ
downstream migration.

### Current code

Feature engineering runs on `TRIMMED_DIR` before dataset preparation.

It uses row-based operations such as:

``` python
df["lag_6"] = df["pm2_5"].shift(6)
df["lag_24"] = df["pm2_5"].shift(24)
df["rolling_mean_6"] = df["pm2_5"].rolling(6).mean()
```

### Validated behavior

After switching the merge to canonical hourly AQ and regenerating
trimmed data, timestamp validation found zero invalid hourly gaps.
Temporal-feature validation then reported 100% timestamp correctness for
`lag_1`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `rolling_3`,
`rolling_6`, and `rolling_24`.

Therefore row-based lag and rolling operations are currently
timestamp-correct on regenerated trimmed data. Remaining missing
lag/rolling values come from missing PM2.5 measurement history within an
otherwise hourly timeline, not from hidden timestamp jumps.

Timestamp-aware reindexing or join-based lag construction is not the
current highest-priority task.

------------------------------------------------------------------------

## 9. Current Feature Engineering

`FeatureEngineer` currently creates:

### Calendar features

-   `hour`
-   `day`
-   `month`
-   `weekday`

### PM2.5 lag features

-   `lag_1`
-   `lag_3`
-   `lag_6`
-   `lag_12`
-   `lag_24`

### Rolling means

-   `rolling_mean_3`
-   `rolling_mean_6`
-   `rolling_mean_24`

### Rolling standard deviations

-   `rolling_std_3`
-   `rolling_std_6`
-   `rolling_std_24`

### Wind-vector features

-   `wind_u`
-   `wind_v`

### Cyclical time encodings

-   `hour_sin`
-   `hour_cos`
-   `month_sin`
-   `month_cos`

Precipitation has also been added to the downloaded weather variables,
although it is not currently in `MODEL_FEATURE_COLUMNS`.

------------------------------------------------------------------------

## 10. Target Generation

`DatasetPreparer` creates the one-hour-ahead target.

Current logic:

1.  parse timestamp,
2.  compute `next_timestamp`,
3.  compute the gap to the next row,
4.  create `target_pm2_5 = pm2_5.shift(-1)`,
5.  invalidate the target unless the next timestamp is exactly one hour
    later,
6.  remove rows where current `pm2_5` or `target_pm2_5` is missing.

Conceptually:

``` python
next_timestamp = df["timestamp"].shift(-1)
time_gap = next_timestamp - df["timestamp"]

df["target_pm2_5"] = df["pm2_5"].shift(-1)

df.loc[
    time_gap != pd.Timedelta(hours=1),
    "target_pm2_5"
] = pd.NA
```

This was introduced after discovering large timestamp gaps.

### Important clarification

A large gap does **not** generate thousands of synthetic NaN rows.

Only the row immediately before an invalid next-timestamp transition
gets an invalid target, because the absent hours do not exist as rows.

------------------------------------------------------------------------

## 11. Dataset Eligibility and Splitting

`MIN_TRAINING_ROWS = 100` is currently configured.

`DatasetSplitter` skips:

-   empty datasets,
-   datasets with fewer than 100 prepared rows.

Usable station data is split chronologically:

-   70% train
-   15% validation
-   15% test

There is no random shuffling.

The minimum-row rule belongs at the modeling/splitting stage rather than
deleting otherwise valid station data during early preprocessing.

------------------------------------------------------------------------

## 12. Missing-Value Handling and Evaluation Frame

An important model-safety rule exists in `BaseModel`:

``` python
evaluation_df = df.dropna(subset=MODEL_FEATURE_COLUMNS + ["target_pm2_5"])
```

This is now exposed as `prepare_evaluation_frame()` and is used by both
persistence and Ridge during evaluation. It preserves the original
source index and timestamp in exported predictions so row identity can
be checked across models.

Current model features are:

``` text
pm2_5
lag_6
lag_24
rolling_mean_6
rolling_std_6
hour_sin
hour_cos
month_sin
month_cos
wind_u
wind_v
temperature
humidity
pressure
dew_point
```

Treat this row mask as the current fair benchmark definition. It is not
feature tuning and it does not change `MODEL_FEATURE_COLUMNS`.

Current generated split coverage under this mask:

``` text
train rows before evaluation frame: 141,120
train rows valid for Ridge features: 115,725
validation rows before evaluation frame: 30,247
validation rows valid for Ridge features: 25,689
test rows before evaluation frame: 30,270
test rows evaluated by both baselines: 26,236
test rows removed by evaluation frame: 4,034
test evaluation coverage: 86.67%
```

------------------------------------------------------------------------

## 13. Baseline Models

### Persistence

Prediction:

``` text
prediction = current pm2_5
```

The persistence baseline uses the shared `BaseModel` evaluation
utilities and now scores only the same Ridge-valid benchmark rows. Its
prediction equation is unchanged.

### Linear model

The class is named `LinearRegressionModel`, but the actual estimator in
the current snapshot is:

``` python
Ridge(alpha=LINEAR_BASELINE_ALPHA)
```

`LINEAR_BASELINE_ALPHA = 1000.0`, selected using validation pooled RMSE
after comparing ordinary least squares, unscaled Ridge, and standardized
Ridge candidates. Future documentation should call this the
**validation-selected Ridge regression baseline** unless the estimator is
changed back to ordinary least-squares linear regression.

### Metrics

All models use:

-   MAE
-   RMSE
-   R2

Current result exports:

``` text
metrics.csv: per-station metrics and evaluation coverage
predictions.csv: station, source_index, timestamp, target, prediction
summary.csv: macro metrics, pooled metrics, and R2 dataset counts
```

Macro metrics average station-level scores equally. Pooled metrics are
computed across all evaluated rows. Macro mean R2 is not a global R2 and
can be dominated by low-variance stations.

------------------------------------------------------------------------

## 14. Feature Correlation Analysis

Stage `14_feature_correlation.py` / `analysis/feature_correlation.py`
was added after the baseline work.

Observed target correlations during development included strong positive
relationships with:

-   current `pm2_5`,
-   rolling PM2.5 means,
-   lagged PM2.5.

Weather variables generally showed weaker direct Pearson correlations.

Important interpretation:

-   correlation measures linear association,
-   correlation does not imply causation,
-   low individual Pearson correlation does not mean a feature is
    useless in a nonlinear or interaction-based model,
-   inter-feature correlation is useful for identifying redundancy and
    multicollinearity, particularly for linear models.

------------------------------------------------------------------------

## 15. Important Bugs and Discoveries So Far

### A. Zero-row / tiny stations

Some stations had no usable OpenAQ data or too little data for
meaningful train/validation/test splitting.

Resolution: - `MIN_TRAINING_ROWS = 100` - skip undersized stations
during splitting.

### B. Split verifier crash

The verifier previously assumed all train/validation/test files
contained rows and could crash on `.iloc[0]`.

This was exposed by small/stale station outputs.

When regenerating splits after changing eligibility rules, remove stale
old split files or ensure output directories are rebuilt cleanly.

### C. Persistence NaN failure

Persistence failed because current `pm2_5` could still be NaN while
`target_pm2_5` was valid.

Dataset preparation was changed to require both:

``` text
pm2_5
target_pm2_5
```

### D. Ridge NaN failure

Ridge failed because lag/rolling columns contained NaNs.

A model-level required-feature `dropna` was added.

### E. Timestamp discontinuities

This is the most important data-quality discovery.

A dataset can have no ordinary NaN in its raw OpenAQ rows while still
having long periods with no rows/observations.

Do not equate "no NaN values" with "continuous hourly time series."

------------------------------------------------------------------------

## 16. Git / Commit State

The last clearly identified committed milestone before the latest
debugging work was **inter-feature correlation analysis**.

After that, work included:

-   timestamp-gap validation,
-   continuity-aware target generation,
-   minimum training-row filtering,
-   current-PM2.5 target-row cleanup,
-   model-level NaN filtering,
-   debugging persistence and Ridge execution.

The exact Git commit hashes are not available from the ZIP snapshots.

Recommended commit separation before further architectural work:

### Commit A

`Add timestamp validation and improve target generation`

Include timestamp validation and one-hour target validity logic.

### Commit B

`Improve model training by handling missing feature values`

Include model-required-column NaN handling and related baseline
stability changes.

Do not bundle a future temporal-feature refactor into either commit.

------------------------------------------------------------------------

## 17. `Nirika-work` Graph Branch

The branch contains a graph pipeline with the following conceptual
stages.

### 1. Station mapping

Creates stable graph node IDs from station metadata.

Outputs include:

``` text
station_mapping.csv
station_mapping.json
```

The uploaded mapping contains 54 node IDs (0--53), including stations
that may not all satisfy the main branch's ML eligibility threshold.

Therefore graph-node eligibility must eventually be reconciled with the
stations actually usable for synchronized forecasting.

### 2. Distance matrix

Computes pairwise geographic distances using the Haversine formula.

Produces matrix and edge-list representations.

### 3. Bearing matrix

Computes directional bearing/azimuth between station pairs.

This is relevant to wind-aware directional connectivity.

### 4. Static graph

Builds geography-based graph structure.

### 5. Dynamic edge weights

Introduces time-varying wind-aware relationships.

### 6. Graph snapshots

Builds graph states across time.

### 7. Sliding windows

Prepares sequential graph samples for spatio-temporal modeling.

### Important integration warning

Before merging this work:

-   reconcile station sets,
-   verify timestamps are synchronized across nodes,
-   verify graph snapshots do not use unavailable/future information,
-   verify wind direction convention,
-   verify source→target bearing convention,
-   verify dynamic edge-weight equations,
-   verify how missing station observations are represented,
-   verify graph windows respect chronological train/validation/test
    boundaries.

------------------------------------------------------------------------

## 18. Relationship Between the Two Branches

The core preprocessing/model pipeline is largely shared between the
uploaded snapshots.

The principal additional work in `Nirika-work` is graph preparation.

Recommended strategy:

1.  stabilize temporal preprocessing on `main`,
2.  define the final eligible station set,
3.  define a trustworthy synchronized time representation,
4.  review Nirika's graph construction against those definitions,
5.  merge graph components incrementally,
6.  add tests/diagnostics before training GNN models.

------------------------------------------------------------------------

## 19. What NOT to Do

Future ChatGPT sessions should not:

1.  reorder Feature Engineering and Dataset Preparation without
    inspecting timestamp semantics;
2.  assume `.shift(6)` means six hours merely because data is sorted;
3.  impute temporal lag values just to make Ridge run without first
    understanding why they are missing;
4.  randomly shuffle time-series train/test data;
5.  remove the persistence baseline;
6.  treat an old baseline metric as final after preprocessing changes;
7.  merge `Nirika-work` wholesale without reviewing graph assumptions;
8.  discard small stations during raw cleaning merely because they are
    not currently useful for ML;
9.  use future PM2.5 information when constructing features;
10. silently change the forecasting horizon from one hour.

------------------------------------------------------------------------

## 20. Exact Current Stopping Point

The project has completed the fair reusable baseline evaluation task.

The next task is **not** Random Forest, XGBoost, GRU, or GNN training.

The current state is:

-   canonical hourly AQ migration is implemented and validated;
-   lag/rolling timestamp correctness is validated at 100%;
-   persistence and Ridge are now evaluated on identical test rows;
-   row-level prediction exports confirm exact station/source-index/
    timestamp/target matches between baselines;
-   Ridge has now been selected using validation data only;
-   production Ridge uses `LINEAR_BASELINE_ALPHA = 1000.0`;
-   the old `Ridge(alpha=10.0)` result remains historical context.

The fair result is that persistence still outperforms the
validation-selected Ridge baseline overall. This is a legitimate
research finding, not an evaluation-row artifact.

------------------------------------------------------------------------

## 21. Recommended Next Development Path

### Step 1 --- Preserve the fair benchmark contract

Any next model must use the same chronological splits and should report
the same coverage, macro, pooled, and row-level prediction outputs.

### Step 2 --- Interpret the Ridge baseline before adding model classes

Use the existing fair results to explain where Ridge fails:

-   one-hour PM2.5 autocorrelation is high, making persistence strong;
-   Ridge is a global linear model trained separately per station;
-   station-level distribution shifts can hurt Ridge;
-   macro mean R2 is sensitive to low-variance stations.

### Step 3 --- Preserve the frozen validation-selected linear result

The selected linear configuration is unscaled `Ridge(alpha=1000.0)`.
Do not change it based on test performance. Any future linear-model
experiment should be a new documented milestone.

### Step 4 --- Keep feature changes explicit

Do not silently add or remove model features. Any feature-set change
should be a separate research decision with regenerated metrics.

### Step 5 --- Only then consider the next baseline family

After the validation-selected Ridge result is documented, the next
reasonable research step is to introduce nonlinear classical baselines.
Do not start Random Forest, XGBoost, GRU, GNN, or graph integration
inside the linear-baseline selection milestone.

------------------------------------------------------------------------

## 22. Instructions for Future ChatGPT Sessions

When continuing this project:

1.  **Inspect the current repository/files before proposing structural
    changes.**
2.  Explain code changes incrementally and identify the exact file and
    method to edit.
3.  Preserve the one-hour-ahead forecasting target unless explicitly
    redesigning the research question.
4.  Treat timestamps as first-class data; never infer elapsed time
    solely from row distance.
5.  Keep preprocessing, model eligibility, and model training
    responsibilities conceptually separate.
6.  Prefer fixing data semantics over suppressing errors with blanket
    `dropna()` or imputation.
7.  Before changing a pipeline stage, state what upstream directory it
    reads and what downstream directory it writes.
8.  After preprocessing changes, regenerate all dependent outputs rather
    than evaluating models on stale CSVs.
9.  Keep Git commits logically separated by research/development
    milestone.
10. Do not assume `Nirika-work` is ready to merge merely because its
    scripts execute.
11. Distinguish **implemented code**, **experimental observations**, and
    **proposed future work**.
12. When a result changes after preprocessing, do not compare it
    directly with an old result without noting the data/pipeline change.

------------------------------------------------------------------------

## 23. Current Key Configuration

From the uploaded `main` snapshot:

``` text
Forecast horizon: 1 hour
Minimum prepared rows for splitting: 100
Train split: 70%
Validation split: 15%
Test split: 15%
Linear baseline estimator: Ridge(alpha=1000.0), no scaler
```

Current model feature list:

``` text
pm2_5
lag_6
lag_24
rolling_mean_6
rolling_std_6
hour_sin
hour_cos
month_sin
month_cos
wind_u
wind_v
temperature
humidity
pressure
dew_point
```

This list is the current fair-benchmark feature set. Do not change it
silently when comparing against the recorded persistence and Ridge
baselines.

------------------------------------------------------------------------

## 24. Definition of a Safe Continuation

A future chat can safely continue when it can answer these questions
from the current repository:

1.  Are persistence and Ridge using the same evaluation frame?
2.  Do `predictions.csv` exports match exactly on station,
    source_index, timestamp, and target?
3.  Which metric is being discussed: macro mean, macro median, or
    pooled?
4.  How many rows were removed by the required-feature evaluation
    frame?
5.  Is Ridge still the frozen validation-selected
    `Ridge(alpha=1000.0)` configuration?
6.  Has the feature set changed from `MODEL_FEATURE_COLUMNS`?
7.  Are generated outputs freshly regenerated after any code or
    preprocessing change?
8.  Are low-variance station R2 values being interpreted carefully?

Until these are answered, do not interpret advanced-model performance as
trustworthy.

------------------------------------------------------------------------

**Handover status:** Ready for continuation in the ChatGPT Project.

**Immediate next conversation title suggestion:**\
`01 - Interpret Fair Persistence vs Ridge Baselines`

------------------------------------------------------------------------

## 25. Implemented Hourly OpenAQ AQ Layer

This milestone has now been implemented and validated on `main`.

Implemented:

- Added OpenAQ `/hours` download support in `scripts/api.py` through a separate paginated helper.
- Added `scripts/02b_download_hourly_air_quality.py` to download raw hourly PM2.5 interval records.
- Added `AIR_QUALITY_HOURLY_RAW_DIR = data/raw/air_quality_hourly/` for untouched OpenAQ `/hours` records.
- Added `AIR_QUALITY_HOURLY_DIR = data/processed/air_quality_hourly/` for canonical hourly PM2.5 records.
- Preserved hourly interval metadata including local/UTC interval start and end, period label/interval, and coverage fields where returned by OpenAQ.
- Added `scripts/preprocessing/hourly_air_quality_preparer.py` and `scripts/02c_prepare_hourly_air_quality.py`.
- Canonical AQ timestamp is the local interval end: `timestamp = datetime_to_local` represented as naive local Kathmandu clock time for later weather compatibility.
- Original timezone-aware OpenAQ local/UTC interval columns are preserved in the canonical files for auditability.
- Canonical preparation keeps only records where `datetime_to_local.minute == 0`.
- Duplicate canonical timestamps are explicitly checked and fail with a diagnostic file rather than being silently deduplicated.
- Added `scripts/validation/hourly_air_quality_validator.py` and `scripts/02d_validate_hourly_air_quality.py`.
- Validation reports are written to `reports/validation/hourly_air_quality_validation.csv` and `reports/validation/hourly_air_quality_validation_summary.csv`.
- Duplicate station-name handling was added for the new hourly layer: duplicated station names such as `Kathmandu University` are stored with sensor-qualified dataset names so distinct PM2.5 sensors do not collide.

Validated results from the full generated hourly layer:

``` text
metadata PM2.5 sensors: 56
raw hourly station/sensor directories: 56
raw hourly files: 114
raw hourly rows: 323,665
canonical hourly files: 56
canonical hourly rows: 251,964
zero-row canonical stations: 0
duplicate timestamp rows: 0
non-clock-hour rows: 0
invalid one-hour intervals: 0
unsorted stations: 0
missing PM2.5 rows in canonical hourly files: 23,837
```

Known-problem stations were re-tested:

- Embassy Kathmandu raw `/hours` includes both `:00` and `:45` interval alignments, but the canonical file has zero duplicate timestamps.
- Phora Durbar Kathman raw `/hours` includes both `:00` and `:45` interval alignments, but the canonical file has zero duplicate timestamps.

Superseded by Section 26:

- `DataMerger` has since been switched to `AIR_QUALITY_HOURLY_DIR`.
- The old downstream AQ timestamp `.floor("h")` merge behavior has
  since been removed.
- `scripts/04_profile_dataset.py` has since been migrated to canonical
  hourly AQ coverage.
- `scripts/run_pipeline.py` now documents the canonical hourly modeling
  pipeline.
- Trimmed, featured, prepared, split, and baseline outputs have since
  been regenerated from the canonical hourly AQ layer.

Separate follow-up issue:

- The existing raw `/measurements` helper `fetch_all_measurements()` still uses `limit=1000` without pagination. This was intentionally not fixed in the hourly-layer milestone because the old raw `/measurements` archive remains separate from the new `/hours` implementation.

That downstream migration is no longer the expected next step.

**Handover status:** Hourly OpenAQ AQ layer implemented and validated;
downstream merger migration has since been completed in Section 26.

------------------------------------------------------------------------

## 26. Implemented Downstream Canonical Hourly Migration

This milestone has now been implemented, regenerated, validated, and
benchmarked on `main`.

Implemented:

- `PROJECT_HANDOVER.md` now exists in the repository root and is the
  canonical handover going forward.
- `scripts/04_profile_dataset.py` now profiles canonical hourly AQ from
  `AIR_QUALITY_HOURLY_DIR` instead of raw OpenAQ `/measurements`.
- `station_coverage.csv` now carries `dataset_name`, `station`, and
  `sensor_id` so duplicated human-readable station names do not collapse
  distinct PM2.5 sensors.
- Coverage now means valid canonical PM2.5 hours that match available
  weather hours divided by unique weather hours.
- `scripts/preprocessing/merger.py` now reads weather from `WEATHER_DIR`
  and canonical AQ from `AIR_QUALITY_HOURLY_DIR`.
- The old AQ timestamp `.floor("h")` merge behavior has been removed.
- The merge remains weather-left, preserving weather hours with missing
  PM2.5 as explicit `NaN`.
- Merged output filenames use canonical `dataset_name`, preserving
  sensor-qualified names for duplicated station names.
- `scripts/run_pipeline.py` now documents the canonical hourly modeling
  pipeline and treats raw `/measurements` as an optional archival source.

Duplicate station-name finding:

``` text
duplicated station name: Kathmandu University
PM2.5 sensors: 15286458, 15286980, 15286975
weather folder reused: data/raw/weather/Kathmandu University/
canonical/merged outputs: sensor-qualified dataset names
```

Regenerated profiling result:

``` text
canonical AQ datasets: 56
weather station folders: 54
coverage_percent definition: valid PM2.5 hours on weather timestamps / unique weather hours
zero exact-coverage datasets: 0
datasets rounding to 0.00% coverage: 3 tiny datasets with one valid hour each
```

Regenerated merge result:

``` text
datasets merged: 56
weather rows: 2,712,192
merged rows: 2,712,192
duplicate merged timestamps: 0
valid merged PM2.5 rows: 208,982
missing merged PM2.5 rows: 2,503,210
weather-left cardinality preserved for every dataset
```

Regenerated timestamp validation:

``` text
datasets checked: 56
trimmed rows: 445,562
valid hourly transitions: 445,506
invalid transitions: 0
largest gap: 1 hour
duplicate timestamps: 0
minimum valid-gap percent: 100.0
```

Regenerated temporal-feature validation:

``` text
lag_1: 100.00%
lag_3: 100.00%
lag_6: 100.00%
lag_12: 100.00%
lag_24: 100.00%
rolling_3: 100.00%
rolling_6: 100.00%
rolling_24: 100.00%
```

Conclusion:

The original row-distance lag/rolling timestamp problem has disappeared
after switching to the canonical hourly AQ merge. Timestamp-aware
reindexing or join-based lag construction is not currently required for
timestamp correctness. PM2.5 missingness remains a separate measurement
availability issue and was not imputed.

Regenerated dataset/split result:

``` text
prepared datasets: 53
prepared rows: 201,658
eligible split datasets: 51
train rows: 141,120
validation rows: 30,247
test rows: 30,270
datasets with no prepared rows: 3
prepared datasets skipped for <100 rows: 2
```

Regenerated baselines after canonical hourly migration:

``` text
Persistence before fair Ridge-valid row matching: 51 datasets, MAE 5.774, RMSE 8.891, R2 0.699
Ridge(alpha=10.0): 51 datasets, macro MAE 9.446, macro RMSE 12.075, macro mean R2 -127.128
Ridge rows after required-feature dropna: 115,725 train, 26,236 test
```

Section 27 supersedes the persistence number above for model comparison,
because persistence now uses the same Ridge-valid test rows. Do not
directly compare canonical-hourly metrics with old metrics from the
pre-migration `datetimeFrom.floor("h")` pipeline.

Still separate follow-up:

- The archival raw `/measurements` downloader still lacks pagination.
  This is technical debt for reproducibility of the raw archive, not a
  blocker for the current canonical-hourly modeling pipeline.

Expected next research step is updated in Section 27.

**Handover status:** Downstream preprocessing has been migrated to
canonical hourly AQ and validated; fair baseline evaluation is recorded
in Section 27.

------------------------------------------------------------------------

## 27. Implemented Fair Baseline Evaluation Framework

This milestone has now been implemented, verified, documented, and
benchmarked on `main`.

Implemented:

- `BaseModel.prepare_evaluation_frame()` defines the fair benchmark row
  mask: `MODEL_FEATURE_COLUMNS + target_pm2_5`.
- `BaseModel` now preserves row identity through `source_index` and
  timestamp in `predictions.csv`.
- `BaseModel` now writes per-station `metrics.csv`, row-level
  `predictions.csv`, and aggregate `summary.csv`.
- Persistence now evaluates on the same Ridge-valid rows while keeping
  the equation `prediction(t + 1) = pm2_5(t)`.
- Ridge remains `Ridge(alpha=10.0)` with no tuning and no feature-set
  change.

Matched-row verification:

``` text
stations compared: 51
persistence prediction rows: 26,236
Ridge prediction rows: 26,236
station/source_index/timestamp/target match: true
timestamp mismatches: 0
source-index mismatches: 0
target max absolute difference: 0.0
```

Fair persistence summary:

``` text
datasets: 51
original test rows: 30,270
evaluated rows: 26,236
removed rows: 4,034
evaluation coverage: 86.67%
macro MAE: 5.830
macro RMSE: 8.815
macro mean R2: 0.692
macro median R2: 0.763
pooled MAE: 6.005
pooled RMSE: 12.083
pooled R2: 0.820
negative R2 datasets: 2
```

Fair Ridge summary:

``` text
datasets: 51
original test rows: 30,270
evaluated rows: 26,236
removed rows: 4,034
evaluation coverage: 86.67%
macro MAE: 9.446
macro RMSE: 12.075
macro mean R2: -127.128
macro median R2: 0.570
pooled MAE: 9.487
pooled RMSE: 14.202
pooled R2: 0.751
negative R2 datasets: 9
```

Interpretation:

Persistence remains stronger than the current Ridge baseline under a
fair row-matched comparison. The one-hour PM2.5 autocorrelation is high,
so current PM2.5 is a difficult benchmark to beat.

The extreme Ridge macro mean R2 is dominated by Sundarighat (SC-23) -
GD Labs, where test target variance is extremely small and the test
distribution is far below the training distribution. Removing that one
station changes Ridge macro R2 from -127.128 to about 0.341, so macro
mean R2 must be interpreted carefully.

Expected next research step:

``` text
use the validation split to investigate whether the linear baseline can
be improved through explicit regularization/linear-variant choices,
without changing the feature set or tuning on test data.
```

Do not start Random Forest, XGBoost, GRU, GNN, or graph integration
until this fair baseline interpretation is reviewed.

**Handover status:** Fair persistence and Ridge evaluation is now
implemented and verified; validation-based linear baseline review is the
next recommended research step.

------------------------------------------------------------------------

## 28. Implemented Validation-Selected Linear Baseline

This milestone has now been implemented, frozen, tested once, documented,
and pushed on `main`.

Validation-only model selection:

- Split used for selection: train -> validation.
- Test split was not used while choosing alpha, scaling, or model
  family.
- Primary selection metric: pooled validation RMSE.
- Feature set: unchanged `MODEL_FEATURE_COLUMNS`.
- Missing lag/rolling values: not imputed.
- Global configuration selected for all station-specific models, not a
  separate alpha per station.

Validation candidates:

``` text
Persistence: rows 25,689, macro RMSE 11.536, median R2 0.769, pooled RMSE 12.300, pooled R2 0.889
LinearRegression: rows 25,689, macro RMSE 14.359, median R2 0.746, pooled RMSE 14.885, pooled R2 0.837
Ridge none alpha=0.001: pooled RMSE 14.885, pooled R2 0.837
Ridge none alpha=0.01: pooled RMSE 14.884, pooled R2 0.837
Ridge none alpha=0.1: pooled RMSE 14.880, pooled R2 0.837
Ridge none alpha=1.0: pooled RMSE 14.843, pooled R2 0.838
Ridge none alpha=10.0: pooled RMSE 14.591, pooled R2 0.843
Ridge none alpha=100.0: pooled RMSE 13.939, pooled R2 0.857
Ridge none alpha=1000.0: pooled RMSE 13.225, pooled R2 0.871
Ridge standard alpha=0.001: pooled RMSE 14.885, pooled R2 0.837
Ridge standard alpha=0.01: pooled RMSE 14.885, pooled R2 0.837
Ridge standard alpha=0.1: pooled RMSE 14.883, pooled R2 0.837
Ridge standard alpha=1.0: pooled RMSE 14.872, pooled R2 0.837
Ridge standard alpha=10.0: pooled RMSE 14.850, pooled R2 0.838
Ridge standard alpha=100.0: pooled RMSE 15.128, pooled R2 0.832
Ridge standard alpha=1000.0: pooled RMSE 17.783, pooled R2 0.767
```

Selected configuration:

``` text
Ridge(alpha=1000.0), no scaler
```

Reason:

Among linear candidates, unscaled `Ridge(alpha=1000.0)` had the lowest
pooled validation RMSE. It did not beat persistence overall on
validation, but it was the strongest tested linear configuration.

Validation station consistency:

``` text
selected Ridge beats persistence by validation RMSE on 17 datasets
persistence beats selected Ridge by validation RMSE on 34 datasets
```

Largest validation wins for selected Ridge:

``` text
Sorakhutte (SC-36)-GD Labs: -3.165 RMSE vs persistence
Pulchowk (SC-44) - GD Labs: -2.536
Phora Durbar Kathman: -2.446
Gokarneshwor (SC-13) - GD Labs: -2.413
Gothatar (SC-12) - GD Labs: -2.158
```

Largest validation losses for selected Ridge:

``` text
Kritipur Ward 3 (SC-25) - GD Labs: +8.852 RMSE vs persistence
Haugal Ganesh Temple, Patan Durbar Square: +8.338
Sifal(SC-03)- GD Labs: +5.651
Teku Ward 12 (SC - 20) - GD Labs: +5.276
Sundarighat (SC-23) - GD Labs: +4.946
```

Frozen final test result:

``` text
selected Ridge test rows: 26,236
macro MAE: 7.437
macro RMSE: 10.076
macro mean R2: -29.748
macro median R2: 0.702
pooled MAE: 7.629
pooled RMSE: 12.591
pooled R2: 0.805
negative R2 datasets: 4
```

Fair test comparison:

``` text
Persistence pooled RMSE: 12.083
selected Ridge pooled RMSE: 12.591
Persistence pooled R2: 0.820
selected Ridge pooled R2: 0.805
```

Conclusion:

Validation-based Ridge selection improved the original untuned
`Ridge(alpha=10.0)` baseline, but the final selected linear model still
does not beat persistence on validation or test overall. This supports
moving next to nonlinear classical baselines after documenting this
linear-baseline result.

**Handover status:** Validation-selected linear baseline is complete.
Next recommended research step: nonlinear classical baselines, starting
with a clearly scoped Random Forest or XGBoost milestone.
