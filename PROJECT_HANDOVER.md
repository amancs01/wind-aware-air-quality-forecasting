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

Current convention after the wind-semantics correction:

-   `wind_direction` is meteorological direction: the direction FROM
    which wind blows, measured clockwise from north.
-   `wind_u` is the physical eastward wind component.
-   `wind_v` is the physical northward wind component.
-   `wind_speed`, `wind_u`, and `wind_v` are in km/h because the
    Open-Meteo downloader still uses the API default wind-speed unit.

The corrected component equations are:

``` python
wind_rad = np.deg2rad(df["wind_direction"])
wind_u = -wind_speed * np.sin(wind_rad)
wind_v = -wind_speed * np.cos(wind_rad)
```

Earlier generated `wind_u`/`wind_v` values used:

``` python
wind_u = wind_speed * np.cos(wind_rad)
wind_v = wind_speed * np.sin(wind_rad)
```

Those earlier values were speed-scaled circular direction encodings, so
the classical-model experiments were not meaningless: they retained the
same wind-speed and wind-direction information through an invertible
swap/sign transform. However, the columns were physically mislabeled.
The correction is necessary for defensible physical interpretation and
for future bearing/wind-alignment graph work.

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

Future directed wind-aware edges must distinguish meteorological FROM
direction from pollution-transport TO direction. Before comparing wind
with a source-to-target station bearing, use:

``` text
transport_direction = (wind_direction + 180) % 360
```

Do not compare raw `wind_direction` directly to a source-to-target
bearing unless the equation is explicitly modeling an upwind direction.
If future graph equations need SI units, consider converting wind speed
and components to m/s deliberately; do not silently mix units.

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

The project has completed the fair reusable baseline evaluation task,
validation-selected Ridge baseline, and validation-selected Random
Forest baseline, and validation-selected XGBoost baseline.

The current state is:

-   canonical hourly AQ migration is implemented and validated;
-   lag/rolling timestamp correctness is validated at 100%;
-   persistence and Ridge are now evaluated on identical test rows;
-   row-level prediction exports confirm exact station/source-index/
    timestamp/target matches between baselines;
-   Ridge has now been selected using validation data only;
-   production Ridge uses `LINEAR_BASELINE_ALPHA = 1000.0`;
-   the old `Ridge(alpha=10.0)` result remains historical context;
-   Random Forest has now been selected using validation data only;
-   production Random Forest uses the frozen constants in `config.py`;
-   XGBoost has now been selected using validation data only;
-   production XGBoost uses the frozen constants in `config.py`.

The fair result is that persistence still outperforms the
validation-selected Ridge baseline overall. Random Forest improves
pooled held-out test RMSE over persistence, and XGBoost very slightly
improves pooled held-out test RMSE over persistence, but Random Forest
remains the strongest pooled-RMSE model among completed baselines.
Persistence still wins station-level test RMSE on most datasets. This is
a legitimate research finding, not an evaluation-row artifact.

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

### Step 5 --- Continue the classical nonlinear sequence carefully

Random Forest and XGBoost are now complete. The next reasonable
experiment is not another test-set tweak; consider either a carefully
defined feature-engineering milestone using validation only, or a
sequence-model milestone if the thesis scope is ready for that step.
Do not start GRU, GNN, or graph integration without a separate plan.

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
Random Forest estimator: 100 trees, max_depth=10, min_samples_leaf=10,
max_features=1.0
XGBoost estimator: learning_rate=0.1, max_depth=3,
min_child_weight=5, early_stopping_rounds=50
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
6.  Is Random Forest still the frozen validation-selected configuration
    in `config.py`?
7.  Is XGBoost still the frozen validation-selected configuration in
    `config.py`?
8.  Has the feature set changed from `MODEL_FEATURE_COLUMNS`?
9.  Are generated outputs freshly regenerated after any code or
    preprocessing change?
10. Are low-variance station R2 values being interpreted carefully?

Until these are answered, do not interpret advanced-model performance as
trustworthy.

------------------------------------------------------------------------

**Handover status:** Ready for continuation in the ChatGPT Project.

**Immediate next conversation title suggestion:**\
`01 - Decide Next Post-Classical Modeling Experiment`

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
- At this milestone, Ridge remained `Ridge(alpha=10.0)` with no tuning
  and no feature-set change. Section 28 supersedes this as the current
  production linear baseline.

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

This next step has since been completed in Section 28.

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
Next recommended research step at that time was Random Forest; this has
since been completed in Section 29.

------------------------------------------------------------------------

## 29. Implemented Validation-Selected Random Forest Baseline

This milestone has now been implemented, frozen, tested once, documented,
and pushed on `main`.

Research question:

``` text
Does nonlinear modeling of the same existing information improve
one-hour-ahead PM2.5 forecasting beyond Persistence and Ridge?
```

Experimental controls:

- Feature set remained exactly `MODEL_FEATURE_COLUMNS`.
- No precipitation was added.
- No PM2.5 lag/rolling imputation was added.
- No preprocessing or graph code was changed.
- Models remained station-specific.
- One global Random Forest configuration was selected for all stations.
- Test data was not used during hyperparameter selection.

Initial design:

- `n_estimators` controls the number of trees. It was fixed rather than
  tuned aggressively because this milestone focuses on tree complexity.
- `max_depth` controls how deep each tree can grow. Shallower trees
  reduce memorization.
- `min_samples_leaf` controls how many rows must remain in a terminal
  leaf. Larger leaves smooth predictions and regularize the model.
- `max_features` controls how many features are considered at each
  split. All features can make individual trees stronger; `sqrt` can
  increase diversity.
- `random_state=42` makes repeated runs reproducible.

Resource note:

The originally suggested unbounded `max_depth=None` candidates exhausted
local memory before producing validation metrics, even with fewer trees
and no worker parallelism. The feasible validation grid therefore used:

``` text
n_estimators: 100
n_jobs: 1
max_depth: [10, 20]
min_samples_leaf: [1, 5, 10]
max_features: [1.0, "sqrt"]
```

Validation candidates:

``` text
Persistence: rows 25,689, macro RMSE 11.536, median R2 0.769, pooled RMSE 12.300, pooled R2 0.889
Ridge(alpha=1000.0): rows 25,689, macro RMSE 12.430, median R2 0.775, pooled RMSE 13.225, pooled R2 0.871
RF depth=10 leaf=1 features=1.0: macro RMSE 12.304, median R2 0.758, pooled RMSE 13.289, pooled R2 0.870
RF depth=10 leaf=1 features=sqrt: macro RMSE 15.328, median R2 0.672, pooled RMSE 16.152, pooled R2 0.808
RF depth=10 leaf=5 features=1.0: macro RMSE 11.891, median R2 0.817, pooled RMSE 12.567, pooled R2 0.884
RF depth=10 leaf=5 features=sqrt: macro RMSE 15.551, median R2 0.681, pooled RMSE 16.412, pooled R2 0.802
RF depth=10 leaf=10 features=1.0: macro RMSE 11.901, median R2 0.800, pooled RMSE 12.450, pooled R2 0.886
RF depth=10 leaf=10 features=sqrt: macro RMSE 16.157, median R2 0.647, pooled RMSE 16.985, pooled R2 0.788
RF depth=20 leaf=1 features=1.0: macro RMSE 12.320, median R2 0.758, pooled RMSE 13.364, pooled R2 0.869
RF depth=20 leaf=1 features=sqrt: macro RMSE 15.135, median R2 0.678, pooled RMSE 15.950, pooled R2 0.813
RF depth=20 leaf=5 features=1.0: macro RMSE 11.892, median R2 0.817, pooled RMSE 12.570, pooled R2 0.884
RF depth=20 leaf=5 features=sqrt: macro RMSE 15.509, median R2 0.671, pooled RMSE 16.336, pooled R2 0.804
RF depth=20 leaf=10 features=1.0: macro RMSE 11.901, median R2 0.800, pooled RMSE 12.450, pooled R2 0.886
RF depth=20 leaf=10 features=sqrt: macro RMSE 16.139, median R2 0.660, pooled RMSE 16.954, pooled R2 0.788
```

Selected configuration by pooled validation RMSE:

``` text
n_estimators = 100
max_depth = 10
min_samples_leaf = 10
max_features = 1.0
random_state = 42
n_jobs = 1
```

Validation comparison:

``` text
Persistence pooled RMSE: 12.300
Ridge(alpha=1000.0) pooled RMSE: 13.225
selected RF pooled RMSE: 12.450
```

Selected RF improves over Ridge on validation, but does not beat
persistence by pooled validation RMSE.

Validation station-level comparison:

``` text
selected RF beats persistence by validation RMSE on 27 datasets
persistence beats selected RF by validation RMSE on 24 datasets
```

Largest validation wins for selected RF:

``` text
Gothatar (SC-12) - GD Labs: -4.562 RMSE vs persistence
Dabali, Handigaun: -3.751
Phora Durbar Kathman: -3.258
Taudaha (SC - 09) - GD Labs: -2.800
Dhathutole, Handigaun: -2.346
```

Largest validation losses for selected RF:

``` text
Farsidol Relocated: +4.745 RMSE vs persistence
Haugal Ganesh Temple, Patan Durbar Square: +4.685
Chovar (SC - 07) - GD Labs: +4.264
Kritipur Ward 3 (SC-25) - GD Labs: +4.259
Kadhaghari (SC-42)-GD Labs: +3.714
```

Final matched test verification:

``` text
persistence rows: 26,236
Ridge rows: 26,236
Random Forest rows: 26,236
persistence/Ridge key match: true
persistence/RF key match: true
Ridge/RF key match: true
timestamp mismatches: 0
source-index mismatches: 0
target max absolute difference: 0.0
```

Final held-out test comparison:

``` text
Persistence: rows 26,236, macro MAE 5.830, macro RMSE 8.815, macro mean R2 0.692, macro median R2 0.763, pooled MAE 6.005, pooled RMSE 12.083, pooled R2 0.820
Ridge(alpha=1000.0): rows 26,236, macro MAE 7.437, macro RMSE 10.076, macro mean R2 -29.748, macro median R2 0.702, pooled MAE 7.629, pooled RMSE 12.591, pooled R2 0.805
Random Forest: rows 26,236, macro MAE 6.735, macro RMSE 9.695, macro mean R2 -78.855, macro median R2 0.704, pooled MAE 6.556, pooled RMSE 11.652, pooled R2 0.833
```

Random Forest relative to persistence on test:

``` text
pooled RMSE difference (RF - Persistence): -0.430
pooled MAE difference (RF - Persistence): +0.551
pooled RMSE improvement: 3.56%
RF beats persistence by test RMSE on 11 datasets
persistence beats RF by test RMSE on 40 datasets
```

Largest test wins for selected RF:

``` text
Phora Durbar Kathman: -7.517 RMSE vs persistence
CEN-SR-02_ Farsidol Brick Factories: -5.652
Dhathutole, Handigaun: -3.905
Embassy Kathmandu: -1.598
Gaushala Chowk (SC-01) - GD Labs: -1.269
```

Largest test losses for selected RF:

``` text
Sundarighat (SC-23) - GD Labs: +7.286 RMSE vs persistence
Pulchowk Engineering Campus - ICE Labs: +7.096
Mid Baneshwor (SC-39)-GD Labs: +6.853
Tarakeswor (SC-14)-GD Labs: +6.154
Kupondole (SC-40)-GD Labs: +5.044
```

Conclusion:

Nonlinear Random Forest interactions helped reduce larger squared errors
overall on held-out test data, giving the first model that beats
persistence by pooled RMSE. The result is heterogeneous: persistence
still wins on most stations and has lower pooled MAE. This supports
testing another nonlinear classical model next, but without changing the
feature set based on the RF test result.

**Handover status:** Validation-selected Random Forest baseline is
complete. The XGBoost follow-up has since been completed in Section 30.

------------------------------------------------------------------------

## 30. Implemented Validation-Selected XGBoost Baseline

This milestone has now been implemented, frozen, tested once, documented,
and pushed on `main`.

Environment:

``` text
Python: 3.12.0
XGBoost: 3.4.0
Dependency: requirements.txt now declares xgboost>=3.4.0
```

Research question:

``` text
Can sequential gradient-boosted trees exploit the same PM2.5-history,
weather, wind, and temporal features more effectively than persistence,
Ridge, and Random Forest?
```

Experimental controls:

- Feature set remained exactly `MODEL_FEATURE_COLUMNS`.
- No precipitation, station IDs, spatial features, missing indicators,
  or new lag/window features were added.
- XGBoost native missing-value handling was not used to expand the
  training/evaluation frame.
- No preprocessing or graph code was changed.
- Models remained station-specific.
- One global XGBoost hyperparameter configuration was selected for all
  stations.
- Station-specific `best_iteration` was allowed through early stopping.
- Test data was not used during hyperparameter selection.

XGBoost design:

- Gradient boosting builds trees sequentially; each new tree tries to
  correct the current ensemble's remaining error.
- `learning_rate` controls how strongly each new tree updates the
  prediction.
- `max_depth` controls each tree's maximum complexity.
- `min_child_weight` controls how easily small specialized branches are
  created.
- `subsample` uses only part of the training rows per tree to reduce
  overfitting.
- `colsample_bytree` uses only part of the features per tree to reduce
  feature over-reliance.
- Early stopping stops adding trees when validation RMSE stops
  improving.

Validation grid:

``` text
learning_rate: [0.03, 0.10]
max_depth: [3, 6]
min_child_weight: [1, 5]
fixed: subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0,
reg_lambda=1.0, n_estimators=1000, early_stopping_rounds=50,
objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
random_state=42, n_jobs=1
```

Validation candidates:

``` text
Persistence: rows 25,689, macro RMSE 11.536, median R2 0.769, pooled RMSE 12.300, pooled R2 0.889
Ridge(alpha=1000.0): rows 25,689, macro RMSE 12.430, median R2 0.775, pooled RMSE 13.225, pooled R2 0.871
Random Forest: rows 25,689, macro RMSE 11.901, median R2 0.800, pooled RMSE 12.450, pooled R2 0.886
XGB lr=0.03 depth=3 child=1: macro RMSE 12.232, median R2 0.788, pooled RMSE 12.904, pooled R2 0.877
XGB lr=0.03 depth=3 child=5: macro RMSE 12.052, median R2 0.786, pooled RMSE 12.725, pooled R2 0.881
XGB lr=0.03 depth=6 child=1: macro RMSE 13.275, median R2 0.737, pooled RMSE 14.150, pooled R2 0.853
XGB lr=0.03 depth=6 child=5: macro RMSE 12.657, median R2 0.761, pooled RMSE 13.481, pooled R2 0.866
XGB lr=0.10 depth=3 child=1: macro RMSE 12.079, median R2 0.782, pooled RMSE 12.839, pooled R2 0.879
XGB lr=0.10 depth=3 child=5: macro RMSE 11.938, median R2 0.787, pooled RMSE 12.700, pooled R2 0.881
XGB lr=0.10 depth=6 child=1: macro RMSE 13.315, median R2 0.749, pooled RMSE 14.215, pooled R2 0.851
XGB lr=0.10 depth=6 child=5: macro RMSE 12.756, median R2 0.762, pooled RMSE 13.575, pooled R2 0.864
```

Selected configuration by pooled validation RMSE:

``` text
learning_rate = 0.1
max_depth = 3
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.0
reg_lambda = 1.0
n_estimators = 1000
early_stopping_rounds = 50
objective = "reg:squarederror"
eval_metric = "rmse"
tree_method = "hist"
random_state = 42
n_jobs = 1
```

Validation best-iteration behavior:

``` text
min: 19
median: 67
mean: 88.78
max: 261
stations hitting n_estimators=1000: 0
```

Validation comparison:

``` text
Persistence pooled RMSE: 12.300
Ridge(alpha=1000.0) pooled RMSE: 13.225
Random Forest pooled RMSE: 12.450
selected XGBoost pooled RMSE: 12.700
```

Selected XGBoost did not beat persistence or Random Forest by pooled
validation RMSE, but it did beat Ridge.

Validation station-level comparison:

``` text
XGBoost beats persistence by validation RMSE on 25 datasets
persistence beats XGBoost by validation RMSE on 26 datasets
XGBoost beats Random Forest by validation RMSE on 23 datasets
Random Forest beats XGBoost by validation RMSE on 28 datasets
```

Largest validation wins for XGBoost over persistence:

``` text
Gothatar (SC-12) - GD Labs: -5.840 RMSE vs persistence
Phora Durbar Kathman: -4.078
Sorakhutte (SC-36)-GD Labs: -3.981
Dabali, Handigaun: -3.386
Nakhipot (SC-08) - GD Labs: -2.914
```

Largest validation losses for XGBoost versus persistence:

``` text
Farsidol Relocated: +6.181 RMSE vs persistence
Chovar (SC - 07) - GD Labs: +6.155
Kadhaghari (SC-42)-GD Labs: +4.804
Sundarighat (SC-23) - GD Labs: +4.675
Kritipur Ward 3 (SC-25) - GD Labs: +4.179
```

Final matched test verification:

``` text
persistence rows: 26,236
Ridge rows: 26,236
Random Forest rows: 26,236
XGBoost rows: 26,236
all model key matches against persistence: true
timestamp mismatches: 0
source-index mismatches: 0
target max absolute difference: 0.0
```

Final held-out test comparison:

``` text
Persistence: rows 26,236, macro MAE 5.830, macro RMSE 8.815, macro mean R2 0.692, macro median R2 0.763, pooled MAE 6.005, pooled RMSE 12.083, pooled R2 0.820
Ridge(alpha=1000.0): rows 26,236, macro MAE 7.437, macro RMSE 10.076, macro mean R2 -29.748, macro median R2 0.702, pooled MAE 7.629, pooled RMSE 12.591, pooled R2 0.805
Random Forest: rows 26,236, macro MAE 6.735, macro RMSE 9.695, macro mean R2 -78.855, macro median R2 0.704, pooled MAE 6.556, pooled RMSE 11.652, pooled R2 0.833
XGBoost: rows 26,236, macro MAE 7.336, macro RMSE 10.257, macro mean R2 -443.742, macro median R2 0.701, pooled MAE 7.184, pooled RMSE 12.043, pooled R2 0.821
```

XGBoost relative to persistence on test:

``` text
pooled RMSE difference (XGB - Persistence): -0.039
pooled RMSE improvement: 0.33%
pooled MAE difference (XGB - Persistence): +1.179
pooled MAE change: approximately 19.63% worse than persistence
XGBoost beats persistence by test RMSE on 11 datasets
persistence beats XGBoost by test RMSE on 40 datasets
```

XGBoost relative to Random Forest on test:

``` text
pooled RMSE difference (XGB - RF): +0.391
pooled MAE difference (XGB - RF): +0.628
pooled R2 difference (XGB - RF): -0.011
XGBoost beats Random Forest by test RMSE on 18 datasets
Random Forest beats XGBoost by test RMSE on 33 datasets
```

Largest test wins for XGBoost over persistence:

``` text
Phora Durbar Kathman: -10.388 RMSE vs persistence
CEN-SR-02_ Farsidol Brick Factories: -5.887
Dhathutole, Handigaun: -3.957
Pulchowk (SC-44) - GD Labs: -3.091
Embassy Kathmandu: -1.208
```

Largest test losses for XGBoost versus persistence:

``` text
Sundarighat (SC-23) - GD Labs: +17.435 RMSE vs persistence
Mid Baneshwor (SC-39)-GD Labs: +9.748
Pulchowk Engineering Campus - ICE Labs: +8.390
Sorakhutte (SC-36)-GD Labs: +7.163
Tarakeswor (SC-14)-GD Labs: +6.178
```

Conclusion:

Sequential gradient boosting did not extract more useful overall signal
than Random Forest from the current feature set. It barely improves
pooled test RMSE over persistence, but the gain is only 0.33%, pooled
MAE is substantially worse, and persistence still wins station-level
test RMSE on 40 of 51 datasets. Random Forest remains the strongest
completed model by pooled held-out test RMSE.

**Handover status:** Validation-selected XGBoost baseline is complete.
Next recommended research step: decide whether to run a validation-only
feature-engineering milestone or move to a sequence-model baseline. Do
not modify features based on XGBoost test results without a new
validation-first experiment.

------------------------------------------------------------------------

## 31. Validation-Only Feature Ablation Study

This milestone has now been implemented, run once, documented, and
pushed on `main`.

Purpose:

``` text
Identify which existing feature groups contribute validation signal,
especially whether physical wind components add predictive information
beyond local PM2.5 history, time, and ordinary meteorology.
```

Important controls:

-   Train and validation splits only.
-   The test split was not loaded or used.
-   Production `MODEL_FEATURE_COLUMNS` was not changed.
-   Random Forest hyperparameters were not retuned.
-   The frozen Random Forest configuration was used:
    `n_estimators=100`, `max_depth=10`, `min_samples_leaf=10`,
    `max_features=1.0`, `random_state=42`, `n_jobs=1`.
-   Every ablation used the same full-feature-valid train/validation
    rows to avoid confounding feature contribution with data
    availability.

Actual current `MODEL_FEATURE_COLUMNS`:

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

Feature groups:

``` text
A0 Persistence:
    prediction = pm2_5

A1 Current PM only:
    pm2_5

A2 PM history:
    pm2_5, lag_6, lag_24, rolling_mean_6, rolling_std_6

A3 PM history + time:
    A2 + hour_sin, hour_cos, month_sin, month_cos

A4 PM history + time + non-wind weather:
    A3 + temperature, humidity, pressure, dew_point

A5 PM history + time + wind:
    A3 + wind_u, wind_v

A6 Full current feature set:
    A3 + weather + wind
```

Fixed frame verification:

``` text
datasets: 51
train rows: 115,725
validation rows: 25,689
train-row mismatching stations across variants: 0
validation-row mismatching stations across variants: 0
validation-target mismatching stations across variants: 0
```

Validation results:

``` text
Persistence: rows 25,689, macro MAE 7.403, macro RMSE 11.536, macro mean R2 0.755, macro median R2 0.769, pooled MAE 7.292, pooled RMSE 12.300, pooled R2 0.889
A1 Current PM only RF: rows 25,689, macro MAE 8.472, macro RMSE 12.787, macro mean R2 0.690, macro median R2 0.768, pooled MAE 8.311, pooled RMSE 13.453, pooled R2 0.867
A2 PM history RF: rows 25,689, macro MAE 8.480, macro RMSE 12.774, macro mean R2 0.687, macro median R2 0.766, pooled MAE 8.303, pooled RMSE 13.407, pooled R2 0.868
A3 PM history + time RF: rows 25,689, macro MAE 7.835, macro RMSE 11.965, macro mean R2 0.717, macro median R2 0.791, pooled MAE 7.586, pooled RMSE 12.499, pooled R2 0.885
A4 Full minus wind RF: rows 25,689, macro MAE 7.700, macro RMSE 11.950, macro mean R2 0.726, macro median R2 0.788, pooled MAE 7.402, pooled RMSE 12.497, pooled R2 0.885
A5 History + time + wind RF: rows 25,689, macro MAE 7.784, macro RMSE 11.912, macro mean R2 0.717, macro median R2 0.803, pooled MAE 7.498, pooled RMSE 12.426, pooled R2 0.886
A6 Full RF: rows 25,689, macro MAE 7.674, macro RMSE 11.900, macro mean R2 0.727, macro median R2 0.800, pooled MAE 7.369, pooled RMSE 12.449, pooled R2 0.886
```

Incremental comparisons use:

``` text
positive improvement = lower error after adding features
```

Findings:

``` text
Historical PM contribution, A2 - A1:
    pooled RMSE improvement +0.046 (+0.34%)

Time contribution, A3 - A2:
    pooled RMSE improvement +0.908 (+6.78%)

Non-wind weather contribution, A4 - A3:
    pooled RMSE improvement +0.002 (+0.01%)

Wind without other weather, A5 - A3:
    pooled RMSE improvement +0.073 (+0.58%)

Conditional wind contribution, A6 - A4:
    pooled RMSE improvement +0.048 (+0.38%)

Conditional non-wind weather contribution, A6 - A5:
    pooled RMSE improvement -0.023 (-0.18%)
```

Primary wind-specific result:

``` text
A4 full-minus-wind pooled RMSE: 12.497
A6 full pooled RMSE: 12.449
conditional wind pooled RMSE improvement: +0.048
A4 pooled MAE: 7.402
A6 pooled MAE: 7.369
A4 pooled R2: 0.885
A6 pooled R2: 0.886
A4 macro RMSE: 11.950
A6 macro RMSE: 11.900
```

Station-level wind result:

``` text
wind improved RMSE: 24 stations
wind worsened RMSE: 27 stations
ties: 0 stations
median station wind RMSE improvement: -0.011
```

Largest validation wind wins:

``` text
Farsidol Relocated: +1.705 RMSE
Sitapaila (SC-30) - GD Labs: +0.392
Tarakeswor (SC-14)-GD Labs: +0.355
Sanepa (SC - 22) - GD Labs: +0.355
Gaushala Chowk (SC-01) - GD Labs: +0.339
Chovar (SC - 07) - GD Labs: +0.274
Baluwatar (SC-02) - GD Labs: +0.264
Tokha (SC - 32) - GD Labs: +0.253
Gokarneshwor (SC-13) - GD Labs: +0.214
Mid Baneshwor (SC-39)-GD Labs: +0.202
```

Largest validation wind losses:

``` text
Tankeshwor (SC- 18) - GD Labs: -0.463 RMSE
Balkumari(SC-28)- GD Labs: -0.452
Teku Ward 12 (SC - 20) - GD Labs: -0.377
Ramkot (SC - 10) - GD Labs: -0.299
Sundarighat (SC-23) - GD Labs: -0.184
Tyanglaphat (SC - 21) - GD Labs: -0.179
Ranibari (SC-43)-GD Labs: -0.178
Imadol(SC-27)- GD Labs: -0.099
Kaushaltar (SC - 33) GD labs: -0.086
Jadibuti (SC-35)-GD Labs: -0.074
```

Interpretation:

-   Persistence remains the best pooled validation RMSE benchmark.
-   Random Forest using only current PM2.5 is not equivalent to
    persistence and performs worse than persistence.
-   PM2.5 history adds only a small pooled validation improvement beyond
    current PM2.5.
-   Cyclical time features provide the largest incremental validation
    gain.
-   Non-wind weather adds almost no pooled RMSE value after PM history
    and time, though it improves pooled MAE.
-   Wind features add a small positive pooled validation signal after
    PM history, time, and ordinary weather, but the benefit is
    heterogeneous and not station-wide.
-   This is predictive evidence, not causal evidence.

Methodological warning:

The existing held-out test split has already been observed during
Ridge, Random Forest, XGBoost, and wind-semantics correction milestones.
Do not use this validation ablation to select a new production feature
subset and then claim the existing test split is a fresh independent
confirmation. If the feature set changes materially later, choose the
evaluation strategy deliberately, such as retaining the full features
for interpretability, using rolling-origin validation, or creating a
fresh chronological holdout.

Generated outputs:

``` text
results/feature_analysis/ablation/ablation_summary.csv
results/feature_analysis/ablation/ablation_station_metrics.csv
results/feature_analysis/ablation/ablation_comparisons.csv
results/feature_analysis/ablation/wind_station_comparison.csv
results/feature_analysis/ablation/row_identity.csv
results/feature_analysis/ablation/feature_groups.csv
```

**Handover status:** Validation-only feature ablation is complete.
Recommended next step: investigate richer temporal modeling and/or
wind-spatial interactions using validation-first methodology. Do not
change production features or use the existing test split as fresh
confirmation without a deliberate evaluation design.

------------------------------------------------------------------------

## 32. Rolling-Origin / Expanding-Window Validation

This milestone has now been implemented, run once, documented, and
pushed on `main`.

Purpose:

``` text
Test temporal robustness of the frozen classical baselines across
multiple chronological development windows without using the existing
final 15% test split.
```

Fold design, applied independently to each prepared station dataset:

``` text
Fold 1: train 0-55%, validate 55-65%
Fold 2: train 0-65%, validate 65-75%
Fold 3: train 0-75%, validate 75-85%
```

Controls:

-   Used only `data/processed/prepared/`.
-   Did not load `data/processed/split/test/`.
-   Did not retune any model.
-   Did not change `MODEL_FEATURE_COLUMNS`.
-   Evaluated only Persistence, `Ridge(alpha=1000)`, and the frozen
    Random Forest configuration.
-   Used the same full-feature-valid train/validation rows for all
    models within each station/fold.

Frozen Random Forest configuration:

``` text
n_estimators = 100
max_depth = 10
min_samples_leaf = 10
max_features = 1.0
random_state = 42
n_jobs = 1
```

Rows:

``` text
Fold 1: 51 datasets, train rows 89,720, validation rows 17,127
Fold 2: 51 datasets, train rows 106,847, validation rows 17,803
Fold 3: 51 datasets, train rows 124,650, validation rows 16,764
```

Two tiny prepared datasets were skipped in every fold because they had
insufficient full-feature-valid rows:

``` text
Kathmandu University__sensor_15286458
Tarakeswor (SC-15)- GD Labs
```

Fold results:

``` text
Fold 1:
Persistence: macro MAE 9.931, macro RMSE 14.681, median R2 0.747, pooled MAE 10.011, pooled RMSE 15.611, pooled R2 0.829
Ridge(alpha=1000): macro MAE 10.313, macro RMSE 14.469, median R2 0.773, pooled MAE 10.540, pooled RMSE 15.541, pooled R2 0.831
Random Forest: macro MAE 9.819, macro RMSE 14.270, median R2 0.773, pooled MAE 9.490, pooled RMSE 14.538, pooled R2 0.852

Fold 2:
Persistence: macro MAE 8.670, macro RMSE 13.314, median R2 0.742, pooled MAE 9.136, pooled RMSE 17.689, pooled R2 0.777
Ridge(alpha=1000): macro MAE 9.330, macro RMSE 13.246, median R2 0.754, pooled MAE 9.959, pooled RMSE 16.503, pooled R2 0.806
Random Forest: macro MAE 8.490, macro RMSE 12.747, median R2 0.786, pooled MAE 8.568, pooled RMSE 15.107, pooled R2 0.837

Fold 3:
Persistence: macro MAE 7.258, macro RMSE 11.014, median R2 0.735, pooled MAE 6.951, pooled RMSE 11.862, pooled R2 0.896
Ridge(alpha=1000): macro MAE 8.449, macro RMSE 12.035, median R2 0.747, pooled MAE 8.218, pooled RMSE 12.978, pooled R2 0.876
Random Forest: macro MAE 7.437, macro RMSE 11.462, median R2 0.725, pooled MAE 7.175, pooled RMSE 12.465, pooled R2 0.885
```

Model-vs-persistence comparisons:

``` text
Fold 1 RF pooled RMSE improvement: +1.073; station wins 31/51
Fold 2 RF pooled RMSE improvement: +2.582; station wins 34/51
Fold 3 RF pooled RMSE improvement: -0.603; station wins 27/51

Fold 1 Ridge pooled RMSE improvement: +0.071; station wins 31/51
Fold 2 Ridge pooled RMSE improvement: +1.185; station wins 24/51
Fold 3 Ridge pooled RMSE improvement: -1.116; station wins 12/51
```

Random Forest station-win consistency:

``` text
RF wins 3/3 folds: 10 stations
RF wins 2/3 folds: 24 stations
RF wins 1/3 folds: 14 stations
RF wins 0/3 folds: 3 stations
```

Largest 3/3 RF-win stations by average RMSE improvement included:

``` text
Dabali, Handigaun
Embassy Kathmandu
Gokarneshwor (SC-13) - GD Labs
Sanepa (SC - 22) - GD Labs
Sorakhutte (SC-36)-GD Labs
Sunakothi (SC - 06) - GD Labs
Chhetrapati (SC - 19) - GD Labs
Dhathutole, Handigaun
Tyanglaphat (SC - 21) - GD Labs
Imadol(SC-27)- GD Labs
```

Distribution shifts:

The validation target distribution shifted substantially for several
station/fold windows. Largest absolute target-mean shifts included:

``` text
Phora Durbar Kathman fold 3: +60.5
Nakhipot (SC-08) - GD Labs fold 1: +57.2
Lamtangil (SC-04)- GD Labs fold 2: -51.7
Jadibuti (SC-35)-GD Labs fold 3: -51.0
Sifal(SC-03)- GD Labs fold 1: -48.7
Bagdol fold 3: -48.2
Lamtangil (SC-04)- GD Labs fold 3: -47.2
Balaju (SC-26)- GD Labs fold 3: -46.5
Baluwatar (SC-02) - GD Labs fold 1: +45.4
Tokha (SC - 32) - GD Labs fold 3: -44.8
```

Sundarighat diagnostic:

``` text
Fold 1:
train target mean 83.4, validation mean 96.0, shift +12.6
Persistence RMSE 14.543, RF RMSE 14.269, Ridge RMSE 15.716

Fold 2:
train target mean 85.4, validation mean 61.6, shift -23.8
Persistence RMSE 11.534, RF RMSE 10.729, Ridge RMSE 12.825

Fold 3:
train target mean 82.0, validation mean 91.1, shift +9.2
Persistence RMSE 13.529, RF RMSE 18.773, Ridge RMSE 19.449
```

Interpretation:

-   Random Forest is more robust than Ridge and beats Persistence in
    pooled RMSE in folds 1 and 2.
-   Persistence remains the strongest model in fold 3 by pooled RMSE
    and pooled MAE.
-   RF wins more stations than it loses in all three folds, but pooled
    performance can still reverse when difficult high-weight windows
    dominate.
-   Temporal distribution shift is a real evaluation issue; single
    fixed-window conclusions are not enough.
-   Next research should focus on richer temporal modeling and/or
    rolling-origin evaluation design before graph integration.

Generated outputs:

``` text
results/rolling_origin/rolling_origin_fold_summary.csv
results/rolling_origin/rolling_origin_station_metrics.csv
results/rolling_origin/rolling_origin_model_comparisons.csv
results/rolling_origin/rolling_origin_rf_station_wins.csv
results/rolling_origin/rolling_origin_distribution_shifts.csv
results/rolling_origin/rolling_origin_fold_frames.csv
```

**Handover status:** Rolling-origin validation is complete. Recommended
next step: use these robustness results to decide whether to build a
sequence-model baseline, refine temporal evaluation, or design
wind-spatial interactions. Do not retune existing classical baselines or
reuse the final test split for feature/model selection.

## 33. LSTM sequence dataset validation

Implemented validation-only sequence dataset design for a future LSTM:

``` text
scripts/19_lstm_sequence_dataset_validation.py
scripts/analysis/lstm_sequence_dataset_validation.py
```

No LSTM was implemented or trained.

Main source decision:

``` text
Use: data/processed/featured/
Do not build sequence windows from: data/processed/prepared/
```

Reason: `data/processed/prepared/` drops rows with missing current or
target PM2.5, so adjacent prepared rows are not necessarily adjacent
hours. The sequence validator confirmed this directly:

``` text
Featured stations checked: 56
Featured invalid hourly gaps: 0
Featured largest gap: 1 hour

Prepared stations with row gaps: 51
Prepared invalid hourly gaps: 5,051
Prepared largest gap: 11,636 hours
```

Future LSTM window definition:

``` text
Input window: 24 consecutive hourly rows
Target: PM2.5 exactly 1 hour after the final input timestamp
No window may cross missing/gap periods
No window may cross train/validation/test split boundaries
No future information is included in input columns
```

The split boundary timestamps are derived from the existing prepared
chronological split convention, but all actual windows are built and
validated from the hourly-continuous featured files.

Input-design comparison:

``` text
Full MODEL_FEATURE_COLUMNS:
pm2_5, lag_6, lag_24, rolling_mean_6, rolling_std_6, hour_sin, hour_cos,
month_sin, month_cos, wind_u, wind_v, temperature, humidity, pressure,
dew_point

Recommended sequence-native first LSTM baseline:
pm2_5, hour_sin, hour_cos, month_sin, month_cos, temperature, humidity,
pressure, dew_point, wind_u, wind_v
```

Recommendation: use the sequence-native 11-column design for the first
LSTM baseline. The LSTM's scientific purpose is to learn temporal
dependencies from the 24-hour sequence itself. Adding hand-engineered
lag/rolling PM2.5 summaries would partially duplicate the LSTM's temporal
role, reduce interpretability, and reject more otherwise valid windows
because lag/rolling columns are often missing after PM2.5 gaps.

Aggregate validation counts:

``` text
Sequence-native design:
train:      252,725 candidate targets, 101,168 accepted,
            150,319 rejected for missing values, 0 discontinuities,
            6 split-boundary crossings, 1,232 insufficient-history cases
validation: 75,370 candidate targets, 22,657 accepted,
            51,486 rejected for missing values, 0 discontinuities,
            1,224 split-boundary crossings, 3 insufficient-history cases
test:       61,818 candidate targets, 22,672 accepted,
            37,918 rejected for missing values, 0 discontinuities,
            1,224 split-boundary crossings, 4 insufficient-history cases

Full MODEL_FEATURE_COLUMNS design:
train:      81,702 accepted
validation: 18,677 accepted
test:       18,928 accepted
```

Proof checks over accepted windows:

``` text
Full MODEL_FEATURE_COLUMNS accepted windows: 119,307
Sequence-native accepted windows: 146,497

Invalid input row counts: 0
Invalid hourly input gaps: 0
Invalid target gaps: 0
Invalid split memberships: 0
```

Stations flagged as too sparse for the recommended design:

``` text
Kathmandu University__sensor_15286458
Kathmandu University__sensor_15286975
Kathmandu University__sensor_15286980
Pulchowk (SC-15)-GD Labs
Purano naikap (SC-29)-GD Labs
Ramkot (SC - 10) - GD Labs
Tarakeswor (SC-15)- GD Labs
```

Generated outputs:

``` text
results/lstm_sequence_validation/sequence_source_assessment.csv
results/lstm_sequence_validation/sequence_station_split_counts.csv
results/lstm_sequence_validation/sequence_summary.csv
results/lstm_sequence_validation/sequence_accepted_window_checks.csv
results/lstm_sequence_validation/sequence_stations_too_few.csv
results/lstm_sequence_validation/sequence_input_designs.csv
results/lstm_sequence_validation/sequence_validation_report.md
```

Recommended future dataset architecture:

-   Build a sequence index from featured station files with station,
    split, input-start timestamp, input-end timestamp, and target
    timestamp.
-   Materialize tensors from that index with shape
    `(n_sequences, 24, 11)` and scalar next-hour PM2.5 targets.
-   Fit any scaler on training input rows only, then apply it unchanged
    to validation and test.
-   Keep station and timestamp metadata with each sequence so later
    station-specific, pooled, or graph-aware models can reuse the same
    validated sequence index.

**Handover status:** LSTM sequence dataset validation is complete. The
next step may be implementation of the actual dataset loader and a first
LSTM baseline, but no LSTM training has been performed yet.

## 34. First station-specific LSTM baseline

Implemented the first validation-only LSTM forecasting baseline:

``` text
scripts/20_lstm_baseline.py
scripts/analysis/lstm_baseline.py
```

The final test split was not evaluated.

Runtime:

``` text
PyTorch: 2.13.0+cpu
CUDA available: False
Device used: CPU
```

Sequence design:

``` text
Source: data/processed/featured/
Input window: 24 consecutive hourly rows
Target: PM2.5 exactly 1 hour after final input timestamp
Input size: 11
Input columns:
pm2_5, hour_sin, hour_cos, month_sin, month_cos, temperature, humidity,
pressure, dew_point, wind_u, wind_v
```

The implementation reuses the exact sequence validity rules from the
LSTM sequence validator. It does not build windows from prepared-row
position and does not use handcrafted lag/rolling columns.

Model configuration:

``` text
Station-specific LSTM
hidden_size=64
num_layers=1
batch_first=True
Linear head -> 1 PM2.5 prediction
Adam learning_rate=0.001
MSE loss
batch_size=64
max_epochs=50
early stopping patience=5
random seed=42
```

Scaling:

-   input scaler fit on each station's training sequences only,
-   target scaler fit on each station's training targets only,
-   validation data used only after train-fitted scalers were fixed,
-   predictions inverse-transformed before PM2.5 metrics.

Stations:

``` text
Trained: 51
Skipped: 5
```

Skipped stations:

``` text
Kathmandu University__sensor_15286458: fewer than 100 training sequences
Kathmandu University__sensor_15286975: missing prepared split boundaries
Kathmandu University__sensor_15286980: missing prepared split boundaries
Pulchowk (SC-15)-GD Labs: missing prepared split boundaries
Tarakeswor (SC-15)- GD Labs: fewer than 100 training sequences
```

Native LSTM validation cohort:

``` text
Validation sequences: 22,657
Macro MAE: 10.895
Macro RMSE: 15.112
Macro median R2: 0.713
Pooled MAE: 9.619
Pooled RMSE: 15.036
Pooled R2: 0.834
```

Matched validation cohort for fair comparison:

``` text
Matched validation rows: 22,477
Unmatched native LSTM validation rows: 180

LSTM:
macro MAE 10.889, macro RMSE 15.103, macro median R2 0.717,
pooled MAE 9.608, pooled RMSE 15.021, pooled R2 0.835

Persistence:
macro MAE 7.180, macro RMSE 11.186, macro median R2 0.769,
pooled MAE 7.142, pooled RMSE 11.848, pooled R2 0.897

Frozen Random Forest:
macro MAE 7.503, macro RMSE 11.606, macro median R2 0.793,
pooled MAE 7.286, pooled RMSE 12.233, pooled R2 0.890
```

Station win counts by RMSE on matched validation timestamps:

``` text
LSTM beats Persistence: 10 / 51 stations
LSTM loses to Persistence: 41 / 51 stations

LSTM beats Random Forest: 7 / 51 stations
LSTM loses to Random Forest: 44 / 51 stations

LSTM beats both Persistence and RF: 4 / 51 stations
Median LSTM minus Persistence RMSE: +2.731
Median LSTM minus RF RMSE: +2.352
```

Best-epoch distribution:

``` text
count 51, mean 16.3, median 15, min 1, max 50
best_epoch <= 5: 9 stations
best_epoch >= 40: 2 stations
hit max epoch: 1 station
```

Interpretation:

-   The first LSTM baseline is technically valid and follows the
    sequence-data constraints.
-   It does not currently add useful temporal signal beyond Persistence
    or the frozen Random Forest.
-   Persistence remains a very strong one-hour benchmark.
-   The LSTM shows heterogeneous behavior: several stations stop very
    early, one station reaches the maximum epoch, and the largest
    train/validation loss gap is at Balkumari.
-   Do not tune LSTM hyperparameters or evaluate test from this result
    alone. A better next step is error diagnostics and possibly a
    simpler sequence baseline, stronger regularization, or pooled
    sequence modeling before moving to Transformer or graph work.

Generated outputs:

``` text
results/lstm/validation_station_metrics.csv
results/lstm/validation_summary.csv
results/lstm/validation_predictions.csv
results/lstm/validation_matched_predictions.csv
results/lstm/validation_matched_summary.csv
results/lstm/validation_station_win_counts.csv
results/lstm/validation_win_summary.csv
results/lstm/training_history.csv
results/lstm/skipped_stations.csv
results/lstm/lstm_validation_report.md
```

**Handover status:** First LSTM baseline is complete on train +
validation only. Do not report any final-test LSTM result yet; none has
been produced.

## 35. Persistence-anchored residual LSTM

Implemented a separate residual LSTM experiment:

``` text
scripts/21_lstm_residual_baseline.py
scripts/analysis/lstm_residual_baseline.py
```

The original direct-target LSTM results under `results/lstm/` were left
unchanged. Residual outputs are written under:

``` text
results/lstm/residual/
```

The final test split was not evaluated.

The experiment kept the same setup as the direct LSTM:

``` text
Source: data/processed/featured/
Window: 24 consecutive hourly rows
Input columns: 11 sequence-native features
Training: station-specific
Architecture: 1-layer LSTM, hidden_size=64, Linear head
Optimizer: Adam learning_rate=0.001
Loss: MSE
Batch size: 64
Max epochs: 50
Patience: 5
Seed: 42
Device: CPU, PyTorch 2.13.0+cpu
```

Only the prediction target changed:

``` text
delta_pm25 = PM2.5(t+1) - PM2.5(t)

final prediction = PM2.5(t) + predicted_delta
```

Residual target scaling was fit on each station's training residuals
only. Input scalers were still fit on training input values only.

Target distribution comparison:

``` text
Train absolute target mean/std: 68.673 / 42.349
Train residual target mean/std: 0.016 / 21.297

Validation absolute target mean/std: 59.297 / 36.950
Validation residual target mean/std: 0.083 / 11.840
```

Cohort:

``` text
Trained stations: 51
Skipped stations: 5
Validation sequences: 22,657
Matched comparison rows: 22,477
```

Skipped stations:

``` text
Kathmandu University__sensor_15286458
Kathmandu University__sensor_15286975
Kathmandu University__sensor_15286980
Pulchowk (SC-15)-GD Labs
Tarakeswor (SC-15)- GD Labs
```

Native residual LSTM validation:

``` text
Macro MAE: 6.653
Macro RMSE: 10.069
Macro median R2: 0.836
Pooled MAE: 6.577
Pooled RMSE: 10.583
Pooled R2: 0.918
```

Matched four-way validation comparison:

``` text
Direct LSTM:
pooled MAE 9.608, pooled RMSE 15.021, pooled R2 0.835,
macro RMSE 15.103, macro median R2 0.717

Residual LSTM:
pooled MAE 6.580, pooled RMSE 10.588, pooled R2 0.918,
macro RMSE 10.081, macro median R2 0.836

Persistence:
pooled MAE 7.142, pooled RMSE 11.848, pooled R2 0.897,
macro RMSE 11.186, macro median R2 0.769

Frozen Random Forest:
pooled MAE 7.286, pooled RMSE 12.233, pooled R2 0.890,
macro RMSE 11.606, macro median R2 0.793
```

Station RMSE wins:

``` text
Residual LSTM beats Persistence: 49 / 51
Residual LSTM beats RF: 41 / 51
Residual LSTM beats direct LSTM: 49 / 51
Residual LSTM beats all three: 39 / 51

Median residual minus Persistence RMSE: -0.709
Median residual minus RF RMSE: -0.936
Median residual minus direct LSTM RMSE: -3.452
```

Best-epoch distribution:

``` text
count 51, mean 6.9, median 5, min 1, max 32
```

Interpretation:

-   Residual learning materially improves the station-specific LSTM.
-   Anchoring the network to Persistence transforms the task from
    learning absolute PM2.5 level to learning one-hour change, whose
    distribution is centered near zero and has much lower variance.
-   The residual LSTM beats Persistence and frozen RF on pooled
    validation metrics and on most station-level RMSE comparisons.
-   This is now the strongest validation-only temporal baseline, but it
    still should not be evaluated on test or tuned further until the next
    research step is chosen.
-   Because residual learning cleared the main baselines, graph design
    remains justified as the next phase; further station-specific LSTM
    tuning is lower priority than adding wind-aware inter-station
    structure.

Generated outputs:

``` text
results/lstm/residual/validation_station_metrics.csv
results/lstm/residual/validation_summary.csv
results/lstm/residual/validation_predictions.csv
results/lstm/residual/validation_matched_predictions.csv
results/lstm/residual/validation_matched_summary.csv
results/lstm/residual/validation_station_win_counts.csv
results/lstm/residual/validation_win_summary.csv
results/lstm/residual/training_history.csv
results/lstm/residual/skipped_stations.csv
results/lstm/residual/target_distribution_by_station.csv
results/lstm/residual/target_distribution_summary.csv
results/lstm/residual/residual_lstm_validation_report.md
```

**Handover status:** Residual LSTM validation is complete. Recommended
next step: move toward graph design and wind-aware station interaction,
using residual LSTM as the sequence baseline to beat.

## 36. Graph design audit before dynamic wind edges

Completed a documentation/audit pass before implementing dynamic wind
edges:

``` text
scripts/22_graph_design_audit.py
scripts/analysis/graph_design_audit.py
docs/graph_design_audit.md
```

No dynamic wind edges, graph snapshots, GNN, or GAT/GNN training were
implemented.

Branch/code review:

``` text
Nirika-work graph scripts 01-04 match main.
Nirika-work graph scripts 05-07 are empty placeholders.
main graph scripts 05-07 are also empty placeholders.
Do not merge Nirika-work as-is.
```

Main identity finding:

``` text
metadata rows: 56
unique human station names: 54
unique PM2.5 sensors: 56
featured datasets: 56
model-usable train+validation datasets: 51
current station_mapping nodes: 54
```

Current `StationMapper` drops duplicate human station names, which
collapses the three Kathmandu University PM2.5 sensors into one graph
node. It also uses raw station names instead of canonical
sensor-qualified dataset names, so it cannot join reliably to featured
files.

Recommended node policy:

-   maintain a canonical 56-row node registry keyed by `dataset_name`;
-   include `node_id`, `dataset_name`, human station name, location id,
    PM2.5 sensor id, latitude, and longitude;
-   use the 51 train+validation model-usable nodes as the first
    supervised graph-model node set;
-   preserve the five non-model-usable featured nodes in the canonical
    registry for reproducibility and later expansion.

Static graph audit:

``` text
K: 5
current mapping nodes: 54
static edge rows: 270
symmetrized adjacency directed edges: 362
symmetrized adjacency undirected pairs: 181
static CSV rows missing reverse directions: 92
```

Recommendation: future dynamic graph candidate edges should be the
directed expansion of the symmetric KNN union. In other words, build the
undirected candidate pair set from the symmetrized KNN adjacency, then
emit both `A -> B` and `B -> A` with their own directed bearings.

Distance/bearing audit:

``` text
distance matrix symmetric: true
distance diagonal zero: true
distance recalculation error: 0.0 km
bearing recalculation error: 0.0 degrees
max reverse-bearing 180-degree error: 0.13 degrees
```

Dynamic wind edge design:

``` text
transport_direction_A(t) = (wind_direction_A(t) + 180) % 360
delta_AB(t) = angular difference between transport_direction_A(t)
              and bearing A -> B
alignment_AB(t) = max(0, cos(delta_AB(t)))
speed_factor_A(t) = wind_speed_A(t) / (wind_speed_A(t) + 5)
distance_factor_AB = exp(-distance_AB / lambda_d)

raw_weight_AB(t) =
    candidate_AB * alignment_AB(t) * speed_factor_A(t) *
    distance_factor_AB
```

Use source-node wind for `A -> B`, because the edge represents possible
transport of pollution leaving source A toward B. Target wind can remain
a node feature or later modifier, but should not control the primary
directed transport edge.

Edge-case policy:

-   near-zero source wind `< 0.5 km/h`: weight 0 with `calm_wind` flag;
-   wind pointing away/perpendicular (`delta >= 90 degrees`): alignment
    0 and weight 0;
-   missing PM2.5: keep graph edge computation separate, use node masks
    and supervised-loss masks;
-   missing weather: keep row with `missing_source_wind` flag and null or
    zero dynamic weight;
-   non-shared usable timestamps: build global hourly graph snapshots and
    use masks rather than silently changing graph shape.

Expected dynamic edge schema is documented in
`docs/graph_design_audit.md`.

Generated audit outputs:

``` text
data/processed/graph/design_audit/recommended_graph_nodes.csv
data/processed/graph/design_audit/identity_summary.csv
data/processed/graph/design_audit/coordinate_summary.csv
data/processed/graph/design_audit/distance_bearing_summary.csv
data/processed/graph/design_audit/static_graph_summary.csv
data/processed/graph/design_audit/graph_design_audit.md
```

**Handover status:** Graph design audit is complete. Before implementing
dynamic edges, correct graph mapping to sensor-qualified node identity
and regenerate distance, bearing, and directed static candidate edges.

## 37. Corrected static graph foundation

Updated and regenerated graph scripts 01-04:

``` text
scripts/graph/01_station_mapping.py
scripts/graph/02_distance_matrix.py
scripts/graph/03_bearing_matrix.py
scripts/graph/04_static_graph.py
```

Dynamic wind weights were not implemented.

Station mapping now preserves canonical sensor-qualified graph identity:

``` text
canonical nodes: 56
unique dataset_name: true
unique pm25_sensor_id: true
model-usable train+validation nodes: 51
missing coordinates: 0
```

Mapping identity:

-   graph identity is `dataset_name` plus retained `pm25_sensor_id`;
-   duplicate human station names are preserved;
-   node ids are deterministic by sorted `dataset_name` and
    `pm25_sensor_id`;
-   human station name, `location_id`, latitude, and longitude are
    retained;
-   `model_usable` identifies the 51 train+validation nodes for the
    first supervised graph model.

Distance/bearing regeneration:

``` text
distance matrix: 56x56
distance symmetric: true
distance diagonal zero: true
complete undirected distance edges: 1,540

bearing matrix: 56x56
bearing directed/non-symmetric: true
complete directed bearing edges: 3,080
```

Distance and bearing edge outputs now preserve joinable source/target
dataset names, PM2.5 sensor ids, and human station names.

Static graph regeneration:

``` text
K: 5
static undirected candidate pairs: 188
static directed candidate edges: 376
adjacency directed edges: 376
static edge rows: 376
adjacency edge set == static CSV edge set: true
candidate pairs missing reverse direction: 0
```

Static graph policy now matches the finalized design: build the
symmetric union of KNN pairs, then emit both `A -> B` and `B -> A` with
directed bearing and distance metadata.

Regenerated ignored artifacts:

``` text
data/metadata/station_mapping.csv
data/metadata/station_mapping.json
data/processed/graph/distance_matrix.csv
data/processed/graph/distance_edges.csv
data/processed/graph/bearing_matrix.csv
data/processed/graph/bearing_edges.csv
data/processed/graph/adjacency_matrix.csv
data/processed/graph/static_graph.csv
data/processed/graph/design_audit/*
```

Validation command:

``` text
python scripts/22_graph_design_audit.py
```

**Handover status:** Static graph foundation is corrected and validated.
Next graph task can implement dynamic wind edge weights on top of this
foundation.

## 38. First dynamic wind-edge stage

Implemented the first actual dynamic wind-edge generator:

``` text
scripts/graph/05_dynamic_edge_weights.py
```

No graph snapshots, sliding windows, GNN, GAT, or model training were
implemented.

Inputs:

``` text
data/processed/graph/static_graph.csv
data/metadata/station_mapping.csv
data/processed/featured/
```

The dynamic stage keeps all 56 canonical nodes and all 376 directed
static candidate edges. It also adds `supervised_edge =
source_model_usable AND target_model_usable` so later graph snapshots can
isolate the 51-node supervised cohort.

Equation:

``` text
transport_direction = (source_wind_direction + 180) % 360
angle_difference = circular difference between transport direction
                   and bearing A->B
alignment = max(0, cos(angle_difference))
speed_factor = wind_speed / (wind_speed + 5)
lambda_d = median static directed candidate distance = 1.930 km
distance_factor = exp(-distance_km / lambda_d)
raw_dynamic_weight = alignment * speed_factor * distance_factor
```

Rules enforced:

-   source-node wind controls `A -> B`;
-   no PM2.5 is used in edge-weight calculation;
-   no future timestamps are used;
-   `wind_speed < 0.5 km/h` gives weight 0 and `calm_wind=True`;
-   missing source wind gives weight 0 and `missing_source_wind=True`;
-   angle `>= 90 degrees` gives alignment 0 and weight 0;
-   weights are raw and not row-normalized.

Generated ignored artifacts:

``` text
data/processed/graph/dynamic_edge_weights.csv
data/processed/graph/dynamic_edge_weights_summary.csv
data/processed/graph/dynamic_edge_weights_validation.csv
data/processed/graph/dynamic_supervised_degree.csv
data/processed/graph/dynamic_reverse_direction_check.csv
```

Dynamic output summary:

``` text
timestamp count: 47,988
total rows: 2,919,724
candidate edges: 376
supervised candidate edges: 326
active-edge percentage: 47.522%
zero-weight percentage: 52.478%
missing-wind percentage: 0.000%
calm-wind percentage: 1.867%
opposite-direction weights can differ: true
all validation checks passed: true
```

Validation checks:

``` text
every generated row corresponds to a static candidate edge: true
no non-candidate edges: true
all static candidates present: true
weights never negative: true
alignment in [0, 1]: true
speed_factor in [0, 1): true
distance_factor in (0, 1]: true
calm wind gives zero weight: true
away/perpendicular wind gives zero weight: true
missing source wind gives zero weight: true
candidate pairs missing reverse direction: 0
uses future rows: false
```

Supervised 51-node subgraph after `supervised_edge=True`:

``` text
supervised nodes: 51
supervised candidate edges: 326
min out-degree: 4
median out-degree: 6
max out-degree: 9
isolated nodes: 0
```

Lowest-degree supervised node:

``` text
Tarakeswor (SC-14)-GD Labs: out-degree 4, in-degree 4, total degree 8
```

This is not problematic enough to change KNN silently.

**Handover status:** Dynamic wind-edge weights are implemented and
validated as raw auditable edge weights. Next step: graph snapshot design
and node/edge masking, not GNN training yet.

## 39. Graph snapshot construction and synchronization analysis

Implemented the first supervised graph snapshot synchronization stage:

``` text
scripts/graph/06_graph_snapshots.py
```

This stage does not train a GNN, row-normalize edge weights, create
sliding windows, or impute missing node values.

Supervised graph policy:

-   use the 51 `model_usable` nodes from `data/metadata/station_mapping.csv`;
-   preserve canonical node IDs from the 56-node registry;
-   keep a fixed 51-node graph frame at every global hourly timestamp;
-   use explicit node input/target masks instead of requiring all nodes
    to be valid.

Node features at timestamp `t`:

``` text
pm2_5
hour_sin
hour_cos
month_sin
month_cos
temperature
humidity
pressure
dew_point
wind_u
wind_v
```

Target:

``` text
residual_pm25(t+1) = pm2_5(t+1) - pm2_5(t)
```

The target is accepted only when `t+1` is exactly one hour after `t` and
does not cross the global chronological train/validation/test split
boundary.

Generated ignored artifacts:

``` text
data/processed/graph/snapshots/supervised_nodes.csv
data/processed/graph/snapshots/snapshot_nodes.csv.gz
data/processed/graph/snapshots/snapshot_edges.csv.gz
data/processed/graph/snapshots/snapshot_timestamp_summary.csv
data/processed/graph/snapshots/snapshot_policy_summary.csv
data/processed/graph/snapshots/snapshot_validation.csv
data/processed/graph/snapshots/snapshot_valid_node_distribution.csv
data/processed/graph/snapshots/snapshot_continuous_runs.csv
```

Artifact size:

``` text
node snapshot rows: 2,447,337
edge snapshot rows: 2,659,101
global hourly timestamps: 47,987
supervised directed static candidates: 326
```

Policy comparison:

``` text
strict policy usable timestamps: 0
strict node-target sequences: 0

masked policy usable timestamps: 30,067
masked train/validation/test timestamps: 17,923 / 4,969 / 7,175
masked node-target sequences: 201,608
```

This decisively rejects the strict all-51-node policy for the first GNN
dataset. It would provide no supervised graph training examples. The
recommended first GNN dataset policy is the masked fixed-graph policy:
retain all 51 canonical nodes, keep raw dynamic edge weights, and apply
node-level input/target masks in the loss and evaluation.

Synchronization distribution:

``` text
valid input nodes per timestamp: min 0, median 1, max 43
valid target nodes per timestamp: min 0, median 1, max 43
valid input+target nodes per timestamp: min 0, median 1, max 42
valid directed edges per timestamp: min 0, median 0, max 234
active dynamic edges per timestamp: min 0, median 0, max 118
```

Coverage thresholds:

``` text
timestamps with 51 valid input nodes: 0
timestamps with >=45 valid input nodes: 0
timestamps with >=40 valid input nodes: 47
timestamps with >=30 valid input nodes: 1,921
```

Longest continuous usable runs:

``` text
masked: 2026-01-04 18:00 to 2026-05-08 13:00, 2,972 hours
masked >=30 input nodes: 2026-01-12 09:00 to 2026-01-23 14:00, 270 hours
masked >=40 input nodes: 2026-05-19 12:00 to 2026-05-20 10:00, 23 hours
```

Validation checks:

``` text
global timestamps hourly: true
target exactly t+1: true
fixed 51-node identity preserved: true
no future node features used: true
every dynamic edge is a supervised static candidate: true
all supervised static candidates present in dynamic edges: true
edge source IDs map to correct nodes: true
edge target IDs map to correct nodes: true
dynamic weights unchanged and non-negative: true
split-boundary crossing target timestamps excluded: 3
```

**Handover status:** Graph snapshots and synchronization masks are
implemented and validated. Next graph step should consume the masked
snapshot artifacts to design graph model batching or sliding temporal
windows. Do not train a GNN until that dataset loader is reviewed.

## 40. Masked 24-hour spatio-temporal graph windows

Implemented compact 24-hour graph window indexing:

``` text
scripts/graph/07_sliding_windows.py
```

This stage consumes outputs from `06_graph_snapshots.py`. It does not
train GAT, GAT-GRU, or any graph model. It also does not impute missing
values, row-normalize edge weights, or use the test split for
model/configuration decisions.

Window definition:

``` text
input graph snapshots: t-23 ... t
window length: 24 consecutive hourly snapshots
prediction target: residual_pm25(t+1) already attached to final timestamp t
```

The window target remains:

``` text
residual_pm25(t+1) = pm2_5(t+1) - pm2_5(t)
```

A node is a supervised target in a window only when:

``` text
sequence_input_valid = input_valid for that node at all 24 input hours
supervised_target_valid = sequence_input_valid AND target_valid at final t
```

This explicitly prevents a node from becoming a supervised graph target
when its own 24-hour input history is incomplete.

Compact artifacts:

``` text
data/processed/graph/snapshots/graph_window_arrays.npz
data/processed/graph/snapshots/graph_window_index.csv
data/processed/graph/snapshots/graph_window_summary.csv
data/processed/graph/snapshots/graph_window_validation.csv
data/processed/graph/snapshots/graph_window_target_distribution.csv
data/processed/graph/snapshots/graph_window_continuous_runs.csv
data/processed/graph/snapshots/graph_window_node_order.csv
data/processed/graph/snapshots/graph_window_edge_order.csv
data/processed/graph/snapshots/graph_window_rejections.csv
```

The `.npz` stores reusable arrays once:

``` text
node_features: (47,987, 51, 11), float32
input_valid_mask: (47,987, 51), bool
target_valid_mask: (47,987, 51), bool
residual_targets: (47,987, 51), float32
edge_weights: (47,987, 326), float32
edge_valid_mask: (47,987, 326), bool
edge_active_mask: (47,987, 326), bool
window_sequence_input_valid_mask: (21,457, 51), bool
window_target_valid_mask: (21,457, 51), bool
```

Overlapping 24-hour feature and edge tensors are not duplicated. Future
dataset code should slice snapshot arrays using `start_idx:end_idx+1`
from `graph_window_index.csv`.

Storage:

``` text
compressed graph_window_arrays.npz: 12.0 MB
uncompressed array memory footprint: about 209.1 MB
graph_window_index.csv: about 2.2 MB
```

Usable windows:

``` text
train: 10,561 windows, 13,238 supervised node-target examples
validation: 4,096 windows, 6,326 supervised node-target examples
test: 6,800 windows, 128,756 supervised node-target examples
all: 21,457 windows, 148,320 supervised node-target examples
```

The test split is indexed only for dataset construction completeness and
must not be used for model or configuration decisions.

Targets per usable window:

``` text
train: min 1, median 1, max 3
validation: min 1, median 1, max 3
test: min 1, median 22, max 39
all: min 1, median 1, max 39
```

Threshold distribution:

``` text
windows with >=1 target: 21,457
windows with >=10 targets: 5,354
windows with >=20 targets: 4,335
windows with >=30 targets: 182
windows with >=40 targets: 0
```

Rejected candidate windows:

``` text
too-short 24h history: 23
non-hourly continuity: 0
split crossing: 49
zero valid supervised targets: 26,458
```

Longest continuous usable runs:

``` text
train: 2023-07-17 18:00 to 2023-08-18 23:00, 774 windows
validation: 2025-07-19 09:00 to 2025-09-14 22:00, 1,382 windows
test: 2026-01-05 17:00 to 2026-04-18 11:00, 2,467 windows
```

Validation checks:

``` text
every accepted window has 24 hourly snapshots: true
no split crossing: true
target exactly t+1 after final input: true
target mask implies complete 24h input history: true
fixed 51-node ordering preserved: true
edge IDs/order consistent across timestamps: true
no future node features used: true
```

**Handover status:** Masked 24-hour graph windows are indexed and
validated. The next graph task can implement a dataset loader/batching
adapter for these artifacts, but should still avoid GNN training until
the loader is inspected.
