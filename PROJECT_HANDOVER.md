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

Air-quality timestamps are normalized and floored to the hour before
merging.

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

## 8. Critical Open Architectural Issue: Lag/Rolling Semantics

This issue is **not fully resolved yet**.

### Current code

Feature engineering runs on `TRIMMED_DIR` before dataset preparation.

It uses row-based operations such as:

``` python
df["lag_6"] = df["pm2_5"].shift(6)
df["lag_24"] = df["pm2_5"].shift(24)
df["rolling_mean_6"] = df["pm2_5"].rolling(6).mean()
```

### Why this requires care

There were two competing ideas in the previous discussion:

1.  **Move dataset preparation before feature engineering.**
2.  **Keep feature engineering before row removal so `.shift()` still
    corresponds to the original timeline.**

Neither should be accepted blindly.

The timestamp validator has already shown that the trimmed datasets
themselves can contain non-hourly jumps. Therefore, even before dataset
preparation removes rows, row-based `.shift(6)` is not guaranteed to
mean six real hours for every station.

At the same time, simply moving feature engineering after dataset
preparation is also unsafe, because dataset preparation deliberately
removes rows, creating additional row-position gaps.

### Correct next investigation

Do **not** solve this by merely swapping stages.

The likely robust solution is to make temporal feature construction
**timestamp-aware**. Possible approaches to evaluate include:

-   reindexing each station onto a complete hourly timeline before
    lag/rolling construction,
-   timestamp-based joins for exact `t-1`, `t-3`, `t-6`, `t-12`, `t-24`,
-   time-based rolling windows with explicit continuity requirements,
-   segmenting the series into continuous hourly blocks and computing
    temporal features within each block.

This must be decided carefully before further model conclusions are
trusted.

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

## 12. Missing-Value Handling

An important recent change exists in `BaseModel.prepare_features()`:

``` python
df = df.dropna(
    subset=MODEL_FEATURE_COLUMNS + ["target_pm2_5"]
)
```

This was added because Ridge regression failed when lag/rolling features
contained NaN values.

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

Treat this `dropna` as a **working model-safety measure**, not
necessarily the final research methodology.

After temporal-feature construction is corrected, revisit whether this
filtering is still needed and quantify how many rows it removes.

------------------------------------------------------------------------

## 13. Baseline Models

### Persistence

Prediction:

``` text
prediction = current pm2_5
```

The persistence baseline uses the shared `BaseModel` evaluation
utilities.

### Linear model

The class is named `LinearRegressionModel`, but the actual estimator in
the current snapshot is:

``` python
Ridge(alpha=10.0)
```

So future documentation should call this a **Ridge regression baseline**
unless the estimator is changed back to ordinary least-squares linear
regression.

### Metrics

All models use:

-   MAE
-   RMSE
-   R²

Earlier experimental results mentioned during development included a
persistence model that substantially outperformed an earlier linear
model. Those values were produced before all later preprocessing fixes,
so they should **not be treated as final results**. Re-run and record
metrics after the temporal-feature issue is resolved.

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

The project stopped immediately after deciding **not to blindly perform
the proposed pipeline swap**.

The next task is **not** Random Forest, XGBoost, GRU, or GNN training.

The next task is:

> Determine and implement the correct timestamp-aware strategy for PM2.5
> lag and rolling features.

We know:

-   the target is now protected against non-one-hour next observations;
-   timestamp validation reports real discontinuities;
-   row-based lag/rolling operations can therefore be temporally
    incorrect;
-   dropping rows before feature engineering would also create
    row-position discontinuities.

This is the highest-priority preprocessing issue.

------------------------------------------------------------------------

## 21. Recommended Next Development Path

### Step 1 --- Quantify the problem

For each station, determine how many engineered lag values correspond to
the exact intended timestamp.

For example, verify whether a `lag_6` value at time `t` came from
exactly `t - 6 hours`.

Do this before changing the implementation.

### Step 2 --- Choose a timestamp-aware temporal-feature strategy

Evaluate the alternatives:

-   hourly reindexing,
-   exact timestamp joins,
-   continuous-segment feature engineering,
-   time-based rolling.

Choose one and document why.

### Step 3 --- Rebuild temporal features

Regenerate `FEATURED_DIR` and downstream datasets from scratch.

### Step 4 --- Re-run dataset preparation and split

Clear stale generated outputs before regeneration.

### Step 5 --- Re-run baselines

Run:

``` text
Persistence
Ridge regression
```

Record per-station and aggregate metrics.

### Step 6 --- Compare against previous behavior

Measure:

-   number of usable stations,
-   number of usable rows,
-   rows removed because temporal history is unavailable,
-   persistence metrics,
-   Ridge metrics.

### Step 7 --- Update documentation

At minimum update:

``` text
docs/architecture.md
docs/preprocessing_plan.md
docs/research_notes.md
docs/changelog.md
```

### Step 8 --- Commit the temporal-feature correction separately

Use a focused commit message such as:

`Make lag and rolling features timestamp-aware`

### Step 9 --- Only then continue classical ML

Suggested next models:

-   Random Forest
-   XGBoost

These should be compared against persistence and Ridge using the same
chronological splits.

### Step 10 --- Then integrate graph work

Review `Nirika-work` and reconcile graph station/time assumptions with
the finalized preprocessing pipeline.

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
Linear baseline estimator: Ridge(alpha=10.0)
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

This list should be revisited after timestamp-aware feature engineering
is finalized.

------------------------------------------------------------------------

## 24. Definition of a Safe Continuation

A future chat can safely continue when it can answer these questions
from the current repository:

1.  Which directory is being used as the source for temporal feature
    engineering?
2.  Are all rows exactly hourly?
3.  If not, how are discontinuities represented?
4.  How is an exact `t-n hours` lag obtained?
5.  How are rolling windows prevented from crossing discontinuities?
6.  How many stations/rows remain after the corrected process?
7.  Are train/validation/test outputs freshly regenerated?
8.  Do persistence and Ridge run without relying on accidental stale
    files?

Until these are resolved, do not interpret advanced-model performance as
trustworthy.

------------------------------------------------------------------------

**Handover status:** Ready for continuation in the ChatGPT Project.

**Immediate next conversation title suggestion:**\
`01 — Fix Timestamp-Aware Lag & Rolling Features`

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

Not yet implemented:

- `DataMerger` has not been switched to `AIR_QUALITY_HOURLY_DIR`.
- The old downstream `.dt.floor("h")` behavior in `scripts/preprocessing/merger.py` has not been removed yet.
- `scripts/04_profile_dataset.py` has not been migrated to canonical hourly AQ coverage yet.
- `scripts/run_pipeline.py` does not yet include the new `02b`, `02c`, or `02d` stages.
- Trimmed, featured, prepared, split, and model outputs have not been regenerated from the canonical hourly AQ layer.
- Baselines and advanced models have not been rerun after this temporal AQ change.

Separate follow-up issue:

- The existing raw `/measurements` helper `fetch_all_measurements()` still uses `limit=1000` without pagination. This was intentionally not fixed in the hourly-layer milestone because the old raw `/measurements` archive remains separate from the new `/hours` implementation.

Expected next step:

``` text
migrate DataMerger and profiling to the validated canonical hourly AQ dataset,
remove downstream timestamp flooring,
regenerate dependent processed outputs,
and rerun temporal validation.
```

**Handover status:** Hourly OpenAQ AQ layer implemented and validated; downstream merger migration is the next task.

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

Regenerated baselines:

``` text
Persistence: 51 datasets, MAE 5.774, RMSE 8.891, R2 0.699
Ridge(alpha=10.0): 51 datasets, MAE 9.446, RMSE 12.075, R2 -127.128
Ridge rows after required-feature dropna: 115,725 train, 26,236 test
```

These are the new valid canonical-hourly baselines. Do not directly
compare them numerically with old metrics from the pre-migration
`datetimeFrom.floor("h")` pipeline.

Still separate follow-up:

- The archival raw `/measurements` downloader still lacks pagination.
  This is technical debt for reproducibility of the raw archive, not a
  blocker for the current canonical-hourly modeling pipeline.

Expected next research step:

``` text
analyze why Ridge underperforms persistence on the canonical hourly data,
quantify feature-row loss from required-feature dropna,
then decide whether to improve linear features/regularization or move to
tree-based baselines.
```

Do not start Random Forest, XGBoost, GRU, GNN, or graph integration
until this baseline interpretation is reviewed.

**Handover status:** Downstream preprocessing has been migrated to
canonical hourly AQ and validated; baseline interpretation is the next
research step.
