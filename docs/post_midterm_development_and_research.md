# Post-Midterm Development and Research Record

## Purpose of this document

This document records the development and research work carried out in the post-midterm phase of the **Wind-Aware Air Quality Forecasting** project, especially the phase where development became more systematic and Codex was used to implement and verify scoped changes.

The goal is different from the ordinary changelog.

- `docs/changelog.md` records what changed.
- `docs/research_notes.md` records individual research observations.
- `PROJECT_HANDOVER.md` records the current state and what should happen next.
- **This document records the full reasoning chain:** what problem we noticed, why it mattered scientifically, how we investigated it, what was implemented, what results were obtained, what conclusions were justified, and how each milestone changed the next research decision.

This is intended to become a major source when writing the final report/thesis methodology, implementation, results, limitations, and discussion chapters.

---

# 1. Starting point of the post-midterm phase

At the beginning of this phase, the project already had a functioning tabular forecasting pipeline:

1. discover Kathmandu Valley monitoring stations,
2. download Open-Meteo weather,
3. download OpenAQ PM2.5 measurements,
4. validate/profile the downloaded data,
5. merge weather and air quality,
6. trim leading PM2.5 missing periods,
7. engineer lag/rolling/time/wind features,
8. create a one-hour-ahead PM2.5 target,
9. split each station chronologically,
10. evaluate persistence and a linear baseline.

The intended research direction was broader than these tabular baselines. The long-term goal was a **wind-aware spatio-temporal model**, eventually using graph relationships between stations. However, before moving to graph or deep-learning models, the main branch had to become scientifically trustworthy.

The main post-midterm question therefore became:

> Are the timestamps, lag features, PM2.5 labels, model splits, and wind variables actually representing the physical and temporal problem we claim they represent?

This led to several important corrections. The biggest lesson from this phase was that getting a model to run is much less important than making sure the data semantics are correct.

---

# 2. Timestamp continuity investigation

## 2.1 Why we investigated timestamps

The feature-engineering code used row-based operations such as:

```python
df["lag_6"] = df["pm2_5"].shift(6)
df["lag_24"] = df["pm2_5"].shift(24)
df["rolling_mean_6"] = df["pm2_5"].rolling(6).mean()
```

These operations are only temporally correct if consecutive rows really represent consecutive hourly timestamps.

For example, `shift(6)` only means "six hours ago" if the six rows between the current row and the source row correspond to six real one-hour transitions.

Because this is a time-series research project, we decided not to assume row distance was equal to elapsed time. A dedicated timestamp validation stage was added.

## 2.2 Initial validator result

The validator checked the trimmed station files and found:

```text
station CSVs checked: 52
total rows: 435,883
adjacent transitions: 435,831
valid one-hour transitions: 431,330
invalid transitions: 4,501
weighted valid percentage: 98.97%
```

At first this looked like a nearly-hourly dataset with a small number of gaps. The important discovery came from inspecting the invalid transitions instead of only looking at the percentage.

The invalid transitions were not long forward gaps. They were mostly **0-hour transitions caused by duplicate timestamps**.

Only two stations were responsible for the problem:

- Embassy Kathmandu
- Phora Durbar Kathman

The rest of the stations were effectively hourly.

## 2.3 Temporal-feature correctness before the fix

A separate validation reproduced the row-based lag and rolling operations and checked whether they pointed to the intended real timestamp.

Across all stations:

```text
lag_1: 98.97% timestamp-correct
lag_3: 97.91%
lag_6: 97.89%
lag_12: 97.85%
lag_24: 97.77%
rolling_3: 97.93%
rolling_6: 97.90%
rolling_24: 97.77%
```

The two problematic stations had much worse correctness. For example, Embassy Kathmandu had roughly 89% correctness for some lag/rolling features.

This was important because it showed that a model could train successfully while some engineered values were associated with the wrong physical time.

---

# 3. Root cause: PM2.5 timestamp flooring created duplicate hours

## 3.1 What the merger was doing

The weather data was already hourly and unique. The air-quality merge code parsed OpenAQ timestamps and then did:

```python
air_df["timestamp"] = air_df["timestamp"].dt.floor("h")
```

The code then performed a weather-left merge.

This looked reasonable if the AQ timestamps were arbitrary measurement moments. It turned out that assumption was wrong.

## 3.2 Embassy and Phora evidence

The raw PM2.5 files contained unique timestamps, but some records occurred at both `:00` and `:45` local time.

For Embassy Kathmandu:

```text
raw rows: 41,915
raw duplicate timestamps: 0
duplicate rows after flooring: 5,202
affected hours: 2,601
```

For Phora Durbar Kathman:

```text
raw rows: 35,815
raw duplicate timestamps: 0
duplicate rows after flooring: 3,800
affected hours: 1,900
```

Example:

```text
06:00 -> floor -> 06:00
06:45 -> floor -> 06:00
```

A weather row at `06:00` therefore matched two PM2.5 rows and became duplicated during the merge.

This explained why the problem appeared in feature engineering even though feature engineering itself was not the root cause.

## 3.3 Key research lesson

The initial mistake was treating OpenAQ timestamps as if they were simple instantaneous observations.

The real question had to become:

> What time interval does each OpenAQ value represent, and what timestamp should represent that interval in our forecasting dataset?

---

# 4. OpenAQ `/hours` investigation and the canonical PM2.5 definition

## 4.1 Why `/hours` was investigated

The original downloader used OpenAQ `/measurements` and saved only:

```text
timestamp
pm2_5
```

It discarded interval metadata such as `datetimeTo`.

The OpenAQ `/hours` endpoint was therefore investigated because it provides precomputed hourly aggregation together with interval information.

## 4.2 Important `/hours` fields

The API exposed fields such as:

```text
value
period.label
period.interval
period.datetimeFrom
period.datetimeTo
coverage.expectedCount
coverage.observedCount
```

The critical finding was that `/hours.value` corresponds to an hourly interval and should not be reduced to a floored start timestamp.

## 4.3 Raw measurement membership

Live checks on Embassy and Phora showed that hourly values matched the arithmetic mean of raw measurements whose interval end fell into the OpenAQ hourly period.

This demonstrated that the saved `datetimeFrom` alone was not sufficient to represent the hourly target correctly.

## 4.4 Mixed interval alignments

Some sensors exposed overlapping hourly alignments.

For Nepal local time, examples included intervals such as:

```text
05:45 -> 06:45
06:00 -> 07:00
```

This explained why the problematic sensors contained both `:00` and `:45` patterns.

An all-sensor audit was then performed.

Result for 56 PM2.5 sensors:

```text
52 sensors: only datetimeTo.minute == 00
4 sensors: mixed alignments including clock-hour endings
0 sensors: only :45 endings
0 sensors: unusable hourly endpoint
```

The four mixed sensors included Embassy, Phora, and two AirGradient sensors.

## 4.5 Canonical modeling convention

The dataset adopted the following explicit convention:

> `timestamp t` represents the PM2.5 hourly interval **ending at local time t**.

Example:

```text
OpenAQ PM2.5 interval: 06:00 -> 07:00
canonical timestamp: 07:00
```

The target remains:

```text
PM2.5(t + 1 hour)
```

Therefore the model row at `t` represents information available for the current hour and predicts the PM2.5 interval ending one hour later.

The canonical selection keeps `/hours` rows whose local `datetimeTo` ends exactly on a clock hour.

No flooring, rounding, or ceiling is used.

---

# 5. Canonical hourly air-quality layer

## 5.1 New architecture

A separate hourly AQ layer was introduced instead of modifying the original raw archive.

```text
OpenAQ /measurements
    -> existing raw archive

OpenAQ /hours
    -> data/raw/air_quality_hourly/
    -> data/processed/air_quality_hourly/
```

The raw hourly directory preserves all returned interval alignments for auditability.

The processed hourly directory contains the canonical clock-ending intervals used by modeling.

## 5.2 Important implementation rules

The canonical preparer:

- sets `timestamp = datetime_to_local`,
- keeps only local clock-hour interval endings,
- preserves original local/UTC interval metadata,
- does not silently deduplicate unexpected canonical timestamps,
- raises an error if duplicates remain,
- sorts chronologically,
- does not synthesize missing PM2.5 hours.

## 5.3 Pagination

The new `/hours` downloader supports pagination.

During this work, another reproducibility issue was identified: the older raw `/measurements` downloader still uses `limit=1000` without full pagination. This remains technical debt for the raw archive, but it is no longer a blocker for the modeling pipeline because modeling now uses the paginated `/hours` source.

## 5.4 Full canonical-hourly result

```text
metadata PM2.5 sensors: 56
raw hourly station/sensor directories: 56
raw hourly files: 114
raw hourly rows: 323,665
canonical hourly files: 56
canonical hourly rows: 251,964
zero-row canonical stations: 0
duplicate canonical timestamps: 0
non-clock-hour canonical rows: 0
invalid one-hour intervals: 0
unsorted stations: 0
missing PM2.5 rows in canonical layer: 23,837
```

Embassy and Phora both had zero duplicate timestamps after canonicalization.

---

# 6. Duplicate station identity problem

During the canonical migration, station identity also had to be made safe.

The metadata contains three PM2.5 sensors under the same human-readable station name:

```text
Kathmandu University
sensor 15286458
sensor 15286980
sensor 15286975
```

If output files were named only from the human-readable station name, one sensor could overwrite another.

The pipeline therefore introduced sensor-qualified dataset names where needed:

```text
Kathmandu University__sensor_15286458.csv
```

The datasets may reuse the same weather station data, but their PM2.5 identities remain separate.

This distinction later became important in profiling, merging, modeling, and future graph-node reconciliation.

---

# 7. Migrating the downstream pipeline to canonical hourly AQ

## 7.1 Profiling semantics changed

The old profile compared raw measurement counts with weather rows. That was scientifically misleading because multiple raw measurements could correspond to the same hour.

The new `coverage_percent` means:

```text
valid canonical PM2.5 hours that match weather timestamps
---------------------------------------------------------
              unique available weather hours
```

This is now a model-hour coverage measure rather than a raw measurement-frequency measure.

## 7.2 Merge semantics changed

The merger now uses:

```text
weather timestamp t
+
PM2.5 hourly interval ending at t
=
one model row at t
```

The weather timeline remains the left side of the merge. Therefore an hour with weather but no PM2.5 remains present with `pm2_5 = NaN`.

The old AQ `.floor("h")` behavior was removed completely.

## 7.3 Merge integrity result

```text
datasets merged: 56
weather rows: 2,712,192
merged rows: 2,712,192
duplicate merged timestamps: 0
valid merged PM2.5 rows: 208,982
missing merged PM2.5 rows: 2,503,210
```

Every dataset preserved weather-left cardinality exactly.

This was a major acceptance criterion because the old bug had multiplied weather rows.

---

# 8. Timestamp and temporal-feature problem was resolved upstream

After rebuilding merged and trimmed data from canonical hourly PM2.5, timestamp validation produced:

```text
datasets checked: 56
trimmed rows: 445,562
valid hourly transitions: 445,506
invalid transitions: 0
duplicate timestamps: 0
largest gap: 1 hour
```

The temporal-feature validator then showed:

```text
lag_1: 100.00%
lag_3: 100.00%
lag_6: 100.00%
lag_12: 100.00%
lag_24: 100.00%
rolling_3: 100.00%
rolling_6: 100.00%
rolling_24: 100.00%
```

This changed an important architectural decision.

Before the canonical fix, we were considering timestamp-aware joins, hourly reindexing, or segment-based rolling windows.

After the fix, those changes were no longer necessary for timestamp correctness because the upstream timeline itself became hourly.

Therefore the project deliberately **did not over-engineer FeatureEngineer**. The current row-based lag/rolling implementation was retained because validation showed it is now temporally correct.

---

# 9. Missing PM2.5 remained a separate problem

Fixing timestamp continuity did not make PM2.5 available at every hour.

After the canonical rebuild:

```text
merged missing PM2.5 rows: 2,503,210
trimmed missing PM2.5 rows: 236,580
prepared missing PM2.5 rows: 0
```

This distinction became central:

- **timestamp continuity** asks whether the hourly row exists,
- **measurement availability** asks whether PM2.5 exists at that hour.

The final timeline can be perfectly hourly while lag features are still `NaN` if the PM2.5 measurement at the required historical timestamp is missing.

No PM2.5 imputation was introduced during this phase.

---

# 10. Dataset preparation and chronological split after canonical migration

Fresh downstream outputs produced:

```text
prepared datasets: 53
prepared rows: 201,658
eligible split datasets: 51
train rows: 141,120
validation rows: 30,247
test rows: 30,270
no prepared rows: 3 datasets
prepared but <100 rows: 2 datasets
```

Splitting remained chronological:

```text
70% train
15% validation
15% test
```

No random shuffle was introduced.

The one-hour target-generation guard remained in place: a target is only valid when the next timestamp is exactly one hour later.

---

# 11. Fair baseline evaluation framework

## 11.1 Why the old comparison was unfair

Persistence originally evaluated all prepared test rows, while Ridge first dropped rows with missing model features.

This meant:

```text
Persistence test rows: 30,270
Ridge test rows: 26,236
```

Therefore their MAE/RMSE values were not directly comparable.

A second reporting problem was that the old "overall R²" was simply:

```text
mean(per-station R²)
```

This is a macro average, not a pooled/global R².

## 11.2 Row-loss audit

The full model feature mask removed:

```text
Train:      25,395 / 141,120 = 18.00%
Validation:  4,558 / 30,247  = 15.07%
Test:        4,034 / 30,270  = 13.33%
```

The missing features were entirely PM2.5-history-derived:

```text
lag_24
rolling_mean_6
rolling_std_6
lag_6
```

Weather and time features had no missing values in the evaluated split rows.

## 11.3 Shared benchmark frame

The project introduced a shared evaluation frame requiring:

```text
MODEL_FEATURE_COLUMNS all non-null
+
target_pm2_5 non-null
```

Persistence, Ridge, Random Forest, and XGBoost now use the same benchmark rows.

The framework records:

- original rows,
- evaluated rows,
- evaluation coverage,
- per-station MAE/RMSE/R²,
- macro mean metrics,
- macro median R²,
- pooled metrics from concatenated predictions,
- prediction rows with station/timestamp/source index.

This became the standard evaluation contract for later models.

---

# 12. Why the original Ridge R² looked catastrophically bad

The first canonical Ridge run reported a macro mean R² around `-127`.

The audit showed this was dominated by one station rather than universal failure.

Sundarighat had:

```text
test target mean: 0.066
test target std: 0.117
Ridge station R²: approximately -6500
```

Its training distribution was dramatically different:

```text
train target mean/std: 83.43 / 42.09
test target mean/std: 0.066 / 0.117
```

Because R² divides error by target variance, very small test variance can produce an enormous negative R² even when the model is not equally catastrophic across all stations.

Evidence:

```text
macro Ridge R² with Sundarighat: about -127
without Sundarighat: about 0.341
pooled Ridge R² on matched rows: about 0.751
```

This led to the decision to report both macro and pooled metrics and to include macro median R².

Low-R² stations were not deleted simply to improve headline numbers.

---

# 13. Persistence strength

On matched rows, current PM2.5 was highly correlated with next-hour PM2.5:

```text
macro correlation: about 0.845
median correlation: about 0.881
pooled correlation: about 0.910
```

This explains why persistence is such a difficult baseline to beat at a one-hour horizon.

The persistence result is not a weak benchmark. It represents a strong physical/time-series property of PM2.5: adjacent hours are highly autocorrelated.

This changed how later model results were interpreted. A complex model failing to beat persistence is scientifically meaningful rather than automatically considered an implementation failure.

---

# 14. Validation-selected Ridge baseline

The original Ridge used `alpha=10` without validation selection.

A validation-only study compared:

- ordinary linear regression,
- unscaled Ridge across a compact alpha grid,
- standardized Ridge across the same alpha grid.

The primary selection criterion was **pooled validation RMSE**.

Important result:

```text
Persistence validation pooled RMSE: 12.300
best selected Ridge pooled RMSE:     13.225
```

The selected linear configuration was:

```text
Ridge(alpha=1000.0)
no scaler
```

The fixed test result after freezing the configuration was:

```text
rows: 26,236
macro MAE: 7.437
macro RMSE: 10.076
macro median R²: 0.702
pooled MAE: 7.629
pooled RMSE: 12.591
pooled R²: 0.805
```

Persistence still won.

This established the linear-model conclusion:

> A tuned linear combination of current/history/weather/time variables did not improve the one-hour forecast over simple persistence.

---

# 15. Random Forest baseline

## 15.1 Why Random Forest was the next experiment

After Ridge, the next hypothesis was:

> Perhaps the same input features contain useful nonlinear interactions that a linear model cannot represent.

Random Forest was selected as a nonlinear classical baseline while holding the data, features, split, and evaluation frame constant.

## 15.2 Validation selection

A compact grid explored structural complexity using:

```text
max_depth
min_samples_leaf
max_features
```

`n_estimators` was initially planned at a larger value, but 300 and 200 trees caused memory problems in the available environment. A stable value of 100 trees was used.

Selected configuration:

```text
n_estimators = 100
max_depth = 10
min_samples_leaf = 10
max_features = 1.0
random_state = 42
n_jobs = 1
```

## 15.3 Result before the wind-component correction

Random Forest became the first model to beat Persistence on pooled held-out RMSE:

```text
Persistence pooled RMSE: 12.083
RF pooled RMSE:          11.652
improvement: about 3.56%
```

However Persistence still had better pooled MAE, and most stations still favored Persistence.

This was the first clear evidence that nonlinear interactions could reduce larger squared errors even though they did not improve typical absolute errors everywhere.

---

# 16. XGBoost baseline

## 16.1 Motivation

XGBoost tested whether sequential gradient boosting could extract more useful nonlinear signal than Random Forest from the same feature set.

The project added `xgboost>=3.4.0` and used XGBoost 3.4.0 under Python 3.12.0.

## 16.2 Validation and early stopping

The model used validation-only configuration selection and station-specific early stopping.

Selected global configuration:

```text
learning_rate = 0.1
max_depth = 3
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.0
reg_lambda = 1.0
n_estimators upper bound = 1000
early_stopping_rounds = 50
objective = reg:squarederror
eval_metric = rmse
tree_method = hist
random_state = 42
n_jobs = 1
```

Before the later wind correction, XGBoost only barely beat Persistence on pooled test RMSE and remained worse than Random Forest overall.

The earlier report accidentally described the negative MAE percentage sign as an "improvement". This wording was later corrected: XGBoost's MAE was worse than Persistence.

---

# 17. Wind-vector semantics audit

## 17.1 Why this audit became necessary

The project is explicitly wind-aware, yet the feature code originally used:

```python
wind_u = wind_speed * cos(direction)
wind_v = wind_speed * sin(direction)
```

These equations are ordinary mathematical polar coordinates. They do not represent meteorological eastward/northward components when direction is the direction **from which** wind blows.

The issue was investigated before performing feature ablation or graph integration.

## 17.2 Correct meteorological convention

Meteorological direction is clockwise from north and indicates where wind comes FROM.

Correct physical components are:

```text
u = -speed * sin(phi)
v = -speed * cos(phi)
```

where:

```text
u = eastward component
v = northward component
```

The relationship to the old variables is:

```text
u_physical = -v_old
v_physical = -u_old
```

Therefore the old representation preserved the same directional information but was physically mislabeled.

## 17.3 Why this mattered

For tabular models, the old variables were still an invertible circular encoding, so the previous experiments were not meaningless.

For future directed wind-aware graph edges, however, physical direction is essential.

If a station bearing A→B is compared directly with meteorological FROM direction, the graph can favor the upwind direction instead of actual transport.

The future transport direction must be:

```text
transport_direction = (wind_direction + 180) % 360
```

before comparing it with source→target bearing.

---

# 18. Physical wind-component correction

Feature engineering was corrected to:

```python
wind_u = -wind_speed * sin(wind_direction)
wind_v = -wind_speed * cos(wind_direction)
```

Wind speed remained in Open-Meteo's existing `km/h` unit. Raw weather was not redownloaded.

## 18.1 Cardinal sanity tests

For speed = 10:

```text
0°   -> u ≈ 0,   v = -10
90°  -> u = -10, v ≈ 0
180° -> u ≈ 0,   v = +10
270° -> u = +10, v ≈ 0
```

## 18.2 Magnitude validation

Across all regenerated featured data:

```text
files checked: 56
rows checked: 445,562
max |sqrt(u² + v²) - speed|:
3.552713678800501e-15
```

This confirmed numerical correctness.

## 18.3 Regeneration boundary

Only outputs downstream of FeatureEngineer were rebuilt:

```text
featured
prepared
split
feature_analysis
persistence results
Ridge results
Random Forest results
XGBoost results
```

Raw weather, raw AQ, canonical AQ, merged data, and trimmed data were unchanged.

All dataset/split counts remained exactly the same.

## 18.4 Frozen models were not retuned

To avoid using known test performance to guide another hyperparameter search:

- Ridge remained `alpha=1000`,
- Random Forest remained at its selected 100-tree configuration,
- XGBoost hyperparameters remained frozen.

This allowed the effect of the physical feature correction itself to be measured.

---

# 19. Final corrected classical-model benchmark

After the physical wind correction, the synchronized held-out test benchmark was:

| Model | Rows | Macro MAE | Macro RMSE | Macro Mean R² | Macro Median R² | Pooled MAE | Pooled RMSE | Pooled R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 26,236 | 5.830 | 8.815 | 0.692 | 0.763 | 6.005 | 12.083 | 0.820 |
| Ridge | 26,236 | 7.437 | 10.076 | -29.748 | 0.702 | 7.629 | 12.591 | 0.805 |
| Random Forest | 26,236 | 6.735 | 9.695 | -78.855 | 0.704 | 6.556 | 11.653 | 0.833 |
| XGBoost | 26,236 | 7.151 | 10.061 | -208.030 | 0.681 | 6.990 | 11.813 | 0.828 |

Interpretation:

- Best pooled RMSE: **Random Forest**.
- Best pooled MAE: **Persistence**.
- Best station-level RMSE consistency: **Persistence**.
- Random Forest remained the strongest completed learned model on pooled RMSE.
- XGBoost improved after the physical wind correction but still did not surpass Random Forest overall.

The fact that different metrics favor different models became an important discussion point. A model can reduce large squared errors while still producing worse average absolute error across observations.

---

# 20. Validation-only feature ablation

## 20.1 Why ablation was performed

After the classical baselines, the key thesis question became more important than simply trying another model:

> Do wind and weather actually add predictive information beyond a station's own PM2.5 history and temporal cycles?

Random Forest was used because it was the strongest completed nonlinear model by pooled held-out RMSE.

No test data was used during ablation.

## 20.2 Critical fixed-row design

All feature variants used the same full-feature-valid rows:

```text
datasets: 51
train rows: 115,725
validation rows: 25,689
row/target mismatches: 0
```

This avoided a major confound: smaller feature subsets were not allowed to gain extra rows simply because they required fewer non-null columns.

## 20.3 Feature groups

The study used:

```text
A0 Persistence
A1 Current PM only
A2 PM history
A3 PM history + cyclical time
A4 A3 + non-wind weather
A5 A3 + wind
A6 Full current feature set
```

Current PM/history group:

```text
pm2_5
lag_6
lag_24
rolling_mean_6
rolling_std_6
```

Time:

```text
hour_sin
hour_cos
month_sin
month_cos
```

Wind:

```text
wind_u
wind_v
```

Non-wind weather:

```text
temperature
humidity
pressure
dew_point
```

## 20.4 Validation results

| Variant | Pooled RMSE | Pooled MAE | Pooled R² | Macro RMSE | Median R² |
|---|---:|---:|---:|---:|---:|
| Persistence | 12.300 | 7.292 | 0.889 | 11.536 | 0.769 |
| A1 Current PM RF | 13.453 | 8.311 | 0.867 | 12.787 | 0.768 |
| A2 PM history RF | 13.407 | 8.303 | 0.868 | 12.774 | 0.766 |
| A3 History + time | 12.499 | 7.586 | 0.885 | 11.965 | 0.791 |
| A4 + non-wind weather | 12.497 | 7.402 | 0.885 | 11.950 | 0.788 |
| A5 + wind | 12.426 | 7.498 | 0.886 | 11.912 | 0.803 |
| A6 Full | 12.449 | 7.369 | 0.886 | 11.900 | 0.800 |

## 20.5 Incremental findings

### PM history beyond current PM

```text
13.453 -> 13.407 pooled RMSE
improvement: 0.046
```

The additional fixed PM history features provided only a small improvement over a Random Forest using current PM2.5 alone.

Persistence itself remained stronger than either A1 or A2.

### Cyclical time

```text
13.407 -> 12.499 pooled RMSE
improvement: 0.908 (~6.78%)
```

This was the clearest ablation gain.

It provided strong evidence that time-of-day and seasonal structure matter in the forecasting problem.

### Non-wind weather

```text
12.499 -> 12.497 pooled RMSE
improvement: 0.002
```

This was essentially neutral for pooled RMSE, although pooled MAE improved more noticeably.

### Wind without other weather

```text
12.499 -> 12.426 pooled RMSE
improvement: 0.073
```

Wind added a small improvement when added to PM history and time.

### Conditional wind contribution

The main wind-specific comparison was:

```text
A4: history + time + non-wind weather
A6: same + wind
```

Result:

```text
A4 pooled RMSE: 12.497
A6 pooled RMSE: 12.449
improvement: 0.048 (~0.38%)

A4 pooled MAE: 7.402
A6 pooled MAE: 7.369

A4 pooled R²: 0.885
A6 pooled R²: 0.886
```

This is evidence that wind provides some additional predictive information, but the effect is small.

## 20.6 Station-level wind heterogeneity

```text
wind improved RMSE: 24 / 51 stations
wind worsened RMSE: 27 / 51 stations
ties: 0
median station effect: slightly negative (-0.011 RMSE)
```

Therefore the pooled wind improvement was not a broad station-wide effect.

Largest validation wind improvements included:

```text
Farsidol Relocated
Sitapaila
Tarakeswor SC-14
Sanepa
Gaushala Chowk
Chovar
Baluwatar
Tokha SC-32
Gokarneshwor
Mid Baneshwor
```

Largest degradations included:

```text
Tankeshwor
Balkumari
Teku Ward 12
Ramkot
Sundarighat
Tyanglaphat
Ranibari
Imadol
Kaushaltar
Jadibuti
```

No causal explanation was assigned to these station differences because the project currently does not contain emissions-source evidence sufficient to justify such claims.

---

# 21. Main scientific conclusions from the post-midterm phase

## 21.1 Data semantics mattered more than adding models

The most important improvement was not Random Forest or XGBoost. It was fixing PM2.5 interval semantics and the merge.

Without that work, some lag and rolling features were attached to incorrect timestamps.

## 21.2 The current timeline is now defensible

After the canonical hourly migration:

- merged timestamps are unique,
- trimmed transitions are hourly,
- lags and rolling windows are timestamp-correct,
- one-hour targets are guarded by actual timestamp difference.

## 21.3 Persistence is a serious benchmark

The one-hour problem has strong autocorrelation. Any future LSTM, Transformer, graph, or GAT-GRU model must be compared against Persistence, not only against weaker learned models.

## 21.4 Linear models did not add enough

Validation-selected Ridge remained below Persistence.

This justified nonlinear baselines.

## 21.5 Nonlinearity helped, but not uniformly

Random Forest achieved the best pooled test RMSE among completed models, but Persistence remained strongest in MAE and across most individual stations.

Therefore "best model" depends on what aspect of error is considered.

## 21.6 Time is currently the strongest extra feature group

The feature-ablation study found a much larger validation gain from cyclical time features than from wind or ordinary meteorology.

This supports investigating richer temporal modeling.

## 21.7 Wind contains some signal, but current tabular use is weak and heterogeneous

Adding physical wind components slightly improved pooled validation metrics after other variables were present, but more stations worsened than improved.

This does not justify saying wind is useless.

A likely reason is that a station-specific tabular model only sees **local wind**, while the actual research hypothesis is about **transport of pollution between locations**.

That spatial interaction is not represented by `wind_u` and `wind_v` alone.

This is an important argument for the eventual graph model.

---

# 22. Why we did not immediately remove weak-looking features

The ablation did not change `MODEL_FEATURE_COLUMNS`.

The project deliberately avoided turning validation analysis into an automatic feature-selection rule because:

1. the test period has already been observed in previous milestones,
2. feature effects were heterogeneous by station,
3. the long-term graph model may use wind differently from a local tabular model,
4. a tiny average gain/loss does not prove a variable is physically irrelevant.

The ablation is currently interpreted as **research evidence**, not as permission to optimize the production feature list against previously seen test outcomes.

---

# 23. Methodological issue discovered late in the phase: repeated use of one fixed test period

The fixed 15% test period was originally treated correctly as a held-out evaluation set for individual model milestones.

However, over the whole post-midterm project phase, its results have now been inspected repeatedly while deciding what research direction to explore next.

This does not erase the historical results, but it means future development should not treat that same period as a completely untouched final decision set.

The planned response is to introduce **rolling-origin / expanding-window validation** over the development period.

This will allow the project to answer whether conclusions are stable across multiple chronological forecast periods rather than depending on one specific validation/test window.

---

# 24. Rolling-origin / expanding-window validation

## 24.1 Why rolling-origin validation was added

The fixed 15% test period had already been inspected during multiple
classical-model milestones. To avoid treating that same period as a
fresh decision set, rolling-origin validation was added over only the
first 85% development portion of each prepared station dataset.

This answered a different question:

> Are the Persistence, Ridge, and Random Forest conclusions stable
> across multiple chronological forecast windows?

The existing final 15% test split was not loaded or evaluated.

## 24.2 Fold design

Each prepared station dataset was split into three expanding windows:

```text
Fold 1: train 0-55%, validate 55-65%
Fold 2: train 0-65%, validate 65-75%
Fold 3: train 0-75%, validate 75-85%
```

The same full-`MODEL_FEATURE_COLUMNS`-valid rows were used for
Persistence, Ridge, and Random Forest within each fold.

Models were frozen:

```text
Persistence
Ridge(alpha=1000)
Random Forest:
    n_estimators=100
    max_depth=10
    min_samples_leaf=10
    max_features=1.0
    random_state=42
    n_jobs=1
```

No features or hyperparameters were changed.

## 24.3 Fold-level results

Each fold evaluated 51 datasets. The same tiny prepared datasets that
were not useful for modeling were skipped because they had insufficient
full-feature-valid rows.

| Fold | Model | Validation Rows | Macro RMSE | Median R² | Pooled MAE | Pooled RMSE | Pooled R² |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Persistence | 17,127 | 14.681 | 0.747 | 10.011 | 15.611 | 0.829 |
| 1 | Ridge | 17,127 | 14.469 | 0.773 | 10.540 | 15.541 | 0.831 |
| 1 | Random Forest | 17,127 | 14.270 | 0.773 | 9.490 | 14.538 | 0.852 |
| 2 | Persistence | 17,803 | 13.314 | 0.742 | 9.136 | 17.689 | 0.777 |
| 2 | Ridge | 17,803 | 13.246 | 0.754 | 9.959 | 16.503 | 0.806 |
| 2 | Random Forest | 17,803 | 12.747 | 0.786 | 8.568 | 15.107 | 0.837 |
| 3 | Persistence | 16,764 | 11.014 | 0.735 | 6.951 | 11.862 | 0.896 |
| 3 | Ridge | 16,764 | 12.035 | 0.747 | 8.218 | 12.978 | 0.876 |
| 3 | Random Forest | 16,764 | 11.462 | 0.725 | 7.175 | 12.465 | 0.885 |

Random Forest improved pooled RMSE over Persistence in folds 1 and 2:

```text
Fold 1: +1.073 RMSE improvement, 31 station wins
Fold 2: +2.582 RMSE improvement, 34 station wins
Fold 3: -0.603 RMSE degradation, 27 station wins
```

Ridge was less robust:

```text
Fold 1: +0.071 RMSE improvement, 31 station wins
Fold 2: +1.185 RMSE improvement, 24 station wins
Fold 3: -1.116 RMSE degradation, 12 station wins
```

## 24.4 Station-level Random Forest robustness

Random Forest beat Persistence by station-level RMSE:

```text
3/3 folds: 10 stations
2/3 folds: 24 stations
1/3 folds: 14 stations
0/3 folds: 3 stations
```

This is more nuanced than the single held-out test result. Random
Forest often helps, but its advantage is not universal across stations
or forecast periods.

## 24.5 Distribution shifts

Large train-to-validation target shifts were observed. Examples:

```text
Phora Durbar Kathman fold 3: +60.5 target-mean shift
Nakhipot fold 1: +57.2
Lamtangil fold 2: -51.7
Jadibuti fold 3: -51.0
Sifal fold 1: -48.7
```

Sundarighat remained especially useful as a diagnostic station:

```text
Fold 1: validation mean 96.0, RF beat Persistence
Fold 2: validation mean 61.6, RF beat Persistence
Fold 3: validation mean 91.1 with higher validation variance,
        Persistence beat RF and Ridge
```

The result strengthened the methodological conclusion: model comparison
must consider temporal robustness and distribution shift, not only a
single chronological window.

---

# 25. Current research story for the final report

A defensible narrative for the project so far is:

```text
Raw AQ + weather pipeline
        ↓
Timestamp audit exposed duplicated hours
        ↓
Root cause traced to flooring OpenAQ interval starts
        ↓
OpenAQ /hours investigated
        ↓
Canonical definition adopted: PM2.5 interval ending at t
        ↓
Canonical hourly AQ layer implemented
        ↓
Merger/profiling migrated
        ↓
Timestamp and lag/rolling correctness reached 100%
        ↓
Fair matched-row model evaluation implemented
        ↓
Persistence shown to be a very strong one-hour baseline
        ↓
Ridge tested linear relationships and lost to Persistence
        ↓
Random Forest tested nonlinear interactions and improved pooled RMSE
        ↓
XGBoost did not surpass Random Forest
        ↓
Wind semantics audited and corrected physically
        ↓
Feature ablation showed strongest extra signal comes from time features
        ↓
Wind provides a small, heterogeneous local-tabular contribution
        ↓
Rolling-origin validation showed RF helps in some windows but not all
        ↓
Sequence dataset validation established featured data as the safe LSTM source
        ↓
First station-specific LSTM baseline trained on validation only and lost to Persistence/RF
        ↓
Persistence-anchored residual LSTM beat Persistence/RF on validation
        ↓
Graph design audit finalized node identity and dynamic edge rules
        ↓
Static graph foundation corrected and regenerated
        ↓
Raw dynamic wind-edge weights implemented and validated
        ↓
Next: graph snapshots and node/edge masks
```

This sequence is much stronger than simply saying "we tried several machine-learning models."

Each major model or preprocessing step was motivated by evidence from the previous one.

---

# 26. Important current limitations

## 25.1 Raw `/measurements` pagination

The archival raw OpenAQ `/measurements` helper still lacks complete pagination. The modeling source uses paginated `/hours`, so this is not currently a modeling blocker, but it should be fixed for full raw-archive reproducibility.

## 25.2 Missing PM2.5 history

Model-valid rows are reduced because lag/rolling PM2.5 values become missing when historical PM2.5 measurements are absent.

No imputation strategy has yet been adopted.

## 25.3 Station-specific models do not represent spatial pollution transport

Ridge, RF, and XGBoost are currently trained separately per dataset/station.

They cannot learn that pollution at an upwind station may affect a downwind station later.

## 25.4 Local wind variables are not the final wind-aware mechanism

The small wind-ablation gain should not be confused with the full research hypothesis. The eventual graph should represent source→target geography together with wind transport direction and strength.

## 25.5 Fixed test has been observed repeatedly

Future model-development conclusions should rely more heavily on rolling-origin validation or another deliberately designed chronological evaluation framework.

## 25.6 Sequence models must not use prepared rows by position

Prepared datasets remove rows with missing current or target PM2.5.
That makes adjacent prepared rows unsuitable for sequence windows because
they are not guaranteed to be adjacent hours. Future LSTM or other
sequence datasets should construct windows from the hourly-continuous
featured stage and reject windows that contain missing values, timestamp
gaps, or split-boundary crossings.

---

# 27. Current state at the end of this record

Implemented and validated:

- canonical OpenAQ hourly AQ layer,
- correct interval-end PM2.5 timestamps,
- safe duplicate station/sensor identity,
- canonical profiling and weather-left merging,
- timestamp validator,
- temporal-feature validator,
- one-hour target guard,
- chronological split,
- fair shared model evaluation,
- Persistence baseline,
- validation-selected Ridge,
- validation-selected Random Forest,
- validation-selected XGBoost with early stopping,
- physically correct meteorological wind components,
- wind-component validator,
- validation-only feature-ablation study,
- rolling-origin / expanding-window validation,
- validation-only LSTM sequence dataset design,
- first station-specific LSTM baseline,
- persistence-anchored residual LSTM baseline,
- graph design audit before dynamic wind edges,
- corrected static graph foundation,
- raw dynamic wind-edge weights.

Current strongest findings:

```text
Best pooled held-out RMSE among completed models:
Random Forest ≈ 11.653

Best pooled held-out MAE:
Persistence ≈ 6.005

Most useful added feature group in validation ablation:
cyclical time features

Conditional wind pooled validation RMSE improvement:
approximately 0.048 (about 0.38%)

Wind station effect:
24 improve / 27 worsen

Rolling-origin result:
Random Forest beats Persistence by pooled RMSE in folds 1 and 2, but
Persistence wins fold 3. RF wins 3/3 folds on 10 stations and at least
2/3 folds on 34 stations.

LSTM sequence dataset validation:
Use data/processed/featured/ rather than prepared rows by position.
Recommended first LSTM input shape is (n_sequences, 24, 11), using
current PM2.5, cyclical time, weather, and physical wind components.
Accepted sequence counts are 101,168 train, 22,657 validation, and
22,672 test with zero accepted-window timestamp or split violations.

First LSTM baseline:
Native validation pooled RMSE 15.036, pooled MAE 9.619, pooled R2 0.834.
On 22,477 matched validation timestamps, Persistence pooled RMSE was
11.848, frozen RF pooled RMSE was 12.233, and LSTM pooled RMSE was
15.021. LSTM beat Persistence on 10/51 stations and RF on 7/51 stations.

Residual LSTM baseline:
Native validation pooled RMSE 10.583, pooled MAE 6.577, pooled R2 0.918.
On the same 22,477 matched validation timestamps, residual LSTM pooled
RMSE was 10.588 versus Persistence 11.848, frozen RF 12.233, and direct
LSTM 15.021. Residual LSTM beat Persistence on 49/51 stations, RF on
41/51 stations, and direct LSTM on 49/51 stations.

Graph design audit:
Canonical featured data has 56 sensor-qualified PM2.5 datasets, but the
current graph mapping has 54 nodes because duplicate human station names
are dropped. The first supervised graph model should use the 51
train+validation model-usable nodes while preserving a canonical 56-node
registry. Static dynamic-edge candidates should be the directed expansion
of the symmetric KNN union. Dynamic A->B wind edges should use source-node
wind and transport direction `(wind_direction + 180) % 360`.

Corrected static graph foundation:
Graph scripts 01-04 now produce 56 canonical nodes, identify 51
train+validation model-usable nodes, regenerate 56x56 distance and
bearing matrices, and build 188 undirected KNN candidate pairs expanded
to 376 directed static edges. The adjacency edge set and static edge CSV
now match exactly.

Dynamic wind-edge weights:
Script 05 now generates raw unnormalized source-wind-controlled dynamic
weights over all 376 static candidate edges. The output has 47,988
timestamps and 2,919,724 rows. Active edges are 47.522%, zero-weight
edges are 52.478%, missing wind is 0.000%, and all validation checks
passed. The 51-node supervised subgraph has 326 supervised candidate
edges, no isolated nodes, minimum out-degree 4, median out-degree 6, and
maximum out-degree 9.
```

Current next methodological direction:

> Build graph snapshots and node/edge masks using the corrected static
> graph foundation and validated dynamic wind-edge weights.

After temporal robustness is understood, likely future phases include:

1. design graph snapshots and node/edge masks,
2. integrate residual temporal baseline into graph-ready snapshots,
3. validate graph snapshot chronology and supervised-node masks,
4. eventual GAT-GRU or related spatio-temporal architecture,
5. targeted graph/temporal diagnostics only if validation requires them.

---

# 28. Notes for final thesis/report writing

When converting this development record into the final report:

- Do not present the work as a sequence of coding bugs only. Present the timestamp and wind corrections as **data-definition and physical-semantics validation**.
- Explain why old metrics from pre-canonical preprocessing are not directly comparable with later metrics.
- State clearly that Persistence is a formal baseline, not a placeholder.
- Separate macro and pooled metrics because they answer different questions.
- Mention R² instability under very small target variance.
- Describe Random Forest's advantage carefully: it improves pooled RMSE, but Persistence remains better in pooled MAE and station-level consistency.
- Describe wind-ablation evidence cautiously: there is small pooled incremental predictive information, but the effect is heterogeneous and not causal evidence.
- Emphasize that local wind features are not equivalent to modeling inter-station pollution transport.
- Explain why future graph alignment must use the wind transport direction rather than raw meteorological FROM direction.
- Preserve chronological evaluation and avoid random splitting for the time-series forecasting task.
- Treat rolling-origin validation as a response to temporal distribution shift and repeated inspection of a single fixed test period.
- Explain that the first LSTM baseline should use sequence-native inputs
  rather than full tabular lag/rolling features, because the sequence
  itself is the temporal representation.

This file should continue to be updated when a **major new research phase** is completed, especially rolling-origin validation, sequence modeling, graph integration, and final wind-aware model evaluation.

---

# 29. LSTM sequence dataset validation

Before implementing or training an LSTM, the sequence data design was
validated explicitly.

The main finding was methodological: `data/processed/prepared/` is not a
safe source for row-position sequence windows. It drops rows with missing
current or target PM2.5, so two neighboring prepared rows can be many
hours apart. The validator therefore recommends `data/processed/featured/`
as the sequence source because it preserves the hourly timeline.

Source evidence:

```text
Featured stations checked: 56
Featured invalid hourly gaps: 0
Prepared stations with row gaps: 51
Prepared invalid hourly gaps: 5,051
Prepared largest adjacent-row gap: 11,636 hours
```

The proposed future LSTM window is:

```text
Input: 24 consecutive hourly timestamps
Target: PM2.5 exactly one hour after the final input timestamp
Rejected: missing input/target values, timestamp gaps, split crossings
```

The first LSTM baseline should use the sequence-native input design:

```text
pm2_5, hour_sin, hour_cos, month_sin, month_cos, temperature, humidity,
pressure, dew_point, wind_u, wind_v
```

This is preferred over full `MODEL_FEATURE_COLUMNS` because an LSTM is
meant to learn temporal dependence from the ordered 24-hour input itself.
Including handcrafted lag and rolling PM2.5 features would mix a tabular
feature-engineering strategy into the first sequence baseline and would
remove additional windows when lag/rolling values are missing.

Accepted sequence counts:

```text
Sequence-native design:
train 101,168; validation 22,657; test 22,672; total 146,497

Full MODEL_FEATURE_COLUMNS:
train 81,702; validation 18,677; test 18,928; total 119,307
```

Accepted-window proofs showed zero invalid input lengths, zero invalid
hourly input gaps, zero invalid one-hour targets, and zero split
membership violations.

Recommended future architecture: create a reusable sequence index from
featured station files, storing station, split, input-start timestamp,
input-end timestamp, and target timestamp. Materialize tensors from this
index with shape `(n_sequences, 24, 11)` and scalar next-hour PM2.5
targets. Fit scaling on training input rows only.

---

# 30. First LSTM baseline

The first LSTM forecasting baseline was implemented after validating the
sequence data design. This was intentionally a simple station-specific
baseline, not a tuned sequence-model study.

Implementation:

```text
scripts/20_lstm_baseline.py
scripts/analysis/lstm_baseline.py
```

Runtime:

```text
PyTorch 2.13.0+cpu
CUDA unavailable
Device: CPU
```

The model used:

```text
input_size = 11
hidden_size = 64
num_layers = 1
batch_first = True
Linear output head
Adam, learning_rate = 0.001
MSE loss
batch_size = 64
max_epochs = 50
early stopping patience = 5
random seed = 42
```

Input windows were 24 consecutive hourly rows from
`data/processed/featured/`, predicting PM2.5 one hour after the final
input timestamp. The input columns were:

```text
pm2_5, hour_sin, hour_cos, month_sin, month_cos, temperature, humidity,
pressure, dew_point, wind_u, wind_v
```

No lag or rolling columns were used. No test split was evaluated.

Scalers were fit per station on training sequences only:

```text
input scaler: fit on training input values only
target scaler: fit on training targets only
validation: transformed with fixed train-fitted scalers
metrics: calculated after inverse-transforming predictions
```

Training cohort:

```text
Stations trained: 51
Stations skipped: 5
Native validation sequences: 22,657
Matched comparison rows: 22,477
```

Skipped stations:

```text
Kathmandu University__sensor_15286458
Kathmandu University__sensor_15286975
Kathmandu University__sensor_15286980
Pulchowk (SC-15)-GD Labs
Tarakeswor (SC-15)- GD Labs
```

Native LSTM validation metrics:

```text
Macro MAE: 10.895
Macro RMSE: 15.112
Macro median R2: 0.713
Pooled MAE: 9.619
Pooled RMSE: 15.036
Pooled R2: 0.834
```

Matched validation comparison:

```text
LSTM pooled RMSE: 15.021
Persistence pooled RMSE: 11.848
Frozen RF pooled RMSE: 12.233

LSTM pooled MAE: 9.608
Persistence pooled MAE: 7.142
Frozen RF pooled MAE: 7.286
```

Station-level RMSE wins:

```text
LSTM beat Persistence: 10/51
LSTM beat RF: 7/51
LSTM beat both: 4/51
```

Best-epoch behavior:

```text
mean best epoch: 16.3
median best epoch: 15
min best epoch: 1
max best epoch: 50
best epoch <= 5: 9 stations
best epoch >= 40: 2 stations
hit max epoch: 1 station
```

Interpretation:

The first LSTM baseline is a valid sequence experiment, but it does not
currently add useful temporal signal beyond Persistence or the frozen
Random Forest. This is an important negative result: for one-hour-ahead
PM2.5 forecasting, current PM2.5 remains extremely difficult to beat, and
the simple station-specific LSTM is likely underpowered, unstable on
small station datasets, or overfitting local validation periods.

This result does not justify moving immediately to a Transformer. The
next sequence-model step should be diagnostic: identify which stations
benefit, inspect loss curves and target shifts, and consider whether a
pooled LSTM, stronger regularization, or a much simpler sequence
baseline is a better bridge before graph-aware modeling.

---

# 31. Persistence-anchored residual LSTM

The direct station-specific LSTM underperformed because it tried to
predict absolute PM2.5 directly. A follow-up experiment kept the same
model and training setup but changed the target to a correction around
Persistence:

```text
delta_pm25 = PM2.5(t+1) - PM2.5(t)
prediction = PM2.5(t) + predicted_delta
```

Implementation:

```text
scripts/21_lstm_residual_baseline.py
scripts/analysis/lstm_residual_baseline.py
```

This experiment used the same 24-hour featured windows, 11
sequence-native input columns, station-specific training, LSTM
architecture, optimizer, learning rate, batch size, maximum epochs,
patience, and seed as the direct LSTM. Only the target changed. No test
split was evaluated.

Residual scaling was fit on training residuals only. Absolute PM2.5
metrics were computed after adding the predicted residual back to the
final PM2.5 value in the input window.

The target distribution shows why this reformulation matters:

```text
Train absolute PM2.5 target mean/std: 68.673 / 42.349
Train residual target mean/std:       0.016 / 21.297

Validation absolute target mean/std:  59.297 / 36.950
Validation residual target mean/std:  0.083 / 11.840
```

Native residual LSTM validation:

```text
Macro MAE: 6.653
Macro RMSE: 10.069
Macro median R2: 0.836
Pooled MAE: 6.577
Pooled RMSE: 10.583
Pooled R2: 0.918
```

Matched four-way validation comparison:

```text
Direct LSTM pooled RMSE:   15.021
Residual LSTM pooled RMSE: 10.588
Persistence pooled RMSE:   11.848
Frozen RF pooled RMSE:     12.233

Direct LSTM pooled MAE:    9.608
Residual LSTM pooled MAE:  6.580
Persistence pooled MAE:    7.142
Frozen RF pooled MAE:      7.286
```

Station-level RMSE wins:

```text
Residual LSTM beat Persistence: 49/51
Residual LSTM beat RF: 41/51
Residual LSTM beat direct LSTM: 49/51
Residual LSTM beat all three: 39/51
```

Best epoch behavior:

```text
mean best epoch: 6.9
median best epoch: 5
min best epoch: 1
max best epoch: 32
```

Interpretation:

Residual learning materially improved the LSTM. Scientifically, this is
not just a better neural-network trick; it changes the learning problem
to modeling departures from the strongest one-hour baseline. The residual
target is centered near zero and lower-variance, so the LSTM can focus on
short-term change rather than relearning the absolute PM2.5 level.

This is now the strongest validation-only temporal baseline. Because it
beats Persistence and frozen RF, the next research phase should move to
wind-aware graph design and inter-station interaction rather than further
station-specific LSTM tuning.

---

# 32. Graph design audit before dynamic wind edges

Before implementing dynamic wind edges, the existing graph code and
Nirika-work graph scripts 01-07 were audited. No dynamic edge generator,
graph snapshots, sliding windows, GNN, or GAT model was implemented in
this phase.

The branch review found:

```text
Nirika-work graph scripts 01-04 match main.
Nirika-work graph scripts 05-07 are empty placeholders.
main graph scripts 05-07 are also empty placeholders.
Do not merge Nirika-work as-is.
```

A reproducible audit helper was added:

```text
scripts/22_graph_design_audit.py
scripts/analysis/graph_design_audit.py
docs/graph_design_audit.md
```

The central issue is graph node identity. Current modeling data is
sensor-qualified where necessary, but the existing `StationMapper` uses
human station names and drops duplicates.

Audit summary:

```text
metadata rows: 56
unique human station names: 54
unique PM2.5 sensors: 56
featured datasets: 56
model-usable train+validation datasets: 51
current station_mapping nodes: 54
```

The three Kathmandu University PM2.5 sensors share the same human station
name, so the current graph mapping collapses distinct datasets. The
correct canonical registry should have 56 rows keyed by featured
`dataset_name`, with `pm25_sensor_id`, `location_id`, latitude, and
longitude retained. The first supervised graph model should use the 51
train+validation model-usable nodes, while preserving the full 56-node
registry for reproducibility.

Distance and directed bearing calculations were verified for the current
54-node artifacts:

```text
distance matrix symmetric: true
distance diagonal zero: true
max distance recalculation error: 0.0 km
bearing diagonal zero: true
max bearing recalculation error: 0.0 degrees
max reverse-bearing 180-degree error: 0.13 degrees
```

These calculations are mathematically acceptable, but they must be
regenerated after the node identity correction.

The static KNN graph currently uses K=5 and symmetrizes the adjacency.
However, the static edge CSV does not represent the same directed edge
set as the adjacency:

```text
current static edge rows: 270
symmetrized adjacency directed edges: 362
symmetrized adjacency undirected pairs: 181
static rows missing reverse directions: 92
```

For dynamic wind edges, the candidate edge table should be the directed
expansion of the symmetric KNN union: build undirected candidate pairs
from the symmetrized KNN adjacency, then emit both A->B and B->A with
their own A-to-B bearings.

The proposed dynamic edge design is:

```text
transport_direction_A(t) = (wind_direction_A(t) + 180) % 360

delta_AB(t) = angular difference between transport_direction_A(t)
              and bearing A->B

alignment_AB(t) = max(0, cos(delta_AB(t)))
speed_factor_A(t) = wind_speed_A(t) / (wind_speed_A(t) + 5)
distance_factor_AB = exp(-distance_AB / lambda_d)

raw_weight_AB(t) =
    candidate_AB * alignment_AB(t) * speed_factor_A(t) *
    distance_factor_AB
```

Use source-node wind for A->B because the edge represents possible
transport of pollution leaving source A toward target B. Target wind may
remain a node feature or later modifier, but it should not control the
primary transport edge.

Edge cases:

```text
near-zero wind (<0.5 km/h): weight 0 with calm_wind flag
wind perpendicular/away from B: alignment 0, weight 0
missing PM2.5: use node/target masks, do not impute only for graph shape
missing weather: keep edge row with missing_source_wind flag
non-shared timestamps: use global hourly snapshots plus masks
```

Before dynamic edge implementation, these graph scripts must be corrected:

```text
01_station_mapping.py: use sensor-qualified dataset identity
02_distance_matrix.py: regenerate from corrected nodes
03_bearing_matrix.py: regenerate directed bearings from corrected nodes
04_static_graph.py: emit directed expansion of symmetric KNN union
05_dynamic_edge_weights.py: implement only after the above corrections
06_graph_snapshots.py and 07_sliding_windows.py: keep pending until
dynamic edge schema is implemented
```

The full schema recommendation for future dynamic edge rows is in
`docs/graph_design_audit.md`.

---

# 33. Corrected static graph foundation

The graph foundation was corrected and regenerated after the audit. This
phase updated only scripts 01-04 and static artifacts; dynamic wind
weights were not implemented.

Updated scripts:

```text
scripts/graph/01_station_mapping.py
scripts/graph/02_distance_matrix.py
scripts/graph/03_bearing_matrix.py
scripts/graph/04_static_graph.py
```

The corrected station mapping now uses canonical sensor-qualified
identity:

```text
identity key: dataset_name with pm25_sensor_id retained
canonical nodes: 56
unique dataset_name: true
unique pm25_sensor_id: true
model-usable train+validation nodes: 51
missing coordinates: 0
```

This fixes the previous `StationMapper` issue where duplicate human
station names collapsed multiple PM2.5 sensors into one graph node.
Human station name, `location_id`, latitude, and longitude are retained
as metadata, and node IDs are deterministic by sorted `dataset_name` and
`pm25_sensor_id`.

Distance and bearing were regenerated from the corrected node registry:

```text
distance matrix: 56x56
distance symmetric: true
distance diagonal zero: true
complete undirected distance edges: 1,540

bearing matrix: 56x56
bearing directed/non-symmetric: true
complete directed bearing edges: 3,080
```

The static KNN graph was regenerated using K=5:

```text
symmetric KNN union: 188 undirected candidate pairs
directed static candidate edges: 376
adjacency directed edges: 376
static edge CSV rows: 376
candidate pairs missing reverse direction: 0
```

The important implementation correction is that the adjacency and static
edge CSV now represent exactly the same directed candidate set. Each
candidate pair appears as both A->B and B->A, with distance, directed
bearing, source/target dataset names, source/target PM2.5 sensor IDs,
and source/target human station names.

This corrected foundation is now ready for dynamic wind edge weights.

---

# 34. Dynamic wind-edge weights

The first actual dynamic wind-edge stage was implemented after correcting
the static graph foundation. This stage computes raw, auditable,
source-wind-controlled edge weights. It does not row-normalize weights,
build graph snapshots, create sliding windows, or train a graph model.

Implementation:

```text
scripts/graph/05_dynamic_edge_weights.py
```

Inputs:

```text
data/processed/graph/static_graph.csv
data/metadata/station_mapping.csv
data/processed/featured/
```

For candidate edge A->B and timestamp `t`, source-node wind controls the
edge:

```text
transport_direction = (source_wind_direction + 180) % 360
angle_difference = circular difference between transport direction
                   and bearing A->B
alignment = max(0, cos(angle_difference))
speed_factor = wind_speed / (wind_speed + 5)
lambda_d = median distance across static directed candidates
distance_factor = exp(-distance_km / lambda_d)
raw_dynamic_weight = alignment * speed_factor * distance_factor
```

The run computed:

```text
lambda_d = 1.930 km
timestamps = 47,988
rows = 2,919,724
candidate edges = 376
supervised candidate edges = 326
active-edge percentage = 47.522%
zero-weight percentage = 52.478%
missing-wind percentage = 0.000%
calm-wind percentage = 1.867%
```

Validation checks passed:

```text
every generated row corresponds to a static candidate edge
no non-candidate edges
all static candidates present
weights are never negative
alignment is in [0, 1]
speed_factor is in [0, 1)
distance_factor is in (0, 1]
calm wind gives zero weight
away/perpendicular wind gives zero weight
missing source wind gives zero weight
candidate pairs missing reverse direction: 0
opposite directions can have different weights
future rows used: false
```

The supervised subgraph after filtering `supervised_edge=True` remains
usable:

```text
supervised nodes = 51
supervised candidate edges = 326
min out-degree = 4
median out-degree = 6
max out-degree = 9
isolated nodes = 0
```

The lowest-degree supervised node is `Tarakeswor (SC-14)-GD Labs`, with
out-degree 4 and in-degree 4. This is acceptable and does not justify
silently changing KNN.

Next step: consume the masked graph snapshot artifacts in a reviewed
graph dataset loader or temporal graph-window builder.

---

# 35. Graph snapshot construction and synchronization analysis

The graph snapshot stage was implemented after the dynamic wind-edge
weights. This stage prepares the first supervised graph-model data
foundation but does not train a GNN, create sliding windows, impute
missing values, or row-normalize dynamic edge weights.

Implementation:

```text
scripts/graph/06_graph_snapshots.py
```

Inputs:

```text
data/metadata/station_mapping.csv
data/processed/featured/
data/processed/graph/static_graph.csv
data/processed/graph/dynamic_edge_weights.csv
```

The first supervised graph uses the 51 `model_usable` nodes and keeps
their canonical node IDs from the 56-node graph registry. Nodes are not
renumbered.

Node features at timestamp `t`:

```text
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

```text
residual_pm25(t+1) = pm2_5(t+1) - pm2_5(t)
```

This matches the residual LSTM formulation that outperformed the direct
LSTM formulation. The target is accepted only when `t+1` is exactly one
hour after `t` and remains inside the same chronological
train/validation/test split.

Generated compact artifacts:

```text
data/processed/graph/snapshots/supervised_nodes.csv
data/processed/graph/snapshots/snapshot_nodes.csv.gz
data/processed/graph/snapshots/snapshot_edges.csv.gz
data/processed/graph/snapshots/snapshot_timestamp_summary.csv
data/processed/graph/snapshots/snapshot_policy_summary.csv
data/processed/graph/snapshots/snapshot_validation.csv
data/processed/graph/snapshots/snapshot_valid_node_distribution.csv
data/processed/graph/snapshots/snapshot_continuous_runs.csv
```

The snapshot node artifact has one row per timestamp and supervised node,
with explicit flags for row existence, complete input features, exact
t+1 target availability, split-safe target validity, and usable
supervised node-target pairs. The edge artifact attaches only supervised
dynamic edges and keeps `raw_dynamic_weight` unchanged.

Policy comparison:

```text
global hourly timestamps = 47,987
node snapshot rows = 2,447,337
edge snapshot rows = 2,659,101

strict usable timestamps = 0
strict node-target sequences = 0

masked usable timestamps = 30,067
masked train/validation/test usable timestamps = 17,923 / 4,969 / 7,175
masked node-target sequences = 201,608
```

The strict policy is unusable because no global timestamp has all 51
supervised nodes with valid inputs and valid one-hour-ahead targets. The
first GNN dataset should therefore use the masked fixed-graph policy:
retain all 51 canonical nodes at each timestamp and apply explicit
input/target masks during training and evaluation.

Synchronization distribution:

```text
valid input nodes per timestamp: min 0, median 1, max 43
valid target nodes per timestamp: min 0, median 1, max 43
valid input+target nodes per timestamp: min 0, median 1, max 42
valid directed edges per timestamp: min 0, median 0, max 234
active dynamic edges per timestamp: min 0, median 0, max 118
```

Coverage thresholds:

```text
timestamps with 51 valid input nodes = 0
timestamps with >=45 valid input nodes = 0
timestamps with >=40 valid input nodes = 47
timestamps with >=30 valid input nodes = 1,921
```

Longest continuous usable runs:

```text
masked: 2026-01-04 18:00 to 2026-05-08 13:00, 2,972 hours
masked >=30 input nodes: 2026-01-12 09:00 to 2026-01-23 14:00, 270 hours
masked >=40 input nodes: 2026-05-19 12:00 to 2026-05-20 10:00, 23 hours
```

Validation checks passed:

```text
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

This result is scientifically important: the graph problem is not a
dense fully synchronized 51-station panel. It is a sparse but fixed-node
spatiotemporal forecasting problem, so masks are part of the dataset
definition rather than a modeling convenience.

---

# 36. Masked 24-hour spatio-temporal graph windows

The masked graph snapshot representation was extended into 24-hour
spatio-temporal graph windows for future temporal GNN experiments. This
stage creates dataset artifacts only. It does not train GAT, GAT-GRU, or
any graph model.

Implementation:

```text
scripts/graph/07_sliding_windows.py
```

The stage consumes outputs from `06_graph_snapshots.py` and preserves
the same 51-node canonical order and 326 supervised directed candidate
edges.

Window definition:

```text
input graph snapshots: t-23 ... t
window length: 24 consecutive hours
prediction target: residual_pm25(t+1) at final timestamp t
```

The target remains the residual formulation:

```text
residual_pm25(t+1) = pm2_5(t+1) - pm2_5(t)
```

The key mask rule is node-specific:

```text
sequence_input_valid = input_valid for that node at all 24 input hours
supervised_target_valid = sequence_input_valid AND target_valid at final t
```

Therefore, a node cannot contribute supervised loss merely because its
final target exists. It must also have a complete 24-hour input history.

The representation is compact. It does not duplicate full 24-hour
feature and edge tensors for every overlapping window. Instead it stores
snapshot arrays once and creates a window index.

Generated artifacts:

```text
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

Stored array shapes:

```text
node_features: (47,987, 51, 11)
input_valid_mask: (47,987, 51)
target_valid_mask: (47,987, 51)
residual_targets: (47,987, 51)
edge_weights: (47,987, 326)
edge_valid_mask: (47,987, 326)
edge_active_mask: (47,987, 326)
window_sequence_input_valid_mask: (21,457, 51)
window_target_valid_mask: (21,457, 51)
```

Approximate storage:

```text
compressed graph_window_arrays.npz = 12.0 MB
uncompressed array memory footprint = 209.1 MB
graph_window_index.csv = 2.2 MB
```

Usable windows:

```text
train = 10,561 windows, 13,238 supervised node-target examples
validation = 4,096 windows, 6,326 supervised node-target examples
test = 6,800 windows, 128,756 supervised node-target examples
all = 21,457 windows, 148,320 supervised node-target examples
```

The test split is indexed for completeness only. It must not be used for
model choice, configuration choice, tuning, or early stopping decisions.

Targets per usable window:

```text
train: min 1, median 1, max 3
validation: min 1, median 1, max 3
test: min 1, median 22, max 39
all: min 1, median 1, max 39
```

Threshold distribution:

```text
windows with >=1 target = 21,457
windows with >=10 targets = 5,354
windows with >=20 targets = 4,335
windows with >=30 targets = 182
windows with >=40 targets = 0
```

Rejected candidate windows:

```text
too-short 24h history = 23
non-hourly continuity = 0
split crossing = 49
zero valid supervised targets = 26,458
```

Longest continuous usable runs:

```text
train: 2023-07-17 18:00 to 2023-08-18 23:00, 774 windows
validation: 2025-07-19 09:00 to 2025-09-14 22:00, 1,382 windows
test: 2026-01-05 17:00 to 2026-04-18 11:00, 2,467 windows
```

Validation checks passed:

```text
every accepted window has 24 hourly snapshots: true
no split crossing: true
target exactly t+1 after final input: true
target mask implies complete 24h input history: true
fixed 51-node ordering preserved: true
edge IDs/order consistent across timestamps: true
no future node features used: true
```

The main modeling implication is that the graph learner will need to
handle highly sparse train/validation target supervision. The future
loader should use the stored window index and masks directly rather than
filtering to dense synchronized windows.
