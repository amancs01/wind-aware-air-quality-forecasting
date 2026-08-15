# Research Notes

Findings and engineering decisions made during development, kept for context on *why* the pipeline is built the way it is.

## OpenAQ API Findings

**Station hierarchy**: Location → Sensor → Measurements. Measurements must be retrieved per sensor, not per location.

**Observations**
- Monthly downloads are significantly more reliable than yearly downloads — the API returns `HTTP 408` for large yearly requests.
- Retry with exponential backoff improves stability against transient failures.
- Some months exceed 1000 measurements, so month-level pagination is required.
- Timestamps contain no duplicates.
- Missing PM2.5 values are returned as `null` rather than being omitted.

**Coverage**
- OpenAQ data for Nepal is sourced from GD Labs.
- Most GD Labs stations only become active in late 2025, so PM2.5 coverage (roughly 8–10 months) is far shorter than weather coverage (roughly five years) for most stations.
- Embassy Kathmandu is the exception, with denser historical coverage.
- Consequence: dataset trimming (removing leading rows with no PM2.5 label) is necessary before model training.

## Timestamp Alignment

Weather (Open-Meteo) and air-quality (OpenAQ) datasets use different timestamp formats, so direct string matching fails. Timestamps are converted to proper `datetime` objects and normalized before merging.

## Forecasting Target

The prediction target is PM2.5 concentration one hour ahead:

```
target_pm2_5 = pm2_5.shift(-1)
```

This turns the task into *forecasting* future air quality rather than estimating current air quality, which better reflects a real-world use case (e.g., "should I avoid going outside in an hour?").

## Baseline Model

A persistence baseline is implemented before any machine learning model:

```
PM2.5(t + 1) = PM2.5(t)
```

This gives a reference point that every subsequent model must outperform — if a model can't beat "tomorrow looks like today," it isn't adding value.

## Evaluation Metrics

All forecasting models are compared using the same three regression metrics, for consistency across experiments:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R2)

Baseline reports now distinguish macro and pooled metrics:

- Macro metrics average station-level scores equally.
- Pooled metrics are computed over all evaluated prediction rows.
- Macro mean R2 is not a global R2 and is unstable for stations with
  very low target variance.

The fair baseline benchmark uses the same row mask for persistence and
Ridge: all configured model features plus `target_pm2_5` must be
present. Persistence still predicts current PM2.5; the shared frame only
controls which rows are scored.

## Model Architecture

A reusable `BaseModel` class standardizes dataset loading, construction
of the fair evaluation frame, evaluation, result storage, and reporting
across models. Future models should inherit from this base class and
implement only their own training/prediction logic.

Current fair benchmark interpretation:

```text
Persistence pooled R2: 0.820
Original Ridge(alpha=10.0) pooled R2: 0.751
Validation-selected Ridge(alpha=1000.0) pooled R2: 0.805
Persistence macro mean R2: 0.692
Original Ridge(alpha=10.0) macro mean R2: -127.128
Validation-selected Ridge(alpha=1000.0) macro mean R2: -29.748
Validation-selected Ridge(alpha=1000.0) macro median R2: 0.702
```

The Ridge macro mean is dominated by Sundarighat (SC-23) - GD Labs,
where the test target variance is extremely small and the test PM2.5
distribution shifts far below the training distribution. Removing that
station from the macro mean changes Ridge macro R2 from -127.128 to
about 0.341, so this is an R2 aggregation warning rather than evidence
that every Ridge forecast collapsed.

Persistence remains the stronger fair benchmark overall. The one-hour
PM2.5 autocorrelation is high, so a model must add real value beyond
current PM2.5 to beat persistence.

## Validation-Based Linear Model Selection

The first linear-model selection milestone used train and validation
splits only. The test split was not used to choose alpha, scaling, or
model family.

Candidates:

- `LinearRegression()`
- unscaled `Ridge(alpha)` for `0.001`, `0.01`, `0.1`, `1.0`, `10.0`,
  `100.0`, and `1000.0`
- train-fitted `StandardScaler() + Ridge(alpha)` for the same alpha grid

Primary selection metric: pooled validation RMSE.

Validation persistence remained the strongest overall benchmark:

```text
Persistence pooled RMSE: 12.300
Persistence pooled R2: 0.889
```

The best tested linear configuration was:

```text
Ridge(alpha=1000.0), no scaler
```

Validation result for the selected linear model:

```text
macro MAE: 8.677
macro RMSE: 12.430
macro mean R2: 0.664
macro median R2: 0.775
pooled MAE: 8.575
pooled RMSE: 13.225
pooled R2: 0.871
```

It beat persistence by validation RMSE on 17 stations; persistence beat
it on 34 stations. After freezing that configuration, final test
evaluation reported pooled RMSE 12.591 and pooled R2 0.805, still below
test persistence at pooled RMSE 12.083 and pooled R2 0.820.

## Random Forest Baseline

Random Forest was added to test whether nonlinear interactions among the
same existing PM2.5, lag, rolling, time, wind, and weather features
improve one-hour-ahead forecasting. No features were added or removed,
and missing lag/rolling values were not imputed.

The Random Forest model remains station-specific, matching the existing
model architecture. Each station trains its own forest, while all
stations share one validation-selected global hyperparameter
configuration.

Tuned parameters:

- `max_depth`: limits how deep each tree can grow. Smaller values make
  each tree simpler and reduce memorization.
- `min_samples_leaf`: requires each terminal leaf to contain at least
  this many training rows. Larger values smooth predictions and improve
  regularization.
- `max_features`: controls how many features each split may consider.
  Using all features can make individual trees stronger; using `sqrt`
  can increase tree diversity.
- `n_estimators`: controls the number of trees. It was fixed rather
  than tuned aggressively because this milestone focuses on structural
  complexity.

Resource note: unbounded forests with `max_depth=None` exhausted local
memory before producing validation metrics, even after reducing worker
parallelism. The final feasible grid used `n_estimators=100`,
`n_jobs=1`, `max_depth` in `[10, 20]`, `min_samples_leaf` in
`[1, 5, 10]`, and `max_features` in `[1.0, "sqrt"]`.

Selected configuration by pooled validation RMSE:

```text
n_estimators: 100
max_depth: 10
min_samples_leaf: 10
max_features: 1.0
random_state: 42
n_jobs: 1
```

Validation result for selected Random Forest:

```text
rows: 25,689
macro MAE: 7.675
macro RMSE: 11.901
macro mean R2: 0.727
macro median R2: 0.800
pooled MAE: 7.370
pooled RMSE: 12.450
pooled R2: 0.886
```

Validation comparison:

```text
Persistence pooled RMSE: 12.300
Ridge(alpha=1000.0) pooled RMSE: 13.225
Random Forest pooled RMSE: 12.450
```

Random Forest improved over Ridge on validation, but it did not beat
persistence overall on pooled validation RMSE. It did beat persistence
by station-level validation RMSE on 27 datasets, while persistence won
on 24 datasets.

Frozen final test result:

```text
rows: 26,236
macro MAE: 6.735
macro RMSE: 9.695
macro mean R2: -78.855
macro median R2: 0.704
pooled MAE: 6.556
pooled RMSE: 11.652
pooled R2: 0.833
```

On held-out test rows, Random Forest improved pooled RMSE over
persistence by 0.430, or 3.56%. However, Random Forest had worse pooled
MAE than persistence and beat persistence by station-level test RMSE on
only 11 of 51 datasets. This means nonlinear interactions helped reduce
larger squared errors overall, but the improvement is not uniform across
stations.

## XGBoost Baseline

XGBoost was added to test whether sequential gradient-boosted trees can
extract more signal from the same existing features than persistence,
Ridge, and Random Forest. The experiment kept the data, target,
chronological splits, `MODEL_FEATURE_COLUMNS`, missing-value policy, and
fair evaluation frame unchanged.

Gradient boosting builds trees one after another. Each new tree tries to
correct errors left by the current ensemble, so the model can learn
nonlinear interactions in a more directed way than a bagged Random
Forest. If boosting continues for too long, validation performance can
stall or worsen, so XGBoost used validation-based early stopping.

Main parameters:

- `learning_rate`: how strongly each new tree changes predictions.
  Smaller values learn more gradually and usually need more trees.
- `max_depth`: maximum complexity of each tree. Deeper trees can model
  richer interactions but may overfit.
- `min_child_weight`: controls how easily small specialized branches are
  created. Larger values make the model more conservative.
- `subsample`: fraction of training rows available to each tree.
- `colsample_bytree`: fraction of features available to each tree.
- `early_stopping_rounds`: stops training when validation RMSE has not
  improved recently.

Environment:

```text
Python: 3.12.0
XGBoost: 3.4.0
```

Validation grid:

```text
learning_rate: [0.03, 0.10]
max_depth: [3, 6]
min_child_weight: [1, 5]
fixed: n_estimators=1000, early_stopping_rounds=50, subsample=0.8,
colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
random_state=42, n_jobs=1
```

Selected configuration by pooled validation RMSE:

```text
learning_rate: 0.1
max_depth: 3
min_child_weight: 5
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.0
reg_lambda: 1.0
n_estimators: 1000
early_stopping_rounds: 50
tree_method: hist
random_state: 42
n_jobs: 1
```

Validation result for selected XGBoost:

```text
rows: 25,689
macro MAE: 7.898
macro RMSE: 11.938
macro mean R2: 0.729
macro median R2: 0.787
pooled MAE: 7.674
pooled RMSE: 12.700
pooled R2: 0.881
```

Best-iteration behavior across validation stations:

```text
min: 19
median: 67
mean: 88.78
max: 261
stations hitting n_estimators=1000: 0
```

Validation comparison:

```text
Persistence pooled RMSE: 12.300
Ridge(alpha=1000.0) pooled RMSE: 13.225
Random Forest pooled RMSE: 12.450
XGBoost pooled RMSE: 12.700
```

XGBoost beat persistence by validation RMSE on 25 stations and Random
Forest on 23 stations, but it did not beat either persistence or Random
Forest by pooled validation RMSE.

Frozen final test result:

```text
rows: 26,236
macro MAE: 7.336
macro RMSE: 10.257
macro mean R2: -443.742
macro median R2: 0.701
pooled MAE: 7.184
pooled RMSE: 12.043
pooled R2: 0.821
```

On held-out test rows, XGBoost improved pooled RMSE over persistence by
0.039, or 0.33%, but worsened pooled MAE by 1.179. Compared with Random
Forest, XGBoost had higher pooled RMSE by 0.391, higher pooled MAE by
0.628, and lower pooled R2 by 0.011. XGBoost beat persistence by
station-level test RMSE on 11 of 51 datasets and beat Random Forest on
18 of 51.

Interpretation: sequential boosting did not provide a meaningful overall
improvement beyond Random Forest on the current feature set. Its tiny
pooled RMSE gain over persistence is concentrated and comes with worse
MAE and station-level heterogeneity.

## Wind-Component Semantics Correction

The project now treats `wind_direction` as meteorological direction: the
direction FROM which wind blows, measured clockwise from north. Feature
engineering converts this to physical components:

```text
wind_u = -wind_speed * sin(wind_direction)
wind_v = -wind_speed * cos(wind_direction)
```

`wind_u` is eastward, `wind_v` is northward, and all three wind columns
remain in km/h because the Open-Meteo downloader uses the API default
wind-speed unit.

The previous `wind_u = speed*cos(direction)` and
`wind_v = speed*sin(direction)` values were speed-scaled circular
direction encodings. They preserved the available wind information, so
the earlier classical baseline results were not meaningless, but the
columns were physically mislabeled. Correcting the semantics is
important for interpretation, thesis wording, and future graph edge
construction.

For future directed pollution-transport reasoning, compare station
source-to-target bearings with:

```text
transport_direction = (wind_direction + 180) % 360
```

Do not compare raw meteorological wind direction directly to a
source-to-target bearing unless deliberately modeling the upwind
direction.

## Validation-Only Feature Ablation

This milestone used train and validation data only. The test split was
not loaded, scored, or used for feature selection. The purpose was to
understand predictive signal from already-established feature groups,
not to improve or replace the production test benchmark.

The ablation used the frozen Random Forest configuration:

```text
n_estimators: 100
max_depth: 10
min_samples_leaf: 10
max_features: 1.0
random_state: 42
n_jobs: 1
```

Every ablation variant used the same full-feature-valid frames:

```text
datasets: 51
train rows: 115,725
validation rows: 25,689
row/target mismatches across variants: 0
```

Feature groups:

```text
A0 Persistence: pm2_5 prediction rule only
A1 Current PM only: pm2_5
A2 PM history: pm2_5, lag_6, lag_24, rolling_mean_6, rolling_std_6
A3 PM history + time: A2 plus hour_sin, hour_cos, month_sin, month_cos
A4 Full minus wind: A3 plus temperature, humidity, pressure, dew_point
A5 PM history + time + wind: A3 plus wind_u, wind_v
A6 Full current feature set: MODEL_FEATURE_COLUMNS
```

Validation summary:

```text
Persistence: pooled RMSE 12.300, pooled MAE 7.292, pooled R2 0.889
A1 current PM RF: pooled RMSE 13.453, pooled MAE 8.311, pooled R2 0.867
A2 PM history RF: pooled RMSE 13.407, pooled MAE 8.303, pooled R2 0.868
A3 PM history + time RF: pooled RMSE 12.499, pooled MAE 7.586, pooled R2 0.885
A4 full minus wind RF: pooled RMSE 12.497, pooled MAE 7.402, pooled R2 0.885
A5 history + time + wind RF: pooled RMSE 12.426, pooled MAE 7.498, pooled R2 0.886
A6 full RF: pooled RMSE 12.449, pooled MAE 7.369, pooled R2 0.886
```

Incremental findings use the sign convention:

```text
positive RMSE improvement = lower validation RMSE after adding features
```

Key increments:

```text
PM history over current PM only: +0.046 RMSE
Time over PM history: +0.908 RMSE
Non-wind weather over PM history + time: +0.002 RMSE
Wind over PM history + time: +0.073 RMSE
Conditional wind over full-minus-wind: +0.048 RMSE
Conditional non-wind weather over wind model: -0.023 RMSE
```

Interpretation:

- Current PM2.5 alone remains very strong, and the forced persistence
  rule beats the learned current-PM-only Random Forest on validation.
- PM2.5 history adds only a small improvement beyond current PM2.5.
- Cyclical time features provide the largest incremental validation
  gain among the tested groups.
- Non-wind weather adds little pooled RMSE value once PM history and
  time are present, though it improves pooled MAE.
- Wind provides a small positive pooled validation signal both without
  other weather and after non-wind meteorology is already included.
- The wind effect is heterogeneous: full features improved RMSE over
  full-minus-wind on 24 stations, worsened it on 27, and had a median
  station RMSE effect of -0.011.

This supports cautious thesis wording: there is validation evidence that
physical wind components add some predictive information beyond local
PM2.5 history, time, and ordinary meteorology in pooled RMSE, but the
benefit is small and not station-wide. This is not causal evidence and
should not be used to change the production feature set without a
deliberate future evaluation design. The existing test split has already
been observed during prior model milestones, so any material
feature-set decision should consider rolling-origin evaluation, a fresh
chronological holdout, or retaining the current full feature set for
interpretability.

## Rolling-Origin Validation

Rolling-origin validation was added to check whether conclusions from a
single validation/test period are stable across multiple chronological
forecast periods. It uses only the first 85% development portion of each
prepared station dataset and does not load the final 15% test split.

Folds:

```text
Fold 1: train 0-55%, validate 55-65%
Fold 2: train 0-65%, validate 65-75%
Fold 3: train 0-75%, validate 75-85%
```

Models were frozen:

```text
Persistence
Ridge(alpha=1000)
Random Forest: n_estimators=100, max_depth=10,
min_samples_leaf=10, max_features=1.0, random_state=42, n_jobs=1
```

Each fold used the same full-`MODEL_FEATURE_COLUMNS`-valid rows for all
models. Two tiny prepared datasets were skipped, leaving 51 datasets in
each fold.

Validation results:

```text
Fold 1:
Persistence pooled RMSE 15.611, pooled MAE 10.011, pooled R2 0.829
Ridge pooled RMSE 15.541, pooled MAE 10.540, pooled R2 0.831
Random Forest pooled RMSE 14.538, pooled MAE 9.490, pooled R2 0.852

Fold 2:
Persistence pooled RMSE 17.689, pooled MAE 9.136, pooled R2 0.777
Ridge pooled RMSE 16.503, pooled MAE 9.959, pooled R2 0.806
Random Forest pooled RMSE 15.107, pooled MAE 8.568, pooled R2 0.837

Fold 3:
Persistence pooled RMSE 11.862, pooled MAE 6.951, pooled R2 0.896
Ridge pooled RMSE 12.978, pooled MAE 8.218, pooled R2 0.876
Random Forest pooled RMSE 12.465, pooled MAE 7.175, pooled R2 0.885
```

Model-vs-persistence pooled RMSE effects:

```text
Fold 1 RF improvement: +1.073 RMSE, 31 station wins
Fold 2 RF improvement: +2.582 RMSE, 34 station wins
Fold 3 RF improvement: -0.603 RMSE, 27 station wins

Fold 1 Ridge improvement: +0.071 RMSE, 31 station wins
Fold 2 Ridge improvement: +1.185 RMSE, 24 station wins
Fold 3 Ridge improvement: -1.116 RMSE, 12 station wins
```

Random Forest station-win consistency:

```text
3/3 folds: 10 stations
2/3 folds: 24 stations
1/3 folds: 14 stations
0/3 folds: 3 stations
```

The strongest 3/3 RF-win stations included Dabali, Embassy Kathmandu,
Gokarneshwor, Sanepa, Sorakhutte, Sunakothi, Chhetrapati,
Dhathutole, Tyanglaphat, and Imadol.

Distribution-shift diagnostics showed that validation periods can differ
substantially from their expanding training histories. Largest target
mean shifts included Phora Durbar Kathman fold 3 (+60.5), Nakhipot fold
1 (+57.2), Lamtangil fold 2 (-51.7), Jadibuti fold 3 (-51.0), and
Sifal fold 1 (-48.7). These shifts help explain why model rankings are
not stable across all chronological windows.

Sundarighat remained an important diagnostic station:

```text
Fold 1 target mean shift: +12.6; RF beat Persistence
Fold 2 target mean shift: -23.8; RF beat Persistence
Fold 3 target mean shift: +9.2 with higher validation variance;
        Persistence beat both RF and Ridge
```

Interpretation:

- Random Forest shows useful nonlinear signal in two of three
  development windows and wins more station comparisons than it loses in
  all folds.
- Persistence remains very difficult to beat, especially in fold 3,
  where it has the best pooled RMSE and MAE.
- Ridge is less robust than Random Forest across folds.
- The project should report temporal robustness rather than relying on a
  single validation or test period.
- These results support a temporal-modeling next step, but they do not
  justify retuning RF or changing the feature set from this analysis
  alone.

## Data Quality Decisions

Stations without sufficient PM2.5 observations are excluded from model training after preprocessing, since they can't provide valid prediction targets (`MIN_TRAINING_ROWS` in `scripts/config.py`).

## Engineering Decisions

| # | Decision |
| - | -------- |
| 1 | The weather downloader should never redownload datasets that already exist on disk. |
| 2 | Shared filesystem utilities belong in `utils.py`, not duplicated per script. |
| 3 | Network retry logic should be reusable across all API clients, not reimplemented per downloader. |
| 4 | Networking is centralized inside a reusable HTTP client (`clients/http_client.py`) rather than living in individual scripts. |

## Canonical Hourly PM2.5 Finding

OpenAQ raw `/measurements` timestamps were not safe for modeling when
downstream code floored `datetimeFrom` to the hour. Some sensors exposed
multiple sub-hour alignments, which could create duplicate local model
timestamps.

The current modeling convention is:

```text
PM2.5(t) = OpenAQ one-hour PM2.5 interval ending at local clock time t
```

Using OpenAQ `/hours` and selecting intervals where
`datetimeTo.local.minute == 0` removed the duplicate-timestamp problem in
the regenerated merge. Timestamp continuity is now separate from PM2.5
availability: the weather-left timeline is hourly and continuous after
trimming, but many PM2.5 values are still missing because measurements
were unavailable.

New baseline metrics from this canonical hourly pipeline are the valid
baseline going forward. They should not be directly compared with older
metrics produced from `datetimeFrom.floor("h")` data.
