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
