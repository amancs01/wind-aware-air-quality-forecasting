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
- Coefficient of Determination (R²)

## Model Architecture

A reusable `BaseModel` class standardizes dataset loading, evaluation, result storage, and reporting across models. Future models (Random Forest, XGBoost, LSTM, Transformer, and eventually the GAT-GRU) inherit from this base class and implement only their own training/prediction logic.

## Data Quality Decisions

Stations without sufficient PM2.5 observations are excluded from model training after preprocessing, since they can't provide valid prediction targets (`MIN_TRAINING_ROWS` in `scripts/config.py`).

## Engineering Decisions

| # | Decision |
| - | -------- |
| 1 | The weather downloader should never redownload datasets that already exist on disk. |
| 2 | Shared filesystem utilities belong in `utils.py`, not duplicated per script. |
| 3 | Network retry logic should be reusable across all API clients, not reimplemented per downloader. |
| 4 | Networking is centralized inside a reusable HTTP client (`clients/http_client.py`) rather than living in individual scripts. |