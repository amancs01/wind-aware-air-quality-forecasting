# Changelog

Notable changes to the project, grouped by milestone. Ongoing "how/why" narrative lives in `development_log.md`.

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