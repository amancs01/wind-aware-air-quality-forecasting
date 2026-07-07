# Changelog

- Initial project structure
- Added weather downloader
- Added configuration system
- Added shared utilities

### Added
- Weather downloader now skips already downloaded yearly datasets.
- Added reusable utility functions in `utils.py`.
- Introduced API abstraction through `api.py`.
- Added OpenAQ station discovery script.
- Added OpenAQ sensor discovery.
- Added first Air Quality downloader prototype.
- Added automatic retry mechanism for failed API requests.
- Added exponential backoff for transient network failures.

### Changed
- Weather downloader now uses `Pathlib` instead of manual path strings.
- Weather downloader creates directories through reusable utilities.
- Air quality downloader switched from yearly requests to monthly partitioning.

### Fixed
- Fixed `.gitignore` encoding issue caused by UTF-16 LE.
- Fixed raw datasets being tracked by Git.
- Fixed Windows path problems caused by invalid station names.
- Fixed downloader repeatedly downloading existing weather files.

### Refactoring
- Introduced BaseDownloader abstraction.
- Refactored Weather Downloader.
- Refactored Air Quality Downloader.
- Centralized utilities into utils.py.
- Added reusable HTTP client.

### Data Processing
- Added dataset validation.
- Added dataset profiling.
- Added merged dataset generation.
- Fixed timestamp alignment between OpenAQ and Open-Meteo.
- Added trimming of leading rows without PM2.5 measurements.

### Feature Engineering
- Started feature engineering pipeline.
- Added temporal features (hour, day, month, weekday).

### Other
- Added support for downloading current-year (2026) weather data.
- Improved filename sanitization for Windows.

## Data Engineering Pipeline Completed

### Added
- Merge pipeline
- Data trimming
- Feature engineering
- Dataset preparation

### Feature Engineering
- Time features
- Lag features
- Rolling statistics
- Wind vector features
- Cyclical time encoding

### Improvements
- Automatic removal of incomplete rows
- ML-ready datasets generated