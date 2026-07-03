# Changelog

## v0.1.0

- Initial project structure
- Added weather downloader
- Added configuration system
- Added shared utilities

---

## Version 0.2.0 (Current Development)

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