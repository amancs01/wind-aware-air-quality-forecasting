# Development Log

A chronological record of how the project was built, kept alongside `changelog.md` (which tracks *what* changed) to capture *how* and *why* decisions were made along the way.

## Phase 1 — Foundations

- Created the repository and initial project structure.
- Built the central `config.py` for paths and constants.
- Implemented the weather downloader with resumable downloads (skips files already on disk).
- Built shared filesystem utilities (`utils.py`).
- Refactored the weather downloader to avoid unnecessary re-downloads.

## Phase 2 — Air Quality API Integration

- Investigated the OpenAQ API and learned the station → sensor → measurement retrieval flow.
- Implemented a first yearly-download prototype.
- Hit API timeout limitations on large yearly requests (`HTTP 408`).
- Redesigned the downloader to use monthly partitions instead.
- Added retry logic and investigated pagination behavior.
- Verified that timestamps contain no duplicates.
- Planned a reusable HTTP client architecture to share retry/backoff logic across downloaders.

## Phase 3 — Validation & Profiling

Built an automated validation pipeline checking:
- Missing values
- Duplicate timestamps
- File sizes
- Dataset summaries

Implemented dataset profiling, generating:
- Weather summary
- Air quality summary
- Station coverage
- Merged dataset analysis

## Phase 4 — Preprocessing

Implemented the preprocessing pipeline as reusable modules:
- `merger.py`
- `analyzer.py`
- `trimmer.py`
- `feature_engineer.py`

Resolved a timestamp mismatch between Open-Meteo and OpenAQ by converting both to compatible `datetime` representations before merging.

**Engineering lessons learned**
- APIs often require partitioned downloads instead of one large request.
- Retry logic shouldn't live inside application code — it belongs in a shared client.
- Git-tracked files should always be UTF-8 (a `.gitignore` encoding bug came from UTF-16 LE).
- Large data pipelines benefit from investing in reusable infrastructure early.

## Milestone — Data Engineering Complete

Completed the full preprocessing pipeline:
- Weather downloader
- Air quality downloader
- Validation
- Profiling
- Dataset merging
- Timestamp alignment
- Dataset trimming
- Feature engineering
- Dataset preparation

**Engineered features**: time features, lag features, rolling statistics, wind vector components, cyclical time encoding.

Prepared datasets were confirmed ready for machine learning.

## Phase 5 — Machine Learning

- Refactored the ML architecture around a reusable `BaseModel` class.
- Implemented the persistence baseline.
- Verified the preprocessing pipeline end-to-end before starting model training.
- Implemented Linear Regression as the first learned model.

## Next Up

- Random Forest and XGBoost baselines
- LSTM / Transformer sequence models
- Wind-weighted graph construction and the GAT-GRU model