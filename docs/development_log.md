# Development Log

### Completed

- Created repository
- Config file
- Weather downloader
- Resumable downloads
- Shared utilities
- Completed reusable utility module.
- Refactored weather downloader.
- Prevented unnecessary weather downloads.
- Investigated OpenAQ API.
- Learned OpenAQ station → sensor → measurement workflow.
- Implemented yearly downloader prototype.
- Identified API timeout limitations.
- Redesigned downloader to use monthly partitions.
- Added retry logic.
- Investigated pagination behaviour.
- Verified timestamps contain no duplicates.
- Planned reusable HTTP client architecture.

## Validation

Created automated validation pipeline.

Checks include:
- Missing values
- Duplicate timestamps
- File sizes
- Dataset summaries

---

## Profiling

Implemented dataset profiling.

Generated:
- Weather summary
- Air quality summary
- Station coverage
- Merged dataset analysis

---

## Preprocessing

Implemented preprocessing pipeline.

Modules:
- merger.py
- analyzer.py
- trimmer.py
- feature_engineer.py

Resolved timestamp mismatch between Open-Meteo and OpenAQ by converting timestamps to compatible datetime representations before merging.

Engineering lessons

- APIs often require partitioned downloads instead of large requests.
- Retry logic should not live inside application code.
- Git files should always be UTF-8.
- Large data pipelines benefit from reusable infrastructure.