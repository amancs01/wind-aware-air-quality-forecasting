# Development Log

## 2026-07-02

### Completed

- Created repository
- Config file
- Weather downloader
- Resumable downloads
- Shared utilities
# Development Log

## Day 1

Major achievements

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

Engineering lessons

- APIs often require partitioned downloads instead of large requests.
- Retry logic should not live inside application code.
- Git files should always be UTF-8.
- Large data pipelines benefit from reusable infrastructure.