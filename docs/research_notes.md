# Research Notes

## OpenAQ API Findings

Station hierarchy

Location
    ↓
Sensor
    ↓
Measurements

Observations

- Measurements are retrieved from sensors rather than locations.
- Monthly downloads are significantly more reliable than yearly downloads.
- API returns HTTP 408 for large yearly requests.
- Retry with exponential backoff improves stability.
- Some months exceed 1000 measurements.
- Month-level pagination will be required.
- Timestamps contain no duplicates.
- API includes missing PM2.5 values represented as null.

## Engineering Decisions

Decision 1

Weather downloader should never redownload existing datasets.

Decision 2

Shared filesystem utilities belong in `utils.py`.

Decision 3

Network retry logic should be reusable across all APIs.

Decision 4

Future networking will be centralized inside a reusable HTTP client.