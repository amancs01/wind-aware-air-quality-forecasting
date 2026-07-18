# Data Sources

This document describes each dataset used in the project, how it is collected, and known limitations.

| Source     | Endpoint | Features           | Years     | Purpose             |
| ---------- | -------- | ------------------- | --------- | ------------------- |
| [OpenAQ](https://openaq.org/) v3 API   | `api.openaq.org/v3` | PM1, PM2.5 | 2021–2026 (sparse; see below) | Air quality |
| [Open-Meteo](https://open-meteo.com/) Archive API | `archive-api.open-meteo.com/v1/archive` | Temperature, humidity, dew point, pressure, wind speed/direction, precipitation | 2021–2026 | Meteorological data |

Study area: stations within `SEARCH_RADIUS` (25 km) of central Kathmandu (`27.7172, 85.3240`), configured in `scripts/config.py`.

## OpenAQ (Air Quality)

**Access**: Requires a free API key (`OPENAQ_API_KEY` in `.env`). Retrieval flow is Location → Sensor → Measurements; only sensors reporting PM2.5 are used (`scripts/00_discover_stations.py`).

**Implemented**
- Station discovery within the study radius
- PM2.5 sensor identification per station
- Monthly-partitioned measurement retrieval (`scripts/02_download_air_quality.py`)
- Retry mechanism with exponential backoff

**Known issues**
- Yearly requests return `HTTP 408`; some months exceed 1000 records, so month-level pagination is required (12 requests per station-year).
- Some measurements contain missing PM2.5 values (nulls), retained for future imputation work.

**Coverage**

Most Nepal stations on OpenAQ are provided by GD Labs, and most GD Labs stations only report consistent measurements starting in late 2025. As a result, air-quality coverage is far shorter than weather coverage for most stations (roughly 8–10 months vs. ~5 years) — Embassy Kathmandu is the notable exception with denser historical coverage. This is handled in preprocessing by:

- Merging weather and PM2.5 on aligned timestamps.
- Trimming leading periods where no PM2.5 measurement exists (`scripts/07_trim_data.py`).
- Retaining later gaps in the record for future imputation experiments.

## Open-Meteo (Weather)

**Access**: No API key required.

**Implemented**
- Per-station, per-year hourly retrieval (`scripts/01_download_weather.py`) for: `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`.
- Downloads skip years already saved on disk, so re-running the script is safe.
- The current year is downloaded up to today's date rather than a fixed year-end.