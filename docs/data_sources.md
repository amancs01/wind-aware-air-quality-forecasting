# Data Sources

This document contains all datasets used in the project.

| Source           | Features          | Years     | Purpose             |
| ---------------- | ----------------- | --------- | ------------------- |
| OpenAQ           | PM1, PM2.5        | 2021–2025 | Air quality         |
| Open-Meteo       | Weather variables | 2021–2025 | Meteorological data |

## OpenAQ

Completed

- Station discovery
- Sensor discovery
- PM2.5 sensor identification
- Monthly data retrieval
- Retry mechanism

Known Issues

- Some months exceed 1000 records.
- Month-level pagination is required.
- Some measurements contain missing PM2.5 values.

Observation:
Most GD Labs stations only contain consistent measurements beginning in late 2025.

Implication:
Weather data spans 2021–2026, while air-quality coverage is much shorter for most stations.

Handling:
- Merge weather and PM2.5 using aligned timestamps.
- Remove leading periods where PM2.5 does not exist.
- Retain later missing observations for future imputation experiments.