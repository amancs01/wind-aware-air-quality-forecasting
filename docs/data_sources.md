# Data Sources

This document contains all datasets used in the project.

| Source           | Features          | Years     | Purpose             |
| ---------------- | ----------------- | --------- | ------------------- |
| OpenAQ           | PM1, PM2.5        | 2021–2025 | Air quality         |
| Open-Meteo       | Weather variables | 2021–2025 | Meteorological data |
| ICIMOD (if used) | Validation        | -         | Cross-checking      |

## OpenAQ

Status:
In Progress

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