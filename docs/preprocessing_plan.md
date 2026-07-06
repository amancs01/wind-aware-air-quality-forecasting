Raw Data
↓
Validation
↓
Profiling
↓
Merge
↓
Timestamp Alignment
↓
Trim Leading Missing Labels
↓
Feature Engineering
    - Time Features
    - Lag Features
    - Rolling Features
↓
Scaling
↓
Train/Test Split

Open-Meteo provides hourly observations.

OpenAQ measurements are timestamped approximately every hour with minute offsets.

During preprocessing, timestamps are converted to datetime objects, normalized, and aligned before merging.