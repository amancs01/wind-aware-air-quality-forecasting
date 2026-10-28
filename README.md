# Wind-Aware Air Quality Forecasting

Final Year Major Project — an explainable, wind-aware forecasting system for PM2.5 air pollution in the Kathmandu Valley.

## Objective

Historical PM2.5 forecasting models typically treat air-quality stations as independent time series and ignore how wind physically transports pollution between locations. This project builds a spatio-temporal forecasting pipeline that incorporates wind direction and speed as first-class features, with the long-term goal of training a Graph Attention + GRU network (GAT-GRU) that treats monitoring stations as nodes in a graph connected along the prevailing wind field.

The current stage of the project focuses on the data engineering pipeline and classical ML baselines; the graph neural network is the next milestone (see [Roadmap](#roadmap)).

## How It Works

1. **Discover** air-quality monitoring stations near Kathmandu via the OpenAQ API.
2. **Download** historical hourly weather (Open-Meteo) and PM2.5 measurements (OpenAQ) per station.
3. **Validate and profile** the raw data to catch missing values, duplicate timestamps, and coverage gaps.
4. **Merge** weather and air-quality data on aligned timestamps, then **trim** leading periods with no PM2.5 readings.
5. **Engineer features**: lag features, rolling statistics, cyclical time encodings, and physical wind vector components.
6. **Split** each station's data into train / validation / test sets and verify the split.
7. **Train and evaluate** forecasting models — starting with a persistence baseline and linear regression — against a shared set of metrics (MAE, RMSE, R²).

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline diagram.

## Wind Convention

Open-Meteo wind direction is treated as meteorological direction: the
direction FROM which wind blows, clockwise from north. Feature
engineering converts it into physical components:

```text
wind_u = eastward component
wind_v = northward component
```

Wind speed and components remain in km/h, matching the current
Open-Meteo default used by the downloader. Future directed graph edges
must compare source-to-target station bearings with pollution transport
direction, not raw meteorological direction:

```text
transport_direction = (wind_direction + 180) % 360
```

## Tech Stack

| Layer               | Tools |
| ------------------- | ----- |
| Language             | Python 3.10+ |
| Data handling        | Pandas, NumPy |
| Machine learning     | scikit-learn (baseline models), PyTorch (planned — GAT-GRU) |
| Data sources         | [OpenAQ](https://openaq.org/) (air quality), [Open-Meteo](https://open-meteo.com/) (weather) |
| Visualization        | Matplotlib |
| API layer (planned)  | FastAPI |
| Dashboard (planned)  | React |

## Project Structure

```
wind-aware-air-quality-forecasting/
├── docs/                        # Design docs, research notes, changelog
├── scripts/
│   ├── 00_discover_stations.py     # Find OpenAQ stations near Kathmandu
│   ├── 01_download_weather.py      # Download hourly weather (Open-Meteo)
│   ├── 02_download_air_quality.py  # Download PM2.5 measurements (OpenAQ)
│   ├── 03_validate_data.py         # Validate raw data quality
│   ├── 04_profile_dataset.py       # Generate dataset profiling reports
│   ├── 05_preprocess_data.py       # Timestamp normalization
│   ├── 06_analyze_merge_data.py    # Merge weather + air quality
│   ├── 07_trim_data.py             # Trim leading rows with no PM2.5
│   ├── 07b_validate_timestamps.py  # Verify merged timestamps
│   ├── 08_feature_engineering.py   # Lag, rolling, cyclical, wind features
│   ├── 09_prepare_dataset.py       # Build the final ML-ready dataset
│   ├── 10_split_dataset.py         # Train / validation / test split
│   ├── 11_verify_split.py          # Sanity-check the split
│   ├── 12_persistence_baseline.py  # Naive persistence baseline model
│   ├── 13_linear_regression.py     # Linear regression model
│   ├── 14_feature_correlation.py   # Feature correlation analysis + plots
│   ├── run_pipeline.py             # Runs all steps above in order
│   ├── clients/, downloaders/      # Reusable HTTP client + downloader base class
│   ├── models/                     # BaseModel + model implementations
│   ├── preprocessing/              # Merging, trimming, feature engineering logic
│   ├── validation/                 # Timestamp validation logic
│   ├── config.py                   # Central paths, constants, feature lists
│   ├── logger.py                   # Shared logging setup
│   └── utils.py                    # Filesystem / filename helpers
├── requirements.txt
└── .env.example
```

Running the pipeline creates `data/`, `reports/`, `models/`, and `results/` directories at the project root (all git-ignored — see [Data & Outputs](#data--outputs)).

## Getting Started

### Prerequisites

- Python 3.10 or later
- A free [OpenAQ API key](https://explore.openaq.org/register)

### Installation

```bash
git clone https://github.com/<your-username>/wind-aware-air-quality-forecasting.git
cd wind-aware-air-quality-forecasting

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Copy the example environment file and add your OpenAQ API key:

```bash
cp .env.example .env
```

```
OPENAQ_API_KEY=your_openaq_api_key_here
```

All other settings (study area coordinates, date range, weather variables, feature list) are defined centrally in [`scripts/config.py`](scripts/config.py).

### Running the Pipeline

Each script is numbered and meant to be run in order from the `scripts/` directory. To run everything end-to-end:

```bash
cd scripts
python run_pipeline.py
```

This executes steps `00` through `14` sequentially and stops immediately if any step fails.

To run an individual stage instead (useful during development):

```bash
cd scripts
python 00_discover_stations.py
python 01_download_weather.py
python 02_download_air_quality.py
# ...and so on
```

> Downloads are resumable — both downloaders skip files that already exist on disk, so re-running a step is safe.

## Data & Outputs

Running the pipeline populates the following directories (all git-ignored, regenerated locally):

| Directory   | Contents |
| ----------- | -------- |
| `data/raw/`       | Raw per-station weather and air-quality CSVs |
| `data/processed/` | Merged, trimmed, feature-engineered, and split datasets |
| `data/metadata/`  | Discovered station metadata |
| `reports/`        | Validation and profiling reports, figures |
| `results/`        | Per-model metrics (MAE, RMSE, R²) and analysis outputs |

See [`docs/data_sources.md`](docs/data_sources.md) for details on each data source, known API quirks, and coverage limitations.

## Documentation

| Document | Description |
| -------- | ----------- |
| [`docs/architecture.md`](docs/architecture.md) | End-to-end pipeline diagram |
| [`docs/data_sources.md`](docs/data_sources.md) | Data sources, coverage, and known issues |
| [`docs/preprocessing_plan.md`](docs/preprocessing_plan.md) | Preprocessing and feature engineering plan |
| [`docs/model_plan.md`](docs/model_plan.md) | Modeling roadmap and progress |
| [`docs/research_notes.md`](docs/research_notes.md) | API findings and engineering decisions |
| [`docs/development_log.md`](docs/development_log.md) | Chronological development log |
| [`docs/changelog.md`](docs/changelog.md) | Notable changes by category |

## Roadmap

- [x] Data collection (OpenAQ + Open-Meteo)
- [x] Validation, profiling, merging, trimming
- [x] Feature engineering (lag, rolling, cyclical, physical wind vectors)
- [x] Train / validation / test split
- [x] Persistence baseline
- [x] Linear regression baseline
- [ ] Random Forest / XGBoost baselines
- [ ] LSTM / Transformer sequence models
- [ ] Graph construction from station geography + wind field
- [ ] Wind-aware GAT-GRU model
- [ ] Explainability (attention weights, feature attribution)
- [ ] FastAPI serving layer
- [ ] React dashboard

## License

No license has been specified yet for this project.
