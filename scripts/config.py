from pathlib import Path

# ----------------------------
# Project Paths
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

WEATHER_DIR = RAW_DIR / "weather"

AIR_QUALITY_DIR = RAW_DIR / "air_quality"

METADATA_DIR = DATA_DIR / "metadata"

STATIONS_FILE = METADATA_DIR / "stations.csv"

# ----------------------------
# Weather Configuration
# ----------------------------

START_YEAR = 2021
END_YEAR = 2025

TIMEZONE = "Asia/Kathmandu"

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]