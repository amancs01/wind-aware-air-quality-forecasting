from dotenv import load_dotenv
import os

load_dotenv()
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

STATIONS_FILE = METADATA_DIR / "stations_metadata.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"

VALIDATION_DIR = REPORTS_DIR / "validation"

PROFILING_DIR = REPORTS_DIR / "profiling"

FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_DIR = DATA_DIR / "processed"

MERGED_DIR = PROCESSED_DIR / "merged"

FINAL_DIR = PROCESSED_DIR / "final"

TRIMMED_DIR = PROCESSED_DIR / "trimmed"

FEATURED_DIR = PROCESSED_DIR / "featured"

PREPARED_DIR = PROCESSED_DIR/ "prepared"

# ----------------------------
# Weather Configuration
# ----------------------------

START_YEAR = 2021
END_YEAR = 2026

TIMEZONE = "Asia/Kathmandu"

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

# -----------------------------
# OpenAQ Configuration
# -----------------------------

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
COUNTRY = "NP"
CITY = "Kathmandu"

# Study Area
CENTER_LATITUDE = 27.7172
CENTER_LONGITUDE = 85.3240
SEARCH_RADIUS = 25000  # meters

STATIONS_METADATA_FILE = DATA_DIR / "metadata" / "stations_metadata.csv"

REQUEST_TIMEOUT = 30