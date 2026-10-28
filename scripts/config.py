import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
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

AIR_QUALITY_HOURLY_RAW_DIR = RAW_DIR / "air_quality_hourly"

METADATA_DIR = DATA_DIR / "metadata"

STATIONS_FILE = METADATA_DIR / "stations_metadata.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"

VALIDATION_DIR = REPORTS_DIR / "validation"

PROFILING_DIR = REPORTS_DIR / "profiling"

FIGURES_DIR = REPORTS_DIR / "figures"

TABLES_DIR = REPORTS_DIR / "tables"

PROCESSED_DIR = DATA_DIR / "processed"

AIR_QUALITY_HOURLY_DIR = PROCESSED_DIR / "air_quality_hourly"

MERGED_DIR = PROCESSED_DIR / "merged"

FINAL_DIR = PROCESSED_DIR / "final"

TRIMMED_DIR = PROCESSED_DIR / "trimmed"

FEATURED_DIR = PROCESSED_DIR / "featured"

PREPARED_DIR = PROCESSED_DIR/ "prepared"

SPLIT_DIR = PROCESSED_DIR / "split"

TRAIN_DIR = SPLIT_DIR / "train"

ML_VALIDATION_DIR = SPLIT_DIR / "validation"

TEST_DIR = SPLIT_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"

PERSISTENCE_RESULTS_DIR = RESULTS_DIR / "persistence"

LINEAR_RESULTS_DIR = RESULTS_DIR / "linear_regression"

RANDOM_FOREST_RESULTS_DIR = RESULTS_DIR / "random_forest"

XGBOOST_RESULTS_DIR = RESULTS_DIR / "xgboost"

LSTM_RESULTS_DIR = RESULTS_DIR / "lstm"

TRANSFORMER_RESULTS_DIR = RESULTS_DIR / "transformer"

FEATURE_ANALYSIS_DIR = RESULTS_DIR / "feature_analysis"

# added part
# ----------------------------
# Graph Processing
# ----------------------------

GRAPH_DIR = PROCESSED_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

STATION_MAPPING_FILE = METADATA_DIR / "station_mapping.csv"

# Distance
DISTANCE_MATRIX_FILE = GRAPH_DIR / "distance_matrix.csv"
DISTANCE_EDGES_FILE = GRAPH_DIR / "distance_edges.csv"

# Bearing
BEARING_MATRIX_FILE = GRAPH_DIR / "bearing_matrix.csv"
BEARING_EDGES_FILE = GRAPH_DIR / "bearing_edges.csv"

# Static Graph
STATIC_GRAPH_FILE = GRAPH_DIR / "static_graph.csv"
ADJACENCY_MATRIX_FILE = GRAPH_DIR / "adjacency_matrix.csv"

# Dynamic Edge Weights
DYNAMIC_EDGE_FILE = GRAPH_DIR / "dynamic_edge_weights.csv"

# Graph Snapshots
GRAPH_SNAPSHOTS_DIR = GRAPH_DIR / "snapshots"
GRAPH_SNAPSHOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
# ----------------------------
# Graph Parameters
# ----------------------------

K_NEIGHBORS = 5
USE_KNN_GRAPH = True
MAX_EDGE_DISTANCE_KM = None

GRAPH_WINDOW_SIZE = 24
TARGET_HORIZON = 1
WIND_ALIGNMENT_THRESHOLD = 45   # degrees

# added end
TIMESTAMP_VALIDATION_DIR = RESULTS_DIR / "timestamp_validation"

TEMPORAL_FEATURE_VALIDATION_DIR = RESULTS_DIR / "temporal_feature_validation"

MIN_TRAINING_ROWS = 100

FEATURE_EXCLUDE_COLUMNS = [
    "timestamp",
    "station",
    "target_pm2_5",
]

MODEL_FEATURE_COLUMNS = [
    "pm2_5", "lag_6", "lag_24",
    "rolling_mean_6", "rolling_std_6",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "wind_u", "wind_v",
    "temperature", "humidity", "pressure", "dew_point",
]

LINEAR_BASELINE_ALPHA = 1000.0

RANDOM_FOREST_N_ESTIMATORS = 100

RANDOM_FOREST_MAX_DEPTH = 10

RANDOM_FOREST_MIN_SAMPLES_LEAF = 10

RANDOM_FOREST_MAX_FEATURES = 1.0

RANDOM_FOREST_RANDOM_STATE = 42

RANDOM_FOREST_N_JOBS = 1

XGBOOST_LEARNING_RATE = 0.1

XGBOOST_MAX_DEPTH = 3

XGBOOST_MIN_CHILD_WEIGHT = 5

XGBOOST_SUBSAMPLE = 0.8

XGBOOST_COLSAMPLE_BYTREE = 0.8

XGBOOST_REG_ALPHA = 0.0

XGBOOST_REG_LAMBDA = 1.0

XGBOOST_N_ESTIMATORS = 1000

XGBOOST_EARLY_STOPPING_ROUNDS = 50

XGBOOST_OBJECTIVE = "reg:squarederror"

XGBOOST_EVAL_METRIC = "rmse"

XGBOOST_TREE_METHOD = "hist"

XGBOOST_RANDOM_STATE = 42

XGBOOST_N_JOBS = 1

for directory in [

    # Reports
    DATA_DIR,
    RAW_DIR,
    WEATHER_DIR,
    AIR_QUALITY_DIR,
    AIR_QUALITY_HOURLY_RAW_DIR,
    METADATA_DIR,
    REPORTS_DIR,
    VALIDATION_DIR,
    PROFILING_DIR,
    FIGURES_DIR,
    TABLES_DIR,

    # Results
    PROCESSED_DIR,
    AIR_QUALITY_HOURLY_DIR,
    MERGED_DIR,
    FINAL_DIR,
    TRIMMED_DIR,
    FEATURED_DIR,
    PREPARED_DIR,
    SPLIT_DIR,
    TRAIN_DIR,
    ML_VALIDATION_DIR,
    TEST_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    PERSISTENCE_RESULTS_DIR,
    LINEAR_RESULTS_DIR,
    RANDOM_FOREST_RESULTS_DIR,
    XGBOOST_RESULTS_DIR,
    LSTM_RESULTS_DIR,
    TRANSFORMER_RESULTS_DIR,
    FEATURE_ANALYSIS_DIR,
    TIMESTAMP_VALIDATION_DIR,
    TEMPORAL_FEATURE_VALIDATION_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    "precipitation"
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
