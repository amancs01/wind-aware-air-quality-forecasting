import subprocess
import sys

STEPS = [
    # Data discovery/download stages are intentionally manual because they
    # touch remote APIs and large ignored datasets.
    # "00_discover_stations.py",
    # "01_download_weather.py",
    # Optional archival raw /measurements source; not used for modeling.
    # "02_download_air_quality.py",
    # "02b_download_hourly_air_quality.py",
    "02c_prepare_hourly_air_quality.py",
    "02d_validate_hourly_air_quality.py",
    "03_validate_data.py",
    "04_profile_dataset.py",
    "05_preprocess_data.py",
    "06_analyze_merge_data.py",
    "07_trim_data.py",
    "07b_validate_timestamps.py",
    "07c_validate_temporal_features.py",
    "08_feature_engineering.py",
    "09_prepare_dataset.py",
    "10_split_dataset.py",
    "11_verify_split.py",
    "12_persistence_baseline.py",
    "13_linear_regression.py",
    # "14_feature_correlation.py",
]

for step in STEPS:
    print(f"\n{'='*60}\nRunning {step}\n{'='*60}")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"FAILED at {step}, stopping.")
        sys.exit(1)
