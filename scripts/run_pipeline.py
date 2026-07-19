import subprocess
import sys

STEPS = [
    # "00_discover_stations.py",
    # "01_download_weather.py",
    # "02_download_air_quality.py",
    # "03_validate_data.py",
    # "04_profile_dataset.py",
    # "05_preprocess_data.py",
    # "06_analyze_merge_data.py",
    # "07_trim_data.py",
    # "07b_validate_timestamps.py",
    # "08_feature_engineering.py",
    # "09_prepare_dataset.py",
    # "10_split_dataset.py",
    # "11_verify_split.py",
    # "12_persistence_baseline.py",
    # "13_linear_regression.py",
    "14_feature_correlation.py",
]

for step in STEPS:
    print(f"\n{'='*60}\nRunning {step}\n{'='*60}")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"FAILED at {step}, stopping.")
        sys.exit(1)