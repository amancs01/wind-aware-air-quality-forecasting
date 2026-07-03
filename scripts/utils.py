from pathlib import Path
import pandas as pd

from config import STATIONS_FILE


def load_stations():
    """
    Load all monitoring stations.
    """
    return pd.read_csv(STATIONS_FILE)


def create_station_folder(base_dir: Path, station: str):
    """
    Create station folder if it doesn't exist.
    """
    station_folder = base_dir / station
    station_folder.mkdir(parents=True, exist_ok=True)
    return station_folder


def print_header(station: str):
    """
    Print station header.
    """
    print("\n" + "=" * 60)
    print(f"Station : {station}")
    print("=" * 60)


def print_summary(downloaded: int, skipped: int, failed: int):
    """
    Print download summary.
    """
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print(f"Downloaded : {downloaded}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {failed}")

def get_headers(api_key):
    return {
        "X-API-Key": api_key,
        "Accept": "application/json"
    }

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path):
    """Check if a file already exists."""
    return Path(path).exists()


def save_dataframe(df: pd.DataFrame, output_path):
    """Save dataframe as CSV."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)

import re

def sanitize_filename(name: str) -> str:
    """
    Convert a string into a filesystem-safe name.
    """

    # Replace invalid Windows filename characters
    name = re.sub(r'[<>:"/\\|?*]', "_", name)

    # Replace multiple spaces with one
    name = re.sub(r"\s+", " ", name)

    return name.strip()