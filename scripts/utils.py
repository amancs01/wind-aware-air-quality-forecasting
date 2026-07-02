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