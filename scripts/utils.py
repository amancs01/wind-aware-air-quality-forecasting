from pathlib import Path
import re

import pandas as pd

from config import STATIONS_FILE


def load_stations():
    """
    Load weather monitoring stations.
    """
    return pd.read_csv(STATIONS_FILE)


def sanitize_filename(name: str) -> str:
    """
    Convert a string into a filesystem-safe filename.
    """

    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def create_station_folder(base_dir: Path, station: str):
    """
    Create (if necessary) and return the station folder.
    """

    safe_station = sanitize_filename(station)

    station_folder = Path(base_dir) / safe_station

    station_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return station_folder


def get_headers(api_key):
    """
    Return OpenAQ request headers.
    """

    return {
        "X-API-Key": api_key,
        "Accept": "application/json",
    }