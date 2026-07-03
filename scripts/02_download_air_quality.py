import requests
import pandas as pd

from api import(
    fetch_all_measurements
)
from config import (
    STATIONS_METADATA_FILE,
    START_YEAR,
    END_YEAR,
    OPENAQ_BASE_URL,
    OPENAQ_API_KEY,
    REQUEST_TIMEOUT,
)

from utils import (
    get_headers,
    ensure_directory,
    file_exists,
    save_dataframe,
)

stations = pd.read_csv(STATIONS_METADATA_FILE)

print(f"Found {len(stations)} stations.")

downloaded = 0
skipped = 0
failed = 0

from pathlib import Path
AIR_QUALITY_DIR = Path("../data/raw/air_quality")
ensure_directory(AIR_QUALITY_DIR)

""""
for _, row in stations.iterrows():

    from utils import sanitize_filename
    station = row["station"]
    safe_station = sanitize_filename(station)

    pm25_sensor = row["pm25_sensor_id"]
    pm1_sensor = row["pm1_sensor_id"]

    print(f"\nStation: {safe_station}")

    station_folder = AIR_QUALITY_DIR / safe_station

    ensure_directory(station_folder)

    for year in range(START_YEAR, END_YEAR + 1):

        output_file = station_folder / f"{safe_station}_{year}.csv"

        print(output_file)
"""
station = "Embassy Kathmandu"
sensor_id = 7710      # use the value from stations_metadata.csv

year = 2023

url = f"{OPENAQ_BASE_URL}/sensors/{int(sensor_id)}/measurements"

params = {
    "datetime_from": f"{year}-01-01T00:00:00Z",
    "datetime_to": f"{year}-12-31T23:59:59Z",
    "limit": 1000,
}

response = requests.get(
    url,
    headers=get_headers(OPENAQ_API_KEY),
    params=params,
    timeout=REQUEST_TIMEOUT,
)

print(response.status_code)

if response.status_code != 200:
    print(response.text)
    exit()

results = fetch_all_measurements(
    sensor_id=sensor_id,
    year=2023,
    base_url=OPENAQ_BASE_URL,
    headers=get_headers(OPENAQ_API_KEY),
    timeout=REQUEST_TIMEOUT,
)
print("First timestamp:")
print(results[0]["period"]["datetimeFrom"]["local"])

print("Last timestamp:")
print(results[-1]["period"]["datetimeFrom"]["local"])

print("Total records:")
print(len(results))
print(results[0])
rows = []

for measurement in results:

    rows.append({

        "timestamp": measurement["period"]["datetimeFrom"]["local"],

        "pm2_5": measurement["value"]

    })

df = pd.DataFrame(rows)

print(df.head())

print(df.shape)

meta = response.json()["meta"]
print(meta)
duplicates = df["timestamp"].duplicated().sum()
print(f"Duplicate timestamps: {duplicates}")

print(response.links)
print(len(results))