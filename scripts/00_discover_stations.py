import requests
import pandas as pd

from config import (
    OPENAQ_BASE_URL,
    OPENAQ_API_KEY,
    REQUEST_TIMEOUT,
    CENTER_LATITUDE,
    CENTER_LONGITUDE,
    SEARCH_RADIUS,
    STATIONS_METADATA_FILE,
)

from utils import get_headers, save_table_sample


url = f"{OPENAQ_BASE_URL}/locations"

params = {
    "coordinates": f"{CENTER_LATITUDE},{CENTER_LONGITUDE}",
    "radius": SEARCH_RADIUS,
    "parameters_id": 2,
    "limit": 100,
}

print("Discovering air quality stations...")
response = requests.get(
    url,
    headers=get_headers(OPENAQ_API_KEY),
    params=params,
    timeout=REQUEST_TIMEOUT,
)

if response.status_code != 200:
    print(response.text)
    raise Exception("Failed to fetch stations.")


else:
    data = response.json()
    locations = data["results"]
    print(f"Found {len(locations)} locations.\n")

rows = []

for location in locations: 
    pm25_sensor = None
    pm1_sensor = None

    for sensor in location["sensors"]:

        parameter = sensor["parameter"]["name"].lower()

        if parameter == "pm25":
            pm25_sensor = sensor["id"]
        
        elif parameter == "pm1":
            pm1_sensor = sensor["id"]

    rows.append({

        "station": location["name"],

        "location_id": location["id"],

        "latitude": location["coordinates"]["latitude"],

        "longitude": location["coordinates"]["longitude"],

        "country": location["country"]["name"],

        "timezone": location["timezone"],

        "owner": location["owner"]["name"],

        "provider": location["provider"]["name"],

        "first_date": (
            location["datetimeFirst"]["utc"]
            if location["datetimeFirst"]
            else None
        ),

        "last_date": (
            location["datetimeLast"]["utc"]
            if location["datetimeLast"]
            else None
        ),

        "pm25_sensor_id": pm25_sensor,

        "pm1_sensor_id": pm1_sensor

    })

df = pd.DataFrame(rows)

df = df.sort_values("station")

df.to_csv(
    STATIONS_METADATA_FILE,
    index=False
)

save_table_sample(
    df,
    "stations_metadata_sample.csv"
)

print(df)

print()

print(f"Saved {len(df)} stations.")

print(f"Metadata saved to:")

print(STATIONS_METADATA_FILE)