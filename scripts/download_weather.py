import requests
import pandas as pd
from config import (
    STATIONS_FILE,
    WEATHER_DIR,
    START_YEAR,
    END_YEAR,
    TIMEZONE,
    WEATHER_VARIABLES,
)
from utils import (
    load_stations,
    create_station_folder,
    print_header,
    print_summary,
)

# Read station list
stations = load_stations()

downloaded = 0
skipped = 0
failed = 0

# Create output folder
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

for _, row in stations.iterrows():

    station = row["station"]
    lat = row["latitude"]
    lon = row["longitude"]
    
    print_header(station)

    station_folder = create_station_folder(
        WEATHER_DIR,
        station
    )

    for year in range(START_YEAR, END_YEAR + 1):
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": TIMEZONE
        }

        output_file = station_folder / f"{station}_{year}.csv"

        if output_file.exists():
            print(f"✓ {year} already exists")
            skipped += 1
            continue

        print(f"⬇ Downloading {year}...")
        

        try:

            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=30
            )

        except Exception as e:

            print(f"❌ Error downloading {station} {year}: {e}")
            failed += 1
            continue
        
        if response.status_code != 200:
            failed += 1
            print(f"❌ Failed ({response.status_code})")
            continue

        if response.status_code == 200:

            data = response.json()

            df = pd.DataFrame({
                "timestamp": data["hourly"]["time"],
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "dew_point": data["hourly"]["dew_point_2m"],
                "pressure": data["hourly"]["surface_pressure"],
                "wind_speed": data["hourly"]["wind_speed_10m"],
                "wind_direction": data["hourly"]["wind_direction_10m"]
            })

            df["station"] = station
            df["latitude"] = lat
            df["longitude"] = lon
            
            df.to_csv(output_file, index=False)
            downloaded += 1
            
            print(f"✅ Saved: {output_file}")

print_summary(
    downloaded,
    skipped,
    failed
)