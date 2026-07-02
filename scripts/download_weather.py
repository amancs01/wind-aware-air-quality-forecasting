import os
import requests
import pandas as pd

YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025
]

# Read station list
stations = pd.read_csv("../data/metadata/stations.csv")

# Create output folder
os.makedirs("../data/raw/weather", exist_ok=True)

for _, row in stations.iterrows():

    station = row["station"]
    lat = row["latitude"]
    lon = row["longitude"]
    
    for year in YEARS:
        print(f"Downloading weather for {station}...")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m"
            ]),
            "timezone": "Asia/Kathmandu"
        }
        
        try:

            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=30
            )

        except Exception as e:

            print(e)
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

            station_folder = f"../data/raw/weather/{station}"
            os.makedirs(station_folder, exist_ok=True)

            output_file = (
                f"{station_folder}/"
                f"{station}_{year}.csv"
            )
            df.to_csv(output_file, index=False)

            print(f"✅ Saved: {output_file}")

        else:
            print(f"❌ Failed for {station}: {response.status_code}")