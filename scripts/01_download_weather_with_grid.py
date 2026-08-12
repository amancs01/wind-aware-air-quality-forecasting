import pandas as pd
from downloaders.base_downloader import BaseDownloader
from clients.http_client import get_json
from logger import logger
from datetime import datetime
from config import (
    STATIONS_FILE,
    WEATHER_DIR,
    START_YEAR,
    END_YEAR,
    TIMEZONE,
    WEATHER_VARIABLES,
)
from utils import (
    sanitize_filename,
    create_station_folder,
)

class WeatherDownloader(BaseDownloader):

    def __init__(self):
        super().__init__(WEATHER_DIR)

    def run(self):
        self.ensure_output_directory()
        stations = pd.read_csv(STATIONS_FILE)
        
        for _, row in stations.iterrows(): 
            station = row["station"]
            lat = row["latitude"]
            lon = row["longitude"]
            station_folder = create_station_folder(
                self.output_dir,
                station
            )

            logger.info("=" * 50)
            logger.info(f"Station: {station}")
            logger.info("=" * 50)
            for year in range(START_YEAR, END_YEAR + 1):
                current_year = datetime.now().year

                if year == current_year:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    end_date = f"{year}-12-31"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": f"{year}-01-01",
                    "end_date": end_date,
                    "hourly": ",".join(WEATHER_VARIABLES),
                    "timezone": TIMEZONE
                }

                safe_station = sanitize_filename(station)

                output_file = station_folder / f"{safe_station}_{year}.csv"

                if self.file_exists(output_file):
                    self.skip_file(output_file)
                    continue

                logger.info(f"⬇ Downloading {year}...")
                try:
                    data = get_json(
                        url="https://archive-api.open-meteo.com/v1/archive",
                        params=params,
                        timeout=30,
                    )
                    if "hourly" not in data or data["hourly"] is None:
                        self.record_failure(
                            f"{station} ({year}): No hourly data returned"
                        )
                        continue
                except Exception as e:

                    self.record_failure(
                        f"{station} ({year}) : {e}"
                    )

                    continue
                        
                df = pd.DataFrame({
                    "timestamp": data["hourly"]["time"],
                    "temperature": data["hourly"]["temperature_2m"],
                    "humidity": data["hourly"]["relative_humidity_2m"],
                    "dew_point": data["hourly"]["dew_point_2m"],
                    "pressure": data["hourly"]["surface_pressure"],
                    "wind_speed": data["hourly"]["wind_speed_10m"],
                    "wind_direction": data["hourly"]["wind_direction_10m"],
                    "precipitation" : data["hourly"]["precipitation"]
                })

                df["station"] = station

                # Coordinates requested from Open-Meteo (the OpenAQ station).
                df["latitude"] = lat
                df["longitude"] = lon

                # Coordinates actually selected by Open-Meteo. The API
                # documents these as the centre of the weather grid cell used
                # to generate the returned series.
                df["weather_grid_latitude"] = data.get("latitude")
                df["weather_grid_longitude"] = data.get("longitude")
                df["weather_grid_elevation"] = data.get("elevation")
                
                self.save_dataframe(
                    df,
                    output_file,
                )
                logger.info(f"✓ Saved {station} ({year})")

if __name__ == "__main__":
    downloader = WeatherDownloader()
    downloader.run()
    downloader.summary()
        
    