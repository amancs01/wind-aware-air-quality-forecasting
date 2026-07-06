import pandas as pd
from downloaders.base_downloader import BaseDownloader
from clients.http_client import get_json
from logger import logger
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
    sanitize_filename,
    create_station_folder,
    print_header,
    print_summary,
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
            
            print_header(station)

            station_folder = create_station_folder(
                WEATHER_DIR,
                station
            )
            logger.info("=" * 50)
            logger.info(f"Station: {station}")
            logger.info("=" * 50)
            for year in range(START_YEAR, END_YEAR + 1):
                
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                    "hourly": ",".join(WEATHER_VARIABLES),
                    "timezone": TIMEZONE
                }

                safe_station = sanitize_filename(station)

                output_file = station_folder / f"{safe_station}_{year}.csv"

                if self.file_exists(output_file):
                    print(f"✓ {year} already exists")
                    self.skip_file(output_file)
                    continue

                logger.info(f"⬇ Downloading {year}...")
                try:
                    data = get_json(
                        url="https://archive-api.open-meteo.com/v1/archive",
                        params=params,
                        timeout=30,
                    )
                
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
                    "wind_direction": data["hourly"]["wind_direction_10m"]
                })

                df["station"] = station
                df["latitude"] = lat
                df["longitude"] = lon
                
                self.save_dataframe(
                    df,
                    output_file,
                )
                logger.info(f"✓ Saved {station} ({year})")

if __name__ == "__main__":
    downloader = WeatherDownloader()
    downloader.run()
    downloader.summary()
        
    