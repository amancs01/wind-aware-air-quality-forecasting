import pandas as pd

from api import fetch_all_measurements
from config import (
    STATIONS_METADATA_FILE,
    START_YEAR,
    END_YEAR,
    OPENAQ_BASE_URL,
    OPENAQ_API_KEY,
    REQUEST_TIMEOUT,
)

from logger import logger

from utils import (
    create_station_folder,
    get_headers,
    sanitize_filename,
)

from downloaders.base_downloader import BaseDownloader


class AirQualityDownloader(BaseDownloader):

    def __init__(self):
        super().__init__("../data/raw/air_quality")

    def run(self):

        self.ensure_output_directory()

        stations = pd.read_csv(STATIONS_METADATA_FILE)

        logger.info(f"Found {len(stations)} stations.")

        headers = get_headers(OPENAQ_API_KEY)

        for _, row in stations.iterrows():

            station = row["station"]
            safe_station = sanitize_filename(station)

            sensor_id = row["pm25_sensor_id"]

            if pd.isna(sensor_id):
                self.record_failure(f"{station}: Missing PM2.5 sensor")
                continue

            logger.info("=" * 50)
            logger.info(f"Station: {station}")
            logger.info("=" * 50)

            station_folder = create_station_folder(
                self.output_dir,
                station,
            )

            for year in range(START_YEAR, END_YEAR + 1):

                output_file = station_folder / f"{safe_station}_{year}.csv"

                if self.file_exists(output_file):
                    self.skip_file(output_file)
                    continue

                logger.info(f"⬇ Downloading {year}...")

                try:

                    results = fetch_all_measurements(
                        sensor_id=int(sensor_id),
                        year=year,
                        base_url=OPENAQ_BASE_URL,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT,
                    )

                except Exception as e:

                    self.record_failure(
                        f"{station} ({year}) : {e}"
                    )

                    continue

                if not results:

                    self.record_failure(
                        f"{station} ({year}) : No data returned"
                    )

                    continue

                rows = []

                for measurement in results:

                    rows.append({
                        "timestamp": measurement["period"]["datetimeFrom"]["local"],
                        "pm2_5": measurement["value"],
                    })

                df = pd.DataFrame(rows)

                if df.empty:

                    self.record_failure(
                        f"{station} ({year}) : Empty dataframe"
                    )

                    continue

                self.save_dataframe(
                    df,
                    output_file,
                )

                logger.info(f"✓ Saved {station} ({year})")


if __name__ == "__main__":

    downloader = AirQualityDownloader()

    downloader.run()

    downloader.summary()