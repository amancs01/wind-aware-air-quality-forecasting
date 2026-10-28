import argparse

import pandas as pd

from api import fetch_all_hourly_measurements
from config import (
    AIR_QUALITY_HOURLY_RAW_DIR,
    END_YEAR,
    OPENAQ_API_KEY,
    OPENAQ_BASE_URL,
    REQUEST_TIMEOUT,
    START_YEAR,
    STATIONS_METADATA_FILE,
)
from downloaders.base_downloader import BaseDownloader
from logger import logger
from utils import create_station_folder, get_headers, station_dataset_name


class HourlyAirQualityDownloader(BaseDownloader):

    def __init__(self):
        super().__init__(AIR_QUALITY_HOURLY_RAW_DIR)

    def _build_row(self, station, sensor_id, measurement):
        period = measurement.get("period", {})
        datetime_from = period.get("datetimeFrom", {})
        datetime_to = period.get("datetimeTo", {})
        coverage = measurement.get("coverage", {})
        parameter = measurement.get("parameter", {})
        summary = measurement.get("summary", {})

        return {
            "station": station,
            "sensor_id": int(sensor_id),
            "parameter": parameter.get("name"),
            "parameter_units": parameter.get("units"),
            "pm2_5": measurement.get("value"),
            "datetime_from_local": datetime_from.get("local"),
            "datetime_to_local": datetime_to.get("local"),
            "datetime_from_utc": datetime_from.get("utc"),
            "datetime_to_utc": datetime_to.get("utc"),
            "period_label": period.get("label"),
            "period_interval": period.get("interval"),
            "coverage_expected_count": coverage.get("expectedCount"),
            "coverage_observed_count": coverage.get("observedCount"),
            "coverage_percent": coverage.get("percentCoverage"),
            "coverage_percent_complete": coverage.get("percentComplete"),
            "coverage_expected_interval": coverage.get("expectedInterval"),
            "coverage_observed_interval": coverage.get("observedInterval"),
            "summary_min": summary.get("min"),
            "summary_max": summary.get("max"),
            "summary_avg": summary.get("avg"),
            "summary_median": summary.get("median"),
            "summary_sd": summary.get("sd"),
        }

    def run(
        self,
        stations_filter=None,
        start_year=START_YEAR,
        end_year=END_YEAR,
    ):

        self.ensure_output_directory()

        stations = pd.read_csv(STATIONS_METADATA_FILE)
        duplicated_station_names = set(
            stations.loc[
                stations["station"].duplicated(keep=False),
                "station",
            ]
        )

        if stations_filter:
            requested = set(stations_filter)
            stations = stations[stations["station"].isin(requested)]
            missing = sorted(requested - set(stations["station"]))

            for station in missing:
                self.record_failure(f"{station}: Station not found")

        logger.info(f"Found {len(stations)} stations.")

        headers = get_headers(OPENAQ_API_KEY)

        for _, row in stations.iterrows():

            station = row["station"]
            sensor_id = row["pm25_sensor_id"]
            station_start_year = max(
                start_year,
                pd.to_datetime(row["first_date"]).year,
            )
            station_end_year = min(
                end_year,
                pd.to_datetime(row["last_date"]).year,
            )

            if pd.isna(sensor_id):
                self.record_failure(f"{station}: Missing PM2.5 sensor")
                continue

            logger.info("=" * 50)
            logger.info(f"Station: {station}")
            logger.info("=" * 50)

            station_folder = create_station_folder(
                self.output_dir,
                station_dataset_name(
                    station,
                    sensor_id=sensor_id,
                    require_sensor_id=station in duplicated_station_names,
                ),
            )
            safe_station = station_folder.name

            for year in range(station_start_year, station_end_year + 1):

                output_file = station_folder / f"{safe_station}_{year}.csv"

                if self.file_exists(output_file):
                    self.skip_file(output_file)
                    continue

                logger.info(f"Downloading hourly AQ for {year}...")

                try:

                    results = fetch_all_hourly_measurements(
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
                        f"{station} ({year}) : No hourly data returned"
                    )

                    continue

                rows = [
                    self._build_row(station, sensor_id, measurement)
                    for measurement in results
                ]

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

                logger.info(f"Saved hourly AQ for {station} ({year})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download OpenAQ /hours PM2.5 data."
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        help="Optional exact station names to download.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    downloader = HourlyAirQualityDownloader()

    downloader.run(
        stations_filter=args.stations,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    downloader.summary()
