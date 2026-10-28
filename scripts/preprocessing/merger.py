from pathlib import Path

import pandas as pd

from logger import logger

from config import (
    AIR_QUALITY_HOURLY_DIR,
    MERGED_DIR,
    PROFILING_DIR,
    WEATHER_DIR,
)
from utils import sanitize_filename


class DataMerger:

    AQ_COLUMNS = [
        "timestamp",
        "sensor_id",
        "pm2_5",
        "datetime_from_local",
        "datetime_to_local",
        "datetime_from_utc",
        "datetime_to_utc",
        "coverage_expected_count",
        "coverage_observed_count",
        "coverage_percent",
    ]

    def __init__(self):

        MERGED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _parse_local_timestamp(series):
        return pd.to_datetime(
            series,
            errors="coerce",
        ).dt.tz_localize(None)

    def load_weather_data(self, station):
        station_folder = WEATHER_DIR / sanitize_filename(station)
        dfs = []

        for csv_file in sorted(station_folder.glob("*.csv")):
            dfs.append(pd.read_csv(csv_file))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(
            dfs,
            ignore_index=True,
        )

    def load_air_quality_data(self, dataset_name):
        csv_file = AIR_QUALITY_HOURLY_DIR / f"{dataset_name}.csv"

        if not csv_file.exists():
            return pd.DataFrame()

        return pd.read_csv(csv_file)

    def _validate_weather(self, dataset_name, weather_df):
        if weather_df.empty:
            raise ValueError(f"{dataset_name}: missing weather data")

        duplicate_rows = weather_df["timestamp"].duplicated(
            keep=False,
        ).sum()

        if duplicate_rows:
            raise ValueError(
                f"{dataset_name}: weather has {duplicate_rows} "
                "duplicate timestamp rows"
            )

    def _validate_air_quality(self, dataset_name, air_df):
        if air_df.empty:
            raise ValueError(
                f"{dataset_name}: missing canonical hourly AQ data"
            )

        duplicate_rows = air_df["timestamp"].duplicated(
            keep=False,
        ).sum()

        if duplicate_rows:
            raise ValueError(
                f"{dataset_name}: canonical AQ has {duplicate_rows} "
                "duplicate timestamp rows"
            )

        non_clock_rows = (
            (air_df["timestamp"].dt.minute != 0) |
            (air_df["timestamp"].dt.second != 0) |
            air_df["timestamp"].isna()
        ).sum()

        if non_clock_rows:
            raise ValueError(
                f"{dataset_name}: canonical AQ has {non_clock_rows} "
                "non-clock-hour timestamp rows"
            )

    def merge_dataset(
        self,
        row,
    ):
        dataset_name = row["dataset_name"]
        station = row["station"]

        weather_df = self.load_weather_data(station)
        air_df = self.load_air_quality_data(dataset_name)

        weather_df["timestamp"] = self._parse_local_timestamp(
            weather_df["timestamp"]
        )
        air_df["timestamp"] = self._parse_local_timestamp(
            air_df["timestamp"]
        )

        weather_df = weather_df.sort_values("timestamp")
        air_df = air_df.sort_values("timestamp")

        self._validate_weather(
            dataset_name,
            weather_df,
        )
        self._validate_air_quality(
            dataset_name,
            air_df,
        )

        available_aq_columns = [
            column for column in self.AQ_COLUMNS
            if column in air_df.columns
        ]
        air_df = air_df[available_aq_columns]

        merged_df = pd.merge(
            weather_df,
            air_df,
            on="timestamp",
            how="left",
        )

        merged_df.insert(
            1,
            "dataset_name",
            dataset_name,
        )

        if len(merged_df) != len(weather_df):
            raise ValueError(
                f"{dataset_name}: merged rows {len(merged_df)} do not "
                f"match weather rows {len(weather_df)}"
            )

        duplicate_rows = merged_df["timestamp"].duplicated(
            keep=False,
        ).sum()

        if duplicate_rows:
            raise ValueError(
                f"{dataset_name}: merged data has {duplicate_rows} "
                "duplicate timestamp rows"
            )

        output_file = MERGED_DIR / f"{dataset_name}.csv"

        merged_df.to_csv(
            output_file,
            index=False,
        )

        logger.info(
            f"Saved {output_file} "
            f"({len(merged_df):,} rows, "
            f"{merged_df['pm2_5'].notna().sum():,} valid PM2.5)"
        )

    def run(self):

        coverage = pd.read_csv(
            PROFILING_DIR / "station_coverage.csv"
        )

        coverage = coverage[
            coverage["air_quality_files"] > 0
        ]

        logger.info(
            f"Merging {len(coverage)} datasets..."
        )

        for _, row in coverage.iterrows():

            logger.info(
                f"Processing {row['dataset_name']}"
            )

            self.merge_dataset(row)
