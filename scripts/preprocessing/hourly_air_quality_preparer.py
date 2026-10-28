from pathlib import Path

import pandas as pd

from config import (
    AIR_QUALITY_HOURLY_DIR,
    AIR_QUALITY_HOURLY_RAW_DIR,
    STATIONS_METADATA_FILE,
)
from logger import logger
from utils import station_dataset_name


class HourlyAirQualityPreparer:

    def __init__(
        self,
        input_dir=AIR_QUALITY_HOURLY_RAW_DIR,
        output_dir=AIR_QUALITY_HOURLY_DIR,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.results = []

    def load_station_data(self, station_folder):
        dfs = []

        for csv_file in sorted(station_folder.glob("*.csv")):
            df = pd.read_csv(csv_file)
            df["source_file"] = csv_file.name
            dfs.append(df)

        if dfs:
            return pd.concat(
                dfs,
                ignore_index=True,
            )

        return pd.DataFrame()

    @staticmethod
    def _local_clock_timestamp(series):
        parsed = pd.to_datetime(
            series,
            errors="coerce",
        )

        return parsed.dt.tz_localize(None)

    def _check_duplicates(self, station, df):
        duplicate_mask = df["timestamp"].duplicated(
            keep=False,
        )

        if not duplicate_mask.any():
            return

        duplicates = df.loc[
            duplicate_mask,
            [
                "timestamp",
                "pm2_5",
                "datetime_from_local",
                "datetime_to_local",
                "source_file",
            ],
        ].sort_values("timestamp")

        output_file = (
            self.output_dir /
            f"{station}_duplicate_canonical_timestamps.csv"
        )

        duplicates.to_csv(
            output_file,
            index=False,
        )

        raise ValueError(
            f"{station}: duplicate canonical timestamps found. "
            f"Details saved to {output_file}"
        )

    def prepare_station(self, station_folder):
        station = station_folder.name
        raw_df = self.load_station_data(station_folder)

        if raw_df.empty:
            logger.warning(f"No raw hourly AQ data for {station}")
            return

        required_columns = [
            "pm2_5",
            "datetime_from_local",
            "datetime_to_local",
            "datetime_from_utc",
            "datetime_to_utc",
        ]
        missing_columns = [
            column for column in required_columns
            if column not in raw_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{station}: missing required columns {missing_columns}"
            )

        raw_df["datetime_to_local_parsed"] = pd.to_datetime(
            raw_df["datetime_to_local"],
            errors="coerce",
        )
        canonical_df = raw_df[
            raw_df["datetime_to_local_parsed"].dt.minute == 0
        ].copy()

        canonical_df["timestamp"] = self._local_clock_timestamp(
            canonical_df["datetime_to_local"]
        )
        canonical_df = canonical_df[
            canonical_df["timestamp"].notna()
        ].copy()
        canonical_df["timestamp"] = canonical_df[
            "timestamp"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")

        self._check_duplicates(
            station,
            canonical_df,
        )

        preferred_columns = [
            "timestamp",
            "station",
            "sensor_id",
            "parameter",
            "parameter_units",
            "pm2_5",
            "datetime_from_local",
            "datetime_to_local",
            "datetime_from_utc",
            "datetime_to_utc",
            "period_label",
            "period_interval",
            "coverage_expected_count",
            "coverage_observed_count",
            "coverage_percent",
            "coverage_percent_complete",
            "coverage_expected_interval",
            "coverage_observed_interval",
            "summary_min",
            "summary_max",
            "summary_avg",
            "summary_median",
            "summary_sd",
            "source_file",
        ]
        output_columns = [
            column for column in preferred_columns
            if column in canonical_df.columns
        ]

        canonical_df = canonical_df[output_columns].sort_values(
            "timestamp"
        )

        if not canonical_df["timestamp"].is_monotonic_increasing:
            raise ValueError(f"{station}: timestamps are not sorted")

        output_file = self.output_dir / f"{station}.csv"

        canonical_df.to_csv(
            output_file,
            index=False,
        )

        self.results.append({
            "station": station,
            "raw_rows": len(raw_df),
            "canonical_rows": len(canonical_df),
            "dropped_non_clock_hour_rows": len(raw_df) - len(canonical_df),
            "first_timestamp": (
                canonical_df["timestamp"].min()
                if len(canonical_df)
                else None
            ),
            "last_timestamp": (
                canonical_df["timestamp"].max()
                if len(canonical_df)
                else None
            ),
        })

        logger.info(f"Saved canonical hourly AQ for {station}")

    def run(self, stations_filter=None):
        if not self.input_dir.exists():
            raise FileNotFoundError(self.input_dir)

        requested = set(stations_filter or [])
        metadata = pd.read_csv(STATIONS_METADATA_FILE)
        duplicated_station_names = set(
            metadata.loc[
                metadata["station"].duplicated(keep=False),
                "station",
            ]
        )

        for _, row in metadata.iterrows():
            if pd.isna(row.get("pm25_sensor_id")):
                continue

            dataset_name = station_dataset_name(
                row["station"],
                sensor_id=row["pm25_sensor_id"],
                require_sensor_id=(
                    row["station"] in duplicated_station_names
                ),
            )

            if requested and (
                dataset_name not in requested and
                row["station"] not in requested
            ):
                continue

            station_folder = self.input_dir / dataset_name

            if not station_folder.exists():
                logger.warning(
                    f"No raw hourly AQ folder for {dataset_name}"
                )
                continue

            self.prepare_station(station_folder)

        return pd.DataFrame(self.results)
