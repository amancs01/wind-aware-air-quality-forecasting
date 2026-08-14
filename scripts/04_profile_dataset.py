import pandas as pd

from logger import logger

from config import (
    AIR_QUALITY_HOURLY_DIR,
    PROFILING_DIR,
    STATIONS_METADATA_FILE,
    WEATHER_DIR,
)
from utils import sanitize_filename, station_dataset_name


class DatasetProfiler:

    def __init__(self):

        self.weather_summary = []
        self.air_summary = []
        self.coverage_rows = []

        PROFILING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _parse_local_timestamp(series):
        return pd.to_datetime(
            series,
            errors="coerce",
        ).dt.tz_localize(None)

    def _load_weather_timestamps(self, station, record_profile=False):
        weather_folder = WEATHER_DIR / sanitize_filename(station)
        dfs = []

        if not weather_folder.exists():
            return pd.DataFrame(columns=["timestamp"])

        for csv_file in sorted(weather_folder.glob("*.csv")):
            df = pd.read_csv(csv_file)
            df["timestamp"] = self._parse_local_timestamp(
                df["timestamp"]
            )
            dfs.append(df)

            if record_profile:
                self.weather_summary.append({
                    "station": station,
                    "weather_station_folder": weather_folder.name,
                    "file": csv_file.name,
                    "year": csv_file.stem.split("_")[-1],
                    "rows": len(df),
                    "unique_timestamps": df["timestamp"].nunique(),
                    "columns": len(df.columns),
                    "first_timestamp": df["timestamp"].min(),
                    "last_timestamp": df["timestamp"].max(),
                    "missing_values": int(df.isna().sum().sum()),
                })

        if dfs:
            return pd.concat(
                dfs,
                ignore_index=True,
            )

        return pd.DataFrame(columns=["timestamp"])

    def profile_weather(self):

        logger.info("=" * 60)
        logger.info("Profiling Weather Dataset")
        logger.info("=" * 60)

        metadata = pd.read_csv(STATIONS_METADATA_FILE)

        for station in sorted(metadata["station"].unique()):
            self._load_weather_timestamps(
                station,
                record_profile=True,
            )

        logger.info(
            f"Weather files: {len(self.weather_summary)}"
        )

    def save_weather_profile(self):

        df = pd.DataFrame(self.weather_summary)

        output_file = PROFILING_DIR / "weather_profile.csv"

        df.to_csv(
            output_file,
            index=False,
        )

        logger.info(f"Saved {output_file}")

        logger.info("=" * 50)
        logger.info("Weather Profile Summary")
        logger.info("=" * 50)
        logger.info(f"Files profiled : {len(df)}")

        if not df.empty:
            logger.info(f"Stations       : {df['station'].nunique()}")
            logger.info(f"Total rows     : {df['rows'].sum():,}")

    def _profile_air_quality_file(
        self,
        dataset_name,
        station,
        sensor_id,
        record_profile=False,
    ):
        csv_file = AIR_QUALITY_HOURLY_DIR / f"{dataset_name}.csv"

        if not csv_file.exists():
            return pd.DataFrame(columns=["timestamp", "pm2_5"])

        df = pd.read_csv(csv_file)
        df["timestamp"] = self._parse_local_timestamp(
            df["timestamp"]
        )

        if record_profile:
            self.air_summary.append({
                "dataset_name": dataset_name,
                "station": station,
                "sensor_id": int(sensor_id),
                "file": csv_file.name,
                "rows": len(df),
                "unique_timestamps": df["timestamp"].nunique(),
                "columns": len(df.columns),
                "valid_pm25": int(df["pm2_5"].notna().sum()),
                "missing_pm25": int(df["pm2_5"].isna().sum()),
                "first_timestamp": df["timestamp"].min(),
                "last_timestamp": df["timestamp"].max(),
            })

        return df

    def profile_air_quality(self):

        logger.info("=" * 60)
        logger.info("Profiling Canonical Hourly Air Quality Dataset")
        logger.info("=" * 60)

        metadata = pd.read_csv(STATIONS_METADATA_FILE)
        duplicated_station_names = set(
            metadata.loc[
                metadata["station"].duplicated(keep=False),
                "station",
            ]
        )

        for _, row in metadata.iterrows():
            if pd.isna(row["pm25_sensor_id"]):
                continue

            dataset_name = station_dataset_name(
                row["station"],
                sensor_id=row["pm25_sensor_id"],
                require_sensor_id=(
                    row["station"] in duplicated_station_names
                ),
            )

            self._profile_air_quality_file(
                dataset_name=dataset_name,
                station=row["station"],
                sensor_id=row["pm25_sensor_id"],
                record_profile=True,
            )

        logger.info(
            f"Canonical hourly air quality files: {len(self.air_summary)}"
        )

    def save_air_quality_profile(self):

        df = pd.DataFrame(self.air_summary)

        output_file = PROFILING_DIR / "air_quality_profile.csv"

        df.to_csv(
            output_file,
            index=False,
        )

        logger.info(f"Saved {output_file}")

        logger.info("=" * 50)
        logger.info("Canonical Hourly Air Quality Profile Summary")
        logger.info("=" * 50)

        logger.info(f"Files profiled : {len(df)}")

        if not df.empty:
            logger.info(f"Datasets       : {df['dataset_name'].nunique()}")
            logger.info(f"Total rows     : {df['rows'].sum():,}")
            logger.info(f"Valid PM2.5    : {df['valid_pm25'].sum():,}")

    def build_station_coverage(self):
        metadata = pd.read_csv(STATIONS_METADATA_FILE)
        duplicated_station_names = set(
            metadata.loc[
                metadata["station"].duplicated(keep=False),
                "station",
            ]
        )

        for _, row in metadata.iterrows():
            if pd.isna(row["pm25_sensor_id"]):
                continue

            station = row["station"]
            sensor_id = int(row["pm25_sensor_id"])
            dataset_name = station_dataset_name(
                station,
                sensor_id=sensor_id,
                require_sensor_id=station in duplicated_station_names,
            )

            weather_df = self._load_weather_timestamps(station)
            air_df = self._profile_air_quality_file(
                dataset_name=dataset_name,
                station=station,
                sensor_id=sensor_id,
            )

            weather_timestamps = pd.Index(
                weather_df["timestamp"]
                .dropna()
                .drop_duplicates()
            )
            air_timestamps = pd.Index(
                air_df["timestamp"]
                .dropna()
                .drop_duplicates()
            )
            valid_air_timestamps = pd.Index(
                air_df.loc[
                    air_df["pm2_5"].notna(),
                    "timestamp",
                ]
                .dropna()
                .drop_duplicates()
            )

            matching_aq_hours = len(
                air_timestamps.intersection(weather_timestamps)
            )
            valid_pm25_hours = len(
                valid_air_timestamps.intersection(weather_timestamps)
            )
            weather_rows = len(weather_timestamps)
            missing_pm25_hours = max(
                weather_rows - valid_pm25_hours,
                0,
            )
            coverage_percent = (
                valid_pm25_hours / weather_rows * 100
                if weather_rows > 0
                else 0
            )

            self.coverage_rows.append({
                "dataset_name": dataset_name,
                "station": station,
                "sensor_id": sensor_id,
                "weather_station_folder": sanitize_filename(station),
                "air_quality_file": f"{dataset_name}.csv",
                "weather_files": (
                    len(list(
                        (
                            WEATHER_DIR /
                            sanitize_filename(station)
                        ).glob("*.csv")
                    ))
                    if (
                        WEATHER_DIR /
                        sanitize_filename(station)
                    ).exists()
                    else 0
                ),
                "air_quality_files": int(
                    (AIR_QUALITY_HOURLY_DIR / f"{dataset_name}.csv")
                    .exists()
                ),
                "weather_rows": weather_rows,
                "canonical_aq_rows": len(air_timestamps),
                "matching_aq_hours": matching_aq_hours,
                "valid_pm25_hours": valid_pm25_hours,
                "missing_pm25_hours": missing_pm25_hours,
                "coverage_percent": round(coverage_percent, 2),
                "first_weather": (
                    weather_timestamps.min()
                    if len(weather_timestamps)
                    else None
                ),
                "last_weather": (
                    weather_timestamps.max()
                    if len(weather_timestamps)
                    else None
                ),
                "first_air": (
                    air_timestamps.min()
                    if len(air_timestamps)
                    else None
                ),
                "last_air": (
                    air_timestamps.max()
                    if len(air_timestamps)
                    else None
                ),
            })

        coverage_df = pd.DataFrame(self.coverage_rows)

        output_file = PROFILING_DIR / "station_coverage.csv"

        coverage_df.to_csv(
            output_file,
            index=False,
        )

        logger.info(f"Saved {output_file}")

    def run(self):
        self.profile_weather()
        self.save_weather_profile()
        self.profile_air_quality()
        self.save_air_quality_profile()
        self.build_station_coverage()


if __name__ == "__main__":

    profiler = DatasetProfiler()

    profiler.run()
