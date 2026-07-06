from pathlib import Path

import pandas as pd

from logger import logger

from config import (
    WEATHER_DIR,
    AIR_QUALITY_DIR,
    REPORTS_DIR,
    PROFILING_DIR,
)

from downloaders.base_downloader import BaseDownloader

class DatasetProfiler:

    def __init__(self):

        self.weather_summary = []

        self.air_summary = []

        PROFILING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def profile_weather(self):
        
        logger.info("=" * 60)
        logger.info("Profiling Weather Dataset")
        logger.info("=" * 60)

        for station_folder in WEATHER_DIR.iterdir():

            if not station_folder.is_dir():
                continue

            for csv_file in station_folder.glob("*.csv"):

                df = pd.read_csv(csv_file)

                self.weather_summary.append({

                    "station": station_folder.name,

                    "year": csv_file.stem.split("_")[-1],

                    "rows": len(df),

                    "columns": len(df.columns),

                    "first_timestamp": df["timestamp"].min(),

                    "last_timestamp": df["timestamp"].max(),

                    "missing_values": int(df.isna().sum().sum())

                })

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
        logger.info(f"Stations       : {df['station'].nunique()}")
        logger.info(f"Total rows     : {df['rows'].sum():,}")
    
    def profile_air_quality(self):

        logger.info("=" * 60)
        logger.info("Profiling Air Quality Dataset")
        logger.info("=" * 60)

        for station_folder in AIR_QUALITY_DIR.iterdir():

            if not station_folder.is_dir():
                continue

            for csv_file in station_folder.glob("*.csv"):

                df = pd.read_csv(csv_file)

                self.air_summary.append({

                    "station": station_folder.name,

                    "year": csv_file.stem.split("_")[-1],

                    "rows": len(df),

                    "columns": len(df.columns),

                    "valid_pm25": df["pm2_5"].notna().sum(),

                    "missing_pm25": df["pm2_5"].isna().sum(),

                    "first_timestamp": df["timestamp"].min(),

                    "last_timestamp": df["timestamp"].max()

                })

        logger.info(
            f"Air quality files: {len(self.air_summary)}"
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
        logger.info("Air Quality Profile Summary")
        logger.info("=" * 50)

        logger.info(f"Files profiled : {len(df)}")
        logger.info(f"Stations       : {df['station'].nunique()}")
        logger.info(f"Total rows     : {df['rows'].sum():,}")
        logger.info(f"Valid PM2.5    : {df['valid_pm25'].sum():,}")

    def build_station_coverage(self):
        weather_df = pd.DataFrame(self.weather_summary)
        air_df = pd.DataFrame(self.air_summary)
        coverage = []

        stations = sorted(
            set(weather_df["station"])
            | set(air_df["station"])
        )

        for station in stations:

            weather_station = weather_df[
                weather_df["station"] == station
            ]

            air_station = air_df[
                air_df["station"] == station
            ]

            weather_rows = weather_station["rows"].sum()
            air_quality_rows = air_station["rows"].sum()
            valid_pm25 = air_station["valid_pm25"].sum()
            missing_pm25 = air_station["missing_pm25"].sum()
            coverage_percent = (
                valid_pm25 / weather_rows * 100
                if weather_rows > 0
                else 0
            )
            coverage.append({
                "station": station,
                "weather_files": len(weather_station),
                "air_quality_files": len(air_station),
                "weather_rows": weather_rows,
                "air_quality_rows": air_quality_rows,
                "valid_pm25": valid_pm25,
                "missing_pm25": missing_pm25,
                "coverage_percent": round(coverage_percent, 2),
                "first_weather":
                    weather_station["first_timestamp"].min(),
                "last_weather":
                    weather_station["last_timestamp"].max(),
                "first_air":
                    air_station["first_timestamp"].min(),
                "last_air":
                    air_station["last_timestamp"].max(),
            })
        coverage_df = pd.DataFrame(coverage)

        output_file = (
            PROFILING_DIR /
            "station_coverage.csv"
        )

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