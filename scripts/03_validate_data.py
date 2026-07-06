from pathlib import Path

import pandas as pd

from logger import logger
from config import (
    WEATHER_DIR,
    AIR_QUALITY_DIR,
    VALIDATION_DIR,
)


class DataValidator:

    def __init__(self):
        self.weather_results = []

        self.air_quality_results = []

        VALIDATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def validate_weather(self):

        logger.info("=" * 60)
        logger.info("Validating Weather Dataset")
        logger.info("=" * 60)

        for station_folder in WEATHER_DIR.iterdir():

            if not station_folder.is_dir():
                continue

            for csv_file in station_folder.glob("*.csv"):

                try:

                    df = pd.read_csv(csv_file)

                except Exception as e:

                    self.weather_results.append({

                        "station": station_folder.name,
                        "file": csv_file.name,
                        "status": "Corrupted",
                        "rows": 0,
                        "duplicates": 0,
                        "missing": 0,
                        "first_timestamp": None,
                        "last_timestamp": None,
                    })

                    continue

                result = {

                    "station": station_folder.name,
                    "file": csv_file.name,
                    "status": "OK",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "duplicates": df["timestamp"].duplicated().sum(),
                    "missing_values": int(df.isna().sum().sum()),
                    "first_timestamp": (
                        df["timestamp"].min()
                        if len(df)
                        else None
                    ),
                    "last_timestamp": (
                        df["timestamp"].max()
                        if len(df)
                        else None
                    ),
                    "sorted": df["timestamp"].is_monotonic_increasing,
                }

                self.weather_results.append(result)
        
        logger.info(
            f"Checked {len(self.weather_results)} weather files."
        )

    def save_weather_report(self):

        df = pd.DataFrame(self.weather_results)

        output_file = VALIDATION_DIR / "weather_validation.csv"

        df.to_csv(
            output_file,
            index=False,
        )

        logger.info(f"Saved {output_file}")
        logger.info("=" * 50)
        logger.info("Weather Validation Summary")
        logger.info("=" * 50)

        logger.info(f"Files checked : {len(df)}")

        logger.info(
            f"Files with duplicates : {(df['duplicates'] > 0).sum()}"
        )

        logger.info(
            f"Files not sorted : {(~df['sorted']).sum()}"
        )

        logger.info(
            f"Files with missing values : {(df['missing_values'] > 0).sum()}"
        )

    def validate_air_quality(self):

        logger.info("=" * 60)
        logger.info("Validating Air Quality Dataset")
        logger.info("=" * 60)

        for station_folder in AIR_QUALITY_DIR.iterdir():

            if not station_folder.is_dir():
                continue

            for csv_file in station_folder.glob("*.csv"):

                try:

                    df = pd.read_csv(csv_file)

                except Exception:

                    self.air_quality_results.append({

                        "station": station_folder.name,
                        "file": csv_file.name,
                        "status": "Corrupted",
                        "rows": 0,
                        "columns": 0,
                        "duplicates": 0,
                        "missing_values": 0,
                        "first_timestamp": None,
                        "last_timestamp": None,
                        "sorted": False,

                    })

                    continue

                result = {

                    "station": station_folder.name,
                    "file": csv_file.name,

                    "status": "OK",

                    "rows": len(df),

                    "columns": len(df.columns),

                    "duplicates": df["timestamp"].duplicated().sum(),

                    "missing_values": int(df.isna().sum().sum()),

                    "first_timestamp": (
                        df["timestamp"].min()
                        if len(df)
                        else None
                    ),

                    "last_timestamp": (
                        df["timestamp"].max()
                        if len(df)
                        else None
                    ),

                    "sorted": df["timestamp"].is_monotonic_increasing,

                }

                self.air_quality_results.append(result)

        logger.info(
            f"Checked {len(self.air_quality_results)} air quality files."
        )

    def save_air_quality_report(self):

        df = pd.DataFrame(self.air_quality_results)

        output_file = VALIDATION_DIR / "air_quality_validation.csv"

        df.to_csv(
            output_file,
            index=False,
        )

        logger.info(f"Saved {output_file}")

        logger.info("=" * 50)
        logger.info("Air Quality Validation Summary")
        logger.info("=" * 50)

        logger.info(f"Files checked : {len(df)}")

        logger.info(
            f"Files with duplicates : {(df['duplicates'] > 0).sum()}"
        )

        logger.info(
            f"Files not sorted : {(~df['sorted']).sum()}"
        )

        logger.info(
            f"Files with missing values : {(df['missing_values'] > 0).sum()}"
        )

    def run(self):

        self.validate_weather()

        self.save_weather_report()

        self.validate_air_quality()

        self.save_air_quality_report()


if __name__ == "__main__":

    validator = DataValidator()

    validator.run()