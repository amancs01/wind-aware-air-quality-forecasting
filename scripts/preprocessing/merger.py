from pathlib import Path

import pandas as pd

from logger import logger

from config import (
    WEATHER_DIR,
    AIR_QUALITY_DIR,
    MERGED_DIR,
    PROFILING_DIR,
)
class DataMerger:

    def __init__(self):

        MERGED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
    
    def load_station_data(self, station_folder):

        dfs = []

        for csv_file in sorted(station_folder.glob("*.csv")):

            df = pd.read_csv(csv_file)

            dfs.append(df)

        if dfs:

            return pd.concat(
                dfs,
                ignore_index=True,
            )

        return pd.DataFrame()
    
    def merge_station(
        self,
        station_name,
    ):
        weather_folder = WEATHER_DIR / station_name

        air_folder = AIR_QUALITY_DIR / station_name

        weather_df = self.load_station_data(weather_folder)

        air_df = self.load_station_data(air_folder)
        weather_df["timestamp"] = (
            pd.to_datetime(weather_df["timestamp"])
            .dt.tz_localize("Asia/Kathmandu")
        )

        air_df["timestamp"] = (
            pd.to_datetime(weather_df["timestamp"])
            .dt.tz_convert("Asia/Kathmandu")
        )
        if weather_df.empty:

            logger.warning(
                f"No weather data for {station_name}"
            )

            return

        if air_df.empty:

            logger.warning(
                f"No air quality data for {station_name}"
            )

            return
        
        merged_df = pd.merge(
            weather_df,
            air_df,
            on="timestamp",
            how="left",
        )

        print(merged_df.head())

        print(merged_df.shape)

        print(merged_df.columns)

        output_file = MERGED_DIR / f"{station_name}.csv"

        merged_df.to_csv(
            output_file,
            index=False,
        )

        logger.info(f"Saved {output_file}")
     
    def run(self):

        coverage = pd.read_csv(
            PROFILING_DIR/"station_coverage.csv"
        )

        coverage = coverage[
            coverage["air_quality_files"] > 0
        ]

        logger.info(
            f"Merging {len(coverage)} stations..."
        )

        for _, row in coverage.iterrows():

            station = row["station"]

            logger.info(f"Processing {station}")

            self.merge_station(station)

