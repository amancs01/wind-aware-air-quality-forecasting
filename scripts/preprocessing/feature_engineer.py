from pathlib import Path

import pandas as pd

import numpy as np

from logger import logger

from config import (
    TRIMMED_DIR,
    FEATURED_DIR,
)


class FeatureEngineer:

    def __init__(self):

        FEATURED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def engineer_features(self, csv_file):
        df = pd.read_csv(csv_file)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df["hour"] = df["timestamp"].dt.hour

        df["day"] = df["timestamp"].dt.day

        df["month"] = df["timestamp"].dt.month

        df["weekday"] = df["timestamp"].dt.dayofweek

        df["lag_1"] = df["pm2_5"].shift(1)

        df["lag_3"] = df["pm2_5"].shift(3)

        df["lag_6"] = df["pm2_5"].shift(6)

        df["lag_12"] = df["pm2_5"].shift(12)

        df["lag_24"] = df["pm2_5"].shift(24)

        df["rolling_mean_3"] = (
            df["pm2_5"]
            .rolling(window=3)
            .mean()
        )

        df["rolling_mean_6"] = (
            df["pm2_5"]
            .rolling(window=6)
            .mean()
        )

        df["rolling_mean_24"] = (
            df["pm2_5"]
            .rolling(window=24)
            .mean()
        )

        df["rolling_std_3"] = (
            df["pm2_5"]
            .rolling(window=3)
            .std()
        )

        df["rolling_std_6"] = (
            df["pm2_5"]
            .rolling(window=6)
            .std()
        )

        df["rolling_std_24"] = (
            df["pm2_5"]
            .rolling(window=24)
            .std()
        )

        wind_rad = np.deg2rad(df["wind_direction"])

        df["wind_u"] = (
            df["wind_speed"] *
            np.cos(wind_rad)
        )

        df["wind_v"] = (
            df["wind_speed"] *
            np.sin(wind_rad)
        )

        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )

        df["month_sin"] = np.sin(
            2 * np.pi * df["month"] / 12
        )

        df["month_cos"] = np.cos(
            2 * np.pi * df["month"] / 12
        )
        
        output = FEATURED_DIR / csv_file.name

        df.to_csv(
            output,
            index=False,
        )

        logger.info(
            f"Saved {output}"
        )

    def run(self):

        for csv_file in sorted(TRIMMED_DIR.glob("*.csv")):

            logger.info(
                f"Engineering {csv_file.stem}"
            )

            self.engineer_features(csv_file)