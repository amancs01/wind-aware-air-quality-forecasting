from pathlib import Path

import pandas as pd

from config import (
    MERGED_DIR,
    REPORTS_DIR,
)

from logger import logger

class MergedDataAnalyzer:

    def __init__(self):

        self.results = []
    
    def analyze_file(self, csv_file):

        df = pd.read_csv(csv_file)

        total_rows = len(df)

        missing_pm25 = df["pm2_5"].isna().sum()

        duplicate_timestamps = df["timestamp"].duplicated().sum()

        missing_weather = (
            df.drop(columns=["timestamp", "pm2_5"])
            .isna()
            .sum()
            .sum()
        )

        self.results.append({

            "station": csv_file.stem,

            "rows": total_rows,

            "missing_pm25": missing_pm25,

            "missing_pm25_percent":
                round(missing_pm25 / total_rows * 100, 2),

            "missing_weather": missing_weather,

            "duplicate_timestamps": duplicate_timestamps,

            "first_timestamp": df["timestamp"].min(),

            "last_timestamp": df["timestamp"].max(),

        })

    def run(self):

        for csv_file in sorted(MERGED_DIR.glob("*.csv")):

            logger.info(f"Analyzing {csv_file.stem}")

            self.analyze_file(csv_file)

        report = pd.DataFrame(self.results)

        output = REPORTS_DIR / "merged_dataset_analysis.csv"

        report.to_csv(
            output,
            index=False,
        )

        logger.info(f"Saved {output}")