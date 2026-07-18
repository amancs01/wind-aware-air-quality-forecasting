import pandas as pd
from logger import logger
from config import (
    TIMESTAMP_VALIDATION_DIR,
    TRIMMED_DIR
)

class TimestampValidator:

    def __init__(self):
        TIMESTAMP_VALIDATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.summary = []
        self.invalid_dir = (
            TIMESTAMP_VALIDATION_DIR / "invalid_rows"
        )
        self.invalid_dir.mkdir(parents=True, exist_ok=True)

    def validate_station(self, csv_file):
        logger.info(
            f"Validating {csv_file.stem}"
        )
        df = pd.read_csv(csv_file)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )
        df["time_gap"] = (
            df["timestamp"]
            .diff()
        )
        expected_gap = pd.Timedelta(hours=1)

        valid = (
            df["time_gap"] ==
            expected_gap
        ).sum()
        invalid = (
            df["time_gap"] !=
            expected_gap
        ).sum()
        largest_gap = (
            df["time_gap"]
            .max()
        )
        largest_gap_hours = (
            largest_gap.total_seconds() / 3600
            if pd.notna(largest_gap)
            else 0
        )
        gap_distribution = (
            df["time_gap"]
            .value_counts()
            .sort_index()
        )
        gap_distribution.to_csv(
            self.invalid_dir /
            f"{csv_file.stem}_gap_distribution.csv"
        )
        valid_percentage = (
            (valid / (len(df) - 1)) * 100
            if len(df) > 1
            else 0
        )
        median_gap_hours = (
            df["time_gap"]
            .dropna()
            .median()
            .total_seconds() / 3600
            if not df["time_gap"].dropna().empty
            else 0
        )
        invalid_rows = df[
            (
                df["time_gap"] !=
                expected_gap
            )
            &
            (
                df["time_gap"]
                .notna()
            )
        ]
        invalid_rows.to_csv(

            self.invalid_dir /
            f"{csv_file.stem}_invalid.csv",

            index=False,

        )

        self.summary.append({

            "station": csv_file.stem,

            "rows": len(df),

            "valid_hourly_gaps": valid,

            "invalid_gaps": invalid - 1,

            "largest_gap_hours": largest_gap_hours,

            "valid_gap_percent": round(valid_percentage, 2),

            "median_gap_hours": median_gap_hours
        })

    def save_summary(self):
        summary_df = pd.DataFrame(
            self.summary
        )

        summary_df.to_csv(

            TIMESTAMP_VALIDATION_DIR /
            "timestamp_validation_summary.csv",

            index=False,

        )
        logger.info("=" * 50)

        logger.info(
            f"Stations checked: {len(self.summary)}"
        )

        logger.info(
            f"Total invalid gaps: "
            f"{summary_df['invalid_gaps'].sum()}"
        )

        logger.info("=" * 50)

    def run(self):

        csv_files = sorted(
            TRIMMED_DIR.glob("*.csv")
        )

        logger.info(
            f"Validating {len(csv_files)} stations..."
        )

        for csv_file in csv_files:

            self.validate_station(
                csv_file
            )

        self.save_summary()
