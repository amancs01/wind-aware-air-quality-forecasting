import pandas as pd

from logger import logger

from config import (
    TEMPORAL_FEATURE_VALIDATION_DIR,
    TRIMMED_DIR,
)


class TemporalFeatureValidator:

    LAG_HOURS = [1, 3, 6, 12, 24]

    ROLLING_WINDOWS = [3, 6, 24]

    EXPECTED_INTERVAL = pd.Timedelta(hours=1)

    REQUIRED_COLUMNS = [
        "timestamp",
        "pm2_5",
    ]

    def __init__(self):

        TEMPORAL_FEATURE_VALIDATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.summary = []

    def validate_station(self, csv_file):
        logger.info(
            f"Validating temporal features for {csv_file.stem}"
        )

        df = pd.read_csv(csv_file)

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{csv_file.name} is missing required columns: "
                f"{', '.join(missing_columns)}"
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        station_summary = {
            "station": csv_file.stem,
            "rows": len(df),
            "timestamps_sorted": df["timestamp"].is_monotonic_increasing,
            "duplicate_timestamp_rows": (
                df["timestamp"]
                .duplicated(keep=False)
                .sum()
            ),
            "invalid_adjacent_gaps": self._count_invalid_adjacent_gaps(df),
            "missing_pm25_rows": df["pm2_5"].isna().sum(),
        }

        station_summary.update(
            self._validate_lags(df)
        )

        station_summary.update(
            self._validate_rolling_windows(df)
        )

        self.summary.append(station_summary)

    def _count_invalid_adjacent_gaps(self, df):
        timestamp_difference = (
            df["timestamp"]
            .diff()
        )

        return (
            timestamp_difference
            .dropna()
            .ne(self.EXPECTED_INTERVAL)
            .sum()
        )

    def _validate_lags(self, df):
        lag_summary = {}

        for lag_hours in self.LAG_HOURS:
            actual_source_timestamp = (
                df["timestamp"]
                .shift(lag_hours)
            )

            expected_source_timestamp = (
                df["timestamp"] -
                pd.Timedelta(hours=lag_hours)
            )

            candidate_mask = (
                actual_source_timestamp
                .notna()
            )

            exact_timestamp_mask = (
                candidate_mask
                &
                (
                    actual_source_timestamp ==
                    expected_source_timestamp
                )
            )

            source_pm25 = (
                df["pm2_5"]
                .shift(lag_hours)
            )

            exact_with_pm25_mask = (
                exact_timestamp_mask
                &
                source_pm25.notna()
            )

            candidate_rows = candidate_mask.sum()
            exact_timestamp_rows = exact_timestamp_mask.sum()

            lag_summary[f"lag_{lag_hours}_candidate_rows"] = candidate_rows
            lag_summary[f"lag_{lag_hours}_exact_timestamp_rows"] = (
                exact_timestamp_rows
            )
            lag_summary[f"lag_{lag_hours}_exact_timestamp_percent"] = (
                self._percentage(
                    exact_timestamp_rows,
                    candidate_rows,
                )
            )
            lag_summary[f"lag_{lag_hours}_exact_with_pm25_rows"] = (
                exact_with_pm25_mask.sum()
            )

        return lag_summary

    def _validate_rolling_windows(self, df):
        rolling_summary = {}

        hourly_transition = (
            df["timestamp"]
            .diff()
            .eq(self.EXPECTED_INTERVAL)
        )

        pm25_available = (
            df["pm2_5"]
            .notna()
        )

        for window in self.ROLLING_WINDOWS:
            candidate_mask = (
                pd.Series(
                    range(len(df)),
                    index=df.index,
                )
                >= window - 1
            )

            continuous_transition_count = (
                hourly_transition
                .rolling(window=window - 1)
                .sum()
            )

            continuous_mask = (
                candidate_mask
                &
                (
                    continuous_transition_count ==
                    window - 1
                )
            )

            pm25_count = (
                pm25_available
                .rolling(window=window)
                .sum()
            )

            continuous_with_pm25_mask = (
                continuous_mask
                &
                (
                    pm25_count ==
                    window
                )
            )

            candidate_rows = candidate_mask.sum()
            continuous_rows = continuous_mask.sum()

            rolling_summary[f"rolling_{window}_candidate_rows"] = (
                candidate_rows
            )
            rolling_summary[f"rolling_{window}_continuous_rows"] = (
                continuous_rows
            )
            rolling_summary[f"rolling_{window}_continuous_percent"] = (
                self._percentage(
                    continuous_rows,
                    candidate_rows,
                )
            )
            rolling_summary[f"rolling_{window}_continuous_with_pm25_rows"] = (
                continuous_with_pm25_mask.sum()
            )

        return rolling_summary

    def save_summary(self):
        summary_df = pd.DataFrame(
            self.summary
        )

        summary_path = (
            TEMPORAL_FEATURE_VALIDATION_DIR /
            "temporal_feature_validation_summary.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        logger.info(
            f"Saved {summary_path}"
        )

        if summary_df.empty:
            logger.warning(
                "No station CSV files were available for temporal feature "
                "validation."
            )
            return

        self._log_aggregate_results(summary_df)

    def _log_aggregate_results(self, summary_df):
        logger.info("=" * 50)
        logger.info("Aggregate lag timestamp correctness")

        for lag_hours in self.LAG_HOURS:
            candidate_rows = (
                summary_df[f"lag_{lag_hours}_candidate_rows"]
                .sum()
            )
            exact_rows = (
                summary_df[f"lag_{lag_hours}_exact_timestamp_rows"]
                .sum()
            )

            logger.info(
                f"lag_{lag_hours}: {exact_rows} / {candidate_rows} "
                f"({self._percentage(exact_rows, candidate_rows)}%)"
            )

        logger.info("=" * 50)
        logger.info("Aggregate rolling-window continuity")

        for window in self.ROLLING_WINDOWS:
            candidate_rows = (
                summary_df[f"rolling_{window}_candidate_rows"]
                .sum()
            )
            continuous_rows = (
                summary_df[f"rolling_{window}_continuous_rows"]
                .sum()
            )

            logger.info(
                f"rolling_{window}: {continuous_rows} / {candidate_rows} "
                f"({self._percentage(continuous_rows, candidate_rows)}%)"
            )

        logger.info("=" * 50)

    @staticmethod
    def _percentage(valid_rows, candidate_rows):
        if candidate_rows == 0:
            return 0

        return round(
            (valid_rows / candidate_rows) * 100,
            2,
        )

    def run(self):
        csv_files = sorted(
            TRIMMED_DIR.glob("*.csv")
        )

        logger.info(
            f"Validating temporal features for {len(csv_files)} stations..."
        )

        for csv_file in csv_files:
            self.validate_station(
                csv_file
            )

        self.save_summary()
