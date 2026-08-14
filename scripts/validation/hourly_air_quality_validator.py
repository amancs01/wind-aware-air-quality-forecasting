from pathlib import Path

import pandas as pd

from config import AIR_QUALITY_HOURLY_DIR, VALIDATION_DIR
from logger import logger


class HourlyAirQualityValidator:

    def __init__(
        self,
        input_dir=AIR_QUALITY_HOURLY_DIR,
        output_dir=VALIDATION_DIR,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.results = []

    @staticmethod
    def _local_clock_timestamp(series):
        parsed = pd.to_datetime(
            series,
            errors="coerce",
        )

        return parsed.dt.tz_localize(None)

    def validate_file(self, csv_file):
        station = csv_file.stem

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            self.results.append({
                "station": station,
                "file": csv_file.name,
                "status": "Corrupted",
                "error": str(e),
                "rows": 0,
            })
            return

        if df.empty:
            self.results.append({
                "station": station,
                "file": csv_file.name,
                "status": "OK",
                "error": None,
                "rows": 0,
                "unique_timestamps": 0,
                "duplicate_timestamp_rows": 0,
                "timestamps_sorted": True,
                "first_timestamp": None,
                "last_timestamp": None,
                "non_clock_hour_rows": 0,
                "missing_pm25": 0,
                "invalid_one_hour_intervals": 0,
                "timestamp_mismatch_rows": 0,
            })
            return

        timestamp = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )
        datetime_from_local = pd.to_datetime(
            df["datetime_from_local"],
            errors="coerce",
        )
        datetime_to_local = pd.to_datetime(
            df["datetime_to_local"],
            errors="coerce",
        )
        datetime_to_clock = self._local_clock_timestamp(
            df["datetime_to_local"]
        )

        interval_delta = datetime_to_local - datetime_from_local
        timestamp_mismatch = (
            timestamp != datetime_to_clock
        ) | timestamp.isna() | datetime_to_clock.isna()

        result = {
            "station": station,
            "file": csv_file.name,
            "status": "OK",
            "error": None,
            "rows": len(df),
            "unique_timestamps": df["timestamp"].nunique(),
            "duplicate_timestamp_rows": int(
                df["timestamp"].duplicated(keep=False).sum()
            ),
            "timestamps_sorted": bool(
                timestamp.is_monotonic_increasing
            ),
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
            "non_clock_hour_rows": int(
                (
                    (timestamp.dt.minute != 0) |
                    (timestamp.dt.second != 0) |
                    timestamp.isna()
                ).sum()
            ),
            "missing_pm25": int(df["pm2_5"].isna().sum()),
            "invalid_one_hour_intervals": int(
                (interval_delta != pd.Timedelta(hours=1)).sum()
            ),
            "timestamp_mismatch_rows": int(timestamp_mismatch.sum()),
        }

        self.results.append(result)

    def save_reports(self):
        df = pd.DataFrame(self.results)

        detail_file = self.output_dir / "hourly_air_quality_validation.csv"
        summary_file = (
            self.output_dir /
            "hourly_air_quality_validation_summary.csv"
        )

        df.to_csv(
            detail_file,
            index=False,
        )

        summary = pd.DataFrame([{
            "stations_checked": df["station"].nunique() if len(df) else 0,
            "files": len(df),
            "rows": int(df.get("rows", pd.Series(dtype=int)).sum()),
            "duplicate_timestamp_rows": int(
                df.get(
                    "duplicate_timestamp_rows",
                    pd.Series(dtype=int),
                ).sum()
            ),
            "non_clock_hour_rows": int(
                df.get("non_clock_hour_rows", pd.Series(dtype=int)).sum()
            ),
            "invalid_one_hour_intervals": int(
                df.get(
                    "invalid_one_hour_intervals",
                    pd.Series(dtype=int),
                ).sum()
            ),
            "missing_pm25": int(
                df.get("missing_pm25", pd.Series(dtype=int)).sum()
            ),
            "unsorted_stations": int(
                (~df.get(
                    "timestamps_sorted",
                    pd.Series(dtype=bool),
                )).sum()
            ) if len(df) else 0,
            "stations_with_duplicate_timestamps": int(
                (
                    df.get(
                        "duplicate_timestamp_rows",
                        pd.Series(dtype=int),
                    ) > 0
                ).sum()
            ) if len(df) else 0,
        }])

        summary.to_csv(
            summary_file,
            index=False,
        )

        logger.info(f"Saved {detail_file}")
        logger.info(f"Saved {summary_file}")

        return df, summary

    def run(self, stations_filter=None):
        if not self.input_dir.exists():
            raise FileNotFoundError(self.input_dir)

        requested = set(stations_filter or [])

        for csv_file in sorted(self.input_dir.glob("*.csv")):
            if requested and csv_file.stem not in requested:
                continue

            self.validate_file(csv_file)

        return self.save_reports()
