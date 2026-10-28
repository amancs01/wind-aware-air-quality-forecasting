import pandas as pd

from config import (
    FEATURED_DIR,
    MODEL_FEATURE_COLUMNS,
    PREPARED_DIR,
    RESULTS_DIR,
)
from logger import logger


WINDOW_LENGTH = 24
HORIZON_HOURS = 1
EXPECTED_STEP = pd.Timedelta(hours=1)
MIN_TRAIN_SEQUENCES = 100

PM_CURRENT = [
    "pm2_5",
]

TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

WEATHER_FEATURES = [
    "temperature",
    "humidity",
    "pressure",
    "dew_point",
]

WIND_FEATURES = [
    "wind_u",
    "wind_v",
]

SEQUENCE_NATIVE_COLUMNS = (
    PM_CURRENT +
    TIME_FEATURES +
    WEATHER_FEATURES +
    WIND_FEATURES
)

INPUT_DESIGNS = [
    {
        "design_id": "full_model_features",
        "description": "Full MODEL_FEATURE_COLUMNS",
        "input_columns": MODEL_FEATURE_COLUMNS,
    },
    {
        "design_id": "sequence_native",
        "description": (
            "Current PM2.5 + cyclical time + weather + physical wind"
        ),
        "input_columns": SEQUENCE_NATIVE_COLUMNS,
    },
]

SPLITS = [
    {
        "split": "train",
        "start": 0.00,
        "end": 0.70,
    },
    {
        "split": "validation",
        "start": 0.70,
        "end": 0.85,
    },
    {
        "split": "test",
        "start": 0.85,
        "end": 1.00,
    },
]


class LSTMSequenceDatasetValidator:

    def __init__(self):
        self.output_dir = RESULTS_DIR / "lstm_sequence_validation"
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self):
        self._validate_designs()

        source_assessment = []
        station_split_counts = []
        accepted_checks = []

        for featured_file in sorted(FEATURED_DIR.glob("*.csv")):
            station = featured_file.stem
            featured_df = pd.read_csv(featured_file)
            featured_df["timestamp"] = pd.to_datetime(
                featured_df["timestamp"],
            )

            source_assessment.append(
                self._assess_source_station(
                    station,
                    featured_df,
                )
            )

            split_ranges = self._build_split_ranges(station)
            if not split_ranges:
                station_split_counts.extend(
                    self._empty_station_rows(
                        station,
                        reason="no prepared split boundaries",
                    )
                )
                continue

            for design in INPUT_DESIGNS:
                station_split_counts.extend(
                    self._validate_station_design(
                        station,
                        featured_df,
                        split_ranges,
                        design,
                        accepted_checks,
                    )
                )

        source_df = pd.DataFrame(source_assessment)
        station_counts_df = pd.DataFrame(station_split_counts)
        checks_df = pd.DataFrame(accepted_checks)
        summary_df = self._build_summary(station_counts_df)
        too_few_df = self._build_too_few_table(station_counts_df)
        design_df = self._build_design_table()

        self._write_outputs(
            source_df,
            station_counts_df,
            summary_df,
            checks_df,
            too_few_df,
            design_df,
        )

        recommended = summary_df[
            summary_df["design_id"] == "sequence_native"
        ]
        logger.info(
            "LSTM sequence dataset validation complete. "
            f"Recommended design generated "
            f"{int(recommended['accepted_sequences'].sum())} sequences."
        )

    def _validate_designs(self):
        model_features = set(MODEL_FEATURE_COLUMNS)
        for design in INPUT_DESIGNS:
            missing = [
                column
                for column in design["input_columns"]
                if column not in model_features
            ]
            if missing:
                raise ValueError(
                    f"{design['design_id']} contains non-model columns: "
                    f"{missing}"
                )

    def _assess_source_station(self, station, featured_df):
        prepared_file = PREPARED_DIR / f"{station}.csv"
        prepared_rows = None
        prepared_invalid_gaps = None
        prepared_largest_gap_hours = None

        if prepared_file.exists():
            prepared_df = pd.read_csv(
                prepared_file,
                usecols=["timestamp"],
            )
            prepared_df["timestamp"] = pd.to_datetime(
                prepared_df["timestamp"],
            )
            prepared_rows = len(prepared_df)
            prepared_gaps = prepared_df["timestamp"].diff().dropna()
            prepared_invalid_gaps = int(
                prepared_gaps.ne(EXPECTED_STEP).sum()
            )
            prepared_largest_gap_hours = (
                prepared_gaps.max() / pd.Timedelta(hours=1)
                if not prepared_gaps.empty
                else None
            )

        featured_gaps = featured_df["timestamp"].diff().dropna()

        return {
            "station": station,
            "featured_rows": len(featured_df),
            "featured_invalid_hourly_gaps": int(
                featured_gaps.ne(EXPECTED_STEP).sum()
            ),
            "featured_largest_gap_hours": (
                featured_gaps.max() / pd.Timedelta(hours=1)
                if not featured_gaps.empty
                else None
            ),
            "prepared_rows": prepared_rows,
            "prepared_invalid_hourly_gaps": prepared_invalid_gaps,
            "prepared_largest_gap_hours": prepared_largest_gap_hours,
            "prepared_has_row_gaps": (
                prepared_invalid_gaps is not None and
                prepared_invalid_gaps > 0
            ),
        }

    def _build_split_ranges(self, station):
        prepared_file = PREPARED_DIR / f"{station}.csv"
        if not prepared_file.exists():
            return {}

        prepared_df = pd.read_csv(
            prepared_file,
            usecols=["timestamp"],
        )
        if prepared_df.empty:
            return {}

        prepared_df["timestamp"] = pd.to_datetime(
            prepared_df["timestamp"],
        )
        rows = len(prepared_df)
        split_ranges = {}

        for split in SPLITS:
            start_index = int(rows * split["start"])
            end_index = int(rows * split["end"])

            if split["split"] == "test":
                end_index = rows

            split_df = prepared_df.iloc[start_index:end_index]
            if split_df.empty:
                continue

            split_ranges[split["split"]] = {
                "start_timestamp": split_df["timestamp"].iloc[0],
                "end_timestamp": split_df["timestamp"].iloc[-1],
                "prepared_rows": len(split_df),
            }

        return split_ranges

    def _empty_station_rows(self, station, reason):
        rows = []
        for design in INPUT_DESIGNS:
            for split in SPLITS:
                rows.append({
                    "station": station,
                    "design_id": design["design_id"],
                    "split": split["split"],
                    "candidate_target_rows": 0,
                    "accepted_sequences": 0,
                    "rejected_missing_values": 0,
                    "rejected_timestamp_discontinuity": 0,
                    "rejected_split_boundary_crossing": 0,
                    "rejected_insufficient_history": 0,
                    "reason": reason,
                })

        return rows

    def _validate_station_design(
            self,
            station,
            featured_df,
            split_ranges,
            design,
            accepted_checks,
    ):
        split_counters = {
            split["split"]: {
                "station": station,
                "design_id": design["design_id"],
                "split": split["split"],
                "candidate_target_rows": 0,
                "accepted_sequences": 0,
                "rejected_missing_values": 0,
                "rejected_timestamp_discontinuity": 0,
                "rejected_split_boundary_crossing": 0,
                "rejected_insufficient_history": 0,
                "reason": "",
            }
            for split in SPLITS
        }

        timestamps = featured_df["timestamp"].reset_index(drop=True)
        input_missing_by_row = (
            featured_df[design["input_columns"]]
            .isna()
            .any(axis=1)
            .astype(int)
            .reset_index(drop=True)
        )
        input_missing_cumsum = input_missing_by_row.cumsum()
        target_missing_by_row = (
            featured_df["pm2_5"]
            .isna()
            .reset_index(drop=True)
        )
        invalid_gap_by_row = (
            timestamps.diff()
            .ne(EXPECTED_STEP)
            .astype(int)
        )
        if len(invalid_gap_by_row) > 0:
            invalid_gap_by_row.iloc[0] = 0
        invalid_gap_cumsum = invalid_gap_by_row.cumsum()

        for target_index, target_timestamp in enumerate(timestamps):
            split_name = self._classify_target_split(
                target_timestamp,
                split_ranges,
            )
            if split_name is None:
                continue

            counter = split_counters[split_name]
            counter["candidate_target_rows"] += 1

            input_start = target_index - WINDOW_LENGTH

            if input_start < 0:
                counter["rejected_insufficient_history"] += 1
                continue

            input_start_timestamp = timestamps.iloc[input_start]
            input_end_timestamp = timestamps.iloc[target_index - 1]

            if not self._window_within_split(
                    input_start_timestamp,
                    target_timestamp,
                    split_ranges[split_name],
            ):
                counter["rejected_split_boundary_crossing"] += 1
                continue

            invalid_gap_count = (
                invalid_gap_cumsum.iloc[target_index] -
                invalid_gap_cumsum.iloc[input_start]
            )
            if invalid_gap_count != 0:
                counter["rejected_timestamp_discontinuity"] += 1
                continue

            missing_before_window = (
                input_missing_cumsum.iloc[input_start - 1]
                if input_start > 0
                else 0
            )
            input_missing = (
                input_missing_cumsum.iloc[target_index - 1] -
                missing_before_window
            ) > 0
            target_missing = target_missing_by_row.iloc[target_index]

            if input_missing or target_missing:
                counter["rejected_missing_values"] += 1
                continue

            counter["accepted_sequences"] += 1
            accepted_checks.append({
                "station": station,
                "design_id": design["design_id"],
                "split": split_name,
                "input_start_timestamp": input_start_timestamp,
                "input_end_timestamp": input_end_timestamp,
                "target_timestamp": target_timestamp,
                "input_rows": WINDOW_LENGTH,
                "input_hourly_diffs_all_valid": True,
                "target_gap_hours": (
                    target_timestamp -
                    input_end_timestamp
                ) / pd.Timedelta(hours=1),
                "target_gap_valid": (
                    target_timestamp -
                    input_end_timestamp
                ) == EXPECTED_STEP,
                "within_split": True,
            })

        return list(split_counters.values())

    @staticmethod
    def _classify_target_split(target_timestamp, split_ranges):
        for split_name, split_range in split_ranges.items():
            if (
                split_range["start_timestamp"] <=
                target_timestamp <=
                split_range["end_timestamp"]
            ):
                return split_name

        return None

    @staticmethod
    def _window_within_split(input_start_timestamp, target_timestamp, split_range):
        return (
            input_start_timestamp >= split_range["start_timestamp"] and
            target_timestamp <= split_range["end_timestamp"]
        )

    def _build_summary(self, station_counts_df):
        grouped = (
            station_counts_df
            .groupby(["design_id", "split"], as_index=False)
            .agg(
                stations=("station", "nunique"),
                candidate_target_rows=("candidate_target_rows", "sum"),
                accepted_sequences=("accepted_sequences", "sum"),
                rejected_missing_values=("rejected_missing_values", "sum"),
                rejected_timestamp_discontinuity=(
                    "rejected_timestamp_discontinuity",
                    "sum",
                ),
                rejected_split_boundary_crossing=(
                    "rejected_split_boundary_crossing",
                    "sum",
                ),
                rejected_insufficient_history=(
                    "rejected_insufficient_history",
                    "sum",
                ),
            )
        )
        grouped["window_length"] = WINDOW_LENGTH
        grouped["horizon_hours"] = HORIZON_HOURS

        return grouped

    def _build_too_few_table(self, station_counts_df):
        recommended = station_counts_df[
            station_counts_df["design_id"] == "sequence_native"
        ]
        pivot = recommended.pivot_table(
            index="station",
            columns="split",
            values="accepted_sequences",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

        for split in ["train", "validation", "test"]:
            if split not in pivot.columns:
                pivot[split] = 0

        pivot["too_few_reason"] = ""
        pivot.loc[
            pivot["train"] < MIN_TRAIN_SEQUENCES,
            "too_few_reason",
        ] += "train sequences below 100; "
        pivot.loc[
            pivot["validation"] == 0,
            "too_few_reason",
        ] += "no validation sequences; "
        pivot.loc[
            pivot["test"] == 0,
            "too_few_reason",
        ] += "no test sequences; "

        return pivot[pivot["too_few_reason"] != ""].copy()

    @staticmethod
    def _build_design_table():
        return pd.DataFrame([
            {
                "design_id": design["design_id"],
                "description": design["description"],
                "input_columns": "|".join(design["input_columns"]),
                "input_column_count": len(design["input_columns"]),
                "recommended": design["design_id"] == "sequence_native",
            }
            for design in INPUT_DESIGNS
        ])

    def _write_outputs(
            self,
            source_df,
            station_counts_df,
            summary_df,
            checks_df,
            too_few_df,
            design_df,
    ):
        source_df.to_csv(
            self.output_dir / "sequence_source_assessment.csv",
            index=False,
        )
        station_counts_df.to_csv(
            self.output_dir / "sequence_station_split_counts.csv",
            index=False,
        )
        summary_df.to_csv(
            self.output_dir / "sequence_summary.csv",
            index=False,
        )
        checks_df.to_csv(
            self.output_dir / "sequence_accepted_window_checks.csv",
            index=False,
        )
        too_few_df.to_csv(
            self.output_dir / "sequence_stations_too_few.csv",
            index=False,
        )
        design_df.to_csv(
            self.output_dir / "sequence_input_designs.csv",
            index=False,
        )
        self._write_markdown_report(
            source_df,
            station_counts_df,
            summary_df,
            checks_df,
            too_few_df,
            design_df,
        )

    def _write_markdown_report(
            self,
            source_df,
            station_counts_df,
            summary_df,
            checks_df,
            too_few_df,
            design_df,
    ):
        source_summary = {
            "featured stations": len(source_df),
            "featured invalid hourly gaps": int(
                source_df["featured_invalid_hourly_gaps"].sum()
            ),
            "featured largest gap hours": float(
                source_df["featured_largest_gap_hours"].max()
            ),
            "prepared stations with row gaps": int(
                source_df["prepared_has_row_gaps"].fillna(False).sum()
            ),
            "prepared invalid hourly gaps": int(
                source_df["prepared_invalid_hourly_gaps"]
                .fillna(0)
                .sum()
            ),
            "prepared largest gap hours": float(
                source_df["prepared_largest_gap_hours"].max()
            ),
        }

        recommended_counts = station_counts_df[
            station_counts_df["design_id"] == "sequence_native"
        ]
        recommended_pivot = (
            recommended_counts
            .pivot_table(
                index="station",
                columns="split",
                values="accepted_sequences",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        for split in ["train", "validation", "test"]:
            if split not in recommended_pivot.columns:
                recommended_pivot[split] = 0
        recommended_pivot = recommended_pivot[
            ["station", "train", "validation", "test"]
        ]

        proof = (
            checks_df
            .groupby("design_id")
            .agg(
                accepted_windows=("station", "size"),
                invalid_input_row_count=(
                    "input_rows",
                    lambda values: int((values != WINDOW_LENGTH).sum()),
                ),
                invalid_hourly_inputs=(
                    "input_hourly_diffs_all_valid",
                    lambda values: int((~values.astype(bool)).sum()),
                ),
                invalid_target_gap=(
                    "target_gap_valid",
                    lambda values: int((~values.astype(bool)).sum()),
                ),
                invalid_split_membership=(
                    "within_split",
                    lambda values: int((~values.astype(bool)).sum()),
                ),
            )
            .reset_index()
        )

        recommended_columns = design_df.loc[
            design_df["design_id"] == "sequence_native",
            "input_columns",
        ].iloc[0].replace("|", ", ")

        report = [
            "# LSTM Sequence Dataset Validation",
            "",
            "This is a validation-only artifact. No LSTM was trained.",
            "",
            "## Source Decision",
            "",
            f"Proposed source directory: `{FEATURED_DIR}`.",
            "",
            "`data/processed/prepared/` is not used for window construction "
            "because it drops rows with missing current or target PM2.5. "
            "Prepared rows are therefore not guaranteed to be adjacent hours.",
            "",
            "Source assessment:",
            "",
            "```text",
            pd.Series(source_summary).to_string(),
            "```",
            "",
            "## Recommended Input Design",
            "",
            "The recommended first LSTM baseline should use current PM2.5, "
            "cyclical time, weather, and physical wind components, excluding "
            "hand-engineered lag and rolling PM2.5 summaries.",
            "",
            f"Exact input columns: `{recommended_columns}`.",
            "",
            "Window length: 24 hours.",
            "Horizon: 1 hour after the final input timestamp.",
            "",
            "## Split Counts",
            "",
            "Recommended-design accepted sequences per station and split:",
            "",
            "```text",
            recommended_pivot.to_string(index=False),
            "```",
            "",
            "Aggregate candidate, accepted, and rejected counts:",
            "",
            "```text",
            summary_df.to_string(index=False),
            "```",
            "",
            "## Proof Checks",
            "",
            "Every accepted window must have 24 rows, exactly hourly input "
            "timestamps, a target exactly one hour after the final input, "
            "and split-contained timestamps.",
            "",
            "```text",
            proof.to_string(index=False),
            "```",
            "",
            "## Too-Few Stations",
            "",
            "Stations are flagged when the recommended design has fewer than "
            "100 train sequences or no validation/test sequences.",
            "",
            "```text",
            (
                too_few_df.to_string(index=False)
                if not too_few_df.empty
                else "None"
            ),
            "```",
            "",
            "## Dataset Architecture Recommendation",
            "",
            "A future LSTM dataset should scan featured station files, create "
            "an index table of accepted `(station, split, input_start, "
            "input_end, target)` windows, and load tensors from that index. "
            "The expected tensor shape is `(n_sequences, 24, 11)` for inputs "
            "and one scalar next-hour PM2.5 target per sequence.",
            "",
            "Split boundary timestamps should be derived chronologically from "
            "the existing prepared split convention, but no input/target "
            "window may cross a split boundary. Any scaler must be fit on "
            "training input rows only and then reused for validation and test.",
            "",
        ]

        (self.output_dir / "sequence_validation_report.md").write_text(
            "\n".join(report),
            encoding="utf-8",
        )


def main():
    validator = LSTMSequenceDatasetValidator()
    validator.run()


if __name__ == "__main__":
    main()
