import gc
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import (
    FEATURE_ANALYSIS_DIR,
    ML_VALIDATION_DIR,
    MODEL_FEATURE_COLUMNS,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MAX_FEATURES,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_N_JOBS,
    RANDOM_FOREST_RANDOM_STATE,
    STATIONS_FILE,
    TRAIN_DIR,
)
from logger import logger
from models.base_model import BaseModel


PM_CURRENT = [
    "pm2_5",
]

PM_HISTORY = [
    "pm2_5",
    "lag_6",
    "lag_24",
    "rolling_mean_6",
    "rolling_std_6",
]

TIME = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

WIND = [
    "wind_u",
    "wind_v",
]

WEATHER = [
    "temperature",
    "humidity",
    "pressure",
    "dew_point",
]

ABLATION_VARIANTS = [
    {
        "variant_id": "A0",
        "variant": "Persistence",
        "model": "Persistence",
        "features": PM_CURRENT,
        "description": "PM2.5(t+1) = PM2.5(t)",
    },
    {
        "variant_id": "A1",
        "variant": "Current PM only",
        "model": "RandomForest",
        "features": PM_CURRENT,
        "description": "Random Forest using current PM2.5 only",
    },
    {
        "variant_id": "A2",
        "variant": "PM history",
        "model": "RandomForest",
        "features": PM_HISTORY,
        "description": "Current PM2.5 plus lag and rolling PM2.5 features",
    },
    {
        "variant_id": "A3",
        "variant": "PM history + time",
        "model": "RandomForest",
        "features": PM_HISTORY + TIME,
        "description": "PM2.5 history plus cyclical time features",
    },
    {
        "variant_id": "A4",
        "variant": "PM history + time + non-wind weather",
        "model": "RandomForest",
        "features": PM_HISTORY + TIME + WEATHER,
        "description": "Full current feature set minus wind components",
    },
    {
        "variant_id": "A5",
        "variant": "PM history + time + wind",
        "model": "RandomForest",
        "features": PM_HISTORY + TIME + WIND,
        "description": "PM2.5 history, time, and physical wind components",
    },
    {
        "variant_id": "A6",
        "variant": "Full current feature set",
        "model": "RandomForest",
        "features": MODEL_FEATURE_COLUMNS,
        "description": "All current MODEL_FEATURE_COLUMNS",
    },
]

COMPARISONS = [
    {
        "comparison": "Historical PM contribution",
        "baseline_variant_id": "A1",
        "added_variant_id": "A2",
    },
    {
        "comparison": "Time contribution",
        "baseline_variant_id": "A2",
        "added_variant_id": "A3",
    },
    {
        "comparison": "Non-wind weather contribution",
        "baseline_variant_id": "A3",
        "added_variant_id": "A4",
    },
    {
        "comparison": "Wind contribution without other weather",
        "baseline_variant_id": "A3",
        "added_variant_id": "A5",
    },
    {
        "comparison": "Conditional wind contribution",
        "baseline_variant_id": "A4",
        "added_variant_id": "A6",
    },
    {
        "comparison": "Conditional non-wind weather contribution",
        "baseline_variant_id": "A5",
        "added_variant_id": "A6",
    },
]


class FeatureAblation:

    def __init__(self):
        self.output_dir = FEATURE_ANALYSIS_DIR / "ablation"
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.evaluator = BaseModel(self.output_dir)

    def run(self):
        self._validate_feature_groups()

        station_frames = self._load_station_frames()
        logger.info(
            f"Loaded {len(station_frames)} fixed train/validation frames"
        )

        station_metrics = []
        summaries = []

        for variant in ABLATION_VARIANTS:
            logger.info(
                f"Evaluating {variant['variant_id']}: {variant['variant']}"
            )
            metrics_df, summary = self._evaluate_variant(
                variant,
                station_frames,
            )
            station_metrics.append(metrics_df)
            summaries.append(summary)

        station_metrics_df = pd.concat(
            station_metrics,
            ignore_index=True,
        )
        summary_df = pd.DataFrame(summaries)
        comparisons_df = self._build_comparisons(summary_df)
        wind_station_df = self._build_wind_station_comparison(
            station_metrics_df,
        )
        row_identity_df = self._build_row_identity_summary(station_frames)
        feature_groups_df = self._build_feature_groups_table()

        self._write_outputs(
            summary_df,
            station_metrics_df,
            comparisons_df,
            wind_station_df,
            row_identity_df,
            feature_groups_df,
        )

        best = summary_df.sort_values("pooled_rmse").iloc[0]
        wind = comparisons_df[
            comparisons_df["comparison"] == "Conditional wind contribution"
        ].iloc[0]

        logger.info(
            "Feature ablation complete. Best validation pooled RMSE: "
            f"{best['variant_id']} {best['variant']} = "
            f"{best['pooled_rmse']:.3f}"
        )
        logger.info(
            "Conditional wind RMSE improvement "
            f"(A4 - A6): {wind['pooled_rmse_improvement']:.3f}"
        )

    def _validate_feature_groups(self):
        expected = set(PM_HISTORY + TIME + WEATHER + WIND)
        actual = set(MODEL_FEATURE_COLUMNS)

        if expected != actual:
            raise ValueError(
                "A6 feature groups do not match MODEL_FEATURE_COLUMNS. "
                f"Missing from groups: {sorted(actual - expected)}. "
                f"Extra in groups: {sorted(expected - actual)}."
            )

        for variant in ABLATION_VARIANTS:
            missing = [
                feature
                for feature in variant["features"]
                if feature not in MODEL_FEATURE_COLUMNS
            ]
            if missing:
                raise ValueError(
                    f"{variant['variant_id']} contains unknown features: "
                    f"{missing}"
                )

    def _load_station_frames(self):
        required_columns = MODEL_FEATURE_COLUMNS + [
            "target_pm2_5",
            "timestamp",
        ]
        station_frames = []

        for train_file in sorted(TRAIN_DIR.glob("*.csv")):
            validation_file = ML_VALIDATION_DIR / train_file.name

            if not validation_file.exists():
                logger.warning(
                    f"Skipping {train_file.stem}: validation file missing"
                )
                continue

            train_df = pd.read_csv(
                train_file,
                usecols=required_columns,
            )
            validation_df = pd.read_csv(
                validation_file,
                usecols=required_columns,
            )

            train_eval = self.evaluator.prepare_evaluation_frame(
                train_df,
                required_features=MODEL_FEATURE_COLUMNS,
            )
            validation_eval = self.evaluator.prepare_evaluation_frame(
                validation_df,
                required_features=MODEL_FEATURE_COLUMNS,
            )

            if train_eval.empty or validation_eval.empty:
                logger.warning(
                    f"Skipping {train_file.stem}: no fixed-frame rows"
                )
                continue

            station_frames.append({
                "station": train_file.stem,
                "original_train_rows": len(train_df),
                "original_validation_rows": len(validation_df),
                "train_eval": train_eval,
                "validation_eval": validation_eval,
            })

        return station_frames

    def _evaluate_variant(self, variant, station_frames):
        station_rows = []
        pooled_targets = []
        pooled_predictions = []

        for station_frame in station_frames:
            station_row, target, prediction = self._evaluate_station_variant(
                variant,
                station_frame,
            )
            station_rows.append(station_row)
            pooled_targets.extend(target.to_list())
            pooled_predictions.extend(pd.Series(prediction).to_list())

        metrics_df = pd.DataFrame(station_rows)
        summary = self._summarize_variant(
            variant,
            metrics_df,
            pooled_targets,
            pooled_predictions,
        )

        return metrics_df, summary

    def _evaluate_station_variant(self, variant, station_frame):
        train_eval = station_frame["train_eval"]
        validation_eval = station_frame["validation_eval"]
        target = validation_eval["target_pm2_5"]

        if variant["model"] == "Persistence":
            prediction = validation_eval["pm2_5"]
        else:
            features = variant["features"]
            estimator = RandomForestRegressor(
                n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                max_depth=RANDOM_FOREST_MAX_DEPTH,
                min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
                max_features=RANDOM_FOREST_MAX_FEATURES,
                random_state=RANDOM_FOREST_RANDOM_STATE,
                n_jobs=RANDOM_FOREST_N_JOBS,
            )
            estimator.fit(
                train_eval[features],
                train_eval["target_pm2_5"],
            )
            prediction = estimator.predict(
                validation_eval[features],
            )
            del estimator
            gc.collect()

        mae, rmse, r2 = self.evaluator.evaluate(
            target,
            prediction,
        )

        station_row = {
            "variant_id": variant["variant_id"],
            "variant": variant["variant"],
            "model": variant["model"],
            "station": station_frame["station"],
            "features": "|".join(variant["features"]),
            "feature_count": len(variant["features"]),
            "original_train_rows": station_frame["original_train_rows"],
            "train_rows": len(train_eval),
            "original_validation_rows": (
                station_frame["original_validation_rows"]
            ),
            "validation_rows": len(validation_eval),
            "target_mean": target.mean(),
            "target_std": target.std(),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        return station_row, target, prediction

    def _summarize_variant(
            self,
            variant,
            metrics_df,
            pooled_targets,
            pooled_predictions,
    ):
        pooled_mae, pooled_rmse, pooled_r2 = self.evaluator.evaluate(
            pooled_targets,
            pooled_predictions,
        )

        train_rows = int(metrics_df["train_rows"].sum())
        validation_rows = int(metrics_df["validation_rows"].sum())
        original_train_rows = int(metrics_df["original_train_rows"].sum())
        original_validation_rows = int(
            metrics_df["original_validation_rows"].sum()
        )

        return {
            "variant_id": variant["variant_id"],
            "variant": variant["variant"],
            "model": variant["model"],
            "description": variant["description"],
            "features": "|".join(variant["features"]),
            "feature_count": len(variant["features"]),
            "datasets": len(metrics_df),
            "original_train_rows": original_train_rows,
            "train_rows": train_rows,
            "removed_train_rows": original_train_rows - train_rows,
            "original_validation_rows": original_validation_rows,
            "validation_rows": validation_rows,
            "removed_validation_rows": (
                original_validation_rows - validation_rows
            ),
            "macro_mae": metrics_df["mae"].mean(),
            "macro_rmse": metrics_df["rmse"].mean(),
            "macro_mean_r2": metrics_df["r2"].mean(),
            "macro_median_r2": metrics_df["r2"].median(),
            "pooled_mae": pooled_mae,
            "pooled_rmse": pooled_rmse,
            "pooled_r2": pooled_r2,
            "negative_r2_datasets": int((metrics_df["r2"] < 0).sum()),
            "positive_r2_datasets": int((metrics_df["r2"] >= 0).sum()),
            "n_estimators": (
                RANDOM_FOREST_N_ESTIMATORS
                if variant["model"] == "RandomForest"
                else None
            ),
            "max_depth": (
                RANDOM_FOREST_MAX_DEPTH
                if variant["model"] == "RandomForest"
                else None
            ),
            "min_samples_leaf": (
                RANDOM_FOREST_MIN_SAMPLES_LEAF
                if variant["model"] == "RandomForest"
                else None
            ),
            "max_features": (
                RANDOM_FOREST_MAX_FEATURES
                if variant["model"] == "RandomForest"
                else None
            ),
            "random_state": (
                RANDOM_FOREST_RANDOM_STATE
                if variant["model"] == "RandomForest"
                else None
            ),
            "n_jobs": (
                RANDOM_FOREST_N_JOBS
                if variant["model"] == "RandomForest"
                else None
            ),
        }

    def _build_comparisons(self, summary_df):
        rows = []
        summary_by_id = summary_df.set_index("variant_id")

        for comparison in COMPARISONS:
            baseline = summary_by_id.loc[
                comparison["baseline_variant_id"]
            ]
            added = summary_by_id.loc[
                comparison["added_variant_id"]
            ]

            pooled_rmse_improvement = (
                baseline["pooled_rmse"] -
                added["pooled_rmse"]
            )
            pooled_mae_improvement = (
                baseline["pooled_mae"] -
                added["pooled_mae"]
            )
            macro_rmse_improvement = (
                baseline["macro_rmse"] -
                added["macro_rmse"]
            )

            rows.append({
                "comparison": comparison["comparison"],
                "baseline_variant_id": baseline.name,
                "baseline_variant": baseline["variant"],
                "added_variant_id": added.name,
                "added_variant": added["variant"],
                "sign_convention": "positive means lower error after adding features",
                "baseline_pooled_rmse": baseline["pooled_rmse"],
                "added_pooled_rmse": added["pooled_rmse"],
                "pooled_rmse_improvement": pooled_rmse_improvement,
                "pooled_rmse_percent_improvement": (
                    pooled_rmse_improvement /
                    baseline["pooled_rmse"] *
                    100
                    if baseline["pooled_rmse"]
                    else 0.0
                ),
                "baseline_pooled_mae": baseline["pooled_mae"],
                "added_pooled_mae": added["pooled_mae"],
                "pooled_mae_improvement": pooled_mae_improvement,
                "pooled_r2_difference": (
                    added["pooled_r2"] -
                    baseline["pooled_r2"]
                ),
                "macro_rmse_improvement": macro_rmse_improvement,
            })

        return pd.DataFrame(rows)

    def _build_wind_station_comparison(self, station_metrics_df):
        no_wind = station_metrics_df[
            station_metrics_df["variant_id"] == "A4"
        ][[
            "station",
            "validation_rows",
            "target_mean",
            "target_std",
            "rmse",
            "mae",
            "r2",
        ]].rename(columns={
            "rmse": "no_wind_rmse",
            "mae": "no_wind_mae",
            "r2": "no_wind_r2",
        })

        full = station_metrics_df[
            station_metrics_df["variant_id"] == "A6"
        ][[
            "station",
            "rmse",
            "mae",
            "r2",
        ]].rename(columns={
            "rmse": "full_rmse",
            "mae": "full_mae",
            "r2": "full_r2",
        })

        comparison = no_wind.merge(
            full,
            on="station",
            how="inner",
        )
        comparison["wind_rmse_improvement"] = (
            comparison["no_wind_rmse"] -
            comparison["full_rmse"]
        )
        comparison["wind_mae_improvement"] = (
            comparison["no_wind_mae"] -
            comparison["full_mae"]
        )
        comparison["wind_r2_difference"] = (
            comparison["full_r2"] -
            comparison["no_wind_r2"]
        )
        comparison["wind_effect"] = comparison[
            "wind_rmse_improvement"
        ].apply(self._classify_effect)

        return comparison.sort_values(
            "wind_rmse_improvement",
            ascending=False,
        )

    @staticmethod
    def _classify_effect(value):
        tolerance = 1e-9
        if value > tolerance:
            return "improved"
        if value < -tolerance:
            return "worsened"
        return "tie"

    def _build_row_identity_summary(self, station_frames):
        rows = []

        for station_frame in station_frames:
            validation_eval = station_frame["validation_eval"]
            train_eval = station_frame["train_eval"]

            rows.append({
                "station": station_frame["station"],
                "train_rows": len(train_eval),
                "validation_rows": len(validation_eval),
                "validation_source_index_min": (
                    validation_eval["source_index"].min()
                ),
                "validation_source_index_max": (
                    validation_eval["source_index"].max()
                ),
                "validation_timestamp_min": (
                    validation_eval["timestamp"].min()
                ),
                "validation_timestamp_max": (
                    validation_eval["timestamp"].max()
                ),
                "validation_target_sum": (
                    validation_eval["target_pm2_5"].sum()
                ),
            })

        return pd.DataFrame(rows)

    def _build_feature_groups_table(self):
        return pd.DataFrame([
            {
                "variant_id": variant["variant_id"],
                "variant": variant["variant"],
                "model": variant["model"],
                "feature_count": len(variant["features"]),
                "features": "|".join(variant["features"]),
                "equals_model_feature_columns": (
                    variant["features"] == MODEL_FEATURE_COLUMNS
                ),
            }
            for variant in ABLATION_VARIANTS
        ])

    def _write_outputs(
            self,
            summary_df,
            station_metrics_df,
            comparisons_df,
            wind_station_df,
            row_identity_df,
            feature_groups_df,
    ):
        summary_df.to_csv(
            self.output_dir / "ablation_summary.csv",
            index=False,
        )
        station_metrics_df.to_csv(
            self.output_dir / "ablation_station_metrics.csv",
            index=False,
        )
        comparisons_df.to_csv(
            self.output_dir / "ablation_comparisons.csv",
            index=False,
        )
        wind_station_df.to_csv(
            self.output_dir / "wind_station_comparison.csv",
            index=False,
        )
        row_identity_df.to_csv(
            self.output_dir / "row_identity.csv",
            index=False,
        )
        feature_groups_df.to_csv(
            self.output_dir / "feature_groups.csv",
            index=False,
        )

        metadata = self._load_station_metadata()
        if metadata is not None:
            wind_station_df.merge(
                metadata,
                on="station",
                how="left",
            ).to_csv(
                self.output_dir / "wind_station_comparison_with_metadata.csv",
                index=False,
            )

    def _load_station_metadata(self):
        if not Path(STATIONS_FILE).exists():
            return None

        metadata = pd.read_csv(STATIONS_FILE)
        keep_columns = [
            column
            for column in [
                "station",
                "dataset_name",
                "provider",
                "latitude",
                "longitude",
                "sensor_id",
            ]
            if column in metadata.columns
        ]

        if "station" not in keep_columns:
            return None

        return (
            metadata[keep_columns]
            .drop_duplicates(subset=["station"])
        )


def main():
    ablation = FeatureAblation()
    ablation.run()


if __name__ == "__main__":
    main()
