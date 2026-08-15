import gc

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from config import (
    LINEAR_BASELINE_ALPHA,
    MODEL_FEATURE_COLUMNS,
    PREPARED_DIR,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MAX_FEATURES,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_N_JOBS,
    RANDOM_FOREST_RANDOM_STATE,
    RESULTS_DIR,
)
from logger import logger
from models.base_model import BaseModel


FOLDS = [
    {
        "fold": 1,
        "train_start": 0.00,
        "train_end": 0.55,
        "validation_start": 0.55,
        "validation_end": 0.65,
    },
    {
        "fold": 2,
        "train_start": 0.00,
        "train_end": 0.65,
        "validation_start": 0.65,
        "validation_end": 0.75,
    },
    {
        "fold": 3,
        "train_start": 0.00,
        "train_end": 0.75,
        "validation_start": 0.75,
        "validation_end": 0.85,
    },
]

MODELS = [
    "Persistence",
    f"Ridge(alpha={LINEAR_BASELINE_ALPHA:g})",
    "Random Forest",
]


class RollingOriginValidation:

    def __init__(self):
        self.output_dir = RESULTS_DIR / "rolling_origin"
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.evaluator = BaseModel(self.output_dir)

    def run(self):
        station_fold_frames = self._load_fold_frames()
        logger.info(
            f"Prepared {len(station_fold_frames)} station-fold frames"
        )

        station_metrics = []
        pooled_predictions = []

        for fold in FOLDS:
            fold_number = fold["fold"]
            fold_frames = [
                frame
                for frame in station_fold_frames
                if frame["fold"] == fold_number
            ]

            logger.info(
                f"Evaluating fold {fold_number} with "
                f"{len(fold_frames)} station frames"
            )

            for frame in fold_frames:
                station_rows, prediction_rows = self._evaluate_station_fold(
                    frame,
                )
                station_metrics.extend(station_rows)
                pooled_predictions.extend(prediction_rows)

        station_metrics_df = pd.DataFrame(station_metrics)
        pooled_predictions_df = pd.DataFrame(pooled_predictions)
        fold_summary_df = self._build_fold_summary(
            station_metrics_df,
            pooled_predictions_df,
        )
        model_comparison_df = self._build_model_comparisons(
            station_metrics_df,
            fold_summary_df,
        )
        rf_station_wins_df = self._build_rf_station_wins(
            station_metrics_df,
        )
        distribution_shift_df = self._build_distribution_shifts(
            station_fold_frames,
        )
        fold_frame_summary_df = self._build_fold_frame_summary(
            station_fold_frames,
        )

        self._write_outputs(
            station_metrics_df,
            fold_summary_df,
            model_comparison_df,
            rf_station_wins_df,
            distribution_shift_df,
            fold_frame_summary_df,
        )

        logger.info("Rolling-origin validation complete.")

    def _load_fold_frames(self):
        required_columns = MODEL_FEATURE_COLUMNS + [
            "target_pm2_5",
            "timestamp",
        ]
        station_fold_frames = []

        for prepared_file in sorted(PREPARED_DIR.glob("*.csv")):
            df = pd.read_csv(
                prepared_file,
                usecols=required_columns,
            )
            total_rows = len(df)

            for fold in FOLDS:
                frame = self._build_station_fold_frame(
                    prepared_file.stem,
                    df,
                    total_rows,
                    fold,
                )

                if frame is not None:
                    station_fold_frames.append(frame)

        return station_fold_frames

    def _build_station_fold_frame(self, station, df, total_rows, fold):
        train_start = int(total_rows * fold["train_start"])
        train_end = int(total_rows * fold["train_end"])
        validation_start = int(total_rows * fold["validation_start"])
        validation_end = int(total_rows * fold["validation_end"])

        train_df = df.iloc[train_start:train_end].copy()
        validation_df = df.iloc[validation_start:validation_end].copy()

        train_eval = self.evaluator.prepare_evaluation_frame(
            train_df,
            required_features=MODEL_FEATURE_COLUMNS,
        )
        validation_eval = self.evaluator.prepare_evaluation_frame(
            validation_df,
            required_features=MODEL_FEATURE_COLUMNS,
        )

        if train_eval.empty or len(validation_eval) < 2:
            logger.warning(
                f"Skipping {station} fold {fold['fold']}: "
                "insufficient full-feature-valid rows"
            )
            return None

        return {
            "station": station,
            "fold": fold["fold"],
            "total_prepared_rows": total_rows,
            "train_start_percent": fold["train_start"],
            "train_end_percent": fold["train_end"],
            "validation_start_percent": fold["validation_start"],
            "validation_end_percent": fold["validation_end"],
            "train_raw_rows": len(train_df),
            "validation_raw_rows": len(validation_df),
            "train_rows": len(train_eval),
            "validation_rows": len(validation_eval),
            "train_eval": train_eval,
            "validation_eval": validation_eval,
        }

    def _evaluate_station_fold(self, frame):
        train_eval = frame["train_eval"]
        validation_eval = frame["validation_eval"]
        target = validation_eval["target_pm2_5"]

        rows = []

        persistence_prediction = validation_eval["pm2_5"]
        rows.append(
            self._evaluate_prediction(
                frame,
                "Persistence",
                target,
                persistence_prediction,
            )
        )

        ridge = Ridge(alpha=LINEAR_BASELINE_ALPHA)
        ridge.fit(
            train_eval[MODEL_FEATURE_COLUMNS],
            train_eval["target_pm2_5"],
        )
        ridge_prediction = ridge.predict(
            validation_eval[MODEL_FEATURE_COLUMNS],
        )
        rows.append(
            self._evaluate_prediction(
                frame,
                f"Ridge(alpha={LINEAR_BASELINE_ALPHA:g})",
                target,
                ridge_prediction,
            )
        )
        del ridge

        random_forest = RandomForestRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            max_features=RANDOM_FOREST_MAX_FEATURES,
            random_state=RANDOM_FOREST_RANDOM_STATE,
            n_jobs=RANDOM_FOREST_N_JOBS,
        )
        random_forest.fit(
            train_eval[MODEL_FEATURE_COLUMNS],
            train_eval["target_pm2_5"],
        )
        random_forest_prediction = random_forest.predict(
            validation_eval[MODEL_FEATURE_COLUMNS],
        )
        rows.append(
            self._evaluate_prediction(
                frame,
                "Random Forest",
                target,
                random_forest_prediction,
            )
        )
        del random_forest
        gc.collect()

        pooled_rows = []

        for row in rows:
            model = row["model"]
            if model == "Persistence":
                prediction = persistence_prediction
            elif model.startswith("Ridge"):
                prediction = ridge_prediction
            else:
                prediction = random_forest_prediction

            pooled_rows.extend(
                self._build_prediction_rows(
                    frame,
                    model,
                    target,
                    prediction,
                )
            )

        return rows, pooled_rows

    def _evaluate_prediction(self, frame, model, target, prediction):
        mae, rmse, r2 = self.evaluator.evaluate(
            target,
            prediction,
        )

        return {
            "fold": frame["fold"],
            "station": frame["station"],
            "model": model,
            "total_prepared_rows": frame["total_prepared_rows"],
            "train_raw_rows": frame["train_raw_rows"],
            "validation_raw_rows": frame["validation_raw_rows"],
            "train_rows": frame["train_rows"],
            "validation_rows": frame["validation_rows"],
            "validation_target_mean": target.mean(),
            "validation_target_std": target.std(),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    def _build_prediction_rows(self, frame, model, target, prediction):
        validation_eval = frame["validation_eval"]
        prediction_series = pd.Series(prediction).reset_index(drop=True)
        target_series = target.reset_index(drop=True)

        return [
            {
                "fold": frame["fold"],
                "station": frame["station"],
                "model": model,
                "source_index": int(source_index),
                "timestamp": timestamp,
                "target_pm2_5": float(target_value),
                "prediction": float(prediction_value),
            }
            for source_index, timestamp, target_value, prediction_value in zip(
                validation_eval["source_index"].to_list(),
                validation_eval["timestamp"].to_list(),
                target_series.to_list(),
                prediction_series.to_list(),
            )
        ]

    def _build_fold_summary(self, station_metrics_df, pooled_predictions_df):
        rows = []

        for fold in sorted(station_metrics_df["fold"].unique()):
            for model in MODELS:
                subset = station_metrics_df[
                    (station_metrics_df["fold"] == fold)
                    &
                    (station_metrics_df["model"] == model)
                ]
                prediction_subset = pooled_predictions_df[
                    (pooled_predictions_df["fold"] == fold)
                    &
                    (pooled_predictions_df["model"] == model)
                ]

                pooled_mae, pooled_rmse, pooled_r2 = self.evaluator.evaluate(
                    prediction_subset["target_pm2_5"],
                    prediction_subset["prediction"],
                )

                rows.append({
                    "fold": fold,
                    "model": model,
                    "datasets": len(subset),
                    "train_rows": int(subset["train_rows"].sum()),
                    "validation_rows": int(
                        subset["validation_rows"].sum()
                    ),
                    "macro_mae": subset["mae"].mean(),
                    "macro_rmse": subset["rmse"].mean(),
                    "macro_mean_r2": subset["r2"].mean(),
                    "macro_median_r2": subset["r2"].median(),
                    "pooled_mae": pooled_mae,
                    "pooled_rmse": pooled_rmse,
                    "pooled_r2": pooled_r2,
                    "negative_r2_datasets": int(
                        (subset["r2"] < 0).sum()
                    ),
                })

        return pd.DataFrame(rows)

    def _build_model_comparisons(self, station_metrics_df, fold_summary_df):
        rows = []

        for fold in sorted(station_metrics_df["fold"].unique()):
            persistence_summary = fold_summary_df[
                (fold_summary_df["fold"] == fold)
                &
                (fold_summary_df["model"] == "Persistence")
            ].iloc[0]

            persistence_station = station_metrics_df[
                (station_metrics_df["fold"] == fold)
                &
                (station_metrics_df["model"] == "Persistence")
            ][["station", "rmse"]].rename(
                columns={"rmse": "persistence_rmse"}
            )

            for model in [
                f"Ridge(alpha={LINEAR_BASELINE_ALPHA:g})",
                "Random Forest",
            ]:
                model_summary = fold_summary_df[
                    (fold_summary_df["fold"] == fold)
                    &
                    (fold_summary_df["model"] == model)
                ].iloc[0]

                model_station = station_metrics_df[
                    (station_metrics_df["fold"] == fold)
                    &
                    (station_metrics_df["model"] == model)
                ][["station", "rmse"]].rename(
                    columns={"rmse": "model_rmse"}
                )

                comparison = persistence_station.merge(
                    model_station,
                    on="station",
                    how="inner",
                )
                rmse_improvement = (
                    persistence_summary["pooled_rmse"] -
                    model_summary["pooled_rmse"]
                )

                rows.append({
                    "fold": fold,
                    "model": model,
                    "sign_convention": "positive means model lower RMSE than persistence",
                    "persistence_pooled_rmse": (
                        persistence_summary["pooled_rmse"]
                    ),
                    "model_pooled_rmse": model_summary["pooled_rmse"],
                    "pooled_rmse_improvement": rmse_improvement,
                    "pooled_rmse_percent_improvement": (
                        rmse_improvement /
                        persistence_summary["pooled_rmse"] *
                        100
                    ),
                    "station_wins": int(
                        (
                            comparison["model_rmse"] <
                            comparison["persistence_rmse"]
                        ).sum()
                    ),
                    "station_losses": int(
                        (
                            comparison["model_rmse"] >
                            comparison["persistence_rmse"]
                        ).sum()
                    ),
                    "station_ties": int(
                        (
                            comparison["model_rmse"] ==
                            comparison["persistence_rmse"]
                        ).sum()
                    ),
                })

        return pd.DataFrame(rows)

    def _build_rf_station_wins(self, station_metrics_df):
        persistence = station_metrics_df[
            station_metrics_df["model"] == "Persistence"
        ][["fold", "station", "rmse"]].rename(
            columns={"rmse": "persistence_rmse"}
        )
        random_forest = station_metrics_df[
            station_metrics_df["model"] == "Random Forest"
        ][["fold", "station", "rmse"]].rename(
            columns={"rmse": "random_forest_rmse"}
        )

        comparison = persistence.merge(
            random_forest,
            on=["fold", "station"],
            how="inner",
        )
        comparison["rf_wins_fold"] = (
            comparison["random_forest_rmse"] <
            comparison["persistence_rmse"]
        )
        comparison["rmse_improvement"] = (
            comparison["persistence_rmse"] -
            comparison["random_forest_rmse"]
        )

        wins = (
            comparison
            .groupby("station")
            .agg(
                rf_win_folds=("rf_wins_fold", "sum"),
                folds_observed=("fold", "count"),
                mean_rmse_improvement=("rmse_improvement", "mean"),
                median_rmse_improvement=("rmse_improvement", "median"),
            )
            .reset_index()
        )
        wins["rf_win_bucket"] = wins["rf_win_folds"].astype(int).astype(str)
        wins["rf_win_bucket"] = wins["rf_win_bucket"] + "/3"

        return wins.sort_values(
            ["rf_win_folds", "mean_rmse_improvement"],
            ascending=[False, False],
        )

    def _build_distribution_shifts(self, station_fold_frames):
        rows = []

        for frame in station_fold_frames:
            train_target = frame["train_eval"]["target_pm2_5"]
            validation_target = frame["validation_eval"]["target_pm2_5"]

            rows.append({
                "fold": frame["fold"],
                "station": frame["station"],
                "train_rows": frame["train_rows"],
                "validation_rows": frame["validation_rows"],
                "train_target_mean": train_target.mean(),
                "validation_target_mean": validation_target.mean(),
                "target_mean_shift": (
                    validation_target.mean() -
                    train_target.mean()
                ),
                "train_target_std": train_target.std(),
                "validation_target_std": validation_target.std(),
                "target_std_ratio": (
                    validation_target.std() /
                    train_target.std()
                    if train_target.std()
                    else None
                ),
                "validation_timestamp_min": (
                    frame["validation_eval"]["timestamp"].min()
                ),
                "validation_timestamp_max": (
                    frame["validation_eval"]["timestamp"].max()
                ),
            })

        shifts = pd.DataFrame(rows)
        shifts["abs_target_mean_shift"] = (
            shifts["target_mean_shift"].abs()
        )

        return shifts.sort_values(
            "abs_target_mean_shift",
            ascending=False,
        )

    @staticmethod
    def _build_fold_frame_summary(station_fold_frames):
        rows = [
            {
                "fold": frame["fold"],
                "station": frame["station"],
                "total_prepared_rows": frame["total_prepared_rows"],
                "train_raw_rows": frame["train_raw_rows"],
                "validation_raw_rows": frame["validation_raw_rows"],
                "train_rows": frame["train_rows"],
                "validation_rows": frame["validation_rows"],
                "train_start_percent": frame["train_start_percent"],
                "train_end_percent": frame["train_end_percent"],
                "validation_start_percent": (
                    frame["validation_start_percent"]
                ),
                "validation_end_percent": frame["validation_end_percent"],
            }
            for frame in station_fold_frames
        ]

        return pd.DataFrame(rows)

    def _write_outputs(
            self,
            station_metrics_df,
            fold_summary_df,
            model_comparison_df,
            rf_station_wins_df,
            distribution_shift_df,
            fold_frame_summary_df,
    ):
        station_metrics_df.to_csv(
            self.output_dir / "rolling_origin_station_metrics.csv",
            index=False,
        )
        fold_summary_df.to_csv(
            self.output_dir / "rolling_origin_fold_summary.csv",
            index=False,
        )
        model_comparison_df.to_csv(
            self.output_dir / "rolling_origin_model_comparisons.csv",
            index=False,
        )
        rf_station_wins_df.to_csv(
            self.output_dir / "rolling_origin_rf_station_wins.csv",
            index=False,
        )
        distribution_shift_df.to_csv(
            self.output_dir / "rolling_origin_distribution_shifts.csv",
            index=False,
        )
        fold_frame_summary_df.to_csv(
            self.output_dir / "rolling_origin_fold_frames.csv",
            index=False,
        )


def main():
    validator = RollingOriginValidation()
    validator.run()


if __name__ == "__main__":
    main()
