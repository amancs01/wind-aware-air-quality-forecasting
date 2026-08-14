import json
import gc

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import(
    LINEAR_BASELINE_ALPHA,
    ML_VALIDATION_DIR,
    MODEL_FEATURE_COLUMNS,
    RANDOM_FOREST_RESULTS_DIR,
    TRAIN_DIR,
)
from logger import logger
from models.base_model import BaseModel
from models.linear_regression import LinearRegressionModel


N_ESTIMATORS = 100
RANDOM_STATE = 42
N_JOBS = 1
PRIMARY_SELECTION_METRIC = "pooled_rmse"

MAX_DEPTH_GRID = [
    10,
    20,
]
MIN_SAMPLES_LEAF_GRID = [
    1,
    5,
    10,
]
MAX_FEATURES_GRID = [
    1.0,
    "sqrt",
]


def build_candidates():
    candidates = []

    for max_depth in MAX_DEPTH_GRID:
        for min_samples_leaf in MIN_SAMPLES_LEAF_GRID:
            for max_features in MAX_FEATURES_GRID:
                candidates.append({
                    "candidate_id": format_candidate_id(
                        max_depth,
                        min_samples_leaf,
                        max_features,
                    ),
                    "n_estimators": N_ESTIMATORS,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "random_state": RANDOM_STATE,
                    "n_jobs": N_JOBS,
                })

    return candidates


def format_candidate_id(max_depth, min_samples_leaf, max_features):
    depth_text = "none" if max_depth is None else str(max_depth)
    features_text = str(max_features).replace(".", "p")

    return (
        f"rf_depth_{depth_text}_leaf_{min_samples_leaf}_"
        f"features_{features_text}"
    )


def make_estimator(candidate):
    return RandomForestRegressor(
        n_estimators=candidate["n_estimators"],
        max_depth=candidate["max_depth"],
        min_samples_leaf=candidate["min_samples_leaf"],
        max_features=candidate["max_features"],
        random_state=candidate["random_state"],
        n_jobs=candidate["n_jobs"],
    )


def load_station_frames():
    evaluator = BaseModel(RANDOM_FOREST_RESULTS_DIR)
    station_frames = []
    required_columns = MODEL_FEATURE_COLUMNS + [
        "target_pm2_5",
        "timestamp",
    ]

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
        train_eval = evaluator.prepare_evaluation_frame(train_df)
        validation_eval = evaluator.prepare_evaluation_frame(validation_df)

        if train_eval.empty or validation_eval.empty:
            logger.warning(
                f"Skipping {train_file.stem}: no valid train/validation rows"
            )
            continue

        station_frames.append((
            train_file.stem,
            train_df,
            validation_df,
            train_eval,
            validation_eval,
        ))

    return station_frames


def summarize_metrics(metrics_df, pooled_targets, pooled_predictions):
    evaluator = BaseModel(RANDOM_FOREST_RESULTS_DIR)
    pooled_mae, pooled_rmse, pooled_r2 = evaluator.evaluate(
        pooled_targets,
        pooled_predictions,
    )

    original_train_rows = int(metrics_df["original_train_rows"].sum())
    valid_train_rows = int(metrics_df["valid_train_rows"].sum())
    original_validation_rows = int(
        metrics_df["original_validation_rows"].sum()
    )
    valid_validation_rows = int(
        metrics_df["valid_validation_rows"].sum()
    )

    return {
        "evaluated_datasets": len(metrics_df),
        "original_train_rows": original_train_rows,
        "valid_train_rows": valid_train_rows,
        "removed_train_rows": original_train_rows - valid_train_rows,
        "train_coverage_percent": (
            valid_train_rows / original_train_rows * 100
            if original_train_rows
            else 0.0
        ),
        "original_validation_rows": original_validation_rows,
        "valid_validation_rows": valid_validation_rows,
        "removed_validation_rows": (
            original_validation_rows - valid_validation_rows
        ),
        "validation_coverage_percent": (
            valid_validation_rows / original_validation_rows * 100
            if original_validation_rows
            else 0.0
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
    }


def add_target_stats(station_row, validation_eval):
    target = validation_eval["target_pm2_5"]
    station_row["target_mean"] = target.mean()
    station_row["target_std"] = target.std()

    return station_row


def evaluate_persistence(station_frames):
    evaluator = BaseModel(RANDOM_FOREST_RESULTS_DIR)
    station_rows = []
    pooled_targets = []
    pooled_predictions = []

    for station, train_df, validation_df, train_eval, validation_eval in station_frames:
        prediction = validation_eval["pm2_5"]
        target = validation_eval["target_pm2_5"]
        mae, rmse, r2 = evaluator.evaluate(
            target,
            prediction,
        )

        station_row = {
            "candidate_id": "persistence",
            "model": "Persistence",
            "station": station,
            "original_train_rows": len(train_df),
            "valid_train_rows": len(train_eval),
            "original_validation_rows": len(validation_df),
            "valid_validation_rows": len(validation_eval),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
        station_rows.append(
            add_target_stats(
                station_row,
                validation_eval,
            )
        )
        pooled_targets.extend(target.to_list())
        pooled_predictions.extend(prediction.to_list())

    metrics_df = pd.DataFrame(station_rows)
    summary = summarize_metrics(
        metrics_df,
        pooled_targets,
        pooled_predictions,
    )
    summary.update({
        "candidate_id": "persistence",
        "model": "Persistence",
        "n_estimators": None,
        "max_depth": None,
        "min_samples_leaf": None,
        "max_features": None,
        "random_state": None,
        "n_jobs": None,
    })

    return metrics_df, summary


def evaluate_ridge(station_frames):
    evaluator = BaseModel(RANDOM_FOREST_RESULTS_DIR)
    station_rows = []
    pooled_targets = []
    pooled_predictions = []

    for station, train_df, validation_df, train_eval, validation_eval in station_frames:
        model = LinearRegressionModel()
        X_train, y_train = evaluator.split_features_target(train_eval)
        X_validation, y_validation = evaluator.split_features_target(
            validation_eval,
        )

        model.fit(
            X_train,
            y_train,
        )
        prediction = model.predict(X_validation)
        mae, rmse, r2 = evaluator.evaluate(
            y_validation,
            prediction,
        )

        station_row = {
            "candidate_id": "ridge_alpha_1000",
            "model": "Ridge",
            "station": station,
            "original_train_rows": len(train_df),
            "valid_train_rows": len(train_eval),
            "original_validation_rows": len(validation_df),
            "valid_validation_rows": len(validation_eval),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
        station_rows.append(
            add_target_stats(
                station_row,
                validation_eval,
            )
        )
        pooled_targets.extend(y_validation.to_list())
        pooled_predictions.extend(pd.Series(prediction).to_list())

    metrics_df = pd.DataFrame(station_rows)
    summary = summarize_metrics(
        metrics_df,
        pooled_targets,
        pooled_predictions,
    )
    summary.update({
        "candidate_id": "ridge_alpha_1000",
        "model": "Ridge",
        "n_estimators": None,
        "max_depth": None,
        "min_samples_leaf": None,
        "max_features": None,
        "random_state": None,
        "n_jobs": None,
    })

    return metrics_df, summary


def evaluate_candidate(candidate, station_frames):
    evaluator = BaseModel(RANDOM_FOREST_RESULTS_DIR)
    station_rows = []
    pooled_targets = []
    pooled_predictions = []

    for station, train_df, validation_df, train_eval, validation_eval in station_frames:
        X_train, y_train = evaluator.split_features_target(train_eval)
        X_validation, y_validation = evaluator.split_features_target(
            validation_eval,
        )
        estimator = make_estimator(candidate)

        estimator.fit(
            X_train,
            y_train,
        )
        prediction = estimator.predict(X_validation)
        mae, rmse, r2 = evaluator.evaluate(
            y_validation,
            prediction,
        )

        station_row = {
            "candidate_id": candidate["candidate_id"],
            "model": "RandomForest",
            "station": station,
            "original_train_rows": len(train_df),
            "valid_train_rows": len(train_eval),
            "original_validation_rows": len(validation_df),
            "valid_validation_rows": len(validation_eval),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
        station_rows.append(
            add_target_stats(
                station_row,
                validation_eval,
            )
        )
        pooled_targets.extend(y_validation.to_list())
        pooled_predictions.extend(pd.Series(prediction).to_list())
        del estimator
        gc.collect()

    metrics_df = pd.DataFrame(station_rows)
    summary = summarize_metrics(
        metrics_df,
        pooled_targets,
        pooled_predictions,
    )
    summary.update({
        "candidate_id": candidate["candidate_id"],
        "model": "RandomForest",
        "n_estimators": candidate["n_estimators"],
        "max_depth": candidate["max_depth"],
        "min_samples_leaf": candidate["min_samples_leaf"],
        "max_features": candidate["max_features"],
        "random_state": candidate["random_state"],
        "n_jobs": candidate["n_jobs"],
    })

    return metrics_df, summary


def write_outputs(station_metrics, summaries):
    metrics_df = pd.concat(
        station_metrics,
        ignore_index=True,
    )
    summary_df = pd.DataFrame(summaries)

    summary_cols = [
        "candidate_id",
        "model",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "random_state",
        "n_jobs",
        "evaluated_datasets",
        "original_train_rows",
        "valid_train_rows",
        "removed_train_rows",
        "train_coverage_percent",
        "original_validation_rows",
        "valid_validation_rows",
        "removed_validation_rows",
        "validation_coverage_percent",
        "macro_mae",
        "macro_rmse",
        "macro_mean_r2",
        "macro_median_r2",
        "pooled_mae",
        "pooled_rmse",
        "pooled_r2",
        "negative_r2_datasets",
        "positive_r2_datasets",
    ]
    summary_df = summary_df[summary_cols]

    metrics_df.to_csv(
        RANDOM_FOREST_RESULTS_DIR / "validation_tuning_station_metrics.csv",
        index=False,
    )
    summary_df.to_csv(
        RANDOM_FOREST_RESULTS_DIR / "validation_tuning.csv",
        index=False,
    )

    return metrics_df, summary_df


def select_best_random_forest(summary_df):
    rf_candidates = summary_df[
        summary_df["model"] == "RandomForest"
    ].copy()

    return rf_candidates.sort_values(
        [
            PRIMARY_SELECTION_METRIC,
            "macro_rmse",
            "pooled_mae",
        ],
        ascending=True,
    ).iloc[0]


def write_selection_outputs(metrics_df, selected):
    selected_id = selected["candidate_id"]
    selected_station_metrics = metrics_df[
        metrics_df["candidate_id"] == selected_id
    ][[
        "station",
        "valid_validation_rows",
        "rmse",
        "target_mean",
        "target_std",
    ]].rename(
        columns={"rmse": "random_forest_rmse"}
    )
    persistence_station_metrics = metrics_df[
        metrics_df["candidate_id"] == "persistence"
    ][["station", "rmse"]].rename(
        columns={"rmse": "persistence_rmse"}
    )

    comparison_df = selected_station_metrics.merge(
        persistence_station_metrics,
        on="station",
        how="inner",
    )
    comparison_df["rmse_delta_vs_persistence"] = (
        comparison_df["random_forest_rmse"] -
        comparison_df["persistence_rmse"]
    )
    comparison_df["winner"] = comparison_df[
        "rmse_delta_vs_persistence"
    ].apply(
        lambda value: "random_forest"
        if value < 0
        else "persistence"
    )
    comparison_df = comparison_df.sort_values(
        "rmse_delta_vs_persistence",
    )

    comparison_df.to_csv(
        RANDOM_FOREST_RESULTS_DIR /
        "validation_selected_vs_persistence.csv",
        index=False,
    )

    selected_config = {
        "candidate_id": selected_id,
        "model": "RandomForest",
        "selection_split": "validation",
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "primary_selection_value": float(selected[PRIMARY_SELECTION_METRIC]),
        "n_estimators": int(selected["n_estimators"]),
        "max_depth": (
            None
            if pd.isna(selected["max_depth"])
            else int(selected["max_depth"])
        ),
        "min_samples_leaf": int(selected["min_samples_leaf"]),
        "max_features": selected["max_features"],
        "random_state": int(selected["random_state"]),
        "n_jobs": int(selected["n_jobs"]),
        "ridge_baseline_alpha": LINEAR_BASELINE_ALPHA,
    }

    with (RANDOM_FOREST_RESULTS_DIR / "selected_random_forest_config.json").open(
            "w",
            encoding="utf-8",
    ) as config_file:
        json.dump(
            selected_config,
            config_file,
            indent=2,
        )

    return comparison_df, selected_config


def main():
    station_frames = load_station_frames()
    logger.info(
        f"Loaded {len(station_frames)} train/validation station pairs"
    )

    station_metrics = []
    summaries = []

    persistence_metrics, persistence_summary = evaluate_persistence(
        station_frames,
    )
    station_metrics.append(persistence_metrics)
    summaries.append(persistence_summary)

    ridge_metrics, ridge_summary = evaluate_ridge(
        station_frames,
    )
    station_metrics.append(ridge_metrics)
    summaries.append(ridge_summary)

    candidates = build_candidates()

    for index, candidate in enumerate(candidates, start=1):
        logger.info(
            f"Evaluating RF candidate {index}/{len(candidates)}: "
            f"{candidate['candidate_id']}"
        )
        metrics_df, summary = evaluate_candidate(
            candidate,
            station_frames,
        )
        station_metrics.append(metrics_df)
        summaries.append(summary)

    metrics_df, summary_df = write_outputs(
        station_metrics,
        summaries,
    )
    selected = select_best_random_forest(summary_df)
    comparison_df, selected_config = write_selection_outputs(
        metrics_df,
        selected,
    )

    logger.info(
        "Random Forest validation tuning complete. "
        f"Selected {selected_config['candidate_id']} by "
        f"{PRIMARY_SELECTION_METRIC}="
        f"{selected_config['primary_selection_value']:.3f}"
    )
    logger.info(
        "Random Forest beats persistence on "
        f"{(comparison_df['winner'] == 'random_forest').sum()} datasets; "
        "persistence wins on "
        f"{(comparison_df['winner'] == 'persistence').sum()} datasets."
    )


if __name__ == "__main__":

    main()
