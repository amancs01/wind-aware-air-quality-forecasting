import gc
import json

import pandas as pd
from xgboost import XGBRegressor

from config import(
    LINEAR_RESULTS_DIR,
    ML_VALIDATION_DIR,
    MODEL_FEATURE_COLUMNS,
    RANDOM_FOREST_RESULTS_DIR,
    TRAIN_DIR,
    XGBOOST_RESULTS_DIR,
)
from logger import logger
from models.base_model import BaseModel


N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_ALPHA = 0.0
REG_LAMBDA = 1.0
OBJECTIVE = "reg:squarederror"
TREE_METHOD = "hist"
EVAL_METRIC = "rmse"
RANDOM_STATE = 42
N_JOBS = 1
PRIMARY_SELECTION_METRIC = "pooled_rmse"

LEARNING_RATE_GRID = [
    0.03,
    0.10,
]
MAX_DEPTH_GRID = [
    3,
    6,
]
MIN_CHILD_WEIGHT_GRID = [
    1,
    5,
]


def build_candidates():
    candidates = []

    for learning_rate in LEARNING_RATE_GRID:
        for max_depth in MAX_DEPTH_GRID:
            for min_child_weight in MIN_CHILD_WEIGHT_GRID:
                candidates.append({
                    "candidate_id": format_candidate_id(
                        learning_rate,
                        max_depth,
                        min_child_weight,
                    ),
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "min_child_weight": min_child_weight,
                    "subsample": SUBSAMPLE,
                    "colsample_bytree": COLSAMPLE_BYTREE,
                    "reg_alpha": REG_ALPHA,
                    "reg_lambda": REG_LAMBDA,
                    "objective": OBJECTIVE,
                    "tree_method": TREE_METHOD,
                    "eval_metric": EVAL_METRIC,
                    "n_estimators": N_ESTIMATORS,
                    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                    "random_state": RANDOM_STATE,
                    "n_jobs": N_JOBS,
                })

    return candidates


def format_candidate_id(learning_rate, max_depth, min_child_weight):
    rate_text = str(learning_rate).replace(".", "p")

    return (
        f"xgb_lr_{rate_text}_depth_{max_depth}_"
        f"child_{min_child_weight}"
    )


def make_estimator(candidate):
    return XGBRegressor(
        n_estimators=candidate["n_estimators"],
        learning_rate=candidate["learning_rate"],
        max_depth=candidate["max_depth"],
        min_child_weight=candidate["min_child_weight"],
        subsample=candidate["subsample"],
        colsample_bytree=candidate["colsample_bytree"],
        reg_alpha=candidate["reg_alpha"],
        reg_lambda=candidate["reg_lambda"],
        objective=candidate["objective"],
        tree_method=candidate["tree_method"],
        eval_metric=candidate["eval_metric"],
        early_stopping_rounds=candidate["early_stopping_rounds"],
        random_state=candidate["random_state"],
        n_jobs=candidate["n_jobs"],
    )


def load_station_frames():
    evaluator = BaseModel(XGBOOST_RESULTS_DIR)
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
    evaluator = BaseModel(XGBOOST_RESULTS_DIR)
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
        "best_iteration_min": int(metrics_df["best_iteration"].min()),
        "best_iteration_median": metrics_df["best_iteration"].median(),
        "best_iteration_mean": metrics_df["best_iteration"].mean(),
        "best_iteration_max": int(metrics_df["best_iteration"].max()),
        "hit_n_estimators_limit": int(
            (metrics_df["best_iteration"] >= N_ESTIMATORS - 1).sum()
        ),
    }


def add_target_stats(station_row, validation_eval):
    target = validation_eval["target_pm2_5"]
    station_row["target_mean"] = target.mean()
    station_row["target_std"] = target.std()

    return station_row


def evaluate_candidate(candidate, station_frames):
    evaluator = BaseModel(XGBOOST_RESULTS_DIR)
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
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )
        prediction = estimator.predict(X_validation)
        mae, rmse, r2 = evaluator.evaluate(
            y_validation,
            prediction,
        )

        station_row = {
            "candidate_id": candidate["candidate_id"],
            "model": "XGBoost",
            "station": station,
            "original_train_rows": len(train_df),
            "valid_train_rows": len(train_eval),
            "original_validation_rows": len(validation_df),
            "valid_validation_rows": len(validation_eval),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "best_iteration": getattr(estimator, "best_iteration", None),
            "best_score": getattr(estimator, "best_score", None),
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
        "model": "XGBoost",
        "learning_rate": candidate["learning_rate"],
        "max_depth": candidate["max_depth"],
        "min_child_weight": candidate["min_child_weight"],
        "subsample": candidate["subsample"],
        "colsample_bytree": candidate["colsample_bytree"],
        "reg_alpha": candidate["reg_alpha"],
        "reg_lambda": candidate["reg_lambda"],
        "n_estimators": candidate["n_estimators"],
        "early_stopping_rounds": candidate["early_stopping_rounds"],
        "tree_method": candidate["tree_method"],
        "random_state": candidate["random_state"],
        "n_jobs": candidate["n_jobs"],
    })

    return metrics_df, summary


def load_baseline_summary():
    baseline_rows = []

    linear_tuning = pd.read_csv(
        LINEAR_RESULTS_DIR / "validation_tuning.csv",
    )
    rf_tuning = pd.read_csv(
        RANDOM_FOREST_RESULTS_DIR / "validation_tuning.csv",
    )

    persistence = linear_tuning[
        linear_tuning["candidate_id"] == "persistence"
    ].iloc[0].to_dict()
    persistence.update({
        "model": "Persistence",
        "candidate_id": "persistence",
    })
    baseline_rows.append(persistence)

    ridge = linear_tuning[
        linear_tuning["candidate_id"] == "ridge_none_alpha_1000p0"
    ].iloc[0].to_dict()
    ridge.update({
        "model": "Ridge",
        "candidate_id": "ridge_alpha_1000",
    })
    baseline_rows.append(ridge)

    random_forest = rf_tuning[
        rf_tuning["candidate_id"] ==
        "rf_depth_10_leaf_10_features_1p0"
    ].iloc[0].to_dict()
    random_forest.update({
        "model": "RandomForest",
        "candidate_id": "random_forest_selected",
    })
    baseline_rows.append(random_forest)

    return baseline_rows


def load_baseline_station_metrics():
    rf_metrics = pd.read_csv(
        RANDOM_FOREST_RESULTS_DIR / "validation_tuning_station_metrics.csv",
    )
    persistence = rf_metrics[
        rf_metrics["candidate_id"] == "persistence"
    ].copy()
    random_forest = rf_metrics[
        rf_metrics["candidate_id"] ==
        "rf_depth_10_leaf_10_features_1p0"
    ].copy()

    return persistence, random_forest


def write_outputs(station_metrics, summaries):
    metrics_df = pd.concat(
        station_metrics,
        ignore_index=True,
    )
    xgb_summary_df = pd.DataFrame(summaries)
    baseline_summary_df = pd.DataFrame(load_baseline_summary())
    summary_df = pd.concat(
        [
            baseline_summary_df,
            xgb_summary_df,
        ],
        ignore_index=True,
        sort=False,
    )

    summary_cols = [
        "candidate_id",
        "model",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "n_estimators",
        "early_stopping_rounds",
        "tree_method",
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
        "best_iteration_min",
        "best_iteration_median",
        "best_iteration_mean",
        "best_iteration_max",
        "hit_n_estimators_limit",
    ]
    summary_df = summary_df.reindex(columns=summary_cols)

    metrics_df.to_csv(
        XGBOOST_RESULTS_DIR / "validation_tuning_station_metrics.csv",
        index=False,
    )
    summary_df.to_csv(
        XGBOOST_RESULTS_DIR / "validation_tuning.csv",
        index=False,
    )

    return metrics_df, summary_df


def select_best_xgboost(summary_df):
    xgb_candidates = summary_df[
        summary_df["model"] == "XGBoost"
    ].copy()

    return xgb_candidates.sort_values(
        [
            PRIMARY_SELECTION_METRIC,
            "macro_rmse",
            "pooled_mae",
        ],
        ascending=True,
    ).iloc[0]


def build_comparison_df(metrics_df, baseline_metrics, selected_id):
    persistence_metrics, random_forest_metrics = baseline_metrics

    selected_metrics = metrics_df[
        metrics_df["candidate_id"] == selected_id
    ][[
        "station",
        "valid_validation_rows",
        "rmse",
        "target_mean",
        "target_std",
    ]].rename(
        columns={"rmse": "xgboost_rmse"}
    )
    persistence = persistence_metrics[[
        "station",
        "rmse",
    ]].rename(
        columns={"rmse": "persistence_rmse"}
    )
    random_forest = random_forest_metrics[[
        "station",
        "rmse",
    ]].rename(
        columns={"rmse": "random_forest_rmse"}
    )

    comparison_df = selected_metrics.merge(
        persistence,
        on="station",
        how="inner",
    ).merge(
        random_forest,
        on="station",
        how="inner",
    )
    comparison_df["xgb_delta_vs_persistence"] = (
        comparison_df["xgboost_rmse"] -
        comparison_df["persistence_rmse"]
    )
    comparison_df["xgb_delta_vs_random_forest"] = (
        comparison_df["xgboost_rmse"] -
        comparison_df["random_forest_rmse"]
    )
    comparison_df["winner_vs_persistence"] = comparison_df[
        "xgb_delta_vs_persistence"
    ].apply(
        lambda value: "xgboost"
        if value < 0
        else "persistence"
    )
    comparison_df["winner_vs_random_forest"] = comparison_df[
        "xgb_delta_vs_random_forest"
    ].apply(
        lambda value: "xgboost"
        if value < 0
        else "random_forest"
    )

    return comparison_df.sort_values(
        "xgb_delta_vs_persistence",
    )


def write_selection_outputs(metrics_df, summary_df, selected):
    selected_id = selected["candidate_id"]
    comparison_df = build_comparison_df(
        metrics_df,
        load_baseline_station_metrics(),
        selected_id,
    )
    comparison_df.to_csv(
        XGBOOST_RESULTS_DIR / "validation_selected_comparison.csv",
        index=False,
    )

    selected_config = {
        "candidate_id": selected_id,
        "model": "XGBoost",
        "selection_split": "validation",
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "primary_selection_value": float(selected[PRIMARY_SELECTION_METRIC]),
        "learning_rate": float(selected["learning_rate"]),
        "max_depth": int(selected["max_depth"]),
        "min_child_weight": int(selected["min_child_weight"]),
        "subsample": float(selected["subsample"]),
        "colsample_bytree": float(selected["colsample_bytree"]),
        "reg_alpha": float(selected["reg_alpha"]),
        "reg_lambda": float(selected["reg_lambda"]),
        "n_estimators": int(selected["n_estimators"]),
        "early_stopping_rounds": int(selected["early_stopping_rounds"]),
        "tree_method": selected["tree_method"],
        "eval_metric": EVAL_METRIC,
        "objective": OBJECTIVE,
        "random_state": int(selected["random_state"]),
        "n_jobs": int(selected["n_jobs"]),
        "best_iteration_min": int(selected["best_iteration_min"]),
        "best_iteration_median": float(selected["best_iteration_median"]),
        "best_iteration_mean": float(selected["best_iteration_mean"]),
        "best_iteration_max": int(selected["best_iteration_max"]),
        "hit_n_estimators_limit": int(selected["hit_n_estimators_limit"]),
    }

    with (XGBOOST_RESULTS_DIR / "selected_xgboost_config.json").open(
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
    candidates = build_candidates()

    for index, candidate in enumerate(candidates, start=1):
        logger.info(
            f"Evaluating XGBoost candidate {index}/{len(candidates)}: "
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
    selected = select_best_xgboost(summary_df)
    comparison_df, selected_config = write_selection_outputs(
        metrics_df,
        summary_df,
        selected,
    )

    logger.info(
        "XGBoost validation tuning complete. "
        f"Selected {selected_config['candidate_id']} by "
        f"{PRIMARY_SELECTION_METRIC}="
        f"{selected_config['primary_selection_value']:.3f}"
    )
    logger.info(
        "XGBoost beats persistence on "
        f"{(comparison_df['winner_vs_persistence'] == 'xgboost').sum()} "
        "datasets; XGBoost beats Random Forest on "
        f"{(comparison_df['winner_vs_random_forest'] == 'xgboost').sum()} "
        "datasets."
    )


if __name__ == "__main__":

    main()
