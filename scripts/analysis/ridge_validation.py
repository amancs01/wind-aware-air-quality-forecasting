import json

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import(
    LINEAR_RESULTS_DIR,
    ML_VALIDATION_DIR,
    TRAIN_DIR,
)
from logger import logger
from models.base_model import BaseModel


ALPHA_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

PRIMARY_SELECTION_METRIC = "pooled_rmse"


def build_candidates():
    candidates = [{
        "model": "LinearRegression",
        "alpha": None,
        "scaling": "none",
        "estimator": LinearRegression(),
    }]

    for alpha in ALPHA_GRID:
        candidates.append({
            "model": "Ridge",
            "alpha": alpha,
            "scaling": "none",
            "estimator": Ridge(alpha=alpha),
        })

    for alpha in ALPHA_GRID:
        candidates.append({
            "model": "Ridge",
            "alpha": alpha,
            "scaling": "standard",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]),
        })

    return candidates


def summarize_metrics(metrics_df, pooled_targets, pooled_predictions):
    evaluator = BaseModel(LINEAR_RESULTS_DIR)
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


def evaluate_persistence(station_frames):
    evaluator = BaseModel(LINEAR_RESULTS_DIR)
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

        station_rows.append({
            "candidate_id": "persistence",
            "model": "Persistence",
            "alpha": None,
            "scaling": "none",
            "station": station,
            "original_train_rows": len(train_df),
            "valid_train_rows": len(train_eval),
            "original_validation_rows": len(validation_df),
            "valid_validation_rows": len(validation_eval),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })
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
        "alpha": None,
        "scaling": "none",
    })

    return metrics_df, summary


def evaluate_candidate(candidate, station_frames):
    evaluator = BaseModel(LINEAR_RESULTS_DIR)
    station_rows = []
    pooled_targets = []
    pooled_predictions = []
    candidate_id = format_candidate_id(candidate)

    for station, train_df, validation_df, train_eval, validation_eval in station_frames:
        X_train, y_train = evaluator.split_features_target(train_eval)
        X_validation, y_validation = evaluator.split_features_target(
            validation_eval,
        )

        estimator = candidate["estimator"]
        estimator.fit(
            X_train,
            y_train,
        )
        prediction = estimator.predict(X_validation)

        mae, rmse, r2 = evaluator.evaluate(
            y_validation,
            prediction,
        )

        station_rows.append({
            "candidate_id": candidate_id,
            "model": candidate["model"],
            "alpha": candidate["alpha"],
            "scaling": candidate["scaling"],
            "station": station,
            "original_train_rows": len(train_df),
            "valid_train_rows": len(train_eval),
            "original_validation_rows": len(validation_df),
            "valid_validation_rows": len(validation_eval),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })
        pooled_targets.extend(y_validation.to_list())
        pooled_predictions.extend(pd.Series(prediction).to_list())

    metrics_df = pd.DataFrame(station_rows)
    summary = summarize_metrics(
        metrics_df,
        pooled_targets,
        pooled_predictions,
    )
    summary.update({
        "candidate_id": candidate_id,
        "model": candidate["model"],
        "alpha": candidate["alpha"],
        "scaling": candidate["scaling"],
    })

    return metrics_df, summary


def format_candidate_id(candidate):
    if candidate["model"] == "LinearRegression":
        return "linear_regression"

    alpha_text = str(candidate["alpha"]).replace(".", "p")
    return f"ridge_{candidate['scaling']}_alpha_{alpha_text}"


def load_station_frames():
    evaluator = BaseModel(LINEAR_RESULTS_DIR)
    station_frames = []

    for train_file in sorted(TRAIN_DIR.glob("*.csv")):
        validation_file = ML_VALIDATION_DIR / train_file.name

        if not validation_file.exists():
            logger.warning(
                f"Skipping {train_file.stem}: validation file missing"
            )
            continue

        train_df = evaluator.load_dataset(train_file)
        validation_df = evaluator.load_dataset(validation_file)
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


def write_outputs(station_metrics, summaries):
    metrics_df = pd.concat(
        station_metrics,
        ignore_index=True,
    )
    summary_df = pd.DataFrame(summaries)

    summary_cols = [
        "candidate_id",
        "model",
        "alpha",
        "scaling",
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
        LINEAR_RESULTS_DIR / "validation_tuning_station_metrics.csv",
        index=False,
    )
    summary_df.to_csv(
        LINEAR_RESULTS_DIR / "validation_tuning.csv",
        index=False,
    )

    return metrics_df, summary_df


def select_best_linear(summary_df):
    linear_candidates = summary_df[
        summary_df["model"] != "Persistence"
    ].copy()

    return linear_candidates.sort_values(
        [
            PRIMARY_SELECTION_METRIC,
            "macro_rmse",
            "pooled_mae",
        ],
        ascending=True,
    ).iloc[0]


def write_selection_outputs(metrics_df, summary_df, selected):
    selected_id = selected["candidate_id"]
    selected_station_metrics = metrics_df[
        metrics_df["candidate_id"] == selected_id
    ][["station", "rmse"]].rename(
        columns={"rmse": "selected_rmse"}
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
        comparison_df["selected_rmse"] -
        comparison_df["persistence_rmse"]
    )
    comparison_df["winner"] = comparison_df[
        "rmse_delta_vs_persistence"
    ].apply(
        lambda value: "selected"
        if value < 0
        else "persistence"
    )
    comparison_df = comparison_df.sort_values(
        "rmse_delta_vs_persistence",
    )

    comparison_df.to_csv(
        LINEAR_RESULTS_DIR / "validation_selected_vs_persistence.csv",
        index=False,
    )

    selected_config = {
        "candidate_id": selected_id,
        "model": selected["model"],
        "alpha": (
            None
            if pd.isna(selected["alpha"])
            else float(selected["alpha"])
        ),
        "scaling": selected["scaling"],
        "selection_split": "validation",
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "primary_selection_value": float(selected[PRIMARY_SELECTION_METRIC]),
    }

    with (LINEAR_RESULTS_DIR / "selected_linear_config.json").open(
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

    for candidate in build_candidates():
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
    selected = select_best_linear(summary_df)
    comparison_df, selected_config = write_selection_outputs(
        metrics_df,
        summary_df,
        selected,
    )

    logger.info(
        "Validation tuning complete. "
        f"Selected {selected_config['candidate_id']} by "
        f"{PRIMARY_SELECTION_METRIC}="
        f"{selected_config['primary_selection_value']:.3f}"
    )
    logger.info(
        "Selected beats persistence on "
        f"{(comparison_df['winner'] == 'selected').sum()} datasets; "
        "persistence wins on "
        f"{(comparison_df['winner'] == 'persistence').sum()} datasets."
    )


if __name__ == "__main__":

    main()
