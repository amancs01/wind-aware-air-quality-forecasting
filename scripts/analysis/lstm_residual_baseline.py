import gc

import numpy as np
import pandas as pd
import torch

from analysis.lstm_baseline import OUTPUT_DIR as DIRECT_LSTM_OUTPUT_DIR
from analysis.lstm_baseline import TRAIN_SPLIT
from analysis.lstm_baseline import VALIDATION_SPLIT
from analysis.lstm_baseline import build_rf_predictions
from analysis.lstm_baseline import build_sequences
from analysis.lstm_baseline import evaluate_arrays
from analysis.lstm_baseline import scale_sequences
from analysis.lstm_baseline import set_random_seed
from analysis.lstm_baseline import train_station_model
from analysis.lstm_baseline import RANDOM_SEED
from analysis.lstm_sequence_dataset_validation import MIN_TRAIN_SEQUENCES
from config import FEATURED_DIR
from config import LSTM_RESULTS_DIR
from logger import logger


OUTPUT_DIR = LSTM_RESULTS_DIR / "residual"


def prepare_residual_sequences(sequences):
    residual_sequences = {}

    for split_name, pack in sequences.items():
        persistence = pack["persistence_prediction"].astype(np.float32)
        target = pack["y"].astype(np.float32)
        residual = (target - persistence).astype(np.float32)
        residual_sequences[split_name] = {
            **pack,
            "absolute_target": target,
            "residual_target": residual,
            "y": residual,
        }

    return residual_sequences


def summarize_target_distribution(station_distribution_df):
    rows = []
    for split_name, split_df in station_distribution_df.groupby("split"):
        rows.append({
            "split": split_name,
            "stations": split_df["station"].nunique(),
            "sequences": int(split_df["sequences"].sum()),
            "pooled_absolute_target_mean": (
                np.average(
                    split_df["absolute_target_mean"],
                    weights=split_df["sequences"],
                )
            ),
            "pooled_absolute_target_std": (
                np.sqrt(
                    np.average(
                        (
                            split_df["absolute_target_std"].fillna(0) ** 2 +
                            split_df["absolute_target_mean"] ** 2
                        ),
                        weights=split_df["sequences"],
                    ) -
                    np.average(
                        split_df["absolute_target_mean"],
                        weights=split_df["sequences"],
                    ) ** 2
                )
            ),
            "pooled_residual_target_mean": (
                np.average(
                    split_df["residual_target_mean"],
                    weights=split_df["sequences"],
                )
            ),
            "pooled_residual_target_std": (
                np.sqrt(
                    np.average(
                        (
                            split_df["residual_target_std"].fillna(0) ** 2 +
                            split_df["residual_target_mean"] ** 2
                        ),
                        weights=split_df["sequences"],
                    ) -
                    np.average(
                        split_df["residual_target_mean"],
                        weights=split_df["sequences"],
                    ) ** 2
                )
            ),
        })

    return pd.DataFrame(rows)


def add_distribution_rows(station, sequences, rows):
    for split_name in [TRAIN_SPLIT, VALIDATION_SPLIT]:
        pack = sequences[split_name]
        absolute = pack["absolute_target"]
        residual = pack["residual_target"]
        rows.append({
            "station": station,
            "split": split_name,
            "sequences": len(absolute),
            "absolute_target_mean": float(np.mean(absolute)),
            "absolute_target_std": float(np.std(absolute)),
            "residual_target_mean": float(np.mean(residual)),
            "residual_target_std": float(np.std(residual)),
        })


def summarize_station_metrics(metrics_df, predictions_df):
    pooled_mae, pooled_rmse, pooled_r2 = evaluate_arrays(
        predictions_df["target_pm2_5"],
        predictions_df["residual_lstm_prediction"],
    )

    return pd.DataFrame([{
        "model": "ResidualLSTM",
        "stations": len(metrics_df),
        "validation_sequences": int(metrics_df["validation_sequences"].sum()),
        "macro_mae": metrics_df["mae"].mean(),
        "macro_rmse": metrics_df["rmse"].mean(),
        "macro_mean_r2": metrics_df["r2"].mean(),
        "macro_median_r2": metrics_df["r2"].median(),
        "pooled_mae": pooled_mae,
        "pooled_rmse": pooled_rmse,
        "pooled_r2": pooled_r2,
    }])


def load_direct_lstm_predictions():
    prediction_file = DIRECT_LSTM_OUTPUT_DIR / "validation_predictions.csv"
    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Direct LSTM predictions not found: {prediction_file}"
        )

    direct_df = pd.read_csv(prediction_file)
    direct_df["target_timestamp"] = pd.to_datetime(
        direct_df["target_timestamp"]
    )
    return direct_df[[
        "station",
        "target_timestamp",
        "target_pm2_5",
        "lstm_prediction",
    ]].rename(columns={
        "target_pm2_5": "direct_lstm_target",
        "lstm_prediction": "direct_lstm_prediction",
    })


def summarize_matched(matched_predictions_df):
    rows = []
    for model, prediction_column in [
            ("DirectLSTM", "direct_lstm_prediction"),
            ("ResidualLSTM", "residual_lstm_prediction"),
            ("Persistence", "persistence_prediction"),
            ("RandomForest", "rf_prediction"),
    ]:
        station_rows = []
        pooled_targets = []
        pooled_predictions = []
        for station, station_df in matched_predictions_df.groupby("station"):
            mae, rmse, r2 = evaluate_arrays(
                station_df["target_pm2_5"],
                station_df[prediction_column],
            )
            station_rows.append({
                "model": model,
                "station": station,
                "matched_validation_rows": len(station_df),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            })
            pooled_targets.extend(station_df["target_pm2_5"].to_list())
            pooled_predictions.extend(station_df[prediction_column].to_list())

        station_metrics = pd.DataFrame(station_rows)
        pooled_mae, pooled_rmse, pooled_r2 = evaluate_arrays(
            pooled_targets,
            pooled_predictions,
        )
        rows.append({
            "model": model,
            "stations": len(station_metrics),
            "matched_validation_rows": int(
                station_metrics["matched_validation_rows"].sum()
            ),
            "macro_mae": station_metrics["mae"].mean(),
            "macro_rmse": station_metrics["rmse"].mean(),
            "macro_mean_r2": station_metrics["r2"].mean(),
            "macro_median_r2": station_metrics["r2"].median(),
            "pooled_mae": pooled_mae,
            "pooled_rmse": pooled_rmse,
            "pooled_r2": pooled_r2,
        })

    return pd.DataFrame(rows)


def build_win_counts(matched_predictions_df):
    station_rows = []
    for station, station_df in matched_predictions_df.groupby("station"):
        _, residual_rmse, _ = evaluate_arrays(
            station_df["target_pm2_5"],
            station_df["residual_lstm_prediction"],
        )
        _, direct_rmse, _ = evaluate_arrays(
            station_df["target_pm2_5"],
            station_df["direct_lstm_prediction"],
        )
        _, persistence_rmse, _ = evaluate_arrays(
            station_df["target_pm2_5"],
            station_df["persistence_prediction"],
        )
        _, rf_rmse, _ = evaluate_arrays(
            station_df["target_pm2_5"],
            station_df["rf_prediction"],
        )

        station_rows.append({
            "station": station,
            "residual_lstm_rmse": residual_rmse,
            "direct_lstm_rmse": direct_rmse,
            "persistence_rmse": persistence_rmse,
            "rf_rmse": rf_rmse,
            "residual_beats_persistence": residual_rmse < persistence_rmse,
            "residual_beats_rf": residual_rmse < rf_rmse,
            "residual_beats_direct_lstm": residual_rmse < direct_rmse,
            "residual_minus_persistence_rmse": (
                residual_rmse - persistence_rmse
            ),
            "residual_minus_rf_rmse": residual_rmse - rf_rmse,
            "residual_minus_direct_lstm_rmse": (
                residual_rmse - direct_rmse
            ),
        })

    station_df = pd.DataFrame(station_rows)
    summary_df = pd.DataFrame([{
        "matched_stations": len(station_df),
        "residual_wins_vs_persistence": int(
            station_df["residual_beats_persistence"].sum()
        ),
        "residual_losses_vs_persistence": int(
            (~station_df["residual_beats_persistence"]).sum()
        ),
        "residual_wins_vs_rf": int(station_df["residual_beats_rf"].sum()),
        "residual_losses_vs_rf": int(
            (~station_df["residual_beats_rf"]).sum()
        ),
        "residual_wins_vs_direct_lstm": int(
            station_df["residual_beats_direct_lstm"].sum()
        ),
        "residual_losses_vs_direct_lstm": int(
            (~station_df["residual_beats_direct_lstm"]).sum()
        ),
        "median_residual_minus_persistence_rmse": (
            station_df["residual_minus_persistence_rmse"].median()
        ),
        "median_residual_minus_rf_rmse": (
            station_df["residual_minus_rf_rmse"].median()
        ),
        "median_residual_minus_direct_lstm_rmse": (
            station_df["residual_minus_direct_lstm_rmse"].median()
        ),
    }])

    return station_df, summary_df


def write_report(
        torch_info,
        skipped_stations,
        station_metrics_df,
        native_summary_df,
        matched_summary_df,
        win_summary_df,
        target_distribution_summary_df,
):
    lines = [
        "# Residual LSTM Validation",
        "",
        "This experiment keeps the original station-specific LSTM setup but "
        "changes the supervised target to "
        "`PM2.5(t+1) - PM2.5(t)`. The final test split was not evaluated.",
        "",
        "## Runtime",
        "",
        "```text",
        pd.Series(torch_info).to_string(),
        "```",
        "",
        "## Cohort",
        "",
        f"Stations trained: {len(station_metrics_df)}",
        f"Stations skipped: {len(skipped_stations)}",
        "",
        "Skipped stations:",
        "",
        "```text",
        pd.DataFrame(skipped_stations).to_string(index=False)
        if skipped_stations else "None",
        "```",
        "",
        "## Target Distribution",
        "",
        "```text",
        target_distribution_summary_df.to_string(index=False),
        "```",
        "",
        "## Native Residual LSTM Validation",
        "",
        "```text",
        native_summary_df.to_string(index=False),
        "```",
        "",
        "## Matched Comparison",
        "",
        "```text",
        matched_summary_df.to_string(index=False),
        "```",
        "",
        "## Station Win Counts",
        "",
        "```text",
        win_summary_df.to_string(index=False),
        "```",
        "",
        "## Best Epochs",
        "",
        "```text",
        station_metrics_df["best_epoch"].describe().to_string(),
        "```",
        "",
        "## Interpretation",
        "",
        "Residual learning is useful only if it materially improves on the "
        "direct LSTM and beats Persistence/RF on matched validation "
        "timestamps. If it does not clear those baselines, the next "
        "research step should move toward graph design rather than further "
        "station-specific LSTM tuning.",
        "",
    ]
    (OUTPUT_DIR / "residual_lstm_validation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }

    direct_predictions_df = load_direct_lstm_predictions()
    station_metrics = []
    predictions = []
    matched_predictions = []
    training_history = []
    skipped_stations = []
    distribution_rows = []

    featured_files = sorted(FEATURED_DIR.glob("*.csv"))
    logger.info(
        f"Training residual LSTM baseline for {len(featured_files)} stations"
    )
    logger.info(f"PyTorch {torch.__version__}; device={device}")

    for featured_file in featured_files:
        station = featured_file.stem
        sequences = build_sequences(station)
        if sequences is None:
            skipped_stations.append({
                "station": station,
                "reason": "missing train/validation split boundaries",
            })
            continue

        residual_sequences = prepare_residual_sequences(sequences)
        train_count = len(residual_sequences[TRAIN_SPLIT]["y"])
        validation_count = len(residual_sequences[VALIDATION_SPLIT]["y"])

        if train_count < MIN_TRAIN_SEQUENCES:
            skipped_stations.append({
                "station": station,
                "reason": "fewer than 100 training sequences",
                "train_sequences": train_count,
                "validation_sequences": validation_count,
            })
            continue
        if validation_count == 0:
            skipped_stations.append({
                "station": station,
                "reason": "no validation sequences",
                "train_sequences": train_count,
                "validation_sequences": validation_count,
            })
            continue

        add_distribution_rows(station, residual_sequences, distribution_rows)
        scaled, residual_scaler = scale_sequences(residual_sequences)
        _, predicted_delta, best_epoch, best_loss, history = train_station_model(
            scaled,
            residual_scaler,
            device,
        )

        validation_pack = residual_sequences[VALIDATION_SPLIT]
        target = validation_pack["absolute_target"]
        persistence = validation_pack["persistence_prediction"]
        residual_prediction = persistence + predicted_delta
        mae, rmse, r2 = evaluate_arrays(target, residual_prediction)

        station_metrics.append({
            "station": station,
            "train_sequences": train_count,
            "validation_sequences": validation_count,
            "best_epoch": best_epoch,
            "best_validation_loss_scaled_mse": best_loss,
            "train_residual_mean": float(
                np.mean(residual_sequences[TRAIN_SPLIT]["residual_target"])
            ),
            "train_residual_std": float(
                np.std(residual_sequences[TRAIN_SPLIT]["residual_target"])
            ),
            "train_absolute_target_mean": float(
                np.mean(residual_sequences[TRAIN_SPLIT]["absolute_target"])
            ),
            "train_absolute_target_std": float(
                np.std(residual_sequences[TRAIN_SPLIT]["absolute_target"])
            ),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })

        station_prediction_df = pd.DataFrame({
            "station": station,
            "input_end_timestamp": validation_pack["input_end_timestamp"],
            "target_timestamp": validation_pack["target_timestamp"],
            "target_pm2_5": target,
            "current_pm2_5": persistence,
            "actual_delta_pm25": validation_pack["residual_target"],
            "predicted_delta_pm25": predicted_delta,
            "residual_lstm_prediction": residual_prediction,
            "persistence_prediction": persistence,
        })
        predictions.append(station_prediction_df)

        rf_predictions = build_rf_predictions(
            station,
            station_prediction_df["target_timestamp"],
        )
        direct_station = direct_predictions_df[
            direct_predictions_df["station"] == station
        ]
        matched = station_prediction_df.merge(
            direct_station,
            on=["station", "target_timestamp"],
            how="inner",
        )
        matched = matched.merge(
            rf_predictions,
            on="target_timestamp",
            how="inner",
        )
        matched = matched[
            np.isclose(matched["target_pm2_5"], matched["direct_lstm_target"])
        ].copy()
        matched = matched[
            np.isclose(matched["target_pm2_5"], matched["rf_target"])
        ].copy()
        matched.drop(
            columns=["direct_lstm_target", "rf_target"],
            inplace=True,
        )
        if not matched.empty:
            matched_predictions.append(matched)

        for row in history:
            training_history.append({
                "station": station,
                **row,
            })

        logger.info(
            f"{station}: train={train_count}, validation={validation_count}, "
            f"best_epoch={best_epoch}, RMSE={rmse:.3f}"
        )
        gc.collect()

    station_metrics_df = pd.DataFrame(station_metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)
    matched_predictions_df = pd.concat(matched_predictions, ignore_index=True)
    training_history_df = pd.DataFrame(training_history)
    skipped_df = pd.DataFrame(skipped_stations)
    distribution_df = pd.DataFrame(distribution_rows)

    native_summary_df = summarize_station_metrics(
        station_metrics_df,
        predictions_df,
    )
    matched_summary_df = summarize_matched(matched_predictions_df)
    station_win_df, win_summary_df = build_win_counts(matched_predictions_df)
    target_distribution_summary_df = summarize_target_distribution(
        distribution_df
    )

    station_metrics_df.to_csv(
        OUTPUT_DIR / "validation_station_metrics.csv",
        index=False,
    )
    native_summary_df.to_csv(
        OUTPUT_DIR / "validation_summary.csv",
        index=False,
    )
    predictions_df.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )
    matched_predictions_df.to_csv(
        OUTPUT_DIR / "validation_matched_predictions.csv",
        index=False,
    )
    matched_summary_df.to_csv(
        OUTPUT_DIR / "validation_matched_summary.csv",
        index=False,
    )
    station_win_df.to_csv(
        OUTPUT_DIR / "validation_station_win_counts.csv",
        index=False,
    )
    win_summary_df.to_csv(
        OUTPUT_DIR / "validation_win_summary.csv",
        index=False,
    )
    training_history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )
    skipped_df.to_csv(
        OUTPUT_DIR / "skipped_stations.csv",
        index=False,
    )
    distribution_df.to_csv(
        OUTPUT_DIR / "target_distribution_by_station.csv",
        index=False,
    )
    target_distribution_summary_df.to_csv(
        OUTPUT_DIR / "target_distribution_summary.csv",
        index=False,
    )

    write_report(
        torch_info,
        skipped_stations,
        station_metrics_df,
        native_summary_df,
        matched_summary_df,
        win_summary_df,
        target_distribution_summary_df,
    )

    logger.info(
        "Residual LSTM baseline complete. "
        f"Trained {len(station_metrics_df)} stations; "
        f"skipped {len(skipped_stations)} stations."
    )


if __name__ == "__main__":
    main()
