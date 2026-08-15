import gc
import random

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from analysis.lstm_sequence_dataset_validation import EXPECTED_STEP
from analysis.lstm_sequence_dataset_validation import MIN_TRAIN_SEQUENCES
from analysis.lstm_sequence_dataset_validation import WINDOW_LENGTH
from analysis.lstm_sequence_dataset_validation import SEQUENCE_NATIVE_COLUMNS
from config import FEATURED_DIR
from config import LSTM_RESULTS_DIR
from config import ML_VALIDATION_DIR
from config import MODEL_FEATURE_COLUMNS
from config import RANDOM_FOREST_MAX_DEPTH
from config import RANDOM_FOREST_MAX_FEATURES
from config import RANDOM_FOREST_MIN_SAMPLES_LEAF
from config import RANDOM_FOREST_N_ESTIMATORS
from config import RANDOM_FOREST_N_JOBS
from config import RANDOM_FOREST_RANDOM_STATE
from config import TRAIN_DIR
from logger import logger
from models.base_model import BaseModel


HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
OUTPUT_DIR = LSTM_RESULTS_DIR


class StandardScaler:

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32) * self.scale_ + self.mean_


class PM25LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(SEQUENCE_NATIVE_COLUMNS),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        last_hidden = hidden[-1]
        return self.output(last_hidden).squeeze(1)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_random_forest():
    return RandomForestRegressor(
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        max_depth=RANDOM_FOREST_MAX_DEPTH,
        min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
        max_features=RANDOM_FOREST_MAX_FEATURES,
        random_state=RANDOM_FOREST_RANDOM_STATE,
        n_jobs=RANDOM_FOREST_N_JOBS,
    )


def build_sequences(station):
    featured_file = FEATURED_DIR / f"{station}.csv"
    featured_df = pd.read_csv(featured_file)
    featured_df["timestamp"] = pd.to_datetime(featured_df["timestamp"])

    split_ranges = build_train_validation_split_ranges(station)
    if not split_ranges:
        return None

    timestamps = featured_df["timestamp"].reset_index(drop=True)
    values = (
        featured_df[SEQUENCE_NATIVE_COLUMNS]
        .to_numpy(dtype=np.float32)
    )
    targets = featured_df["pm2_5"].to_numpy(dtype=np.float32)
    missing_input_by_row = (
        featured_df[SEQUENCE_NATIVE_COLUMNS]
        .isna()
        .any(axis=1)
        .astype(int)
        .reset_index(drop=True)
    )
    missing_input_cumsum = missing_input_by_row.cumsum()
    missing_target_by_row = (
        featured_df["pm2_5"]
        .isna()
        .reset_index(drop=True)
    )
    invalid_gap_by_row = timestamps.diff().ne(EXPECTED_STEP).astype(int)
    if len(invalid_gap_by_row) > 0:
        invalid_gap_by_row.iloc[0] = 0
    invalid_gap_cumsum = invalid_gap_by_row.cumsum()

    sequences = {
        TRAIN_SPLIT: empty_sequence_pack(),
        VALIDATION_SPLIT: empty_sequence_pack(),
    }

    for target_index, target_timestamp in enumerate(timestamps):
        split_name = classify_target_split(target_timestamp, split_ranges)
        if split_name not in sequences:
            continue

        input_start = target_index - WINDOW_LENGTH
        if input_start < 0:
            continue

        input_start_timestamp = timestamps.iloc[input_start]
        input_end_timestamp = timestamps.iloc[target_index - 1]
        split_range = split_ranges[split_name]
        if (
                input_start_timestamp < split_range["start_timestamp"] or
                target_timestamp > split_range["end_timestamp"]
        ):
            continue

        invalid_gap_count = (
            invalid_gap_cumsum.iloc[target_index] -
            invalid_gap_cumsum.iloc[input_start]
        )
        if invalid_gap_count != 0:
            continue

        missing_before_window = (
            missing_input_cumsum.iloc[input_start - 1]
            if input_start > 0
            else 0
        )
        input_missing = (
            missing_input_cumsum.iloc[target_index - 1] -
            missing_before_window
        ) > 0
        if input_missing or missing_target_by_row.iloc[target_index]:
            continue

        sequences[split_name]["X"].append(values[input_start:target_index])
        sequences[split_name]["y"].append(targets[target_index])
        sequences[split_name]["input_end_timestamp"].append(input_end_timestamp)
        sequences[split_name]["target_timestamp"].append(target_timestamp)
        sequences[split_name]["persistence_prediction"].append(
            targets[target_index - 1]
        )

    return finalize_sequence_pack(sequences)


def build_train_validation_split_ranges(station):
    split_files = {
        TRAIN_SPLIT: TRAIN_DIR / f"{station}.csv",
        VALIDATION_SPLIT: ML_VALIDATION_DIR / f"{station}.csv",
    }
    split_ranges = {}

    for split_name, split_file in split_files.items():
        if not split_file.exists():
            return {}

        split_df = pd.read_csv(split_file, usecols=["timestamp"])
        if split_df.empty:
            continue

        timestamps = pd.to_datetime(split_df["timestamp"])
        split_ranges[split_name] = {
            "start_timestamp": timestamps.iloc[0],
            "end_timestamp": timestamps.iloc[-1],
        }

    return split_ranges


def empty_sequence_pack():
    return {
        "X": [],
        "y": [],
        "input_end_timestamp": [],
        "target_timestamp": [],
        "persistence_prediction": [],
    }


def finalize_sequence_pack(sequences):
    for split_name, pack in sequences.items():
        if pack["X"]:
            pack["X"] = np.stack(pack["X"]).astype(np.float32)
            pack["y"] = np.asarray(pack["y"], dtype=np.float32)
            pack["input_end_timestamp"] = pd.to_datetime(
                pack["input_end_timestamp"]
            )
            pack["target_timestamp"] = pd.to_datetime(
                pack["target_timestamp"]
            )
            pack["persistence_prediction"] = np.asarray(
                pack["persistence_prediction"],
                dtype=np.float32,
            )
        else:
            pack["X"] = np.empty(
                (0, WINDOW_LENGTH, len(SEQUENCE_NATIVE_COLUMNS)),
                dtype=np.float32,
            )
            pack["y"] = np.empty((0,), dtype=np.float32)
            pack["input_end_timestamp"] = pd.to_datetime([])
            pack["target_timestamp"] = pd.to_datetime([])
            pack["persistence_prediction"] = np.empty((0,), dtype=np.float32)

    return sequences


def classify_target_split(target_timestamp, split_ranges):
    for split_name in [TRAIN_SPLIT, VALIDATION_SPLIT]:
        split_range = split_ranges.get(split_name)
        if split_range is None:
            continue
        if (
                split_range["start_timestamp"] <=
                target_timestamp <=
                split_range["end_timestamp"]
        ):
            return split_name

    return None


def scale_sequences(sequences):
    train = sequences[TRAIN_SPLIT]
    validation = sequences[VALIDATION_SPLIT]

    input_scaler = StandardScaler().fit(
        train["X"].reshape(-1, len(SEQUENCE_NATIVE_COLUMNS))
    )
    target_scaler = StandardScaler().fit(train["y"].reshape(-1, 1))

    scaled = {}
    for split_name, pack in [
            (TRAIN_SPLIT, train),
            (VALIDATION_SPLIT, validation),
    ]:
        X_shape = pack["X"].shape
        scaled_X = input_scaler.transform(
            pack["X"].reshape(-1, len(SEQUENCE_NATIVE_COLUMNS))
        ).reshape(X_shape)
        scaled_y = target_scaler.transform(
            pack["y"].reshape(-1, 1)
        ).reshape(-1)
        scaled[split_name] = {
            **pack,
            "X_scaled": scaled_X.astype(np.float32),
            "y_scaled": scaled_y.astype(np.float32),
        }

    return scaled, target_scaler


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train_station_model(sequences, target_scaler, device):
    model = PM25LSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loader = make_loader(
        sequences[TRAIN_SPLIT]["X_scaled"],
        sequences[TRAIN_SPLIT]["y_scaled"],
        shuffle=True,
    )
    validation_loader = make_loader(
        sequences[VALIDATION_SPLIT]["X_scaled"],
        sequences[VALIDATION_SPLIT]["y_scaled"],
        shuffle=False,
    )

    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )
        train_loss = float(np.mean(train_losses))
        history.append({
            "epoch": epoch,
            "train_loss_scaled_mse": train_loss,
            "validation_loss_scaled_mse": validation_loss,
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    scaled_prediction = predict_scaled(model, validation_loader, device)
    prediction = target_scaler.inverse_transform(
        scaled_prediction.reshape(-1, 1)
    ).reshape(-1)

    return model, prediction, best_epoch, best_validation_loss, history


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(X_batch)
            losses.append(criterion(prediction, y_batch).item())

    return float(np.mean(losses))


def predict_scaled(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            prediction = model(X_batch)
            predictions.append(prediction.detach().cpu().numpy())

    return np.concatenate(predictions).astype(np.float32)


def evaluate_arrays(target, prediction):
    evaluator = BaseModel(OUTPUT_DIR)
    return evaluator.evaluate(target, prediction)


def build_rf_predictions(station, validation_target_timestamps):
    train_file = TRAIN_DIR / f"{station}.csv"
    validation_file = ML_VALIDATION_DIR / f"{station}.csv"
    if not train_file.exists() or not validation_file.exists():
        return pd.DataFrame()

    required_columns = MODEL_FEATURE_COLUMNS + ["target_pm2_5", "timestamp"]
    train_df = pd.read_csv(train_file, usecols=required_columns)
    validation_df = pd.read_csv(validation_file, usecols=required_columns)
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    validation_df["timestamp"] = pd.to_datetime(validation_df["timestamp"])

    evaluator = BaseModel(OUTPUT_DIR)
    train_eval = evaluator.prepare_evaluation_frame(train_df)
    validation_eval = evaluator.prepare_evaluation_frame(validation_df)
    if train_eval.empty or validation_eval.empty:
        return pd.DataFrame()

    validation_eval = validation_eval.copy()
    validation_eval["target_timestamp"] = (
        pd.to_datetime(validation_eval["timestamp"]) + EXPECTED_STEP
    )
    matched = validation_eval[
        validation_eval["target_timestamp"].isin(validation_target_timestamps)
    ].copy()
    if matched.empty:
        return pd.DataFrame()

    X_train, y_train = evaluator.split_features_target(train_eval)
    estimator = make_random_forest()
    estimator.fit(X_train, y_train)
    matched["rf_prediction"] = estimator.predict(matched[MODEL_FEATURE_COLUMNS])
    matched["rf_target"] = matched["target_pm2_5"]

    return matched[[
        "target_timestamp",
        "rf_target",
        "rf_prediction",
    ]]


def summarize_station_metrics(metrics_df, predictions_df):
    pooled_mae, pooled_rmse, pooled_r2 = evaluate_arrays(
        predictions_df["target_pm2_5"],
        predictions_df["lstm_prediction"],
    )

    return pd.DataFrame([{
        "model": "LSTM",
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


def summarize_matched(matched_predictions_df):
    rows = []
    for model, prediction_column in [
            ("LSTM", "lstm_prediction"),
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
        _, lstm_rmse, _ = evaluate_arrays(
            station_df["target_pm2_5"],
            station_df["lstm_prediction"],
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
            "lstm_rmse": lstm_rmse,
            "persistence_rmse": persistence_rmse,
            "rf_rmse": rf_rmse,
            "lstm_beats_persistence": lstm_rmse < persistence_rmse,
            "lstm_beats_rf": lstm_rmse < rf_rmse,
            "lstm_minus_persistence_rmse": lstm_rmse - persistence_rmse,
            "lstm_minus_rf_rmse": lstm_rmse - rf_rmse,
        })

    station_df = pd.DataFrame(station_rows)
    summary_df = pd.DataFrame([{
        "matched_stations": len(station_df),
        "lstm_wins_vs_persistence": int(
            station_df["lstm_beats_persistence"].sum()
        ),
        "lstm_losses_vs_persistence": int(
            (~station_df["lstm_beats_persistence"]).sum()
        ),
        "lstm_wins_vs_rf": int(station_df["lstm_beats_rf"].sum()),
        "lstm_losses_vs_rf": int((~station_df["lstm_beats_rf"]).sum()),
        "median_lstm_minus_persistence_rmse": (
            station_df["lstm_minus_persistence_rmse"].median()
        ),
        "median_lstm_minus_rf_rmse": station_df["lstm_minus_rf_rmse"].median(),
    }])

    return station_df, summary_df


def write_report(
        torch_info,
        trained_stations,
        skipped_stations,
        station_metrics_df,
        summary_df,
        matched_summary_df,
        win_summary_df,
):
    best_epochs = station_metrics_df["best_epoch"]
    lines = [
        "# LSTM Baseline Validation",
        "",
        "This run trained the first station-specific LSTM baseline on train "
        "and validation only. The final test split was not evaluated.",
        "",
        "## Runtime",
        "",
        "```text",
        pd.Series(torch_info).to_string(),
        "```",
        "",
        "## Cohort",
        "",
        f"Stations trained: {len(trained_stations)}",
        f"Stations skipped: {len(skipped_stations)}",
        "",
        "Skipped stations:",
        "",
        "```text",
        pd.DataFrame(skipped_stations).to_string(index=False)
        if skipped_stations else "None",
        "```",
        "",
        "## Native LSTM Validation",
        "",
        "```text",
        summary_df.to_string(index=False),
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
        best_epochs.describe().to_string(),
        "```",
        "",
        "## Interpretation",
        "",
        "The first LSTM baseline is considered useful only if it improves on "
        "Persistence and the frozen Random Forest on matched validation "
        "timestamps. Overfitting is indicated by early validation stopping "
        "with train loss continuing lower than validation loss.",
        "",
    ]
    (OUTPUT_DIR / "lstm_validation_report.md").write_text(
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

    station_metrics = []
    predictions = []
    matched_predictions = []
    training_history = []
    skipped_stations = []
    trained_stations = []

    featured_files = sorted(FEATURED_DIR.glob("*.csv"))
    logger.info(f"Training LSTM baseline for {len(featured_files)} stations")
    logger.info(f"PyTorch {torch.__version__}; device={device}")

    for featured_file in featured_files:
        station = featured_file.stem
        sequences = build_sequences(station)
        if sequences is None:
            skipped_stations.append({
                "station": station,
                "reason": "missing prepared split boundaries",
            })
            continue

        train_count = len(sequences[TRAIN_SPLIT]["y"])
        validation_count = len(sequences[VALIDATION_SPLIT]["y"])
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

        scaled, target_scaler = scale_sequences(sequences)
        _, lstm_prediction, best_epoch, best_loss, history = train_station_model(
            scaled,
            target_scaler,
            device,
        )
        validation_pack = sequences[VALIDATION_SPLIT]
        target = validation_pack["y"]
        mae, rmse, r2 = evaluate_arrays(target, lstm_prediction)
        station_metrics.append({
            "station": station,
            "train_sequences": train_count,
            "validation_sequences": validation_count,
            "best_epoch": best_epoch,
            "best_validation_loss_scaled_mse": best_loss,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })
        trained_stations.append(station)

        station_prediction_df = pd.DataFrame({
            "station": station,
            "input_end_timestamp": validation_pack["input_end_timestamp"],
            "target_timestamp": validation_pack["target_timestamp"],
            "target_pm2_5": target,
            "lstm_prediction": lstm_prediction,
            "persistence_prediction": validation_pack[
                "persistence_prediction"
            ],
        })
        predictions.append(station_prediction_df)

        rf_predictions = build_rf_predictions(
            station,
            station_prediction_df["target_timestamp"],
        )
        if not rf_predictions.empty:
            matched = station_prediction_df.merge(
                rf_predictions,
                on="target_timestamp",
                how="inner",
            )
            matched = matched[
                np.isclose(matched["target_pm2_5"], matched["rf_target"])
            ].copy()
            matched.drop(columns=["rf_target"], inplace=True)
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

    summary_df = summarize_station_metrics(station_metrics_df, predictions_df)
    matched_summary_df = summarize_matched(matched_predictions_df)
    station_win_df, win_summary_df = build_win_counts(matched_predictions_df)

    station_metrics_df.to_csv(
        OUTPUT_DIR / "validation_station_metrics.csv",
        index=False,
    )
    summary_df.to_csv(
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

    write_report(
        torch_info,
        trained_stations,
        skipped_stations,
        station_metrics_df,
        summary_df,
        matched_summary_df,
        win_summary_df,
    )

    logger.info(
        "LSTM baseline complete. "
        f"Trained {len(trained_stations)} stations; "
        f"skipped {len(skipped_stations)} stations."
    )


if __name__ == "__main__":
    main()
