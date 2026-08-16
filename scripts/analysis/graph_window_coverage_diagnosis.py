"""
Diagnose masked graph-window coverage before GNN training.

This analysis is intentionally read-only with respect to graph dataset
artifacts. It inspects the outputs of graph stages 06 and 07 and explains
why 24-hour supervised target coverage is imbalanced across chronological
splits.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import GRAPH_SNAPSHOTS_DIR, RESULTS_DIR
from logger import logger


WINDOW_LENGTH = 24
EXPECTED_STEP = pd.Timedelta(hours=1)
THRESHOLDS = [5, 10, 20, 30, 40]

NODE_SNAPSHOT_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_nodes.csv.gz"
WINDOW_ARRAYS_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_arrays.npz"
WINDOW_INDEX_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_index.csv"
SUPERVISED_NODES_FILE = GRAPH_SNAPSHOTS_DIR / "supervised_nodes.csv"

OUTPUT_DIR = RESULTS_DIR / "graph_window_coverage"
NODE_REPORT_FILE = OUTPUT_DIR / "node_coverage_report.csv"
WINDOW_COVERAGE_FILE = OUTPUT_DIR / "window_coverage_by_time.csv"
THRESHOLD_FILE = OUTPUT_DIR / "coverage_thresholds.csv"
RUNS_FILE = OUTPUT_DIR / "coverage_threshold_runs.csv"
MONTHLY_FILE = OUTPUT_DIR / "node_coverage_by_month.csv"
FACTOR_FILE = OUTPUT_DIR / "coverage_factor_decomposition.csv"
SUMMARY_FILE = OUTPUT_DIR / "coverage_diagnosis_summary.csv"
RECOMMENDATION_FILE = OUTPUT_DIR / "methodology_recommendation.md"

SPLIT_CODE_TO_NAME = {
    0: "train",
    1: "validation",
    2: "test",
}


class GraphWindowCoverageDiagnosis:

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        nodes = self.load_nodes()
        arrays = np.load(WINDOW_ARRAYS_FILE)
        window_index = pd.read_csv(
            WINDOW_INDEX_FILE,
            parse_dates=[
                "start_timestamp",
                "end_timestamp",
                "target_timestamp",
            ],
        )
        snapshot_nodes = self.load_snapshot_nodes()

        candidate_windows = self.build_candidate_window_coverage(arrays)
        node_report = self.build_node_report(
            nodes,
            arrays,
            window_index,
            snapshot_nodes,
        )
        threshold_summary = self.build_threshold_summary(candidate_windows)
        threshold_runs = self.build_threshold_runs(candidate_windows)
        monthly = self.build_monthly_coverage(candidate_windows)
        factors = self.build_factor_decomposition(
            arrays,
            candidate_windows,
            snapshot_nodes,
        )
        summary = self.build_summary(
            window_index,
            candidate_windows,
            factors,
        )

        node_report.to_csv(NODE_REPORT_FILE, index=False)
        candidate_windows.to_csv(WINDOW_COVERAGE_FILE, index=False)
        threshold_summary.to_csv(THRESHOLD_FILE, index=False)
        threshold_runs.to_csv(RUNS_FILE, index=False)
        monthly.to_csv(MONTHLY_FILE, index=False)
        factors.to_csv(FACTOR_FILE, index=False)
        summary.to_csv(SUMMARY_FILE, index=False)
        self.write_recommendation(summary, factors, threshold_summary)

        logger.info("=" * 50)
        logger.info("Graph Window Coverage Diagnosis")
        logger.info("=" * 50)
        for row in summary.itertuples(index=False):
            logger.info(
                f"{row.split}: {row.usable_windows} usable windows, "
                f"{row.mean_targets_per_usable_window:.2f} targets/window"
            )
        logger.info(f"Outputs written to {OUTPUT_DIR}")

    @staticmethod
    def load_nodes():
        nodes = pd.read_csv(SUPERVISED_NODES_FILE)
        nodes = nodes.sort_values("node_id").reset_index(drop=True)
        nodes["node_position"] = np.arange(len(nodes), dtype=np.int16)
        return nodes

    @staticmethod
    def load_snapshot_nodes():
        columns = [
            "timestamp",
            "node_id",
            "node_exists",
            "input_valid",
            "target_exists_t_plus_1",
            "target_valid",
            "pm2_5",
            "temperature",
            "humidity",
            "pressure",
            "dew_point",
            "wind_u",
            "wind_v",
        ]
        snapshot_nodes = pd.read_csv(
            NODE_SNAPSHOT_FILE,
            usecols=columns,
            parse_dates=["timestamp"],
        )
        return snapshot_nodes

    @staticmethod
    def build_candidate_window_coverage(arrays):
        timestamps = pd.to_datetime(arrays["timestamps"])
        target_timestamps = pd.to_datetime(arrays["target_timestamps"])
        split_codes = arrays["split_codes"]
        split_names = np.array([
            SPLIT_CODE_TO_NAME[int(code)]
            for code in split_codes
        ])
        input_valid_mask = arrays["input_valid_mask"]
        target_valid_mask = arrays["target_valid_mask"]

        rows = []
        for end_idx in range(len(timestamps)):
            start_idx = end_idx - WINDOW_LENGTH + 1
            too_short = start_idx < 0
            hourly = False
            same_split_window = False
            target_same_split = bool(arrays["same_split_target"][end_idx])
            sequence_input_count = 0
            final_target_count = int(target_valid_mask[end_idx, :].sum())
            supervised_count = 0

            if not too_short:
                window_timestamps = timestamps[start_idx:end_idx + 1]
                hourly = bool(
                    pd.Series(window_timestamps).diff().dropna()
                    .eq(EXPECTED_STEP)
                    .all()
                )
                same_split_window = bool(
                    np.all(split_codes[start_idx:end_idx + 1] ==
                           split_codes[end_idx])
                )
                sequence_input_valid = input_valid_mask[
                    start_idx:end_idx + 1,
                    :,
                ].all(axis=0)
                supervised_target_valid = (
                    sequence_input_valid & target_valid_mask[end_idx, :]
                )
                sequence_input_count = int(sequence_input_valid.sum())
                supervised_count = int(supervised_target_valid.sum())

            split_crossing = (
                (not too_short) and
                hourly and
                (not same_split_window or not target_same_split)
            )
            zero_valid_targets = (
                (not too_short) and
                hourly and
                same_split_window and
                target_same_split and
                supervised_count == 0
            )
            usable = (
                (not too_short) and
                hourly and
                same_split_window and
                target_same_split and
                supervised_count > 0
            )

            rows.append({
                "end_idx": end_idx,
                "start_idx": start_idx if not too_short else None,
                "start_timestamp": (
                    timestamps[start_idx] if not too_short else pd.NaT
                ),
                "end_timestamp": timestamps[end_idx],
                "target_timestamp": target_timestamps[end_idx],
                "split": split_names[end_idx],
                "too_short_history": too_short,
                "hourly_24h": hourly,
                "same_split_window": same_split_window,
                "target_same_split": target_same_split,
                "split_crossing": split_crossing,
                "final_target_valid_node_count": final_target_count,
                "sequence_input_valid_node_count": sequence_input_count,
                "supervised_target_node_count": supervised_count,
                "zero_valid_targets": zero_valid_targets,
                "usable": usable,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def build_node_report(nodes, arrays, window_index, snapshot_nodes):
        timestamps = pd.to_datetime(arrays["timestamps"])
        input_valid = arrays["input_valid_mask"]
        target_valid = arrays["target_valid_mask"]
        window_target_valid = arrays["window_target_valid_mask"]

        rows = []
        for node in nodes.itertuples(index=False):
            pos = int(node.node_position)
            node_snapshots = snapshot_nodes[
                snapshot_nodes["node_id"] == node.node_id
            ].copy()

            valid_input_indices = np.flatnonzero(input_valid[:, pos])
            valid_target_indices = np.flatnonzero(target_valid[:, pos])
            window_valid_indices = np.flatnonzero(window_target_valid[:, pos])
            valid_windows = (
                window_index.iloc[window_valid_indices]
                if len(window_valid_indices)
                else pd.DataFrame(columns=window_index.columns)
            )

            pm_missing_when_exists = int(
                (
                    node_snapshots["node_exists"].astype(bool) &
                    node_snapshots["pm2_5"].isna()
                ).sum()
            )
            weather_columns = [
                "temperature",
                "humidity",
                "pressure",
                "dew_point",
                "wind_u",
                "wind_v",
            ]
            weather_missing_when_exists = int(
                (
                    node_snapshots["node_exists"].astype(bool) &
                    node_snapshots[weather_columns].isna().any(axis=1)
                ).sum()
            )

            rows.append({
                "node_id": int(node.node_id),
                "dataset_name": node.dataset_name,
                "station": node.station,
                "first_valid_input_timestamp": (
                    timestamps[valid_input_indices[0]]
                    if len(valid_input_indices)
                    else pd.NaT
                ),
                "last_valid_input_timestamp": (
                    timestamps[valid_input_indices[-1]]
                    if len(valid_input_indices)
                    else pd.NaT
                ),
                "first_valid_24h_sequence_target_timestamp": (
                    valid_windows["end_timestamp"].iloc[0]
                    if len(valid_windows)
                    else pd.NaT
                ),
                "last_valid_24h_sequence_target_timestamp": (
                    valid_windows["end_timestamp"].iloc[-1]
                    if len(valid_windows)
                    else pd.NaT
                ),
                "train_supervised_target_count": int(
                    (valid_windows["split"] == "train").sum()
                ),
                "validation_supervised_target_count": int(
                    (valid_windows["split"] == "validation").sum()
                ),
                "test_supervised_target_count": int(
                    (valid_windows["split"] == "test").sum()
                ),
                "has_train_targets": bool(
                    (valid_windows["split"] == "train").any()
                ),
                "has_validation_targets": bool(
                    (valid_windows["split"] == "validation").any()
                ),
                "valid_input_timestamps": int(len(valid_input_indices)),
                "valid_final_target_timestamps": int(len(valid_target_indices)),
                "valid_24h_sequence_target_windows": int(
                    len(window_valid_indices)
                ),
                "node_exists_timestamps": int(
                    node_snapshots["node_exists"].astype(bool).sum()
                ),
                "pm25_missing_when_node_exists": pm_missing_when_exists,
                "weather_missing_when_node_exists": weather_missing_when_exists,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def build_threshold_summary(candidate_windows):
        rows = []
        usable = candidate_windows[candidate_windows["usable"]].copy()
        for threshold in THRESHOLDS:
            reached = usable[
                usable["supervised_target_node_count"] >= threshold
            ]
            rows.append({
                "threshold_nodes": threshold,
                "first_reached_timestamp": (
                    reached["end_timestamp"].min()
                    if len(reached)
                    else pd.NaT
                ),
                "usable_windows_at_or_above_threshold": len(reached),
                "train_windows_at_or_above_threshold": int(
                    (reached["split"] == "train").sum()
                ),
                "validation_windows_at_or_above_threshold": int(
                    (reached["split"] == "validation").sum()
                ),
                "test_windows_at_or_above_threshold": int(
                    (reached["split"] == "test").sum()
                ),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def build_threshold_runs(candidate_windows):
        rows = []
        candidate_windows = candidate_windows.sort_values("end_idx")
        for threshold in THRESHOLDS:
            mask = (
                candidate_windows["usable"] &
                (candidate_windows["supervised_target_node_count"] >=
                 threshold)
            )
            run_start = None
            run_end = None
            run_split = None
            length = 0
            previous_idx = None
            for row in candidate_windows.assign(
                    threshold_usable=mask.values
            ).itertuples(index=False):
                if row.threshold_usable:
                    if previous_idx is None or row.end_idx != previous_idx + 1:
                        if run_start is not None:
                            rows.append({
                                "threshold_nodes": threshold,
                                "split": run_split,
                                "start_timestamp": run_start,
                                "end_timestamp": run_end,
                                "length_windows": length,
                            })
                        run_start = row.end_timestamp
                        run_split = row.split
                        length = 1
                    else:
                        length += 1
                    run_end = row.end_timestamp
                    previous_idx = row.end_idx
                else:
                    if run_start is not None:
                        rows.append({
                            "threshold_nodes": threshold,
                            "split": run_split,
                            "start_timestamp": run_start,
                            "end_timestamp": run_end,
                            "length_windows": length,
                        })
                    run_start = None
                    run_end = None
                    run_split = None
                    length = 0
                    previous_idx = None
            if run_start is not None:
                rows.append({
                    "threshold_nodes": threshold,
                    "split": run_split,
                    "start_timestamp": run_start,
                    "end_timestamp": run_end,
                    "length_windows": length,
                })

        runs = pd.DataFrame(rows)
        if runs.empty:
            return runs
        return runs.sort_values(
            ["threshold_nodes", "length_windows"],
            ascending=[True, False],
        )

    @staticmethod
    def build_monthly_coverage(candidate_windows):
        usable = candidate_windows[candidate_windows["usable"]].copy()
        usable["year_month"] = usable["end_timestamp"].dt.to_period(
            "M",
        ).astype(str)
        rows = []
        for (split, year_month), group in usable.groupby(
                ["split", "year_month"],
                sort=True,
        ):
            counts = group["supervised_target_node_count"]
            rows.append({
                "split": split,
                "year_month": year_month,
                "usable_windows": len(group),
                "supervised_node_targets": int(counts.sum()),
                "mean_targets_per_window": float(counts.mean()),
                "median_targets_per_window": float(counts.median()),
                "max_targets_per_window": int(counts.max()),
                "windows_ge_5_targets": int((counts >= 5).sum()),
                "windows_ge_10_targets": int((counts >= 10).sum()),
                "windows_ge_20_targets": int((counts >= 20).sum()),
                "windows_ge_30_targets": int((counts >= 30).sum()),
                "windows_ge_40_targets": int((counts >= 40).sum()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def build_factor_decomposition(arrays, candidate_windows, snapshot_nodes):
        input_valid = arrays["input_valid_mask"]
        target_valid = arrays["target_valid_mask"]
        split_codes = arrays["split_codes"]
        split_names = np.array([
            SPLIT_CODE_TO_NAME[int(code)]
            for code in split_codes
        ])

        valid_input_count = input_valid.sum(axis=1)
        valid_target_count = target_valid.sum(axis=1)
        factor_rows = []
        for split_name in ["train", "validation", "test", "all"]:
            if split_name == "all":
                ts_mask = np.ones(len(split_names), dtype=bool)
                win_mask = np.ones(len(candidate_windows), dtype=bool)
                node_subset = snapshot_nodes
            else:
                ts_mask = split_names == split_name
                win_mask = candidate_windows["split"].to_numpy() == split_name
                node_subset = snapshot_nodes[
                    snapshot_nodes["timestamp"].isin(
                        pd.to_datetime(arrays["timestamps"])[ts_mask]
                    )
                ]

            split_windows = candidate_windows[win_mask]
            valid_window_candidates = split_windows[
                (~split_windows["too_short_history"]) &
                split_windows["hourly_24h"] &
                split_windows["same_split_window"] &
                split_windows["target_same_split"]
            ]
            usable_windows = valid_window_candidates[
                valid_window_candidates["supervised_target_node_count"] > 0
            ]

            exists = node_subset["node_exists"].astype(bool)
            pm_missing_when_exists = int(
                (exists & node_subset["pm2_5"].isna()).sum()
            )
            weather_columns = [
                "temperature",
                "humidity",
                "pressure",
                "dew_point",
                "wind_u",
                "wind_v",
            ]
            weather_missing_when_exists = int(
                (exists & node_subset[weather_columns].isna().any(axis=1))
                .sum()
            )

            factor_rows.append({
                "split": split_name,
                "snapshot_timestamps": int(ts_mask.sum()),
                "node_timestamp_slots": int(ts_mask.sum() *
                                            input_valid.shape[1]),
                "node_exists_slots": int(exists.sum()),
                "valid_input_slots": int(valid_input_count[ts_mask].sum()),
                "valid_final_target_slots": int(
                    valid_target_count[ts_mask].sum()
                ),
                "pm25_missing_when_node_exists": pm_missing_when_exists,
                "weather_missing_when_node_exists": (
                    weather_missing_when_exists
                ),
                "candidate_window_endpoints": int(win_mask.sum()),
                "too_short_history_rejections": int(
                    split_windows["too_short_history"].sum()
                ),
                "split_crossing_rejections": int(
                    split_windows["split_crossing"].sum()
                ),
                "zero_target_rejections": int(
                    split_windows["zero_valid_targets"].sum()
                ),
                "usable_windows": len(usable_windows),
                "supervised_node_targets": int(
                    usable_windows["supervised_target_node_count"].sum()
                ),
                "mean_sequence_valid_nodes_before_target_mask": (
                    float(valid_window_candidates[
                        "sequence_input_valid_node_count"
                    ].mean())
                    if len(valid_window_candidates)
                    else 0.0
                ),
                "mean_final_target_valid_nodes_before_24h_mask": (
                    float(valid_window_candidates[
                        "final_target_valid_node_count"
                    ].mean())
                    if len(valid_window_candidates)
                    else 0.0
                ),
                "mean_supervised_targets_after_both_masks": (
                    float(valid_window_candidates[
                        "supervised_target_node_count"
                    ].mean())
                    if len(valid_window_candidates)
                    else 0.0
                ),
            })

        return pd.DataFrame(factor_rows)

    @staticmethod
    def build_summary(window_index, candidate_windows, factors):
        rows = []
        for split_name in ["train", "validation", "test", "all"]:
            if split_name == "all":
                split_windows = window_index
            else:
                split_windows = window_index[
                    window_index["split"] == split_name
                ]
            counts = split_windows["supervised_target_node_count"]
            factor = factors[factors["split"] == split_name].iloc[0]
            rows.append({
                "split": split_name,
                "usable_windows": len(split_windows),
                "supervised_node_targets": int(counts.sum()),
                "mean_targets_per_usable_window": (
                    float(counts.mean()) if len(counts) else 0.0
                ),
                "median_targets_per_usable_window": (
                    float(counts.median()) if len(counts) else 0.0
                ),
                "max_targets_per_usable_window": (
                    int(counts.max()) if len(counts) else 0
                ),
                "candidate_window_endpoints": int(
                    factor["candidate_window_endpoints"]
                ),
                "zero_target_rejections": int(
                    factor["zero_target_rejections"]
                ),
                "split_crossing_rejections": int(
                    factor["split_crossing_rejections"]
                ),
                "mean_sequence_valid_nodes_before_target_mask": float(
                    factor[
                        "mean_sequence_valid_nodes_before_target_mask"
                    ]
                ),
                "mean_final_target_valid_nodes_before_24h_mask": float(
                    factor[
                        "mean_final_target_valid_nodes_before_24h_mask"
                    ]
                ),
                "mean_supervised_targets_after_both_masks": float(
                    factor["mean_supervised_targets_after_both_masks"]
                ),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def write_recommendation(summary, factors, threshold_summary):
        train = summary[summary["split"] == "train"].iloc[0]
        validation = summary[summary["split"] == "validation"].iloc[0]
        train_factor = factors[factors["split"] == "train"].iloc[0]
        validation_factor = factors[
            factors["split"] == "validation"
        ].iloc[0]
        first_ge_10 = threshold_summary[
            threshold_summary["threshold_nodes"] == 10
        ]["first_reached_timestamp"].iloc[0]

        text = f"""# Graph Window Coverage Diagnosis

This diagnosis did not change the dataset, split, masks, graph, or
generated stage-06/stage-07 artifacts.

Train and validation coverage is too sparse for a conventional spatial
graph model under the current global 70/15/15 timeline:

- train mean supervised targets/window: {train.mean_targets_per_usable_window:.2f}
- validation mean supervised targets/window: {validation.mean_targets_per_usable_window:.2f}
- train max targets/window: {int(train.max_targets_per_usable_window)}
- validation max targets/window: {int(validation.max_targets_per_usable_window)}

The dominant cause is station deployment/start-date mismatch on the
global timeline, compounded by PM2.5 missingness and the strict 24-hour
per-node sequence-completeness rule. Split-boundary rejection is small:
train split-crossing rejections are {int(train_factor.split_crossing_rejections)}
and validation split-crossing rejections are
{int(validation_factor.split_crossing_rejections)}.

Coverage first reaches >=10 supervised target nodes at
{first_ge_10}, which is outside the usable train/validation evidence for
model selection. Do not use test coverage or performance to choose a new
cutoff, threshold, graph cohort, or hyperparameter.

Recommendation before GNN training: treat the current global 70/15/15
timeline as unsuitable for training a spatial graph model. Defensible
alternatives to evaluate using train+validation evidence only include:

1. A graph-specific chronological split over the period where enough
   supervised nodes are deployed and observable.
2. A smaller graph cohort chosen using train+validation availability
   only, with frozen node identities before any test evaluation.
3. A minimum-supervised-target threshold for graph windows selected
   using train+validation only.
4. A non-spatial temporal baseline or per-node model for sparse early
   periods, reserving graph modeling for synchronized deployment periods.

Do not implement any of these alternatives until the methodology choice
is reviewed.
"""
        RECOMMENDATION_FILE.write_text(text, encoding="utf-8")


def main():
    GraphWindowCoverageDiagnosis().run()


if __name__ == "__main__":
    main()
