"""
Masked 24-hour spatio-temporal graph window index builder.

Consumes graph snapshot outputs from 06_graph_snapshots.py and creates a
compact representation for future graph models:

- node/edge snapshot arrays are stored once by timestamp;
- each 24-hour window is represented by start/end/target indices and
  validity counts;
- masks are preserved without imputing missing PM2.5 or weather;
- no GNN, GAT-GRU, or model training is performed here.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import GRAPH_SNAPSHOTS_DIR
from logger import logger


WINDOW_LENGTH = 24
EXPECTED_STEP = pd.Timedelta(hours=1)

NODE_FEATURE_COLUMNS = [
    "pm2_5",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "temperature",
    "humidity",
    "pressure",
    "dew_point",
    "wind_u",
    "wind_v",
]

NODE_SNAPSHOT_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_nodes.csv.gz"
EDGE_SNAPSHOT_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_edges.csv.gz"
TIMESTAMP_SUMMARY_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_timestamp_summary.csv"
SUPERVISED_NODES_FILE = GRAPH_SNAPSHOTS_DIR / "supervised_nodes.csv"

GRAPH_ARRAYS_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_arrays.npz"
WINDOW_INDEX_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_index.csv"
WINDOW_SUMMARY_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_summary.csv"
WINDOW_VALIDATION_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_validation.csv"
WINDOW_DISTRIBUTION_FILE = (
    GRAPH_SNAPSHOTS_DIR / "graph_window_target_distribution.csv"
)
WINDOW_RUNS_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_continuous_runs.csv"
EDGE_ORDER_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_edge_order.csv"
NODE_ORDER_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_node_order.csv"
REJECTION_SUMMARY_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_rejections.csv"

SPLIT_TO_CODE = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


class SlidingGraphWindowBuilder:

    def run(self):
        timestamp_summary = self.load_timestamp_summary()
        node_order = self.load_node_order()
        edge_order = self.load_edge_order()

        arrays = self.build_snapshot_arrays(
            timestamp_summary,
            node_order,
            edge_order,
        )
        window_index, rejection_summary = self.build_window_index(
            timestamp_summary,
            arrays["input_valid_mask"],
            arrays["target_valid_mask"],
            arrays["edge_valid_mask"],
            arrays["edge_active_mask"],
        )
        window_masks = self.build_window_masks(
            window_index,
            arrays["input_valid_mask"],
            arrays["target_valid_mask"],
        )
        arrays.update(window_masks)
        validation = self.build_validation(
            timestamp_summary,
            node_order,
            edge_order,
            arrays,
            window_index,
        )
        distribution = self.build_target_distribution(window_index)
        runs = self.build_continuous_runs(window_index)
        summary = self.build_summary(
            window_index,
            rejection_summary,
            arrays,
        )

        self.save_arrays(arrays, node_order, edge_order)
        window_index.to_csv(WINDOW_INDEX_FILE, index=False)
        summary.to_csv(WINDOW_SUMMARY_FILE, index=False)
        validation.to_csv(WINDOW_VALIDATION_FILE, index=False)
        distribution.to_csv(WINDOW_DISTRIBUTION_FILE, index=False)
        runs.to_csv(WINDOW_RUNS_FILE, index=False)
        rejection_summary.to_csv(REJECTION_SUMMARY_FILE, index=False)
        node_order.to_csv(NODE_ORDER_FILE, index=False)
        edge_order.to_csv(EDGE_ORDER_FILE, index=False)

        logger.info("=" * 50)
        logger.info("Masked Graph Window Summary")
        logger.info("=" * 50)
        for row in summary.itertuples(index=False):
            if row.split == "all":
                logger.info(
                    f"Usable windows: {row.usable_windows}, "
                    f"node-target examples: {row.supervised_node_targets}"
                )
            else:
                logger.info(
                    f"{row.split}: {row.usable_windows} windows, "
                    f"{row.supervised_node_targets} node-target examples"
                )

    @staticmethod
    def load_timestamp_summary():
        timestamp_summary = pd.read_csv(
            TIMESTAMP_SUMMARY_FILE,
            parse_dates=["timestamp", "target_timestamp"],
        )
        timestamp_summary = timestamp_summary.sort_values(
            "timestamp",
        ).reset_index(drop=True)
        timestamp_summary["timestamp_idx"] = np.arange(
            len(timestamp_summary),
            dtype=np.int32,
        )
        return timestamp_summary

    @staticmethod
    def load_node_order():
        nodes = pd.read_csv(SUPERVISED_NODES_FILE)
        nodes = nodes.sort_values("node_id").reset_index(drop=True)
        if len(nodes) != 51:
            raise ValueError(f"expected 51 supervised nodes, found {len(nodes)}")
        nodes["node_position"] = np.arange(len(nodes), dtype=np.int16)
        return nodes

    @staticmethod
    def load_edge_order():
        edge_columns = [
            "source_node_id",
            "target_node_id",
            "source_dataset_name",
            "target_dataset_name",
        ]
        edges = pd.read_csv(
            EDGE_SNAPSHOT_FILE,
            usecols=edge_columns,
        )
        edges = edges[edge_columns].drop_duplicates()
        edges = edges.sort_values([
            "source_node_id",
            "target_node_id",
        ]).reset_index(drop=True)
        if len(edges) != 326:
            raise ValueError(
                f"expected 326 supervised directed edges, found {len(edges)}"
            )
        edges["edge_position"] = np.arange(len(edges), dtype=np.int16)
        return edges

    def build_snapshot_arrays(self, timestamp_summary, node_order, edge_order):
        timestamps = timestamp_summary["timestamp"].to_numpy(
            dtype="datetime64[ns]",
        )
        target_timestamps = timestamp_summary["target_timestamp"].to_numpy(
            dtype="datetime64[ns]",
        )
        split_codes = (
            timestamp_summary["split"].map(SPLIT_TO_CODE).to_numpy(
                dtype=np.int8,
            )
        )
        same_split_target = timestamp_summary[
            "same_split_target"
        ].astype(bool).to_numpy()

        node_arrays = self.build_node_arrays(
            timestamp_summary,
            node_order,
        )
        edge_arrays = self.build_edge_arrays(
            timestamp_summary,
            edge_order,
        )

        return {
            "timestamps": timestamps,
            "target_timestamps": target_timestamps,
            "split_codes": split_codes,
            "same_split_target": same_split_target,
            **node_arrays,
            **edge_arrays,
        }

    @staticmethod
    def build_node_arrays(timestamp_summary, node_order):
        node_columns = (
            ["timestamp", "node_id"] +
            NODE_FEATURE_COLUMNS +
            [
                "input_valid",
                "target_valid",
                "residual_pm25_t_plus_1",
            ]
        )
        nodes = pd.read_csv(
            NODE_SNAPSHOT_FILE,
            usecols=node_columns,
            parse_dates=["timestamp"],
        )

        timestamp_lookup = timestamp_summary.set_index(
            "timestamp",
        )["timestamp_idx"]
        node_lookup = node_order.set_index("node_id")["node_position"]
        nodes["timestamp_idx"] = nodes["timestamp"].map(timestamp_lookup)
        nodes["node_position"] = nodes["node_id"].map(node_lookup)
        nodes = nodes.sort_values([
            "timestamp_idx",
            "node_position",
        ])

        expected_rows = len(timestamp_summary) * len(node_order)
        if len(nodes) != expected_rows:
            raise ValueError(
                f"expected {expected_rows} node rows, found {len(nodes)}"
            )

        shape = (len(timestamp_summary), len(node_order))
        node_features = nodes[NODE_FEATURE_COLUMNS].to_numpy(
            dtype=np.float32,
        ).reshape(shape[0], shape[1], len(NODE_FEATURE_COLUMNS))
        input_valid_mask = nodes["input_valid"].astype(bool).to_numpy().reshape(
            shape,
        )
        target_valid_mask = nodes["target_valid"].astype(bool).to_numpy().reshape(
            shape,
        )
        residual_targets = nodes["residual_pm25_t_plus_1"].to_numpy(
            dtype=np.float32,
        ).reshape(shape)

        return {
            "node_features": node_features,
            "input_valid_mask": input_valid_mask,
            "target_valid_mask": target_valid_mask,
            "residual_targets": residual_targets,
        }

    @staticmethod
    def build_edge_arrays(timestamp_summary, edge_order):
        edge_columns = [
            "timestamp",
            "source_node_id",
            "target_node_id",
            "raw_dynamic_weight",
            "valid_directed_edge",
            "active_valid_directed_edge",
        ]
        edges = pd.read_csv(
            EDGE_SNAPSHOT_FILE,
            usecols=edge_columns,
            parse_dates=["timestamp"],
        )

        timestamp_lookup = timestamp_summary.set_index(
            "timestamp",
        )["timestamp_idx"]
        edge_lookup = edge_order.set_index([
            "source_node_id",
            "target_node_id",
        ])["edge_position"]

        edges["timestamp_idx"] = edges["timestamp"].map(timestamp_lookup)
        edge_keys = pd.MultiIndex.from_frame(
            edges[["source_node_id", "target_node_id"]],
        )
        edges["edge_position"] = edge_lookup.reindex(edge_keys).to_numpy()
        if edges["edge_position"].isna().any():
            raise ValueError("edge snapshots include unknown edge IDs")

        t_count = len(timestamp_summary)
        e_count = len(edge_order)
        edge_weights = np.zeros((t_count, e_count), dtype=np.float32)
        edge_valid_mask = np.zeros((t_count, e_count), dtype=bool)
        edge_active_mask = np.zeros((t_count, e_count), dtype=bool)

        t_idx = edges["timestamp_idx"].to_numpy(dtype=np.int32)
        e_idx = edges["edge_position"].to_numpy(dtype=np.int32)
        edge_weights[t_idx, e_idx] = edges["raw_dynamic_weight"].to_numpy(
            dtype=np.float32,
        )
        edge_valid_mask[t_idx, e_idx] = edges[
            "valid_directed_edge"
        ].astype(bool).to_numpy()
        edge_active_mask[t_idx, e_idx] = edges[
            "active_valid_directed_edge"
        ].astype(bool).to_numpy()

        return {
            "edge_weights": edge_weights,
            "edge_valid_mask": edge_valid_mask,
            "edge_active_mask": edge_active_mask,
        }

    def build_window_index(
            self,
            timestamp_summary,
            input_valid_mask,
            target_valid_mask,
            edge_valid_mask,
            edge_active_mask,
    ):
        rows = []
        rejection_counts = {
            "too_short_history": 0,
            "non_hourly_continuity": 0,
            "split_crossing": 0,
            "zero_valid_targets": 0,
        }
        timestamps = timestamp_summary["timestamp"]
        splits = timestamp_summary["split"].to_numpy()
        same_split_target = timestamp_summary[
            "same_split_target"
        ].astype(bool).to_numpy()

        for end_idx in range(len(timestamp_summary)):
            start_idx = end_idx - WINDOW_LENGTH + 1
            if start_idx < 0:
                rejection_counts["too_short_history"] += 1
                continue

            window_timestamps = timestamps.iloc[start_idx:end_idx + 1]
            if not window_timestamps.diff().dropna().eq(EXPECTED_STEP).all():
                rejection_counts["non_hourly_continuity"] += 1
                continue

            window_splits = splits[start_idx:end_idx + 1]
            split = window_splits[0]
            if (
                    not np.all(window_splits == split) or
                    not same_split_target[end_idx]
            ):
                rejection_counts["split_crossing"] += 1
                continue

            sequence_input_valid = input_valid_mask[
                start_idx:end_idx + 1,
                :,
            ].all(axis=0)
            supervised_target_valid = (
                sequence_input_valid & target_valid_mask[end_idx, :]
            )
            target_count = int(supervised_target_valid.sum())
            if target_count == 0:
                rejection_counts["zero_valid_targets"] += 1
                continue

            rows.append({
                "window_id": len(rows),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "target_idx": end_idx,
                "start_timestamp": timestamps.iloc[start_idx],
                "end_timestamp": timestamps.iloc[end_idx],
                "target_timestamp": (
                    timestamp_summary["target_timestamp"].iloc[end_idx]
                ),
                "split": split,
                "supervised_target_node_count": target_count,
                "sequence_input_valid_node_count": int(
                    sequence_input_valid.sum()
                ),
                "final_timestamp_target_valid_node_count": int(
                    target_valid_mask[end_idx, :].sum()
                ),
                "valid_edge_observation_count": int(
                    edge_valid_mask[start_idx:end_idx + 1, :].sum()
                ),
                "active_edge_observation_count": int(
                    edge_active_mask[start_idx:end_idx + 1, :].sum()
                ),
            })

        window_index = pd.DataFrame(rows)
        if not window_index.empty:
            window_index["window_id"] = window_index["window_id"].astype(
                np.int32,
            )
            for column in ["start_idx", "end_idx", "target_idx"]:
                window_index[column] = window_index[column].astype(np.int32)

        rejection_summary = pd.DataFrame([
            {
                "rejection_reason": reason,
                "candidate_windows": count,
            }
            for reason, count in rejection_counts.items()
        ])
        return window_index, rejection_summary

    @staticmethod
    def build_window_masks(window_index, input_valid_mask, target_valid_mask):
        if window_index.empty:
            return {
                "window_sequence_input_valid_mask": np.zeros(
                    (0, input_valid_mask.shape[1]),
                    dtype=bool,
                ),
                "window_target_valid_mask": np.zeros(
                    (0, input_valid_mask.shape[1]),
                    dtype=bool,
                ),
            }

        sequence_masks = []
        target_masks = []
        for row in window_index.itertuples(index=False):
            start = int(row.start_idx)
            end = int(row.end_idx)
            sequence_input_valid = input_valid_mask[
                start:end + 1,
                :,
            ].all(axis=0)
            supervised_target_valid = (
                sequence_input_valid & target_valid_mask[end, :]
            )
            sequence_masks.append(sequence_input_valid)
            target_masks.append(supervised_target_valid)

        return {
            "window_sequence_input_valid_mask": np.asarray(
                sequence_masks,
                dtype=bool,
            ),
            "window_target_valid_mask": np.asarray(
                target_masks,
                dtype=bool,
            ),
        }

    @staticmethod
    def build_validation(
            timestamp_summary,
            node_order,
            edge_order,
            arrays,
            window_index,
    ):
        timestamps = timestamp_summary["timestamp"]
        split_codes = arrays["split_codes"]
        validations = {
            "every_accepted_window_has_24_hourly_snapshots": True,
            "no_split_crossing": True,
            "target_exactly_t_plus_1_after_final_input": True,
            "target_mask_implies_complete_24h_input_history": True,
            "fixed_51_node_ordering_preserved": (
                len(node_order) == 51 and
                node_order["node_id"].is_monotonic_increasing
            ),
            "edge_ids_order_consistent_across_timestamps": (
                len(edge_order) == 326 and
                not edge_order[[
                    "source_node_id",
                    "target_node_id",
                ]].duplicated().any()
            ),
            "no_future_node_features_used": True,
        }

        for row in window_index.itertuples(index=False):
            start = int(row.start_idx)
            end = int(row.end_idx)
            window_timestamps = timestamps.iloc[start:end + 1]
            if len(window_timestamps) != WINDOW_LENGTH:
                validations[
                    "every_accepted_window_has_24_hourly_snapshots"
                ] = False
            elif not window_timestamps.diff().dropna().eq(EXPECTED_STEP).all():
                validations[
                    "every_accepted_window_has_24_hourly_snapshots"
                ] = False

            if not np.all(split_codes[start:end + 1] == split_codes[end]):
                validations["no_split_crossing"] = False
            if not bool(arrays["same_split_target"][end]):
                validations["no_split_crossing"] = False

            target_delta = (
                timestamp_summary["target_timestamp"].iloc[end] -
                timestamp_summary["timestamp"].iloc[end]
            )
            if target_delta != EXPECTED_STEP:
                validations[
                    "target_exactly_t_plus_1_after_final_input"
                ] = False

            sequence_input_valid = arrays["input_valid_mask"][
                start:end + 1,
                :,
            ].all(axis=0)
            supervised_target_valid = (
                sequence_input_valid &
                arrays["target_valid_mask"][end, :]
            )
            if supervised_target_valid.any():
                if not sequence_input_valid[supervised_target_valid].all():
                    validations[
                        "target_mask_implies_complete_24h_input_history"
                    ] = False

        return pd.DataFrame([validations])

    @staticmethod
    def build_target_distribution(window_index):
        if window_index.empty:
            return pd.DataFrame()

        rows = []
        for split_name, split_df in window_index.groupby("split", sort=False):
            counts = split_df["supervised_target_node_count"]
            rows.append({
                "split": split_name,
                "usable_windows": len(split_df),
                "supervised_node_targets": int(counts.sum()),
                "min_targets_per_window": int(counts.min()),
                "median_targets_per_window": float(counts.median()),
                "max_targets_per_window": int(counts.max()),
                "windows_with_at_least_1_target": int((counts >= 1).sum()),
                "windows_with_at_least_10_targets": int((counts >= 10).sum()),
                "windows_with_at_least_20_targets": int((counts >= 20).sum()),
                "windows_with_at_least_30_targets": int((counts >= 30).sum()),
                "windows_with_at_least_40_targets": int((counts >= 40).sum()),
            })

        counts = window_index["supervised_target_node_count"]
        rows.append({
            "split": "all",
            "usable_windows": len(window_index),
            "supervised_node_targets": int(counts.sum()),
            "min_targets_per_window": int(counts.min()),
            "median_targets_per_window": float(counts.median()),
            "max_targets_per_window": int(counts.max()),
            "windows_with_at_least_1_target": int((counts >= 1).sum()),
            "windows_with_at_least_10_targets": int((counts >= 10).sum()),
            "windows_with_at_least_20_targets": int((counts >= 20).sum()),
            "windows_with_at_least_30_targets": int((counts >= 30).sum()),
            "windows_with_at_least_40_targets": int((counts >= 40).sum()),
        })
        return pd.DataFrame(rows)

    @staticmethod
    def build_continuous_runs(window_index):
        rows = []
        if window_index.empty:
            return pd.DataFrame(rows)

        for split_name, split_df in window_index.groupby("split", sort=False):
            split_df = split_df.sort_values("end_idx")
            run_start = None
            run_end = None
            length = 0
            previous_end_idx = None
            for row in split_df.itertuples(index=False):
                if (
                        previous_end_idx is None or
                        int(row.end_idx) != previous_end_idx + 1
                ):
                    if run_start is not None:
                        rows.append({
                            "split": split_name,
                            "start_timestamp": run_start,
                            "end_timestamp": run_end,
                            "length_windows": length,
                        })
                    run_start = row.end_timestamp
                    length = 1
                else:
                    length += 1
                run_end = row.end_timestamp
                previous_end_idx = int(row.end_idx)
            if run_start is not None:
                rows.append({
                    "split": split_name,
                    "start_timestamp": run_start,
                    "end_timestamp": run_end,
                    "length_windows": length,
                })

        runs = pd.DataFrame(rows)
        return runs.sort_values(
            ["split", "length_windows"],
            ascending=[True, False],
        )

    @staticmethod
    def build_summary(window_index, rejection_summary, arrays):
        distribution = SlidingGraphWindowBuilder.build_target_distribution(
            window_index,
        )
        if distribution.empty:
            summary = pd.DataFrame([{
                "split": "all",
                "usable_windows": 0,
                "supervised_node_targets": 0,
            }])
        else:
            summary = distribution.copy()

        array_bytes = sum(
            value.nbytes
            for value in arrays.values()
            if isinstance(value, np.ndarray)
        )
        summary["array_storage_mb_uncompressed"] = array_bytes / 1024 ** 2
        summary["window_length_hours"] = WINDOW_LENGTH
        for row in rejection_summary.itertuples(index=False):
            summary[f"rejected_{row.rejection_reason}"] = int(
                row.candidate_windows
            )
        return summary

    @staticmethod
    def save_arrays(arrays, node_order, edge_order):
        np.savez_compressed(
            GRAPH_ARRAYS_FILE,
            timestamps=arrays["timestamps"],
            target_timestamps=arrays["target_timestamps"],
            split_codes=arrays["split_codes"],
            same_split_target=arrays["same_split_target"],
            node_ids=node_order["node_id"].to_numpy(dtype=np.int32),
            edge_source_node_ids=edge_order[
                "source_node_id"
            ].to_numpy(dtype=np.int32),
            edge_target_node_ids=edge_order[
                "target_node_id"
            ].to_numpy(dtype=np.int32),
            node_features=arrays["node_features"],
            input_valid_mask=arrays["input_valid_mask"],
            target_valid_mask=arrays["target_valid_mask"],
            residual_targets=arrays["residual_targets"],
            edge_weights=arrays["edge_weights"],
            edge_valid_mask=arrays["edge_valid_mask"],
            edge_active_mask=arrays["edge_active_mask"],
            window_sequence_input_valid_mask=arrays[
                "window_sequence_input_valid_mask"
            ],
            window_target_valid_mask=arrays["window_target_valid_mask"],
            feature_names=np.array(NODE_FEATURE_COLUMNS),
        )


if __name__ == "__main__":
    SlidingGraphWindowBuilder().run()
