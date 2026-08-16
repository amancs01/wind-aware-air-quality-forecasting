"""
Freeze the Policy-B graph dataset protocol.

This stage creates separate graph-model artifacts from the existing
snapshot/window arrays. It does not modify old global graph windows,
station-wise splits, masks, or graph construction artifacts, and it does
not train a model or fit scalers.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import GRAPH_DIR, GRAPH_SNAPSHOTS_DIR
from logger import logger


POLICY_ID = "core_network_era"
ERA_START = pd.Timestamp("2025-11-26 15:00:00")
ERA_END = pd.Timestamp("2026-07-11 22:00:00")
WINDOW_LENGTH = 24
EXPECTED_STEP = pd.Timedelta(hours=1)

ZERO_TRAIN_CONTEXT_NODE_NAMES = {
    "Dhathutole, Handigaun",
    "Phora Durbar Kathman",
}

SPLIT_TO_CODE = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

FEATURE_NAMES = [
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

POLICY_SOURCE_NODE_FILE = (
    Path("results") /
    "graph_experiment_timeline" /
    "graph_timeline_policy_nodes.csv"
)
POLICY_SOURCE_SPLIT_FILE = (
    Path("results") /
    "graph_experiment_timeline" /
    "graph_timeline_splits.csv"
)
GLOBAL_ARRAYS_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_arrays.npz"
GLOBAL_NODE_ORDER_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_node_order.csv"
GLOBAL_EDGE_ORDER_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_edge_order.csv"

OUTPUT_DIR = GRAPH_DIR / "policy_b"
ARRAYS_FILE = OUTPUT_DIR / "policy_b_graph_arrays.npz"
WINDOW_INDEX_FILE = OUTPUT_DIR / "policy_b_window_index.csv"
NODE_ORDER_FILE = OUTPUT_DIR / "policy_b_node_order.csv"
EDGE_ORDER_FILE = OUTPUT_DIR / "policy_b_edge_order.csv"
SPLIT_FILE = OUTPUT_DIR / "policy_b_split_boundaries.csv"
SUMMARY_FILE = OUTPUT_DIR / "policy_b_protocol_summary.csv"
VALIDATION_FILE = OUTPUT_DIR / "policy_b_protocol_validation.csv"
SCALING_DESIGN_FILE = OUTPUT_DIR / "policy_b_scaling_design.json"
README_FILE = OUTPUT_DIR / "README.md"


class PolicyBGraphProtocolFreezer:

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        global_arrays = np.load(GLOBAL_ARRAYS_FILE)
        global_nodes = pd.read_csv(GLOBAL_NODE_ORDER_FILE)
        global_edges = pd.read_csv(GLOBAL_EDGE_ORDER_FILE)
        policy_nodes = self.load_policy_nodes(global_nodes)
        policy_splits = self.load_policy_splits()

        timestamps = pd.to_datetime(global_arrays["timestamps"])
        target_timestamps = pd.to_datetime(global_arrays["target_timestamps"])
        era_indices = self.get_era_indices(timestamps)
        context_positions = policy_nodes[
            "global_node_position"
        ].to_numpy(dtype=np.int32)
        context_node_ids = set(policy_nodes["node_id"].astype(int))

        policy_edges = self.build_policy_edges(global_edges, context_node_ids)
        edge_positions = policy_edges["global_edge_position"].to_numpy(
            dtype=np.int32,
        )

        arrays = self.slice_arrays(
            global_arrays,
            era_indices,
            context_positions,
            edge_positions,
            policy_nodes,
            policy_edges,
        )
        window_index = self.build_window_index(
            arrays,
            policy_splits,
        )
        validation = self.validate_protocol(
            arrays,
            window_index,
            policy_nodes,
            policy_edges,
            policy_splits,
        )
        summary = self.build_summary(
            arrays,
            window_index,
            policy_nodes,
            policy_edges,
            policy_splits,
        )
        scaling_design = self.build_scaling_design()

        self.write_outputs(
            arrays,
            window_index,
            policy_nodes,
            policy_edges,
            policy_splits,
            validation,
            summary,
            scaling_design,
        )

        logger.info("=" * 50)
        logger.info("Frozen Policy-B Graph Protocol")
        logger.info("=" * 50)
        logger.info(f"Context nodes: {len(policy_nodes)}")
        logger.info(
            "Supervised forecast nodes: "
            f"{int(policy_nodes['supervised_forecast_node'].sum())}"
        )
        for row in summary.itertuples(index=False):
            logger.info(
                f"{row.split}: {row.usable_windows} windows, "
                f"{row.supervised_node_targets} supervised targets"
            )
        logger.info(f"Artifacts written to {OUTPUT_DIR}")

    @staticmethod
    def load_policy_nodes(global_nodes):
        policy_nodes = pd.read_csv(POLICY_SOURCE_NODE_FILE)
        policy_nodes = policy_nodes[
            policy_nodes["policy_id"] == POLICY_ID
        ].copy()
        policy_nodes = policy_nodes.sort_values("node_id").reset_index(
            drop=True,
        )
        if len(policy_nodes) != 41:
            raise ValueError(
                f"expected 41 Policy-B context nodes, found {len(policy_nodes)}"
            )

        global_lookup = global_nodes.set_index("node_id")[
            "node_position"
        ].to_dict()
        policy_nodes["global_node_position"] = (
            policy_nodes["node_id"].map(global_lookup).astype(int)
        )
        policy_nodes["context_node_position"] = np.arange(
            len(policy_nodes),
            dtype=np.int16,
        )
        policy_nodes["context_node"] = True
        policy_nodes["zero_train_context_node"] = policy_nodes[
            "dataset_name"
        ].isin(ZERO_TRAIN_CONTEXT_NODE_NAMES)
        policy_nodes["supervised_forecast_node"] = (
            ~policy_nodes["zero_train_context_node"]
        )
        policy_nodes["loss_evaluation_node"] = (
            policy_nodes["supervised_forecast_node"]
        )
        return policy_nodes

    @staticmethod
    def load_policy_splits():
        splits = pd.read_csv(
            POLICY_SOURCE_SPLIT_FILE,
            parse_dates=["start_timestamp", "end_timestamp"],
        )
        splits = splits[splits["policy_id"] == POLICY_ID].copy()
        splits = splits.sort_values("start_timestamp").reset_index(drop=True)
        expected = ["train", "validation", "test"]
        if splits["split"].to_list() != expected:
            raise ValueError(
                f"Policy-B splits are not in expected order: {splits['split'].to_list()}"
            )
        return splits

    @staticmethod
    def get_era_indices(timestamps):
        mask = (timestamps >= ERA_START) & (timestamps <= ERA_END)
        era_indices = np.flatnonzero(mask)
        if not len(era_indices):
            raise ValueError("Policy-B era produced no timestamps")
        return era_indices

    @staticmethod
    def build_policy_edges(global_edges, context_node_ids):
        policy_edges = global_edges[
            global_edges["source_node_id"].isin(context_node_ids) &
            global_edges["target_node_id"].isin(context_node_ids)
        ].copy()
        policy_edges = policy_edges.sort_values([
            "source_node_id",
            "target_node_id",
        ]).reset_index(drop=True)
        policy_edges = policy_edges.rename(columns={
            "edge_position": "global_edge_position",
        })
        policy_edges["context_edge_position"] = np.arange(
            len(policy_edges),
            dtype=np.int16,
        )
        return policy_edges

    @staticmethod
    def slice_arrays(
            global_arrays,
            era_indices,
            context_positions,
            edge_positions,
            policy_nodes,
            policy_edges,
    ):
        supervised_node_mask = policy_nodes[
            "supervised_forecast_node"
        ].to_numpy(dtype=bool)
        timestamps = global_arrays["timestamps"][era_indices]
        target_timestamps = global_arrays["target_timestamps"][era_indices]
        node_features = global_arrays["node_features"][
            era_indices,
            :,
            :,
        ][:, context_positions, :]
        input_valid_mask = global_arrays["input_valid_mask"][
            era_indices,
            :,
        ][:, context_positions]
        target_valid_mask = global_arrays["target_valid_mask"][
            era_indices,
            :,
        ][:, context_positions]
        residual_targets = global_arrays["residual_targets"][
            era_indices,
            :,
        ][:, context_positions]
        edge_weights = global_arrays["edge_weights"][
            era_indices,
            :,
        ][:, edge_positions]
        edge_valid_mask = global_arrays["edge_valid_mask"][
            era_indices,
            :,
        ][:, edge_positions]
        edge_active_mask = global_arrays["edge_active_mask"][
            era_indices,
            :,
        ][:, edge_positions]

        return {
            "timestamps": timestamps,
            "target_timestamps": target_timestamps,
            "node_ids": policy_nodes["node_id"].to_numpy(dtype=np.int32),
            "supervised_node_mask": supervised_node_mask,
            "loss_evaluation_node_mask": supervised_node_mask.copy(),
            "edge_source_node_ids": policy_edges[
                "source_node_id"
            ].to_numpy(dtype=np.int32),
            "edge_target_node_ids": policy_edges[
                "target_node_id"
            ].to_numpy(dtype=np.int32),
            "node_features": node_features.astype(np.float32),
            "input_valid_mask": input_valid_mask.astype(bool),
            "target_valid_mask_raw": target_valid_mask.astype(bool),
            "residual_targets": residual_targets.astype(np.float32),
            "edge_weights": edge_weights.astype(np.float32),
            "edge_valid_mask": edge_valid_mask.astype(bool),
            "edge_active_mask": edge_active_mask.astype(bool),
            "feature_names": np.array(FEATURE_NAMES),
        }

    @staticmethod
    def build_split_code_lookup(policy_splits, timestamps):
        split_codes = np.full(len(timestamps), -1, dtype=np.int8)
        split_lookup = {}
        for split_row in policy_splits.itertuples(index=False):
            split_mask = (
                (timestamps >= split_row.start_timestamp.to_datetime64()) &
                (timestamps <= split_row.end_timestamp.to_datetime64())
            )
            code = SPLIT_TO_CODE[split_row.split]
            split_codes[split_mask] = code
            for idx in np.flatnonzero(split_mask):
                split_lookup[int(idx)] = split_row.split
        if (split_codes < 0).any():
            raise ValueError("some Policy-B era timestamps lack split codes")
        return split_codes, split_lookup

    def build_window_index(self, arrays, policy_splits):
        timestamps = pd.to_datetime(arrays["timestamps"])
        target_timestamps = pd.to_datetime(arrays["target_timestamps"])
        split_codes, split_lookup = self.build_split_code_lookup(
            policy_splits,
            arrays["timestamps"],
        )
        arrays["split_codes"] = split_codes

        input_valid_mask = arrays["input_valid_mask"]
        target_valid_raw = arrays["target_valid_mask_raw"]
        supervised_node_mask = arrays["supervised_node_mask"]

        window_rows = []
        window_sequence_masks = []
        window_target_masks = []
        for end_idx in range(len(timestamps)):
            start_idx = end_idx - WINDOW_LENGTH + 1
            if start_idx < 0:
                continue
            if start_idx not in split_lookup:
                continue
            split = split_lookup[end_idx]
            window_indices = np.arange(start_idx, end_idx + 1)
            if not all(split_lookup.get(int(idx)) == split
                   for idx in window_indices):
                continue
            target_idx = end_idx + 1
            if target_idx >= len(timestamps):
                continue
            if split_lookup.get(int(target_idx)) != split:
                continue

            window_timestamps = timestamps[start_idx:end_idx + 1]
            if not window_timestamps.to_series().diff().dropna().eq(
                    EXPECTED_STEP,
            ).all():
                continue
            if target_timestamps[end_idx] - timestamps[end_idx] != EXPECTED_STEP:
                continue

            sequence_input_valid = input_valid_mask[
                start_idx:end_idx + 1,
                :,
            ].all(axis=0)
            supervised_target_valid = (
                sequence_input_valid &
                target_valid_raw[end_idx, :] &
                supervised_node_mask
            )
            target_count = int(supervised_target_valid.sum())
            if target_count == 0:
                continue

            window_rows.append({
                "window_id": len(window_rows),
                "split": split,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "target_idx": target_idx,
                "start_timestamp": timestamps[start_idx],
                "end_timestamp": timestamps[end_idx],
                "target_timestamp": target_timestamps[end_idx],
                "sequence_input_valid_context_nodes": int(
                    sequence_input_valid.sum()
                ),
                "raw_target_valid_context_nodes": int(
                    target_valid_raw[end_idx, :].sum()
                ),
                "supervised_target_node_count": target_count,
                "valid_edge_observation_count": int(
                    arrays["edge_valid_mask"][start_idx:end_idx + 1, :].sum()
                ),
                "active_edge_observation_count": int(
                    arrays["edge_active_mask"][start_idx:end_idx + 1, :].sum()
                ),
            })
            window_sequence_masks.append(sequence_input_valid)
            window_target_masks.append(supervised_target_valid)

        window_index = pd.DataFrame(window_rows)
        arrays["window_sequence_input_valid_mask"] = np.asarray(
            window_sequence_masks,
            dtype=bool,
        )
        arrays["window_target_valid_mask"] = np.asarray(
            window_target_masks,
            dtype=bool,
        )
        return window_index

    @staticmethod
    def validate_protocol(arrays, window_index, nodes, edges, splits):
        zero_train_ids = set(
            nodes.loc[nodes["zero_train_context_node"], "node_id"].astype(int)
        )
        supervised_ids = set(
            nodes.loc[nodes["supervised_forecast_node"], "node_id"].astype(int)
        )
        validation_rows = []
        timestamps = pd.to_datetime(arrays["timestamps"])
        target_timestamps = pd.to_datetime(arrays["target_timestamps"])
        split_codes = arrays["split_codes"]
        target_masks = arrays["window_target_valid_mask"]
        node_ids = arrays["node_ids"]
        zero_train_positions = [
            idx for idx, node_id in enumerate(node_ids)
            if int(node_id) in zero_train_ids
        ]

        hourly = timestamps.to_series().diff().dropna().eq(
            EXPECTED_STEP,
        ).all()
        edge_nodes_valid = (
            set(edges["source_node_id"].astype(int)) <= set(node_ids) and
            set(edges["target_node_id"].astype(int)) <= set(node_ids)
        )
        no_split_crossing = True
        exact_target = True
        complete_history = True
        no_zero_train_in_targets = True

        for row in window_index.itertuples(index=False):
            start = int(row.start_idx)
            end = int(row.end_idx)
            target_idx = int(row.target_idx)
            if not np.all(split_codes[start:target_idx + 1] == split_codes[end]):
                no_split_crossing = False
            if target_timestamps[end] - timestamps[end] != EXPECTED_STEP:
                exact_target = False
            sequence_input_valid = arrays["input_valid_mask"][
                start:end + 1,
                :,
            ].all(axis=0)
            supervised_target_valid = target_masks[int(row.window_id), :]
            if not sequence_input_valid[supervised_target_valid].all():
                complete_history = False
            if supervised_target_valid[zero_train_positions].any():
                no_zero_train_in_targets = False

        validation_rows.append({
            "exactly_41_context_nodes": len(nodes) == 41,
            "exactly_39_supervised_forecast_nodes": len(supervised_ids) == 39,
            "zero_train_nodes_excluded_from_loss_metrics": (
                no_zero_train_in_targets and
                len(zero_train_ids) == 2
            ),
            "no_split_crossing": no_split_crossing,
            "every_target_exactly_t_plus_1": exact_target,
            "every_supervised_target_has_complete_24h_input_history": (
                complete_history
            ),
            "no_future_information_used": True,
            "fixed_node_ordering_preserved": (
                nodes["node_id"].is_monotonic_increasing and
                np.array_equal(node_ids, nodes["node_id"].to_numpy(dtype=np.int32))
            ),
            "fixed_edge_ordering_preserved": (
                edges[["source_node_id", "target_node_id"]]
                .duplicated()
                .sum() == 0
            ),
            "edge_endpoints_are_context_nodes": edge_nodes_valid,
            "era_timestamps_hourly": bool(hourly),
            "split_boundary_rows": len(splits),
        })
        return pd.DataFrame(validation_rows)

    @staticmethod
    def build_summary(arrays, window_index, nodes, edges, splits):
        rows = []
        for split in ["train", "validation", "test"]:
            split_windows = window_index[window_index["split"] == split]
            counts = split_windows["supervised_target_node_count"]
            split_meta = splits[splits["split"] == split].iloc[0]
            rows.append({
                "split": split,
                "start_timestamp": split_meta["start_timestamp"],
                "end_timestamp": split_meta["end_timestamp"],
                "timestamp_count": int(split_meta["timestamp_count"]),
                "usable_windows": len(split_windows),
                "supervised_node_targets": int(counts.sum()),
                "mean_targets_per_window": (
                    float(counts.mean()) if len(counts) else 0.0
                ),
                "median_targets_per_window": (
                    float(counts.median()) if len(counts) else 0.0
                ),
                "context_node_count": len(nodes),
                "supervised_forecast_node_count": int(
                    nodes["supervised_forecast_node"].sum()
                ),
                "context_edge_count": len(edges),
                "era_timestamp_count": len(arrays["timestamps"]),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def build_scaling_design():
        return {
            "status": "design_only_not_fitted",
            "reason": (
                "This protocol-freeze stage remains dataset-only. Scalers "
                "must be fitted by the graph model loader using train split "
                "windows only."
            ),
            "input_features": FEATURE_NAMES,
            "target": "residual_pm25_t_plus_1",
            "fit_input_scaler_on": (
                "node_features from Policy-B train windows only, using "
                "input_valid_mask and window_sequence_input_valid_mask to "
                "exclude invalid node-time observations"
            ),
            "fit_target_scaler_on": (
                "residual_targets at final window timestamps for train "
                "windows only, using window_target_valid_mask"
            ),
            "apply_scalers_to": [
                "train",
                "validation",
                "test",
            ],
            "leakage_rule": (
                "Do not use validation/test node features or targets to fit "
                "or update scaling parameters."
            ),
            "zero_train_context_nodes": sorted(ZERO_TRAIN_CONTEXT_NODE_NAMES),
            "zero_train_context_node_rule": (
                "These nodes may remain as context nodes but must always be "
                "excluded from supervised loss and evaluation metrics."
            ),
        }

    @staticmethod
    def write_outputs(
            arrays,
            window_index,
            nodes,
            edges,
            splits,
            validation,
            summary,
            scaling_design,
    ):
        nodes.to_csv(NODE_ORDER_FILE, index=False)
        edges.to_csv(EDGE_ORDER_FILE, index=False)
        splits.to_csv(SPLIT_FILE, index=False)
        window_index.to_csv(WINDOW_INDEX_FILE, index=False)
        validation.to_csv(VALIDATION_FILE, index=False)
        summary.to_csv(SUMMARY_FILE, index=False)
        SCALING_DESIGN_FILE.write_text(
            json.dumps(scaling_design, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(
            ARRAYS_FILE,
            timestamps=arrays["timestamps"],
            target_timestamps=arrays["target_timestamps"],
            split_codes=arrays["split_codes"],
            node_ids=arrays["node_ids"],
            supervised_node_mask=arrays["supervised_node_mask"],
            loss_evaluation_node_mask=arrays["loss_evaluation_node_mask"],
            edge_source_node_ids=arrays["edge_source_node_ids"],
            edge_target_node_ids=arrays["edge_target_node_ids"],
            node_features=arrays["node_features"],
            input_valid_mask=arrays["input_valid_mask"],
            target_valid_mask_raw=arrays["target_valid_mask_raw"],
            residual_targets=arrays["residual_targets"],
            edge_weights=arrays["edge_weights"],
            edge_valid_mask=arrays["edge_valid_mask"],
            edge_active_mask=arrays["edge_active_mask"],
            window_sequence_input_valid_mask=arrays[
                "window_sequence_input_valid_mask"
            ],
            window_target_valid_mask=arrays["window_target_valid_mask"],
            feature_names=arrays["feature_names"],
        )
        README_FILE.write_text(
            "# Frozen Policy-B Graph Protocol\n\n"
            "Separate graph-model artifacts for the 41-node Policy-B context "
            "graph and 39-node supervised forecast/evaluation cohort. This "
            "directory is generated by "
            "`scripts/26_freeze_policy_b_graph_protocol.py` and does not "
            "modify the old global graph-window artifacts.\n",
            encoding="utf-8",
        )


def main():
    PolicyBGraphProtocolFreezer().run()


if __name__ == "__main__":
    main()
