"""
Redesign graph experiment timeline from deployment/data availability.

This script is analysis-only. It reads existing graph snapshot/window
artifacts, evaluates graph-era split policies, and writes separate
protocol artifacts. It does not modify station-wise splits, graph masks,
dynamic edges, snapshots, windows, or train any model.
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
CORE_DEPLOYED_FRACTION = 0.80
OUTPUT_DIR = RESULTS_DIR / "graph_experiment_timeline"

NODE_SNAPSHOT_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_nodes.csv.gz"
WINDOW_ARRAYS_FILE = GRAPH_SNAPSHOTS_DIR / "graph_window_arrays.npz"
SUPERVISED_NODES_FILE = GRAPH_SNAPSHOTS_DIR / "supervised_nodes.csv"

POLICY_SUMMARY_FILE = OUTPUT_DIR / "graph_timeline_policy_summary.csv"
POLICY_NODE_FILE = OUTPUT_DIR / "graph_timeline_policy_nodes.csv"
POLICY_SPLIT_FILE = OUTPUT_DIR / "graph_timeline_splits.csv"
POLICY_MONTHLY_FILE = OUTPUT_DIR / "graph_timeline_monthly_coverage.csv"
POLICY_RECOMMENDATION_FILE = OUTPUT_DIR / "graph_timeline_recommendation.md"


class GraphExperimentTimelineRedesign:

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        nodes = self.load_nodes()
        arrays = np.load(WINDOW_ARRAYS_FILE)
        snapshot_nodes = self.load_snapshot_nodes()
        timestamps = pd.to_datetime(arrays["timestamps"])
        target_timestamps = pd.to_datetime(arrays["target_timestamps"])
        first_sequence_times = self.compute_first_sequence_times(
            nodes,
            arrays,
            timestamps,
        )
        policies = self.define_policies(
            nodes,
            first_sequence_times,
            timestamps,
        )

        policy_summaries = []
        split_rows = []
        node_rows = []
        monthly_rows = []

        for policy in policies:
            result = self.evaluate_policy(
                policy,
                nodes,
                arrays,
                snapshot_nodes,
                timestamps,
                target_timestamps,
                first_sequence_times,
            )
            policy_summaries.append(result["summary"])
            split_rows.extend(result["splits"])
            node_rows.extend(result["nodes"])
            monthly_rows.extend(result["monthly"])

        summary_df = pd.DataFrame(policy_summaries)
        split_df = pd.DataFrame(split_rows)
        node_df = pd.DataFrame(node_rows)
        monthly_df = pd.DataFrame(monthly_rows)

        summary_df.to_csv(POLICY_SUMMARY_FILE, index=False)
        split_df.to_csv(POLICY_SPLIT_FILE, index=False)
        node_df.to_csv(POLICY_NODE_FILE, index=False)
        monthly_df.to_csv(POLICY_MONTHLY_FILE, index=False)
        self.write_recommendation(summary_df, split_df)

        logger.info("=" * 50)
        logger.info("Graph Timeline Redesign")
        logger.info("=" * 50)
        for row in summary_df.itertuples(index=False):
            logger.info(
                f"{row.policy_id}: {row.node_count} nodes, "
                f"start {row.era_start}, "
                f"train/validation nodes with targets "
                f"{row.train_nodes_with_targets}/"
                f"{row.validation_nodes_with_targets}"
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
            "target_valid",
            "pm2_5",
        ]
        return pd.read_csv(
            NODE_SNAPSHOT_FILE,
            usecols=columns,
            parse_dates=["timestamp"],
        )

    @staticmethod
    def compute_first_sequence_times(nodes, arrays, timestamps):
        input_valid = arrays["input_valid_mask"]
        target_valid = arrays["target_valid_mask"]
        rows = []
        for node in nodes.itertuples(index=False):
            pos = int(node.node_position)
            first_idx = None
            last_idx = None
            for end_idx in range(WINDOW_LENGTH - 1, len(timestamps)):
                sequence_input_valid = input_valid[
                    end_idx - WINDOW_LENGTH + 1:end_idx + 1,
                    pos,
                ].all()
                supervised = sequence_input_valid and target_valid[end_idx, pos]
                if supervised:
                    if first_idx is None:
                        first_idx = end_idx
                    last_idx = end_idx
            rows.append({
                "node_id": int(node.node_id),
                "node_position": pos,
                "dataset_name": node.dataset_name,
                "first_sequence_timestamp": (
                    timestamps[first_idx] if first_idx is not None else pd.NaT
                ),
                "last_sequence_timestamp": (
                    timestamps[last_idx] if last_idx is not None else pd.NaT
                ),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def define_policies(nodes, first_sequence_times, timestamps):
        deployed = first_sequence_times.dropna(
            subset=["first_sequence_timestamp"],
        ).sort_values("first_sequence_timestamp")
        if len(deployed) != len(nodes):
            raise ValueError(
                "all supervised nodes must have a first valid sequence time "
                "for all-node common-era policy"
            )

        all_start = deployed["first_sequence_timestamp"].max()
        all_node_ids = nodes["node_id"].astype(int).to_list()

        core_count = int(np.ceil(len(nodes) * CORE_DEPLOYED_FRACTION))
        core_start = deployed["first_sequence_timestamp"].iloc[core_count - 1]
        core_node_ids = deployed.iloc[:core_count]["node_id"].astype(
            int,
        ).sort_values().to_list()

        era_end = timestamps[-1]
        return [
            {
                "policy_id": "all_node_common_era",
                "description": (
                    "Start when all 51 supervised nodes have begun producing "
                    "valid 24h sequence targets; keep all nodes."
                ),
                "era_start": all_start,
                "era_end": era_end,
                "node_ids": all_node_ids,
                "selected_by": "latest first valid 24h sequence timestamp",
            },
            {
                "policy_id": "core_network_era",
                "description": (
                    "Start when at least 80% of supervised nodes have begun "
                    "producing valid 24h sequence targets; keep that fixed "
                    "deployed cohort."
                ),
                "era_start": core_start,
                "era_end": era_end,
                "node_ids": core_node_ids,
                "selected_by": (
                    f"earliest timestamp with >= {core_count}/51 deployed "
                    "sequence-capable nodes"
                ),
            },
        ]

    def evaluate_policy(
            self,
            policy,
            nodes,
            arrays,
            snapshot_nodes,
            timestamps,
            target_timestamps,
            first_sequence_times,
    ):
        node_positions = nodes[
            nodes["node_id"].isin(policy["node_ids"])
        ]["node_position"].to_numpy(dtype=np.int16)
        era_mask = (
            (timestamps >= policy["era_start"]) &
            (timestamps <= policy["era_end"])
        )
        era_indices = np.flatnonzero(era_mask)
        split_lookup, split_rows = self.build_policy_splits(
            policy,
            timestamps,
            era_indices,
        )
        window_rows, per_node_counts = self.evaluate_policy_windows(
            policy,
            arrays,
            timestamps,
            target_timestamps,
            node_positions,
            split_lookup,
        )
        window_df = pd.DataFrame(window_rows)
        node_rows = self.build_policy_node_rows(
            policy,
            nodes,
            node_positions,
            first_sequence_times,
            per_node_counts,
            snapshot_nodes,
        )
        monthly_rows = self.build_monthly_rows(policy, window_df)
        summary = self.build_policy_summary(
            policy,
            window_df,
            split_rows,
            node_rows,
            snapshot_nodes,
        )
        return {
            "summary": summary,
            "splits": split_rows,
            "nodes": node_rows,
            "monthly": monthly_rows,
        }

    @staticmethod
    def build_policy_splits(policy, timestamps, era_indices):
        n = len(era_indices)
        split_specs = [
            ("train", 0.00, 0.70),
            ("validation", 0.70, 0.85),
            ("test", 0.85, 1.00),
        ]
        split_lookup = {}
        rows = []
        for split_name, start_frac, end_frac in split_specs:
            start_pos = int(np.floor(n * start_frac))
            end_pos = int(np.floor(n * end_frac))
            if split_name == "test":
                end_pos = n
            split_indices = era_indices[start_pos:end_pos]
            for idx in split_indices:
                split_lookup[int(idx)] = split_name
            rows.append({
                "policy_id": policy["policy_id"],
                "split": split_name,
                "start_timestamp": (
                    timestamps[split_indices[0]]
                    if len(split_indices)
                    else pd.NaT
                ),
                "end_timestamp": (
                    timestamps[split_indices[-1]]
                    if len(split_indices)
                    else pd.NaT
                ),
                "timestamp_count": len(split_indices),
            })
        return split_lookup, rows

    @staticmethod
    def evaluate_policy_windows(
            policy,
            arrays,
            timestamps,
            target_timestamps,
            node_positions,
            split_lookup,
    ):
        input_valid = arrays["input_valid_mask"]
        target_valid = arrays["target_valid_mask"]
        per_node_counts = {
            split: np.zeros(len(node_positions), dtype=np.int32)
            for split in ["train", "validation", "test"]
        }
        rows = []
        for end_idx in sorted(split_lookup):
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
            if not pd.Series(window_timestamps).diff().dropna().eq(
                    EXPECTED_STEP,
            ).all():
                continue
            if target_timestamps[end_idx] - timestamps[end_idx] != EXPECTED_STEP:
                continue

            sequence_valid = input_valid[
                start_idx:end_idx + 1,
                :,
            ][:, node_positions].all(axis=0)
            target_mask = (
                sequence_valid &
                target_valid[end_idx, node_positions]
            )
            target_count = int(target_mask.sum())
            if target_count == 0:
                continue

            per_node_counts[split] += target_mask.astype(np.int32)
            rows.append({
                "policy_id": policy["policy_id"],
                "split": split,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "target_idx": target_idx,
                "start_timestamp": timestamps[start_idx],
                "end_timestamp": timestamps[end_idx],
                "target_timestamp": target_timestamps[end_idx],
                "supervised_target_node_count": target_count,
                "sequence_valid_node_count": int(sequence_valid.sum()),
                "final_target_valid_node_count": int(
                    target_valid[end_idx, node_positions].sum()
                ),
            })
        return rows, per_node_counts

    @staticmethod
    def build_policy_node_rows(
            policy,
            nodes,
            node_positions,
            first_sequence_times,
            per_node_counts,
            snapshot_nodes,
    ):
        rows = []
        position_to_offset = {
            int(position): offset
            for offset, position in enumerate(node_positions)
        }
        for node in nodes[
                nodes["node_position"].isin(node_positions)
        ].itertuples(index=False):
            offset = position_to_offset[int(node.node_position)]
            first_row = first_sequence_times[
                first_sequence_times["node_id"] == node.node_id
            ].iloc[0]
            node_snapshots = snapshot_nodes[
                (snapshot_nodes["node_id"] == node.node_id) &
                (snapshot_nodes["timestamp"] >= policy["era_start"]) &
                (snapshot_nodes["timestamp"] <= policy["era_end"])
            ]
            deployed_rows = node_snapshots[
                node_snapshots["node_exists"].astype(bool)
            ]
            pm_missing_after_deployment = int(
                deployed_rows["pm2_5"].isna().sum()
            )
            rows.append({
                "policy_id": policy["policy_id"],
                "node_id": int(node.node_id),
                "dataset_name": node.dataset_name,
                "station": node.station,
                "first_valid_24h_sequence_target_timestamp": (
                    first_row["first_sequence_timestamp"]
                ),
                "last_valid_24h_sequence_target_timestamp": (
                    first_row["last_sequence_timestamp"]
                ),
                "train_supervised_target_count": int(
                    per_node_counts["train"][offset]
                ),
                "validation_supervised_target_count": int(
                    per_node_counts["validation"][offset]
                ),
                "test_supervised_target_count": int(
                    per_node_counts["test"][offset]
                ),
                "has_train_targets": bool(
                    per_node_counts["train"][offset] > 0
                ),
                "has_validation_targets": bool(
                    per_node_counts["validation"][offset] > 0
                ),
                "node_exists_after_era_start": int(len(deployed_rows)),
                "pm25_missing_after_deployment_in_era": (
                    pm_missing_after_deployment
                ),
                "pm25_missing_after_deployment_rate": (
                    pm_missing_after_deployment / len(deployed_rows)
                    if len(deployed_rows)
                    else np.nan
                ),
            })
        return rows

    @staticmethod
    def build_monthly_rows(policy, window_df):
        if window_df.empty:
            return []
        window_df = window_df.copy()
        window_df["year_month"] = window_df["end_timestamp"].dt.to_period(
            "M",
        ).astype(str)
        rows = []
        for (split, year_month), group in window_df.groupby(
                ["split", "year_month"],
                sort=True,
        ):
            counts = group["supervised_target_node_count"]
            rows.append({
                "policy_id": policy["policy_id"],
                "split": split,
                "year_month": year_month,
                "usable_windows": len(group),
                "supervised_node_targets": int(counts.sum()),
                "mean_targets_per_window": float(counts.mean()),
                "median_targets_per_window": float(counts.median()),
                "max_targets_per_window": int(counts.max()),
            })
        return rows

    @staticmethod
    def build_policy_summary(
            policy,
            window_df,
            split_rows,
            node_rows,
            snapshot_nodes,
    ):
        node_df = pd.DataFrame(node_rows)
        split_df = pd.DataFrame(split_rows)
        duration_hours = int(
            (
                pd.Timestamp(policy["era_end"]) -
                pd.Timestamp(policy["era_start"])
            ) / EXPECTED_STEP
        ) + 1
        rows = {
            "policy_id": policy["policy_id"],
            "description": policy["description"],
            "selected_by": policy["selected_by"],
            "era_start": policy["era_start"],
            "era_end": policy["era_end"],
            "duration_hours": duration_hours,
            "duration_days": duration_hours / 24,
            "node_count": len(policy["node_ids"]),
            "train_nodes_with_targets": int(
                node_df["has_train_targets"].sum()
            ),
            "validation_nodes_with_targets": int(
                node_df["has_validation_targets"].sum()
            ),
            "pm25_missing_after_deployment_in_era": int(
                node_df["pm25_missing_after_deployment_in_era"].sum()
            ),
            "node_exists_after_era_start": int(
                node_df["node_exists_after_era_start"].sum()
            ),
        }
        rows["pm25_missing_after_deployment_rate"] = (
            rows["pm25_missing_after_deployment_in_era"] /
            rows["node_exists_after_era_start"]
            if rows["node_exists_after_era_start"]
            else np.nan
        )
        for split in ["train", "validation", "test"]:
            split_windows = window_df[window_df["split"] == split]
            counts = split_windows["supervised_target_node_count"]
            split_meta = split_df[split_df["split"] == split].iloc[0]
            rows[f"{split}_start"] = split_meta["start_timestamp"]
            rows[f"{split}_end"] = split_meta["end_timestamp"]
            rows[f"{split}_timestamp_count"] = int(
                split_meta["timestamp_count"]
            )
            rows[f"{split}_usable_windows"] = len(split_windows)
            rows[f"{split}_supervised_node_targets"] = int(counts.sum())
            rows[f"{split}_mean_targets_per_window"] = (
                float(counts.mean()) if len(counts) else 0.0
            )
            rows[f"{split}_median_targets_per_window"] = (
                float(counts.median()) if len(counts) else 0.0
            )
        return rows

    @staticmethod
    def write_recommendation(summary_df, split_df):
        all_node = summary_df[
            summary_df["policy_id"] == "all_node_common_era"
        ].iloc[0]
        core = summary_df[
            summary_df["policy_id"] == "core_network_era"
        ].iloc[0]
        text = f"""# Graph Experiment Timeline Redesign

This analysis used station deployment and data-availability timing only.
No graph model was trained, no performance was inspected, and no existing
station-wise split or graph artifact was overwritten.

The old global 2021-2026 test split has already been inspected for graph
coverage, so it is no longer a pristine graph-model test split. No graph
model performance has been inspected yet.

## Policy A: All-Node Common Era

- start: {all_node.era_start}
- end: {all_node.era_end}
- nodes: {int(all_node.node_count)}
- duration days: {all_node.duration_days:.1f}
- train usable windows: {int(all_node.train_usable_windows)}
- validation usable windows: {int(all_node.validation_usable_windows)}
- train nodes represented: {int(all_node.train_nodes_with_targets)}/{int(all_node.node_count)}
- validation nodes represented: {int(all_node.validation_nodes_with_targets)}/{int(all_node.node_count)}

This policy keeps all 51 nodes, but creates a tiny common era and does not
give every evaluated node train and validation supervision.

## Policy B: Core-Network Era

- start: {core.era_start}
- end: {core.era_end}
- nodes: {int(core.node_count)}
- duration days: {core.duration_days:.1f}
- train usable windows: {int(core.train_usable_windows)}
- validation usable windows: {int(core.validation_usable_windows)}
- train nodes represented: {int(core.train_nodes_with_targets)}/{int(core.node_count)}
- validation nodes represented: {int(core.validation_nodes_with_targets)}/{int(core.node_count)}

Recommendation: use Policy B as the graph-specific dataset protocol,
pending review. It keeps a large enough fixed cohort for spatial
learning, provides a longer chronological training era, and avoids the
tiny common-era problem. The graph-specific splits must be stored and
reported separately from the old station-wise/global splits.

Do not use the old inspected global test split as a pristine graph-model
test. For graph modeling, create a fresh graph-specific chronological
holdout inside the selected graph era, and do not inspect graph-model
performance on that holdout until the protocol is frozen.
"""
        POLICY_RECOMMENDATION_FILE.write_text(text, encoding="utf-8")


def main():
    GraphExperimentTimelineRedesign().run()


if __name__ == "__main__":
    main()
