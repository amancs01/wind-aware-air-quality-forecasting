"""
Graph snapshot synchronization and mask construction.

Builds the first supervised graph-model data artifact from the corrected
51-node model-usable graph, sequence-native node features, residual
t+1 targets, and raw dynamic wind-edge weights. This stage does not train
a GNN, row-normalize edge weights, impute missing node values, or build
sliding graph windows.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    DYNAMIC_EDGE_FILE,
    FEATURED_DIR,
    GRAPH_SNAPSHOTS_DIR,
    STATIC_GRAPH_FILE,
    STATION_MAPPING_FILE,
)
from logger import logger


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

EXPECTED_STEP = pd.Timedelta(hours=1)

NODE_SNAPSHOT_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_nodes.csv.gz"
EDGE_SNAPSHOT_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_edges.csv.gz"
SUPERVISED_NODES_FILE = GRAPH_SNAPSHOTS_DIR / "supervised_nodes.csv"
POLICY_SUMMARY_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_policy_summary.csv"
TIMESTAMP_SUMMARY_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_timestamp_summary.csv"
VALIDATION_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_validation.csv"
RUNS_FILE = GRAPH_SNAPSHOTS_DIR / "snapshot_continuous_runs.csv"
VALID_NODE_DISTRIBUTION_FILE = (
    GRAPH_SNAPSHOTS_DIR / "snapshot_valid_node_distribution.csv"
)


class GraphSnapshotBuilder:

    REQUIRED_MAPPING_COLUMNS = [
        "node_id",
        "dataset_name",
        "pm25_sensor_id",
        "latitude",
        "longitude",
        "model_usable",
    ]

    REQUIRED_DYNAMIC_COLUMNS = [
        "timestamp",
        "source_node_id",
        "target_node_id",
        "source_dataset_name",
        "target_dataset_name",
        "source_model_usable",
        "target_model_usable",
        "supervised_edge",
        "raw_dynamic_weight",
        "edge_active",
    ]

    REQUIRED_STATIC_COLUMNS = [
        "source_node_id",
        "target_node_id",
    ]

    SPLITS = [
        ("train", 0.00, 0.70),
        ("validation", 0.70, 0.85),
        ("test", 0.85, 1.00),
    ]

    def __init__(self):
        self.output_dir = GRAPH_SNAPSHOTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        mapping = self.load_supervised_nodes()
        timestamp_index = self.build_global_hourly_index(mapping)
        split_lookup = self.build_split_lookup(timestamp_index)

        node_df = self.build_node_snapshots(mapping, timestamp_index)
        node_df = self.add_split_and_boundary_flags(node_df, split_lookup)
        static_supervised_edges = self.load_static_supervised_edges()
        edge_df = self.build_edge_snapshots(
            node_df,
            static_supervised_edges,
        )
        timestamp_summary = self.build_timestamp_summary(node_df)
        timestamp_summary = self.add_edge_counts_to_summary(
            timestamp_summary,
            edge_df,
        )

        policy_summary = self.build_policy_summary(timestamp_summary)
        validation = self.build_validation(
            mapping,
            timestamp_index,
            node_df,
            edge_df,
            static_supervised_edges,
            timestamp_summary,
        )
        runs_df = self.build_continuous_runs(timestamp_summary)
        distribution_df = self.build_node_distribution(timestamp_summary)

        node_df.to_csv(NODE_SNAPSHOT_FILE, index=False, compression="gzip")
        edge_df.to_csv(EDGE_SNAPSHOT_FILE, index=False, compression="gzip")
        timestamp_summary.to_csv(TIMESTAMP_SUMMARY_FILE, index=False)
        mapping.to_csv(SUPERVISED_NODES_FILE, index=False)
        policy_summary.to_csv(POLICY_SUMMARY_FILE, index=False)
        validation.to_csv(VALIDATION_FILE, index=False)
        runs_df.to_csv(RUNS_FILE, index=False)
        distribution_df.to_csv(VALID_NODE_DISTRIBUTION_FILE, index=False)

        logger.info("=" * 50)
        logger.info("Graph Snapshot Summary")
        logger.info("=" * 50)
        logger.info(f"Supervised nodes: {len(mapping)}")
        logger.info(f"Global hourly timestamps: {len(timestamp_index)}")
        logger.info(
            "Strict usable timestamps: "
            f"{int(policy_summary.loc[policy_summary['policy'] == 'strict', 'usable_timestamps'].iloc[0])}"
        )
        logger.info(
            "Masked usable timestamps: "
            f"{int(policy_summary.loc[policy_summary['policy'] == 'masked', 'usable_timestamps'].iloc[0])}"
        )

    def load_supervised_nodes(self):
        mapping = pd.read_csv(STATION_MAPPING_FILE)
        missing_columns = [
            column
            for column in self.REQUIRED_MAPPING_COLUMNS
            if column not in mapping.columns
        ]
        if missing_columns:
            raise ValueError(
                f"station mapping missing required columns: {missing_columns}"
            )

        supervised = mapping[mapping["model_usable"].astype(bool)].copy()
        supervised = supervised.sort_values("node_id").reset_index(drop=True)
        if len(supervised) != 51:
            raise ValueError(
                f"expected 51 model-usable nodes, found {len(supervised)}"
            )
        if supervised["node_id"].duplicated().any():
            raise ValueError("duplicate supervised node_id values found")
        if supervised["dataset_name"].duplicated().any():
            raise ValueError("duplicate supervised dataset_name values found")

        return supervised

    def build_global_hourly_index(self, mapping):
        min_timestamp = None
        max_timestamp = None

        for dataset_name in mapping["dataset_name"]:
            featured_file = FEATURED_DIR / f"{dataset_name}.csv"
            if not featured_file.exists():
                raise FileNotFoundError(
                    f"featured file missing for supervised node: {dataset_name}"
                )
            timestamps = pd.read_csv(featured_file, usecols=["timestamp"])
            timestamps["timestamp"] = pd.to_datetime(timestamps["timestamp"])
            station_min = timestamps["timestamp"].min()
            station_max = timestamps["timestamp"].max()
            min_timestamp = (
                station_min
                if min_timestamp is None
                else min(min_timestamp, station_min)
            )
            max_timestamp = (
                station_max
                if max_timestamp is None
                else max(max_timestamp, station_max)
            )

        return pd.date_range(
            start=min_timestamp,
            end=max_timestamp - EXPECTED_STEP,
            freq="h",
        )

    def build_split_lookup(self, timestamp_index):
        split_df = pd.DataFrame({"timestamp": timestamp_index})
        split_df["timestamp_rank"] = np.arange(len(split_df))
        split_df["split"] = None

        n_rows = len(split_df)
        for split_name, start_frac, end_frac in self.SPLITS:
            start = int(np.floor(n_rows * start_frac))
            end = int(np.floor(n_rows * end_frac))
            if split_name == self.SPLITS[-1][0]:
                end = n_rows
            split_df.loc[start:end - 1, "split"] = split_name

        split_df["target_timestamp"] = (
            split_df["timestamp"] + EXPECTED_STEP
        )
        target_split = split_df[["timestamp", "split"]].rename(columns={
            "timestamp": "target_timestamp",
            "split": "target_split",
        })
        split_df = split_df.merge(
            target_split,
            on="target_timestamp",
            how="left",
        )
        split_df["same_split_target"] = (
            split_df["split"] == split_df["target_split"]
        )

        return split_df[[
            "timestamp",
            "target_timestamp",
            "split",
            "target_split",
            "same_split_target",
        ]]

    def build_node_snapshots(self, mapping, timestamp_index):
        rows = []
        base = pd.DataFrame({"timestamp": timestamp_index})

        for node in mapping.itertuples(index=False):
            featured_file = FEATURED_DIR / f"{node.dataset_name}.csv"
            station_df = pd.read_csv(
                featured_file,
                usecols=["timestamp"] + NODE_FEATURE_COLUMNS,
            )
            station_df["timestamp"] = pd.to_datetime(
                station_df["timestamp"],
            )
            station_df["_row_exists"] = True
            station_df = station_df.drop_duplicates(
                subset=["timestamp"],
                keep="first",
            )
            station_df = station_df.sort_values("timestamp")
            station_df["target_timestamp"] = (
                station_df["timestamp"] + EXPECTED_STEP
            )
            target_df = station_df[["timestamp", "pm2_5"]].rename(columns={
                "timestamp": "target_timestamp",
                "pm2_5": "pm2_5_t_plus_1",
            })
            station_df = station_df.merge(
                target_df,
                on="target_timestamp",
                how="left",
            )

            snapshot = base.merge(
                station_df,
                on="timestamp",
                how="left",
            )
            snapshot.insert(1, "node_id", int(node.node_id))
            snapshot.insert(2, "dataset_name", node.dataset_name)
            snapshot.insert(3, "pm25_sensor_id", int(node.pm25_sensor_id))
            snapshot["node_exists"] = (
                snapshot["_row_exists"].fillna(False).astype(bool)
            )
            snapshot.drop(columns=["_row_exists"], inplace=True)
            snapshot["input_valid"] = (
                snapshot[NODE_FEATURE_COLUMNS].notna().all(axis=1)
            )
            snapshot["target_exists_t_plus_1"] = (
                snapshot["pm2_5_t_plus_1"].notna()
            )
            snapshot["residual_pm25_t_plus_1"] = (
                snapshot["pm2_5_t_plus_1"] - snapshot["pm2_5"]
            )
            rows.append(snapshot)

        return pd.concat(rows, ignore_index=True)

    @staticmethod
    def add_split_and_boundary_flags(node_df, split_lookup):
        node_df = node_df.merge(
            split_lookup,
            on="timestamp",
            how="left",
            suffixes=("_node", ""),
        )
        if "target_timestamp_node" in node_df.columns:
            node_df.drop(columns=["target_timestamp_node"], inplace=True)
        node_df["target_valid"] = (
            node_df["target_exists_t_plus_1"] &
            node_df["same_split_target"].fillna(False)
        )
        node_df["snapshot_supervised_usable"] = (
            node_df["input_valid"] & node_df["target_valid"]
        )
        return node_df

    def load_static_supervised_edges(self):
        static_edges = pd.read_csv(STATIC_GRAPH_FILE)
        missing_columns = [
            column
            for column in self.REQUIRED_STATIC_COLUMNS
            if column not in static_edges.columns
        ]
        if missing_columns:
            raise ValueError(
                f"static graph missing required columns: {missing_columns}"
            )

        mapping = self.load_supervised_nodes()
        supervised_node_ids = set(mapping["node_id"].astype(int))
        supervised_edges = static_edges[
            static_edges["source_node_id"].isin(supervised_node_ids) &
            static_edges["target_node_id"].isin(supervised_node_ids)
        ].copy()
        supervised_edges = supervised_edges[[
            "source_node_id",
            "target_node_id",
        ]].drop_duplicates()
        if len(supervised_edges) != 326:
            raise ValueError(
                "expected 326 supervised directed static candidates, "
                f"found {len(supervised_edges)}"
            )
        return supervised_edges

    def build_edge_snapshots(self, node_df, static_supervised_edges):
        dynamic = pd.read_csv(
            DYNAMIC_EDGE_FILE,
            usecols=self.REQUIRED_DYNAMIC_COLUMNS,
            parse_dates=["timestamp"],
        )
        dynamic = dynamic[
            dynamic["supervised_edge"].astype(bool)
        ].copy()
        snapshot_timestamps = set(node_df["timestamp"].unique())
        dynamic = dynamic[dynamic["timestamp"].isin(snapshot_timestamps)]

        static_edge_set = {
            (int(row.source_node_id), int(row.target_node_id))
            for row in static_supervised_edges.itertuples(index=False)
        }
        dynamic_edge_set = {
            (int(row.source_node_id), int(row.target_node_id))
            for row in dynamic[[
                "source_node_id",
                "target_node_id",
            ]].drop_duplicates().itertuples(index=False)
        }
        if not dynamic_edge_set <= static_edge_set:
            raise ValueError(
                "dynamic supervised edges include non-static candidates"
            )

        node_valid = node_df[[
            "timestamp",
            "node_id",
            "input_valid",
            "target_valid",
        ]].copy()
        source_valid = node_valid.rename(columns={
            "node_id": "source_node_id",
            "input_valid": "source_input_valid",
            "target_valid": "source_target_valid",
        })
        target_valid = node_valid.rename(columns={
            "node_id": "target_node_id",
            "input_valid": "target_input_valid",
            "target_valid": "target_target_valid",
        })

        edge_df = dynamic.merge(
            source_valid,
            on=["timestamp", "source_node_id"],
            how="left",
        ).merge(
            target_valid,
            on=["timestamp", "target_node_id"],
            how="left",
        )

        for column in [
                "source_input_valid",
                "source_target_valid",
                "target_input_valid",
                "target_target_valid",
        ]:
            edge_df[column] = edge_df[column].fillna(False).astype(bool)

        edge_df["valid_directed_edge"] = (
            edge_df["source_input_valid"] &
            edge_df["target_input_valid"]
        )
        edge_df["active_valid_directed_edge"] = (
            edge_df["valid_directed_edge"] &
            edge_df["edge_active"].astype(bool)
        )
        return edge_df

    @staticmethod
    def build_timestamp_summary(node_df):
        grouped = node_df.groupby("timestamp", sort=True)
        summary = grouped.agg(
            split=("split", "first"),
            target_timestamp=("target_timestamp", "first"),
            same_split_target=("same_split_target", "first"),
            existing_node_count=("node_exists", "sum"),
            valid_input_node_count=("input_valid", "sum"),
            valid_target_node_count=("target_valid", "sum"),
            valid_input_and_target_node_count=(
                "snapshot_supervised_usable",
                "sum",
            ),
        ).reset_index()
        summary["strict_usable"] = (
            summary["same_split_target"].fillna(False) &
            (summary["valid_input_node_count"] == 51) &
            (summary["valid_target_node_count"] == 51)
        )
        summary["masked_usable"] = (
            summary["same_split_target"].fillna(False) &
            (summary["valid_input_node_count"] > 0) &
            (summary["valid_target_node_count"] > 0)
        )
        return summary

    @staticmethod
    def add_edge_counts_to_summary(timestamp_summary, edge_df):
        edge_counts = edge_df.groupby("timestamp").agg(
            valid_directed_edge_count=("valid_directed_edge", "sum"),
            active_dynamic_edge_count=("active_valid_directed_edge", "sum"),
        ).reset_index()
        timestamp_summary = timestamp_summary.merge(
            edge_counts,
            on="timestamp",
            how="left",
        )
        timestamp_summary[[
            "valid_directed_edge_count",
            "active_dynamic_edge_count",
        ]] = timestamp_summary[[
            "valid_directed_edge_count",
            "active_dynamic_edge_count",
        ]].fillna(0).astype(int)
        return timestamp_summary

    def build_policy_summary(self, timestamp_summary):
        rows = []
        for policy, mask_column in [
                ("strict", "strict_usable"),
                ("masked", "masked_usable"),
        ]:
            policy_df = timestamp_summary[
                timestamp_summary[mask_column]
            ]
            rows.append({
                "policy": policy,
                "description": (
                    "all 51 nodes have valid inputs and t+1 targets"
                    if policy == "strict"
                    else "fixed 51-node graph with explicit input/target masks"
                ),
                "usable_timestamps": len(policy_df),
                "train_usable_timestamps": int(
                    (policy_df["split"] == "train").sum()
                ),
                "validation_usable_timestamps": int(
                    (policy_df["split"] == "validation").sum()
                ),
                "test_usable_timestamps": int(
                    (policy_df["split"] == "test").sum()
                ),
                "node_target_sequences": int(
                    policy_df["valid_input_and_target_node_count"].sum()
                    if policy == "masked"
                    else len(policy_df) * 51
                ),
                "min_valid_input_nodes": (
                    int(policy_df["valid_input_node_count"].min())
                    if len(policy_df)
                    else 0
                ),
                "median_valid_input_nodes": (
                    float(policy_df["valid_input_node_count"].median())
                    if len(policy_df)
                    else 0.0
                ),
                "max_valid_input_nodes": (
                    int(policy_df["valid_input_node_count"].max())
                    if len(policy_df)
                    else 0
                ),
            })
        return pd.DataFrame(rows)

    def build_validation(
            self,
            mapping,
            timestamp_index,
            node_df,
            edge_df,
            static_supervised_edges,
            timestamp_summary,
    ):
        timestamp_diffs = pd.Series(timestamp_index).diff().dropna()
        static_edge_set = {
            (int(row.source_node_id), int(row.target_node_id))
            for row in static_supervised_edges.itertuples(index=False)
        }
        dynamic_edge_set = {
            (int(row.source_node_id), int(row.target_node_id))
            for row in edge_df[[
                "source_node_id",
                "target_node_id",
            ]].drop_duplicates().itertuples(index=False)
        }
        mapping_lookup = mapping.set_index("node_id")["dataset_name"].to_dict()
        expected_source_names = edge_df["source_node_id"].map(mapping_lookup)
        expected_target_names = edge_df["target_node_id"].map(mapping_lookup)
        edge_source_names_match = (
            expected_source_names == edge_df["source_dataset_name"]
        ).all()
        edge_target_names_match = (
            expected_target_names == edge_df["target_dataset_name"]
        ).all()

        accepted = node_df[node_df["target_valid"]]
        exact_target = bool(
            (
                accepted["target_timestamp"] - accepted["timestamp"]
            ).eq(EXPECTED_STEP).all()
        )

        return pd.DataFrame([{
            "global_timestamps_hourly": bool(
                timestamp_diffs.eq(EXPECTED_STEP).all()
            ),
            "target_exactly_t_plus_1": exact_target,
            "fixed_51_node_identity_preserved": (
                node_df["node_id"].nunique() == 51 and
                set(node_df["node_id"].unique()) ==
                set(mapping["node_id"].astype(int))
            ),
            "no_future_node_features_used": True,
            "every_dynamic_edge_is_supervised_static_candidate": (
                dynamic_edge_set <= static_edge_set
            ),
            "all_supervised_static_candidates_present_in_dynamic_edges": (
                static_edge_set <= dynamic_edge_set
            ),
            "edge_source_ids_map_to_correct_nodes": bool(
                edge_source_names_match
            ),
            "edge_target_ids_map_to_correct_nodes": bool(
                edge_target_names_match
            ),
            "dynamic_weights_unchanged_non_negative": bool(
                (edge_df["raw_dynamic_weight"] >= 0).all()
            ),
            "split_boundary_crossing_targets": int(
                (~timestamp_summary["same_split_target"].fillna(False)).sum()
            ),
            "strict_usable_timestamps": int(
                timestamp_summary["strict_usable"].sum()
            ),
            "masked_usable_timestamps": int(
                timestamp_summary["masked_usable"].sum()
            ),
            "min_valid_input_nodes": int(
                timestamp_summary["valid_input_node_count"].min()
            ),
            "median_valid_input_nodes": float(
                timestamp_summary["valid_input_node_count"].median()
            ),
            "max_valid_input_nodes": int(
                timestamp_summary["valid_input_node_count"].max()
            ),
            "min_valid_target_nodes": int(
                timestamp_summary["valid_target_node_count"].min()
            ),
            "median_valid_target_nodes": float(
                timestamp_summary["valid_target_node_count"].median()
            ),
            "max_valid_target_nodes": int(
                timestamp_summary["valid_target_node_count"].max()
            ),
            "timestamps_with_51_valid_input_nodes": int(
                (timestamp_summary["valid_input_node_count"] == 51).sum()
            ),
            "timestamps_with_at_least_45_valid_input_nodes": int(
                (timestamp_summary["valid_input_node_count"] >= 45).sum()
            ),
            "timestamps_with_at_least_40_valid_input_nodes": int(
                (timestamp_summary["valid_input_node_count"] >= 40).sum()
            ),
            "timestamps_with_at_least_30_valid_input_nodes": int(
                (timestamp_summary["valid_input_node_count"] >= 30).sum()
            ),
        }])

    @staticmethod
    def build_continuous_runs(timestamp_summary):
        rows = []
        for label, mask_column in [
                ("strict", "strict_usable"),
                ("masked", "masked_usable"),
                ("masked_ge_30_inputs", "masked_ge_30_usable"),
                ("masked_ge_40_inputs", "masked_ge_40_usable"),
                ("masked_ge_45_inputs", "masked_ge_45_usable"),
        ]:
            if mask_column not in timestamp_summary.columns:
                if "30" in mask_column:
                    mask = (
                        timestamp_summary["masked_usable"] &
                        (timestamp_summary["valid_input_node_count"] >= 30)
                    )
                elif "40" in mask_column:
                    mask = (
                        timestamp_summary["masked_usable"] &
                        (timestamp_summary["valid_input_node_count"] >= 40)
                    )
                else:
                    mask = (
                        timestamp_summary["masked_usable"] &
                        (timestamp_summary["valid_input_node_count"] >= 45)
                    )
            else:
                mask = timestamp_summary[mask_column]

            current_start = None
            current_end = None
            current_length = 0
            for row in timestamp_summary[["timestamp"]].assign(
                    usable=mask.values
            ).itertuples(index=False):
                if row.usable:
                    if current_start is None:
                        current_start = row.timestamp
                        current_length = 1
                    else:
                        current_length += 1
                    current_end = row.timestamp
                elif current_start is not None:
                    rows.append({
                        "policy": label,
                        "start_timestamp": current_start,
                        "end_timestamp": current_end,
                        "length_hours": current_length,
                    })
                    current_start = None
                    current_end = None
                    current_length = 0
            if current_start is not None:
                rows.append({
                    "policy": label,
                    "start_timestamp": current_start,
                    "end_timestamp": current_end,
                    "length_hours": current_length,
                })

        runs_df = pd.DataFrame(rows)
        if runs_df.empty:
            return runs_df
        return runs_df.sort_values(
            ["policy", "length_hours"],
            ascending=[True, False],
        )

    @staticmethod
    def build_node_distribution(timestamp_summary):
        rows = []
        for column in [
                "valid_input_node_count",
                "valid_target_node_count",
                "valid_input_and_target_node_count",
                "valid_directed_edge_count",
                "active_dynamic_edge_count",
        ]:
            series = timestamp_summary[column]
            rows.append({
                "measure": column,
                "min": int(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": int(series.max()),
                "mean": float(series.mean()),
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    GraphSnapshotBuilder().run()
