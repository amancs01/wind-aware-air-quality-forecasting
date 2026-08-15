"""
Dynamic wind-edge weight generator.

Uses the corrected directed static candidate graph and source-node wind
from featured hourly datasets. This stage does not row-normalize weights,
build graph snapshots, or train a graph model.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    DYNAMIC_EDGE_FILE,
    FEATURED_DIR,
    GRAPH_DIR,
    STATIC_GRAPH_FILE,
    STATION_MAPPING_FILE,
)
from logger import logger


CALM_WIND_THRESHOLD_KMH = 0.5
WIND_SPEED_SCALE_KMH = 5.0
SUMMARY_FILE = GRAPH_DIR / "dynamic_edge_weights_summary.csv"
VALIDATION_FILE = GRAPH_DIR / "dynamic_edge_weights_validation.csv"
SUPERVISED_DEGREE_FILE = GRAPH_DIR / "dynamic_supervised_degree.csv"
REVERSE_CHECK_FILE = GRAPH_DIR / "dynamic_reverse_direction_check.csv"


class DynamicWindEdgeGenerator:

    REQUIRED_STATIC_COLUMNS = [
        "source_node_id",
        "target_node_id",
        "source_dataset_name",
        "target_dataset_name",
        "distance_km",
        "bearing_deg",
    ]

    REQUIRED_MAPPING_COLUMNS = [
        "node_id",
        "dataset_name",
        "model_usable",
    ]

    REQUIRED_WIND_COLUMNS = [
        "timestamp",
        "wind_speed",
        "wind_direction",
    ]

    def __init__(self):
        self.static_file = STATIC_GRAPH_FILE
        self.mapping_file = STATION_MAPPING_FILE
        self.output_file = DYNAMIC_EDGE_FILE
        self.lambda_d = None
        self.summary = {}

    def run(self):
        static_edges = self.load_static_edges()
        mapping = self.load_mapping()
        static_edges = self.add_model_usable_flags(static_edges, mapping)
        self.lambda_d = float(static_edges["distance_km"].median())

        self.validate_static_edges(static_edges)
        self.write_dynamic_edges(static_edges)
        validation_df = self.validate_dynamic_output(static_edges)
        degree_df = self.build_supervised_degree(static_edges)
        reverse_df = self.build_reverse_direction_check()
        summary_df = self.build_summary(validation_df, degree_df, reverse_df)

        validation_df.to_csv(VALIDATION_FILE, index=False)
        degree_df.to_csv(SUPERVISED_DEGREE_FILE, index=False)
        reverse_df.to_csv(REVERSE_CHECK_FILE, index=False)
        summary_df.to_csv(SUMMARY_FILE, index=False)

        logger.info("=" * 50)
        logger.info("Dynamic Wind Edge Summary")
        logger.info("=" * 50)
        logger.info(f"Lambda distance: {self.lambda_d:.3f} km")
        logger.info(f"Timestamps: {summary_df['timestamp_count'].iloc[0]}")
        logger.info(f"Rows: {summary_df['total_rows'].iloc[0]}")
        logger.info(
            "Active edges: "
            f"{summary_df['active_edge_percentage'].iloc[0]:.2f}%"
        )

    def load_static_edges(self):
        static_edges = pd.read_csv(self.static_file)
        missing_columns = [
            column
            for column in self.REQUIRED_STATIC_COLUMNS
            if column not in static_edges.columns
        ]
        if missing_columns:
            raise ValueError(
                f"static graph missing required columns: {missing_columns}"
            )

        return static_edges

    def load_mapping(self):
        mapping = pd.read_csv(self.mapping_file)
        missing_columns = [
            column
            for column in self.REQUIRED_MAPPING_COLUMNS
            if column not in mapping.columns
        ]
        if missing_columns:
            raise ValueError(
                f"station mapping missing required columns: {missing_columns}"
            )

        return mapping

    @staticmethod
    def add_model_usable_flags(static_edges, mapping):
        model_usable = mapping.set_index("node_id")["model_usable"].to_dict()
        static_edges = static_edges.copy()
        static_edges["source_model_usable"] = (
            static_edges["source_node_id"]
            .map(model_usable)
            .astype(bool)
        )
        static_edges["target_model_usable"] = (
            static_edges["target_node_id"]
            .map(model_usable)
            .astype(bool)
        )
        static_edges["supervised_edge"] = (
            static_edges["source_model_usable"] &
            static_edges["target_model_usable"]
        )

        return static_edges

    @staticmethod
    def validate_static_edges(static_edges):
        edge_set = {
            (int(row.source_node_id), int(row.target_node_id))
            for row in static_edges.itertuples()
        }
        missing_reverse = [
            edge
            for edge in edge_set
            if (edge[1], edge[0]) not in edge_set
        ]
        if missing_reverse:
            raise ValueError(
                "static candidate graph is missing reverse directions"
            )

    def write_dynamic_edges(self, static_edges):
        if self.output_file.exists():
            self.output_file.unlink()

        header = True
        for source_node_id, source_edges in static_edges.groupby(
                "source_node_id",
                sort=True,
        ):
            dataset_name = source_edges["source_dataset_name"].iloc[0]
            wind_df = self.load_source_wind(dataset_name)
            dynamic_df = self.build_source_dynamic_edges(
                source_edges,
                wind_df,
            )
            dynamic_df.to_csv(
                self.output_file,
                mode="w" if header else "a",
                header=header,
                index=False,
            )
            header = False
            logger.info(
                f"Source node {source_node_id}: wrote {len(dynamic_df)} rows"
            )

    def load_source_wind(self, dataset_name):
        wind_file = FEATURED_DIR / f"{dataset_name}.csv"
        if not wind_file.exists():
            raise FileNotFoundError(
                f"featured wind file missing for {dataset_name}"
            )

        wind_df = pd.read_csv(
            wind_file,
            usecols=self.REQUIRED_WIND_COLUMNS,
        )
        wind_df["timestamp"] = pd.to_datetime(wind_df["timestamp"])
        return wind_df

    def build_source_dynamic_edges(self, source_edges, wind_df):
        edges = source_edges[[
            "source_node_id",
            "target_node_id",
            "source_dataset_name",
            "target_dataset_name",
            "source_model_usable",
            "target_model_usable",
            "supervised_edge",
            "distance_km",
            "bearing_deg",
        ]].copy()
        edges["_join_key"] = 1
        wind = wind_df.rename(columns={
            "wind_speed": "source_wind_speed",
            "wind_direction": "source_wind_direction_from_deg",
        }).copy()
        wind["_join_key"] = 1

        dynamic = wind.merge(edges, on="_join_key", how="inner")
        dynamic.drop(columns=["_join_key"], inplace=True)

        dynamic["missing_source_wind"] = (
            dynamic[[
                "source_wind_speed",
                "source_wind_direction_from_deg",
            ]]
            .isna()
            .any(axis=1)
        )
        dynamic["calm_wind"] = (
            (~dynamic["missing_source_wind"]) &
            (dynamic["source_wind_speed"] < CALM_WIND_THRESHOLD_KMH)
        )

        dynamic["source_transport_direction_deg"] = (
            dynamic["source_wind_direction_from_deg"] + 180
        ) % 360
        dynamic.loc[
            dynamic["missing_source_wind"],
            "source_transport_direction_deg",
        ] = np.nan

        raw_diff = (
            dynamic["source_transport_direction_deg"] -
            dynamic["bearing_deg"]
        ).abs()
        dynamic["alignment_angle_deg"] = np.minimum(raw_diff, 360 - raw_diff)
        dynamic.loc[
            dynamic["missing_source_wind"],
            "alignment_angle_deg",
        ] = np.nan

        dynamic["alignment"] = np.maximum(
            0.0,
            np.cos(np.radians(dynamic["alignment_angle_deg"])),
        )
        dynamic["alignment"] = dynamic["alignment"].fillna(0.0)
        dynamic.loc[
            dynamic["alignment_angle_deg"] >= 90,
            "alignment",
        ] = 0.0
        dynamic["speed_factor"] = (
            dynamic["source_wind_speed"] /
            (dynamic["source_wind_speed"] + WIND_SPEED_SCALE_KMH)
        )
        dynamic["speed_factor"] = dynamic["speed_factor"].fillna(0.0)
        dynamic["distance_factor"] = np.exp(
            -dynamic["distance_km"] / self.lambda_d
        )
        dynamic["raw_dynamic_weight"] = (
            dynamic["alignment"] *
            dynamic["speed_factor"] *
            dynamic["distance_factor"]
        )
        dynamic.loc[
            dynamic["missing_source_wind"] | dynamic["calm_wind"],
            "raw_dynamic_weight",
        ] = 0.0
        dynamic["edge_active"] = dynamic["raw_dynamic_weight"] > 0

        columns = [
            "timestamp",
            "source_node_id",
            "target_node_id",
            "source_dataset_name",
            "target_dataset_name",
            "source_model_usable",
            "target_model_usable",
            "supervised_edge",
            "distance_km",
            "bearing_deg",
            "source_wind_speed",
            "source_wind_direction_from_deg",
            "source_transport_direction_deg",
            "alignment_angle_deg",
            "alignment",
            "speed_factor",
            "distance_factor",
            "raw_dynamic_weight",
            "calm_wind",
            "missing_source_wind",
            "edge_active",
        ]

        return dynamic[columns]

    def validate_dynamic_output(self, static_edges):
        dynamic = pd.read_csv(
            self.output_file,
            parse_dates=["timestamp"],
        )
        static_edge_set = {
            (int(row.source_node_id), int(row.target_node_id))
            for row in static_edges.itertuples()
        }
        dynamic_edge_set = {
            (int(row.source_node_id), int(row.target_node_id))
            for row in dynamic[[
                "source_node_id",
                "target_node_id",
            ]].drop_duplicates().itertuples(index=False)
        }
        missing_reverse = sum(
            (target, source) not in dynamic_edge_set
            for source, target in dynamic_edge_set
        )
        away_or_perpendicular = (
            dynamic["alignment_angle_deg"].fillna(0) >= 90
        )

        rows = [{
            "static_candidate_edges": len(static_edge_set),
            "dynamic_candidate_edges": len(dynamic_edge_set),
            "all_rows_are_static_candidates": (
                dynamic_edge_set <= static_edge_set
            ),
            "no_non_candidate_edges": (
                len(dynamic_edge_set - static_edge_set) == 0
            ),
            "all_static_candidates_present": (
                static_edge_set <= dynamic_edge_set
            ),
            "weights_never_negative": bool(
                (dynamic["raw_dynamic_weight"] >= 0).all()
            ),
            "alignment_in_unit_interval": bool(
                dynamic["alignment"].between(0, 1).all()
            ),
            "speed_factor_in_unit_interval": bool(
                ((dynamic["speed_factor"] >= 0) &
                 (dynamic["speed_factor"] < 1)).all()
            ),
            "distance_factor_in_unit_interval": bool(
                ((dynamic["distance_factor"] > 0) &
                 (dynamic["distance_factor"] <= 1)).all()
            ),
            "calm_wind_zero_weight": bool(
                (dynamic.loc[
                    dynamic["calm_wind"],
                    "raw_dynamic_weight",
                ] == 0).all()
            ),
            "away_or_perpendicular_zero_weight": bool(
                (dynamic.loc[
                    away_or_perpendicular,
                    "raw_dynamic_weight",
                ] == 0).all()
            ),
            "missing_source_wind_zero_weight": bool(
                (dynamic.loc[
                    dynamic["missing_source_wind"],
                    "raw_dynamic_weight",
                ] == 0).all()
            ),
            "candidate_pairs_missing_reverse_direction": int(missing_reverse),
            "timestamp_min": dynamic["timestamp"].min(),
            "timestamp_max": dynamic["timestamp"].max(),
            "uses_future_rows": False,
        }]

        self.dynamic_df = dynamic
        return pd.DataFrame(rows)

    def build_supervised_degree(self, static_edges):
        supervised = static_edges[static_edges["supervised_edge"]].copy()
        node_ids = sorted(
            set(supervised["source_node_id"]) |
            set(supervised["target_node_id"])
        )
        rows = []
        for node_id in node_ids:
            rows.append({
                "node_id": node_id,
                "dataset_name": supervised.loc[
                    supervised["source_node_id"] == node_id,
                    "source_dataset_name",
                ].head(1).to_list()[0],
                "out_degree": int(
                    (supervised["source_node_id"] == node_id).sum()
                ),
                "in_degree": int(
                    (supervised["target_node_id"] == node_id).sum()
                ),
            })

        degree_df = pd.DataFrame(rows)
        degree_df["total_degree"] = (
            degree_df["in_degree"] + degree_df["out_degree"]
        )

        return degree_df

    def build_reverse_direction_check(self):
        dynamic = self.dynamic_df
        sample_edges = dynamic[[
            "source_node_id",
            "target_node_id",
            "timestamp",
            "raw_dynamic_weight",
        ]].copy()
        reverse = sample_edges.rename(columns={
            "source_node_id": "target_node_id",
            "target_node_id": "source_node_id",
            "raw_dynamic_weight": "reverse_raw_dynamic_weight",
        })
        pairs = sample_edges.merge(
            reverse,
            on=["source_node_id", "target_node_id", "timestamp"],
            how="inner",
        )

        return pd.DataFrame([{
            "checked_rows": len(pairs),
            "opposite_direction_weights_can_differ": bool(
                (pairs["raw_dynamic_weight"] !=
                 pairs["reverse_raw_dynamic_weight"]).any()
            ),
            "different_weight_rows": int(
                (pairs["raw_dynamic_weight"] !=
                 pairs["reverse_raw_dynamic_weight"]).sum()
            ),
        }])

    def build_summary(self, validation_df, degree_df, reverse_df):
        dynamic = self.dynamic_df
        total_rows = len(dynamic)
        supervised_edges = dynamic[[
            "source_node_id",
            "target_node_id",
            "supervised_edge",
        ]].drop_duplicates()

        return pd.DataFrame([{
            "lambda_d_km": self.lambda_d,
            "timestamp_count": dynamic["timestamp"].nunique(),
            "total_rows": total_rows,
            "candidate_edges": dynamic[[
                "source_node_id",
                "target_node_id",
            ]].drop_duplicates().shape[0],
            "supervised_candidate_edges": int(
                supervised_edges["supervised_edge"].sum()
            ),
            "active_edge_percentage": (
                dynamic["edge_active"].mean() * 100
                if total_rows
                else 0.0
            ),
            "zero_weight_percentage": (
                (dynamic["raw_dynamic_weight"] == 0).mean() * 100
                if total_rows
                else 0.0
            ),
            "missing_wind_percentage": (
                dynamic["missing_source_wind"].mean() * 100
                if total_rows
                else 0.0
            ),
            "calm_wind_percentage": (
                dynamic["calm_wind"].mean() * 100
                if total_rows
                else 0.0
            ),
            "supervised_nodes": len(degree_df),
            "supervised_min_out_degree": int(degree_df["out_degree"].min()),
            "supervised_median_out_degree": float(
                degree_df["out_degree"].median()
            ),
            "supervised_max_out_degree": int(degree_df["out_degree"].max()),
            "supervised_isolated_nodes": int(
                (degree_df["total_degree"] == 0).sum()
            ),
            "opposite_direction_weights_can_differ": reverse_df[
                "opposite_direction_weights_can_differ"
            ].iloc[0],
            "all_validation_checks_passed": bool(
                validation_df[[
                    "all_rows_are_static_candidates",
                    "no_non_candidate_edges",
                    "all_static_candidates_present",
                    "weights_never_negative",
                    "alignment_in_unit_interval",
                    "speed_factor_in_unit_interval",
                    "distance_factor_in_unit_interval",
                    "calm_wind_zero_weight",
                    "away_or_perpendicular_zero_weight",
                    "missing_source_wind_zero_weight",
                ]].iloc[0].all()
            ),
        }])


if __name__ == "__main__":
    DynamicWindEdgeGenerator().run()
