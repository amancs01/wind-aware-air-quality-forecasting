"""
Static directed candidate graph construction.

Builds a KNN graph, forms the symmetric union of nearest-neighbor pairs,
then expands every undirected pair into both directed candidates. The
adjacency matrix and static edge CSV therefore represent exactly the same
directed edge set.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    ADJACENCY_MATRIX_FILE,
    BEARING_MATRIX_FILE,
    DISTANCE_MATRIX_FILE,
    K_NEIGHBORS,
    STATIC_GRAPH_FILE,
    STATION_MAPPING_FILE,
)
from logger import logger


class StaticGraphBuilder:

    def __init__(self):
        self.mapping_file = STATION_MAPPING_FILE
        self.distance_file = DISTANCE_MATRIX_FILE
        self.bearing_file = BEARING_MATRIX_FILE
        self.graph_output = STATIC_GRAPH_FILE
        self.adjacency_output = ADJACENCY_MATRIX_FILE

    def load_mapping(self):
        return pd.read_csv(self.mapping_file)

    def load_matrix(self, path):
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(int)
        df.columns = df.columns.astype(int)
        return df

    def build_graph(self, mapping, distance_matrix, bearing_matrix):
        logger.info("Constructing directed static candidate graph...")

        node_ids = distance_matrix.index.tolist()
        node_meta = mapping.set_index("node_id").to_dict(orient="index")
        candidate_pairs = set()

        for source in node_ids:
            distances = distance_matrix.loc[source].drop(labels=source)
            neighbors = distances.nsmallest(K_NEIGHBORS)
            for target in neighbors.index:
                candidate_pairs.add(tuple(sorted((int(source), int(target)))))

        directed_edges = sorted(
            (source, target)
            for node_a, node_b in candidate_pairs
            for source, target in [(node_a, node_b), (node_b, node_a)]
        )

        adjacency = pd.DataFrame(
            0,
            index=node_ids,
            columns=node_ids,
            dtype=int,
        )
        rows = []

        for source, target in directed_edges:
            source_meta = node_meta[source]
            target_meta = node_meta[target]
            distance = float(distance_matrix.loc[source, target])
            bearing = float(bearing_matrix.loc[source, target])
            adjacency.loc[source, target] = 1
            rows.append({
                "source": source,
                "target": target,
                "source_node_id": source,
                "target_node_id": target,
                "source_dataset_name": source_meta["dataset_name"],
                "target_dataset_name": target_meta["dataset_name"],
                "source_pm25_sensor_id": source_meta["pm25_sensor_id"],
                "target_pm25_sensor_id": target_meta["pm25_sensor_id"],
                "source_station": source_meta["station"],
                "target_station": target_meta["station"],
                "distance_km": round(distance, 3),
                "bearing_deg": round(bearing, 2),
                "static_distance_weight": round(1 / (distance + 1e-6), 6),
                "candidate_edge": 1,
            })

        edge_df = pd.DataFrame(rows)
        return adjacency, edge_df

    def validate_outputs(self, adjacency, edge_df):
        adjacency_edges = {
            (int(source), int(target))
            for source in adjacency.index
            for target in adjacency.columns
            if source != target and int(adjacency.loc[source, target]) == 1
        }
        csv_edges = {
            (int(row["source"]), int(row["target"]))
            for _, row in edge_df.iterrows()
        }

        if adjacency_edges != csv_edges:
            raise ValueError(
                "static graph edge CSV does not match adjacency edge set"
            )

        missing_reverse = [
            (source, target)
            for source, target in csv_edges
            if (target, source) not in csv_edges
        ]
        if missing_reverse:
            raise ValueError(
                "directed static graph is missing reverse candidates: "
                f"{missing_reverse[:5]}"
            )

    def save_outputs(self, adjacency, edge_df):
        adjacency.to_csv(self.adjacency_output)
        edge_df.to_csv(self.graph_output, index=False)
        logger.info(f"Adjacency matrix saved -> {self.adjacency_output}")
        logger.info(f"Static graph saved -> {self.graph_output}")

    def print_summary(self, adjacency, edge_df):
        logger.info("=" * 50)
        logger.info("Static Graph Summary")
        logger.info("=" * 50)
        logger.info(f"Nodes : {len(adjacency)}")
        logger.info(f"Directed Edges : {len(edge_df)}")
        logger.info(f"Undirected Pairs : {len(edge_df) // 2}")
        logger.info(f"K Neighbours : {K_NEIGHBORS}")
        density = adjacency.values.sum() / (
            len(adjacency) * (len(adjacency) - 1)
        )
        logger.info(f"Density : {density:.4f}")

    def run(self):
        mapping = self.load_mapping()
        distance_matrix = self.load_matrix(self.distance_file)
        bearing_matrix = self.load_matrix(self.bearing_file)
        adjacency, edge_df = self.build_graph(
            mapping,
            distance_matrix,
            bearing_matrix,
        )
        self.validate_outputs(adjacency, edge_df)
        self.save_outputs(adjacency, edge_df)
        self.print_summary(adjacency, edge_df)


if __name__ == "__main__":
    StaticGraphBuilder().run()
