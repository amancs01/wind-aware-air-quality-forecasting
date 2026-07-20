"""
Static Graph Construction

Builds the initial spatial graph using geographical distance
between monitoring stations.

Edges are created using K-Nearest Neighbors (KNN).

Inputs
------
data/processed/graph/distance_matrix.csv

Outputs
-------
data/processed/graph/static_graph.csv
data/processed/graph/adjacency_matrix.csv

Author:
    Nirika Lamichhane
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from logger import logger

from config import (
    DISTANCE_MATRIX_FILE,
    STATIC_GRAPH_FILE,
    ADJACENCY_MATRIX_FILE,
    K_NEIGHBORS,
)


class StaticGraphBuilder:

    def __init__(self):

        self.distance_file = DISTANCE_MATRIX_FILE

        self.graph_output = STATIC_GRAPH_FILE

        self.adjacency_output = ADJACENCY_MATRIX_FILE

    # -----------------------------------------------------
    # Load Distance Matrix
    # -----------------------------------------------------

    def load_distance_matrix(self):

        logger.info("Loading distance matrix...")

        df = pd.read_csv(
            self.distance_file,
            index_col=0,
        )

        # Convert both rows and columns to integers
        df.index = df.index.astype(int)
        df.columns = df.columns.astype(int)

        return df

    # -----------------------------------------------------
    # Build KNN Graph
    # -----------------------------------------------------

    def build_graph(self, distance_matrix):

        logger.info("Constructing static graph...")

        node_ids = distance_matrix.index.tolist()

        adjacency = pd.DataFrame(
            0,
            index=node_ids,
            columns=node_ids,
            dtype=int,
        )

        edges = []

        for source in node_ids:

            distances = distance_matrix.loc[source].copy()

            # remove self node
            distances = distances.drop(labels=source)

            # K nearest neighbours
            neighbors = distances.nsmallest(K_NEIGHBORS)

            for target, distance in neighbors.items():

                adjacency.loc[source, target] = 1
                adjacency.loc[target, source] = 1

                weight = round(
                    1 / (distance + 1e-6),
                    6,
                )

                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "distance_km": round(float(distance), 3),
                        "weight": weight,
                    }
                )

        edge_df = (
            pd.DataFrame(edges)
            .drop_duplicates(subset=["source", "target"])
            .reset_index(drop=True)
        )

        return adjacency, edge_df

    # -----------------------------------------------------
    # Save Outputs
    # -----------------------------------------------------

    def save_outputs(
        self,
        adjacency,
        edge_df,
    ):

        adjacency.to_csv(
            self.adjacency_output
        )

        edge_df.to_csv(
            self.graph_output,
            index=False,
        )

        logger.info(
            f"Adjacency matrix saved -> {self.adjacency_output}"
        )

        logger.info(
            f"Static graph saved -> {self.graph_output}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    def print_summary(
        self,
        adjacency,
        edge_df,
    ):

        logger.info("=" * 50)
        logger.info("Static Graph Summary")
        logger.info("=" * 50)

        logger.info(
            f"Nodes : {len(adjacency)}"
        )

        logger.info(
            f"Edges : {len(edge_df)}"
        )

        logger.info(
            f"K Neighbours : {K_NEIGHBORS}"
        )

        density = adjacency.values.sum() / (
            len(adjacency) * (len(adjacency) - 1)
        )

        logger.info(
            f"Density : {density:.4f}"
        )

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------

    def run(self):

        distance_matrix = self.load_distance_matrix()

        adjacency, edge_df = self.build_graph(
            distance_matrix
        )

        self.save_outputs(
            adjacency,
            edge_df,
        )

        self.print_summary(
            adjacency,
            edge_df,
        )


if __name__ == "__main__":

    StaticGraphBuilder().run()