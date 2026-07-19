"""
Static Graph Construction

Builds the initial spatial graph using geographical distance
between monitoring stations.

Edges are created using K-Nearest Neighbors (KNN).

Inputs
------
data/metadata/station_mapping.csv
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
import numpy as np

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

        return pd.read_csv(
            self.distance_file,
            index_col=0,
        )

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
        )

        edges = []

        for source in node_ids:

            distances = distance_matrix.loc[source].copy()

            # remove self
            distances = distances.drop(source)

            # nearest K stations
            neighbors = distances.nsmallest(K_NEIGHBORS)

            for target, distance in neighbors.items():

                adjacency.loc[source, target] = 1
                adjacency.loc[target, source] = 1

                weight = round(
                    1 / (distance + 1e-6),
                    6,
                )

                edges.append({

                    "source": source,

                    "target": target,

                    "distance_km": round(distance,3),

                    "weight": weight,

                })

        edge_df = (
            pd.DataFrame(edges)
            .drop_duplicates(
                subset=["source","target"]
            )
        )

        return adjacency, edge_df

    # -----------------------------------------------------
    # Save
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
            f"Saved adjacency matrix -> {self.adjacency_output}"
        )

        logger.info(
            f"Saved static graph -> {self.graph_output}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    def print_summary(
        self,
        adjacency,
        edge_df,
    ):

        logger.info("="*50)
        logger.info("Static Graph Summary")
        logger.info("="*50)

        logger.info(
            f"Nodes : {len(adjacency)}"
        )

        logger.info(
            f"Edges : {len(edge_df)}"
        )

        logger.info(
            f"K     : {K_NEIGHBORS}"
        )

        density = (
            adjacency.values.sum()
            / (len(adjacency)**2)
        )

        logger.info(
            f"Density : {density:.3f}"
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