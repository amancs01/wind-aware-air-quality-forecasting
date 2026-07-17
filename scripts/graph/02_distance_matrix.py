"""
Distance Matrix Generator

Generates pairwise Haversine distances between monitoring
stations for graph construction.

Input:
    data/metadata/station_mapping.csv

Outputs:
    data/processed/graph/distance_matrix.csv
    data/processed/graph/distance_edges.csv

Author:
    Nirika Lamichhane
"""

import math
import pandas as pd

from logger import logger

from config import (
    STATION_MAPPING_FILE,
    DISTANCE_MATRIX_FILE,
    DISTANCE_EDGES_FILE,
)


class DistanceMatrixGenerator:

    EARTH_RADIUS_KM = 6371.0

    def __init__(self):

        self.input_file = STATION_MAPPING_FILE

        self.matrix_output = DISTANCE_MATRIX_FILE

        self.edges_output = DISTANCE_EDGES_FILE

    # -------------------------------------------------------
    # Load Station Mapping
    # -------------------------------------------------------

    def load_mapping(self):

        logger.info(
            "Loading station mapping..."
        )

        return pd.read_csv(
            self.input_file
        )

    # -------------------------------------------------------
    # Haversine Distance
    # -------------------------------------------------------

    @staticmethod
    def haversine(
        lat1,
        lon1,
        lat2,
        lon2,
    ):

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)

        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )

        return self.EARTH_RADIUS_KM * c

    # -------------------------------------------------------
    # Generate Distance Matrix
    # -------------------------------------------------------

    def generate_distance_matrix(
        self,
        df,
    ):

        logger.info(
            "Generating distance matrix..."
        )

        node_ids = df["node_id"].tolist()

        matrix = pd.DataFrame(
            0.0,
            index=node_ids,
            columns=node_ids,
        )

        edge_list = []

        n = len(df)

        for i in range(n):

            source = df.iloc[i]

            for j in range(i, n):

                target = df.iloc[j]

                distance = self.haversine(

                    source["latitude"],
                    source["longitude"],

                    target["latitude"],
                    target["longitude"],

                )

                distance = round(
                    distance,
                    3,
                )

                # Fill symmetric matrix
                matrix.loc[
                    source["node_id"],
                    target["node_id"],
                ] = distance

                matrix.loc[
                    target["node_id"],
                    source["node_id"],
                ] = distance

                # Skip self-loop in edge table
                if i != j:

                    edge_list.append({

                        "source": source["node_id"],

                        "target": target["node_id"],

                        "distance_km": distance,

                    })

        edge_df = pd.DataFrame(
            edge_list
        )

        return matrix, edge_df

    # -------------------------------------------------------
    # Save Outputs
    # -------------------------------------------------------

    def save_outputs(
        self,
        matrix,
        edge_df,
    ):

        matrix.to_csv(
            self.matrix_output,
            index=True,
        )

        edge_df.to_csv(
            self.edges_output,
            index=False,
        )

        logger.info(
            f"Distance matrix saved to: {self.matrix_output}"
        )

        logger.info(
            f"Distance edge table saved to: {self.edges_output}"
        )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    def print_summary(
        self,
        matrix,
        edge_df,
    ):

        logger.info("=" * 50)
        logger.info("Distance Matrix Summary")
        logger.info("=" * 50)

        logger.info(
            f"Stations      : {len(matrix)}"
        )

        logger.info(
            f"Matrix Shape  : {matrix.shape}"
        )

        logger.info(
            f"Unique Edges  : {len(edge_df)}"
        )

    # -------------------------------------------------------
    # Run Pipeline
    # -------------------------------------------------------

    def run(self):

        df = self.load_mapping()

        matrix, edge_df = self.generate_distance_matrix(
            df
        )

        self.save_outputs(
            matrix,
            edge_df,
        )

        self.print_summary(
            matrix,
            edge_df,
        )


if __name__ == "__main__":

    DistanceMatrixGenerator().run()