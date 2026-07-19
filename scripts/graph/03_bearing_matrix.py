"""
Bearing Matrix Generator

Generates pairwise bearings (azimuths) between monitoring
stations for wind-aware graph construction.

Input:
    data/metadata/station_mapping.csv

Outputs:
    data/processed/graph/bearing_matrix.csv
    data/processed/graph/bearing_edges.csv

Author:
    Nirika Lamichhane
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import math
import pandas as pd

from logger import logger

from config import (
    STATION_MAPPING_FILE,
    BEARING_MATRIX_FILE,
    BEARING_EDGES_FILE,
)


class BearingMatrixGenerator:

    def __init__(self):

        self.input_file = STATION_MAPPING_FILE

        self.matrix_output = BEARING_MATRIX_FILE

        self.edges_output = BEARING_EDGES_FILE

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
    # Bearing Calculation
    # -------------------------------------------------------

    @staticmethod
    def calculate_bearing(
        lat1,
        lon1,
        lat2,
        lon2,
    ):
        """
        Returns initial bearing (degrees)
        from station A to station B.
        """

        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        dlon = math.radians(
            lon2 - lon1
        )

        x = math.sin(dlon) * math.cos(lat2)

        y = (
            math.cos(lat1) * math.sin(lat2)
            -
            math.sin(lat1)
            * math.cos(lat2)
            * math.cos(dlon)
        )

        bearing = math.degrees(
            math.atan2(x, y)
        )

        bearing = (bearing + 360) % 360

        return round(
            bearing,
            2
        )

    # -------------------------------------------------------
    # Generate Bearing Matrix
    # -------------------------------------------------------

    def generate_bearing_matrix(
        self,
        df,
    ):

        logger.info(
            "Generating bearing matrix..."
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

            for j in range(n):

                target = df.iloc[j]

                if i == j:

                    bearing = 0.0

                else:

                    bearing = self.calculate_bearing(

                        source["latitude"],
                        source["longitude"],

                        target["latitude"],
                        target["longitude"],

                    )

                matrix.loc[
                    source["node_id"],
                    target["node_id"],
                ] = bearing

                if i != j:

                    edge_list.append({

                        "source": source["node_id"],

                        "target": target["node_id"],

                        "bearing_deg": bearing,

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
            f"Bearing matrix saved to: {self.matrix_output}"
        )

        logger.info(
            f"Bearing edge table saved to: {self.edges_output}"
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
        logger.info("Bearing Matrix Summary")
        logger.info("=" * 50)

        logger.info(
            f"Stations     : {len(matrix)}"
        )

        logger.info(
            f"Matrix Shape : {matrix.shape}"
        )

        logger.info(
            f"Directed Edges : {len(edge_df)}"
        )

    # -------------------------------------------------------
    # Run Pipeline
    # -------------------------------------------------------

    def run(self):

        df = self.load_mapping()

        matrix, edge_df = self.generate_bearing_matrix(
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

    BearingMatrixGenerator().run()