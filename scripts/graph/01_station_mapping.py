"""
Station Mapping Generator

This script prepares the station metadata for graph construction.

Input:
    data/metadata/stations_metadata.csv

Outputs:
    data/metadata/station_mapping.csv
    data/metadata/station_mapping.json

Author:
    Nirika Lamichhane
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from logger import logger
from config import (
    STATIONS_FILE,
    METADATA_DIR,
    STATION_MAPPING_FILE
)


class StationMapper:
    """
    Generates a clean mapping between station names and graph node IDs.
    """

    REQUIRED_COLUMNS = [
        "station",
        "latitude",
        "longitude",
    ]

    def __init__(self):

        self.input_file = STATIONS_FILE

        self.output_csv = (
            METADATA_DIR /
            "station_mapping.csv"
        )

        self.output_json = (
            METADATA_DIR /
            "station_mapping.json"
        )

    
    # Load Metadata
    
    def load_metadata(self):

        logger.info(
            "Loading station metadata..."
        )

        return pd.read_csv(
            self.input_file
        )

    # Validate Metadata

    def validate(self, df):

        logger.info(
            "Validating metadata..."
        )

        missing_columns = [

            column

            for column in self.REQUIRED_COLUMNS

            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(

                f"Missing required columns: {missing_columns}"

            )

    # Clean Metadata

    def clean(self, df):

        logger.info(
            "Cleaning metadata..."
        )

        initial_count = len(df)

        # Remove duplicate stations
        df = df.drop_duplicates(
            subset="station"
        )

        # Remove rows without coordinates
        df = df.dropna(
            subset=[
                "latitude",
                "longitude",
            ]
        )

        # Sort alphabetically
        df = df.sort_values(
            by="station"
        ).reset_index(
            drop=True
        )

        removed = initial_count - len(df)

        logger.info(
            f"Removed {removed} invalid/duplicate stations."
        )

        return df

    # Assign Node IDs

    def assign_node_ids(self, df):

        logger.info(
            "Assigning node IDs..."
        )

        df.insert(
            0,
            "node_id",
            range(len(df))
        )

        return df

    # Save CSV

    def save_csv(self, df):

        df.to_csv(
            self.output_csv,
            index=False,
        )

        logger.info(
            f"Saved mapping CSV -> {self.output_csv}"
        )

    # Save JSON

    def save_json(self, df):

        station_mapping = dict(

            zip(

                df["station"],

                df["node_id"]

            )

        )

        with open(
            self.output_json,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(

                station_mapping,

                f,

                indent=4,

                ensure_ascii=False,

            )

        logger.info(
            f"Saved mapping JSON -> {self.output_json}"
        )

    # Summary

    def print_summary(self, df):

        logger.info("=" * 50)
        logger.info("Station Mapping Summary")
        logger.info("=" * 50)

        logger.info(
            f"Total Stations : {len(df)}"
        )

        logger.info(
            f"CSV Output      : {self.output_csv}"
        )

        logger.info(
            f"JSON Output     : {self.output_json}"
        )

    # Run Pipeline

    def run(self):

        df = self.load_metadata()

        self.validate(df)

        df = self.clean(df)

        df = self.assign_node_ids(df)

        self.save_csv(df)

        self.save_json(df)

        self.print_summary(df)


if __name__ == "__main__":

    mapper = StationMapper()

    mapper.run()