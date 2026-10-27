"""
Station Mapping Generator

Generates graph node mapping from the trimmed station datasets.

Input:
    data/processed/trimmed/*.csv

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
    TRIMMED_DIR,
    METADATA_DIR,
    STATION_MAPPING_FILE,
)


class StationMapper:

    REQUIRED_COLUMNS = [
        "station",
        "latitude",
        "longitude",
    ]

    def __init__(self):

        self.input_dir = TRIMMED_DIR

        self.output_csv = STATION_MAPPING_FILE

        self.output_json = (
            METADATA_DIR /
            "station_mapping.json"
        )

    # -------------------------------------------------------
    # Load Metadata
    # -------------------------------------------------------

    def load_metadata(self):

        logger.info(
            "Loading trimmed station files..."
        )

        csv_files = sorted(
            self.input_dir.glob("*.csv")
        )

        if len(csv_files) == 0:

            raise FileNotFoundError(
                f"No station CSV files found in {self.input_dir}"
            )

        station_records = []

        for csv_file in csv_files:

            df = pd.read_csv(
                csv_file,
                nrows=1,
            )

            missing = [
                col
                for col in self.REQUIRED_COLUMNS
                if col not in df.columns
            ]

            if missing:

                raise ValueError(
                    f"{csv_file.name} is missing columns: {missing}"
                )

            station_records.append({

                "station": df.loc[0, "station"],

                "latitude": df.loc[0, "latitude"],

                "longitude": df.loc[0, "longitude"],

                "file_name": csv_file.name,

            })

        return pd.DataFrame(
            station_records
        )

    # -------------------------------------------------------
    # Clean Metadata
    # -------------------------------------------------------

    def clean(self, df):

        logger.info(
            "Cleaning metadata..."
        )

        initial_count = len(df)

        df = df.drop_duplicates(
            subset="station"
        )

        df = df.dropna(
            subset=[
                "latitude",
                "longitude",
            ]
        )

        df = df.sort_values(
            by="station"
        ).reset_index(
            drop=True
        )

        removed = initial_count - len(df)

        logger.info(
            f"Removed {removed} duplicate/invalid stations."
        )

        return df

    # -------------------------------------------------------
    # Assign Node IDs
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------

    def save_csv(self, df):

        df.to_csv(
            self.output_csv,
            index=False,
        )

        logger.info(
            f"Saved mapping CSV -> {self.output_csv}"
        )

    # -------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------

    def save_json(self, df):

        mapping = dict(

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

                mapping,

                f,

                indent=4,

                ensure_ascii=False,

            )

        logger.info(
            f"Saved mapping JSON -> {self.output_json}"
        )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    def print_summary(self, df):

        logger.info("=" * 50)
        logger.info("Station Mapping Summary")
        logger.info("=" * 50)

        logger.info(
            f"Total Stations : {len(df)}"
        )

        logger.info(
            f"CSV Output : {self.output_csv}"
        )

        logger.info(
            f"JSON Output : {self.output_json}"
        )

    # -------------------------------------------------------
    # Run Pipeline
    # -------------------------------------------------------

    def run(self):

        df = self.load_metadata()

        df = self.clean(df)

        df = self.assign_node_ids(df)

        self.save_csv(df)

        self.save_json(df)

        self.print_summary(df)


if __name__ == "__main__":

    StationMapper().run()