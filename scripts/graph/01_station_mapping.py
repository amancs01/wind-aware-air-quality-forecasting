"""
Canonical graph node registry generator.

Graph identity is dataset-level, not human-station-level. Duplicate human
station names are preserved by using the same canonical dataset naming
rule as the processed featured datasets.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd

from config import (
    FEATURED_DIR,
    METADATA_DIR,
    ML_VALIDATION_DIR,
    STATIONS_FILE,
    TRAIN_DIR,
)
from logger import logger
from utils import station_dataset_name


class StationMapper:
    """
    Generates a canonical sensor-qualified graph node registry.
    """

    REQUIRED_COLUMNS = [
        "station",
        "location_id",
        "pm25_sensor_id",
        "latitude",
        "longitude",
    ]

    def __init__(self):
        self.input_file = STATIONS_FILE
        self.output_csv = METADATA_DIR / "station_mapping.csv"
        self.output_json = METADATA_DIR / "station_mapping.json"

    def load_metadata(self):
        logger.info("Loading station metadata...")
        return pd.read_csv(self.input_file)

    def validate(self, df):
        logger.info("Validating metadata...")
        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df["pm25_sensor_id"].isna().any():
            raise ValueError("station metadata contains missing PM2.5 sensors")

        if df[["latitude", "longitude"]].isna().any(axis=1).any():
            raise ValueError("station metadata contains missing coordinates")

    def build_mapping(self, df):
        logger.info("Building canonical graph node registry...")

        duplicated_station_names = set(
            df.loc[
                df["station"].duplicated(keep=False),
                "station",
            ]
        )
        featured_names = {path.stem for path in FEATURED_DIR.glob("*.csv")}
        train_names = {path.stem for path in TRAIN_DIR.glob("*.csv")}
        validation_names = {
            path.stem
            for path in ML_VALIDATION_DIR.glob("*.csv")
        }

        rows = []
        for _, row in df.iterrows():
            dataset_name = station_dataset_name(
                row["station"],
                sensor_id=row["pm25_sensor_id"],
                require_sensor_id=row["station"] in duplicated_station_names,
            )
            rows.append({
                "dataset_name": dataset_name,
                "station": row["station"],
                "location_id": int(row["location_id"]),
                "pm25_sensor_id": int(row["pm25_sensor_id"]),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "in_featured": dataset_name in featured_names,
                "in_train": dataset_name in train_names,
                "in_validation": dataset_name in validation_names,
                "model_usable": (
                    dataset_name in train_names and
                    dataset_name in validation_names
                ),
            })

        mapping = (
            pd.DataFrame(rows)
            .sort_values(["dataset_name", "pm25_sensor_id"])
            .reset_index(drop=True)
        )
        mapping.insert(0, "node_id", range(len(mapping)))

        if not mapping["dataset_name"].is_unique:
            raise ValueError("dataset_name is not unique in graph mapping")

        if not mapping["pm25_sensor_id"].is_unique:
            raise ValueError("pm25_sensor_id is not unique in graph mapping")

        missing_featured = mapping.loc[~mapping["in_featured"], "dataset_name"]
        if not missing_featured.empty:
            raise ValueError(
                "metadata datasets missing from featured outputs: "
                f"{missing_featured.to_list()}"
            )

        return mapping

    def save_csv(self, df):
        df.to_csv(self.output_csv, index=False)
        logger.info(f"Saved mapping CSV -> {self.output_csv}")

    def save_json(self, df):
        mapping = {
            row["dataset_name"]: {
                "node_id": int(row["node_id"]),
                "pm25_sensor_id": int(row["pm25_sensor_id"]),
                "station": row["station"],
                "model_usable": bool(row["model_usable"]),
            }
            for _, row in df.iterrows()
        }

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved mapping JSON -> {self.output_json}")

    def print_summary(self, df):
        logger.info("=" * 50)
        logger.info("Graph Node Registry Summary")
        logger.info("=" * 50)
        logger.info(f"Canonical nodes : {len(df)}")
        logger.info(f"Featured nodes  : {int(df['in_featured'].sum())}")
        logger.info(f"Model usable    : {int(df['model_usable'].sum())}")
        logger.info(f"CSV Output      : {self.output_csv}")
        logger.info(f"JSON Output     : {self.output_json}")

    def run(self):
        df = self.load_metadata()
        self.validate(df)
        mapping = self.build_mapping(df)
        self.save_csv(mapping)
        self.save_json(mapping)
        self.print_summary(mapping)


if __name__ == "__main__":
    StationMapper().run()
