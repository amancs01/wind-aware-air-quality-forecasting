"""
Classify weather stations by Open-Meteo grid cell and detect stations
with identical weather time series.

Place this file in the project's ``scripts`` directory and run:

    python scripts/01b_classify_weather_grids.py

Outputs are written to ``reports/weather_grids``.

The script supports two cases:
1. New weather files containing Open-Meteo's returned grid coordinates:
   ``weather_grid_latitude`` and ``weather_grid_longitude``.
2. Existing weather files without those columns. In that case, grid groups
   are inferred from stations whose weather values are exactly identical on
   their common timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, WEATHER_DIR


WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "dew_point",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
]

GRID_LATITUDE_COLUMN = "weather_grid_latitude"
GRID_LONGITUDE_COLUMN = "weather_grid_longitude"
GRID_ELEVATION_COLUMN = "weather_grid_elevation"

# Require at least one week of overlapping hourly data before declaring two
# stations identical. Increase this value for an even stricter comparison.
MIN_COMMON_ROWS = 24 * 7

# CSV values downloaded from the same Open-Meteo grid should normally be
# exactly equal. This tiny tolerance only protects against floating-point
# representation differences while reading CSV files.
ABSOLUTE_TOLERANCE = 1e-10

OUTPUT_DIR = PROJECT_ROOT / "reports" / "weather_grids"


@dataclass
class DisjointSet:
    """Small union-find structure used to build inferred station groups."""

    parent: dict[str, str]

    @classmethod
    def from_items(cls, items: Iterable[str]) -> "DisjointSet":
        return cls(parent={item: item for item in items})

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def load_weather_data() -> pd.DataFrame:
    """Load and combine every raw weather CSV in the project."""

    files = sorted(Path(WEATHER_DIR).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No weather CSV files were found under: {WEATHER_DIR}"
        )

    frames: list[pd.DataFrame] = []
    required_columns = {"timestamp", "station", *WEATHER_COLUMNS}

    for file_path in files:
        frame = pd.read_csv(file_path)
        missing = required_columns.difference(frame.columns)
        if missing:
            print(
                f"Skipping {file_path.name}: missing columns "
                f"{sorted(missing)}"
            )
            continue

        frame["source_file"] = str(file_path.relative_to(PROJECT_ROOT))
        frames.append(frame)

    if not frames:
        raise ValueError("No valid weather CSV files could be loaded.")

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", "station"])

    for column in WEATHER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # The downloader creates one record per station and timestamp. Keeping the
    # last occurrence prevents accidental duplicate files from inflating counts.
    data = (
        data.sort_values(["station", "timestamp", "source_file"])
        .drop_duplicates(subset=["station", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    return data


def values_match(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return one Boolean per row indicating whether all variables match."""

    equal_values = np.isclose(
        first,
        second,
        rtol=0.0,
        atol=ABSOLUTE_TOLERANCE,
        equal_nan=True,
    )
    return equal_values.all(axis=1)


def compare_station_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Compare every pair of stations over their common timestamps."""

    stations = sorted(data["station"].unique())
    station_frames = {
        station: (
            data.loc[data["station"] == station, ["timestamp", *WEATHER_COLUMNS]]
            .set_index("timestamp")
            .sort_index()
        )
        for station in stations
    }

    rows: list[dict[str, object]] = []

    for index, first_station in enumerate(stations):
        first = station_frames[first_station]

        for second_station in stations[index + 1 :]:
            second = station_frames[second_station]
            common_index = first.index.intersection(second.index)
            common_rows = len(common_index)

            if common_rows == 0:
                matching_rows = 0
                match_ratio = np.nan
                exactly_same = False
            else:
                first_values = first.loc[common_index, WEATHER_COLUMNS].to_numpy(
                    dtype=float
                )
                second_values = second.loc[common_index, WEATHER_COLUMNS].to_numpy(
                    dtype=float
                )
                row_matches = values_match(first_values, second_values)
                matching_rows = int(row_matches.sum())
                match_ratio = matching_rows / common_rows
                exactly_same = (
                    common_rows >= MIN_COMMON_ROWS and matching_rows == common_rows
                )

            rows.append(
                {
                    "station_1": first_station,
                    "station_2": second_station,
                    "common_rows": common_rows,
                    "matching_rows": matching_rows,
                    "match_ratio": match_ratio,
                    "exactly_same_weather": exactly_same,
                }
            )

    return pd.DataFrame(rows)


def actual_grid_classification(data: pd.DataFrame) -> pd.DataFrame | None:
    """Classify stations from Open-Meteo's returned grid coordinates."""

    required = {GRID_LATITUDE_COLUMN, GRID_LONGITUDE_COLUMN}
    if not required.issubset(data.columns):
        return None

    grid_data = data.dropna(subset=list(required)).copy()
    if grid_data.empty:
        return None

    # Open-Meteo coordinates are stable decimals. Rounding avoids grouping
    # failures caused only by CSV floating-point representation.
    grid_data[GRID_LATITUDE_COLUMN] = grid_data[GRID_LATITUDE_COLUMN].round(6)
    grid_data[GRID_LONGITUDE_COLUMN] = grid_data[GRID_LONGITUDE_COLUMN].round(6)

    station_rows: list[dict[str, object]] = []

    for station, group in grid_data.groupby("station", sort=True):
        cells = (
            group[[GRID_LATITUDE_COLUMN, GRID_LONGITUDE_COLUMN]]
            .drop_duplicates()
            .sort_values([GRID_LATITUDE_COLUMN, GRID_LONGITUDE_COLUMN])
        )

        cell_labels = [
            f"{row[GRID_LATITUDE_COLUMN]:.6f},{row[GRID_LONGITUDE_COLUMN]:.6f}"
            for _, row in cells.iterrows()
        ]

        requested_latitude = (
            group["latitude"].dropna().iloc[0]
            if "latitude" in group and group["latitude"].notna().any()
            else np.nan
        )
        requested_longitude = (
            group["longitude"].dropna().iloc[0]
            if "longitude" in group and group["longitude"].notna().any()
            else np.nan
        )
        elevation = (
            group[GRID_ELEVATION_COLUMN].dropna().iloc[0]
            if GRID_ELEVATION_COLUMN in group
            and group[GRID_ELEVATION_COLUMN].notna().any()
            else np.nan
        )

        station_rows.append(
            {
                "station": station,
                "requested_latitude": requested_latitude,
                "requested_longitude": requested_longitude,
                "grid_id": "GRID_" + "__".join(cell_labels),
                "grid_cells_used": len(cell_labels),
                "weather_grid_cells": " | ".join(cell_labels),
                "weather_grid_elevation": elevation,
                "classification_method": "open_meteo_returned_coordinates",
            }
        )

    classification = pd.DataFrame(station_rows)
    counts = classification.groupby("grid_id")["station"].transform("count")
    classification["stations_in_same_grid"] = counts
    classification["shares_grid_with_other_station"] = counts > 1

    return classification.sort_values(
        ["stations_in_same_grid", "grid_id", "station"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def inferred_grid_classification(
    data: pd.DataFrame,
    pair_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Infer grid groups from exactly identical weather time series."""

    stations = sorted(data["station"].unique())
    groups = DisjointSet.from_items(stations)

    identical_pairs = pair_comparison.loc[
        pair_comparison["exactly_same_weather"]
    ]
    for row in identical_pairs.itertuples(index=False):
        groups.union(row.station_1, row.station_2)

    components: dict[str, list[str]] = {}
    for station in stations:
        root = groups.find(station)
        components.setdefault(root, []).append(station)

    ordered_components = sorted(
        components.values(),
        key=lambda members: (-len(members), members[0]),
    )

    grid_id_by_station: dict[str, str] = {}
    for number, members in enumerate(ordered_components, start=1):
        grid_id = f"INFERRED_GRID_{number:02d}"
        for station in members:
            grid_id_by_station[station] = grid_id

    station_metadata = (
        data.groupby("station", as_index=False)
        .agg(
            requested_latitude=("latitude", "first")
            if "latitude" in data.columns
            else ("station", lambda _: np.nan),
            requested_longitude=("longitude", "first")
            if "longitude" in data.columns
            else ("station", lambda _: np.nan),
            weather_rows=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
    )

    station_metadata["grid_id"] = station_metadata["station"].map(
        grid_id_by_station
    )
    counts = station_metadata.groupby("grid_id")["station"].transform("count")
    station_metadata["stations_in_same_grid"] = counts
    station_metadata["shares_grid_with_other_station"] = counts > 1
    station_metadata["grid_cells_used"] = np.nan
    station_metadata["weather_grid_cells"] = "unknown"
    station_metadata["weather_grid_elevation"] = np.nan
    station_metadata["classification_method"] = (
        "inferred_from_identical_weather_series"
    )

    return station_metadata.sort_values(
        ["stations_in_same_grid", "grid_id", "station"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_grid_summary(classification: pd.DataFrame) -> pd.DataFrame:
    """Create one concise row for each actual or inferred grid group."""

    summary = (
        classification.groupby("grid_id", as_index=False)
        .agg(
            station_count=("station", "size"),
            stations=("station", lambda values: " | ".join(sorted(values))),
            classification_method=("classification_method", "first"),
            weather_grid_cells=("weather_grid_cells", "first"),
        )
        .sort_values(["station_count", "grid_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def print_summary(
    classification: pd.DataFrame,
    grid_summary: pd.DataFrame,
    pair_comparison: pd.DataFrame,
) -> None:
    station_count = classification["station"].nunique()
    grid_count = classification["grid_id"].nunique()
    shared_station_count = int(
        classification["shares_grid_with_other_station"].sum()
    )
    duplicate_pair_count = int(
        pair_comparison["exactly_same_weather"].sum()
    )

    print("\n" + "=" * 72)
    print("WEATHER GRID CLASSIFICATION")
    print("=" * 72)
    print(f"Stations analysed                  : {station_count}")
    print(f"Grid groups identified             : {grid_count}")
    print(f"Stations sharing a grid/group      : {shared_station_count}")
    print(f"Exactly identical station pairs    : {duplicate_pair_count}")
    print(
        "Classification method              : "
        f"{classification['classification_method'].iloc[0]}"
    )

    shared_groups = grid_summary.loc[grid_summary["station_count"] > 1]
    if shared_groups.empty:
        print("\nNo station group with shared weather/grid was found.")
    else:
        print("\nShared grid groups:")
        for row in shared_groups.itertuples(index=False):
            print(f"  {row.grid_id}: {row.station_count} stations")
            print(f"    {row.stations}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_weather_data()
    pair_comparison = compare_station_pairs(data)

    classification = actual_grid_classification(data)
    if classification is None:
        classification = inferred_grid_classification(data, pair_comparison)
        print(
            "Open-Meteo grid-coordinate columns were not found. "
            "Grid groups are therefore inferred from identical weather series."
        )
    else:
        print("Using Open-Meteo returned coordinates for grid classification.")

    grid_summary = build_grid_summary(classification)

    classification_path = OUTPUT_DIR / "station_grid_classification.csv"
    grid_summary_path = OUTPUT_DIR / "weather_grid_summary.csv"
    pair_comparison_path = OUTPUT_DIR / "duplicate_weather_pairs.csv"

    classification.to_csv(classification_path, index=False)
    grid_summary.to_csv(grid_summary_path, index=False)
    pair_comparison.sort_values(
        ["exactly_same_weather", "match_ratio", "common_rows"],
        ascending=[False, False, False],
    ).to_csv(pair_comparison_path, index=False)

    print_summary(classification, grid_summary, pair_comparison)
    print("\nSaved reports:")
    print(f"  {classification_path}")
    print(f"  {grid_summary_path}")
    print(f"  {pair_comparison_path}")


if __name__ == "__main__":
    main()
