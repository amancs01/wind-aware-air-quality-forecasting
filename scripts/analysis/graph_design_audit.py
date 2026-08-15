import math

import pandas as pd

from config import (
    ADJACENCY_MATRIX_FILE,
    BEARING_EDGES_FILE,
    BEARING_MATRIX_FILE,
    DISTANCE_EDGES_FILE,
    DISTANCE_MATRIX_FILE,
    FEATURED_DIR,
    GRAPH_DIR,
    K_NEIGHBORS,
    ML_VALIDATION_DIR,
    STATIONS_METADATA_FILE,
    STATIC_GRAPH_FILE,
    STATION_MAPPING_FILE,
    TRAIN_DIR,
)
from utils import station_dataset_name


OUTPUT_DIR = GRAPH_DIR / "design_audit"


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_difference(deg_a, deg_b):
    return abs((deg_a - deg_b + 180) % 360 - 180)


def build_canonical_node_table():
    metadata = pd.read_csv(STATIONS_METADATA_FILE)
    duplicated_station_names = set(
        metadata.loc[
            metadata["station"].duplicated(keep=False),
            "station",
        ]
    )
    rows = []

    featured_names = {path.stem for path in FEATURED_DIR.glob("*.csv")}
    train_names = {path.stem for path in TRAIN_DIR.glob("*.csv")}
    validation_names = {path.stem for path in ML_VALIDATION_DIR.glob("*.csv")}

    for _, row in metadata.iterrows():
        dataset_name = station_dataset_name(
            row["station"],
            sensor_id=row["pm25_sensor_id"],
            require_sensor_id=row["station"] in duplicated_station_names,
        )
        rows.append({
            "dataset_name": dataset_name,
            "station": row["station"],
            "location_id": row["location_id"],
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

    node_df = pd.DataFrame(rows).sort_values(
        ["dataset_name", "pm25_sensor_id"],
    ).reset_index(drop=True)
    node_df.insert(0, "recommended_node_id", range(len(node_df)))

    return node_df


def audit_identity(node_df):
    current_mapping = pd.read_csv(STATION_MAPPING_FILE)
    featured_names = {path.stem for path in FEATURED_DIR.glob("*.csv")}
    mapping_names = set(current_mapping["dataset_name"])

    return pd.DataFrame([{
        "metadata_rows": len(node_df),
        "metadata_unique_human_station_names": node_df["station"].nunique(),
        "metadata_unique_pm25_sensors": node_df["pm25_sensor_id"].nunique(),
        "featured_datasets": len(featured_names),
        "model_usable_train_validation_datasets": int(
            node_df["model_usable"].sum()
        ),
        "current_station_mapping_nodes": len(current_mapping),
        "human_station_names_with_duplicates": int(
            node_df["station"].duplicated(keep=False).sum()
        ),
        "current_mapping_missing_featured_datasets": len(
            featured_names - mapping_names
        ),
    }])


def audit_coordinates(node_df):
    grouped = node_df.groupby("dataset_name")
    return pd.DataFrame([{
        "rows": len(node_df),
        "dataset_name_unique": node_df["dataset_name"].is_unique,
        "pm25_sensor_id_unique": node_df["pm25_sensor_id"].is_unique,
        "missing_coordinates": int(
            node_df[["latitude", "longitude"]].isna().any(axis=1).sum()
        ),
        "dataset_names_with_multiple_coordinate_pairs": int(
            grouped[["latitude", "longitude"]]
            .nunique()
            .gt(1)
            .any(axis=1)
            .sum()
        ),
    }])


def audit_distance_bearing():
    mapping = pd.read_csv(STATION_MAPPING_FILE)
    distance_matrix = pd.read_csv(DISTANCE_MATRIX_FILE, index_col=0)
    bearing_matrix = pd.read_csv(BEARING_MATRIX_FILE, index_col=0)
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)
    bearing_matrix.index = bearing_matrix.index.astype(int)
    bearing_matrix.columns = bearing_matrix.columns.astype(int)

    distance_edges = pd.read_csv(DISTANCE_EDGES_FILE)
    bearing_edges = pd.read_csv(BEARING_EDGES_FILE)

    max_distance_error = 0.0
    max_bearing_error = 0.0
    max_reverse_bearing_error = 0.0
    for _, src in mapping.iterrows():
        for _, dst in mapping.iterrows():
            if src["node_id"] == dst["node_id"]:
                continue
            expected_distance = round(
                haversine(
                    src["latitude"],
                    src["longitude"],
                    dst["latitude"],
                    dst["longitude"],
                ),
                3,
            )
            actual_distance = distance_matrix.loc[
                src["node_id"],
                dst["node_id"],
            ]
            max_distance_error = max(
                max_distance_error,
                abs(expected_distance - actual_distance),
            )

            expected_bearing = round(
                bearing(
                    src["latitude"],
                    src["longitude"],
                    dst["latitude"],
                    dst["longitude"],
                ),
                2,
            )
            actual_bearing = bearing_matrix.loc[
                src["node_id"],
                dst["node_id"],
            ]
            max_bearing_error = max(
                max_bearing_error,
                angle_difference(expected_bearing, actual_bearing),
            )
            reverse_bearing = bearing_matrix.loc[
                dst["node_id"],
                src["node_id"],
            ]
            max_reverse_bearing_error = max(
                max_reverse_bearing_error,
                abs(angle_difference(actual_bearing + 180, reverse_bearing)),
            )

    return pd.DataFrame([{
        "nodes_in_current_mapping": len(mapping),
        "distance_matrix_shape": f"{distance_matrix.shape[0]}x{distance_matrix.shape[1]}",
        "distance_edges_rows": len(distance_edges),
        "expected_undirected_complete_edges": len(mapping) * (len(mapping) - 1) // 2,
        "distance_matrix_symmetric": distance_matrix.equals(
            distance_matrix.T
        ),
        "distance_diagonal_zero": bool(
            (distance_matrix.values.diagonal() == 0).all()
        ),
        "max_distance_recalc_error_km": max_distance_error,
        "bearing_matrix_shape": f"{bearing_matrix.shape[0]}x{bearing_matrix.shape[1]}",
        "bearing_edges_rows": len(bearing_edges),
        "expected_directed_complete_edges": len(mapping) * (len(mapping) - 1),
        "bearing_diagonal_zero": bool(
            (bearing_matrix.values.diagonal() == 0).all()
        ),
        "max_bearing_recalc_error_deg": max_bearing_error,
        "max_reverse_bearing_180_error_deg": max_reverse_bearing_error,
    }])


def audit_static_graph():
    adjacency = pd.read_csv(ADJACENCY_MATRIX_FILE, index_col=0)
    adjacency.index = adjacency.index.astype(int)
    adjacency.columns = adjacency.columns.astype(int)
    static_edges = pd.read_csv(STATIC_GRAPH_FILE)

    adjacency_directed_edges = {
        (int(src), int(dst))
        for src in adjacency.index
        for dst in adjacency.columns
        if src != dst and int(adjacency.loc[src, dst]) == 1
    }
    static_directed_edges = {
        (int(row["source"]), int(row["target"]))
        for _, row in static_edges.iterrows()
    }
    symmetric_candidate_edges = {
        (src, dst)
        for src, dst in adjacency_directed_edges
    }
    static_undirected = {
        tuple(sorted(edge))
        for edge in static_directed_edges
    }
    adjacency_undirected = {
        tuple(sorted(edge))
        for edge in adjacency_directed_edges
    }

    return pd.DataFrame([{
        "k_neighbors": K_NEIGHBORS,
        "nodes": len(adjacency),
        "static_edge_rows": len(static_edges),
        "adjacency_directed_edges": len(adjacency_directed_edges),
        "adjacency_undirected_edges": len(adjacency_undirected),
        "static_undirected_pairs": len(static_undirected),
        "adjacency_is_symmetric": adjacency.equals(adjacency.T),
        "static_missing_reverse_rows": int(
            sum(
                (dst, src) not in static_directed_edges
                for src, dst in static_directed_edges
            )
        ),
        "static_rows_not_in_adjacency": len(
            static_directed_edges - adjacency_directed_edges
        ),
        "adjacency_directed_edges_missing_from_static_rows": len(
            adjacency_directed_edges - static_directed_edges
        ),
        "recommended_candidate_directed_edges": len(symmetric_candidate_edges),
    }])


def write_report(identity_df, coordinate_df, geometry_df, static_df):
    lines = [
        "# Graph Design Audit",
        "",
        "This audit reviews current graph identity, coordinate, geometry, "
        "and static KNN artifacts before dynamic wind edges are implemented.",
        "",
        "## Identity",
        "",
        "```text",
        identity_df.to_string(index=False),
        "```",
        "",
        "## Coordinates",
        "",
        "```text",
        coordinate_df.to_string(index=False),
        "```",
        "",
        "## Distance And Bearing",
        "",
        "```text",
        geometry_df.to_string(index=False),
        "```",
        "",
        "## Static Graph",
        "",
        "```text",
        static_df.to_string(index=False),
        "```",
        "",
    ]
    (OUTPUT_DIR / "graph_design_audit.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    node_df = build_canonical_node_table()
    identity_df = audit_identity(node_df)
    coordinate_df = audit_coordinates(node_df)
    geometry_df = audit_distance_bearing()
    static_df = audit_static_graph()

    node_df.to_csv(OUTPUT_DIR / "recommended_graph_nodes.csv", index=False)
    identity_df.to_csv(OUTPUT_DIR / "identity_summary.csv", index=False)
    coordinate_df.to_csv(OUTPUT_DIR / "coordinate_summary.csv", index=False)
    geometry_df.to_csv(OUTPUT_DIR / "distance_bearing_summary.csv", index=False)
    static_df.to_csv(OUTPUT_DIR / "static_graph_summary.csv", index=False)
    write_report(identity_df, coordinate_df, geometry_df, static_df)

    print("Graph design audit complete")
    print(identity_df.to_string(index=False))
    print(static_df.to_string(index=False))


if __name__ == "__main__":
    main()
