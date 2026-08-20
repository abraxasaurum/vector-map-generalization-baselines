from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.polygon import orient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_context_source.geojson"
TARGET_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_context_target.geojson"

SEQUENCE_FILE = PROJECT_ROOT / "data" / "processed" / "context_adaptive_sequences.npz"
METADATA_FILE = PROJECT_ROOT / "data" / "processed" / "context_adaptive_metadata.csv"

N_POINTS = 32

CONTEXT_COLUMNS = [
    "nearest_neighbor_distance_m",
    "neighbors_within_25m",
    "local_building_density_per_km2",
]


def canonical_ring(polygon):
    polygon = orient(polygon, sign=1.0)
    coordinates = np.asarray(polygon.exterior.coords[:-1])

    start_index = np.lexsort((coordinates[:, 1], coordinates[:, 0]))[0]
    coordinates = np.vstack(
        [coordinates[start_index:], coordinates[:start_index]]
    )

    return np.vstack([coordinates, coordinates[0]])


def sample_polygon_boundary(polygon, n_points):
    ring = LineString(canonical_ring(polygon))
    distances = np.linspace(0, ring.length, num=n_points, endpoint=False)

    return np.array(
        [
            [ring.interpolate(distance).x, ring.interpolate(distance).y]
            for distance in distances
        ],
        dtype=np.float32,
    )


source = gpd.read_file(SOURCE_FILE).set_index("building_id").sort_index()
target = gpd.read_file(TARGET_FILE).set_index("building_id").sort_index()

if not source.index.equals(target.index):
    raise ValueError("Source- und Target-IDs stimmen nicht überein.")

source_sequences = []
target_sequences = []
context_rows = []
metadata_rows = []

for building_id in source.index:
    source_row = source.loc[building_id]
    target_row = target.loc[building_id]

    source_points = sample_polygon_boundary(source_row.geometry, N_POINTS)
    target_points = sample_polygon_boundary(target_row.geometry, N_POINTS)

    center = source_points.mean(axis=0)
    scale = np.linalg.norm(source_points - center, axis=1).max()

    source_normalized = (source_points - center) / scale
    target_normalized = (target_points - center) / scale

    source_sequences.append(source_normalized)
    target_sequences.append(target_normalized)

    context_rows.append(
        [source_row[column] for column in CONTEXT_COLUMNS]
    )

    metadata_rows.append(
        {
            "building_id": building_id,
            "source_area": source_row["source_area"],
            "context_tolerance_m": source_row["context_tolerance_m"],
            "center_x": center[0],
            "center_y": center[1],
            "scale_m": scale,
            "area_m2": source_row["area_m2"],
            "n_vertices_source": source_row["n_vertices_source"],
            "n_vertices_target": source_row["n_vertices_target"],
            "vertex_reduction": source_row["vertex_reduction"],
            **{
                column: source_row[column]
                for column in CONTEXT_COLUMNS
            },
        }
    )

X = np.stack(source_sequences)
y = np.stack(target_sequences)
context_raw = np.asarray(context_rows, dtype=np.float32)

context_mean = context_raw.mean(axis=0)
context_std = context_raw.std(axis=0)

context = (context_raw - context_mean) / context_std

metadata = pd.DataFrame(metadata_rows)

np.savez_compressed(
    SEQUENCE_FILE,
    X=X,
    y=y,
    context=context,
    context_mean=context_mean,
    context_std=context_std,
)

metadata.to_csv(METADATA_FILE, index=False)

print(f"Input tensor X: {X.shape}")
print(f"Target tensor y: {y.shape}")
print(f"Context tensor: {context.shape}")

print("\nContext normalization:")
for column, mean, std in zip(CONTEXT_COLUMNS, context_mean, context_std):
    print(f"{column}: mean={mean:.3f}, std={std:.3f}")

print(f"\nSequence dataset: {SEQUENCE_FILE}")
print(f"Metadata: {METADATA_FILE}")
