from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.polygon import orient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_source.geojson"
TARGET_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_target_dp_2m.geojson"
SEQUENCE_FILE = PROJECT_ROOT / "data" / "processed" / "building_sequences_32points.npz"
METADATA_FILE = PROJECT_ROOT / "data" / "processed" / "sequence_metadata.csv"
FIGURE_FILE = PROJECT_ROOT / "outputs" / "sequence_representation_example.png"

N_POINTS = 32


def canonical_ring(polygon):
    """Orientiert ein Polygon gegen den Uhrzeigersinn und setzt einen festen Startpunkt."""
    polygon = orient(polygon, sign=1.0)
    coordinates = np.asarray(polygon.exterior.coords[:-1])

    # Deterministischer Start: westlichster Punkt, bei Gleichstand südlichster Punkt.
    start_index = np.lexsort((coordinates[:, 1], coordinates[:, 0]))[0]
    coordinates = np.vstack(
        [coordinates[start_index:], coordinates[:start_index]]
    )

    return np.vstack([coordinates, coordinates[0]])


def sample_polygon_boundary(polygon, n_points):
    """Erzeugt n gleichmäßig entlang des äußeren Polygonrings verteilte Punkte."""
    ring = LineString(canonical_ring(polygon))
    distances = np.linspace(0, ring.length, num=n_points, endpoint=False)

    return np.array(
        [[ring.interpolate(distance).x, ring.interpolate(distance).y]
         for distance in distances],
        dtype=np.float32,
    )


source = gpd.read_file(SOURCE_FILE).set_index("building_id").sort_index()
target = gpd.read_file(TARGET_FILE).set_index("building_id").sort_index()

if not source.index.equals(target.index):
    raise ValueError("Source- und Target-IDs stimmen nicht überein.")

source_sequences = []
target_sequences = []
metadata = []

for building_id in source.index:
    source_geometry = source.loc[building_id].geometry
    target_geometry = target.loc[building_id].geometry

    source_points = sample_polygon_boundary(source_geometry, N_POINTS)
    target_points = sample_polygon_boundary(target_geometry, N_POINTS)

    # Jede Geometrie wird relativ zum Quellgebäude zentriert und skaliert.
    center = source_points.mean(axis=0)
    scale = np.linalg.norm(source_points - center, axis=1).max()

    source_normalized = (source_points - center) / scale
    target_normalized = (target_points - center) / scale

    source_sequences.append(source_normalized)
    target_sequences.append(target_normalized)

    metadata.append(
        {
            "building_id": building_id,
            "source_area": source.loc[building_id, "source_area"],
            "center_x": center[0],
            "center_y": center[1],
            "scale_m": scale,
            "area_m2": source.loc[building_id, "area_m2"],
            "n_vertices_source": source.loc[building_id, "n_vertices_source"],
            "n_vertices_target": source.loc[building_id, "n_vertices_target"],
            "vertex_reduction": source.loc[building_id, "vertex_reduction"],
        }
    )

X = np.stack(source_sequences)
y = np.stack(target_sequences)
metadata = pd.DataFrame(metadata)

np.savez_compressed(SEQUENCE_FILE, X=X, y=y)
metadata.to_csv(METADATA_FILE, index=False)

print(f"Input-Tensor X:  {X.shape}")
print(f"Target-Tensor y: {y.shape}")
print(f"Wertebereich X:  [{X.min():.3f}, {X.max():.3f}]")
print(f"Wertebereich y:  [{y.min():.3f}, {y.max():.3f}]")

example_index = 0

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(
    X[example_index, :, 0],
    X[example_index, :, 1],
    marker="o",
    markersize=3,
    color="#4C78A8",
)
axes[0].set_title("Source: 32 sampled boundary points")
axes[0].set_aspect("equal")
axes[0].grid(alpha=0.3)

axes[1].plot(
    y[example_index, :, 0],
    y[example_index, :, 1],
    marker="o",
    markersize=3,
    color="#E45756",
)
axes[1].set_title("Target: 32 sampled boundary points")
axes[1].set_aspect("equal")
axes[1].grid(alpha=0.3)

fig.suptitle("Normalized fixed-length polygon sequence representation", y=0.98)
plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nSequenzen gespeichert: {SEQUENCE_FILE}")
print(f"Metadaten gespeichert: {METADATA_FILE}")
print(f"Abbildung gespeichert: {FIGURE_FILE}")
