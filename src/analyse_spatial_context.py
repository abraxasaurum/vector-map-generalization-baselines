from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_source.geojson"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_with_spatial_context.geojson"
SUMMARY_FILE = PROJECT_ROOT / "reports" / "spatial_context_summary.csv"
FIGURE_FILE = PROJECT_ROOT / "reports" / "spatial_context_overview.png"

NEIGHBOR_RADIUS_M = 25.0

buildings = gpd.read_file(INPUT_FILE)

if buildings.crs is None or not buildings.crs.is_projected:
    raise ValueError("Die Daten benötigen ein projiziertes CRS in Metern.")

centroids = buildings.geometry.centroid
coordinates = np.column_stack([centroids.x, centroids.y])

tree = cKDTree(coordinates)

distances, indices = tree.query(coordinates, k=2)
buildings["nearest_neighbor_distance_m"] = distances[:, 1]

neighbour_lists = tree.query_ball_point(coordinates, r=NEIGHBOR_RADIUS_M)
buildings["neighbors_within_25m"] = [
    len(neighbours) - 1 for neighbours in neighbour_lists
]

circle_area_m2 = np.pi * NEIGHBOR_RADIUS_M ** 2
buildings["local_building_density_per_km2"] = (
    buildings["neighbors_within_25m"] / circle_area_m2 * 1_000_000
)

context_columns = [
    "nearest_neighbor_distance_m",
    "neighbors_within_25m",
    "local_building_density_per_km2",
]

summary = buildings[context_columns].describe().T
summary.to_csv(SUMMARY_FILE)

print(f"Buildings: {len(buildings)}")
print(f"Neighbour radius: {NEIGHBOR_RADIUS_M:.0f} m")
print("\nSpatial-context summary:")
print(summary.round(2))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
buildings.to_file(OUTPUT_FILE, driver="GeoJSON")

example_area = buildings[
    buildings["source_area"] == "kreuzberg"
].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

example_area.plot(
    ax=axes[0],
    column="nearest_neighbor_distance_m",
    cmap="viridis",
    legend=True,
    edgecolor="white",
    linewidth=0.15,
)
axes[0].set_title("Nearest-neighbour distance [m]")
axes[0].set_axis_off()

example_area.plot(
    ax=axes[1],
    column="neighbors_within_25m",
    cmap="magma",
    legend=True,
    edgecolor="white",
    linewidth=0.15,
)
axes[1].set_title("Buildings within 25 m")
axes[1].set_axis_off()

axes[2].scatter(
    buildings["nearest_neighbor_distance_m"],
    buildings["vertex_reduction"],
    alpha=0.35,
    color="#4C78A8",
    edgecolor="none",
)
axes[2].set_xlabel("Nearest-neighbour distance [m]")
axes[2].set_ylabel("Douglas-Peucker vertex reduction")
axes[2].set_title("Context versus geometric simplification")
axes[2].grid(alpha=0.25)

fig.suptitle(
    "Spatial context features for building-footprint generalization",
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nContext dataset: {OUTPUT_FILE}")
print(f"Summary: {SUMMARY_FILE}")
print(f"Figure: {FIGURE_FILE}")
