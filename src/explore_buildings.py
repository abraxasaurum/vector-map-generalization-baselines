from pathlib import Path

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "osm_buildings_tu_berlin.geojson"
SUMMARY_FILE = PROJECT_ROOT / "outputs" / "building_summary.csv"
FIGURE_FILE = PROJECT_ROOT / "outputs" / "building_dataset_overview.png"

def outer_vertex_count(geometry):
    if geometry.geom_type == "Polygon":
        return len(geometry.exterior.coords) - 1
    if geometry.geom_type == "MultiPolygon":
        return sum(len(polygon.exterior.coords) - 1 for polygon in geometry.geoms)
    return np.nan

buildings = gpd.read_file(INPUT_FILE)

print(f"Eingelesene Gebäude: {len(buildings)}")
print("\nGeometrietypen:")
print(buildings.geometry.geom_type.value_counts())

buildings["area_m2"] = buildings.geometry.area
buildings["perimeter_m"] = buildings.geometry.length
buildings["n_vertices"] = buildings.geometry.apply(outer_vertex_count)
buildings["compactness"] = (
    4 * np.pi * buildings["area_m2"] / buildings["perimeter_m"] ** 2
)

summary = buildings[
    ["area_m2", "perimeter_m", "n_vertices", "compactness"]
].describe().T
summary.to_csv(SUMMARY_FILE)

print("\nDeskriptive Statistik:")
print(summary.round(2))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].hist(buildings["area_m2"], bins=30, color="#4C78A8", edgecolor="white")
axes[0].set_title("Building area")
axes[0].set_xlabel("Area [m²]")

axes[1].hist(buildings["n_vertices"], bins=range(3, int(buildings["n_vertices"].max()) + 2),
             color="#F58518", edgecolor="white")
axes[1].set_title("Outer-ring vertices")
axes[1].set_xlabel("Number of vertices")

axes[2].scatter(
    buildings["area_m2"],
    buildings["n_vertices"],
    alpha=0.65,
    color="#54A24B",
    edgecolor="none",
)
axes[2].set_xscale("log")
axes[2].set_title("Geometry complexity versus area")
axes[2].set_xlabel("Area [m², log scale]")
axes[2].set_ylabel("Number of vertices")

fig.suptitle("OSM building footprint dataset: initial exploration", y=1.03)
plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nStatistik gespeichert: {SUMMARY_FILE}")
print(f"Abbildung gespeichert: {FIGURE_FILE}")
