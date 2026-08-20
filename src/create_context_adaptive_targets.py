from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_with_spatial_context.geojson"

SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_context_source.geojson"
TARGET_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_context_target.geojson"

SUMMARY_FILE = PROJECT_ROOT / "reports" / "context_adaptive_target_summary.csv"
FIGURE_FILE = PROJECT_ROOT / "reports" / "context_adaptive_targets.png"


def outer_vertex_count(geometry):
    if geometry.geom_type == "Polygon":
        return len(geometry.exterior.coords) - 1
    return np.nan


def context_tolerance(neighbor_count):
    if neighbor_count == 0:
        return 1.0
    if neighbor_count <= 2:
        return 2.0
    return 3.0


buildings = gpd.read_file(INPUT_FILE)

required_columns = {
    "building_id",
    "source_area",
    "nearest_neighbor_distance_m",
    "neighbors_within_25m",
    "local_building_density_per_km2",
}

missing_columns = required_columns - set(buildings.columns)
if missing_columns:
    raise ValueError(f"Fehlende Spalten: {sorted(missing_columns)}")

buildings = buildings[
    buildings.geometry.geom_type == "Polygon"
].copy()

buildings["context_tolerance_m"] = buildings[
    "neighbors_within_25m"
].apply(context_tolerance)

buildings["geometry_target"] = buildings.apply(
    lambda row: row.geometry.simplify(
        tolerance=row["context_tolerance_m"],
        preserve_topology=True,
    ),
    axis=1,
)

buildings["n_vertices_source"] = buildings.geometry.apply(outer_vertex_count)
buildings["n_vertices_target"] = buildings["geometry_target"].apply(
    outer_vertex_count
)

buildings = buildings[
    buildings["n_vertices_target"] < buildings["n_vertices_source"]
].copy()

buildings["vertex_reduction"] = (
    1
    - buildings["n_vertices_target"] / buildings["n_vertices_source"]
)

buildings = buildings.reset_index(drop=True)

source_columns = [
    "building_id",
    "source_area",
    "area_m2",
    "nearest_neighbor_distance_m",
    "neighbors_within_25m",
    "local_building_density_per_km2",
    "context_tolerance_m",
    "n_vertices_source",
    "n_vertices_target",
    "vertex_reduction",
    "geometry",
]

target_columns = [
    "building_id",
    "source_area",
    "area_m2",
    "nearest_neighbor_distance_m",
    "neighbors_within_25m",
    "local_building_density_per_km2",
    "context_tolerance_m",
    "n_vertices_source",
    "n_vertices_target",
    "vertex_reduction",
    "geometry_target",
]

source = buildings[source_columns].copy()

target = buildings[target_columns].copy()
target = target.set_geometry("geometry_target").rename_geometry("geometry")
target = target.set_crs(source.crs, allow_override=True)

SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
source.to_file(SOURCE_FILE, driver="GeoJSON")
target.to_file(TARGET_FILE, driver="GeoJSON")

summary = (
    source.groupby("context_tolerance_m")
    .agg(
        buildings=("building_id", "count"),
        mean_neighbors=("neighbors_within_25m", "mean"),
        median_nearest_neighbor_m=("nearest_neighbor_distance_m", "median"),
        median_source_vertices=("n_vertices_source", "median"),
        median_target_vertices=("n_vertices_target", "median"),
        mean_vertex_reduction=("vertex_reduction", "mean"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_FILE, index=False)

print(f"Input buildings: {len(gpd.read_file(INPUT_FILE))}")
print(f"Buildings after target generation: {len(source)}")

print("\nContext-adaptive target summary:")
print(summary.round(3).to_string(index=False))

print("\nTolerance counts:")
print(source["context_tolerance_m"].value_counts().sort_index())

examples = []

for tolerance in [1.0, 2.0, 3.0]:
    candidates = source[
        source["context_tolerance_m"] == tolerance
    ].sort_values("n_vertices_source", ascending=False)

    if not candidates.empty:
        examples.append(candidates.iloc[0])

fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 5))

if len(examples) == 1:
    axes = [axes]

for axis, row in zip(axes, examples):
    source_geometry = row.geometry
    target_geometry = target.loc[
        target["building_id"] == row["building_id"]
    ].geometry.iloc[0]

    gpd.GeoSeries([source_geometry], crs=source.crs).plot(
        ax=axis,
        facecolor="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
    )

    gpd.GeoSeries([target_geometry], crs=source.crs).plot(
        ax=axis,
        facecolor="none",
        edgecolor="#E45756",
        linewidth=1.5,
    )

    axis.set_title(
        f"{row['context_tolerance_m']:.0f} m tolerance\n"
        f"{int(row['neighbors_within_25m'])} neighbours within 25 m\n"
        f"{int(row['n_vertices_source'])} → "
        f"{int(row['n_vertices_target'])} vertices"
    )
    axis.set_axis_off()

fig.suptitle(
    "Context-adaptive Douglas-Peucker targets: source (blue), target (red)",
    y=0.98,
)
plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nContext source dataset: {SOURCE_FILE}")
print(f"Context target dataset: {TARGET_FILE}")
print(f"Summary: {SUMMARY_FILE}")
print(f"Figure: {FIGURE_FILE}")
