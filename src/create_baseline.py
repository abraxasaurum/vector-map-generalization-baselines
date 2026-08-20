from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "osm_buildings_berlin_multiarea.geojson"
SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_source.geojson"
TARGET_FILE = PROJECT_ROOT / "data" / "processed" / "buildings_target_dp_2m.geojson"
SUMMARY_FILE = PROJECT_ROOT / "outputs" / "baseline_summary.csv"
FIGURE_FILE = PROJECT_ROOT / "outputs" / "douglas_peucker_examples.png"

MIN_AREA_M2 = 50
MIN_VERTICES = 8
SIMPLIFICATION_TOLERANCE_M = 2.0


def outer_vertex_count(geometry):
    if geometry.geom_type == "Polygon":
        return len(geometry.exterior.coords) - 1
    return np.nan


buildings = gpd.read_file(INPUT_FILE)

# MultiPolygone in einzelne, unabhängige Polygonflächen zerlegen.
buildings = buildings.explode(index_parts=False).reset_index(drop=True)
buildings = buildings[buildings.geometry.geom_type == "Polygon"].copy()
buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty].copy()

buildings["area_m2"] = buildings.geometry.area
buildings["n_vertices_source"] = buildings.geometry.apply(outer_vertex_count)

selected = buildings[
    (buildings["area_m2"] >= MIN_AREA_M2)
    & (buildings["n_vertices_source"] >= MIN_VERTICES)
].copy()

selected["geometry_target"] = selected.geometry.simplify(
    tolerance=SIMPLIFICATION_TOLERANCE_M,
    preserve_topology=True,
)
selected["n_vertices_target"] = selected["geometry_target"].apply(outer_vertex_count)

# Nur Beispiele behalten, bei denen die Baseline tatsächlich vereinfacht.
selected = selected[
    selected["n_vertices_target"] < selected["n_vertices_source"]
].copy()

selected = selected.reset_index(drop=True)
selected["building_id"] = selected.index
selected["vertex_reduction"] = (
    1 - selected["n_vertices_target"] / selected["n_vertices_source"]
)

source = selected[
    ["building_id", "source_area", "area_m2", "n_vertices_source", "n_vertices_target", "vertex_reduction", "geometry"]
].copy()

target = selected[
    ["building_id", "source_area", "area_m2", "n_vertices_source", "n_vertices_target", "vertex_reduction", "geometry_target"]
].copy()
target = target.set_geometry("geometry_target").rename_geometry("geometry")

SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
source.to_file(SOURCE_FILE, driver="GeoJSON")
target.to_file(TARGET_FILE, driver="GeoJSON")

summary = pd.DataFrame(
    {
        "metric": [
            "input_buildings",
            "selected_buildings",
            "tolerance_m",
            "median_source_vertices",
            "median_target_vertices",
            "mean_vertex_reduction",
        ],
        "value": [
            len(buildings),
            len(selected),
            SIMPLIFICATION_TOLERANCE_M,
            selected["n_vertices_source"].median(),
            selected["n_vertices_target"].median(),
            selected["vertex_reduction"].mean(),
        ],
    }
)
summary.to_csv(SUMMARY_FILE, index=False)

print(f"Gebäude nach Qualitätsfilter: {len(selected)}")
print(f"Douglas-Peucker-Toleranz: {SIMPLIFICATION_TOLERANCE_M:.1f} m")
print(
    "Median Vertices: "
    f"{selected['n_vertices_source'].median():.0f} → "
    f"{selected['n_vertices_target'].median():.0f}"
)
print(
    "Mittlere Vertex-Reduktion: "
    f"{selected['vertex_reduction'].mean():.1%}"
)

examples = selected.nlargest(6, "n_vertices_source")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for axis, (_, row) in zip(axes.flat, examples.iterrows()):
    gpd.GeoSeries([row.geometry], crs=source.crs).plot(
        ax=axis,
        facecolor="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
    )
    gpd.GeoSeries([row.geometry_target], crs=source.crs).plot(
        ax=axis,
        facecolor="none",
        edgecolor="#E45756",
        linewidth=1.4,
    )
    axis.set_title(
        f"{int(row['n_vertices_source'])} → "
        f"{int(row['n_vertices_target'])} vertices"
    )
    axis.set_axis_off()

fig.suptitle(
    "Douglas-Peucker baseline: source geometry (blue) and simplified target (red)",
    y=0.98,
)
plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nSource dataset: {SOURCE_FILE}")
print(f"Target dataset: {TARGET_FILE}")
print(f"Summary: {SUMMARY_FILE}")
print(f"Examples: {FIGURE_FILE}")
