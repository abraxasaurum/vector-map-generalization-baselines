from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "osm_buildings_tu_berlin.geojson"
FIGURE = PROJECT_ROOT / "outputs" / "osm_buildings_tu_berlin.png"

# Kleines Gebiet nahe dem TU-Berlin-Campus, WGS84:
# (west, south, east, north)
BBOX = (13.315, 52.511, 13.326, 52.518)

ox.settings.use_cache = True
ox.settings.log_console = True

print("Lade Gebäude aus OpenStreetMap ...")
buildings = ox.features_from_bbox(BBOX, tags={"building": True})

# Nur Flächengeometrien behalten; Linien oder Punkte sind für dieses Projekt ungeeignet.
buildings = buildings[
    buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()

# Ein metrisches CRS ist Voraussetzung für sinnvolle Flächen, Distanzen und Toleranzen.
buildings = buildings.to_crs("EPSG:25833")
buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty].copy()
buildings = buildings.reset_index(drop=True)

print(f"Gebäude-Polygone: {len(buildings)}")
print(f"CRS: {buildings.crs}")

RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
FIGURE.parent.mkdir(parents=True, exist_ok=True)

buildings.to_file(RAW_DATA, driver="GeoJSON")

ax = buildings.plot(
    figsize=(10, 10),
    facecolor="#4C78A8",
    edgecolor="white",
    linewidth=0.25,
)
ax.set_title("OpenStreetMap building footprints near TU Berlin")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(FIGURE, dpi=180, bbox_inches="tight")
plt.close()

print(f"GeoJSON gespeichert: {RAW_DATA}")
print(f"Abbildung gespeichert: {FIGURE}")
