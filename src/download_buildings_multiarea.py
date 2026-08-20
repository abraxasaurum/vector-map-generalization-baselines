from pathlib import Path
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "osm_buildings_berlin_multiarea.geojson"

# Bounding boxes: (west, south, east, north), WGS84 / EPSG:4326
AREAS = {
    "charlottenburg": (13.315, 52.511, 13.326, 52.518),
    "moabit": (13.335, 52.525, 13.348, 52.534),
    "schoeneberg": (13.342, 52.478, 13.356, 52.488),
    "kreuzberg": (13.395, 52.492, 13.409, 52.502),
    "friedrichshain": (13.420, 52.510, 13.434, 52.520),
    "prenzlauer_berg": (13.405, 52.535, 13.419, 52.545),
    "wedding": (13.355, 52.545, 13.369, 52.555),
    "lichtenberg": (13.475, 52.511, 13.489, 52.521),
}

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.overpass_rate_limit = True

all_buildings = []
failed_areas = []

for area_name, bbox in AREAS.items():
    print(f"\n{'=' * 70}")
    print(f"Lade Gebiet: {area_name}")

    try:
        buildings = ox.features_from_bbox(
            bbox=bbox,
            tags={"building": True},
        )

        buildings = buildings[
            buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()

        buildings["source_area"] = area_name
        all_buildings.append(buildings)

        print(f"Gebäudeobjekte in {area_name}: {len(buildings)}")

    except Exception as error:
        failed_areas.append(area_name)
        print(f"FEHLER in {area_name}: {error}")

    # Höflicher Abstand zwischen Overpass-Anfragen.
    time.sleep(3)

if not all_buildings:
    raise RuntimeError("Keine Gebäude konnten heruntergeladen werden.")

buildings = gpd.GeoDataFrame(
    pd.concat(all_buildings, ignore_index=True),
    crs="EPSG:4326",
)

buildings = buildings.to_crs("EPSG:25833")
buildings = buildings[
    buildings.geometry.is_valid
    & ~buildings.geometry.is_empty
].copy()

buildings = buildings.reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
buildings.to_file(OUTPUT_FILE, driver="GeoJSON")

print(f"\n{'=' * 70}")
print(f"Erfolgreich geladene Gebiete: {len(all_buildings)} / {len(AREAS)}")
print(f"Gebäude-Polygone insgesamt: {len(buildings)}")
print(f"CRS: {buildings.crs}")
print(f"Datensatz gespeichert: {OUTPUT_FILE}")

if failed_areas:
    print(f"Nicht geladene Gebiete: {', '.join(failed_areas)}")
