"""Existing shelter facility locations for the 5 Florida panhandle counties.

FGDL's current data catalog (452 datasets, checked via its file manifest)
has no "shelter" layer - closest hits were parks/recreation datasets, not
emergency shelters. Falling back to Florida's state emergency management
agency (FDEM), which publishes an authoritative statewide "Risk Shelter
Inventory" via a public ArcGIS REST service - a better statewide source
than stitching together 5 separate county GIS portals.

Source: https://maps.floridadisaster.org/gis/rest/services/Facilities/Critical_Facilities/MapServer
Layers: 39 (General), 40 (Pet Friendly), 41 (Special Needs)
"""

import json
import os
import urllib.parse
import urllib.request

import geopandas as gpd

BASE = "https://maps.floridadisaster.org/gis/rest/services/Facilities/Critical_Facilities/MapServer"
LAYERS = {39: "general", 40: "pet_friendly", 41: "special_needs"}
COUNTIES = ["ESCAMBIA", "SANTA ROSA", "OKALOOSA", "WALTON", "BAY"]

OUT_DIR = "data/processed/shelters"
os.makedirs(OUT_DIR, exist_ok=True)


def query_layer(layer_id):
    county_list = ",".join(f"'{c}'" for c in COUNTIES)
    where = f"COUNTY IN ({county_list})"
    params = {
        "where": where,
        "outFields": "*",
        "f": "geojson",
    }
    url = f"{BASE}/{layer_id}/query?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    return data


all_gdfs = []
for layer_id, label in LAYERS.items():
    geojson = query_layer(layer_id)
    n = len(geojson.get("features", []))
    print(f"Layer {layer_id} ({label}): {n} features in the 5 counties")
    if n == 0:
        continue
    out_path = os.path.join(OUT_DIR, f"shelters_{label}.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    gdf = gpd.read_file(out_path)
    gdf["shelter_layer"] = label
    all_gdfs.append(gdf)
    print(f"  saved to {out_path}")
    print(f"  counties present: {sorted(gdf['COUNTY'].unique().tolist())}")

if all_gdfs:
    combined = gpd.GeoDataFrame(gpd.pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
    combined_path = os.path.join(OUT_DIR, "shelters_panhandle_combined.geojson")
    combined.to_file(combined_path, driver="GeoJSON")
    print(f"\nCombined: {len(combined)} total shelter features -> {combined_path}")
    print(combined["shelter_layer"].value_counts())
