"""Road-network routing analysis: for every land block group, the drive time
to the nearest existing shelter along the OSM road network, and which block
groups (and how many people) fall outside 5 / 10 / 15-minute shelter
coverage.

Method:
  - load the drivable OSM graph, impute edge speeds from OSM maxspeed tags
    (osmnx defaults by highway class where missing), derive per-edge
    travel_time in seconds
  - snap each shelter and each block-group internal point to its nearest
    graph node
  - one multi-source Dijkstra from all shelter nodes at once gives every
    graph node its travel time to the *nearest* shelter; look up each
    block group's node
  - classify coverage by 5 / 10 / 15-minute thresholds; population outside
    15 minutes is the coverage gap

Water block groups (tract 99xxxx, zero population) are excluded, consistent
with the anomaly and suitability deliverables.
"""

import os

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

GRAPH_PATH = "data/processed/osm/panhandle_roads.graphml"
BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
SHELTER_PATH = "data/processed/shelters/shelters_panhandle_combined.geojson"
OUT_DIR = "data/processed/routing"
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLDS_MIN = [5, 10, 15]

print("Loading road graph...")
G = ox.load_graphml(GRAPH_PATH)
G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)
print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# --- block groups (land only) ---
bg = gpd.read_file(BG_PATH)
water_mask = bg["TRACTCE"].str.startswith("99") | (bg["P1_001N"] == 0)
bg = bg[~water_mask].reset_index(drop=True)
bg["lon"] = bg["INTPTLON"].astype(float)
bg["lat"] = bg["INTPTLAT"].astype(float)
print(f"Land block groups: {len(bg)}")

# --- shelters ---
shelters = gpd.read_file(SHELTER_PATH)
shelters = shelters[shelters.geometry.notna()].reset_index(drop=True)
print(f"Shelters: {len(shelters)}")

# --- snap to nearest graph nodes ---
bg["node"] = ox.distance.nearest_nodes(G, bg["lon"].values, bg["lat"].values)
shelter_nodes = ox.distance.nearest_nodes(
    G, shelters.geometry.x.values, shelters.geometry.y.values
)
shelter_nodes = list(set(shelter_nodes))
print(f"Shelters snapped to {len(shelter_nodes)} distinct graph nodes")

# --- multi-source Dijkstra: every node -> travel time to nearest shelter ---
print("Running multi-source Dijkstra from all shelter nodes...")
dist = nx.multi_source_dijkstra_path_length(G, shelter_nodes, weight="travel_time")

bg["drive_seconds_to_shelter"] = bg["node"].map(dist)
bg["drive_min_to_shelter"] = bg["drive_seconds_to_shelter"] / 60

n_unreachable = bg["drive_min_to_shelter"].isna().sum()
print(f"Block groups with no network path to any shelter: {n_unreachable}")

for t in THRESHOLDS_MIN:
    bg[f"within_{t}min"] = bg["drive_min_to_shelter"] <= t

# --- coverage summary ---
total_pop = bg["P1_001N"].sum()
print("\n" + "=" * 70)
print("SHELTER COVERAGE (drive time along road network)")
print("=" * 70)
print(f"{'threshold':>12} {'block groups':>14} {'population':>14} {'pop %':>8}")
for t in THRESHOLDS_MIN:
    covered = bg[bg[f"within_{t}min"]]
    print(f"{'<= ' + str(t) + ' min':>12} {len(covered):>14} {covered['P1_001N'].sum():>14,} "
          f"{covered['P1_001N'].sum() / total_pop * 100:>7.1f}%")
outside = bg[(bg["drive_min_to_shelter"] > 15) | bg["drive_min_to_shelter"].isna()]
print(f"{'> 15 min / none':>12} {len(outside):>14} {outside['P1_001N'].sum():>14,} "
      f"{outside['P1_001N'].sum() / total_pop * 100:>7.1f}%")

print(f"\nDrive-time-to-nearest-shelter distribution (minutes):")
print(bg["drive_min_to_shelter"].describe())

print("\nCoverage gap (>15 min) block groups by county:")
print(outside["COUNTY_NAME"].value_counts())

print("\nTop 15 most-isolated block groups (longest drive to a shelter):")
worst = bg.sort_values("drive_min_to_shelter", ascending=False).head(15)
with pd.option_context("display.width", 200):
    print(worst[["GEOID", "COUNTY_NAME", "P1_001N", "drive_min_to_shelter"]].to_string(index=False))

# --- save ---
keep = [
    "GEOID", "COUNTY_NAME", "NAMELSAD", "P1_001N",
    "drive_seconds_to_shelter", "drive_min_to_shelter",
    "within_5min", "within_10min", "within_15min", "geometry",
]
out = bg[keep].sort_values("drive_min_to_shelter", ascending=False)
out.to_file(os.path.join(OUT_DIR, "blockgroup_shelter_drivetime.geojson"), driver="GeoJSON")
out.drop(columns="geometry").to_csv(
    os.path.join(OUT_DIR, "blockgroup_shelter_drivetime.csv"), index=False
)
print(f"\nSaved to {OUT_DIR}/blockgroup_shelter_drivetime.(geojson|csv)")
