"""OpenStreetMap road/path network for the 5 Florida panhandle counties,
for later closest-facility / service-area routing analysis.

Uses osmnx to pull the drivable road network by place boundary (queried
against the OSM/Nominatim place database), then saves both the routable
graph (GraphML) and the edges/nodes as GeoJSON for use outside networkx.
"""

import os

import osmnx as ox

# Default overpass-api.de mirror was intermittently timing out; kumi.systems
# is a standard, reliable public Overpass mirror
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
ox.settings.overpass_rate_limit = True

COUNTIES = [
    "Escambia County, Florida, USA",
    "Santa Rosa County, Florida, USA",
    "Okaloosa County, Florida, USA",
    "Walton County, Florida, USA",
    "Bay County, Florida, USA",
]

OUT_DIR = "data/processed/osm"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Downloading drivable road network for: {COUNTIES}")
G = ox.graph_from_place(COUNTIES, network_type="drive")

n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
print(f"Graph: {n_nodes} nodes, {n_edges} edges")

graphml_path = os.path.join(OUT_DIR, "panhandle_roads.graphml")
ox.save_graphml(G, graphml_path)
print(f"Saved routable graph to {graphml_path}")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

edges_path = os.path.join(OUT_DIR, "panhandle_roads_edges.geojson")
edges_gdf.reset_index().to_file(edges_path, driver="GeoJSON")
print(f"Saved {len(edges_gdf)} edges to {edges_path}")

nodes_path = os.path.join(OUT_DIR, "panhandle_roads_nodes.geojson")
nodes_gdf.reset_index().to_file(nodes_path, driver="GeoJSON")
print(f"Saved {len(nodes_gdf)} nodes to {nodes_path}")

print(f"\nEdges file size: {os.path.getsize(edges_path) / (1024*1024):.1f} MB")
print(f"Nodes file size: {os.path.getsize(nodes_path) / (1024*1024):.1f} MB")
print(f"GraphML file size: {os.path.getsize(graphml_path) / (1024*1024):.1f} MB")
