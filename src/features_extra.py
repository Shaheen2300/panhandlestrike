"""Engineer the genuinely-new, mostly-non-circular features for the ML layer:

  - spatial_lag_flash_density : mean baseline flash density (flash_per_km2)
    of the queen-contiguity neighbours of each block group. Storms cross
    block-group lines, so neighbouring activity is real independent signal
    about the local regime (semi-circular - neighbour flashes correlate
    with own - but it adds neighbourhood information the point value does not).
  - coast_dist_km             : distance from the block group's internal
    point to the nearest coastline segment (TIGER national coastline).
    Pure geography, fully non-circular. Sea-breeze convergence is a
    documented driver of Florida warm-season lightning.

Output: data/processed/features/blockgroup_features_extra.csv
"""

import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

COMBINED = "data/processed/final/blockgroup_combined_priority.geojson"
OUT = "data/processed/features/blockgroup_features_extra.csv"
COAST_ZIP = "data/raw/coastline/tl_2022_us_coastline.zip"
PROJ = "EPSG:5070"  # CONUS Albers, metres
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(os.path.dirname(COAST_ZIP), exist_ok=True)

gdf = gpd.read_file(COMBINED).to_crs(PROJ)

# --- spatial lag: queen-contiguity neighbour mean of flash_per_km2 ---
pairs = gpd.sjoin(
    gdf[["GEOID", "geometry"]],
    gdf[["GEOID", "flash_per_km2", "geometry"]].rename(columns={"GEOID": "GEOID_nb"}),
    predicate="touches", how="left",
)
pairs = pairs[pairs["GEOID"] != pairs["GEOID_nb"]]
lag = pairs.groupby("GEOID")["flash_per_km2"].mean().rename("spatial_lag_flash_density")
n_nb = pairs.groupby("GEOID").size().rename("n_neighbours")
gdf = gdf.merge(lag, on="GEOID", how="left").merge(n_nb, on="GEOID", how="left")
# island block groups with no touching neighbour -> fall back to study-area mean
gdf["spatial_lag_flash_density"] = gdf["spatial_lag_flash_density"].fillna(
    gdf["flash_per_km2"].mean()
)
gdf["n_neighbours"] = gdf["n_neighbours"].fillna(0).astype(int)
print(f"spatial lag: {int((gdf['n_neighbours'] == 0).sum())} block groups had no touching neighbour")

# --- distance to coastline ---
if not os.path.exists(COAST_ZIP):
    url = "https://www2.census.gov/geo/tiger/TIGER2022/COASTLINE/tl_2022_us_coastline.zip"
    r = requests.get(url, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(COAST_ZIP, "wb") as f:
        f.write(r.content)
    print(f"downloaded coastline ({len(r.content) / 1e6:.1f} MB)")

coast = gpd.read_file(f"zip://{COAST_ZIP}").to_crs(PROJ)
# clip to a generous box around the study area to keep the distance query fast
study_box = gdf.total_bounds
pad = 200_000  # 200 km
from shapely.geometry import box as _box
clip_geom = _box(study_box[0] - pad, study_box[1] - pad, study_box[2] + pad, study_box[3] + pad)
coast = coast[coast.intersects(clip_geom)]
print(f"coastline segments near study area: {len(coast)}")

bg_pts = gpd.GeoDataFrame(
    gdf[["GEOID"]],
    geometry=gpd.points_from_xy(
        gpd.read_file(COMBINED).to_crs(PROJ).geometry.centroid.x,
        gpd.read_file(COMBINED).to_crs(PROJ).geometry.centroid.y,
    ),
    crs=PROJ,
)
coast_union = coast.union_all()
bg_pts["coast_dist_km"] = bg_pts.geometry.distance(coast_union) / 1000
gdf = gdf.merge(bg_pts[["GEOID", "coast_dist_km"]], on="GEOID", how="left")

out = gdf[[
    "GEOID", "COUNTY_NAME", "P1_001N", "flash_per_km2", "drive_min_to_shelter",
    "spatial_lag_flash_density", "n_neighbours", "coast_dist_km",
    "n_days_both_flag", "has_confirmed_anomaly",
]].copy()
out.to_csv(OUT, index=False)

print(f"\nSaved {len(out)} rows to {OUT}")
print(out[["spatial_lag_flash_density", "coast_dist_km"]].describe().round(2))
print("\ncoast_dist_km by county (mean):")
print(out.groupby("COUNTY_NAME")["coast_dist_km"].mean().round(1))
print("\ncorrelation of new features with flash_per_km2 (circularity check):")
print(out[["flash_per_km2", "spatial_lag_flash_density", "coast_dist_km"]].corr().round(3)["flash_per_km2"])
