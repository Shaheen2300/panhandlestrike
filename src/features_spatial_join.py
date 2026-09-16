"""Foundation step for both downstream deliverables: assign every GLM flash to
a Census block group via point-in-polygon, and write one combined flash table
with a GEOID column.

Flashes that fall outside every block group (over water, or outside the
5-county land area but still inside the bbox) are dropped and counted.
"""

import glob
import os

import geopandas as gpd
import pandas as pd

GLM_DIR = "data/processed/glm"
BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
OUT_PATH = "data/processed/features/flashes_with_blockgroup.parquet"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# 1. Load and concatenate all monthly flash files
files = sorted(glob.glob(os.path.join(GLM_DIR, "glm_panhandle_*.parquet")))
flashes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
flashes["timestamp"] = pd.to_datetime(flashes["timestamp"])
print(f"Total flashes loaded: {len(flashes):,} from {len(files)} monthly files")

# 2. Build flash points GeoDataFrame in WGS84
gdf = gpd.GeoDataFrame(
    flashes,
    geometry=gpd.points_from_xy(flashes["flash_lon"], flashes["flash_lat"]),
    crs="EPSG:4326",
)

# 3. Block groups -> WGS84 to match
bg = gpd.read_file(BG_PATH).to_crs("EPSG:4326")
bg_slim = bg[["GEOID", "COUNTY_NAME", "P1_001N", "geometry"]]

# 4. Point-in-polygon spatial join
joined = gpd.sjoin(gdf, bg_slim, how="left", predicate="within")
joined = joined[~joined.index.duplicated(keep="first")]  # guard against polygon overlaps

n_outside = joined["GEOID"].isna().sum()
n_inside = joined["GEOID"].notna().sum()
print(f"Flashes inside a block group: {n_inside:,} ({n_inside/len(joined)*100:.1f}%)")
print(f"Flashes outside all block groups (water/off-land): {n_outside:,} ({n_outside/len(joined)*100:.1f}%)")

# 5. Write slim flash table (drop geometry, keep GEOID + core fields)
out = joined.loc[
    joined["GEOID"].notna(),
    ["flash_id", "flash_lat", "flash_lon", "flash_area", "flash_energy", "timestamp", "GEOID", "COUNTY_NAME"],
].reset_index(drop=True)
out.to_parquet(OUT_PATH, index=False)
print(f"\nWrote {len(out):,} block-group-assigned flashes to {OUT_PATH}")
print(f"File size: {os.path.getsize(OUT_PATH) / (1024*1024):.1f} MB")

print("\nFlashes per county:")
print(out["COUNTY_NAME"].value_counts())
print(f"\nBlock groups with at least one flash: {out['GEOID'].nunique()} / {len(bg)}")
