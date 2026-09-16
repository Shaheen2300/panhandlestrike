"""US Census block-group population for the 5 Florida panhandle counties,
joined onto the geometry already pulled in pull_census_geometry.py.

Uses the 2020 Decennial Census (table P1: total population), the
authoritative exact-count source at block-group granularity, now that a
free API key is available (the API returns "Missing Key" for all
endpoints without one, confirmed earlier).
"""

import os

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CENSUS_API_KEY")
if not API_KEY:
    raise SystemExit("Missing CENSUS_API_KEY in .env")

STATE_FIPS = "12"
COUNTY_FIPS = {
    "033": "ESCAMBIA",
    "113": "SANTA ROSA",
    "091": "OKALOOSA",
    "131": "WALTON",
    "005": "BAY",
}

GEOM_PATH = "data/processed/census/panhandle_block_groups.geojson"
OUT_DIR = "data/processed/census"

frames = []
for county_fips, county_name in COUNTY_FIPS.items():
    url = "https://api.census.gov/data/2020/dec/pl"
    params = {
        "get": "NAME,P1_001N",
        "for": "block group:*",
        "in": f"state:{STATE_FIPS} county:{county_fips} tract:*",
        "key": API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["COUNTY_NAME"] = county_name
    frames.append(df)
    print(f"{county_name}: {len(df)} block groups, "
          f"total population {df['P1_001N'].astype(int).sum():,}")

pop = pd.concat(frames, ignore_index=True)
pop["GEOID"] = pop["state"] + pop["county"] + pop["tract"] + pop["block group"]
pop["P1_001N"] = pop["P1_001N"].astype(int)

print(f"\nTotal block groups with population data: {len(pop)}")
print(f"Total panhandle population (2020 Census): {pop['P1_001N'].sum():,}")

geom = gpd.read_file(GEOM_PATH)
print(f"Geometry block groups: {len(geom)}")

merged = geom.merge(pop[["GEOID", "P1_001N", "NAME"]], on="GEOID", how="left")
n_missing = merged["P1_001N"].isna().sum()
print(f"Block groups missing a population match after merge: {n_missing}")

merged["pop_density_per_sqkm"] = merged["P1_001N"] / (merged["ALAND"] / 1_000_000)

out_path = os.path.join(OUT_DIR, "panhandle_block_groups_population.geojson")
merged.to_file(out_path, driver="GeoJSON")
print(f"\nSaved merged geometry+population to {out_path}")
print(f"File size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")
print(f"Columns: {list(merged.columns)}")
