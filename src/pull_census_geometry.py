"""US Census block-group geometry for the 5 Florida panhandle counties, via
TIGER/Line (direct file download, no API key needed).

Note: the Census API itself (api.census.gov) now requires a free API key
for population attribute data (confirmed empirically - both the 2020 dec/pl
and ACS5 endpoints return "Missing Key" even for small anonymous requests).
This script gets the geometry only; population attributes are pulled
separately once a key is available.
"""

import os
import zipfile

import geopandas as gpd
import requests

STATE_FIPS = "12"  # Florida
COUNTY_FIPS = {
    "033": "ESCAMBIA",
    "113": "SANTA ROSA",
    "091": "OKALOOSA",
    "131": "WALTON",
    "005": "BAY",
}

OUT_DIR = "data/processed/census"
os.makedirs(OUT_DIR, exist_ok=True)


def download(url, local_path):
    if os.path.exists(local_path):
        return
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)


# 1. County boundaries - verify our FIPS-to-name assumptions empirically
county_zip = os.path.join(OUT_DIR, "tl_2022_us_county.zip")
download(
    "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip",
    county_zip,
)
counties = gpd.read_file(f"zip://{county_zip}")
fl_target = counties[(counties["STATEFP"] == STATE_FIPS) & (counties["COUNTYFP"].isin(COUNTY_FIPS.keys()))]
print("FIPS verification against real TIGER county file:")
for _, row in fl_target.iterrows():
    expected = COUNTY_FIPS[row["COUNTYFP"]]
    actual = row["NAME"].upper()
    match = "OK" if expected == actual else "MISMATCH"
    print(f"  {row['COUNTYFP']}: expected={expected} actual={actual} [{match}]")

# 2. Block group geometry for the whole state, then filter to our 5 counties
bg_zip = os.path.join(OUT_DIR, "tl_2022_12_bg.zip")
download(
    "https://www2.census.gov/geo/tiger/TIGER2022/BG/tl_2022_12_bg.zip",
    bg_zip,
)
bg = gpd.read_file(f"zip://{bg_zip}")
print(f"\nTotal FL block groups: {len(bg)}")

panhandle_bg = bg[bg["COUNTYFP"].isin(COUNTY_FIPS.keys())].copy()
panhandle_bg["COUNTY_NAME"] = panhandle_bg["COUNTYFP"].map(COUNTY_FIPS)
print(f"Panhandle 5-county block groups: {len(panhandle_bg)}")
print(panhandle_bg["COUNTY_NAME"].value_counts())

out_path = os.path.join(OUT_DIR, "panhandle_block_groups.geojson")
panhandle_bg.to_file(out_path, driver="GeoJSON")
print(f"\nSaved geometry to {out_path}")
print(f"Columns: {list(panhandle_bg.columns)}")
