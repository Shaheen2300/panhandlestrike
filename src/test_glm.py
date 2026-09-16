"""Test pull of GOES-16 GLM lightning flash data directly from NOAA's public
AWS S3 bucket (no goes2go/cartopy dependency - direct s3fs + xarray access).

Pulls one hour on a known Florida storm-season afternoon and checks:
- file count / download success
- whether flash_lat/flash_lon/flash_id contain real data
- how many flashes fall inside the panhandle bounding box
"""

import os

import s3fs
import xarray as xr

BUCKET = "noaa-goes16"
PRODUCT = "GLM-L2-LCFA"
YEAR, DOY, HOUR = 2023, 227  , 18  # 2023-08-15, hour 18 UTC
LOCAL_DIR = "data/raw/glm_test"

MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 29.5, 31.5, -88.0, -84.5

fs = s3fs.S3FileSystem(anon=True)
prefix = f"{BUCKET}/{PRODUCT}/{YEAR}/{DOY:03d}/{HOUR:02d}/"

remote_files = fs.ls(prefix)
print(f"Files listed on S3 for this hour: {len(remote_files)}")

os.makedirs(LOCAL_DIR, exist_ok=True)
local_paths = []
for rf in remote_files:
    local_path = os.path.join(LOCAL_DIR, os.path.basename(rf))
    if not os.path.exists(local_path):
        fs.get(rf, local_path)
    local_paths.append(local_path)

print(f"Files downloaded/present locally: {len(local_paths)}")

ds0 = xr.open_dataset(local_paths[0])
print("\nFirst file contents check:")
print(f"flash_lat: {ds0['flash_lat'].values}")
print(f"flash_lon: {ds0['flash_lon'].values}")
print(f"flash_id: {ds0['flash_id'].values}")
ds0.close()

total_flashes = 0
panhandle_flashes = 0
for lp in local_paths:
    ds = xr.open_dataset(lp)
    lats = ds["flash_lat"].values
    lons = ds["flash_lon"].values
    total_flashes += len(lats)
    mask = (
        (lats > MIN_LAT) & (lats < MAX_LAT) & (lons > MIN_LON) & (lons < MAX_LON)
    )
    panhandle_flashes += mask.sum()
    ds.close()

print(f"\nTotal flashes across full hour (whole GOES-16 disk): {total_flashes}")
print(f"Flashes inside panhandle bounding box during this hour: {panhandle_flashes}")
