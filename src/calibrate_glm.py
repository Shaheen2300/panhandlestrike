"""Calibration pass: sample a handful of hours across different conditions
(overnight, shoulder-season, different days) to see how much panhandle flash
volume actually varies, before locking in a storage/runtime estimate for the
full 2017-2025 batch pull.

Reads each file straight from S3 into memory (h5netcdf + BytesIO) - nothing
is ever written to disk here, matching the design for the real batch pull.
Files within an hour are fetched concurrently since each is small (~300-450KB)
and independent.
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import s3fs
import xarray as xr

BUCKET = "noaa-goes16"
PRODUCT = "GLM-L2-LCFA"
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 29.5, 31.5, -88.0, -84.5

# label, year, day-of-year, hour(UTC)
SAMPLES = [
    ("Overnight, peak season (2023-08-15 06 UTC / ~2am EDT)", 2023, 227, 6),
    ("May shoulder-season afternoon (2023-05-15 20 UTC / ~4pm EDT)", 2023, 135, 20),
    ("May shoulder-season overnight (2023-05-15 06 UTC / ~2am EDT)", 2023, 135, 6),
    ("Different August afternoon (2023-08-01 20 UTC / ~4pm EDT)", 2023, 213, 20),
    ("September shoulder-season afternoon (2023-09-20 20 UTC / ~4pm EDT)", 2023, 263, 20),
]

fs = s3fs.S3FileSystem(anon=True)


def process_file(remote_path):
    with fs.open(remote_path, "rb") as f:
        data = f.read()
    ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
    lats = ds["flash_lat"].values
    lons = ds["flash_lon"].values
    total = len(lats)
    mask = (lats > MIN_LAT) & (lats < MAX_LAT) & (lons > MIN_LON) & (lons < MAX_LON)
    panhandle = int(mask.sum())
    ds.close()
    return total, panhandle


results = []
for label, year, doy, hour in SAMPLES:
    prefix = f"{BUCKET}/{PRODUCT}/{year}/{doy:03d}/{hour:02d}/"
    remote_files = fs.ls(prefix)

    t0 = time.time()
    total_flashes = 0
    panhandle_flashes = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(process_file, rf) for rf in remote_files]
        for fut in as_completed(futures):
            total, panhandle = fut.result()
            total_flashes += total
            panhandle_flashes += panhandle
    elapsed = time.time() - t0

    print(f"\n{label}")
    print(f"  files: {len(remote_files)} | elapsed: {elapsed:.1f}s")
    print(f"  total flashes (full disk): {total_flashes}")
    print(f"  panhandle flashes: {panhandle_flashes}")
    results.append((label, len(remote_files), total_flashes, panhandle_flashes, elapsed))

print("\n=== Summary ===")
for label, nfiles, total, panhandle, elapsed in results:
    print(f"{panhandle:6d} panhandle flashes | {elapsed:5.1f}s | {label}")
