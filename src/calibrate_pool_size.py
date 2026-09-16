"""Same isolated single-hour throughput test as calibrate_concurrency.py,
but with the underlying botocore connection pool explicitly raised via
config_kwargs, since the flat 20/100/200-thread results suggested threads
were queuing behind a small default pool (max_pool_connections=10) rather
than actually hitting S3 concurrently.
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import s3fs
import xarray as xr

BUCKET = "noaa-goes16"
PRODUCT = "GLM-L2-LCFA"
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 29.5, 31.5, -88.0, -84.5

YEAR, DOY, HOUR = 2023, 227, 22  # 2023-08-15, 22 UTC - fresh hour
THREAD_COUNTS = [100, 200]
POOL_SIZE = 200

fs = s3fs.S3FileSystem(anon=True, config_kwargs={"max_pool_connections": POOL_SIZE})


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


prefix = f"{BUCKET}/{PRODUCT}/{YEAR}/{DOY:03d}/{HOUR:02d}/"
remote_files = fs.ls(prefix)
print(f"Hour: {YEAR}-{DOY:03d} {HOUR:02d} UTC | files: {len(remote_files)} | pool_size={POOL_SIZE}\n")

TOTAL_FILES_FULL_PULL = 5_948_640  # 2017-2025 storm season, all hours, 180 files/hr

for workers in THREAD_COUNTS:
    t0 = time.time()
    total_flashes = 0
    panhandle_flashes = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_file, rf) for rf in remote_files]
        for fut in as_completed(futures):
            total, panhandle = fut.result()
            total_flashes += total
            panhandle_flashes += panhandle
    elapsed = time.time() - t0
    rate = len(remote_files) / elapsed

    est_seconds = TOTAL_FILES_FULL_PULL / rate
    est_hours = est_seconds / 3600
    est_days = est_hours / 24

    print(f"workers={workers:4d} | elapsed={elapsed:5.1f}s | rate={rate:5.1f} files/sec "
          f"| total_flashes={total_flashes} | panhandle={panhandle_flashes}")
    print(f"  -> full-pull estimate at this rate: {est_hours:.1f} hours (~{est_days:.1f} days)\n")
