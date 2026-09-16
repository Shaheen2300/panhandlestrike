"""Multiprocessing-based parse test: same isolated single-hour workload, but
using a persistent ProcessPoolExecutor (worker processes created once via an
initializer, then reused across all 180 files - not spawned per file) since
threading was shown to be lock-serialized and actively counterproductive for
the HDF5 parse step.

Each worker process gets its own s3fs filesystem instance (set up once in
the initializer) and its own HDF5/h5py state, so there's no shared lock
across processes the way there was across threads.
"""

import io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import s3fs
import xarray as xr

BUCKET = "noaa-goes16"
PRODUCT = "GLM-L2-LCFA"
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 29.5, 31.5, -88.0, -84.5
YEAR, DOY, HOUR = 2023, 229, 18  # 2023-08-17, 18 UTC - fresh hour, not used before
TOTAL_FILES_FULL_PULL = 5_948_640  # 2017-2025 storm season, all hours, 180 files/hr

_worker_fs = None


def _init_worker():
    global _worker_fs
    _worker_fs = s3fs.S3FileSystem(anon=True)


def process_file(remote_path):
    with _worker_fs.open(remote_path, "rb") as f:
        data = f.read()
    ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
    lats = ds["flash_lat"].values
    lons = ds["flash_lon"].values
    total = len(lats)
    mask = (lats > MIN_LAT) & (lats < MAX_LAT) & (lons > MIN_LON) & (lons < MAX_LON)
    panhandle = int(mask.sum())
    ds.close()
    return total, panhandle


def run_test(remote_files, workers):
    t0 = time.time()
    total_flashes = 0
    panhandle_flashes = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futures = [ex.submit(process_file, rf) for rf in remote_files]
        for fut in as_completed(futures):
            total, panhandle = fut.result()
            total_flashes += total
            panhandle_flashes += panhandle
    elapsed = time.time() - t0
    rate = len(remote_files) / elapsed
    return elapsed, rate, total_flashes, panhandle_flashes


if __name__ == "__main__":
    lister = s3fs.S3FileSystem(anon=True)
    prefix = f"{BUCKET}/{PRODUCT}/{YEAR}/{DOY:03d}/{HOUR:02d}/"
    remote_files = lister.ls(prefix)
    n = len(remote_files)
    print(f"Hour: {YEAR}-{DOY:03d} {HOUR:02d} UTC | files: {n}\n")

    baseline_rate = None
    for workers in [8, 16, 28]:
        elapsed, rate, total, panhandle = run_test(remote_files, workers)
        est_hours = TOTAL_FILES_FULL_PULL / rate / 3600
        est_days = est_hours / 24

        if baseline_rate is None:
            baseline_rate = rate
            scaling_note = " (baseline)"
        else:
            ideal_rate = baseline_rate * (workers / 8)
            efficiency = rate / ideal_rate * 100
            scaling_note = f" | vs ideal linear scaling from 8-worker baseline: {efficiency:.0f}% efficient"

        print(f"workers={workers:3d} | elapsed={elapsed:5.1f}s | rate={rate:6.1f} files/sec{scaling_note}")
        print(f"  total_flashes={total} panhandle={panhandle}")
        print(f"  -> full-pull estimate: {est_hours:.1f}h (~{est_days:.2f} days)\n")
