"""Breaks the per-file pipeline into separate phases to find out whether the
flat ~3-4 files/sec throughput (unchanged across 20/100/200 threads and
default/200-size connection pools) is a network-bound problem or an
HDF5-parse-bound problem.

Phase A: download all 180 files' raw bytes concurrently (threads), nothing
         else - pure network I/O, should benefit from concurrency if the
         bottleneck is network wait.
Phase B: parse the already-downloaded bytes sequentially (single thread) -
         baseline parse cost with zero concurrency.
Phase C: parse the same already-downloaded bytes again, but through a
         thread pool - if HDF5/h5py serializes internally (global lock),
         this should show no real speedup over Phase B despite N threads.
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import s3fs
import xarray as xr

BUCKET = "noaa-goes16"
PRODUCT = "GLM-L2-LCFA"
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 29.5, 31.5, -88.0, -84.5

YEAR, DOY, HOUR = 2023, 228, 18  # 2023-08-16, 18 UTC - fresh hour, not used before

fs = s3fs.S3FileSystem(anon=True)
prefix = f"{BUCKET}/{PRODUCT}/{YEAR}/{DOY:03d}/{HOUR:02d}/"
remote_files = fs.ls(prefix)
n = len(remote_files)
print(f"Hour: {YEAR}-{DOY:03d} {HOUR:02d} UTC | files: {n}\n")


def download_one(path):
    with fs.open(path, "rb") as f:
        return f.read()


def parse_one(data):
    ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
    lats = ds["flash_lat"].values
    lons = ds["flash_lon"].values
    total = len(lats)
    mask = (lats > MIN_LAT) & (lats < MAX_LAT) & (lons > MIN_LON) & (lons < MAX_LON)
    panhandle = int(mask.sum())
    ds.close()
    return total, panhandle


# --- Phase A: download only, concurrent ---
t0 = time.time()
blobs = [None] * n
with ThreadPoolExecutor(max_workers=100) as ex:
    futures = {ex.submit(download_one, p): i for i, p in enumerate(remote_files)}
    for fut in as_completed(futures):
        i = futures[fut]
        blobs[i] = fut.result()
download_elapsed = time.time() - t0
download_rate = n / download_elapsed
print(f"Phase A - download only (100 threads): {download_elapsed:.1f}s | {download_rate:.1f} files/sec")

# --- Phase B: parse only, sequential ---
t0 = time.time()
total_flashes = 0
panhandle_flashes = 0
for blob in blobs:
    total, panhandle = parse_one(blob)
    total_flashes += total
    panhandle_flashes += panhandle
parse_seq_elapsed = time.time() - t0
parse_seq_rate = n / parse_seq_elapsed
print(f"Phase B - parse only, sequential (1 thread): {parse_seq_elapsed:.1f}s | {parse_seq_rate:.1f} files/sec")
print(f"  total_flashes={total_flashes} panhandle={panhandle_flashes}")

# --- Phase C: parse only, threaded ---
t0 = time.time()
total_flashes_c = 0
panhandle_flashes_c = 0
with ThreadPoolExecutor(max_workers=100) as ex:
    futures = [ex.submit(parse_one, blob) for blob in blobs]
    for fut in as_completed(futures):
        total, panhandle = fut.result()
        total_flashes_c += total
        panhandle_flashes_c += panhandle
parse_thread_elapsed = time.time() - t0
parse_thread_rate = n / parse_thread_elapsed
print(f"Phase C - parse only, threaded (100 threads): {parse_thread_elapsed:.1f}s | {parse_thread_rate:.1f} files/sec")
print(f"  total_flashes={total_flashes_c} panhandle={panhandle_flashes_c}")

print("\n=== Summary ===")
print(f"Download (concurrent):  {download_rate:6.1f} files/sec")
print(f"Parse (sequential):     {parse_seq_rate:6.1f} files/sec")
print(f"Parse (100 threads):    {parse_thread_rate:6.1f} files/sec")
speedup = parse_thread_rate / parse_seq_rate
print(f"Parse threading speedup: {speedup:.2f}x")
