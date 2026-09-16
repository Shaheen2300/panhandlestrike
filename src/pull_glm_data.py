"""Full GLM historical pull: May-Sep 2024 and May-Sep 2025, filtered to the
Florida panhandle bounding box, using the validated 16-process persistent
pool approach (threading was confirmed lock-serialized and counterproductive
for the HDF5 parse step; see src/calibrate_*.py for the benchmarking work
that led here).

Only the filtered per-flash fields are ever kept in memory/disk - raw
NetCDF bytes are read, parsed, and discarded per file. Output is chunked
by month (not one giant file) so a multi-hour run is resumable: if this
script is re-run, months that already have an output parquet are skipped.

Note: GOES-16 was retired as the operational GOES-East satellite and
replaced by GOES-19 around April 2025 (confirmed empirically - GOES-16 has
no GLM data on S3 past early April 2025). 2024 data comes from noaa-goes16,
2025 data comes from noaa-goes19.
"""

import calendar
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import pandas as pd
import s3fs
import xarray as xr

MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 29.5, 31.5, -88.0, -84.5
PRODUCT = "GLM-L2-LCFA"
MONTHS = range(5, 10)  # May-Sep
YEAR_BUCKETS = {2024: "noaa-goes16", 2025: "noaa-goes19"}
N_WORKERS = 16

OUT_DIR = "data/processed/glm"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "glm_pull_log.txt")

_worker_fs = None


def _init_worker():
    global _worker_fs
    _worker_fs = s3fs.S3FileSystem(anon=True)


def process_file(remote_path):
    try:
        with _worker_fs.open(remote_path, "rb") as f:
            data = f.read()
        ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
        lats = ds["flash_lat"].values
        lons = ds["flash_lon"].values
        mask = (lats > MIN_LAT) & (lats < MAX_LAT) & (lons > MIN_LON) & (lons < MAX_LON)
        if not mask.any():
            ds.close()
            return None
        result = {
            "flash_id": ds["flash_id"].values[mask].tolist(),
            "flash_lat": lats[mask].tolist(),
            "flash_lon": lons[mask].tolist(),
            "flash_area": ds["flash_area"].values[mask].tolist(),
            "flash_energy": ds["flash_energy"].values[mask].tolist(),
            "timestamp": ds.attrs.get("time_coverage_start"),
        }
        ds.close()
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "path": remote_path}


def list_month_files(bucket, year, month):
    fs = s3fs.S3FileSystem(anon=True)
    _, ndays = calendar.monthrange(year, month)
    files = []
    for day in range(1, ndays + 1):
        doy = date(year, month, day).timetuple().tm_yday
        prefix = f"{bucket}/{PRODUCT}/{year}/{doy:03d}/"
        try:
            files.extend(sorted(fs.find(prefix)))
        except FileNotFoundError:
            pass
    return files


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    overall_start = time.time()
    total_flash_count = 0
    total_files_processed = 0
    total_errors = 0

    log("=== GLM panhandle pull starting: May-Sep 2024, May-Sep 2025 ===")

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        for year in [2024, 2025]:
            bucket = YEAR_BUCKETS[year]
            for month in MONTHS:
                out_path = os.path.join(OUT_DIR, f"glm_panhandle_{year}_{month:02d}.parquet")
                if os.path.exists(out_path):
                    existing = pd.read_parquet(out_path)
                    log(f"SKIP {year}-{month:02d} (already done, {len(existing)} flashes)")
                    total_flash_count += len(existing)
                    continue

                log(f"Listing files for {year}-{month:02d} (bucket={bucket})...")
                files = list_month_files(bucket, year, month)
                log(f"  {len(files)} files to process for {year}-{month:02d}")

                month_start = time.time()
                rows = {k: [] for k in ["flash_id", "flash_lat", "flash_lon", "flash_area", "flash_energy", "timestamp"]}
                n_errors = 0
                n_done = 0

                futures = [ex.submit(process_file, p) for p in files]
                for fut in as_completed(futures):
                    res = fut.result()
                    n_done += 1
                    if res is None:
                        pass
                    elif "error" in res:
                        n_errors += 1
                        if n_errors <= 5:
                            log(f"  ERROR on {res['path']}: {res['error']}")
                    else:
                        n = len(res["flash_id"])
                        rows["flash_id"].extend(res["flash_id"])
                        rows["flash_lat"].extend(res["flash_lat"])
                        rows["flash_lon"].extend(res["flash_lon"])
                        rows["flash_area"].extend(res["flash_area"])
                        rows["flash_energy"].extend(res["flash_energy"])
                        rows["timestamp"].extend([res["timestamp"]] * n)

                    if n_done % 20000 == 0:
                        elapsed = time.time() - month_start
                        rate = n_done / elapsed
                        log(f"  progress {year}-{month:02d}: {n_done}/{len(files)} files "
                            f"({rate:.1f} files/sec) | {len(rows['flash_id'])} flashes so far | {n_errors} errors")

                df = pd.DataFrame(rows)
                if len(df):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.to_parquet(out_path, index=False)
                month_elapsed = time.time() - month_start

                log(f"DONE {year}-{month:02d}: {len(files)} files | {len(df)} flashes | "
                    f"{n_errors} errors | {month_elapsed/3600:.2f}h | saved to {out_path}")

                total_flash_count += len(df)
                total_files_processed += len(files)
                total_errors += n_errors

    overall_elapsed = time.time() - overall_start
    log("\n=== ALL MONTHS DONE ===")
    log(f"Total files processed this run: {total_files_processed}")
    log(f"Total flashes (all months, including skipped): {total_flash_count}")
    log(f"Total errors: {total_errors}")
    log(f"Total runtime this run: {overall_elapsed/3600:.2f} hours")

    all_paths = sorted(
        os.path.join(OUT_DIR, f)
        for f in os.listdir(OUT_DIR)
        if f.startswith("glm_panhandle_") and f.endswith(".parquet")
    )
    total_size_mb = sum(os.path.getsize(p) for p in all_paths) / (1024 * 1024)
    log(f"Total parquet size: {total_size_mb:.1f} MB across {len(all_paths)} monthly files")

    # Sample CSV from the earliest month for quick visual inspection
    first_df = pd.read_parquet(all_paths[0]).sort_values("timestamp")
    sample_path = os.path.join(OUT_DIR, "glm_panhandle_sample_1000.csv")
    first_df.head(1000).to_csv(sample_path, index=False)
    log(f"Sample CSV (first 1000 rows of {os.path.basename(all_paths[0])}) written to {sample_path}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
