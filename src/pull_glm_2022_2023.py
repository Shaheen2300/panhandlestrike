"""GLM pull for May-Sep 2022 and 2023, to extend the training window to 4
storm seasons spanning genuinely different ENSO states (2022 La Nina, 2023
transitioning to El Nino, 2024-25 near-neutral). Same 16-process pool, same
panhandle bbox, as pull_glm_data.py / pull_glm_2026.py.

Satellite confirmed empirically: GOES-16 was the operational GOES-East
satellite for both 2022 and 2023 (GOES-19 did not exist until 2024 and was not
operational until April 2025) - both years pull from noaa-goes16.
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
BUCKET = "noaa-goes16"
YEARS = [2022, 2023]
MONTHS = range(5, 10)
N_WORKERS = 16

OUT_DIR = "data/processed/glm"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "glm_pull_2022_2023_log.txt")

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


def list_month_files(year, month):
    fs = s3fs.S3FileSystem(anon=True)
    _, ndays = calendar.monthrange(year, month)
    files = []
    for day in range(1, ndays + 1):
        doy = date(year, month, day).timetuple().tm_yday
        try:
            files.extend(sorted(fs.find(f"{BUCKET}/{PRODUCT}/{year}/{doy:03d}/")))
        except FileNotFoundError:
            pass
    return files


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    t0 = time.time()
    total_flashes = total_files = total_errors = 0
    log(f"=== GLM pull starting: May-Sep 2022 & 2023, {BUCKET} ===")

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        for year in YEARS:
            for month in MONTHS:
                out_path = os.path.join(OUT_DIR, f"glm_panhandle_{year}_{month:02d}.parquet")
                if os.path.exists(out_path):
                    log(f"SKIP {year}-{month:02d} (already done)")
                    continue
                log(f"Listing {year}-{month:02d}...")
                files = list_month_files(year, month)
                log(f"  {len(files)} files for {year}-{month:02d}")

                m0 = time.time()
                rows = {k: [] for k in ["flash_id", "flash_lat", "flash_lon", "flash_area", "flash_energy", "timestamp"]}
                n_err = n_done = 0
                for fut in as_completed([ex.submit(process_file, p) for p in files]):
                    res = fut.result()
                    n_done += 1
                    if res is None:
                        pass
                    elif "error" in res:
                        n_err += 1
                    else:
                        n = len(res["flash_id"])
                        for k in ("flash_id", "flash_lat", "flash_lon", "flash_area", "flash_energy"):
                            rows[k].extend(res[k])
                        rows["timestamp"].extend([res["timestamp"]] * n)
                    if n_done % 20000 == 0:
                        log(f"  {year}-{month:02d}: {n_done}/{len(files)} files, {len(rows['flash_id'])} flashes, {n_err} err")

                df = pd.DataFrame(rows)
                if len(df):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.to_parquet(out_path, index=False)
                log(f"DONE {year}-{month:02d}: {len(files)} files | {len(df)} flashes | {n_err} err | "
                    f"{(time.time() - m0) / 3600:.2f}h -> {out_path}")
                total_flashes += len(df); total_files += len(files); total_errors += n_err

    log(f"\n=== DONE === files {total_files} | flashes {total_flashes} | errors {total_errors} | "
        f"{(time.time() - t0) / 3600:.2f}h")


if __name__ == "__main__":
    main()
