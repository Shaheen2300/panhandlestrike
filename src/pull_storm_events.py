"""Pull NOAA Storm Events Database details files (2016-2025) and filter to
the 5 Florida panhandle counties in scope for this project, to see the real
row count, real variables, and whether this dataset alone gets anywhere
near 1000+ observations before deciding how it fits into the project.
"""

import gzip
import io
import os
import urllib.request

import pandas as pd

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FILES = {
    2016: "StormEvents_details-ftp_v1.0_d2016_c20260323.csv.gz",
    2017: "StormEvents_details-ftp_v1.0_d2017_c20260519.csv.gz",
    2018: "StormEvents_details-ftp_v1.0_d2018_c20260323.csv.gz",
    2019: "StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz",
    2020: "StormEvents_details-ftp_v1.0_d2020_c20260323.csv.gz",
    2021: "StormEvents_details-ftp_v1.0_d2021_c20260323.csv.gz",
    2022: "StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz",
    2023: "StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz",
    2024: "StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz",
    2025: "StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz",
}

PANHANDLE_COUNTIES = {"ESCAMBIA", "SANTA ROSA", "OKALOOSA", "WALTON", "BAY"}

LOCAL_DIR = "data/raw/storm_events"
os.makedirs(LOCAL_DIR, exist_ok=True)

all_frames = []
for year, filename in FILES.items():
    local_path = os.path.join(LOCAL_DIR, filename)
    if not os.path.exists(local_path):
        req = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)

    with gzip.open(local_path, "rt", encoding="latin-1") as f:
        df = pd.read_csv(f, low_memory=False)

    fl = df[df["STATE"] == "FLORIDA"]
    panhandle = fl[fl["CZ_NAME"].isin(PANHANDLE_COUNTIES)]
    print(f"{year}: {len(df)} total rows | {len(fl)} FL rows | {len(panhandle)} panhandle-county rows")
    all_frames.append(panhandle)

combined = pd.concat(all_frames, ignore_index=True)
print(f"\nTotal panhandle rows (2016-2025): {len(combined)}")
print(f"\nColumns ({len(combined.columns)}):")
print(list(combined.columns))

print("\nEvent type breakdown:")
print(combined["EVENT_TYPE"].value_counts())

print("\nCounty breakdown:")
print(combined["CZ_NAME"].value_counts())

lightning_only = combined[combined["EVENT_TYPE"].str.contains("Lightning", case=False, na=False)]
print(f"\nLightning-tagged events only: {len(lightning_only)}")

combined.to_csv(os.path.join(LOCAL_DIR, "panhandle_storm_events_2016_2025.csv"), index=False)
print(f"\nSaved combined filtered file to {LOCAL_DIR}/panhandle_storm_events_2016_2025.csv")
