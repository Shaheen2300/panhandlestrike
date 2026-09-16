"""Land-cover fractions per block group from NLCD 2021 (30 m), via zonal
statistics. Adds developed / forest / wetland share as features - urban land
can locally enhance afternoon convection, so this is information the current
feature set (all population / geography / calendar) does not carry.

Output: data/processed/features/blockgroup_nlcd.csv
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats

TIF = "data/raw/nlcd/nlcd_2021_panhandle.tif"
BG = "data/processed/census/panhandle_block_groups_population.geojson"
OUT = "data/processed/features/blockgroup_nlcd.csv"

DEVELOPED = {21, 22, 23, 24}
FOREST = {41, 42, 43}
WETLAND = {90, 95}

bg = gpd.read_file(BG).to_crs("EPSG:5070")  # match NLCD Albers

stats = zonal_stats(bg, TIF, categorical=True, nodata=0, geojson_out=False)

rows = []
for geoid, s in zip(bg["GEOID"], stats):
    total = sum(s.values()) if s else 0
    if total == 0:
        rows.append((geoid, np.nan, np.nan, np.nan, np.nan))
        continue
    dev = sum(v for k, v in s.items() if k in DEVELOPED) / total
    forest = sum(v for k, v in s.items() if k in FOREST) / total
    wet = sum(v for k, v in s.items() if k in WETLAND) / total
    # dominant class code
    dom = max(s, key=s.get)
    rows.append((geoid, dev, forest, wet, int(dom)))

out = pd.DataFrame(rows, columns=["GEOID", "pct_developed", "pct_forest", "pct_wetland", "nlcd_dominant"])
# a couple of block groups can be all-nodata (tiny slivers over water); fill with study medians
for c in ["pct_developed", "pct_forest", "pct_wetland"]:
    out[c] = out[c].fillna(out[c].median())
out["nlcd_dominant"] = out["nlcd_dominant"].fillna(-1).astype(int)

out.to_csv(OUT, index=False)
print(f"Saved {len(out)} rows to {OUT}")
print(out[["pct_developed", "pct_forest", "pct_wetland"]].describe().round(3))
print("\nDominant NLCD class counts:")
print(out["nlcd_dominant"].value_counts())
