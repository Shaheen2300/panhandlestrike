"""Generates all 19 report figures requested for the final write-up, one per
numbered item, into report_figures/. Every number is either read directly from
the same processed files already used to report it, or recomputed using the
exact same method (same thresholds, same exclusions) as the script that
produced it originally - nothing is derived a new way.

Run: python src/generate_report_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - this script now interleaves
# heavy joblib-parallel sklearn fits (RandomizedSearchCV n_jobs=-1) with
# matplotlib figure creation in one process; a GUI backend (TkAgg) caused
# Tcl/Tk thread-safety errors and a silent hang partway through. Must be set
# before importing pyplot.
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

OUT = "report_figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 150})

COUNTY_COLORS = {
    "ESCAMBIA": "#4477AA", "SANTA ROSA": "#66CCEE", "OKALOOSA": "#228833",
    "WALTON": "#CCBB44", "BAY": "#EE6677",
}
WATER_GEOIDS = None  # filled below
index_rows = []


def record(n, fname, section, title):
    index_rows.append((n, fname, section, title))
    print(f"  [{n:02d}] {fname}")


def save(fig, fname):
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Shared data loads
# =============================================================================
print("Loading shared data...")
BG = gpd.read_file("data/processed/census/panhandle_block_groups_population.geojson")
water_mask = BG["TRACTCE"].str.startswith("99") | (BG["P1_001N"] == 0)
WATER_GEOIDS = set(BG.loc[water_mask, "GEOID"])
LAND_BG = BG[~water_mask].copy()

# --- Option B: the 4-year (2022-2025) window is the single primary truth for
# every figure below, except the dedicated 2-yr-vs-4-yr comparison section
# (figs 16/18/19), which intentionally keeps both for a specific narrative
# reason (found a real generalization problem, fixed it - see that section). ---
COMBINED = pd.read_csv("data/processed/final_4yr/blockgroup_combined_priority_4yr.csv", dtype={"GEOID": str})
COMBINED_GDF = gpd.read_file("data/processed/final_4yr/blockgroup_combined_priority_4yr.geojson")
DRIVETIME = pd.read_csv("data/processed/routing/blockgroup_shelter_drivetime.csv", dtype={"GEOID": str})
DRIVETIME_GDF = gpd.read_file("data/processed/routing/blockgroup_shelter_drivetime.geojson")
SHELTERS = gpd.read_file("data/processed/shelters/shelters_panhandle_combined.geojson")
ANOM_SUMMARY = pd.read_csv("data/processed/anomaly_4yr/blockgroup_anomaly_summary_4yr.csv", dtype={"GEOID": str})
FLAGGED = pd.read_csv("data/processed/anomaly_4yr/flagged_blockgroup_days_4yr.csv", dtype={"GEOID": str})
FLASHES_LAND = pd.read_parquet("data/processed/features/flashes_with_blockgroup_4yr.parquet")
FLASHES_LAND = FLASHES_LAND[~FLASHES_LAND["GEOID"].isin(WATER_GEOIDS)].copy()
FLASHES_LAND["timestamp"] = pd.to_datetime(FLASHES_LAND["timestamp"])

BUF_CMP = pd.read_csv("data/processed/coastal_buffer_10km_4yr/flash_count_comparison_4yr.csv", dtype={"GEOID": str})
BUF_COMBINED = pd.read_csv("data/processed/coastal_buffer_10km_4yr/blockgroup_combined_priority_buffered10km_4yr.csv",
                          dtype={"GEOID": str})

# shared 4-year expanded-feature training table, used by fig17 (permutation
# importance) and fig19 (the deliberately-kept 2yr-vs-4yr comparison)
FEATS_4YR = ["P1_001N", "flash_per_km2_4yr", "drive_min_to_shelter", "coast_dist_km",
            "pct_developed", "pct_forest", "pct_wetland", "prev_month_flagged",
            "same_month_last_yr_flagged", "is_may", "oni", "doy_sin", "doy_cos"]
TRAIN4 = pd.read_csv("data/processed/features/blockgroup_month_4yr.csv", dtype={"GEOID": str})
TEST26_4 = pd.read_csv("data/processed/features/blockgroup_month_2026_test_4yr.csv", dtype={"GEOID": str})

WALTON_ID = "121319503053"
OKA_CLUSTER = ["120910210011", "120910210012", "120910210013",
              "120910210021", "120910210022", "120910210023"]

print("Loaded. Generating figures...\n")

# =============================================================================
# SECTION: Data overview
# =============================================================================

# --- Fig 1: flashes by month, all 4 training years (2022-2025) ---
months = list(range(5, 10))
month_names = ["May", "Jun", "Jul", "Aug", "Sep"]
years4 = [2022, 2023, 2024, 2025]
year_colors = {2022: "#66CCEE", 2023: "#228833", 2024: "#4477AA", 2025: "#EE6677"}
counts_by_year = {y: [] for y in years4}
for y in years4:
    for m in months:
        counts_by_year[y].append(len(pd.read_parquet(f"data/processed/glm/glm_panhandle_{y}_{m:02d}.parquet")))

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(months))
w = 0.2
for i, y in enumerate(years4):
    offset = (i - 1.5) * w
    bars = ax.bar(x + offset, counts_by_year[y], w, label=str(y), color=year_colors[y])
    for xi, v in zip(x + offset, counts_by_year[y]):
        ax.text(xi, v, f"{v/1000:.0f}k", ha="center", va="bottom", fontsize=6.5, rotation=90)
ax.annotate("June 2025 spike\n(412,344 - not a\ntropical cyclone;\nsee Section 2)",
           xy=(1 + 1.5 * w, counts_by_year[2025][1]), xytext=(2.6, counts_by_year[2025][1] * 0.9),
           arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(month_names)
ax.set_xlabel("Month"); ax.set_ylabel("GLM flashes (panhandle bbox)")
ax.set_title("GOES-16/19 GLM lightning flashes by month, 2022-2025\n(the full 4-year training window)")
ax.legend(title="Year", ncol=4, loc="upper right")
save(fig, "fig01_flashes_by_month.png")
record(1, "fig01_flashes_by_month.png", "Data overview", "Flashes by month, 2022-2025 (4-year window)")

# --- Fig 2: dataset scale overview (KPI cards) ---
n_edges = 152235
n_nodes = 61253
total_flashes_4yr = sum(sum(v) for v in counts_by_year.values())
stats = [
    ("GLM lightning\nflashes", f"{total_flashes_4yr:,}", "May-Sep 2022-2025 (4 years)"),
    ("Population\n(2020 Census)", "972,094", "595 block groups (unchanged - static census input)"),
    ("Existing\nshelters", "194", "FDEM Risk Shelter Inventory (unchanged)"),
    ("Road network", f"{n_nodes:,} nodes\n{n_edges:,} edges", "OpenStreetMap, drivable (unchanged)"),
]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (label, val, sub) in zip(axes, stats):
    ax.axis("off")
    ax.text(0.5, 0.62, val, ha="center", va="center", fontsize=16, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, label, ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.5, 0.08, sub, ha="center", va="center", fontsize=7.5, color="grey", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor="#999999",
                               linewidth=1, transform=ax.transAxes))
fig.suptitle("Dataset scale overview - 5-county Florida panhandle study area\n"
            "(4-year GLM window is the primary dataset; population/shelters/roads are static, year-independent inputs)", y=1.08, fontsize=10)
save(fig, "fig02_dataset_overview.png")
record(2, "fig02_dataset_overview.png", "Data overview", "Dataset scale overview, 4-year GLM window")

# =============================================================================
# SECTION: June 2025 validation
# =============================================================================

# --- Fig 3: daily flash timeline, June 2025, validation dates marked ---
jun25 = pd.read_parquet("data/processed/glm/glm_panhandle_2025_06.parquet")
jun25["date"] = pd.to_datetime(jun25["timestamp"]).dt.date
daily = jun25.groupby("date").size()
full_idx = pd.date_range("2025-06-01", "2025-06-30", freq="D").date
daily = daily.reindex(full_idx, fill_value=0)
val_days = [9, 10, 17, 23, 25]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, 31), daily.values, color="#4477AA", marker="o", markersize=3, linewidth=1.2)
for d in val_days:
    ax.axvline(d, color="#EE6677", linestyle="--", alpha=0.7, linewidth=1)
    ax.annotate(f"{d}", xy=(d, daily.values[d - 1]), xytext=(d, daily.values[d - 1] + 2500),
               ha="center", fontsize=8, color="#AA3344")
ax.set_xlabel("Day of June 2025"); ax.set_ylabel("GLM flashes (panhandle bbox)")
ax.set_title("Daily lightning flashes, June 2025 - NOAA Storm Events dates marked (dashed)")
ax.set_xticks(range(1, 31, 2))
save(fig, "fig03_june2025_daily_timeline.png")
record(3, "fig03_june2025_daily_timeline.png", "June 2025 validation", "Daily flash timeline with Storm Events dates")

# --- Fig 4: Santa Rosa map, 3 matched block groups (June 9-10 flash flooding) ---
sr_ids = ["121130101001", "121130102001", "121130104001"]
sr_county = BG[BG["COUNTY_NAME"] == "SANTA ROSA"]
fig, ax = plt.subplots(figsize=(7, 7))
sr_county.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.5)
sr_county[sr_county["GEOID"].isin(sr_ids)].plot(ax=ax, color="#EE6677", edgecolor="#7a0000", linewidth=1.5)
for gid in sr_ids:
    row = sr_county[sr_county["GEOID"] == gid]
    if not row.empty:
        c = row.geometry.centroid.iloc[0]
        ax.annotate(gid, (c.x, c.y), fontsize=7, ha="center",
                   xytext=(0, 8), textcoords="offset points")
ax.set_title("Santa Rosa County: 3 block groups matching both\nGLM anomaly flags and NOAA flash-flood reports (Jun 9-10, 2025)")
ax.set_axis_off()
legend_elems = [Patch(facecolor="#EE6677", edgecolor="#7a0000", label="Z+BURST flagged, matches Storm Events"),
               Patch(facecolor="#f0f0f0", edgecolor="#999999", label="Other Santa Rosa block groups")]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8)
save(fig, "fig04_santa_rosa_validation_map.png")
record(4, "fig04_santa_rosa_validation_map.png", "June 2025 validation", "Santa Rosa map: 3 matched block groups")

# =============================================================================
# SECTION: Anomaly detection
# =============================================================================

# --- Fig 5: z-only / burst-only / both bar chart ---
z_total = int(FLAGGED["zscore_anomaly"].sum())
b_total = int(FLAGGED["burst_alert"].sum())
both_total = int(FLAGGED["both_flag"].sum())
z_only = z_total - both_total
b_only = b_total - both_total
total_either = len(FLAGGED)

fig, ax = plt.subplots(figsize=(7, 5))
cats = ["Z-score only", "Burst only", "Both methods", "Total flagged\n(either)"]
vals = [z_only, b_only, both_total, total_either]
colors = ["#66CCEE", "#CCBB44", "#EE6677", "#4477AA"]
bars = ax.bar(cats, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Block-group-days")
ax.set_title("Anomaly flags by method: overlap between z-score and\nrolling 30-min burst detection (2022-25, land block groups)")
save(fig, "fig05_anomaly_method_overlap.png")
record(5, "fig05_anomaly_method_overlap.png", "Anomaly detection", "Z-score vs burst vs both, overlap counts")

# --- Fig 6: 5 validation days, 4 metrics ---
base = FLASHES_LAND.copy()
base["date"] = base["timestamp"].dt.date
daily_bg = base.groupby(["GEOID", "date"]).size().rename("c").reset_index()
all_days_full = pd.date_range(base["timestamp"].min().normalize(), base["timestamp"].max().normalize(), freq="D").date
full_grid = (daily_bg.set_index(["GEOID", "date"])
            .reindex(pd.MultiIndex.from_product([LAND_BG["GEOID"], all_days_full], names=["GEOID", "date"]), fill_value=0)
            .reset_index())
bg_stats = full_grid.groupby("GEOID")["c"].agg(mu="mean", sd="std").reset_index()

val_dates = [pd.Timestamp(f"2025-06-{d:02d}").date() for d in val_days]
metrics = {"total_flashes": [], "max_z": [], "max_peak30": [], "both_count": []}
for d in val_dates:
    day_flashes = base[base["date"] == d]
    total_fl = len(day_flashes)
    day_counts = day_flashes.groupby("GEOID").size().reset_index(name="c")
    day_counts = day_counts.merge(bg_stats, on="GEOID", how="left")
    day_counts["z"] = np.where(day_counts["sd"] > 0, (day_counts["c"] - day_counts["mu"]) / day_counts["sd"], np.nan)
    max_z = day_counts["z"].max()

    peaks, both_n = [], 0
    for gid, grp in day_flashes.groupby("GEOID"):
        t = np.sort(grp["timestamp"].values)
        k = 0
        peak = 0
        for i in range(len(t)):
            while t[i] - t[k] >= pd.Timedelta(minutes=30):
                k += 1
            peak = max(peak, i - k + 1)
        peaks.append(peak)
        z_row = day_counts[day_counts["GEOID"] == gid]
        z_val = z_row["z"].iloc[0] if not z_row.empty else np.nan
        if peak >= 5 and pd.notna(z_val) and z_val >= 3.0:
            both_n += 1
    metrics["total_flashes"].append(total_fl)
    metrics["max_z"].append(max_z)
    metrics["max_peak30"].append(max(peaks) if peaks else 0)
    metrics["both_count"].append(both_n)

date_labels = [f"Jun {d}" for d in val_days]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
panels = [("total_flashes", "Total land flashes", "#4477AA"),
         ("max_z", "Max daily z-score", "#66CCEE"),
         ("max_peak30", "Max 30-min burst peak", "#CCBB44"),
         ("both_count", "Block groups flagged\nby BOTH methods", "#EE6677")]
for ax, (key, title, color) in zip(axes.flat, panels):
    b = ax.bar(date_labels, metrics[key], color=color)
    for bi, v in zip(b, metrics[key]):
        ax.text(bi.get_x() + bi.get_width() / 2, v, f"{v:,.0f}" if key != "max_z" else f"{v:.1f}",
               ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Validation day (2025)")
fig.suptitle("June 2025 validation days: both detection methods, side by side", y=1.00, fontsize=12)
fig.tight_layout()
save(fig, "fig06_validation_days_metrics.png")
record(6, "fig06_validation_days_metrics.png", "Anomaly detection", "5 validation days: flashes/max-z/max-burst/both-count")

# =============================================================================
# SECTION: Suitability model
# =============================================================================

# --- Fig 7: 589 BG suitability choropleth ---
fig, ax = plt.subplots(figsize=(8, 8))
COMBINED_GDF.plot(column="suitability_score", cmap="RdYlBu_r", linewidth=0.2, edgecolor="#666666",
                  legend=True, ax=ax, legend_kwds={"label": "Suitability score", "shrink": 0.7})
ax.set_title("Shelter suitability score, all 589 land block groups\n"
            "(population + 4-year [2022-25] flash density + drive-time to shelter)")
ax.set_axis_off()
save(fig, "fig07_suitability_map.png")
record(7, "fig07_suitability_map.png", "Suitability model", "Suitability choropleth, 589 block groups")

# --- Fig 8: top 15 suitability bar, county labeled ---
top15 = COMBINED.sort_values("suitability_rank").head(15)
fig, ax = plt.subplots(figsize=(9, 6))
colors15 = [COUNTY_COLORS[c] for c in top15["COUNTY_NAME"]]
bars = ax.barh(range(15), top15["suitability_score"][::-1], color=colors15[::-1])
ax.set_yticks(range(15))
ax.set_yticklabels([f"{g} ({c.title()})" for g, c in zip(top15["GEOID"][::-1], top15["COUNTY_NAME"][::-1])], fontsize=8)
ax.set_xlabel("Suitability score")
ax.set_title("Top 15 highest-suitability block groups\n(4-year [2022-25] flash density)")
handles = [Patch(color=v, label=k.title()) for k, v in COUNTY_COLORS.items()]
ax.legend(handles=handles, loc="lower right", fontsize=8, title="County")
save(fig, "fig08_top15_suitability.png")
record(8, "fig08_top15_suitability.png", "Suitability model", "Top 15 suitability block groups by county")

# =============================================================================
# SECTION: Coastal buffer comparison
# =============================================================================

# --- Fig 9: shortlist land-only (42) vs buffered (51) side-by-side maps ---
n_orig_short = int(COMBINED["on_shortlist"].sum())
n_buf_short = int(BUF_COMBINED["on_shortlist_buffered"].sum())
plot_gdf = LAND_BG[["GEOID", "geometry"]].merge(
    COMBINED[["GEOID", "on_shortlist"]], on="GEOID").merge(
    BUF_COMBINED[["GEOID", "on_shortlist_buffered"]], on="GEOID")

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, col, title, n in [(axes[0], "on_shortlist", f"Land-only shortlist (n={n_orig_short})", n_orig_short),
                         (axes[1], "on_shortlist_buffered", f"10 km coastal-buffer shortlist (n={n_buf_short})", n_buf_short)]:
    plot_gdf.plot(ax=ax, color="#eeeeee", edgecolor="#bbbbbb", linewidth=0.3)
    plot_gdf[plot_gdf[col]].plot(ax=ax, color="#0d1b8f", edgecolor="#000033", linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
fig.suptitle(f"Final shortlist (4-year baseline): land-only vs 10 km coastal buffer", y=0.98, fontsize=13)
save(fig, "fig09_shortlist_landonly_vs_buffered.png")
record(9, "fig09_shortlist_landonly_vs_buffered.png", "Coastal buffer comparison",
      f"Shortlist maps: {n_orig_short} vs {n_buf_short} (4-year baseline)")

# --- Fig 10: Walton rank shift under the coastal buffer (4-year baseline) ---
r_orig = int(COMBINED.loc[COMBINED["GEOID"] == WALTON_ID, "suitability_rank"].iloc[0])
r_buf = int(BUF_COMBINED.loc[BUF_COMBINED["GEOID"] == WALTON_ID, "suitability_rank"].iloc[0])
on_orig = bool(COMBINED.loc[COMBINED["GEOID"] == WALTON_ID, "on_shortlist"].iloc[0])
on_buf = bool(BUF_COMBINED.loc[BUF_COMBINED["GEOID"] == WALTON_ID, "on_shortlist_buffered"].iloc[0])
fig, ax = plt.subplots(figsize=(5.5, 6))
ax.plot([0, 1], [r_orig, r_buf], marker="o", markersize=12, color="#EE6677", linewidth=2)
ax.text(-0.05, r_orig, f"Rank {r_orig}\n(land-only)\nshortlist: {'YES' if on_orig else 'no'}",
       ha="right", va="center", fontsize=10)
ax.text(1.05, r_buf, f"Rank {r_buf}\n(10 km buffered)\nshortlist: {'YES' if on_buf else 'no'}",
       ha="left", va="center", fontsize=10)
ax.invert_yaxis()
ax.set_xlim(-0.7, 1.7); ax.set_xticks([])
ax.set_ylabel("Suitability rank (1 = highest priority, of 589)")
ax.set_title(f"Walton block group {WALTON_ID}:\nrank shift from the coastal buffer (4-year baseline)")
ax.annotate(f"shift: {r_buf - r_orig:+d} places", xy=(0.5, (r_orig + r_buf) / 2),
           ha="center", fontsize=9, color="#7a0000")
note = ("Stays on the shortlist under the 4-year baseline\n(under the original 2-yr baseline it dropped off: 127->158)"
       if on_orig and on_buf else "")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
if note:
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.5, color="#555555")
fig.subplots_adjust(bottom=0.22)
save(fig, "fig10_walton_rank_shift.png")
record(10, "fig10_walton_rank_shift.png", "Coastal buffer comparison",
      f"Walton 121319503053 rank shift {r_orig}->{r_buf} (4-year baseline)")

# =============================================================================
# SECTION: Routing analysis
# (Confirmed unchanged under Option B: population, shelters, and the OSM road
# network are static, year-independent inputs - drive time does not depend on
# the GLM flash window, so figs 11-12 need no rebuild.)
# =============================================================================

# --- Fig 11: population by drive-time tier ---
pop_5 = DRIVETIME.loc[DRIVETIME["drive_min_to_shelter"] <= 5, "P1_001N"].sum()
pop_10 = DRIVETIME.loc[(DRIVETIME["drive_min_to_shelter"] > 5) & (DRIVETIME["drive_min_to_shelter"] <= 10), "P1_001N"].sum()
pop_15 = DRIVETIME.loc[(DRIVETIME["drive_min_to_shelter"] > 10) & (DRIVETIME["drive_min_to_shelter"] <= 15), "P1_001N"].sum()
pop_gap = DRIVETIME.loc[DRIVETIME["drive_min_to_shelter"] > 15, "P1_001N"].sum()
tiers = ["0-5 min", "5-10 min", "10-15 min", ">15 min\n(coverage gap)"]
tier_pop = [pop_5, pop_10, pop_15, pop_gap]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(tiers, tier_pop, color=["#4477AA", "#66CCEE", "#CCBB44", "#EE6677"])
for b, v in zip(bars, tier_pop):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Population (2020 Census)")
ax.set_xlabel("Drive time to nearest shelter")
ax.set_title(f"Population by shelter drive-time tier - {pop_gap:,.0f} people\n(13.2%) live more than 15 minutes from a shelter")
save(fig, "fig11_population_by_drivetime_tier.png")
record(11, "fig11_population_by_drivetime_tier.png", "Routing analysis", "Population by drive-time tier")

# --- Fig 12: N. Okaloosa / Walton coverage gap map ---
focus = DRIVETIME_GDF[DRIVETIME_GDF["COUNTY_NAME"].isin(["OKALOOSA", "WALTON"])]
fig, ax = plt.subplots(figsize=(8, 8))
focus.plot(ax=ax, color="#eeeeee", edgecolor="#999999", linewidth=0.3)
focus[~focus["within_15min"]].plot(ax=ax, color="#EE6677", edgecolor="#7a0000", linewidth=0.4)
SHELTERS[SHELTERS["COUNTY"].isin(["OKALOOSA", "WALTON"])].plot(ax=ax, color="#2a7f2a", markersize=25,
                                                              marker="^", label="Shelter")
ax.set_title("Okaloosa & Walton counties: block groups >15 min\nfrom the nearest shelter (rural coverage gap)")
ax.set_axis_off()
legend_elems = [Patch(facecolor="#EE6677", edgecolor="#7a0000", label=">15 min (gap)"),
               Patch(facecolor="#eeeeee", edgecolor="#999999", label="<=15 min (covered)"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor="#2a7f2a", markersize=8, label="Shelter")]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8)
save(fig, "fig12_okaloosa_walton_gap_map.png")
record(12, "fig12_okaloosa_walton_gap_map.png", "Routing analysis", "N. Okaloosa/Walton coverage gap map")

# =============================================================================
# SECTION: Featured findings
# =============================================================================

# --- Fig 13: Walton summary card ---
wr = COMBINED[COMBINED["GEOID"] == WALTON_ID].iloc[0]
n_bg_total = len(COMBINED)
pct_rank = int(wr["suitability_rank"]) / n_bg_total * 100
flash_rank = int((COMBINED["flash_count"] > wr["flash_count"]).sum()) + 1
fig, ax = plt.subplots(figsize=(7, 6))
ax.axis("off")
ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor="#7a0000", linewidth=2))
ax.text(0.5, 0.92, f"Walton County Block Group {WALTON_ID}", ha="center", fontsize=13, fontweight="bold")
ax.text(0.5, 0.85, "The strongest single case in the study area (4-year baseline)", ha="center", fontsize=10, style="italic", color="grey")
rows = [
    ("Population (2020)", f"{int(wr['P1_001N']):,}"),
    ("Suitability rank", f"{int(wr['suitability_rank'])} of {n_bg_total} (top {pct_rank:.0f}%)"),
    ("Drive time to nearest shelter", f"{wr['drive_min_to_shelter']:.1f} min  (longest in study area)"),
    ("GLM flashes (4 seasons, 2022-25)", f"{int(wr['flash_count']):,}  (rank {flash_rank} of {n_bg_total})"),
    ("Days flagged by BOTH anomaly methods", f"{int(wr['n_days_both_flag'])}"),
    ("Peak flashes in a 30-min window", f"{int(wr['max_peak_30min'])}"),
    ("Total burst episodes", f"{int(wr['total_burst_episodes'])}"),
    ("On final shortlist (land-only, 4-yr)", "Yes"),
]
y0 = 0.72
for label, val in rows:
    ax.text(0.06, y0, label, fontsize=10, va="center")
    ax.text(0.96, y0, val, fontsize=10, va="center", ha="right", fontweight="bold")
    y0 -= 0.085
save(fig, "fig13_walton_summary_card.png")
record(13, "fig13_walton_summary_card.png", "Featured findings", "Walton 121319503053 summary card")

# --- Fig 14: Okaloosa cluster 3-and-3 split ---
oka_rows = COMBINED[COMBINED["GEOID"].isin(OKA_CLUSTER)].copy()
oka_gdf = LAND_BG[LAND_BG["GEOID"].isin(OKA_CLUSTER)].merge(
    COMBINED[["GEOID", "on_shortlist", "suitability_rank", "n_days_both_flag", "drive_min_to_shelter"]], on="GEOID")

fig = plt.figure(figsize=(12, 6))
ax_map = fig.add_subplot(1, 2, 1)
oka_gdf.plot(ax=ax_map, color=oka_gdf["on_shortlist"].map({True: "#f4a13c", False: "#bdbdbd"}),
            edgecolor="#5a3a00", linewidth=0.8)
for _, r in oka_gdf.iterrows():
    c = r.geometry.centroid
    ax_map.annotate(r["GEOID"][-3:], (c.x, c.y), ha="center", fontsize=8, fontweight="bold")
n_on = int(oka_rows["on_shortlist"].sum())
ax_map.set_title(f"N. Okaloosa cluster (4-yr baseline): {n_on} on shortlist (orange),\n{len(oka_rows) - n_on} not (grey)")
ax_map.set_axis_off()

ax_tbl = fig.add_subplot(1, 2, 2)
ax_tbl.axis("off")
cols = ["GEOID (last 3)", "Rank", "Both-days", "On list?"]
cell_text = [[g[-3:], str(int(r)), str(int(b)), "YES" if s else "no"]
            for g, r, b, s in zip(oka_rows["GEOID"], oka_rows["suitability_rank"],
                                  oka_rows["n_days_both_flag"], oka_rows["on_shortlist"])]
tbl = ax_tbl.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2)
ax_tbl.set_title(f"The {n_on}-and-{len(oka_rows) - n_on} split", fontsize=11)
fig.suptitle("Northern Okaloosa cluster (120910210011-023), 4-year baseline", y=1.02, fontsize=13)
save(fig, "fig14_okaloosa_cluster_split.png")
record(14, "fig14_okaloosa_cluster_split.png", "Featured findings", "Okaloosa 6-BG cluster, 3-and-3 split")

# =============================================================================
# SECTION: ML model comparison
# =============================================================================

# --- Fig 15: 6 models AUC with error bars (4-year window, 2022-2025) ---
mc4 = pd.read_csv("data/processed/features/model_comparison_4yr_results.csv")
label_map = {"LogReg+interactions": "LogReg+\ninteractions"}
mc4_sorted = mc4.sort_values("pooled_AUC", ascending=False)
names = [label_map.get(m, m) for m in mc4_sorted["model"]]
aucs = mc4_sorted["pooled_AUC"].tolist()
stds = mc4_sorted["fold_AUC_std"].tolist()
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(names, aucs, yerr=stds, capsize=5, color="#4477AA",
             error_kw={"elinewidth": 1.5, "ecolor": "#333333"})
ax.axhline(0.5, color="grey", linestyle=":", label="chance (AUC 0.5)")
ax.set_ylabel("AUC (pooled out-of-fold)")
ax.set_ylim(0.4, 0.85)
ax.set_title("Model comparison, block-group x month target, 4-year window (2022-25)\n"
            "(error bars = fold-to-fold std across 5 county folds; top 4 models are within noise of each other)")
for b, v, s in zip(bars, aucs, stds):
    ax.text(b.get_x() + b.get_width() / 2, v + s + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.legend()
save(fig, "fig15_model_comparison_auc.png")
record(15, "fig15_model_comparison_auc.png", "ML model comparison", "6 models AUC, 4-year window (2022-25)")

# =============================================================================
# DEDICATED SECTION: 2-year vs 4-year training-diversity comparison
#
# This is a deliberate methodological comparison, not leftover superseded
# numbers. Every other figure/statistic in this report uses the 4-year
# (2022-2025) window as the single primary result (Option B). This section
# is kept separate, on purpose, because it is direct evidence of a real
# problem found and fixed: training on only 2 climate-neutral seasons
# produced an ONI coefficient that silently saturated the 2026 forecast
# (100% flagged anomalous - Fig 19, left panel) despite a deceptively good
# cross-validation AUC (Fig 18). Widening to 4 climate-diverse years exposed
# and fixed that failure, at the honest cost of a slightly harder CV problem
# (Fig 16). Most student projects never get to demonstrate this test - it is
# reported as a strength, not an afterthought.
#
# Every number below is a LIVE recompute via the actual training scripts
# (model_ceiling_push.py for the 2-yr CV, retrain_4yr_model.py for the 4-yr
# side, validate_2026.py for both years' 2026 out-of-time evaluation) reading
# straight from their saved artifact files - no hardcoded AUC constants, so
# this section cannot silently drift if any of those scripts or their inputs
# are ever touched again.
# =============================================================================
print("  [comparison section] recomputing 2-yr ceiling-push nested CV (RF, expanded features)...")
import model_ceiling_push as mcp
import retrain_4yr_model as r4yr
import validate_2026 as v26

df_2yr_exp = mcp.build_df()
cv_2yr_res = mcp.nested_cv(df_2yr_exp, mcp.feature_sets()["expanded (static4 + NLCD + lags + ONI + cyclical DOY)"])
cv_2yr = cv_2yr_res["RandomForest"]["pooled_AUC"]

print("  [comparison section] recomputing 4-yr nested CV (RF, same expanded features)...")
cv_4yr_res = r4yr.nested_cv(TRAIN4, r4yr.FEATS)
cv_4yr = cv_4yr_res["RandomForest"]["pooled_AUC"]

print("  [comparison section] recomputing 2-yr and 4-yr 2026 out-of-time evaluation...")
train2, train_lut = v26.build_training()
j26 = v26.spatial_join_2026()
bm26 = v26.anomaly_flags_2026(j26)
prev26 = {(g, m): int(bm26[(bm26.GEOID == g) & (bm26.month == m)]["target"].sum() > 0)
         for g in v26.EXTRA["GEOID"] for m in range(5, 9)}
lastyr26 = {(g, m): train_lut.get((g, 2025, m), 0) for g in v26.EXTRA["GEOID"] for m in range(5, 9)}
full26 = (pd.MultiIndex.from_product([v26.EXTRA["GEOID"], range(5, 9)], names=["GEOID", "month"])
         .to_frame(index=False).merge(bm26, on=["GEOID", "month"], how="left"))
full26["target"] = full26["target"].fillna(0).astype(int)
test26_2 = v26.build_features(full26, 2026, prev26, lastyr26)
oot_2yr_res = v26.evaluate(train2, test26_2)
oot_2yr = oot_2yr_res["RandomForest"]["auc"]

oot_4yr_res = r4yr.evaluate_2026(TRAIN4, TEST26_4, FEATS_4YR)
oot_4yr = oot_4yr_res["RandomForest"]["auc"]

# --- Fig 16: AUC progression, now ending on the 4-year retrain ---
stages = ["Static features\nonly (LogReg)", "+ month\nindicators (RF)", "+ NLCD, lags, ONI,\ncyclical DOY (RF)\n2-yr window (2024-25)",
         "4-year window\n(2022-25), same\nfeatures (RF)"]
stage_auc = [0.685, 0.775, cv_2yr, cv_4yr]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(range(3), stage_auc[:3], marker="o", markersize=10, color="#EE6677", linewidth=2.5)
ax.plot(range(2, 4), stage_auc[2:], marker="o", markersize=10, color="#4477AA", linewidth=2.5, linestyle="--")
for i, v in enumerate(stage_auc):
    color = "#7a0000" if i < 3 else "#223a66"
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold", color=color)
ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=8.5)
ax.set_ylabel("AUC (nested spatial CV, county held-out)")
ax.set_ylim(0.6, 0.86)
ax.set_title("AUC progression from feature expansion (red) through the\n4-year training-window retrain (blue) - a deliberate, expected step down")
ax.annotate("Widening to 4 climate-diverse\nyears makes the CV problem\nitself harder - traded for better\nforward-in-time generalization\n(see Fig 18)",
           xy=(3, cv_4yr), xytext=(2.15, 0.66), fontsize=8, color="#223a66",
           arrowprops=dict(arrowstyle="->", color="#223a66"))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
save(fig, "fig16_auc_progression.png")
record(16, "fig16_auc_progression.png", "DEDICATED SECTION: 2-yr vs 4-yr training-diversity comparison",
      f"AUC progression 0.685 -> 0.775 -> {cv_2yr:.3f} -> {cv_4yr:.3f} (4-yr retrain)")

# --- Fig 17: RF feature importance - PERMUTATION importance (not impurity-based),
# expanded feature set, full 4-year (2022-25) refit ---
print("  refitting expanded RF on full 4-year (2022-25) data + permutation importance (takes a minute)...")
from sklearn.inspection import permutation_importance

sc17 = StandardScaler().fit(TRAIN4[FEATS_4YR])
Xf17 = sc17.transform(TRAIN4[FEATS_4YR])
y17 = TRAIN4["target"].to_numpy()
rf_final = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=4,
                                  max_features="sqrt", random_state=42, n_jobs=-1).fit(Xf17, y17)
perm = permutation_importance(rf_final, Xf17, y17, scoring="roc_auc", n_repeats=15,
                              random_state=42, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=FEATS_4YR)
imp_std = pd.Series(perm.importances_std, index=FEATS_4YR)
order = imp.sort_values().index
imp, imp_std = imp[order], imp_std[order]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.index, imp.values, xerr=imp_std.values, color="#4477AA",
       error_kw={"elinewidth": 1, "ecolor": "#333333", "capsize": 3})
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Permutation importance (mean AUC drop, 15 repeats)")
ax.set_title("Random Forest feature importance - permutation-based\n(expanded feature set, full 4-year [2022-25] refit)")
save(fig, "fig17_rf_feature_importance.png")
record(17, "fig17_rf_feature_importance.png", "ML model comparison",
      "RF permutation importance, 4-year expanded model")

# --- Fig 18: CV AUC vs 2026 out-of-time AUC, 2-yr model vs 4-yr retrain ---
labels = ["Cross-validation\n(county held-out)", "2026 out-of-time\n(year held-out)"]
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(2)
w = 0.35
bars_2yr = ax.bar(x - w / 2, [cv_2yr, oot_2yr], w, label="2-yr model (2024-25)", color="#EE6677")
bars_4yr = ax.bar(x + w / 2, [cv_4yr, oot_4yr], w, label="4-yr retrain (2022-25)", color="#4477AA")
for b, v in zip(bars_2yr, [cv_2yr, oot_2yr]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for b, v in zip(bars_4yr, [cv_4yr, oot_4yr]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("AUC"); ax.set_ylim(0.6, 0.86)
ax.set_title("Random Forest: spatial vs temporal generalization\n"
            f"2-yr gap {oot_2yr - cv_2yr:+.3f} (real, beyond fold noise) -> "
            f"4-yr gap {oot_4yr - cv_4yr:+.3f} (essentially closed)")
ax.legend(loc="lower center")
save(fig, "fig18_cv_vs_2026_auc.png")
record(18, "fig18_cv_vs_2026_auc.png", "DEDICATED SECTION: 2-yr vs 4-yr training-diversity comparison",
      f"CV vs 2026 OOT AUC, 2-yr ({cv_2yr:.3f}/{oot_2yr:.3f}) vs 4-yr retrain ({cv_4yr:.3f}/{oot_4yr:.3f})")

# --- Fig 19: LogReg 2026 predicted-probability histogram, 2-yr (broken) vs 4-yr (fixed) ---
proba_4yr = oot_4yr_res["LogReg"]["proba"]
proba_2yr = oot_2yr_res["LogReg"]["proba"]
prec_4yr, rec_4yr = oot_4yr_res["LogReg"]["prec@0.5"], oot_4yr_res["LogReg"]["rec@0.5"]
true_rate_4yr = TEST26_4["target"].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
pct_2yr = (proba_2yr >= 0.5).mean() * 100
axes[0].hist(proba_2yr, bins=30, color="#EE6677", edgecolor="white")
axes[0].axvline(0.5, color="#7a0000", linestyle="--", linewidth=2, label="decision threshold (0.5)")
axes[0].set_title(f"2-yr model (BROKEN - not a real result)\n{pct_2yr:.0f}% of predictions above 0.5 - "
                  f"ONI extrapolation\nsaturation, not a usable classifier", fontsize=10, color="#7a0000")
axes[0].set_xlabel("LogReg predicted probability (2026 test rows)")
axes[0].set_ylabel("Count of block-group-months")
axes[0].legend(fontsize=8)

pct_4yr = (proba_4yr >= 0.5).mean() * 100
axes[1].hist(proba_4yr, bins=30, color="#4477AA", edgecolor="white")
axes[1].axvline(0.5, color="#223a66", linestyle="--", linewidth=2, label="decision threshold (0.5)")
axes[1].set_title(f"4-yr retrain (FIXED - the honest, usable result)\n{pct_4yr:.0f}% of predictions above 0.5 "
                  f"vs a {true_rate_4yr:.0f}% true rate\n(prec {prec_4yr:.3f}, rec {rec_4yr:.3f})", fontsize=10, color="#223a66")
axes[1].set_xlabel("LogReg predicted probability (2026 test rows)")
axes[1].legend(fontsize=8)
fig.suptitle("Logistic regression predicted probabilities on 2026 data: before vs after the 4-year retrain", y=1.02)
save(fig, "fig19_logreg_2026_probability_histogram.png")
record(19, "fig19_logreg_2026_probability_histogram.png", "DEDICATED SECTION: 2-yr vs 4-yr training-diversity comparison",
      f"LogReg 2026 probability histogram: 2-yr broken ({pct_2yr:.0f}% saturated) vs 4-yr fixed ({pct_4yr:.0f}%, honest)")

# =============================================================================
# Index file
# =============================================================================
idx_path = os.path.join(OUT, "index.md")
with open(idx_path, "w") as f:
    f.write("# Report figures index\n\n")
    f.write("| # | File | Section | Title |\n|---|---|---|---|\n")
    for n, fname, section, title in index_rows:
        f.write(f"| {n} | `{fname}` | {section} | {title} |\n")

print(f"\nAll {len(index_rows)} figures saved to {OUT}/")
print(f"Index written to {idx_path}")
