"""4-year rerun of build_dashboard.py: identical map/layers/popup structure,
built from the 4-year combined priority table (blockgroup_combined_priority_4yr,
hazard = 2022-2025 flash density, anomaly = 2022-2025 baseline) instead of the
original 2024-25-only table. Labels updated to say "4 seasons (2022-2025)"
where the original said "2 seasons".

Output: dashboard/panhandle_priority_dashboard.html (replaces the previous,
2-year-baseline version, which is archived alongside it as
panhandle_priority_dashboard_2yr_baseline.html).
"""

import os

import branca.colormap as cm
import folium
import geopandas as gpd

SRC = "data/processed/final_4yr/blockgroup_combined_priority_4yr.geojson"
OUT = "dashboard/panhandle_priority_dashboard.html"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

WALTON_FEATURED = "121319503053"
OKALOOSA_CLUSTER = [
    "120910210011", "120910210012", "120910210013",
    "120910210021", "120910210022", "120910210023",
]

gdf = gpd.read_file(SRC).to_crs("EPSG:4326")
gdf["on_shortlist"] = gdf["on_shortlist"].astype(bool)

_cent = gdf.to_crs("EPSG:5070").geometry.centroid.to_crs("EPSG:4326")
gdf["cx"] = _cent.x.values
gdf["cy"] = _cent.y.values

score = gdf["suitability_score"]
smin, smax = float(score.min()), float(score.max())
colormap = cm.LinearColormap(
    ["#2c7bb6", "#ffffbf", "#d7191c"],
    vmin=smin, vmax=smax,
    caption=(f"Suitability score  (pop + 4-yr flash density + drive-time to shelter)  "
             f"— range {smin:.3f} to {smax:.3f}"),
)
print(f"colormap range: {smin:.4f} .. {smax:.4f}")

center = [gdf["cy"].mean(), gdf["cx"].mean()]
m = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap", control_scale=True)

POPUP_FIELDS = [
    ("GEOID", "GEOID"),
    ("COUNTY_NAME", "County"),
    ("NAMELSAD", "Block group"),
    ("P1_001N", "Population (2020)"),
    ("flash_count", "GLM flashes (4 seasons, 2022-25)"),
    ("flash_per_km2", "Flash density / km2 (4-yr)"),
    ("drive_min_to_shelter", "Drive time to nearest shelter (min)"),
    ("coverage_tier", "Coverage tier"),
    ("suitability_rank", "Suitability rank (of 589)"),
    ("suitability_score", "Suitability score"),
    ("n_days_zscore_flag", "Anomaly days - z-score (4-yr baseline)"),
    ("n_days_burst_flag", "Anomaly days - 30-min burst"),
    ("n_days_both_flag", "Anomaly days - BOTH methods"),
    ("max_zscore", "Max daily z-score"),
    ("max_peak_30min", "Max flashes in a 30-min window"),
    ("on_shortlist", "On final shortlist"),
]
fields = [f for f, _ in POPUP_FIELDS]
aliases = [a for _, a in POPUP_FIELDS]

def style_fn(feat):
    return {
        "fillColor": colormap(feat["properties"]["suitability_score"]),
        "color": "#666666", "weight": 0.4, "fillOpacity": 0.75,
    }

folium.GeoJson(
    gdf.to_json(),
    name="Suitability score (all 589 block groups, 4-yr baseline)",
    style_function=style_fn,
    highlight_function=lambda f: {"weight": 2, "color": "#000000", "fillOpacity": 0.9},
    popup=folium.GeoJsonPopup(fields=fields, aliases=aliases, localize=True, max_width=360),
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "COUNTY_NAME", "suitability_rank", "suitability_score"],
        aliases=["GEOID", "County", "Rank", "Score"], localize=True,
    ),
).add_to(m)

shortlist = gdf[gdf["on_shortlist"]]
folium.GeoJson(
    shortlist.to_json(),
    name=f"Final shortlist ({len(shortlist)} block groups, 4-yr baseline)",
    style_function=lambda f: {
        "fillColor": "#00000000", "color": "#0d1b8f", "weight": 3, "fillOpacity": 0.0,
    },
    highlight_function=lambda f: {"weight": 5, "color": "#3f51ff"},
    popup=folium.GeoJsonPopup(fields=fields, aliases=aliases, localize=True, max_width=360),
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "COUNTY_NAME", "suitability_rank"],
        aliases=["SHORTLIST -", "County", "Rank"],
    ),
).add_to(m)

featured_group = folium.FeatureGroup(name="Featured findings (Walton + N. Okaloosa cluster)")

walton = gdf[gdf["GEOID"] == WALTON_FEATURED]
folium.GeoJson(
    walton.to_json(),
    style_function=lambda f: {"fillColor": "#d7191c", "color": "#7a0000", "weight": 3, "fillOpacity": 0.35},
).add_to(featured_group)
wr = walton.iloc[0]
folium.Marker(
    [wr["cy"], wr["cx"]],
    icon=folium.Icon(color="red", icon="bolt", prefix="fa"),
    popup=folium.Popup(
        f"<b>Walton BG {WALTON_FEATURED}</b> &mdash; strongest single case<br>"
        f"Population {int(wr['P1_001N'])}<br>"
        f"Suitability rank {int(wr['suitability_rank'])} / 589 (4-yr baseline; was 127 under 2-yr)<br>"
        f"Drive to shelter <b>{wr['drive_min_to_shelter']:.1f} min</b> (longest in study area)<br>"
        f"GLM flashes <b>{int(wr['flash_count']):,}</b> (4 seasons, 2022-25)<br>"
        f"Both-method anomaly days: <b>{int(wr['n_days_both_flag'])}</b>; "
        f"max 30-min burst <b>{int(wr['max_peak_30min'])}</b><br>"
        f"On shortlist: <b>{bool(wr['on_shortlist'])}</b>",
        max_width=340,
    ),
).add_to(featured_group)

for geoid in OKALOOSA_CLUSTER:
    row = gdf[gdf["GEOID"] == geoid]
    if row.empty:
        continue
    r = row.iloc[0]
    on = bool(r["on_shortlist"])
    folium.GeoJson(
        row.to_json(),
        style_function=lambda f, on=on: {
            "fillColor": "#fdae61" if on else "#bdbdbd",
            "color": "#8a4b08" if on else "#636363",
            "weight": 2.5, "fillOpacity": 0.4,
        },
    ).add_to(featured_group)
    folium.Marker(
        [r["cy"], r["cx"]],
        icon=folium.Icon(color="orange" if on else "lightgray", icon="users", prefix="fa"),
        popup=folium.Popup(
            f"<b>N. Okaloosa cluster &mdash; BG {geoid}</b><br>"
            f"Population {int(r['P1_001N'])}<br>"
            f"Drive to shelter {r['drive_min_to_shelter']:.1f} min (all 6 in the &gt;15-min gap)<br>"
            f"Suitability rank {int(r['suitability_rank'])} / 589 (4-yr baseline)<br>"
            f"GLM flashes {int(r['flash_count'])} (4 seasons, 2022-25)<br>"
            f"Anomaly days &mdash; z-score {int(r['n_days_zscore_flag'])}, "
            f"burst {int(r['n_days_burst_flag'])}, BOTH {int(r['n_days_both_flag'])}<br>"
            f"On shortlist: <b>{on}</b>"
            + ("" if on else "<br><i>not on shortlist under the 4-yr baseline</i>"),
            max_width=340,
        ),
    ).add_to(featured_group)

featured_group.add_to(m)

colormap.add_to(m)
folium.LayerControl(collapsed=True, position="topright").add_to(m)

title_html = (
    '<div style="position: fixed; top: 10px; left: 50px; z-index: 9999; '
    'background: white; padding: 8px 14px; border: 1px solid #999; border-radius: 4px; '
    'font-family: sans-serif; font-size: 14px;">'
    '<b>PanhandleStrike &mdash; lightning shelter siting priority (4-year baseline, 2022-2025)</b><br>'
    f'<span style="font-size: 12px;">589 block groups by suitability &middot; '
    f'{int(shortlist.shape[0])}-BG shortlist outlined &middot; featured findings pinned</span></div>'
)
m.get_root().html.add_child(folium.Element(title_html))

m.save(OUT)
size_kb = os.path.getsize(OUT) / 1024
print(f"Saved {OUT}  ({size_kb:.0f} KB)")
print(f"Layers: suitability choropleth (589), shortlist outline ({len(shortlist)}), "
      f"featured findings (1 Walton + {len(OKALOOSA_CLUSTER)} Okaloosa)")
