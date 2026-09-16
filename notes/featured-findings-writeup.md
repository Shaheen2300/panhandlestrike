# Featured findings — draft for the results section

Two block-group results anchor the results section. They were selected because
each illustrates a different way the combined method (suitability score +
dual-method anomaly detection + network coverage gap) behaves, and together they
show the shortlist is not a single-signal artifact.

All figures are over the two study seasons (May–September 2024 and 2025), 589
land block groups after the water-polygon exclusion.

---

## Finding 1 — Walton County Block Group `121319503053`: the strongest single case

This block group clears every shortlist criterion and is extreme on the two that
matter most for a lightning shelter.

| Measure | Value | Standing in study area |
|---|---|---|
| Population (2020) | 1,135 | modest |
| GLM flashes | 22,066 | **2nd-highest of 589** (max is 34,097) |
| Flash density | 37.5 / km² | top decile |
| Drive time to nearest shelter | 28.2 min | **longest in the study area** (median 5.5 min) |
| Days flagged, z-score method | 14 | — |
| Days flagged, rolling-30-min burst method | 113 | — |
| Days flagged by **both** methods | 14 | — |
| Peak flashes in a 30-min window | 994 | — |
| Total burst episodes (≥5 flashes / 30 min) | 374 | — |
| Suitability rank | 127 / 589 (top 22%) | — |

The suitability rank of 127 understates the case: the score is held down only by
the small resident population, which caps the exposure term. On the other two
axes the block group is at or near the study-area maximum. It has the longest
shelter drive time of any inhabited block group in the five counties, and it
recorded the second-largest flash total while being one of only a handful of
block groups with a 30-minute burst approaching a thousand flashes. Its 374 burst
episodes and 113 burst-flagged days mean intense lightning here is a recurring
condition across both seasons, not a single outlier storm.

**Siting implication.** A shelter placed to serve this block group would address
the worst-covered population in the study area while sitting under one of its most
active lightning regimes. It is the clearest candidate to lead the
recommendations.

---

## Finding 2 — Northern Okaloosa cluster `120910210011`–`023`: a 3-and-3 split

Six contiguous block groups (Census tracts 021001 and 021002, three block groups
each; ~11,900 residents combined) form the largest single coverage gap in the
study area — every one is 21–25 minutes from the nearest shelter. They do **not**
behave uniformly under the rest of the method.

| GEOID | Pop | Drive (min) | Flashes | z-score days | burst days | both-method days | max 30-min peak | Suitability rank | On shortlist |
|---|---|---|---|---|---|---|---|---|---|
| `...210011` | 2,229 | 25.2 | 81 | 5 | 2 | 2 | 12 | 16 | **yes** |
| `...210023` | 2,389 | 21.8 | 54 | 6 | 3 | 2 | 9 | 44 | **yes** |
| `...210012` | 2,000 | 24.2 | 40 | 9 | 3 | 3 | 7 | 142 | **yes** |
| `...210013` | 1,870 | 24.3 | 26 | 18 | 0 | 0 | 4 | 206 | no |
| `...210021` | 1,828 | 25.0 | 28 | 7 | 0 | 0 | 4 | 201 | no |
| `...210022` | 1,549 | 22.5 | 9 | 6 | 0 | 0 | 3 | 278 | no |

Three of the six reach the shortlist (6,618 residents); three do not (5,247
residents). The split is driven by the **absolute** burst method, not the
relative z-score. All six carry z-score anomalies — their baseline flash rate is
near zero, so even a handful of flashes on one day registers as unusual for that
area. But the three that fail the shortlist never once recorded a 30-minute
window with five or more flashes (their peaks are 3–4), so they have zero
both-method days and cannot meet the "confirmed anomaly" criterion. Their
suitability ranks (201–278) also fall outside the top quartile.

It is worth being explicit that even the three cluster members that do make the
shortlist qualify on a thin lightning signal: 40–81 flashes over two seasons and
2–3 minor confirmed bursts (peaks of 7–12). They reach the shortlist mainly on
population and isolation, and only marginally on hazard — the opposite balance
from Walton `121319503053`.

**Method implication.** This cluster is the cleanest illustration that a relative
anomaly is not the same as an absolute hazard. Requiring agreement between a
baseline-relative test and a baseline-independent absolute threshold prevents
near-zero-baseline areas from entering the shortlist on statistical noise alone,
while still admitting them when a genuine burst is recorded.

**Siting implication.** The northern Okaloosa gap is real and large regardless of
lightning: ~11,900 people more than 20 minutes from shelter. But the lightning
case for prioritizing it *within* this project is modest and concentrated in the
three shortlisted block groups. If the recommendation is framed purely around
lightning exposure, this cluster ranks below Walton `121319503053` and the
higher-flash Santa Rosa and Bay entries; if framed around coverage equity, it is
the single largest underserved population block.
