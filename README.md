# Pharmacy Desert Planner

> **Where should new pharmacies be built to maximize healthcare accessibility in Chicago?**

An interactive geospatial planning tool that identifies Chicago's most underserved communities and recommends optimal pharmacy locations using a principled optimization algorithm, multi-source data integration, and a narrative-driven UI.

---

## What It Does

Chicago's pharmacy access gap is real and uneven. Residents in the South and West Sides face significantly longer distances to the nearest pharmacy than residents in wealthier neighborhoods — and that gap compounds when 40%+ of households in those communities lack vehicle access.

This tool:
1. **Maps the access gap** — visualizes a Pharmacy Need Index across all 801 Chicago census tracts, combining six demographic and health data dimensions
2. **Identifies pharmacy deserts** — tracts where residents live beyond the walkable coverage threshold
3. **Optimizes new locations** — runs a greedy weighted max-coverage algorithm to recommend where new pharmacies would do the most good
4. **Reports equity outcomes** — shows what share of newly-covered residents live in majority-Black or majority-Hispanic/Latino communities (post-hoc, never a scoring input)

---

## Technical Highlights

| Skill | What's Demonstrated |
|---|---|
| **Optimization** | Greedy weighted maximum coverage location problem (WMCLP) with marginal discounting — 1−1/e (~63%) provable optimality guarantee |
| **Geospatial** | GeoPandas, Shapely, osmnx; residential land clipping; street-network isochrones; spatial joins |
| **Data Engineering** | 6 integrated data sources with disk-caching, graceful degradation on missing API keys, and proper error typing |
| **Machine Learning** | Percentile-rank normalization for multi-dimensional need index; feature engineering for pharmacy-specific need |
| **Data Sources** | Census ACS (B/DP/S series), CDC PLACES (Socrata), TIGERweb, HUD ArcGIS, Chicago Data Portal, OSM |
| **Product Thinking** | Named strategy presets instead of raw sliders; narrative UI flow; equity reporting framing |
| **Software Engineering** | `core/` pipeline with zero Streamlit dependency; independently testable; multi-city architecture |
| **UI/UX** | Portfolio-matched dark theme (Fraunces + Inter, teal accent); Folium dark maps; custom HTML tables |
| **Testing** | 7 pytest files covering algorithms, config, coverage math, integration |

---

## Algorithm

### Pharmacy Need Index (PNI)

```
PNI(tract) = Population × Σ( percentile_rank(factor_i) × weight_i )
```

Six factors, each **percentile-ranked within Chicago** before combining — so a vehicle-ownership percentage, a disease prevalence rate, and a poverty rate (different units, different scales) are comparable:

| Factor | Source | Default Weight |
|---|---|---|
| No vehicle access | ACS B08201 | 28% |
| Poverty rate | ACS B17001 | 22% |
| Chronic medication burden (diabetes + hypertension) | CDC PLACES | 20% |
| Age 65+ | ACS DP05 | 13% |
| Mobility / ambulatory disability | ACS S1810 | 12% |
| Uninsured rate | ACS DP03 | 5% |

The original CMU model summed 13 CDC PLACES measures with arbitrary weights. This rebuild narrows to 6 factors that specifically predict *recurring pharmacy need* — the conditions that mean ongoing prescriptions, not every chronic condition on file.

### Greedy Max-Coverage Optimizer

Replaces the original "Multi-Armed Bandit" (which was actually just sorting by a static reward — picking candidate #2 never accounted for what candidate #1 already covered).

```python
remaining_need = { geoid: need_score[geoid] × (1 - existing_coverage[geoid]) }

for each pick:
    best = argmax Σ remaining_need[geoid] × coverage_fraction(candidate, geoid)
    select(best)
    for each geoid:
        remaining_need[geoid] *= (1 - coverage_fraction(best, geoid))  # discount
```

This is the standard greedy approach to the Weighted Maximum Coverage Location Problem. It has a **provable 1−1/e ≈ 63% optimality guarantee** (from the submodularity of coverage functions), is deterministic, and is fast enough to re-run on every UI change for Chicago-sized candidate sets.

### Equity Reporting

Race/ethnicity is **reported, never scored**. After optimization, the app shows what share of newly-covered residents live in majority-Black or majority-Hispanic/Latino tracts — a post-hoc equity check, not a variable the optimizer can see or chase. The same equity story emerges naturally from socioeconomic factors without making race an optimization target.

---

## Data Sources

| Source | Module | Data | Cache |
|---|---|---|---|
| Census ACS 5-Year Estimates | `core/acs_need_factors.py` | Vehicle access, poverty, age, mobility, uninsured, race/ethnicity by tract | parquet |
| CDC PLACES (Socrata) | `core/places_api.py` | Diabetes & hypertension prevalence by tract | via city_data |
| TIGERweb REST API | `core/tiger_tracts.py` | Census tract boundaries (any US state/county) | via city_data |
| HUD ArcGIS | `core/opportunity_zones.py` | Federally designated Opportunity Zone tracts | parquet |
| Chicago Data Portal | `core/chicago_community_areas.py` | 77 official Chicago Community Area polygons | parquet |
| 2020 Census Blocks | `core/census_blocks.py` | Block-level population for accurate coverage weighting (opt-in) | parquet |
| OpenStreetMap via osmnx | `core/candidates.py`, `core/isochrones.py` | Land use polygons, street network for isochrones | JSON, parquet |
| Local files | `core/city_data.py` | Chicago tract boundaries, PLACES scores, pharmacy locations | N/A |

All external data is disk-cached under `.cache/` (gitignored). Cold-start time (empty cache) is approximately 2–5 minutes for Chicago; subsequent runs are instant.

---

## Architecture

```
app/streamlit_app.py          ← narrative UI, Folium maps, CSS (Streamlit only)
         │
         ↓ calls
core/pipeline.py              ← orchestration (no Streamlit imports)
  ├── prepare_city()          ← load data → need index → residential clip
  ├── get_preview_layer()     ← pre-optimization need map with rich tooltips
  └── run_optimization()      ← candidates → coverage → greedy select → results
         │
core/  need_index.py          ← PNI formula
       optimize.py            ← greedy WMCLP
       coverage.py            ← area + block-weighted coverage fractions
       candidates.py          ← programmatic grid generation from OSM land
       city_data.py           ← abstract CityDataSource + Chicago/NYC impls
       config.py              ← all decision variables as dataclasses
       cache.py               ← disk cache (parquet + JSON)
       acs*.py / places_api.py / tiger_tracts.py / ...   ← data integrations
```

The `core/` package has **zero Streamlit dependency** — the full pipeline is independently testable and scriptable.

### Multi-City Architecture

Chicago and New York City are both wired up. Adding another US city is a ~5-line subclass of `OpenDataCitySource` (display name, OSM place name, state FIPS, county FIPS list, map center). See `NewYorkDataSource` in `core/city_data.py`.

---

## Setup

Geospatial packages (GDAL/GEOS/PROJ via geopandas, osmnx, fiona) install most reliably on Windows via conda:

```bash
# From the pharmacy-desert-app/ directory, using a conda Python:
<path-to-conda-python> -m venv --system-site-packages .venv
.venv/Scripts/pip install -r requirements.txt
```

If not on conda, plain pip may work — expect to resolve GDAL/GEOS wheel availability for your platform.

**Optional:** Copy `.env.example` to `.env` and add a free Census API key from https://api.census.gov/data/key_signup.html to enable:
- Tiered desert radius (adjusts threshold by income + vehicle access)
- Population-weighted coverage (real 2020 Census block data instead of area estimate)
- Full 6-factor PNI (without key, falls back to chronic medication burden only)

Without a key, the app still works and all other features remain available.

---

## Running

```bash
pharmacy-desert-app/.venv/Scripts/python -m streamlit run app/streamlit_app.py
```

Or double-click `Start Pharmacy Planner.bat` on Windows.

The app is one continuously-scrolling narrative page:

1. **Hero + context** — problem statement and three grounding statistics
2. **Planning strategy** — pick a named preset or customize factor weights
3. **Access threshold** — define what "covered" means (default: 0.5 mi)
4. **Need map** — Pharmacy Need Index choropleth over residential land, with existing coverage overlay and rich hover tooltips
5. **Intervention** — set pharmacy count, optional Opportunity Zone filter
6. **Optimization** — click Run; step-by-step progress messages show what's happening
7. **Results** — impact metrics, plain-English summary, equity check, results map, ranked site table, community-area impact table
8. **Methodology** — collapsed accordion explaining all decisions and limitations

---

## Testing

```bash
pharmacy-desert-app/.venv/Scripts/python -m pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `test_need_index.py` | PNI scales linearly with population; no double-normalization |
| `test_optimize.py` | Greedy matches brute-force on small cases; marginal discounting works; min-distance respected |
| `test_candidates.py` | Grid generation is deterministic, stays in-zone, respects exclusion distance |
| `test_coverage.py` | Coverage fraction area math |
| `test_pipeline.py` | End-to-end integration |
| `test_config.py` | Weight sums, label/key correspondence, strategy preset completeness |

---

## What Changed from the Original CMU Project

The original notebook (`../Pharmacy Desert Modeling (original)/`) had three fundamental problems:

1. **Fake MAB**: The "Multi-Armed Bandit" computed each candidate's reward once before any selection. Its epsilon-greedy loop just repeatedly sampled a fixed value — equivalent to sorting by reward. Picking candidate #2 never discounted for what candidate #1 already covered. Fixed by implementing the correct greedy WMCLP.

2. **Need index math bug**: CDC PLACES `Data_Value` is a population-normalized prevalence rate. The original divided by population, normalized 0-100, then *re-multiplied by population* — the divide/normalize/multiply sequence doesn't cancel cleanly and distorts relative weight between large/small-population tracts. Fixed by computing `weighted_need = percentile_composite × population` directly.

3. **No generalization path**: Candidate sites came from a manually-curated Chicago-only LoopNet CSV. Replaced with programmatic OSM land-use grid generation — any city is a ~5-line config change.

---

## Project Layout

```
core/            Pure-Python pipeline — no Streamlit dependency, independently testable
  config.py        Decision variables (NeedWeights, OptConfig, CandidateConfig, STRATEGY_PRESETS)
  city_data.py     Abstract CityDataSource + ChicagoDataSource + NewYorkDataSource
  need_index.py    Pharmacy Need Index: percentile-ranked 6-factor composite
  optimize.py      Greedy weighted max-coverage selector
  coverage.py      Area-based + block-population-weighted coverage fractions
  candidates.py    Candidate site generation from OSM land use
  pipeline.py      prepare_city() + get_preview_layer() + run_optimization()
  acs.py           Census ACS tiered-radius lookups
  acs_need_factors.py  6 ACS need factors per tract
  census_blocks.py 2020 Census block population (opt-in accurate coverage)
  tiger_tracts.py  TIGERweb tract boundaries (any US city)
  places_api.py    CDC PLACES API (any US city)
  isochrones.py    Street-network isochrones (visualization only)
  opportunity_zones.py  HUD ArcGIS OZ designation
  chicago_community_areas.py  Chicago Data Portal 77 community areas
  cache.py         Disk cache (parquet + JSON) keyed by city slug
app/
  streamlit_app.py  Single-page narrative UI + dark Folium maps + styled tables
tests/             7-file pytest suite
data/              Local Chicago data (tracts, PLACES scores, pharmacies)
.cache/            Disk-cached API responses (gitignored)
```
