"""Chicago Pharmacy Desert Planner — visual design matches hosseinkay.github.io

Design tokens from hosseinkay/Personal-Portfolio:
  bg #0b0d0f · elevated #111417 · border #22272c
  fg #e8eaed · fg-muted #9aa4ab · fg-subtle #6b747b
  accent #4fa89a · accent-strong #74c4b7
  fonts: Fraunces (display/headings) + Inter (body)
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import branca.colormap as bcm
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import acs, isochrones, pipeline
from core import cache as cache_mod
from core import coverage as cov_mod
from core.city_data import get_osm_place_name
from core.config import (
    DEFAULT_WEIGHTS,
    STRATEGY_DESCRIPTIONS,
    STRATEGY_PRESETS,
    CandidateConfig,
    NeedWeights,
    OptConfig,
)
from core.demo_data import load_demo_snapshot

CITY_KEY = "chicago"
CITY_NAME = "Chicago"

# ---------------------------------------------------------------------------
# Secrets: inject Census API key into the environment so core/ can find it.
# On Streamlit Cloud, add CENSUS_API_KEY under Settings → Secrets.
# Locally, use .env (python-dotenv in core/acs.py handles that).
# ---------------------------------------------------------------------------
import os as _os
try:
    _ck = st.secrets.get("CENSUS_API_KEY")
    if _ck and not _os.environ.get("CENSUS_API_KEY"):
        _os.environ["CENSUS_API_KEY"] = _ck
except (AttributeError, FileNotFoundError):
    pass  # no secrets file locally — .env handles it

st.set_page_config(
    page_title="Chicago Pharmacy Desert Planner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Portfolio design system — injected once at the top of every render
# ---------------------------------------------------------------------------
st.markdown(
    """
    <!-- Google Fonts: Fraunces (display) + Inter (body) -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
    /* ── Design tokens ────────────────────────────────────────────── */
    :root {
      --bg:           #0b0d0f;
      --bg-e:         #111417;
      --bg-e2:        #161a1e;
      --fg:           #e8eaed;
      --fg-m:         #9aa4ab;
      --fg-s:         #6b747b;
      --border:       #22272c;
      --accent:       #4fa89a;
      --accent-s:     #74c4b7;
      --r:            12px;
      --font-d:       'Fraunces', ui-serif, Georgia, serif;
      --font-b:       'Inter', ui-sans-serif, system-ui, sans-serif;
    }

    /* ── Global reset ─────────────────────────────────────────────── */
    html, body,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
      background-color: var(--bg) !important;
      color: var(--fg) !important;
      font-family: var(--font-b) !important;
    }

    /* Hide Streamlit chrome */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    .stDeployButton,
    #MainMenu { display: none !important; }

    /* ── Layout ───────────────────────────────────────────────────── */
    .block-container {
      padding-top: 3.5rem !important;
      padding-bottom: 5rem !important;
      max-width: 1120px !important;
    }

    /* ── Typography ───────────────────────────────────────────────── */
    h1, [data-testid="stHeadingWithActionElements"] h1 {
      font-family: var(--font-d) !important;
      font-size: 2.75rem !important;
      font-weight: 500 !important;
      line-height: 1.1 !important;
      letter-spacing: -0.01em !important;
      color: var(--fg) !important;
      margin-bottom: 0.25rem !important;
    }
    h2, [data-testid="stHeadingWithActionElements"] h2 {
      font-family: var(--font-d) !important;
      font-size: 1.55rem !important;
      font-weight: 500 !important;
      line-height: 1.2 !important;
      color: var(--fg) !important;
      margin-top: 0 !important;
      margin-bottom: 0.15rem !important;
    }
    h3, h4, [data-testid="stHeadingWithActionElements"] h3 {
      font-family: var(--font-d) !important;
      font-size: 1.05rem !important;
      font-weight: 500 !important;
      color: var(--fg) !important;
      margin-top: 1rem !important;
      margin-bottom: 0.1rem !important;
    }
    p, li,
    .stMarkdown p,
    .stMarkdown li {
      font-family: var(--font-b) !important;
      color: var(--fg-m) !important;
      line-height: 1.7 !important;
    }
    .stMarkdown strong { color: var(--fg) !important; font-weight: 600 !important; }
    .stMarkdown em    { color: var(--fg-m) !important; font-style: italic !important; }

    /* ── Hero custom classes ──────────────────────────────────────── */
    .ph-eyebrow {
      font-family: var(--font-b) !important;
      font-size: 0.75rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.22em !important;
      text-transform: uppercase !important;
      color: var(--accent) !important;
      margin-bottom: 0.75rem !important;
      display: block !important;
    }
    .ph-hero-title {
      font-family: var(--font-d) !important;
      font-size: 3.1rem !important;
      font-weight: 500 !important;
      line-height: 1.08 !important;
      letter-spacing: -0.015em !important;
      color: var(--fg) !important;
      margin: 0 0 1.5rem !important;
    }
    .ph-hero-body {
      font-family: var(--font-b) !important;
      font-size: 1.05rem !important;
      line-height: 1.7 !important;
      color: var(--fg-m) !important;
      max-width: 660px !important;
      margin-bottom: 0.85rem !important;
    }
    .ph-hero-body strong { color: var(--fg) !important; }
    .ph-hero-cta {
      font-family: var(--font-b) !important;
      font-size: 0.95rem !important;
      color: var(--fg-s) !important;
      font-style: italic !important;
      margin-top: 1rem !important;
    }

    /* ── Section headings (portfolio SectionHeading pattern) ──────── */
    .ph-sh { margin-bottom: 1.1rem !important; }
    .ph-sh-idx {
      display: block !important;
      font-family: var(--font-d) !important;
      font-size: 0.72rem !important;
      letter-spacing: 0.28em !important;
      text-transform: uppercase !important;
      color: var(--accent) !important;
      margin-bottom: 0.35rem !important;
    }
    .ph-sh-title {
      font-family: var(--font-d) !important;
      font-size: 1.55rem !important;
      font-weight: 500 !important;
      color: var(--fg) !important;
      line-height: 1.2 !important;
    }

    /* Sub-heading inside a results section */
    .ph-sub-h {
      font-family: var(--font-d) !important;
      font-size: 0.9rem !important;
      font-weight: 500 !important;
      color: var(--fg-m) !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase !important;
      margin: 1.4rem 0 0.6rem !important;
      padding-bottom: 0.4rem !important;
      border-bottom: 1px solid var(--border) !important;
    }

    /* Equity label */
    .ph-eq-label {
      font-family: var(--font-b) !important;
      font-size: 0.72rem !important;
      letter-spacing: 0.2em !important;
      text-transform: uppercase !important;
      color: var(--accent) !important;
      margin: 1.2rem 0 0.5rem !important;
      display: block !important;
    }

    /* ── Cards: st.container(border=True) ────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:    var(--bg-e) !important;
      border:        1px solid var(--border) !important;
      border-radius: var(--r) !important;
      padding:       1.75rem 2rem !important;
      transition:    border-color 0.2s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
      border-color: rgba(79, 168, 154, 0.28) !important;
    }

    /* ── Form control labels ──────────────────────────────────────── */
    label[data-testid="stWidgetLabel"],
    [data-baseweb="form-control-label"],
    .stSlider label,
    .stCheckbox label,
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label {
      font-family: var(--font-b) !important;
      font-size:   0.85rem !important;
      font-weight: 500 !important;
      color:       var(--fg) !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"],
    .stCaption, small {
      font-family: var(--font-b) !important;
      font-size:   0.78rem !important;
      color:       var(--fg-s) !important;
    }

    /* ── Selectbox ────────────────────────────────────────────────── */
    [data-baseweb="select"] [data-baseweb="input"],
    [data-baseweb="select"] > div:first-child {
      background:   var(--bg-e2) !important;
      border-color: var(--border) !important;
      border-radius: 8px !important;
      color:        var(--fg) !important;
      font-family:  var(--font-b) !important;
    }
    [data-baseweb="menu"],
    [data-baseweb="popover"] {
      background:   var(--bg-e) !important;
      border:       1px solid var(--border) !important;
      border-radius: 10px !important;
    }
    [data-baseweb="option"] {
      color:       var(--fg) !important;
      font-family: var(--font-b) !important;
      background:  transparent !important;
    }
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {
      background: var(--bg-e2) !important;
    }
    [data-baseweb="select"] svg { fill: var(--fg-s) !important; }

    /* ── Number / text inputs ─────────────────────────────────────── */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
      background:  var(--bg-e2) !important;
      border:      1px solid var(--border) !important;
      border-radius: 8px !important;
      color:       var(--fg) !important;
      font-family: var(--font-b) !important;
    }
    [data-testid="stNumberInput"] button {
      background:  var(--bg-e2) !important;
      border-color: var(--border) !important;
      color:       var(--fg-m) !important;
    }
    [data-testid="stNumberInput"] button:hover {
      border-color: var(--accent) !important;
      color:        var(--accent) !important;
    }

    /* ── Slider ───────────────────────────────────────────────────── */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
      background:   var(--accent) !important;
      border-color: var(--accent) !important;
      box-shadow:   0 0 0 3px rgba(79, 168, 154, 0.25) !important;
    }
    /* Filled track */
    [data-testid="stSlider"] div[data-testid="stTickBar"] + div > div > div:first-child {
      background: var(--accent) !important;
    }

    /* ── Checkbox ─────────────────────────────────────────────────── */
    [data-baseweb="checkbox"] span[role="presentation"] {
      border-color:     var(--border) !important;
      background-color: var(--bg-e2) !important;
      border-radius:    4px !important;
    }
    [data-baseweb="checkbox"] input:checked ~ span[role="presentation"],
    [data-baseweb="checkbox"] [data-checked="true"] {
      background-color: var(--accent)  !important;
      border-color:     var(--accent)  !important;
    }
    [data-baseweb="checkbox"] > div > div:last-child {
      color:       var(--fg-m) !important;
      font-family: var(--font-b) !important;
      font-size:   0.875rem !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────── */
    .stButton > button,
    [data-testid="baseButton-secondary"] {
      font-family:  var(--font-b) !important;
      font-size:    0.85rem !important;
      font-weight:  500 !important;
      background:   var(--bg-e2) !important;
      color:        var(--fg-m) !important;
      border:       1px solid var(--border) !important;
      border-radius: 8px !important;
      transition:   border-color 0.18s ease, color 0.18s ease !important;
    }
    .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover {
      border-color: var(--accent) !important;
      color:        var(--accent-s) !important;
      background:   var(--bg-e2) !important;
    }
    /* Primary button — "Run optimization" */
    [data-testid="baseButton-primary"],
    .stButton > button[kind="primary"] {
      background:   var(--accent) !important;
      color:        #0b0d0f !important;
      border-color: var(--accent) !important;
      font-weight:  600 !important;
    }
    /* Streamlit wraps button label in <div><p>text</p></div> — target both */
    [data-testid="baseButton-primary"] p,
    [data-testid="baseButton-primary"] div,
    [data-testid="baseButton-primary"] span,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] div,
    .stButton > button[kind="primary"] span {
      color: #0b0d0f !important;
    }
    [data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
      background:   var(--accent-s) !important;
      border-color: var(--accent-s) !important;
      color:        #0b0d0f !important;
    }
    [data-testid="baseButton-primary"]:hover p,
    [data-testid="baseButton-primary"]:hover div,
    .stButton > button[kind="primary"]:hover p,
    .stButton > button[kind="primary"]:hover div {
      color: #0b0d0f !important;
    }

    /* Download button */
    .stDownloadButton > button {
      font-family:  var(--font-b) !important;
      font-size:    0.8rem !important;
      background:   var(--bg-e2) !important;
      color:        var(--fg-s) !important;
      border:       1px solid var(--border) !important;
      border-radius: 8px !important;
      transition:   border-color 0.18s, color 0.18s !important;
    }
    .stDownloadButton > button:hover {
      border-color: var(--accent) !important;
      color:        var(--accent) !important;
    }

    /* ── Expanders ────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
      background:    var(--bg-e) !important;
      border:        1px solid var(--border) !important;
      border-radius: var(--r) !important;
    }
    /* Summary row: background + text color only.
       IMPORTANT: do NOT set font-family/font-size on summary * — that
       overrides Streamlit's internal icon font and causes the expand arrow
       to render as raw ligature text (e.g. "_arr") instead of the ▾ glyph. */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details > summary,
    [data-testid="stExpander"] details[open] > summary {
      background-color: var(--bg-e) !important;
      color:            var(--fg-m) !important;
      list-style:       none !important;
    }
    /* Apply Inter only to the text label — not to icon/svg children */
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary > div > p,
    [data-testid="stExpander"] summary > div > span {
      font-family: var(--font-b) !important;
      font-size:   0.85rem !important;
      font-weight: 500 !important;
      color:       var(--fg-m) !important;
    }
    [data-testid="stExpander"] summary:hover p,
    [data-testid="stExpander"] summary:hover > div > p,
    [data-testid="stExpander"] summary:hover > div > span { color: var(--fg) !important; }
    /* SVG expand/collapse arrow — ensure visible and tinted */
    [data-testid="stExpander"] svg {
      fill:    var(--fg-s) !important;
      display: inline-block !important;
    }
    [data-testid="stExpander"] summary:hover svg { fill: var(--fg) !important; }
    /* Content area below summary when open */
    [data-testid="stExpander"] details > div,
    [data-testid="stExpander"] details > div > div {
      background-color: var(--bg-e) !important;
    }

    /* ── Metric cards ─────────────────────────────────────────────── */
    [data-testid="metric-container"] {
      background:    var(--bg-e) !important;
      border:        1px solid var(--border) !important;
      border-radius: var(--r) !important;
      padding:       1rem 1.1rem !important;
    }
    [data-testid="stMetricValue"] {
      font-family: var(--font-d) !important;
      font-size:   1.5rem !important;
      font-weight: 500 !important;
      color:       var(--fg) !important;
    }
    [data-testid="stMetricLabel"] {
      font-family:     var(--font-b) !important;
      font-size:       0.7rem !important;
      font-weight:     500 !important;
      color:           var(--fg-s) !important;
      text-transform:  uppercase !important;
      letter-spacing:  0.08em !important;
    }

    /* ── Spinner ──────────────────────────────────────────────────── */
    [data-testid="stSpinner"] p { color: var(--fg-m) !important; font-family: var(--font-b) !important; }

    /* ── Dataframe wrapper ────────────────────────────────────────── */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"] {
      border:        1px solid var(--border) !important;
      border-radius: var(--r) !important;
      overflow:      hidden !important;
    }

    /* ── Select / dropdown — dark popup ──────────────────────────── */
    /* Streamlit's selectbox popup: [data-baseweb="popover"] wraps
       several divs using generated st-* emotion classes (no stable
       attribute).  The inspection-confirmed white elements are the
       direct child div of popover, and the ul with the Streamlit
       test-id.  Target them explicitly. */
    [data-baseweb="popover"] {
      background:    var(--bg-e) !important;
      border:        1px solid var(--border) !important;
      border-radius: 10px !important;
      box-shadow:    0 8px 24px rgba(0,0,0,0.55) !important;
      overflow:      hidden !important;
    }
    /* Direct child div (first white wrapper inside popover) */
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] > div > div,
    [data-baseweb="popover"] > div > div > div {
      background: var(--bg-e) !important;
    }
    /* The virtualised list — stable Streamlit test-id */
    [data-testid="stSelectboxVirtualDropdown"] {
      background: var(--bg-e) !important;
    }
    /* Each option row */
    [data-testid="stSelectboxVirtualDropdown"] li,
    [data-baseweb="popover"] li[role="option"] {
      background:  transparent !important;
      color:       var(--fg-m) !important;
      font-family: var(--font-b) !important;
      font-size:   0.875rem !important;
    }
    [data-testid="stSelectboxVirtualDropdown"] li:hover,
    [data-baseweb="popover"] li[role="option"]:hover {
      background: var(--bg-e2) !important;
      color:      var(--fg) !important;
    }
    [data-baseweb="popover"] li[aria-selected="true"] {
      background: rgba(79, 168, 154, 0.12) !important;
      color:      var(--fg) !important;
    }
    /* Text nodes inside options */
    [data-testid="stSelectboxVirtualDropdown"] div,
    [data-baseweb="popover"] li div {
      color: var(--fg-m) !important;
      font-family: var(--font-b) !important;
    }

    /* Select trigger (closed state) */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
      background:    var(--bg-e2) !important;
      border-color:  var(--border) !important;
      border-radius: 8px !important;
    }
    [data-testid="stSelectbox"] span { color: var(--fg) !important; }

    /* ── Custom dark HTML tables (ph-table) ───────────────────────── */
    .ph-table-wrap {
      overflow-x: auto !important;
      border-radius: var(--r) !important;
      border: 1px solid var(--border) !important;
      margin: 0.5rem 0 1rem !important;
    }
    .ph-table {
      border-collapse: collapse !important;
      width: 100% !important;
      font-family: var(--font-b) !important;
      font-size: 0.82rem !important;
    }
    .ph-table thead tr {
      background: var(--bg-e2) !important;
      border-bottom: 1px solid var(--border) !important;
    }
    .ph-table th {
      color: var(--fg-s) !important;
      font-family: var(--font-b) !important;
      font-weight: 500 !important;
      font-size: 0.68rem !important;
      letter-spacing: 0.1em !important;
      text-transform: uppercase !important;
      padding: 0.65rem 1rem !important;
      text-align: left !important;
      white-space: nowrap !important;
    }
    .ph-table tbody tr { background: var(--bg-e) !important; }
    .ph-table tbody tr:hover { background: var(--bg-e2) !important; }
    .ph-table tbody tr:last-child td { border-bottom: none !important; }
    .ph-table td {
      color:           var(--fg-m) !important;
      padding:         0.5rem 0.85rem !important;
      border-bottom:   1px solid var(--border) !important;
      vertical-align:  top !important;
      white-space:     normal !important;
      max-width:       260px !important;
      word-break:      break-word !important;
    }
    .ph-table td:first-child {
      color:           var(--fg) !important;
      font-weight:     500 !important;
      white-space:     nowrap !important;
      max-width:       none !important;
    }
    /* Rank number badge */
    .ph-table td.rank-cell {
      font-family:  var(--font-d) !important;
      color:        var(--accent) !important;
      font-weight:  600 !important;
      font-size:    0.95rem !important;
      white-space:  nowrap !important;
      width:        2.5rem !important;
      text-align:   center !important;
    }

    /* ── Scrollbar ────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--fg-s); }

    /* ── Horizontal dividers ──────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── How-it-works step cards (hero intro) ────────────────────── */
    /* ── Narrative pullquote + map lead-in ──────────────────────── */
    .ph-pullquote {
      font-family: var(--font-d) !important;
      font-size: 1.75rem !important;
      font-weight: 600 !important;
      color: var(--fg) !important;
      border-left: 4px solid var(--accent) !important;
      padding: 0.35rem 0 0.35rem 1.2rem !important;
      margin: 1.4rem 0 1.1rem !important;
      line-height: 1.25 !important;
      letter-spacing: -0.01em !important;
    }
    .ph-map-lead {
      background: var(--bg-e) !important;
      border: 1px solid var(--border) !important;
      border-left: 3px solid var(--accent) !important;
      border-radius: 0 var(--r) var(--r) 0 !important;
      padding: 0.9rem 1.25rem !important;
      margin: 1.1rem 0 1.5rem !important;
    }
    .ph-map-lead p {
      font-family: var(--font-b) !important;
      font-size: 0.9rem !important;
      color: var(--fg-m) !important;
      line-height: 1.65 !important;
      margin: 0 !important;
    }

    /* ── Stat burst (3-up context cards) ─────────────────────────── */
    .ph-stat-burst {
      display: grid !important;
      grid-template-columns: repeat(3, 1fr) !important;
      gap: 1rem !important;
      margin: 0 0 2.5rem !important;
    }
    .ph-stat-card {
      background: var(--bg-e) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--r) !important;
      padding: 1.4rem 1.6rem !important;
      display: flex !important;
      flex-direction: column !important;
      gap: 0.45rem !important;
    }
    .ph-stat-number {
      font-family: var(--font-d) !important;
      font-size: 2.3rem !important;
      font-weight: 500 !important;
      color: var(--accent-s) !important;
      line-height: 1 !important;
      letter-spacing: -0.02em !important;
    }
    .ph-stat-label {
      font-family: var(--font-b) !important;
      font-size: 0.80rem !important;
      color: var(--fg-m) !important;
      line-height: 1.55 !important;
    }

    /* ── Run button — oversized CTA ──────────────────────────────── */
    [data-testid="baseButton-primary"][kind="primary"] {
      min-height: 3rem !important;
      font-size: 0.95rem !important;
      letter-spacing: 0.01em !important;
    }

    /* ── Explore-further label ────────────────────────────────────── */
    .ph-explore-label {
      font-family: var(--font-b) !important;
      font-size: 0.72rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.22em !important;
      text-transform: uppercase !important;
      color: var(--fg-s) !important;
      margin: 0 0 0.85rem !important;
      display: block !important;
    }

    /* ── Optimizer result narrative ───────────────────────────────── */
    .ph-optimizer-narrative {
      font-family: var(--font-b) !important;
      font-size: 0.975rem !important;
      line-height: 1.75 !important;
      color: var(--fg-m) !important;
      margin: 1.1rem 0 1.4rem !important;
      padding: 1rem 1.25rem !important;
      background: rgba(79, 168, 154, 0.06) !important;
      border-left: 3px solid var(--accent) !important;
      border-radius: 0 var(--r) var(--r) 0 !important;
    }
    .ph-optimizer-narrative strong { color: var(--fg) !important; }

    /* ── Tab bar separator ────────────────────────────────────────── */
    [data-testid="stTabs"] > div:first-child {
      border-bottom: 1px solid var(--border) !important;
      margin-bottom: 1.25rem !important;
    }
    [data-testid="stTabs"] button[role="tab"] {
      font-family: var(--font-b) !important;
      font-size: 0.85rem !important;
      letter-spacing: 0.02em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Folium dark tooltip CSS — injected into every map object
_FOLIUM_DARK_CSS = """<style>
.leaflet-tooltip {
  background: #111417 !important;
  border: 1px solid #22272c !important;
  color: #e8eaed !important;
  font-family: 'Inter', ui-sans-serif, sans-serif !important;
  font-size: 12px !important;
  line-height: 1.5 !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.55) !important;
}
.leaflet-tooltip-top::before    { border-top-color:    #22272c !important; }
.leaflet-tooltip-bottom::before { border-bottom-color: #22272c !important; }
.leaflet-tooltip-left::before   { border-left-color:   #22272c !important; }
.leaflet-tooltip-right::before  { border-right-color:  #22272c !important; }
.leaflet-tooltip th, .leaflet-tooltip tr, .leaflet-tooltip td {
  color: #e8eaed !important; border: none !important;
}
</style>"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def walk_minutes(miles: float) -> int:
    return round(miles * 19.35)  # ~3.1 mph walking pace


def section_header(idx: str, title: str) -> None:
    """Replica of the portfolio's SectionHeading component."""
    st.markdown(
        f"""<div class="ph-sh">
          <span class="ph-sh-idx">{idx}</span>
          <span class="ph-sh-title">{title}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def subsection_header(title: str) -> None:
    st.markdown(f'<p class="ph-sub-h">{title}</p>', unsafe_allow_html=True)


def build_map(center, fit_to=None):
    m = folium.Map(
        location=list(center),
        tiles="cartodbdark_matter",
        prefer_canvas=True,
    )
    if fit_to is not None:
        minx, miny, maxx, maxy = fit_to.total_bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]])
    m.get_root().html.add_child(folium.Element(_FOLIUM_DARK_CSS))
    return m


def to_wgs84(geom_3857):
    if geom_3857 is None or geom_3857.is_empty:
        return None
    return gpd.GeoSeries([geom_3857], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]


def add_filled_geometry(m, geom_4326, color, opacity, tooltip):
    if geom_4326 is None or geom_4326.is_empty:
        return
    folium.GeoJson(
        {"type": "Feature", "geometry": geom_4326.__geo_interface__, "properties": {}},
        style_function=lambda _f, c=color, o=opacity: {
            "fillColor": c, "color": c, "weight": 2.0, "fillOpacity": o,
        },
        tooltip=folium.Tooltip(tooltip),
    ).add_to(m)


def numbered_icon(rank) -> folium.DivIcon:
    """Numbered circle marker using the portfolio accent teal."""
    return folium.DivIcon(
        html=f"""<div style="
            background-color:#4fa89a; color:#0b0d0f; border-radius:50%;
            width:26px; height:26px; display:flex; align-items:center;
            justify-content:center; font-family:'Inter',sans-serif;
            font-weight:700; font-size:12px;
            border:2px solid #74c4b7; box-shadow:0 2px 6px rgba(0,0,0,0.5);
        ">{int(rank)}</div>""",
        icon_size=(26, 26),
        icon_anchor=(13, 13),
    )


def fmt_pct(x, decimals=0) -> str:
    return "—" if pd.isna(x) else f"{x:.{decimals}f}%"


def fmt_miles(x) -> str:
    return "—" if pd.isna(x) else f"{x:.2f} mi"


def fmt_int(x) -> str:
    return "—" if pd.isna(x) else f"{x:,.0f}"


def yes_no(x) -> str:
    if pd.isna(x):
        return "—"
    return "Yes" if x else "No"


def render_dark_table(df: pd.DataFrame, rank_col: str | None = None) -> None:
    """Render a DataFrame as a styled dark HTML table.

    st.dataframe() renders in an iframe — external CSS cannot reach it.
    This replaces it with a plain HTML table that our design system CSS
    can fully style.  rank_col names the column whose cells get the
    accent-teal rank-badge treatment.
    """
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col_name, val in zip(df.columns, row):
            css = ' class="rank-cell"' if col_name == rank_col else ""
            escaped = str(val).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cells += f"<td{css}>{escaped}</td>"
        rows_html += f"<tr>{cells}</tr>"
    st.markdown(
        f'<div class="ph-table-wrap"><table class="ph-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


# Need-index choropleth: warm yellow→red pops on the dark map tiles
NEED_COLORMAP = bcm.LinearColormap(
    colors=["#FFEC80", "#FEB24C", "#FC4E2A", "#BD0026"],
    vmin=0, vmax=100,
    caption="Pharmacy Need Index (0–100)",
    text_color="#e8eaed",
)

# Coverage-gain choropleth: teal gradient shows where optimizer made the biggest difference
GAIN_COLORMAP = bcm.LinearColormap(
    colors=["#0b1f1d", "#1a4d40", "#2d7a65", "#4fa89a", "#74c4b7"],
    vmin=0, vmax=40,
    caption="Coverage gain (percentage points)",
)


def need_choropleth_layer(
    gdf_4326: gpd.GeoDataFrame, tooltip_cols: list[tuple[str, str]]
) -> folium.GeoJson:
    fields = [c for c, _ in tooltip_cols]
    aliases = [a for _, a in tooltip_cols]
    return folium.GeoJson(
        gdf_4326.__geo_interface__,
        style_function=lambda f: {
            "fillColor": NEED_COLORMAP(f["properties"].get("normalized_need") or 0),
            "color": "rgba(0,0,0,0.15)",
            "weight": 0.3,
            "fillOpacity": 0.72,
        },
        highlight_function=lambda f: {"weight": 1.5, "color": "#74c4b7"},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
    )


def gain_choropleth_layer(
    gdf_4326: gpd.GeoDataFrame, tooltip_cols: list[tuple[str, str]]
) -> folium.GeoJson:
    fields = [c for c, _ in tooltip_cols]
    aliases = [a for _, a in tooltip_cols]
    return folium.GeoJson(
        gdf_4326.__geo_interface__,
        style_function=lambda f: {
            "fillColor": GAIN_COLORMAP(min(f["properties"].get("growth_pct") or 0, 40)),
            "color": "rgba(0,0,0,0.15)",
            "weight": 0.3,
            "fillOpacity": 0.85,
        },
        highlight_function=lambda f: {"weight": 1.5, "color": "#74c4b7"},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
    )


def _network_cache_available(radius_miles: float) -> tuple[bool, bool]:
    """Return (isochrone_parquet_exists, graphml_exists).

    Checks whether the pre-computed drive network or per-radius isochrone
    parquet already lives on disk.  When neither is present (e.g. fresh
    Streamlit Cloud deployment) we skip network computation entirely and
    fall back to straight-line buffers immediately — attempting to download
    the full Chicago drive network from OSM would time out or exhaust RAM.
    """
    place_name = get_osm_place_name(CITY_KEY)
    slug = cache_mod.slugify(place_name)
    city_dir = cache_mod.CACHE_DIR / slug
    iso_exists = (city_dir / f"existing_isochrone_r{radius_miles:.2f}.parquet").exists()
    graph_exists = (city_dir / "drive_network.graphml").exists()
    return iso_exists, graph_exists


def get_existing_coverage_shape(pharmacies: gpd.GeoDataFrame, radius_miles: float):
    cache_store = st.session_state.setdefault("existing_coverage_cache", {})
    key = round(radius_miles, 4)
    if key in cache_store:
        return cache_store[key]

    iso_cached, graph_cached = _network_cache_available(radius_miles)

    if iso_cached or graph_cached:
        # Network data is on disk — use it (fast path: loads from parquet or graphml)
        try:
            place_name = get_osm_place_name(CITY_KEY)
            geom = isochrones.get_merged_existing_isochrone_cached(place_name, pharmacies, radius_miles)
            status = "Street-network isochrone"
        except Exception:
            # Corrupted cache or unexpected error — fall back gracefully
            geom = to_wgs84(cov_mod.merged_buffer(pharmacies, radius_miles))
            status = "Straight-line buffer"
    else:
        # No cached network on disk — use straight-line buffer immediately.
        # Downloading the full drive network from OSM is too slow and
        # memory-intensive for Streamlit Cloud cold starts.
        geom = to_wgs84(cov_mod.merged_buffer(pharmacies, radius_miles))
        status = "Straight-line buffer"

    cache_store[key] = (geom, status)
    return cache_store[key]


def get_new_coverage_shape(selected_4326: gpd.GeoDataFrame, radius_miles: float):
    """Return merged street-network isochrone for the selected candidate sites.

    Lookup strategy (no circle fallback — ever):

    1. candidate_iso_by_coord_r{r:.2f}.parquet  — all 7283 demo candidates
       indexed by coord_key "lon_lat" (5 decimal EPSG:4326, ≈ 1 m precision).
       The coordinate key is stable across pipeline runs because generate_grid_
       candidates is deterministic on committed zoned_land.parquet.  Works for
       any live optimization regardless of candidate_id value.

    2. new_sites_iso_n{N:02d}_r{r:.2f}.parquet  — pre-committed per-N merged
       isochrones for the default demo greedy rank order.  Only used when the
       all-candidates parquet is absent (e.g., first deploy before precompute).
       May not align exactly with the live selected sites in that case.

    If nothing matches: omit the blue layer.  Better to show nothing than a
    misleading straight-line buffer (circle) or wrong-location isochrone.
    """
    if selected_4326.empty:
        return None, ""

    # Session cache keyed by exact selected coordinates + radius (not just N —
    # two different sets of N candidates must produce different isochrones).
    coord_keys = tuple(sorted(
        f"{r.geometry.x:.5f}_{r.geometry.y:.5f}"
        for _, r in selected_4326.iterrows()
    ))
    cache_store = st.session_state.setdefault("new_coverage_cache", {})
    cache_key = (coord_keys, round(radius_miles, 4))
    if cache_key in cache_store:
        return cache_store[cache_key]

    place_name = get_osm_place_name(CITY_KEY)
    slug = cache_mod.slugify(place_name)
    cache_dir = cache_mod.city_cache_dir(slug)

    # ── 1. Coordinate-keyed all-candidate lookup ────────────────────────────
    coord_path = cache_dir / f"candidate_iso_by_coord_r{radius_miles:.2f}.parquet"
    if coord_path.exists():
        if "cand_iso_by_coord" not in st.session_state:
            st.session_state["cand_iso_by_coord"] = gpd.read_parquet(coord_path)
        cand_iso = st.session_state["cand_iso_by_coord"]
        sel_keys = list(coord_keys)
        matched = cand_iso.loc[[k for k in sel_keys if k in cand_iso.index]]
        if not matched.empty:
            geom = matched.geometry.union_all()
            if geom is not None and not geom.is_empty:
                result = (geom, "Street-network isochrone")
                cache_store[cache_key] = result
                return result

    # ── 2. Per-N fallback (pre-committed, used only when coord parquet absent)
    n = len(selected_4326)
    n_path = cache_dir / f"new_sites_iso_n{n:02d}_r{radius_miles:.2f}.parquet"
    if n_path.exists():
        gdf = gpd.read_parquet(n_path)
        if not gdf.empty:
            result = (gdf.geometry.iloc[0], "Street-network isochrone")
            cache_store[cache_key] = result
            return result

    # Nothing matched — omit blue layer.
    return None, ""


# ---------------------------------------------------------------------------
# Bootstrap: show results on first visit — fast path uses precomputed demo data,
# slow path runs the live pipeline (development / first deploy before precompute).
# ---------------------------------------------------------------------------
if "bootstrapped" not in st.session_state:
    _demo = load_demo_snapshot()

    if _demo is not None:
        # ── Fast path (deployed demo) ────────────────────────────────────────
        # Load the precomputed need map and pharmacy locations so section 01
        # renders instantly.  "result" is intentionally NOT stored here —
        # results only appear after the user clicks Run, keeping the opening
        # view clean (map + intro, no pre-filled optimization output).
        st.session_state["preview_layer"]  = _demo.preview_layer
        st.session_state["map_center"]     = _demo.result.center
        st.session_state["map_pharmacies"] = _demo.result.pharmacies
        st.session_state["base_radius"]    = _demo.meta.get("base_radius", 0.5)
        # prepared_key prevents a spurious prepare_city() call when the
        # default strategy widget renders for the first time.
        st.session_state["prepared_key"]   = (CITY_KEY, "Balanced need (recommended)")
        st.session_state["bootstrapped"]   = True
        st.session_state["using_demo"]     = True
    else:
        # ── Slow path (development / first-run before precompute script) ─────
        with st.status("Setting up Chicago Pharmacy Desert Planner…", expanded=True) as _boot:
            st.write("Loading city data from OpenStreetMap and Census sources…")
            _boot_weights = NeedWeights(weights=dict(DEFAULT_WEIGHTS))
            _boot_prepared = pipeline.prepare_city(CITY_KEY, _boot_weights, use_health_priorities=True)
            st.session_state["prepared_default"] = _boot_prepared
            st.session_state["prepared"]         = _boot_prepared
            st.session_state["prepared_key"]     = (CITY_KEY, "Balanced need (recommended)")
            st.session_state["map_center"]       = _boot_prepared.bundle.center
            st.session_state["map_pharmacies"]   = _boot_prepared.bundle.pharmacies
            st.write("Pre-computing need index map…")
            st.session_state["preview_layer"] = pipeline.get_preview_layer(_boot_prepared, 0.5)
            st.session_state["base_radius"]   = 0.5
            st.session_state["bootstrapped"]  = True
            st.session_state["using_demo"]    = False
            _boot.update(label="Ready. Map loaded below.", state="complete")

elif "prepared_default" not in st.session_state and st.session_state.get("prepared") is not None:
    st.session_state["prepared_default"] = st.session_state.get("prepared")

# ---------------------------------------------------------------------------
# Tab layout — Planner (main interactive flow) + Methodology (deep reference)
# ---------------------------------------------------------------------------
_planner_tab, _method_tab = st.tabs(
    ["Planner", "Methodology & Model Design"]
)
_planner_tab.__enter__()

# ---------------------------------------------------------------------------
# 01 · Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 0.5rem 0 0.5rem;">
      <span class="ph-eyebrow">Chicago &middot; Pharmacy Access Planning</span>
      <h1 class="ph-hero-title">Chicago Pharmacy<br>Desert Planner</h1>
      <p class="ph-hero-body">
        Pharmacy deserts are areas where getting to a pharmacy is difficult enough
        that routine access to prescriptions becomes a real problem.
        <strong>Chicago is one of the largest cities where that gap can vary
        dramatically from one neighborhood to another.</strong>
      </p>
      <p class="ph-hero-body">
        So where does the problem show up most clearly?
      </p>
      <p class="ph-hero-body">
        A useful starting point is to look at where pharmacies are located today
        and compare that with where pharmacy-related need is highest across the city.
      </p>
      <p class="ph-hero-body">
        Distance matters, but it is only part of the story. Limited pharmacy access
        can have a much bigger effect in communities where more households lack a
        vehicle, incomes are lower, residents are older or have mobility limitations,
        or conditions like diabetes and hypertension make regular prescriptions
        more important.
      </p>
      <div class="ph-map-lead">
        <p><strong>The map below brings those pieces together, showing existing
        pharmacy locations alongside the parts of Chicago where pharmacy need
        is highest.</strong></p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 01 · Current pharmacy need (full-width map — always balanced-need baseline)
# ---------------------------------------------------------------------------
st.markdown(
    "## Who is most affected by limited access?\n\n"
    "A pharmacy being far away is not equally disruptive for every neighborhood.\n\n"
    "To better understand where access matters most, this analysis looks at vehicle "
    "access, poverty, age, mobility limitations, uninsured status, and chronic conditions "
    "such as diabetes and hypertension. These factors are combined into the "
    "**Pharmacy Need Index**, which helps compare pharmacy-related need across "
    "Chicago's census tracts."
)

with st.container(border=True):
    section_header("01", "Where is pharmacy need highest?")
    st.caption(
        "Residential land colored by Pharmacy Need Index. "
        "Teal dots show existing pharmacy locations."
    )

    # Use map_center + map_pharmacies from session_state — works in both demo
    # mode (populated from precomputed data) and live mode (from prepared city).
    _map_center     = st.session_state.get("map_center")
    _map_pharmacies = st.session_state.get("map_pharmacies")
    _preview_layer  = st.session_state.get("preview_layer")

    if _preview_layer is not None and _map_center is not None:
        preview_layer = _preview_layer.copy()
        preview_layer["covered_display"] = preview_layer["is_covered"].map(yes_no)
        for _col, _fmt in [
            ("normalized_need",           lambda v: "—" if pd.isna(v) else f"{v:.0f} / 100"),
            ("nearest_pharmacy_miles",    fmt_miles),
            ("TotalPopulation",           fmt_int),
            ("no_vehicle_pct",            fmt_pct),
            ("poverty_pct",               fmt_pct),
            ("age65_pct",                 fmt_pct),
            ("ambulatory_disability_pct", fmt_pct),
            ("chronic_burden_pct",        fmt_pct),
            ("uninsured_pct",             fmt_pct),
        ]:
            preview_layer[f"{_col}_display"] = (
                preview_layer[_col].map(_fmt) if _col in preview_layer.columns else "—"
            )
        preview_layer["community_area_display"] = preview_layer["community_area"].fillna("Unknown")

        _preview_tooltip_cols = [
            ("community_area_display",            "Community area:"),
            ("normalized_need_display",           "Need index (0–100):"),
            ("nearest_pharmacy_miles_display",    "Nearest pharmacy:"),
            ("covered_display",                   "Currently covered:"),
            ("TotalPopulation_display",           "Population:"),
            ("no_vehicle_pct_display",            "No vehicle:"),
            ("poverty_pct_display",               "Poverty rate:"),
            ("age65_pct_display",                 "Age 65+:"),
            ("ambulatory_disability_pct_display", "Mobility difficulty:"),
            ("chronic_burden_pct_display",        "Diabetes / hypertension:"),
            ("uninsured_pct_display",             "Uninsured:"),
        ]

        m_preview = build_map(_map_center, fit_to=preview_layer)
        need_choropleth_layer(preview_layer, _preview_tooltip_cols).add_to(m_preview)
        NEED_COLORMAP.add_to(m_preview)

        if _map_pharmacies is not None:
            pharmacies_4326 = _map_pharmacies.to_crs("EPSG:4326")
            folium.GeoJson(
                pharmacies_4326[["geometry"]].__geo_interface__,
                marker=folium.CircleMarker(
                    radius=2, color="#4fa89a", fill=True, fill_color="#4fa89a", fill_opacity=0.55,
                ),
                tooltip=folium.Tooltip("Existing pharmacy"),
            ).add_to(m_preview)

        st_folium(m_preview, height=500, use_container_width=True, returned_objects=[], key="map_section01")
        st.caption("Teal dots = ~850 existing pharmacies. Hover any tract for details.")

        st.markdown(
            "**What are we looking at?**  \n"
            "The highlighted areas represent **residential land across Chicago**, rather than the full "
            "physical area of each census tract. This keeps the map focused on the places where "
            "people actually live.\n\n"
            "Each area is shaded using the **Pharmacy Need Index**, a score that combines population "
            "with health and demographic factors from the U.S. Census and CDC. Higher values represent "
            "areas where more people live and where the factors associated with pharmacy access and "
            "recurring medication needs are more concentrated.\n\n"
            "Existing pharmacies are shown as points on top of the map. "
            "For the full breakdown of the index, factor weights, data sources, and calculation, "
            "see the **Methodology & Model Design** tab."
        )
    else:
        st.info("Map loading. If this persists, reload the page.", icon="🗺️")

# Stat burst — three context numbers after the map (not before)
st.markdown(
    """
    <div class="ph-stat-burst">
      <div class="ph-stat-card">
        <span class="ph-stat-number">~850</span>
        <span class="ph-stat-label">
          pharmacies serve Chicago's 2.7&nbsp;million residents, but access
          clusters in wealthier neighborhoods, leaving South &amp; West Sides
          significantly underserved.
        </span>
      </div>
      <div class="ph-stat-card">
        <span class="ph-stat-number">40%+</span>
        <span class="ph-stat-label">
          of households in the most pharmacy-poor tracts lack vehicle access,
          making a pharmacy more than 0.5&nbsp;miles away effectively unreachable
          for routine prescriptions.
        </span>
      </div>
      <div class="ph-stat-card">
        <span class="ph-stat-number">6</span>
        <span class="ph-stat-label">
          data dimensions combined into the Pharmacy Need Index: vehicle access,
          poverty, chronic disease burden, age 65+, mobility, and uninsured rate.
          Each is percentile-ranked across all 801&nbsp;Chicago tracts.
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Visual placeholder: results render here (above controls), filled after controls execute
_results_slot = st.empty()

# ---------------------------------------------------------------------------
# 03 · Adjust and re-run (controls — de-emphasized; results appear above via _results_slot)
# ---------------------------------------------------------------------------
st.markdown(
    "## So where would a new pharmacy actually help?\n\n"
    "Finding a high-need area is only part of the problem.\n\n"
    "A new pharmacy is most useful when it reaches people who are not already well served, "
    "and adding several locations too close together can create redundant coverage. "
    "The planner evaluates candidate locations across the city and looks for the sites "
    "that create the greatest additional access to unmet need."
)

with st.container(border=True):
    section_header("03", "Adjust and re-run")

    _p_col, _n_col, _d_col = st.columns([5, 3, 3], gap="large")

    with _p_col:
        st.markdown("**Planning strategy**")
        strategy_options = list(STRATEGY_PRESETS.keys())
        strategy = st.selectbox(
            "Planning strategy", strategy_options, index=0,
            label_visibility="collapsed",
        )
        st.caption(STRATEGY_DESCRIPTIONS[strategy])

    with _n_col:
        st.markdown("**New pharmacies to site**")
        num_pharmacies = st.number_input(
            "New pharmacies", min_value=1, max_value=30, value=5,
            label_visibility="collapsed",
        )

    with _d_col:
        st.markdown("**Access distance**")
        access_mode = st.radio(
            "Access distance",
            ["Walking  ½ mi", "Driving  1 mi"],
            index=0,
            label_visibility="collapsed",
        )
        base_radius = 0.5 if access_mode.startswith("Walking") else 1.0
        st.caption(f"~{walk_minutes(base_radius)}-min walk")

    # Advanced options — collapsed, for power users
    with st.expander("Advanced options"):
        _adv_a, _adv_b = st.columns(2)
        with _adv_a:
            restrict_to_oz = st.checkbox(
                "Limit candidates to Opportunity Zones",
                value=False,
                help="Policy eligibility filter. OZ status never influences the need index.",
            )
            use_tiered = st.checkbox(
                "Flexible desert thresholds (ACS)",
                value=False,
                disabled=not acs.get_census_api_key(),
                help="Stricter threshold in low-car areas. Requires CENSUS_API_KEY.",
            )
        with _adv_b:
            grid_spacing = st.slider("Candidate spacing (m)", 100, 500, 200, 25)
            use_pop_weighted = st.checkbox(
                "Use 2020 Census block population",
                value=False,
                disabled=not acs.get_census_api_key(),
                help="Replaces area-based density with real block population. Requires CENSUS_API_KEY.",
            )

    extended_radius = base_radius
    if use_tiered:
        extended_radius = st.slider(
            "Extended radius (vehicle-accessible areas):",
            base_radius, 2.0, max(base_radius, 1.0), 0.05, format="%.2f mi",
        )

# Resolve strategy weights and prepare city data
_strat_weights = STRATEGY_PRESETS[strategy]
use_health_priorities = _strat_weights is not None
factor_weights = _strat_weights if use_health_priorities else dict(DEFAULT_WEIGHTS)
need_weights = NeedWeights(weights=factor_weights) if use_health_priorities else NeedWeights(weights={})

prepared_key = (CITY_KEY, strategy)
_in_demo_mode = st.session_state.get("using_demo", False) and st.session_state.get("prepared") is None

if st.session_state.get("prepared_key") != prepared_key and not _in_demo_mode:
    # Strategy changed — re-prepare with new weights. Skip in demo mode:
    # the user hasn't clicked Run yet so there's nothing to recompute.
    with st.spinner(f"Updating need scores for '{strategy}'…"):
        st.session_state["prepared"] = pipeline.prepare_city(
            CITY_KEY, need_weights, use_health_priorities=use_health_priorities
        )
        st.session_state["prepared_key"] = prepared_key
        st.session_state["result"] = None
        st.session_state["using_demo"] = False

prepared = st.session_state.get("prepared")

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
run_clicked = st.button(
    f"Run optimization: {int(num_pharmacies)} pharmacies · {strategy.split('(')[0].strip()}",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    opt = OptConfig(
        num_pharmacies=int(num_pharmacies),
        min_distance_miles=base_radius,
        desert_radius_miles=base_radius,
        use_tiered_radius=use_tiered,
        extended_radius_miles=extended_radius,
        use_population_weighted_coverage=use_pop_weighted,
        restrict_to_opportunity_zones=restrict_to_oz,
    )
    candidate_cfg = CandidateConfig(grid_spacing_meters=float(grid_spacing))
    pop_note = " (first run fetches Census block data, ~1 min)" if use_pop_weighted else ""

    with st.status(
        f"Optimizing {int(num_pharmacies)} pharmacy locations…{pop_note}", expanded=True
    ) as _status:
        # ── Lazy prepare_city() in demo mode ─────────────────────────────
        if prepared is None:
            st.write("Loading city data for the first time (one-time, ~30 s)…")
            _lazy_prepared = pipeline.prepare_city(
                CITY_KEY, need_weights, use_health_priorities=use_health_priorities
            )
            st.session_state["prepared"]         = _lazy_prepared
            st.session_state["prepared_default"] = _lazy_prepared
            st.session_state["prepared_key"]     = (CITY_KEY, strategy)
            st.session_state["using_demo"]       = False
            # Update map layers so section 01 uses the live-pipeline pharmacy data
            st.session_state["map_center"]       = _lazy_prepared.bundle.center
            st.session_state["map_pharmacies"]   = _lazy_prepared.bundle.pharmacies
            prepared = _lazy_prepared

        st.write(f"Generating candidate sites from OpenStreetMap land use (every {int(grid_spacing)} m)…")
        if restrict_to_oz:
            st.write("Filtering to federally designated Opportunity Zone tracts…")
        if use_pop_weighted:
            st.write("Fetching 2020 Census block population for coverage weighting…")
        st.write("Scoring coverage fractions across all census tracts…")
        st.write("Running greedy max-coverage selection (1−1/e ≈ 63% optimality guarantee)…")
        result_new = pipeline.run_optimization(prepared, opt, candidate_cfg)
        n_sel = len(result_new.candidates[result_new.candidates["selected"]])
        n_cand = result_new.summary["num_candidates_evaluated"]
        _status.update(
            label=f"✓ Selected {n_sel} sites from {n_cand:,} candidates",
            state="complete", expanded=False,
        )
    st.session_state["result"]      = result_new
    st.session_state["base_radius"] = base_radius

# ---------------------------------------------------------------------------
# 02 · Optimization results — rendered into _results_slot (appears above controls)
# ---------------------------------------------------------------------------
_result_radius = st.session_state.get("base_radius", 0.5)
result = st.session_state.get("result")
with _results_slot.container():
  if result is not None:
    st.divider()
    st.markdown(
        "## What changes after new pharmacies are added?\n\n"
        "The results below show which locations were selected, how much additional need "
        "they reach, and which communities see the largest change in access.\n\n"
        "The model gives each new location credit for the need it newly covers, so later "
        "selections are less likely to simply duplicate the impact of sites already chosen."
    )
    with st.container(border=True):
        section_header("02", "Optimization results")

        s = result.summary
        selected = result.candidates[result.candidates["selected"]]

        # Two rows of three — labels read in full, metrics don't truncate
        row1 = st.columns(3)
        row1[0].metric("New sites selected",          f"{s['num_selected']} / {s['num_requested']}")
        row1[1].metric("Newly covered residents",     fmt_int(s["newly_covered_population"]))
        row1[2].metric("High-need residents covered", fmt_int(s["high_need_newly_covered_population"]))
        row2 = st.columns(3)
        row2[0].metric(s["impact_label"],             s["impact_count"])
        row2[1].metric("Avg. residential coverage",  f"{s['avg_coverage_after_pct']:.1f}%")
        row2[2].metric("Coverage gain",               f"+{s['avg_coverage_after_pct'] - s['avg_coverage_before_pct']:.1f} pts")
        st.caption(s["radius_status"] + " · " + s["coverage_status"])

        # Plain-English summary of what the optimizer found
        _cov_before = s["avg_coverage_before_pct"]
        _cov_after  = s["avg_coverage_after_pct"]
        _cov_gain   = _cov_after - _cov_before
        _impact_noun = s["impact_label"].lower()
        st.markdown(
            f'<p class="ph-optimizer-narrative">'
            f'The optimizer evaluated <strong>{s["num_candidates_evaluated"]:,} candidate sites</strong> '
            f'across Chicago&rsquo;s residential and commercial land, '
            f'selecting the <strong>{s["num_selected"]} locations</strong> that together '
            f'maximize access for the highest-need residents. '
            f'An estimated <strong>{fmt_int(s["newly_covered_population"])} additional residents</strong> '
            f'gain a pharmacy within the chosen radius, including '
            f'<strong>{fmt_int(s["high_need_newly_covered_population"])}</strong> in '
            f'the city&rsquo;s highest-need tracts. '
            f'Average residential coverage rises from '
            f'<strong>{_cov_before:.1f}%</strong> to '
            f'<strong>{_cov_after:.1f}%</strong> '
            f'(+{_cov_gain:.1f}&nbsp;pts) across '
            f'<strong>{s["impact_count"]} {_impact_noun}</strong>.'
            f'</p>',
            unsafe_allow_html=True,
        )

        # Map ---------------------------------------------------------------
        subsection_header("Map")
        map_view_col, show_all_col = st.columns([2, 1])
        with map_view_col:
            map_view = st.radio(
                "View",
                ["After optimization", "Coverage gain", "Existing coverage only"],
                horizontal=True,
                help=(
                    "After optimization: need index + existing (green) + new sites (blue). "
                    "Coverage gain: teal gradient shows where each tract improved most. "
                    "Existing only: need index + baseline coverage before optimization."
                ),
            )
        with show_all_col:
            show_all = st.checkbox(
                "Show all candidates", value=False,
                help="Show every evaluated candidate site (can be slow with fine grid spacing).",
            )

        # Format tract columns (shared by all map modes)
        tract_res = result.tract_residential.copy()
        tract_res["covered_display"] = (tract_res["coverage_after_pct"] >= 50).map(yes_no)
        for col, fmt in [
            ("normalized_need",          lambda v: "—" if pd.isna(v) else f"{v:.0f} / 100"),
            ("nearest_pharmacy_miles",   fmt_miles),
            ("TotalPopulation",          fmt_int),
            ("no_vehicle_pct",           fmt_pct),
            ("poverty_pct",              fmt_pct),
            ("age65_pct",                fmt_pct),
            ("ambulatory_disability_pct",fmt_pct),
            ("chronic_burden_pct",       fmt_pct),
            ("uninsured_pct",            fmt_pct),
        ]:
            tract_res[f"{col}_display"] = (
                tract_res[col].map(fmt) if col in tract_res.columns else "—"
            )
        tract_res["community_area_display"] = tract_res["community_area"].fillna("Unknown")
        tract_res["growth_pct_display"] = tract_res["growth_pct"].map(
            lambda v: "—" if pd.isna(v) else f"+{v:.1f} pts"
        )
        tract_res["coverage_before_pct_display"] = tract_res["coverage_before_pct"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.1f}%"
        )
        tract_res["coverage_after_pct_display"] = tract_res["coverage_after_pct"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.1f}%"
        )
        _newly_cov = (
            (tract_res["coverage_after_pct"] - tract_res["coverage_before_pct"])
            / 100.0 * tract_res["TotalPopulation"]
        ).clip(lower=0)
        tract_res["newly_covered_display"] = _newly_cov.map(
            lambda v: "—" if pd.isna(v) else f"{v:,.0f}"
        )

        # tooltip_cols for the results-section need-choropleth — defined here
        # (not inherited from section 01) so it's always in scope when results
        # render, regardless of whether the preview map section ran.
        tooltip_cols = [
            ("community_area_display",            "Community area:"),
            ("normalized_need_display",           "Need index (0–100):"),
            ("nearest_pharmacy_miles_display",    "Nearest pharmacy:"),
            ("covered_display",                   "Currently covered:"),
            ("TotalPopulation_display",           "Population:"),
            ("no_vehicle_pct_display",            "No vehicle:"),
            ("poverty_pct_display",               "Poverty rate:"),
            ("age65_pct_display",                 "Age 65+:"),
            ("ambulatory_disability_pct_display", "Mobility difficulty:"),
            ("chronic_burden_pct_display",        "Diabetes / hypertension:"),
            ("uninsured_pct_display",             "Uninsured:"),
        ]

        # Compute coverage shapes once (disk-cached after first run)
        with st.spinner("Computing coverage areas…"):
            existing_shape, existing_status = get_existing_coverage_shape(
                result.pharmacies, _result_radius
            )
            new_shape, new_status = get_new_coverage_shape(selected, _result_radius)

        # Blue layer = street-network isochrone for selected new sites.
        # Shown at full extent (no difference subtraction) so the coverage
        # corridor of each recommended site is clearly visible, even where it
        # overlaps the existing green layer.
        new_only = new_shape  # None when selected is empty

        def _add_selected_markers(folium_map) -> None:
            for _, row in selected.iterrows():
                popup_html = (
                    f"<b>#{int(row['rank'])}: {row['site_label']}</b><br>"
                    f"Community area: {row['community_area'] or '—'}<br>"
                    f"Newly covered: {fmt_int(row['population_reached'])} residents<br>"
                    f"High-need residents: {fmt_int(row['high_need_population_reached'])}<br>"
                    f"Opportunity Zone: {yes_no(row['is_opportunity_zone'])}<br>"
                    f"<em style='color:#6b747b'>High need · low current access</em>"
                )
                folium.Marker(
                    [row.geometry.y, row.geometry.x],
                    icon=numbered_icon(row["rank"]),
                    tooltip=folium.Tooltip(popup_html),
                ).add_to(folium_map)

        m = build_map(result.center, fit_to=tract_res)

        if map_view == "Coverage gain":
            gain_tooltip_cols = [
                ("community_area_display",        "Community area:"),
                ("growth_pct_display",            "Coverage gain:"),
                ("coverage_before_pct_display",   "Before:"),
                ("coverage_after_pct_display",    "After:"),
                ("newly_covered_display",         "Newly covered residents:"),
                ("TotalPopulation_display",       "Population:"),
            ]
            gain_choropleth_layer(tract_res, gain_tooltip_cols).add_to(m)
            GAIN_COLORMAP.add_to(m)
            _add_selected_markers(m)
            st_folium(m, height=560, use_container_width=True, returned_objects=[], key="map_result_gain")
            st.caption(
                "Teal intensity = coverage gained per tract (percentage points). "
                "Numbered markers = selected pharmacy sites. Hover any tract or marker for details."
            )

        elif map_view == "Existing coverage only":
            need_choropleth_layer(tract_res, tooltip_cols).add_to(m)
            NEED_COLORMAP.add_to(m)
            add_filled_geometry(m, existing_shape, color="#4ade80", opacity=0.45, tooltip="Existing coverage")
            st_folium(m, height=560, use_container_width=True, returned_objects=[], key="map_result_existing")
            st.caption(
                f"Baseline view: Pharmacy Need Index choropleth + existing coverage ({existing_status}). "
                "No new sites shown. Use 'After optimization' to see recommendations."
            )

        else:  # "After optimization" (default)
            need_choropleth_layer(tract_res, tooltip_cols).add_to(m)
            NEED_COLORMAP.add_to(m)
            if show_all:
                not_selected = result.candidates[~result.candidates["selected"]]
                folium.GeoJson(
                    not_selected[["label", "geometry"]].__geo_interface__,
                    marker=folium.CircleMarker(
                        radius=2, color="#6b747b", fill=True, fill_opacity=0.5
                    ),
                    tooltip=folium.GeoJsonTooltip(fields=["label"]),
                ).add_to(m)
            add_filled_geometry(m, existing_shape, color="#4ade80", opacity=0.45, tooltip="Existing coverage")
            if new_only is not None:
                add_filled_geometry(
                    m, new_only, color="#38bdf8", opacity=0.65,
                    tooltip="Newly covered by selected sites",
                )
            _add_selected_markers(m)
            st_folium(m, height=560, use_container_width=True, returned_objects=[], key="map_result_after_opt")
            st.caption(
                "Green: existing pharmacy coverage. Blue: area newly covered by the selected sites. "
                "Hover a numbered marker for site details."
            )

        # Selected sites table ----------------------------------------------
        subsection_header("Selected pharmacy sites")
        site_table = selected[
            ["rank", "site_label", "community_area", "is_opportunity_zone",
             "population_reached", "high_need_population_reached",
             "nearest_pharmacy_miles"]
        ].sort_values("rank").copy()
        site_table["rank"]                         = site_table["rank"].astype(int)
        site_table["community_area"]               = site_table["community_area"].fillna("—")
        site_table["is_opportunity_zone"]          = site_table["is_opportunity_zone"].map(yes_no)
        site_table["population_reached"]           = site_table["population_reached"].round(0).astype(int).map(fmt_int)
        site_table["high_need_population_reached"] = site_table["high_need_population_reached"].round(0).astype(int).map(fmt_int)
        site_table["nearest_pharmacy_miles"]       = site_table["nearest_pharmacy_miles"].map(fmt_miles)
        site_table = site_table.rename(columns={
            "rank":                        "#",
            "site_label":                  "Site",
            "community_area":              "Community area",
            "is_opportunity_zone":         "OZ?",
            "population_reached":          "Newly covered",
            "high_need_population_reached":"High-need covered",
            "nearest_pharmacy_miles":      "Nearest pharmacy",
        })
        render_dark_table(site_table, rank_col="#")
        st.download_button(
            "Download selected sites (CSV)",
            site_table.to_csv(index=False),
            file_name="selected_pharmacy_sites.csv",
        )

        with st.expander("Community & equity breakdown", expanded=False):
            if s.get("equity_available"):
                st.markdown(
                    '<span class="ph-eq-label">Equity check: '
                    'reported after optimization, never a scoring input</span>',
                    unsafe_allow_html=True,
                )
                eq_cols = st.columns(3)
                eq_cols[0].metric(
                    "Newly covered in majority-Black areas",
                    f"{s['equity_pct_majority_black']:.0f}%",
                )
                eq_cols[1].metric(
                    "…in majority-Hispanic/Latino areas",
                    f"{s['equity_pct_majority_hispanic']:.0f}%",
                )
                if not selected.empty:
                    oz_share = float(selected["is_opportunity_zone"].mean() * 100.0)
                    eq_cols[2].metric("Selected sites in Opportunity Zones", f"{oz_share:.0f}%")
            elif use_health_priorities:
                st.caption("Equity check unavailable. Requires a Census API key.")

            # Community impact table
            subsection_header("Community impact")
            if result.community_impact is not None:
                community_table = result.community_impact.copy()
                community_table["coverage_before_pct"]    = community_table["coverage_before_pct"].round(1)
                community_table["coverage_after_pct"]     = community_table["coverage_after_pct"].round(1)
                community_table["coverage_gain_pct"]      = community_table["coverage_gain_pct"].round(1)
                community_table["newly_covered_population"] = (
                    community_table["newly_covered_population"].round(0).astype(int)
                )
                if "majority_black" in community_table.columns:
                    community_table["equity_flag"] = community_table.apply(
                        lambda r: ", ".join(
                            f for f, v in [
                                ("Majority-Black", r["majority_black"]),
                                ("Majority-Hispanic/Latino", r["majority_hispanic"]),
                            ] if v
                        ) or "—",
                        axis=1,
                    )
                    community_table = community_table.drop(columns=["majority_black", "majority_hispanic"])
                community_table = community_table.drop(columns=["population"]).rename(columns={
                    "community_area":          "Community area",
                    "coverage_before_pct":     "Before (%)",
                    "coverage_after_pct":      "After (%)",
                    "coverage_gain_pct":       "Gain (pts)",
                    "newly_covered_population":"Newly covered",
                    "selected_site_ranks":     "Sites",
                    "equity_flag":             "Equity",
                })
                render_dark_table(community_table)
            else:
                st.caption("Community-area data unavailable for this run.")

# ---------------------------------------------------------------------------
# 04 · Explore further (sensitivity — collapsed by default)
# ---------------------------------------------------------------------------
if result is not None:
    st.divider()
    st.markdown('<span class="ph-explore-label">Explore further</span>', unsafe_allow_html=True)
    with st.expander("Sensitivity analysis: how stable are these recommendations?", expanded=False):
        st.markdown(
            "Reruns the optimizer across **20 random weight combinations** "
            "plus the **3 named strategy presets** (23 total runs) and measures how "
            "often each candidate site appears. Sites selected in 80%+ of runs are "
            "robust to weighting disagreements."
        )

        sa_key = (
            CITY_KEY,
            round(base_radius, 4),
            int(num_pharmacies),
            restrict_to_oz,
            int(grid_spacing),
        )
        if st.button("Run sensitivity analysis (~30–60 seconds)", key="run_sensitivity"):
            with st.status("Running sensitivity analysis…", expanded=True) as _sa_status:
                st.write("Generating candidates and computing coverage fractions…")
                st.write("Running optimizer with 20 random Dirichlet weight samples…")
                st.write("Running optimizer with 3 named strategy presets…")
                sa_result = pipeline.run_sensitivity_analysis(
                    prepared, OptConfig(
                        num_pharmacies=int(num_pharmacies),
                        min_distance_miles=base_radius,
                        desert_radius_miles=base_radius,
                        use_tiered_radius=use_tiered,
                        extended_radius_miles=extended_radius,
                        use_population_weighted_coverage=use_pop_weighted,
                        restrict_to_opportunity_zones=restrict_to_oz,
                    ),
                    CandidateConfig(grid_spacing_meters=float(grid_spacing)),
                    n_random_samples=20,
                )
                st.session_state["sa_result"] = sa_result
                st.session_state["sa_key"] = sa_key
                n_high = sa_result["high_stability_count"]
                _sa_status.update(
                    label=f"✓ Complete: {n_high} site{'s' if n_high != 1 else ''} selected in ≥80% of all runs",
                    state="complete",
                    expanded=False,
                )

        if st.session_state.get("sa_key") == sa_key and st.session_state.get("sa_result"):
            sa = st.session_state["sa_result"]
            sa_candidates = sa["candidates"]
            n_high = sa["high_stability_count"]
            total = sa["total_runs"]

            if n_high > 0:
                st.markdown(
                    f'<p class="ph-optimizer-narrative">'
                    f'<strong>{n_high} site{"s" if n_high != 1 else ""}</strong> '
                    f'{"were" if n_high != 1 else "was"} selected in '
                    f'<strong>80%+ of all {total} optimizer runs</strong>, '
                    f'across {sa["n_random"]} random weight profiles and '
                    f'{len(sa["strategy_names"])} named strategies '
                    f'({", ".join(sa["strategy_names"])}). '
                    f'These sites represent the most robust recommendations.'
                    f'</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p class="ph-optimizer-narrative">'
                    f'No single site appeared in 80%+ of all {total} runs. '
                    f'The recommendations are sensitive to the chosen weighting.'
                    f'</p>',
                    unsafe_allow_html=True,
                )

            subsection_header("Site stability map")
            st.caption(
                f"Each dot is a candidate site. Bright teal = selected in most of the {total} runs. "
                "Dim = selected only under specific weight assumptions."
            )

            sa_candidates["stability_pct"] = (sa_candidates["stability"] * 100).round(1)
            m_sa = build_map(prepared.bundle.center)
            for _, row in sa_candidates.iterrows():
                stab = float(row["stability"])
                if stab <= 0:
                    continue
                r_hex = int(107 + (79 - 107) * stab)
                g_hex = int(116 + (168 - 116) * stab)
                b_hex = int(123 + (154 - 123) * stab)
                color = f"#{r_hex:02x}{g_hex:02x}{b_hex:02x}"
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=3 + int(stab * 7),
                    color=color, fill=True, fill_color=color, fill_opacity=0.85,
                    tooltip=folium.Tooltip(
                        f"<b>{row['label']}</b><br>Selected in {row['stability_pct']:.0f}% of runs"
                    ),
                ).add_to(m_sa)
            st_folium(m_sa, height=480, use_container_width=True, returned_objects=[])

            top_stable = (
                sa_candidates[sa_candidates["stability"] > 0]
                .sort_values("stability", ascending=False).head(10).copy()
            )
            if not top_stable.empty:
                subsection_header("Most consistently recommended sites")
                render_dark_table(pd.DataFrame({
                    "Site": top_stable["label"].values,
                    "Selected in": top_stable["stability_pct"].map(lambda v: f"{v:.0f}% of runs").values,
                }))

# ---------------------------------------------------------------------------
# Methodology overview (concise — full reference in the Methodology tab)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Methodology, data sources & assumptions", expanded=False):
    st.caption(
        "This section gives a concise overview of how the planner works. "
        "For the full optimization formulation, algorithm design, data sources, "
        "and analytical assumptions, see the **Methodology & Model Design** tab."
    )
    section_header("", "Pharmacy Need Index")
    st.markdown(
        "Six factors, each **percentile-ranked across Chicago's 801 census tracts** before "
        "combining, so vehicle ownership (%), disease prevalence (%), and poverty rate (%) "
        "become directly comparable. The result is scaled by tract population:"
    )
    render_dark_table(pd.DataFrame([
        ("No vehicle access",                   "ACS B08201", "28%", "Primary transportation barrier"),
        ("Poverty rate",                        "ACS B17001", "22%", "Economic vulnerability"),
        ("Chronic medication burden",           "CDC PLACES", "20%", "Diabetes + hypertension prevalence"),
        ("Age 65+",                             "ACS DP05",   "13%", "Higher prescription volume"),
        ("Mobility / ambulatory disability",    "ACS S1810",  "12%", "Physical barrier to reaching pharmacy"),
        ("Uninsured rate",                      "ACS DP03",   " 5%", "Financial barrier (weakest pharmacy predictor)"),
    ], columns=["Factor", "Source", "Default weight", "Rationale"]))
    st.markdown(
        "**Formula:** `PNI(tract) = Population × Σ( percentile_rank(factor_i) × weight_i ) / W`  \n"
        "where *W* = sum of weights for available factors. "
        "The 0–100 display scale used for map coloring is computed afterward "
        "and is never fed back into the optimizer."
    )

    section_header("", "Location Optimizer")
    st.markdown(
        "Greedy weighted maximum coverage: picks sites one at a time, each time choosing "
        "the candidate that covers the most remaining unmet need. Marginal gains are "
        "recomputed after every selection, so earlier picks properly discount later ones.\n\n"
        "Coverage fractions use fast Euclidean buffers for scoring thousands of candidates; "
        "the final map shows real drive-network isochrones for selected sites.\n\n"
        "**Race/ethnicity** is never an optimization input. It is reported after the fact only. "
        "**Opportunity Zones** act as an eligibility constraint applied before scoring, not a reward signal."
    )

    st.caption(
        "For the full optimization formulation, algorithm pseudocode, approximation guarantee, "
        "data sources, and design decisions, see the **Methodology & Model Design** tab."
    )

_planner_tab.__exit__(None, None, None)

# ---------------------------------------------------------------------------
# Methodology & Model Design tab — full technical reference
# ---------------------------------------------------------------------------
with _method_tab:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem;">
          <span class="ph-eyebrow">Technical Reference</span>
          <h1 class="ph-hero-title" style="font-size:2rem">Methodology &amp; Model Design</h1>
          <p class="ph-hero-body">
            A full account of every modelling decision: the problem formulation,
            algorithm, data sources, design trade-offs, and known limitations.
            Written for a technical audience.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 1. How the model works ────────────────────────────────────────────────
    section_header("1", "How the model works")
    st.markdown(
        "The model answers a single question: *given a fixed budget of new pharmacy "
        "locations, which sites serve the most people who need them most?*\n\n"
        "It has two stages:\n\n"
        "1. **Score every census tract** by how badly it needs another pharmacy. "
        "A composite of six socioeconomic and health factors, each percentile-ranked "
        "so different units become comparable, then scaled by population.\n\n"
        "2. **Select sites greedily.** Starting from nothing, pick one candidate at a "
        "time: whichever site covers the most remaining unmet need. Repeat until the "
        "requested number of sites is reached."
    )

    # ── 2. Pharmacy Need Index ────────────────────────────────────────────────
    st.divider()
    section_header("2", "Pharmacy Need Index (PNI)")

    st.markdown(
        "The index measures **medication-access burden**, not general health status. "
        "Each of the six factors is **percentile-ranked within Chicago's 801 census tracts** "
        "before combining. A vehicle-ownership percentage, a disease prevalence, "
        "and a poverty rate (different units, different scales) all become 0–1 scores "
        "measuring relative disadvantage within the city."
    )

    render_dark_table(pd.DataFrame([
        ("No vehicle access",                   "ACS B08201",  "28%",
         "Primary transportation barrier. Households without a car face a hard distance cutoff"),
        ("Poverty rate",                        "ACS B17001",  "22%",
         "Economic vulnerability. Reduced ability to pay for transport or OTC alternatives"),
        ("Chronic medication burden",           "CDC PLACES",  "20%",
         "Diabetes + hypertension prevalence: conditions requiring monthly prescription refills"),
        ("Age 65+",                             "ACS DP05",    "13%",
         "Older adults fill ~3x more prescriptions than working-age adults"),
        ("Mobility / ambulatory disability",    "ACS S1810",   "12%",
         "Physical barrier. Ambulatory difficulty makes a long walk to a pharmacy prohibitive"),
        ("Uninsured rate",                      "ACS DP03",    " 5%",
         "Financial barrier, but weighted low: uninsured residents still need pharmacies"),
    ], columns=["Factor", "Source", "Default weight", "Rationale"]))

    st.markdown(
        r"""
**Formula**

$$\text{PNI}_i = P_i \times \frac{1}{W} \sum_k \bigl( w_k \times r_{ik} \bigr)$$

where:
- $P_i$ = total population of tract $i$
- $r_{ik}$ = percentile rank of tract $i$ on factor $k$ (0 = lowest need, 1 = highest)
- $w_k$ = user-configured weight for factor $k$
- $W = \sum_k w_k$ over factors that are actually available (graceful degradation when a data source is absent)

The result `weighted_need_i` is fed directly into the optimizer.
The 0–100 display value shown on the map (`normalized_need`) is computed afterward:
`normalized_need_i = 100 × weighted_need_i / max(weighted_need)`.
It is never used in optimization.
        """
    )

    # ── 3. Location optimization ──────────────────────────────────────────────
    st.divider()
    section_header("3", "Location Optimization")

    subsection_header("Objective function")
    st.markdown(
        r"""
The model solves the **Weighted Maximum Coverage Location Problem (WMCLP)**:

$$\max_{S \subseteq C,\; |S| = K} \sum_{i \in T} u_i \cdot \mathbf{1}\!\left[\text{tract } i \text{ is covered by } S\right]$$

where:
- $T$ = set of census tracts, $C$ = set of candidate sites, $K$ = requested new pharmacies
- $u_i = N_i \times (1 - e_i)$ = **unmet need** of tract $i$:
  - $N_i$ = `weighted_need_i` = PNI score (population-scaled)
  - $e_i$ = existing coverage fraction of tract $i$ before any new sites
- Coverage is **fractional** (a continuous relaxation): a candidate partially overlapping a tract
  covers that fraction of the tract's unmet need

The fractional objective is submodular and monotone, which enables the greedy algorithm's guarantee.
        """
    )

    subsection_header("Greedy algorithm")
    st.markdown(
        "A greedy algorithm with **multiplicative marginal discounting** solves the problem "
        "in seconds on Chicago-sized data and is fully deterministic:"
    )
    st.code(
        """\
# Initialization
remaining_need[tract] = weighted_need[tract] × (1 − existing_coverage[tract])

# Greedy loop — one site per iteration
for pick in range(K):
    # Score every un-selected candidate
    for candidate in candidates − selected:
        # Skip if too close to an already-selected site
        if min_distance(candidate, selected) < spacing_threshold:
            continue
        gain[candidate] = Σ_tract  remaining_need[tract] × coverage_frac(candidate, tract)

    # Select the highest-gain candidate
    best = argmax(gain)
    selected.add(best)

    # Discount remaining need for newly covered tracts
    for tract, frac in coverage_frac(best, *):
        remaining_need[tract] ×= (1 − frac)
""",
        language="python",
    )

    subsection_header("Approximation guarantee")
    st.markdown(
        "The greedy algorithm for submodular maximization over a uniform cardinality constraint "
        "achieves a worst-case **1−1/e ≈ 63% optimality guarantee**, meaning the greedy "
        "solution is always at least 63% as good as the true optimum.\n\n"
        "This model adds a **minimum-spacing constraint** (candidates must be at least "
        "`desert_radius_miles` apart) to prevent the optimizer from clustering all sites "
        "in one high-need neighborhood. This constraint is beyond the formal guarantee's "
        "assumptions. In practice on Chicago-sized data the spacing constraint rarely "
        "binds. High-need tracts are geographically distributed, and the greedy "
        "solution closely approximates the unconstrained optimum."
    )

    subsection_header("Coverage fractions")
    st.markdown(
        "Coverage is computed as the **fraction of a tract's residential land area** that "
        "falls within the access radius of a candidate site (Euclidean buffer, not network "
        "distance). This is fast enough to score thousands of candidates across 801 tracts "
        "in under a second.\n\n"
        "When *'Use 2020 Census block population'* is enabled, each block's population "
        "replaces the area proxy: coverage becomes the share of block-level population "
        "within radius, weighted by population. This requires a Census API key and is "
        "slower but more accurate in tracts with irregular residential density.\n\n"
        "**Coverage and isochrones are intentionally separate:**  \n"
        "Scoring uses fast Euclidean buffers. Visualization uses real street-network "
        "isochrones (drive network, 40 m buffer around reachable road edges). "
        "The isochrones shown on the final map are pre-computed for all candidate sites "
        "and loaded by coordinate key. They are never used as scoring inputs."
    )

    # ── 4. Siting constraints ─────────────────────────────────────────────────
    st.divider()
    section_header("4", "Siting Constraints")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Minimum spacing**  \n"
            "Each new site must be at least `desert_radius_miles` (Euclidean distance) "
            "from every already-selected site. This prevents the optimizer from stacking "
            "all picks in a single dense high-need cluster, which would saturate one "
            "neighborhood while leaving others uncovered.\n\n"
            "**Candidate generation**  \n"
            "Candidates are programmatically generated on a regular grid (default 200 m "
            "spacing) clipped to residential and commercial land use from OpenStreetMap. "
            "Points within one access radius of an existing pharmacy are excluded before "
            "scoring. Adding a pharmacy right next to an existing one provides no new coverage."
        )
    with col2:
        st.markdown(
            "**Opportunity Zone filter (opt-in)**  \n"
            "When enabled, all candidate sites outside federally designated "
            "Opportunity Zone tracts are removed from the pool *before* any scoring. "
            "OZ status never appears in the need index. The constraint is on "
            "*eligibility*, not on *desirability*. The optimizer still maximizes "
            "need coverage; it simply can only choose from the OZ-eligible subset.\n\n"
            "**Interpretation**: OZ filtering answers the question *'where should we "
            "build if we also want to attract tax-incentivized capital?'*, not *'where "
            "is need highest?'* The two questions often have different answers."
        )

    # ── 5. Deliberate design choices ─────────────────────────────────────────
    st.divider()
    section_header("5", "Deliberate Design Choices")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Race/ethnicity: reported, never scored**  \n"
            "The equity section shows what share of newly covered residents live in "
            "majority-Black or majority-Hispanic/Latino tracts, a post-hoc observation "
            "reported after the optimizer has already made its selections, not an input "
            "to the optimization. The same equity outcomes emerge naturally from the "
            "socioeconomic factors without making race a variable the algorithm can "
            "directly chase. This is the methodologically correct separation: the optimizer "
            "maximizes coverage of need; the equity audit checks who benefited.\n\n"
            "**Narrow variable set**  \n"
            "Obesity, smoking, depression, asthma, COPD, high cholesterol, dental visits, "
            "and vision disability are all available in CDC PLACES but deliberately "
            "excluded. They don't reliably predict *recurring pharmacy need* the way "
            "diabetes and hypertension do. Conditions that require monthly prescription "
            "refills are the reliable predictor. Adding correlated variables adds noise without improving signal."
        )
    with col_b:
        st.markdown(
            "**Opportunity Zones: constraint, not reward**  \n"
            "See §4. OZ status is never in the need index, only ever an eligibility gate.\n\n"
            "**Percentile ranking before combining**  \n"
            "Without ranking, a factor measured as a percentage (0–100%) and one "
            "measured as a count (0–5,000 people) would produce wildly incomparable "
            "values, and whichever happened to have larger raw numbers would dominate "
            "the index. Percentile ranking maps every factor to [0, 1] regardless of "
            "original units or scale, so weights become meaningful.\n\n"
            "**Buffers for speed, isochrones for presentation**  \n"
            "A 0.5-mile Euclidean buffer is a good proxy for a 10-minute walk in a "
            "grid-plan city like Chicago. Computing real street-network isochrones for "
            "7,283 candidate sites would take 30+ minutes; buffers take under a second. "
            "The final visualization uses pre-computed drive-network isochrones for the "
            "selected sites only. The map shows what those sites actually cover, "
            "while the optimizer used buffers to find them."
        )

    # ── 6. Data sources ───────────────────────────────────────────────────────
    st.divider()
    section_header("6", "Data Sources & Freshness")
    st.markdown(
        "Every data layer is open and publicly reproducible."
    )
    render_dark_table(pd.DataFrame([
        ("CDC PLACES (2023 release)",
         "Chronic medication burden: diabetes + hypertension tract-level prevalence",
         "2021 model year",
         "Annual",
         "Socrata API · data.cdc.gov · local CSV fallback for Chicago"),
        ("Census ACS 5-Year (2022)",
         "Five need factors: vehicle access, poverty, age 65+, mobility, uninsured; race/ethnicity for equity audit",
         "2018–2022 survey pool",
         "Annual",
         "Census API (CENSUS_API_KEY) · graceful PLACES-only fallback if absent"),
        ("TIGERweb 2020 Tract Boundaries",
         "Tract geometries for all spatial joins",
         "2020 decennial",
         "Fixed (2020 census geography)",
         "Census TIGERweb REST · tigerweb.geo.census.gov · disk-cached"),
        ("HUD Opportunity Zones",
         "OZ eligibility filter: which tracts are federally designated",
         "2018 designation",
         "Fixed (2017 TCJA designation, no amendments)",
         "HUD ArcGIS REST · services.arcgis.com"),
        ("OpenStreetMap",
         "Existing pharmacy locations; residential/commercial land for candidate grid",
         "Live crowdsourced snapshot",
         "Continuous",
         "osmnx + Overpass API · disk-cached per city after first fetch"),
        ("2020 Census Blocks",
         "Block-level population for population-weighted coverage (opt-in)",
         "2020 decennial",
         "Fixed",
         "Census API · only fetched when option is enabled"),
        ("Chicago Data Portal",
         "77 community area boundaries for community impact table",
         "Fixed (official city boundaries)",
         "Rarely changes",
         "Public GeoJSON · data.cityofchicago.org"),
    ], columns=["Source", "Powers in this model", "Vintage", "Update frequency", "Access method"]))

    # ── 7. Assumptions & data quality ────────────────────────────────────────
    st.divider()
    section_header("7", "Assumptions & Data Quality")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(
            "**Modelled estimates, not measurements**  \n"
            "CDC PLACES disease prevalences are derived from BRFSS survey responses "
            "via Bayesian small-area estimation. Tracts with fewer than ~1,000 residents "
            "carry higher uncertainty. The point estimate is used, not the confidence interval.\n\n"
            "**ACS margins of error**  \n"
            "5-Year ACS estimates are based on pooled surveys, not complete counts. "
            "The need index uses point estimates. Tracts near index boundaries should be "
            "treated as approximate, not precise cutoffs.\n\n"
            "**OSM pharmacy completeness**  \n"
            "OpenStreetMap's Chicago coverage is high but not exhaustive. Independent "
            "and informal pharmacies are more likely to be missing than chain pharmacies. "
            "Undercounted pharmacies bias the model toward over-estimating desert extent."
        )
    with col_q2:
        st.markdown(
            "**Tract-level resolution**  \n"
            "Both ACS and PLACES publish at census tract granularity (~4,000 residents). "
            "Real within-tract variation is invisible. A tract scored as high-need "
            "may have well-served pockets and severely underserved corners.\n\n"
            "**Area-based density (default)**  \n"
            "Without the Census block option, population is assumed uniformly distributed "
            "across residential land within each tract. In reality, density varies. "
            "Enable *'Use 2020 Census block population'* for population-weighted coverage.\n\n"
            "**Static pharmacy and OZ snapshots**  \n"
            "Existing pharmacies come from the data snapshot at load time. Openings, "
            "closures, and hours changes are not reflected. Opportunity Zone designations "
            "are fixed at the 2017 TCJA list; no post-2018 amendments are reflected.\n\n"
            "**Drive network as walk proxy**  \n"
            "The full pedestrian network for Chicago exceeds available RAM. The drive "
            "network respects real routing barriers (rivers, rail, highways) but slightly "
            "over-estimates walkable distance in some areas."
        )
