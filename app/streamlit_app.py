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
    colors=["#FFEC80", "#FEB24C", "#FC4E2A", "#BD0026"], vmin=0, vmax=100
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

    Loads one of the pre-computed per-N parquets committed to .cache/:
      new_sites_iso_n{N:02d}_r{radius:.2f}.parquet
    Each contains a single merged EPSG:4326 polygon (union of N ego_graph
    isochrones for the top-N demo sites in greedy rank order).

    This mirrors exactly how get_existing_coverage_shape loads the green
    layer — simple parquet read, gdf.geometry.iloc[0], no runtime graph
    loading, no fallback to circles.  If the parquet is absent (N > 10,
    custom radius, or different strategy picked) the map simply omits the
    blue layer rather than showing misleading straight-line buffers.
    """
    if selected_4326.empty:
        return None, ""

    # Session cache — keyed by (N, radius) since the per-N parquet is
    # independent of which specific candidates were chosen.
    n = len(selected_4326)
    cache_store = st.session_state.setdefault("new_coverage_cache", {})
    cache_key = (n, round(radius_miles, 4))
    if cache_key in cache_store:
        return cache_store[cache_key]

    place_name = get_osm_place_name(CITY_KEY)
    slug = cache_mod.slugify(place_name)
    iso_path = cache_mod.city_cache_dir(slug) / f"new_sites_iso_n{n:02d}_r{radius_miles:.2f}.parquet"

    if iso_path.exists():
        gdf = gpd.read_parquet(iso_path)
        if not gdf.empty:
            result = (gdf.geometry.iloc[0], "Street-network isochrone")
            cache_store[cache_key] = result
            return result

    # Parquet not available (N > 10 or custom settings) — omit blue layer.
    # Never fall back to straight-line buffers; circles are visually wrong.
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
            _boot.update(label="Ready — map loaded below", state="complete")

elif "prepared_default" not in st.session_state and st.session_state.get("prepared") is not None:
    st.session_state["prepared_default"] = st.session_state.get("prepared")

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
        that routine access to prescriptions becomes a real problem. Chicago is one
        of the largest cities where that gap can vary dramatically from one
        neighborhood to another.
      </p>
      <div class="ph-pullquote">So how big is the problem?</div>
      <p class="ph-hero-body">
        A good place to start is simply by looking at where every pharmacy in Chicago
        is today and asking how much of the city is actually within a reasonable
        walking distance. Here, that means about a 10-minute walk along the street
        network.
      </p>
      <p class="ph-hero-body" style="margin-top:0.85rem !important;">
        But distance alone only tells part of the story. Limited pharmacy access
        matters differently in a neighborhood where most households have a car than
        it does in one where many do not. The same is true for communities with
        higher poverty, older populations, mobility limitations, or greater rates
        of conditions that require regular prescriptions.
      </p>
      <div class="ph-map-lead">
        <p>The map below brings those pieces together to show where pharmacy access
        is already strong, where it falls off, and where limited access overlaps
        with greater community need.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 01 · Current pharmacy need (full-width map — always balanced-need baseline)
# ---------------------------------------------------------------------------
with st.container(border=True):
    section_header("01", "Where is pharmacy need highest?")
    st.caption(
        "Tracts colored by Pharmacy Need Index — a composite of vehicle access, "
        "poverty, chronic medication burden, age 65+, mobility, and uninsured rate, "
        "each percentile-ranked across Chicago's 801 census tracts. "
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
        NEED_COLORMAP.caption = "Pharmacy Need Index (0–100)"
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
    else:
        st.info("Map loading — if this persists, reload the page.", icon="🗺️")

# Stat burst — three context numbers after the map (not before)
st.markdown(
    """
    <div class="ph-stat-burst">
      <div class="ph-stat-card">
        <span class="ph-stat-number">~850</span>
        <span class="ph-stat-label">
          pharmacies serve Chicago's 2.7&nbsp;million residents — but access
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
          poverty, chronic disease burden, age 65+, mobility, and uninsured rate
          — each percentile-ranked across all 801&nbsp;Chicago tracts.
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
                help="Policy eligibility filter — OZ status never influences the need index.",
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
    f"Run optimization — {int(num_pharmacies)} pharmacies · {strategy.split('(')[0].strip()}",
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
            f'gain a pharmacy within the chosen radius — '
            f'<strong>{fmt_int(s["high_need_newly_covered_population"])}</strong> of them in '
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
                    f"<b>#{int(row['rank'])} — {row['site_label']}</b><br>"
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
                "No new sites shown — use 'After optimization' to see recommendations."
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
                    '<span class="ph-eq-label">Equity check — '
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
                st.caption("Equity check unavailable — requires a Census API key.")

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
    with st.expander("Sensitivity analysis — how stable are these recommendations?", expanded=False):
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
                    label=f"✓ Complete — {n_high} site{'s' if n_high != 1 else ''} selected in ≥80% of all runs",
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
                    f'<strong>80%+ of all {total} optimizer runs</strong> — '
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
# Methodology & data quality (always available at the bottom)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Methodology, data sources & assumptions"):
    section_header("", "Pharmacy Need Index")
    st.markdown(
        "The index measures **medication-access burden**, not general health status. "
        "Each factor is **percentile-ranked within Chicago's tracts** before combining — "
        "so a vehicle-ownership percentage, a disease prevalence rate, and a poverty rate "
        "(different units, different scales) become comparable:"
    )
    render_dark_table(pd.DataFrame([
        ("No vehicle access",                   "ACS B08201",  "28%", "Primary transportation barrier"),
        ("Poverty rate",                        "ACS B17001",  "22%", "Economic vulnerability"),
        ("Chronic medication burden",           "CDC PLACES",  "20%", "Diabetes + hypertension: conditions requiring ongoing prescriptions"),
        ("Age 65+",                             "ACS DP05",    "13%", "Older adults have higher prescription volume"),
        ("Mobility / ambulatory disability",    "ACS S1810",   "12%", "Physical barrier to reaching a pharmacy"),
        ("Uninsured rate",                      "ACS DP03",    " 5%", "Financial barrier; weighted low — weaker pharmacy-specific predictor"),
    ], columns=["Factor", "Source", "Default weight", "Rationale"]))
    st.markdown(
        "**Formula:** `PNI(tract) = Population × Σ( percentile_rank(factor_i) × weight_i )`  \n"
        "The population-scaled `weighted_need` goes directly into the optimizer. "
        "The `normalized_need` (0–100 display scale) is computed afterward for map coloring only — "
        "it is never fed back into the optimization."
    )

    section_header("", "Greedy Max-Coverage Algorithm")
    st.markdown(
        "The optimizer solves the **Weighted Maximum Coverage Location Problem (WMCLP)** "
        "— the standard operations research formulation for this class of siting decision.\n\n"
        "A greedy approach with marginal discounting achieves a provable **1−1/e ≈ 63% "
        "optimality guarantee** (from the submodularity of coverage functions), "
        "is deterministic, and runs in seconds on Chicago-sized data:\n\n"
        "1. Initialize `remaining_need[tract] = need_score × (1 − existing_coverage)`\n"
        "2. For each pick: choose the candidate with highest `Σ remaining_need[tract] × coverage_fraction(candidate, tract)`\n"
        "3. Update: `remaining_need[tract] ×= (1 − coverage_fraction(selected, tract))`\n"
        "4. Repeat until `num_pharmacies` sites selected\n\n"
        "The original CMU project used an epsilon-greedy \"Multi-Armed Bandit\" that computed "
        "each candidate's reward once before any selection — equivalent to sorting by static "
        "reward. Picking candidate #2 never accounted for what candidate #1 already covered. "
        "This rebuild fixes that by recomputing marginal gain after every selection."
    )

    section_header("", "Deliberate Design Choices")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Race/ethnicity: reported, never scored**  \n"
            "The equity check shows what share of newly covered residents live in "
            "majority-Black or majority-Hispanic/Latino tracts — a post-hoc observation, "
            "not an optimization input. The same equity outcomes emerge naturally from "
            "socioeconomic factors without making race a variable the optimizer can chase.\n\n"
            "**Opportunity Zones: constraint, not reward**  \n"
            "When OZ filtering is enabled, candidates outside designated tracts are removed "
            "from the pool before scoring. OZ status never appears in the need index — "
            "the constraint is on *eligibility*, not on *desirability*."
        )
    with col_b:
        st.markdown(
            "**Narrow variable set**  \n"
            "Obesity, smoking, depression, asthma, COPD, high cholesterol, dental visits, "
            "and vision disability are deliberately excluded. CDC PLACES publishes all of "
            "these, but they don't reliably predict *recurring pharmacy need* the way "
            "diabetes and hypertension do. More variables add noise, not signal.\n\n"
            "**Coverage via buffers, visualized via isochrones**  \n"
            "Scoring thousands of candidates with real street-network isochrones isn't "
            "practical at scale. The optimizer uses fast Euclidean buffers for candidate "
            "scoring; the final map shows actual drive-network isochrones for the selected "
            "sites and existing pharmacies."
        )

    section_header("", "Data Sources & Freshness")
    st.markdown(
        "Every data layer used in this analysis is open and publicly reproducible. "
        "The table below lists each source, what it powers, its vintage, and how it is accessed."
    )
    render_dark_table(pd.DataFrame([
        ("CDC PLACES (2023)",
         "Chronic medication burden factor (diabetes + hypertension prevalence)",
         "2021 model year",
         "Annual",
         "Socrata API — `data.cdc.gov` · local CSV fallback for Chicago"),
        ("Census ACS 5-Year (2022)",
         "5 need factors: vehicle access, poverty, age 65+, mobility, uninsured; race/ethnicity for equity reporting",
         "2018–2022 survey",
         "Annual",
         "Census API — requires `CENSUS_API_KEY`; graceful fallback to PLACES-only if absent"),
        ("TIGERweb — 2020 Tracts",
         "Census tract boundaries (geometry for all spatial joins)",
         "2020 decennial",
         "Fixed (2020 geographies)",
         "Census TIGERweb REST — `tigerweb.geo.census.gov`; disk-cached after first load"),
        ("HUD Opportunity Zones",
         "OZ eligibility filter (removes non-OZ candidates when enabled)",
         "2018 designation",
         "Fixed (2017 TCJA, no subsequent amendments)",
         "HUD ArcGIS REST — `services.arcgis.com`"),
        ("OpenStreetMap",
         "Existing pharmacy locations; residential/commercial land for candidate generation",
         "Live snapshot",
         "Crowdsourced (continuous)",
         "osmnx + Overpass API; disk-cached per city after first fetch"),
        ("2020 Census Blocks",
         "Block-level population for population-weighted coverage (opt-in)",
         "2020 decennial",
         "Fixed",
         "Census API — requires `CENSUS_API_KEY`; only fetched when option is enabled"),
        ("Chicago Data Portal",
         "77 community area boundaries for community impact rollup",
         "Fixed (official city boundaries)",
         "Rarely changes",
         "Public GeoJSON endpoint — `data.cityofchicago.org`"),
    ], columns=["Source", "Powers", "Vintage", "Update freq.", "Access"]))

    # Dynamic cache status — two slug conventions:
    # osm_place_name slug ("chicago-illinois-usa") for OSM-fetched layers
    # city_key slug ("chicago") for API-fetched layers (ACS, blocks, OZ, community areas)
    cache_dir = cache_mod.CACHE_DIR
    has_api_key = bool(acs.get_census_api_key())

    osm_slug = cache_mod.slugify("Chicago, Illinois, USA")   # "chicago-illinois-usa"
    key_slug = cache_mod.slugify(CITY_KEY)                   # "chicago"
    cache_files = {
        "OSM land use (zoned land)": f"{osm_slug}/zoned_land.parquet",
        "OSM pharmacies": f"{osm_slug}/pharmacies.parquet",
        "ACS need factors": f"{key_slug}/acs_need_factors.parquet",
        "Census block population": f"{key_slug}/census_blocks.parquet",
        "Opportunity Zone tracts": f"{key_slug}/opportunity_zone_tracts.parquet",
        "Community areas (Chicago)": f"{key_slug}/community_areas.parquet",
    }

    status_rows = []
    for label, rel_path in cache_files.items():
        p = cache_dir / rel_path
        if p.exists():
            mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
            status_rows.append((label, "✓ Cached", mtime.strftime("%Y-%m-%d %H:%M")))
        else:
            status_rows.append((label, "Will fetch on first use", "—"))

    subsection_header("Local cache status")
    st.caption(
        f"Data layers below are disk-cached in `pharmacy-desert-app/.cache/` after their first fetch. "
        f"Census API key: {'configured ✓' if has_api_key else 'not configured — ACS factors and block population unavailable'}."
    )
    render_dark_table(pd.DataFrame(status_rows, columns=["Data layer", "Status", "Cached at"]))

    section_header("", "Known Data Quality Considerations")
    st.markdown(
        "- **CDC PLACES uses modeled estimates**, not direct measurement. Values are derived from "
        "BRFSS survey responses via small-area estimation. Tracts with fewer than ~1,000 residents "
        "carry higher statistical uncertainty.\n"
        "- **ACS 5-Year estimates carry margins of error.** The need index uses point estimates. "
        "Tracts at the high-need boundary should be treated as approximate, not precise cutoffs.\n"
        "- **OSM pharmacy data may be incomplete.** OpenStreetMap's Chicago coverage is high but "
        "not exhaustive — independent or informal pharmacies are more likely to be missing than "
        "chain pharmacies. Missing pharmacies bias the model toward over-estimating desert area.\n"
        "- **Opportunity Zone designations are fixed at 2018.** The model uses the original "
        "2017 TCJA list. No amendments or de-designations after 2018 are reflected.\n"
        "- **Tract-level data**: both ACS and PLACES are published at census tract resolution. "
        "Real within-tract variation is invisible to the model.\n"
        "- **Area-based coverage by default**: population is assumed to be distributed evenly "
        "across residential land. Enable *'Use real neighborhood population data'* to substitute "
        "actual 2020 Census block population."
    )

    section_header("", "Limitations")
    st.markdown(
        "- **Drive network as proxy**: the full pedestrian network for Chicago exceeds available "
        "RAM; the drive network respects real routing barriers (rivers, highways, rail) "
        "but slightly over-estimates walkable access.\n"
        "- **Static pharmacy set**: existing pharmacies come from the loaded data snapshot. "
        "Pharmacy openings, closures, and hours are not reflected."
    )
