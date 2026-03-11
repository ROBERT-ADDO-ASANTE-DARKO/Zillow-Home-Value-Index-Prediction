#!/usr/bin/env python3
"""
Accra Home Price Index – Interactive Dash Dashboard
====================================================
Four-tab financial dashboard visualising the AHPI dataset.

Tabs
----
  Overview          – AHPI target variable with event annotations & overlays
  Macro Drivers     – Selectable macroeconomic regressor panel with normalise toggle
  Commodities       – Gold, Brent oil, and Cocoa price time series
  Regressor Explorer– Scatter / correlation analysis between any two variables
  Districts         – Per-district mid-market AHPI (compare all or drill into one)
  Prime Areas       – Per-prime-area AHPI (compare all or drill into one)
  Map               – Interactive OpenStreetMap of all 11 locations with hover metrics

Run
---
  python accra_dashboard.py          # http://localhost:8050
  python accra_dashboard.py --port 8080
"""

import io
import json
import math
import zipfile
import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# dash-leaflet — GIS choropleth maps
import dash_leaflet as dl
from dash_extensions.javascript import assign

# reportlab — PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.platypus import KeepTogether

warnings.filterwarnings("ignore")

# ── data ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "accra_home_price_index.csv",
)
DISTRICT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "accra_district_prices.csv",
)
PRIME_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "accra_prime_prices.csv",
)
DF = pd.read_csv(DATA_PATH, parse_dates=["ds"])
DF_DISTRICT = pd.read_csv(DISTRICT_DATA_PATH, parse_dates=["ds"])
DF_PRIME = pd.read_csv(PRIME_DATA_PATH, parse_dates=["ds"])
YEARS = list(range(DF["ds"].dt.year.min(), DF["ds"].dt.year.max() + 1))

# ── Prophet forecast outputs ───────────────────────────────────────────────────
_FORECASTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecasts")
DF_TEST_EVAL = pd.read_csv(
    os.path.join(_FORECASTS_DIR, "ahpi_test_eval.csv"), parse_dates=["ds"])
DF_FC_BEAR = pd.read_csv(
    os.path.join(_FORECASTS_DIR, "ahpi_forecast_bear.csv"), parse_dates=["ds"])
DF_FC_BASE = pd.read_csv(
    os.path.join(_FORECASTS_DIR, "ahpi_forecast_base.csv"), parse_dates=["ds"])
DF_FC_BULL = pd.read_csv(
    os.path.join(_FORECASTS_DIR, "ahpi_forecast_bull.csv"), parse_dates=["ds"])

# Pre-compute test-set accuracy metrics (used in the static info card)
_TEST_MAE  = (DF_TEST_EVAL["y"] - DF_TEST_EVAL["yhat"]).abs().mean()
_TEST_RMSE = ((DF_TEST_EVAL["y"] - DF_TEST_EVAL["yhat"]) ** 2).mean() ** 0.5
_TEST_MAPE = (
    (DF_TEST_EVAL["y"] - DF_TEST_EVAL["yhat"]).abs() / DF_TEST_EVAL["y"]
).mean() * 100
DISTRICTS   = DF_DISTRICT["district"].unique().tolist()
PRIME_AREAS = DF_PRIME["district"].unique().tolist()

# ── Prime areas forecast outputs ───────────────────────────────────────────────
PRIME_AREA_SLUGS: dict[str, str] = {
    "East Legon":           "east_legon",
    "Cantonments":          "cantonments",
    "Airport Residential":  "airport_residential",
    "Labone / Roman Ridge": "labone_roman_ridge",
    "Dzorwulu / Abelenkpe": "dzorwulu_abelenkpe",
    "Trasacco Valley":      "trasacco_valley",
}

_PRIME_TEST_EVALS: dict[str, pd.DataFrame] = {
    area: pd.read_csv(
        os.path.join(_FORECASTS_DIR, f"prime_test_eval_{slug}.csv"),
        parse_dates=["ds"],
    )
    for area, slug in PRIME_AREA_SLUGS.items()
}

_PRIME_FC: dict[tuple[str, str], pd.DataFrame] = {
    (sc, area): pd.read_csv(
        os.path.join(_FORECASTS_DIR, f"prime_forecast_{sc}_{slug}.csv"),
        parse_dates=["ds"],
    )
    for sc in ("bear", "base", "bull")
    for area, slug in PRIME_AREA_SLUGS.items()
}

_PRIME_TEST_SUMMARY = pd.read_csv(
    os.path.join(_FORECASTS_DIR, "prime_test_summary.csv"))


def _avg_dfs(dfs: list[pd.DataFrame], val_cols: list[str]) -> pd.DataFrame:
    """Average val_cols across a list of same-length DataFrames."""
    base = dfs[0][["ds"]].copy()
    for col in val_cols:
        base[col] = sum(df[col].values for df in dfs) / len(dfs)
    return base


_PRIME_TEST_EVAL_AGG = _avg_dfs(
    list(_PRIME_TEST_EVALS.values()),
    ["y", "yhat", "yhat_lower", "yhat_upper", "residual"],
)
_PRIME_FC_AGG: dict[str, pd.DataFrame] = {
    sc: _avg_dfs(
        [_PRIME_FC[(sc, area)] for area in PRIME_AREA_SLUGS],
        ["yhat", "yhat_lower", "yhat_upper", "trend"],
    )
    for sc in ("bear", "base", "bull")
}
_PRIME_HIST_AGG = DF_PRIME.groupby("ds")[["y"]].mean().reset_index()

# ── District forecast outputs ──────────────────────────────────────────────────
DISTRICT_SLUGS: dict[str, str] = {
    "Spintex Road": "spintex_road",
    "Adenta":       "adenta",
    "Tema":         "tema",
    "Dome":         "dome",
    "Kasoa":        "kasoa",
}

_DISTRICT_TEST_EVALS: dict[str, pd.DataFrame] = {
    d: pd.read_csv(
        os.path.join(_FORECASTS_DIR, f"district_test_eval_{slug}.csv"),
        parse_dates=["ds"],
    )
    for d, slug in DISTRICT_SLUGS.items()
}

_DISTRICT_FC: dict[tuple[str, str], pd.DataFrame] = {
    (sc, d): pd.read_csv(
        os.path.join(_FORECASTS_DIR, f"district_forecast_{sc}_{slug}.csv"),
        parse_dates=["ds"],
    )
    for sc in ("bear", "base", "bull")
    for d, slug in DISTRICT_SLUGS.items()
}

_DISTRICT_TEST_SUMMARY = pd.read_csv(
    os.path.join(_FORECASTS_DIR, "district_test_summary.csv"))

_DISTRICT_TEST_EVAL_AGG = _avg_dfs(
    list(_DISTRICT_TEST_EVALS.values()),
    ["y", "yhat", "yhat_lower", "yhat_upper", "residual"],
)
_DISTRICT_FC_AGG: dict[str, pd.DataFrame] = {
    sc: _avg_dfs(
        [_DISTRICT_FC[(sc, d)] for d in DISTRICT_SLUGS],
        ["yhat", "yhat_lower", "yhat_upper", "trend"],
    )
    for sc in ("bear", "base", "bull")
}
_DISTRICT_HIST_AGG = DF_DISTRICT.groupby("ds")[["y"]].mean().reset_index()

# Prime aggregate: mean of all six areas per month, with macro columns joined
# so the KPI callback can read exchange_rate / inflation / gold from one place.
_prime_agg = (
    DF_PRIME.groupby("ds")[["y", "price_ghs_per_sqm", "price_usd_per_sqm"]]
    .mean()
    .round(2)
    .reset_index()
)
DF_PRIME_FULL = _prime_agg.merge(
    DF[["ds", "exchange_rate_ghs_usd", "inflation_cpi_pct", "gold_price_usd"]],
    on="ds",
    how="left",
)

# ── palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "border":   "#30363d",
    "hover":    "#21262d",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
    "gold":     "#d4a017",
    "green":    "#3fb950",
    "red":      "#f85149",
    "blue":     "#58a6ff",
    "purple":   "#bc8cff",
    "teal":     "#39d0b1",
    "orange":   "#ffa657",
    "cocoa":    "#a0522d",
}

MACRO_META = {
    "gdp_growth_pct":         ("GDP Growth (%)",               C["green"]),
    "inflation_cpi_pct":      ("Inflation CPI (%)",            C["red"]),
    "exchange_rate_ghs_usd":  ("GHS / USD Exchange Rate",      C["gold"]),
    "lending_rate_pct":       ("Lending Rate (%)",             C["purple"]),
    "unemployment_pct":       ("Unemployment (%)",             C["teal"]),
    "govt_debt_pct_gdp":      ("Govt Debt (% GDP)",            C["orange"]),
    "remittances_pct_gdp":    ("Remittances (% GDP)",          C["blue"]),
    "fdi_pct_gdp":            ("FDI Net Inflows (% GDP)",      "#64b5f6"),
    "credit_private_pct_gdp": ("Private Credit (% GDP)",       "#e91e63"),
    "gross_capital_form_pct": ("Gross Capital Formation (%)",  "#26c6da"),
    "broad_money_pct_gdp":    ("Broad Money M2 (% GDP)",       "#607d8b"),
    "gdp_per_capita_usd":     ("GDP per Capita (USD)",         C["green"]),
}

DISTRICT_COLORS = {
    "Spintex Road": "#58a6ff",   # blue
    "Adenta":       "#3fb950",   # green
    "Tema":         "#d4a017",   # gold
    "Dome":         "#bc8cff",   # purple
    "Kasoa":        "#ffa657",   # orange
}

PRIME_COLORS = {
    "East Legon":           "#ff7b72",   # red-salmon
    "Cantonments":          "#e3b341",   # yellow-gold
    "Airport Residential":  "#d2a8ff",   # lavender
    "Labone / Roman Ridge": "#56d364",   # bright green
    "Dzorwulu / Abelenkpe": "#79c0ff",   # sky blue
    "Trasacco Valley":      "#f0883e",   # amber
}

DISTRICT_COLORS = {
    "Spintex Road": "#58a6ff",   # sky blue
    "Adenta":       "#56d364",   # green
    "Tema":         "#e3b341",   # gold
    "Dome":         "#d2a8ff",   # lavender
    "Kasoa":        "#f0883e",   # amber
}

# ── geographic coordinates (lat, lon) ─────────────────────────────────────────
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "Spintex Road": (5.620, -0.128),
    "Adenta":       (5.712, -0.168),
    "Tema":         (5.668,  0.017),
    "Dome":         (5.650, -0.235),
    "Kasoa":        (5.534, -0.420),
}

PRIME_COORDS: dict[str, tuple[float, float]] = {
    "East Legon":           (5.636, -0.151),
    "Cantonments":          (5.587, -0.186),
    "Airport Residential":  (5.605, -0.166),
    "Labone / Roman Ridge": (5.574, -0.174),
    "Dzorwulu / Abelenkpe": (5.597, -0.210),
    "Trasacco Valley":      (5.662, -0.135),
}

# Pre-compute Jan 2010 → Dec 2024 snapshots for map hover tooltips
def _make_snapshots(df: pd.DataFrame) -> dict:
    first = df.groupby("district").first()
    last  = df.groupby("district").last()
    out: dict = {}
    for loc in last.index:
        usd0 = first.loc[loc, "price_usd_per_sqm"]
        usd1 = last.loc[loc, "price_usd_per_sqm"]
        out[loc] = dict(
            ahpi    = last.loc[loc, "y"],
            ghs_sqm = last.loc[loc, "price_ghs_per_sqm"],
            usd_sqm = usd1,
            usd_pct = (usd1 - usd0) / usd0 * 100,
        )
    return out

DISTRICT_SNAP = _make_snapshots(DF_DISTRICT)
PRIME_SNAP    = _make_snapshots(DF_PRIME)

# Global bounds across the full 2010-2029 timeline for a consistent animation
# color scale (prevents each frame from looking the same due to per-frame rescaling).
# AHPI upper bound is set above the max forecast value (~928) for headroom.
_ANIM_BOUNDS: dict[str, tuple[float, float]] = {
    "ahpi": (
        float(min(DF_DISTRICT["y"].min(), DF_PRIME["y"].min())),
        950.0,
    ),
    "usd_sqm": (
        float(min(DF_DISTRICT["price_usd_per_sqm"].min(), DF_PRIME["price_usd_per_sqm"].min())),
        float(max(DF_DISTRICT["price_usd_per_sqm"].max(), DF_PRIME["price_usd_per_sqm"].max())),
    ),
    "ghs_sqm": (
        float(min(DF_DISTRICT["price_ghs_per_sqm"].min(), DF_PRIME["price_ghs_per_sqm"].min())),
        float(max(DF_DISTRICT["price_ghs_per_sqm"].max(), DF_PRIME["price_ghs_per_sqm"].max())),
    ),
}

# ── GeoJSON boundaries ─────────────────────────────────────────────────────────
_GEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "accra_boundaries.geojson")
with open(_GEO_PATH) as _f:
    BOUNDARIES_GEOJSON: dict = json.load(_f)

# Build lookup: name → feature index (for efficient access)
_BOUNDARY_IDX: dict[str, int] = {
    feat["properties"]["name"]: i
    for i, feat in enumerate(BOUNDARIES_GEOJSON["features"])
}

# ── GIS colour scales ──────────────────────────────────────────────────────────
# Continuous colour ramps for choropleth use (8-stop tuples)
_CS_GOLD_TO_RED = [
    [0.0,  "#0d1117"],
    [0.15, "#1a2a0a"],
    [0.30, "#2d5a1a"],
    [0.45, "#a0850a"],
    [0.60, "#d4a017"],
    [0.70, "#e07820"],
    [0.80, "#d05030"],
    [1.0,  "#f85149"],
]
_CS_BLUE_TO_GREEN = [
    [0.0,  "#0d1117"],
    [0.15, "#0a1e30"],
    [0.30, "#0d3b5e"],
    [0.45, "#1060a0"],
    [0.60, "#3090d0"],
    [0.75, "#30b060"],
    [0.90, "#3fb950"],
    [1.0,  "#a0f070"],
]
_CS_DIVERGE = [           # for growth vs decline (red → neutral → green)
    [0.0,  "#f85149"],
    [0.25, "#a03030"],
    [0.45, "#30363d"],
    [0.55, "#30363d"],
    [0.75, "#207840"],
    [1.0,  "#3fb950"],
]


def _hex_to_rgba(hex_color: str, alpha: float = 0.75) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _interpolate_color(value: float, stops: list) -> str:
    """Map a 0-1 value to a colour using the given colour-scale stops."""
    v = max(0.0, min(1.0, value))
    for i in range(len(stops) - 1):
        lo_v, lo_c = stops[i]
        hi_v, hi_c = stops[i + 1]
        if lo_v <= v <= hi_v:
            t = (v - lo_v) / (hi_v - lo_v + 1e-9)
            lo_h = lo_c.lstrip("#")
            hi_h = hi_c.lstrip("#")
            r = int(int(lo_h[0:2], 16) * (1 - t) + int(hi_h[0:2], 16) * t)
            g = int(int(lo_h[2:4], 16) * (1 - t) + int(hi_h[2:4], 16) * t)
            b = int(int(lo_h[4:6], 16) * (1 - t) + int(hi_h[4:6], 16) * t)
            return f"#{r:02x}{g:02x}{b:02x}"
    return stops[-1][1]


def _normalise(val: float, lo: float, hi: float) -> float:
    return (val - lo) / (hi - lo + 1e-9) if hi > lo else 0.5


def _build_price_geojson(metric: str = "price_usd_per_sqm") -> dict:
    """
    Return a copy of BOUNDARIES_GEOJSON with price data injected into
    each feature's properties (for Plotly choropleth and Leaflet styling).
    """
    all_snaps = {**DISTRICT_SNAP, **PRIME_SNAP}
    vals = [all_snaps[n].get(metric, 0) for n in all_snaps]
    lo, hi = min(vals), max(vals)

    gj = json.loads(json.dumps(BOUNDARIES_GEOJSON))   # deep copy
    for feat in gj["features"]:
        name = feat["properties"]["name"]
        snap = all_snaps.get(name, {})
        v    = snap.get(metric, 0)
        feat["properties"]["value"]     = round(v, 1)
        feat["properties"]["norm"]      = round(_normalise(v, lo, hi), 4)
        feat["properties"]["ahpi"]      = round(snap.get("ahpi", 0), 1)
        feat["properties"]["ghs_sqm"]   = round(snap.get("ghs_sqm", 0), 0)
        feat["properties"]["usd_sqm"]   = round(snap.get("usd_sqm", 0), 0)
        feat["properties"]["usd_pct"]   = round(snap.get("usd_pct", 0), 1)
        feat["properties"]["fill"]      = _interpolate_color(
            _normalise(v, lo, hi), _CS_GOLD_TO_RED)
    return gj, lo, hi


def _build_forecast_geojson(scenario: str = "base", year: int = 2027) -> dict:
    """
    Return a copy of BOUNDARIES_GEOJSON annotated with forecast AHPI and
    % growth vs Dec 2024 for the chosen scenario/year.
    """
    gj = json.loads(json.dumps(BOUNDARIES_GEOJSON))   # deep copy
    growths = []
    for feat in gj["features"]:
        name = feat["properties"]["name"]
        mkt  = "composite" if name not in (DISTRICTS + PRIME_AREAS) else name
        fc   = _get_fc_dec(mkt, scenario, year)
        hist = _get_hist_dec(mkt, 2024)
        if fc is not None and hist is not None:
            ahpi_now  = float(hist.get("y", 1))
            ahpi_fc   = float(fc["yhat"])
            growth    = (ahpi_fc - ahpi_now) / ahpi_now * 100
        else:
            growth = 0.0
        feat["properties"]["growth_pct"] = round(growth, 1)
        feat["properties"]["fc_ahpi"]    = round(float(fc["yhat"]), 1) if fc is not None else None
        growths.append(growth)

    lo, hi = min(growths), max(growths)
    mid    = (lo + hi) / 2
    for feat in gj["features"]:
        g    = feat["properties"]["growth_pct"]
        # Normalise: 0=lo, 0.5=mid, 1=hi (diverging from neutral)
        norm = _normalise(g, lo, hi)
        feat["properties"]["norm"]   = round(norm, 4)
        feat["properties"]["fill"]   = _interpolate_color(norm, _CS_BLUE_TO_GREEN)
    return gj, lo, hi


def _build_timeline_geojson(year: int, metric: str = "usd_sqm",
                             scenario: str = "base") -> tuple[dict, float, float]:
    """
    Build a GeoJSON snapshot for a specific year.

    - Years 2010-2024: pulls actual December values from historical CSVs.
    - Years 2025-2029: uses Prophet forecast (AHPI only; metric is ignored and
      forced to 'ahpi' because USD/GHS predictions are unavailable).

    Color normalisation uses _ANIM_BOUNDS (fixed global range) so that the
    animation frames are directly comparable across years.
    """
    col_map = {
        "usd_sqm": "price_usd_per_sqm",
        "ghs_sqm": "price_ghs_per_sqm",
        "ahpi":    "y",
    }
    # Forecast years carry AHPI only
    eff_metric = "ahpi" if year >= 2025 else (metric or "usd_sqm")
    col        = col_map.get(eff_metric, "price_usd_per_sqm")
    lo, hi     = _ANIM_BOUNDS.get(eff_metric, _ANIM_BOUNDS["ahpi"])

    gj = json.loads(json.dumps(BOUNDARIES_GEOJSON))   # deep copy
    for feat in gj["features"]:
        name = feat["properties"]["name"]

        if year <= 2024:
            if name in DISTRICTS:
                row = _HIST_DEC_DISTRICT.get(name, {}).get(year)
            elif name in PRIME_AREAS:
                row = _HIST_DEC_PRIME.get(name, {}).get(year)
            else:
                row = None

            if row is not None:
                v      = float(row[col])
                ahpi_v = float(row["y"])
                ghs_v  = float(row["price_ghs_per_sqm"])
                usd_v  = float(row["price_usd_per_sqm"])
            else:
                v = ahpi_v = ghs_v = usd_v = 0.0
        else:
            mkt = name if (name in DISTRICTS or name in PRIME_AREAS) else "composite"
            fc  = _get_fc_dec(mkt, scenario, year)
            ahpi_v = float(fc["yhat"]) if fc is not None else 0.0
            v      = ahpi_v
            ghs_v  = 0.0
            usd_v  = 0.0

        norm = _normalise(v, lo, hi)
        feat["properties"]["value"]     = round(v, 1)
        feat["properties"]["norm"]      = round(norm, 4)
        feat["properties"]["ahpi"]      = round(ahpi_v, 1)
        feat["properties"]["ghs_sqm"]   = round(ghs_v, 0)
        feat["properties"]["usd_sqm"]   = round(usd_v, 0)
        feat["properties"]["year"]      = year
        feat["properties"]["projected"] = year >= 2025
        feat["properties"]["fill"]      = _interpolate_color(norm, _CS_GOLD_TO_RED)

    return gj, lo, hi


# JavaScript style functions for dash-leaflet GeoJSON layers
_style_price = assign("""function(feature) {
    return {
        fillColor:   feature.properties.fill || '#30363d',
        fillOpacity: 0.72,
        color:       '#d4a017',
        weight:      1.5,
        opacity:     0.9,
        dashArray:   feature.properties.type === 'prime' ? '4 3' : null,
    };
}""")

_style_forecast = assign("""function(feature) {
    return {
        fillColor:   feature.properties.fill || '#30363d',
        fillOpacity: 0.72,
        color:       feature.properties.growth_pct >= 0 ? '#3fb950' : '#f85149',
        weight:      1.5,
        opacity:     0.9,
        dashArray:   feature.properties.type === 'prime' ? '4 3' : null,
    };
}""")

_on_each_feature_price = assign("""function(feature, layer) {
    var p = feature.properties;
    layer.bindTooltip(
        '<div style="background:#161b22;border:1px solid #30363d;padding:8px 12px;border-radius:6px;font-family:monospace;min-width:180px;">' +
        '<div style="color:#d4a017;font-weight:700;font-size:13px;margin-bottom:4px;">' + p.name + '</div>' +
        '<div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">' + p.type + '</div>' +
        '<hr style="border-color:#30363d;margin:5px 0"/>' +
        '<div style="color:#e6edf3;font-size:12px;">AHPI Dec 2024: <b>' + p.ahpi.toFixed(1) + '</b></div>' +
        '<div style="color:#e6edf3;font-size:12px;">GHS/sqm: <b>' + p.ghs_sqm.toLocaleString() + '</b></div>' +
        '<div style="color:#e6edf3;font-size:12px;">USD/sqm: <b>' + p.usd_sqm.toLocaleString() + '</b></div>' +
        '<div style="color:#3fb950;font-size:12px;">USD gain 2010–24: <b>+' + p.usd_pct.toFixed(1) + '%</b></div>' +
        '</div>',
        {sticky: true, opacity: 1}
    );
    layer.on('mouseover', function(e) { layer.setStyle({fillOpacity: 0.92, weight: 3}); });
    layer.on('mouseout',  function(e) { layer.setStyle({fillOpacity: 0.72, weight: 1.5}); });
}""")

_on_each_feature_forecast = assign("""function(feature, layer) {
    var p = feature.properties;
    var growthColor = p.growth_pct >= 0 ? '#3fb950' : '#f85149';
    var sign = p.growth_pct >= 0 ? '+' : '';
    layer.bindTooltip(
        '<div style="background:#161b22;border:1px solid #30363d;padding:8px 12px;border-radius:6px;font-family:monospace;min-width:190px;">' +
        '<div style="color:#d4a017;font-weight:700;font-size:13px;margin-bottom:4px;">' + p.name + '</div>' +
        '<div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">' + p.type + '</div>' +
        '<hr style="border-color:#30363d;margin:5px 0"/>' +
        '<div style="color:#e6edf3;font-size:12px;">Forecast AHPI: <b>' + (p.fc_ahpi ? p.fc_ahpi.toFixed(1) : "—") + '</b></div>' +
        '<div style="font-size:13px;font-weight:700;color:' + growthColor + ';">Growth vs 2024: ' + sign + p.growth_pct.toFixed(1) + '%</div>' +
        '</div>',
        {sticky: true, opacity: 1}
    );
    layer.on('mouseover', function(e) { layer.setStyle({fillOpacity: 0.92, weight: 3}); });
    layer.on('mouseout',  function(e) { layer.setStyle({fillOpacity: 0.72, weight: 1.5}); });
}""")


_on_each_feature_timeline = assign("""function(feature, layer) {
    var p = feature.properties;
    var proj = p.projected;
    var priceRows = proj
        ? '<div style="color:#58a6ff;font-size:11px;font-style:italic;margin-top:3px;">Projected — AHPI only</div>'
        : '<div style="color:#e6edf3;font-size:12px;">GHS/sqm: <b>' + (p.ghs_sqm ? p.ghs_sqm.toLocaleString() : '—') + '</b></div>' +
          '<div style="color:#e6edf3;font-size:12px;">USD/sqm: <b>' + (p.usd_sqm ? p.usd_sqm.toLocaleString() : '—') + '</b></div>';
    layer.bindTooltip(
        '<div style="background:#161b22;border:1px solid #30363d;padding:8px 12px;border-radius:6px;font-family:monospace;min-width:180px;">' +
        '<div style="color:#d4a017;font-weight:700;font-size:13px;margin-bottom:4px;">' + p.name + '</div>' +
        '<div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">' + p.type + ' · ' + p.year + '</div>' +
        '<hr style="border-color:#30363d;margin:5px 0"/>' +
        '<div style="color:#e6edf3;font-size:12px;">AHPI' + (proj ? ' (proj)' : '') + ': <b>' + p.ahpi.toFixed(1) + '</b></div>' +
        priceRows +
        '</div>',
        {sticky: true, opacity: 1}
    );
    layer.on('mouseover', function(e) { layer.setStyle({fillOpacity: 0.92, weight: 3}); });
    layer.on('mouseout',  function(e) { layer.setStyle({fillOpacity: 0.72, weight: 1.5}); });
}""")


COMMODITY_META = {
    "gold_price_usd":   ("Gold (USD / troy oz)",  C["gold"],   "left"),
    "oil_brent_usd":    ("Brent Oil (USD / bbl)", C["muted"],  "left"),
    "cocoa_price_usd":  ("Cocoa (USD / MT)",      C["cocoa"],  "right"),
}

KEY_EVENTS = [
    ("2014-01-01", "Cedi crisis\n(−30% vs USD)",       "below"),
    ("2016-03-01", "IMF bailout\n(USD 918 m ECF)",      "above"),
    ("2020-03-01", "COVID-19",                          "below"),
    ("2022-01-01", "Debt crisis /\ncurrency collapse",  "below"),
    ("2023-05-01", "IMF programme\n(USD 3 bn)",         "above"),
    ("2024-01-01", "Cocoa surge",                       "above"),
]

ALL_VARS = {
    "y":                     "AHPI (index)",
    **{k: v[0] for k, v in MACRO_META.items()},
    **{k: v[0] for k, v in COMMODITY_META.items()},
    "price_ghs_per_sqm":    "Price GHS/sqm",
    "price_usd_per_sqm":    "Price USD/sqm",
}

# ── base chart layout ─────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor=C["card"],
    plot_bgcolor=C["bg"],
    font=dict(family="'Inter', 'Segoe UI', Arial, sans-serif",
              color=C["text"], size=11),
    margin=dict(l=55, r=20, t=44, b=44),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C["hover"], font_color=C["text"],
                    bordercolor=C["border"], namelength=-1),
)

# Axis / legend defaults kept separate so figure builders can extend them
# without triggering duplicate-keyword errors when spreading BASE_LAYOUT.
BASE_XAXIS  = dict(showgrid=False, linecolor=C["border"],
                   tickcolor=C["muted"], tickfont=dict(size=10))
BASE_YAXIS  = dict(showgrid=True, gridcolor=C["border"],
                   linecolor=C["border"], tickfont=dict(size=10))
BASE_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                   borderwidth=1, font_size=10)

# ── helpers ───────────────────────────────────────────────────────────────────
def filter_df(start_yr, end_yr):
    return DF[(DF["ds"].dt.year >= start_yr) & (DF["ds"].dt.year <= end_yr)].copy()


def filter_df_district(start_yr, end_yr, district="all"):
    mask = (
        (DF_DISTRICT["ds"].dt.year >= start_yr) &
        (DF_DISTRICT["ds"].dt.year <= end_yr)
    )
    dff = DF_DISTRICT[mask].copy()
    if district != "all":
        dff = dff[dff["district"] == district]
    return dff


def filter_df_prime(start_yr, end_yr, area="all"):
    mask = (
        (DF_PRIME["ds"].dt.year >= start_yr) &
        (DF_PRIME["ds"].dt.year <= end_yr)
    )
    dff = DF_PRIME[mask].copy()
    if area != "all":
        dff = dff[dff["district"] == area]
    return dff


def add_event_lines(fig, dff):
    """Add vertical event annotation lines to a plain (non-subplots) figure."""
    for date_str, label, pos in KEY_EVENTS:
        dt = pd.Timestamp(date_str)
        if dt < dff["ds"].min() or dt > dff["ds"].max():
            continue
        ypos = 1.02 if pos == "above" else -0.06
        fig.add_vline(
            x=dt.timestamp() * 1000,
            line_width=1, line_dash="dot",
            line_color=C["muted"],
            opacity=0.5,
        )
        fig.add_annotation(
            x=dt, y=ypos,
            xref="x", yref="paper",
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=8, color=C["muted"]),
            align="center",
            bgcolor=C["hover"],
            bordercolor=C["border"],
            borderwidth=1,
            borderpad=3,
            opacity=0.88,
        )
    return fig


def linreg(x, y):
    """Return slope, intercept, r_squared for two arrays."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None, None, None
    coeffs = np.polyfit(x, y, 1)
    y_hat  = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return coeffs[0], coeffs[1], r2


# ── figure builders ───────────────────────────────────────────────────────────
def build_ahpi_fig(dff, overlays, show_events, segment="mid", dff_prime_full=None):
    fig = go.Figure()

    # ── choose main series by segment ─────────────────────────────────────────
    if segment == "prime" and dff_prime_full is not None:
        main_dff    = dff_prime_full
        main_color  = C["red"]
        fill_color  = "rgba(248,81,73,0.10)"
        main_label  = "AHPI – Prime avg"
        chart_title = ("<b>Accra Home Price Index</b>  ·  "
                       "<span style='color:#f85149'>Prime Areas Average</span>"
                       "  (GHS, 2015 = 100)")
    else:
        main_dff    = dff
        main_color  = C["gold"]
        fill_color  = "rgba(212,160,23,0.12)"
        main_label  = "AHPI – Mid-Market"
        chart_title = ("<b>Accra Home Price Index</b>  ·  "
                       "<span style='color:#d4a017'>Mid-Market</span>"
                       "  (GHS, 2015 = 100)")

    if segment == "both":
        chart_title = ("<b>Accra Home Price Index</b>  ·  "
                       "<span style='color:#d4a017'>Mid-Market</span>"
                       "  vs  "
                       "<span style='color:#f85149'>Prime avg</span>"
                       "  (GHS, 2015 = 100)")

    # ── area fill (main series) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=main_dff["ds"], y=main_dff["y"],
        fill="tozeroy", fillcolor=fill_color,
        line=dict(color=main_color, width=2.5),
        name=main_label,
        hovertemplate="%{y:.1f}<extra>" + main_label + "</extra>",
    ))

    # ── prime overlay when comparing both ─────────────────────────────────────
    if segment == "both" and dff_prime_full is not None:
        fig.add_trace(go.Scatter(
            x=dff_prime_full["ds"], y=dff_prime_full["y"],
            line=dict(color=C["red"], width=2, dash="dash"),
            name="AHPI – Prime avg",
            hovertemplate="%{y:.1f}<extra>Prime avg</extra>",
        ))

    # ── 2015 baseline ──────────────────────────────────────────────────────────
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.9, opacity=0.5,
                  annotation_text="2015 baseline",
                  annotation_font_color=C["muted"],
                  annotation_font_size=9)

    # ── price overlays (mid-market or prime only, not when comparing) ──────────
    if segment != "both":
        if "ghs" in overlays:
            fig.add_trace(go.Scatter(
                x=main_dff["ds"], y=main_dff["price_ghs_per_sqm"],
                line=dict(color=C["green"], width=1.6, dash="dash"),
                name="GHS / sqm",
                yaxis="y2",
                hovertemplate="GHS %{y:,.0f}<extra>GHS/sqm</extra>",
            ))
        if "usd" in overlays:
            fig.add_trace(go.Scatter(
                x=main_dff["ds"], y=main_dff["price_usd_per_sqm"],
                line=dict(color=C["blue"], width=1.6, dash="dot"),
                name="USD / sqm",
                yaxis="y3",
                hovertemplate="$%{y:,.0f}<extra>USD/sqm</extra>",
            ))

    fig.update_layout(
        BASE_LAYOUT,
        xaxis=BASE_XAXIS,
        yaxis=dict(**BASE_YAXIS, title="Index (2015 = 100)"),
        yaxis2=dict(title=dict(text="GHS / sqm", font=dict(color=C["green"])),
                    overlaying="y", side="right",
                    showgrid=False, tickformat=",",
                    tickfont=dict(color=C["green"], size=10)),
        yaxis3=dict(title=dict(text="USD / sqm", font=dict(color=C["blue"])),
                    overlaying="y", side="right",
                    anchor="free", position=0.97, showgrid=False,
                    tickfont=dict(color=C["blue"], size=10)),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        title=dict(text=chart_title, font_size=14, x=0.01),
        height=440,
    )
    if show_events:
        add_event_lines(fig, main_dff)
    return fig


def build_macro_fig(dff, selected_vars, normalise):
    if not selected_vars:
        selected_vars = ["inflation_cpi_pct", "exchange_rate_ghs_usd", "gdp_growth_pct"]

    fig = go.Figure()
    for var in selected_vars:
        if var not in MACRO_META:
            continue
        label, color = MACRO_META[var]
        series = dff[var].copy()
        if normalise:
            mn, mx = series.min(), series.max()
            series = (series - mn) / (mx - mn) * 100 if mx != mn else series * 0
        fig.add_trace(go.Scatter(
            x=dff["ds"], y=series,
            line=dict(color=color, width=2),
            name=label,
            hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
        ))

    y_title = "Normalised (0–100)" if normalise else "Value"
    fig.update_layout(
        BASE_LAYOUT,
        xaxis=BASE_XAXIS,
        yaxis=dict(**BASE_YAXIS, title=y_title),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.08),
        title=dict(text="<b>Macroeconomic Regressors</b>", font_size=14, x=0.01),
        height=420,
    )
    return fig


def build_macro_grid(dff):
    """Small-multiples grid: 4×3 sparklines for all macro variables."""
    keys   = list(MACRO_META.keys())
    ncols  = 3
    nrows  = int(np.ceil(len(keys) / ncols))
    titles = [MACRO_META[k][0] for k in keys]

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    for i, key in enumerate(keys):
        row = i // ncols + 1
        col = i %  ncols + 1
        _, color = MACRO_META[key]
        fig.add_trace(
            go.Scatter(x=dff["ds"], y=dff[key],
                       line=dict(color=color, width=1.5),
                       showlegend=False,
                       hovertemplate=f"%{{y:.2f}}<extra>{MACRO_META[key][0]}</extra>"),
            row=row, col=col,
        )
        fig.update_xaxes(showgrid=False, tickfont=dict(size=8),
                         row=row, col=col)
        fig.update_yaxes(showgrid=True, gridcolor=C["border"],
                         tickfont=dict(size=8), row=row, col=col)

    fig.update_layout(
        paper_bgcolor=C["card"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], size=10),
        margin=dict(l=40, r=10, t=55, b=30),
        height=520,
    )
    for ann in fig.layout.annotations:
        ann.font.size  = 9
        ann.font.color = C["muted"]
    return fig


def build_commodity_fig(dff):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Gold & Brent Crude Oil", "Cocoa Prices"),
        horizontal_spacing=0.10,
    )

    # Gold
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["gold_price_usd"],
        line=dict(color=C["gold"], width=2),
        name="Gold (USD/oz)",
        hovertemplate="$%{y:,.0f}<extra>Gold</extra>",
    ), row=1, col=1)

    # Brent
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["oil_brent_usd"],
        line=dict(color=C["muted"], width=1.8, dash="dash"),
        name="Brent Oil (USD/bbl)",
        yaxis="y2",
        hovertemplate="$%{y:.1f}<extra>Brent</extra>",
    ), row=1, col=1)

    # Cocoa (area fill)
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["cocoa_price_usd"],
        fill="tozeroy", fillcolor="rgba(160,82,45,0.15)",
        line=dict(color=C["cocoa"], width=2),
        name="Cocoa (USD/MT)",
        hovertemplate="$%{y:,.0f}<extra>Cocoa</extra>",
    ), row=1, col=2)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text="<b>Commodity Prices</b>  –  Key drivers of the Ghanaian economy",
                   font_size=14, x=0.01),
        yaxis=dict(**BASE_YAXIS, title="Gold (USD/oz)"),
        yaxis2=dict(title=dict(text="Brent (USD/bbl)", font=dict(color=C["muted"])),
                    overlaying="y", side="right",
                    showgrid=False,
                    tickfont=dict(color=C["muted"], size=10)),
        yaxis3=dict(**BASE_YAXIS, title="Cocoa (USD/MT)"),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.08),
        height=380,
    )
    for ann in fig.layout.annotations:
        ann.font.size  = 12
        ann.font.color = C["muted"]
    return fig


def build_scatter_fig(dff, x_var, y_var):
    x_arr = dff[x_var].values.astype(float)
    y_arr = dff[y_var].values.astype(float)
    years = dff["ds"].dt.year.values

    slope, intercept, r2 = linreg(x_arr, y_arr)
    x_line = np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), 200)
    y_line = slope * x_line + intercept if slope is not None else None

    x_label = ALL_VARS.get(x_var, x_var)
    y_label = ALL_VARS.get(y_var, y_var)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_arr, y=y_arr,
        mode="markers",
        marker=dict(
            color=years,
            colorscale="Viridis",
            size=6,
            opacity=0.85,
            colorbar=dict(title="Year", thickness=12,
                          tickfont=dict(size=9), outlinecolor=C["border"]),
            line=dict(width=0),
        ),
        text=[str(d)[:7] for d in dff["ds"]],
        hovertemplate=(
            f"{x_label}: %{{x:.2f}}<br>"
            f"{y_label}: %{{y:.2f}}<br>"
            "%{text}<extra></extra>"
        ),
        name="Monthly observation",
    ))

    if y_line is not None:
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color=C["red"], width=1.8, dash="dash"),
            name=f"OLS trend  (R² = {r2:.3f})",
            hoverinfo="skip",
        ))

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(
            text=f"<b>{y_label}</b>  vs  <b>{x_label}</b>"
                 + (f"   <span style='color:{C['muted']}; font-size:12px'>R² = {r2:.3f}</span>"
                    if r2 is not None else ""),
            font_size=13, x=0.01,
        ),
        xaxis=dict(**BASE_XAXIS, title=x_label),
        yaxis=dict(**BASE_YAXIS, title=y_label),
        height=430,
    )
    return fig


def build_heatmap_fig(dff):
    cols = [c for c in dff.columns if c not in ("ds", "population_total")]
    labels = [ALL_VARS.get(c, c) for c in cols]
    corr = dff[cols].corr().values

    fig = go.Figure(go.Heatmap(
        z=corr,
        x=labels, y=labels,
        colorscale=[
            [0.0,  "#1a3a5c"],
            [0.25, "#2980b9"],
            [0.5,  C["card"]],
            [0.75, "#c0392b"],
            [1.0,  "#7b241c"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}",
        textfont=dict(size=8, color=C["text"]),
        colorbar=dict(title="ρ", thickness=12,
                      tickfont=dict(size=9), outlinecolor=C["border"]),
        hoverongaps=False,
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=C["card"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], size=10),
        title=dict(text="<b>Pearson Correlation Matrix</b>  (all variables)",
                   font_size=13, x=0.01, font_color=C["text"]),
        margin=dict(l=160, r=20, t=50, b=160),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8.5)),
        yaxis=dict(tickfont=dict(size=8.5), autorange="reversed"),
        height=560,
    )
    return fig


def build_district_comparison_fig(dff, show_events=True):
    """Overlay all five district AHPI lines on a single chart."""
    fig = go.Figure()
    for district in DISTRICTS:
        d = dff[dff["district"] == district]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["ds"], y=d["y"],
            line=dict(color=DISTRICT_COLORS[district], width=2),
            name=district,
            hovertemplate=f"{district}: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.9, opacity=0.5,
                  annotation_text="2015 baseline",
                  annotation_font_color=C["muted"],
                  annotation_font_size=9)
    fig.update_layout(
        BASE_LAYOUT,
        xaxis=BASE_XAXIS,
        yaxis=dict(**BASE_YAXIS, title="Index (2015 = 100)"),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        title=dict(text="<b>AHPI by District</b>  — all five mid-market areas (2015 = 100 per district)",
                   font_size=14, x=0.01),
        height=460,
    )
    if show_events:
        add_event_lines(fig, dff)
    return fig


def build_district_single_fig(dff, district):
    """Area chart for one district's AHPI + GHS/sqm and USD/sqm overlays."""
    color = DISTRICT_COLORS.get(district, C["gold"])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["y"],
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
        line=dict(color=color, width=2.5),
        name=f"AHPI – {district}",
        hovertemplate="%{y:.1f}<extra>AHPI</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["price_ghs_per_sqm"],
        line=dict(color=C["green"], width=1.6, dash="dash"),
        name="GHS / sqm",
        yaxis="y2",
        hovertemplate="GHS %{y:,.0f}<extra>GHS/sqm</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["price_usd_per_sqm"],
        line=dict(color=C["blue"], width=1.6, dash="dot"),
        name="USD / sqm",
        yaxis="y3",
        hovertemplate="$%{y:,.0f}<extra>USD/sqm</extra>",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.9, opacity=0.5,
                  annotation_text="2015 baseline",
                  annotation_font_color=C["muted"],
                  annotation_font_size=9)
    fig.update_layout(
        BASE_LAYOUT,
        xaxis=BASE_XAXIS,
        yaxis=dict(**BASE_YAXIS, title="Index (2015 = 100)"),
        yaxis2=dict(title=dict(text="GHS / sqm", font=dict(color=C["green"])),
                    overlaying="y", side="right",
                    showgrid=False, tickformat=",",
                    tickfont=dict(color=C["green"], size=10)),
        yaxis3=dict(title=dict(text="USD / sqm", font=dict(color=C["blue"])),
                    overlaying="y", side="right",
                    anchor="free", position=0.97, showgrid=False,
                    tickfont=dict(color=C["blue"], size=10)),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        title=dict(text=f"<b>AHPI — {district}</b>  (GHS-denominated, base 2015 = 100)",
                   font_size=14, x=0.01),
        height=460,
    )
    add_event_lines(fig, dff)
    return fig


def build_district_price_table(yr_range):
    """Summary table: latest GHS/sqm and USD/sqm per district."""
    dff = filter_df_district(*yr_range)
    latest_date = dff["ds"].max()
    rows = []
    for district in DISTRICTS:
        row = dff[(dff["district"] == district) & (dff["ds"] == latest_date)]
        if row.empty:
            continue
        r = row.iloc[0]
        dot = html.Span("●", style={"color": DISTRICT_COLORS[district],
                                    "marginRight": "6px", "fontSize": "1.1rem"})
        rows.append(html.Tr([
            html.Td([dot, district],
                    style={"color": C["text"], "fontWeight": "600", "padding": "6px 12px"}),
            html.Td(f"{r['y']:.1f}",
                    style={"color": C["gold"], "textAlign": "right", "padding": "6px 12px"}),
            html.Td(f"GHS {r['price_ghs_per_sqm']:,.0f}",
                    style={"color": C["green"], "textAlign": "right", "padding": "6px 12px"}),
            html.Td(f"${r['price_usd_per_sqm']:,.0f}",
                    style={"color": C["blue"], "textAlign": "right", "padding": "6px 12px"}),
        ]))
    header = html.Tr([
        html.Th("District",   style={"color": C["muted"], "padding": "6px 12px",
                                     "borderBottom": f"1px solid {C['border']}"}),
        html.Th("AHPI",       style={"color": C["muted"], "textAlign": "right",
                                     "padding": "6px 12px",
                                     "borderBottom": f"1px solid {C['border']}"}),
        html.Th("GHS / sqm",  style={"color": C["muted"], "textAlign": "right",
                                     "padding": "6px 12px",
                                     "borderBottom": f"1px solid {C['border']}"}),
        html.Th("USD / sqm",  style={"color": C["muted"], "textAlign": "right",
                                     "padding": "6px 12px",
                                     "borderBottom": f"1px solid {C['border']}"}),
    ])
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "fontSize": "0.85rem", "borderCollapse": "collapse"},
    )


def build_prime_comparison_fig(dff, show_events=True):
    """Overlay all six prime-area AHPI lines on a single chart."""
    fig = go.Figure()
    for area in PRIME_AREAS:
        d = dff[dff["district"] == area]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["ds"], y=d["y"],
            line=dict(color=PRIME_COLORS[area], width=2),
            name=area,
            hovertemplate=f"{area}: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.9, opacity=0.5,
                  annotation_text="2015 baseline",
                  annotation_font_color=C["muted"],
                  annotation_font_size=9)
    fig.update_layout(
        BASE_LAYOUT,
        xaxis=BASE_XAXIS,
        yaxis=dict(**BASE_YAXIS, title="Index (2015 = 100)"),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        title=dict(text="<b>AHPI — Prime Areas</b>  (USD-indexed markets, base 2015 = 100 per area)",
                   font_size=14, x=0.01),
        height=460,
    )
    if show_events:
        add_event_lines(fig, dff)
    return fig


def build_prime_single_fig(dff, area):
    """Area chart for one prime location's AHPI + GHS/sqm and USD/sqm."""
    color = PRIME_COLORS.get(area, C["red"])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["y"],
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
        line=dict(color=color, width=2.5),
        name=f"AHPI – {area}",
        hovertemplate="%{y:.1f}<extra>AHPI</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["price_ghs_per_sqm"],
        line=dict(color=C["green"], width=1.6, dash="dash"),
        name="GHS / sqm",
        yaxis="y2",
        hovertemplate="GHS %{y:,.0f}<extra>GHS/sqm</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["price_usd_per_sqm"],
        line=dict(color=C["blue"], width=1.6, dash="dot"),
        name="USD / sqm",
        yaxis="y3",
        hovertemplate="$%{y:,.0f}<extra>USD/sqm</extra>",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.9, opacity=0.5,
                  annotation_text="2015 baseline",
                  annotation_font_color=C["muted"],
                  annotation_font_size=9)
    fig.update_layout(
        BASE_LAYOUT,
        xaxis=BASE_XAXIS,
        yaxis=dict(**BASE_YAXIS, title="Index (2015 = 100)"),
        yaxis2=dict(title=dict(text="GHS / sqm", font=dict(color=C["green"])),
                    overlaying="y", side="right",
                    showgrid=False, tickformat=",",
                    tickfont=dict(color=C["green"], size=10)),
        yaxis3=dict(title=dict(text="USD / sqm", font=dict(color=C["blue"])),
                    overlaying="y", side="right",
                    anchor="free", position=0.97, showgrid=False,
                    tickfont=dict(color=C["blue"], size=10)),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        title=dict(text=f"<b>AHPI — {area}</b>  (USD-indexed prime market, base 2015 = 100)",
                   font_size=14, x=0.01),
        height=460,
    )
    add_event_lines(fig, dff)
    return fig


def build_prime_price_table(yr_range):
    """Summary table showing latest prices across all prime areas."""
    dff = filter_df_prime(*yr_range)
    latest_date = dff["ds"].max()
    rows = []
    for area in PRIME_AREAS:
        row = dff[(dff["district"] == area) & (dff["ds"] == latest_date)]
        if row.empty:
            continue
        r = row.iloc[0]
        dot = html.Span("●", style={"color": PRIME_COLORS[area],
                                    "marginRight": "6px", "fontSize": "1.1rem"})
        rows.append(html.Tr([
            html.Td([dot, area],
                    style={"color": C["text"], "fontWeight": "600", "padding": "6px 12px"}),
            html.Td(f"{r['y']:.1f}",
                    style={"color": C["gold"], "textAlign": "right", "padding": "6px 12px"}),
            html.Td(f"GHS {r['price_ghs_per_sqm']:,.0f}",
                    style={"color": C["green"], "textAlign": "right", "padding": "6px 12px"}),
            html.Td(f"${r['price_usd_per_sqm']:,.0f}",
                    style={"color": C["blue"], "textAlign": "right", "padding": "6px 12px"}),
        ]))
    header = html.Tr([
        html.Th("Prime Area",  style={"color": C["muted"], "padding": "6px 12px",
                                      "borderBottom": f"1px solid {C['border']}"}),
        html.Th("AHPI",        style={"color": C["muted"], "textAlign": "right",
                                      "padding": "6px 12px",
                                      "borderBottom": f"1px solid {C['border']}"}),
        html.Th("GHS / sqm",   style={"color": C["muted"], "textAlign": "right",
                                      "padding": "6px 12px",
                                      "borderBottom": f"1px solid {C['border']}"}),
        html.Th("USD / sqm",   style={"color": C["muted"], "textAlign": "right",
                                      "padding": "6px 12px",
                                      "borderBottom": f"1px solid {C['border']}"}),
    ])
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "fontSize": "0.85rem", "borderCollapse": "collapse"},
    )


# ── map figure ────────────────────────────────────────────────────────────────
def build_map_fig(segment: str = "both") -> go.Figure:
    """Interactive open-street-map showing district / prime area locations."""
    fig = go.Figure()

    def _traces(coords: dict, snap: dict, colors: dict,
                group: str, gtitle: str) -> None:
        first = True
        for name, (lat, lon) in coords.items():
            info = snap[name]
            # Scale marker size by AHPI (range ~390–850 across both segments)
            size = max(14, min(44, 10 + info["ahpi"] / 900 * 36))
            fig.add_trace(go.Scattermap(
                lat=[lat], lon=[lon],
                mode="markers",
                marker=dict(size=size, color=colors[name], opacity=0.88),
                name=name,
                customdata=[[
                    info["ahpi"], info["ghs_sqm"],
                    info["usd_sqm"], info["usd_pct"],
                ]],
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "AHPI Dec 2024: <b>%{customdata[0]:.1f}</b><br>"
                    "GHS / sqm: <b>%{customdata[1]:,.0f}</b><br>"
                    "USD / sqm: <b>%{customdata[2]:,.0f}</b><br>"
                    "USD gain 2010–24: <b>+%{customdata[3]:.0f}%%</b>"
                    "<extra></extra>"
                ),
                legendgroup=group,
                legendgrouptitle_text=gtitle if first else "",
            ))
            first = False

    if segment in ("mid", "both"):
        _traces(DISTRICT_COORDS, DISTRICT_SNAP, DISTRICT_COLORS,
                group="mid", gtitle="Mid-Market Districts")
    if segment in ("prime", "both"):
        _traces(PRIME_COORDS, PRIME_SNAP, PRIME_COLORS,
                group="prime", gtitle="Prime Areas")

    fig.update_layout(
        paper_bgcolor=C["card"],
        font=dict(family="'Inter', 'Segoe UI', Arial, sans-serif",
                  color=C["text"], size=11),
        hovermode="closest",
        hoverlabel=dict(bgcolor=C["hover"], font_color=C["text"],
                        bordercolor=C["border"], namelength=-1),
        map=dict(
            style="open-street-map",
            center=dict(lat=5.610, lon=-0.195),
            zoom=9.5,
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.88)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(size=11, color=C["text"]),
            orientation="v",
            x=0.01, y=0.99,
            tracegroupgap=8,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
    )
    return fig


# ── forecast figure ────────────────────────────────────────────────────────────
SCENARIO_STYLES = {
    "bear": (C["red"],    "dot",   "Bear  (GHS/USD → 20)"),
    "base": (C["blue"],   "solid", "Base  (GHS/USD → 15)"),
    "bull": (C["green"],  "dot",   "Bull  (GHS/USD → 12)"),
}

# Scenario FX endpoint (used by calculators)
SCENARIO_FX   = {"bear": 20.0, "base": 15.0, "bull": 12.0}
SCENARIO_TBILL = {"bear": 28.0, "base": 20.0, "bull": 15.0}   # annual GHS risk-free rate (%)
USD_DEPOSIT_RATE = 5.0   # annual USD fixed-deposit benchmark (%)

# Forecast years available (CSVs now cover Jan 2025 – Dec 2029)
FC_YEARS     = list(range(2025, 2030))
FC_YEAR_OPTS = [{"label": str(y), "value": y} for y in FC_YEARS]


def _fc_dec(df_fc: pd.DataFrame, year: int) -> pd.Series:
    """Return the December row for a given forecast year."""
    mask = (df_fc["ds"].dt.year == year) & (df_fc["ds"].dt.month == 12)
    rows = df_fc[mask]
    return rows.iloc[0] if len(rows) else df_fc.iloc[-1]


def build_forecast_fig(show_ci: bool = True) -> go.Figure:
    """
    Two-panel figure:
      row 1 — historical actuals (2010-2024) + test-period predicted vs actual
               + three scenario forecasts (2025-2026) with 90% CI bands
      row 2 — test-period residuals (bar chart, 2023-2024)
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "AHPI  ·  Historical Actuals  ·  Test-Period Evaluation  ·  2025–2026 Scenario Forecasts",
            "Residuals  (Actual − Predicted)  ·  Test Period 2023–2024",
        ),
    )

    # ── 1. Historical actuals 2010-2024 (area fill) ───────────────────────────
    fig.add_trace(go.Scatter(
        x=DF["ds"], y=DF["y"],
        fill="tozeroy", fillcolor="rgba(212,160,23,0.10)",
        line=dict(color=C["gold"], width=2.5),
        name="Actual AHPI",
        hovertemplate="%{y:.1f}<extra>Actual</extra>",
    ), row=1, col=1)

    # ── 2. Test-period prediction + 90% CI ───────────────────────────────────
    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(DF_TEST_EVAL["ds"]) + list(DF_TEST_EVAL["ds"])[::-1],
            y=list(DF_TEST_EVAL["yhat_upper"]) + list(DF_TEST_EVAL["yhat_lower"])[::-1],
            fill="toself",
            fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Test 90% CI",
            hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=DF_TEST_EVAL["ds"], y=DF_TEST_EVAL["yhat"],
        line=dict(color=C["blue"], width=2, dash="dash"),
        name="Predicted (test 2023–24)",
        hovertemplate="%{y:.1f}<extra>Predicted</extra>",
    ), row=1, col=1)

    # ── 3. Scenario forecasts 2025-2026 ──────────────────────────────────────
    for fc_name, fc_df in [("bear", DF_FC_BEAR), ("base", DF_FC_BASE), ("bull", DF_FC_BULL)]:
        color, dash, label = SCENARIO_STYLES[fc_name]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(fc_df["ds"]) + list(fc_df["ds"])[::-1],
                y=list(fc_df["yhat_upper"]) + list(fc_df["yhat_lower"])[::-1],
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=fc_df["ds"], y=fc_df["yhat"],
            line=dict(color=color, width=2, dash=dash),
            name=label,
            customdata=fc_df[["yhat_lower", "yhat_upper"]].values,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "yhat: %{y:.1f}<br>"
                "90% CI: [%{customdata[0]:.1f} – %{customdata[1]:.1f}]"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # ── 4. Separator vertical lines ───────────────────────────────────────────
    for vdate, vlabel in [
        ("2023-01-01", "Test start"),
        ("2025-01-01", "Forecast start"),
    ]:
        fig.add_vline(
            x=pd.Timestamp(vdate).timestamp() * 1000,
            line_dash="dot", line_color=C["muted"], line_width=1.2, opacity=0.55,
            row=1, col=1,
        )
        fig.add_annotation(
            x=pd.Timestamp(vdate), y=1.0,
            xref="x", yref="paper",
            text=vlabel, showarrow=False,
            font=dict(size=8, color=C["muted"]),
            bgcolor=C["hover"], bordercolor=C["border"],
            borderwidth=1, borderpad=3, opacity=0.9, xanchor="left",
        )

    # ── 5. 2015 baseline ─────────────────────────────────────────────────────
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.8, opacity=0.4, row=1, col=1)

    # ── 6. Residual bars ──────────────────────────────────────────────────────
    res_colors = [C["green"] if r >= 0 else C["red"]
                  for r in DF_TEST_EVAL["residual"]]
    fig.add_trace(go.Bar(
        x=DF_TEST_EVAL["ds"],
        y=DF_TEST_EVAL["residual"],
        marker_color=res_colors,
        name="Residual",
        showlegend=False,
        hovertemplate="%{y:+.2f}<extra>Residual</extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"],
                  line_width=0.8, opacity=0.5, row=2, col=1)

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=C["card"],
        plot_bgcolor=C["bg"],
        font=dict(family="'Inter', 'Segoe UI', Arial, sans-serif",
                  color=C["text"], size=11),
        margin=dict(l=55, r=20, t=50, b=44),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["hover"], font_color=C["text"],
                        bordercolor=C["border"], namelength=-1),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        height=640,
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"],
                     tickcolor=C["muted"], tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=C["border"],
                     linecolor=C["border"], tickfont=dict(size=10))
    fig.update_yaxes(title_text="Index (2015 = 100)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (pts)", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font.size  = 10
        ann.font.color = C["muted"]
    return fig


def build_prime_forecast_fig(area: str = "all", show_ci: bool = True) -> go.Figure:
    """
    Two-panel forecast figure for prime areas.
    area='all'  — aggregate (average) across all 6 prime areas
    area=<name> — single-area model output
    """
    if area == "all":
        hist_df    = _PRIME_HIST_AGG
        test_eval  = _PRIME_TEST_EVAL_AGG
        fc_data    = {sc: _PRIME_FC_AGG[sc] for sc in ("bear", "base", "bull")}
        main_color = C["red"]
        fill_color = "rgba(248,81,73,0.10)"
        hist_name  = "Actual AHPI  (6-area avg)"
        area_label = "All Prime Areas (average)"
    else:
        hist_df    = DF_PRIME[DF_PRIME["district"] == area][["ds", "y"]].reset_index(drop=True)
        test_eval  = _PRIME_TEST_EVALS[area]
        fc_data    = {sc: _PRIME_FC[(sc, area)] for sc in ("bear", "base", "bull")}
        main_color = PRIME_COLORS.get(area, C["red"])
        r, g, b    = int(main_color[1:3], 16), int(main_color[3:5], 16), int(main_color[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.10)"
        hist_name  = f"Actual AHPI  ({area})"
        area_label = area

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"Prime AHPI  ·  {area_label}  ·  Historical  ·  Test Eval  ·  2025–2026 Scenarios",
            "Residuals  (Actual − Predicted)  ·  Test Period 2023–2024",
        ),
    )

    # 1. Historical actuals 2010-2024
    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["y"],
        fill="tozeroy", fillcolor=fill_color,
        line=dict(color=main_color, width=2.5),
        name=hist_name,
        hovertemplate="%{y:.1f}<extra>Actual</extra>",
    ), row=1, col=1)

    # 2. Test-period prediction + 90% CI
    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(test_eval["ds"]) + list(test_eval["ds"])[::-1],
            y=list(test_eval["yhat_upper"]) + list(test_eval["yhat_lower"])[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Test 90% CI", hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=test_eval["ds"], y=test_eval["yhat"],
        line=dict(color=C["blue"], width=2, dash="dash"),
        name="Predicted (test 2023–24)",
        hovertemplate="%{y:.1f}<extra>Predicted</extra>",
    ), row=1, col=1)

    # 3. Scenario forecasts 2025-2026
    for fc_name, fc_df in [(sc, fc_data[sc]) for sc in ("bear", "base", "bull")]:
        color, dash, label = SCENARIO_STYLES[fc_name]
        r2, g2, b2 = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(fc_df["ds"]) + list(fc_df["ds"])[::-1],
                y=list(fc_df["yhat_upper"]) + list(fc_df["yhat_lower"])[::-1],
                fill="toself", fillcolor=f"rgba({r2},{g2},{b2},0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=fc_df["ds"], y=fc_df["yhat"],
            line=dict(color=color, width=2, dash=dash),
            name=label,
            customdata=fc_df[["yhat_lower", "yhat_upper"]].values,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "yhat: %{y:.1f}<br>"
                "90% CI: [%{customdata[0]:.1f} – %{customdata[1]:.1f}]"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # 4. Separator vertical lines
    for vdate, vlabel in [("2023-01-01", "Test start"), ("2025-01-01", "Forecast start")]:
        fig.add_vline(
            x=pd.Timestamp(vdate).timestamp() * 1000,
            line_dash="dot", line_color=C["muted"], line_width=1.2, opacity=0.55,
            row=1, col=1,
        )
        fig.add_annotation(
            x=pd.Timestamp(vdate), y=1.0,
            xref="x", yref="paper",
            text=vlabel, showarrow=False,
            font=dict(size=8, color=C["muted"]),
            bgcolor=C["hover"], bordercolor=C["border"],
            borderwidth=1, borderpad=3, opacity=0.9, xanchor="left",
        )

    # 5. 2015 baseline
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.8, opacity=0.4, row=1, col=1)

    # 6. Residual bars
    res_colors = [C["green"] if r >= 0 else C["red"] for r in test_eval["residual"]]
    fig.add_trace(go.Bar(
        x=test_eval["ds"], y=test_eval["residual"],
        marker_color=res_colors, name="Residual",
        showlegend=False, hovertemplate="%{y:+.2f}<extra>Residual</extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"],
                  line_width=0.8, opacity=0.5, row=2, col=1)

    fig.update_layout(
        paper_bgcolor=C["card"], plot_bgcolor=C["bg"],
        font=dict(family="'Inter', 'Segoe UI', Arial, sans-serif",
                  color=C["text"], size=11),
        margin=dict(l=55, r=20, t=50, b=44),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["hover"], font_color=C["text"],
                        bordercolor=C["border"], namelength=-1),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        height=640,
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"],
                     tickcolor=C["muted"], tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=C["border"],
                     linecolor=C["border"], tickfont=dict(size=10))
    fig.update_yaxes(title_text="Index (2015 = 100)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (pts)", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font.size  = 10
        ann.font.color = C["muted"]
    return fig


def _build_prime_metrics_div(area: str) -> html.Div:
    if area == "all":
        mae  = _PRIME_TEST_SUMMARY["mae"].mean()
        rmse = _PRIME_TEST_SUMMARY["rmse"].mean()
        mape = _PRIME_TEST_SUMMARY["mape_pct"].mean()
        note = "Average across all 6 prime areas (n = 24 per area)."
    else:
        row  = _PRIME_TEST_SUMMARY[_PRIME_TEST_SUMMARY["area"] == area].iloc[0]
        mae, rmse, mape = row["mae"], row["rmse"], row["mape_pct"]
        note = f"Evaluation model for {area}; trained 2010–2022 (n = 24 test months)."
    return html.Div([
        _stat_row("MAE",  f"{mae:.2f} index pts",  C["gold"]),
        _stat_row("RMSE", f"{rmse:.2f} index pts", C["orange"]),
        _stat_row("MAPE", f"{mape:.1f}%",          C["red"]),
        html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
        html.Div(note, style={"fontSize": "0.75rem", "color": C["muted"]}),
        _METRIC_GLOSSARY,
    ])


def _build_prime_targets_div(area: str, year: int = 2026) -> html.Div:
    rows = []
    for sc in ("bear", "base", "bull"):
        color, _, label = SCENARIO_STYLES[sc]
        fc_df = _PRIME_FC_AGG[sc] if area == "all" else _PRIME_FC[(sc, area)]
        dec   = _fc_dec(fc_df, year)
        rows.append(html.Div([
            html.Span("● ", style={"color": color, "fontSize": "1rem"}),
            html.Span(f"{label}: ", style={"color": C["muted"], "fontSize": "0.8rem"}),
            html.Span(f"{dec['yhat']:.1f}",
                      style={"color": color, "fontWeight": "700", "fontSize": "0.9rem"}),
            html.Span(f"  [{dec['yhat_lower']:.1f} – {dec['yhat_upper']:.1f}]",
                      style={"color": C["muted"], "fontSize": "0.78rem"}),
        ], className="mb-2"))
    return html.Div([
        *rows,
        html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
        html.Div(f"Dec {year} AHPI under each scenario. 90% credible interval in brackets.",
                 style={"fontSize": "0.75rem", "color": C["muted"]}),
    ])


def build_district_forecast_fig(district: str = "all", show_ci: bool = True) -> go.Figure:
    """
    Two-panel forecast figure for mid-market districts.
    district='all'  — aggregate (average) across all 5 districts
    district=<name> — single-district model output
    """
    if district == "all":
        hist_df      = _DISTRICT_HIST_AGG
        test_eval    = _DISTRICT_TEST_EVAL_AGG
        fc_data      = {sc: _DISTRICT_FC_AGG[sc] for sc in ("bear", "base", "bull")}
        main_color   = C["blue"]
        fill_color   = "rgba(88,166,255,0.10)"
        hist_name    = "Actual AHPI  (5-district avg)"
        area_label   = "All Districts (average)"
    else:
        hist_df    = DF_DISTRICT[DF_DISTRICT["district"] == district][["ds", "y"]].reset_index(drop=True)
        test_eval  = _DISTRICT_TEST_EVALS[district]
        fc_data    = {sc: _DISTRICT_FC[(sc, district)] for sc in ("bear", "base", "bull")}
        main_color = DISTRICT_COLORS.get(district, C["blue"])
        r, g, b    = int(main_color[1:3], 16), int(main_color[3:5], 16), int(main_color[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.10)"
        hist_name  = f"Actual AHPI  ({district})"
        area_label = district

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"District AHPI  ·  {area_label}  ·  Historical  ·  Test Eval  ·  2025–2026 Scenarios",
            "Residuals  (Actual − Predicted)  ·  Test Period 2023–2024",
        ),
    )

    # 1. Historical actuals
    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["y"],
        fill="tozeroy", fillcolor=fill_color,
        line=dict(color=main_color, width=2.5),
        name=hist_name,
        hovertemplate="%{y:.1f}<extra>Actual</extra>",
    ), row=1, col=1)

    # 2. Test-period prediction + 90% CI
    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(test_eval["ds"]) + list(test_eval["ds"])[::-1],
            y=list(test_eval["yhat_upper"]) + list(test_eval["yhat_lower"])[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Test 90% CI", hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=test_eval["ds"], y=test_eval["yhat"],
        line=dict(color=C["blue"], width=2, dash="dash"),
        name="Predicted (test 2023–24)",
        hovertemplate="%{y:.1f}<extra>Predicted</extra>",
    ), row=1, col=1)

    # 3. Scenario forecasts 2025-2026
    for fc_name, fc_df in [(sc, fc_data[sc]) for sc in ("bear", "base", "bull")]:
        color, dash, label = SCENARIO_STYLES[fc_name]
        r2, g2, b2 = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(fc_df["ds"]) + list(fc_df["ds"])[::-1],
                y=list(fc_df["yhat_upper"]) + list(fc_df["yhat_lower"])[::-1],
                fill="toself", fillcolor=f"rgba({r2},{g2},{b2},0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=fc_df["ds"], y=fc_df["yhat"],
            line=dict(color=color, width=2, dash=dash),
            name=label,
            customdata=fc_df[["yhat_lower", "yhat_upper"]].values,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "yhat: %{y:.1f}<br>"
                "90% CI: [%{customdata[0]:.1f} – %{customdata[1]:.1f}]"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # 4. Vertical separators
    for vdate, vlabel in [("2023-01-01", "Test start"), ("2025-01-01", "Forecast start")]:
        fig.add_vline(
            x=pd.Timestamp(vdate).timestamp() * 1000,
            line_dash="dot", line_color=C["muted"], line_width=1.2, opacity=0.55,
            row=1, col=1,
        )
        fig.add_annotation(
            x=pd.Timestamp(vdate), y=1.0,
            xref="x", yref="paper",
            text=vlabel, showarrow=False,
            font=dict(size=8, color=C["muted"]),
            bgcolor=C["hover"], bordercolor=C["border"],
            borderwidth=1, borderpad=3, opacity=0.9, xanchor="left",
        )

    # 5. Baseline
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.8, opacity=0.4, row=1, col=1)

    # 6. Residual bars
    res_colors = [C["green"] if r >= 0 else C["red"] for r in test_eval["residual"]]
    fig.add_trace(go.Bar(
        x=test_eval["ds"], y=test_eval["residual"],
        marker_color=res_colors, name="Residual",
        showlegend=False, hovertemplate="%{y:+.2f}<extra>Residual</extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"],
                  line_width=0.8, opacity=0.5, row=2, col=1)

    fig.update_layout(
        paper_bgcolor=C["card"], plot_bgcolor=C["bg"],
        font=dict(family="'Inter', 'Segoe UI', Arial, sans-serif",
                  color=C["text"], size=11),
        margin=dict(l=55, r=20, t=50, b=44),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["hover"], font_color=C["text"],
                        bordercolor=C["border"], namelength=-1),
        legend=dict(**BASE_LEGEND, orientation="h", x=0.01, y=1.06),
        height=640,
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"],
                     tickcolor=C["muted"], tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=C["border"],
                     linecolor=C["border"], tickfont=dict(size=10))
    fig.update_yaxes(title_text="Index (2015 = 100)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (pts)", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font.size  = 10
        ann.font.color = C["muted"]
    return fig


def _build_district_metrics_div(district: str) -> html.Div:
    if district == "all":
        mae  = _DISTRICT_TEST_SUMMARY["mae"].mean()
        rmse = _DISTRICT_TEST_SUMMARY["rmse"].mean()
        mape = _DISTRICT_TEST_SUMMARY["mape_pct"].mean()
        note = "Average across all 5 mid-market districts (n = 24 per district)."
    else:
        row  = _DISTRICT_TEST_SUMMARY[_DISTRICT_TEST_SUMMARY["district"] == district].iloc[0]
        mae, rmse, mape = row["mae"], row["rmse"], row["mape_pct"]
        note = f"Evaluation model for {district}; trained 2010–2022 (n = 24 test months)."
    return html.Div([
        _stat_row("MAE",  f"{mae:.2f} index pts",  C["gold"]),
        _stat_row("RMSE", f"{rmse:.2f} index pts", C["orange"]),
        _stat_row("MAPE", f"{mape:.1f}%",          C["red"]),
        html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
        html.Div(note, style={"fontSize": "0.75rem", "color": C["muted"]}),
        _METRIC_GLOSSARY,
    ])


def _build_district_targets_div(district: str, year: int = 2026) -> html.Div:
    rows = []
    for sc in ("bear", "base", "bull"):
        color, _, label = SCENARIO_STYLES[sc]
        fc_df = _DISTRICT_FC_AGG[sc] if district == "all" else _DISTRICT_FC[(sc, district)]
        dec   = _fc_dec(fc_df, year)
        rows.append(html.Div([
            html.Span("● ", style={"color": color, "fontSize": "1rem"}),
            html.Span(f"{label}: ", style={"color": C["muted"], "fontSize": "0.8rem"}),
            html.Span(f"{dec['yhat']:.1f}",
                      style={"color": color, "fontWeight": "700", "fontSize": "0.9rem"}),
            html.Span(f"  [{dec['yhat_lower']:.1f} – {dec['yhat_upper']:.1f}]",
                      style={"color": C["muted"], "fontSize": "0.78rem"}),
        ], className="mb-2"))
    return html.Div([
        *rows,
        html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
        html.Div(f"Dec {year} AHPI under each scenario. 90% credible interval in brackets.",
                 style={"fontSize": "0.75rem", "color": C["muted"]}),
    ])


# ── reusable layout pieces ────────────────────────────────────────────────────
def kpi_card(label, val_id, icon, color=C["gold"]):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Span(icon, style={"fontSize": "1.2rem", "marginRight": "6px"}),
                    html.Span(label,
                              id=f"{val_id}-label",
                              style={"fontSize": "0.68rem", "textTransform": "uppercase",
                                     "letterSpacing": "0.07em", "color": C["muted"],
                                     "cursor": "help", "borderBottom": f"1px dotted {C['border']}"}),
                ], className="d-flex align-items-center mb-1"),
                html.Div(id=val_id,
                         style={"fontSize": "1.35rem", "fontWeight": "700",
                                "color": color, "lineHeight": "1.2"}),
                html.Div(id=f"{val_id}-sub",
                         style={"fontSize": "0.72rem", "color": C["muted"],
                                "marginTop": "2px"}),
            ], style={"padding": "0.8rem 1rem"}),
        ], style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}",
                  "borderRadius": "8px", "height": "100%"}),
        xs=6, sm=4, md=2,
    )


def section_card(*children, **kwargs):
    return dbc.Card(
        dbc.CardBody(list(children)),
        style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}",
               "borderRadius": "8px", **kwargs},
        className="mb-3",
    )


def toggle_btn(btn_id, label, active=True):
    return dbc.Button(
        label, id=btn_id, size="sm", outline=not active,
        color="warning" if active else "secondary",
        className="me-1",
        style={"fontSize": "0.75rem", "padding": "2px 10px"},
    )


def _dl_btn(btn_id, label="⬇  Download CSV"):
    """Small right-aligned download trigger button."""
    return html.Div(
        dbc.Button(label, id=btn_id, size="sm", outline=True, color="secondary",
                   style={"fontSize": "0.72rem", "padding": "2px 10px"}),
        className="text-end mb-1",
    )


def _zip_dfs(files: dict) -> bytes:
    """Zip a {filename: DataFrame} dict into a bytes object."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, df in files.items():
            zf.writestr(fname, df.to_csv(index=False))
    return buf.getvalue()


# ── methodology modal ──────────────────────────────────────────────────────────
_METHODOLOGY_MODAL = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("About the Accra Home Price Index (AHPI)"),
                    close_button=True),
    dbc.ModalBody([
        html.H6("What is the AHPI?", style={"color": C["gold"], "marginBottom": "6px"}),
        html.P(
            "The Accra Home Price Index measures how residential property values in Accra, Ghana "
            "have changed over time, expressed as an index where January 2015 = 100. "
            "An AHPI of 400 means property prices are 4× their 2015 level in Ghanaian Cedis (GHS). "
            "Think of it like a stock market index — but for Accra real estate.",
            style={"fontSize": "0.85rem"},
        ),
        html.H6("Two markets tracked", style={"color": C["gold"], "marginBottom": "6px"}),
        dbc.Row([
            dbc.Col(html.Div([
                html.Strong("Mid-Market Districts (5 areas)"),
                html.Br(),
                html.Small(
                    "Spintex Road · Adenta · Tema · Dome · Kasoa. "
                    "Prices are quoted in GHS. AHPI reflects true GHS appreciation.",
                    style={"color": C["muted"]},
                ),
            ], style={"backgroundColor": C["hover"], "padding": "10px",
                      "borderRadius": "6px", "border": f"1px solid {C['border']}"}), md=6),
            dbc.Col(html.Div([
                html.Strong("Prime Areas (6 locations)"),
                html.Br(),
                html.Small(
                    "East Legon · Cantonments · Airport Residential · Labone / Roman Ridge · "
                    "Dzorwulu / Abelenkpe · Trasacco Valley. Prices are USD-anchored. "
                    "AHPI here reflects USD appreciation plus GHS depreciation — "
                    "so the prime index grows much faster in GHS terms.",
                    style={"color": C["muted"]},
                ),
            ], style={"backgroundColor": C["hover"], "padding": "10px",
                      "borderRadius": "6px", "border": f"1px solid {C['border']}"}), md=6),
        ], className="mb-3"),
        html.H6("Data construction", style={"color": C["gold"], "marginBottom": "6px"}),
        html.P(
            "Quarterly USD/sqm benchmarks from Global Property Guide, Numbeo, JLL Africa, and "
            "Knight Frank Africa are converted to GHS using Bank of Ghana exchange rates, "
            "normalised so that January 2015 = 100, then interpolated to monthly frequency. "
            "Dataset covers January 2010 – December 2024 (180 months).",
            style={"fontSize": "0.85rem"},
        ),
        html.H6("What are the 2025–2026 forecasts?",
                style={"color": C["gold"], "marginBottom": "6px"}),
        html.P(
            "Facebook Prophet — a machine-learning time-series model — was trained on 15 years of "
            "historical data using six economic drivers: GHS/USD exchange rate, inflation (CPI), "
            "urban population growth, broad money supply (M2), gold price, and cocoa price. "
            "Separate models were built for the composite mid-market index, each of the 6 prime areas, "
            "and each of the 5 mid-market districts (12 models total).",
            style={"fontSize": "0.85rem"},
        ),
        html.H6("Bear / Base / Bull scenarios",
                style={"color": C["gold"], "marginBottom": "6px"}),
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("🐻  Bear", style={"color": C["red"], "fontWeight": "700"}),
                html.Br(),
                html.Small(
                    "GHS/USD reaches 20 by end-2026. Continued cedi depreciation, "
                    "inflation stays elevated (~31%). Worst case for GHS-earning buyers; "
                    "best case for USD-denominated asset holders.",
                    style={"color": C["muted"]},
                ),
            ], style={"backgroundColor": C["hover"], "padding": "10px",
                      "borderRadius": "6px", "border": f"1px solid {C['border']}"}), md=4),
            dbc.Col(html.Div([
                html.Span("📊  Base", style={"color": C["blue"], "fontWeight": "700"}),
                html.Br(),
                html.Small(
                    "GHS/USD stabilises at 15. Gradual cedi recovery, "
                    "inflation moderating (~20%). Most likely outcome under the current "
                    "IMF Extended Credit Facility programme.",
                    style={"color": C["muted"]},
                ),
            ], style={"backgroundColor": C["hover"], "padding": "10px",
                      "borderRadius": "6px", "border": f"1px solid {C['border']}"}), md=4),
            dbc.Col(html.Div([
                html.Span("🐂  Bull", style={"color": C["green"], "fontWeight": "700"}),
                html.Br(),
                html.Small(
                    "GHS/USD recovers to 12. Strong cedi driven by high cocoa and gold export "
                    "revenues, inflation falls to ~14%. Optimistic scenario.",
                    style={"color": C["muted"]},
                ),
            ], style={"backgroundColor": C["hover"], "padding": "10px",
                      "borderRadius": "6px", "border": f"1px solid {C['border']}"}), md=4),
        ], className="mb-3"),
        html.H6("Understanding accuracy metrics",
                style={"color": C["gold"], "marginBottom": "6px"}),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Metric"), html.Th("What it measures"), html.Th("How to read it"),
            ])),
            html.Tbody([
                html.Tr([html.Td("MAE"), html.Td("Mean Absolute Error — average error in index points"),
                         html.Td("28.5 pts → predictions were off by 28.5 AHPI pts on average")]),
                html.Tr([html.Td("RMSE"), html.Td("Root Mean Squared Error — like MAE but penalises large errors more"),
                         html.Td("Useful for catching occasional big misses")]),
                html.Tr([html.Td("MAPE"), html.Td("Mean Absolute Percentage Error — error as % of actual value"),
                         html.Td("7.8% → predicted within ±7.8% of actual, on average")]),
            ]),
        ], size="sm", bordered=True, striped=True, className="mb-3",
           style={"color": C["text"], "backgroundColor": C["card"]}),
        html.Hr(style={"borderColor": C["border"]}),
        html.Small(
            "AHPI is an estimated research index. It should not be used as the sole basis for "
            "investment or lending decisions. Forecasts are illustrative scenario projections, "
            "not financial advice. Dataset version 2.1 · January 2010 – December 2024.",
            style={"color": C["muted"]},
        ),
    ]),
    dbc.ModalFooter(
        dbc.Button("Close", id="methodology-modal-close", color="secondary", size="sm"),
    ),
], id="methodology-modal", size="lg", is_open=False, scrollable=True,
   style={"fontFamily": "'Inter', 'Segoe UI', Arial, sans-serif"})


# ── range slider ──────────────────────────────────────────────────────────────
range_slider = section_card(
    html.Div([
        html.Span("Date range ", style={"color": C["muted"], "fontSize": "0.8rem",
                                        "marginRight": "10px"}),
        html.Span(id="range-label",
                  style={"color": C["gold"], "fontWeight": "600", "fontSize": "0.85rem"}),
    ], className="mb-2 d-flex align-items-center"),
    dcc.RangeSlider(
        id="year-range",
        min=YEARS[0], max=YEARS[-1],
        step=1,
        value=[YEARS[0], YEARS[-1]],
        marks={y: {"label": str(y),
                   "style": {"color": C["muted"], "fontSize": "0.72rem"}}
               for y in YEARS},
        tooltip={"always_visible": False},
        allowCross=False,
    ),
)

# ── tab: Overview ─────────────────────────────────────────────────────────────
tab_overview = html.Div([
    section_card(
        # ── market segment selector ───────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Span("Market segment:", style={"color": C["muted"],
                                                        "fontSize": "0.8rem",
                                                        "marginRight": "10px"}),
                    dbc.RadioItems(
                        id="overview-segment",
                        options=[
                            {"label": " Mid-Market",       "value": "mid"},
                            {"label": " Prime Areas (avg)", "value": "prime"},
                            {"label": " Compare Both",      "value": "both"},
                        ],
                        value="mid",
                        inline=True,
                        style={"fontSize": "0.8rem"},
                    ),
                ], className="d-flex align-items-center"),
            ),
        ], className="mb-2"),
        # ── overlays + events row ─────────────────────────────────────────────
        html.Div([
            html.Span("Overlays:", style={"color": C["muted"], "fontSize": "0.8rem",
                                          "marginRight": "8px"}),
            dbc.Checklist(
                id="ahpi-overlays",
                options=[
                    {"label": " GHS/sqm", "value": "ghs"},
                    {"label": " USD/sqm", "value": "usd"},
                ],
                value=["ghs"],
                inline=True,
                switch=True,
                style={"fontSize": "0.8rem"},
            ),
            html.Span("  |  ", style={"color": C["border"], "margin": "0 8px"}),
            dbc.Checklist(
                id="ahpi-events",
                options=[{"label": " Show key events", "value": "show"}],
                value=["show"],
                inline=True,
                switch=True,
                style={"fontSize": "0.8rem"},
            ),
            html.Span("  |  ", style={"color": C["border"], "margin": "0 8px",
                                      "id": "overlay-separator"}),
            html.Span("Overlays disabled in Compare mode",
                      id="overlay-note",
                      style={"color": C["muted"], "fontSize": "0.75rem",
                             "fontStyle": "italic", "display": "none"}),
        ], className="d-flex align-items-center flex-wrap mb-2"),
        _dl_btn("dl-overview-btn"),
        dcc.Download(id="dl-overview"),
        dcc.Graph(id="ahpi-chart", config={"displayModeBar": True,
                                            "modeBarButtonsToRemove": ["lasso2d"],
                                            "toImageButtonOptions": {"scale": 2}}),
    ),
])

# ── tab: Macro Drivers ────────────────────────────────────────────────────────
tab_macro = html.Div([
    section_card(
        dbc.Row([
            dbc.Col([
                html.Label("Select indicators", style={"fontSize": "0.78rem",
                                                        "color": C["muted"]}),
                dcc.Dropdown(
                    id="macro-vars",
                    options=[{"label": v[0], "value": k}
                             for k, v in MACRO_META.items()],
                    value=["inflation_cpi_pct", "exchange_rate_ghs_usd",
                           "gdp_growth_pct", "lending_rate_pct"],
                    multi=True,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                    className="dash-dropdown-dark",
                ),
            ], md=9),
            dbc.Col([
                html.Label("Normalise (0–100)", style={"fontSize": "0.78rem",
                                                        "color": C["muted"]}),
                dbc.Switch(id="macro-normalise", value=False, label=""),
            ], md=3, className="d-flex flex-column justify-content-start"),
        ], className="mb-2"),
        _dl_btn("dl-macro-btn"),
        dcc.Download(id="dl-macro"),
        dcc.Graph(id="macro-chart", config={"displayModeBar": True,
                                             "modeBarButtonsToRemove": ["lasso2d"],
                                             "toImageButtonOptions": {"scale": 2}}),
    ),
    section_card(
        html.P("All macro indicators – small multiples",
               style={"color": C["muted"], "fontSize": "0.78rem", "marginBottom": "4px"}),
        dcc.Graph(id="macro-grid", config={"displayModeBar": False}),
    ),
])

# ── tab: Commodities ──────────────────────────────────────────────────────────
tab_commodities = html.Div([
    section_card(
        _dl_btn("dl-commodities-btn"),
        dcc.Download(id="dl-commodities"),
        dcc.Graph(id="commodity-chart",
                  config={"displayModeBar": True,
                          "modeBarButtonsToRemove": ["lasso2d"],
                          "toImageButtonOptions": {"scale": 2}}),
    ),
    section_card(
        html.P("Cocoa and gold prices are macro anchors for Accra property demand: "
               "Ghana's export revenues, fiscal space, and diaspora remittances are "
               "tightly coupled to commodity cycles.",
               style={"color": C["muted"], "fontSize": "0.8rem", "margin": 0}),
    ),
])

# ── tab: Regressor Explorer ───────────────────────────────────────────────────
tab_explorer = html.Div([
    dbc.Row([
        dbc.Col(
            section_card(
                html.Label("X-axis variable", style={"color": C["muted"],
                                                     "fontSize": "0.78rem"}),
                dcc.Dropdown(
                    id="scatter-x",
                    options=[{"label": v, "value": k} for k, v in ALL_VARS.items()],
                    value="exchange_rate_ghs_usd",
                    clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem"},
                ),
                html.Label("Y-axis variable", style={"color": C["muted"],
                                                     "fontSize": "0.78rem",
                                                     "marginTop": "10px"}),
                dcc.Dropdown(
                    id="scatter-y",
                    options=[{"label": v, "value": k} for k, v in ALL_VARS.items()],
                    value="y",
                    clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem"},
                ),
                html.Div(id="scatter-stats",
                         style={"marginTop": "12px", "fontSize": "0.8rem",
                                "color": C["muted"]}),
            ), md=3,
        ),
        dbc.Col(
            section_card(
                dcc.Graph(id="scatter-chart",
                          config={"displayModeBar": True,
                                  "toImageButtonOptions": {"scale": 2}}),
            ), md=9,
        ),
    ]),
    section_card(
        html.P("Pearson correlation matrix – all variables",
               style={"color": C["muted"], "fontSize": "0.78rem", "marginBottom": "4px"}),
        dcc.Graph(id="heatmap-chart", config={"displayModeBar": False}),
    ),
])

# ── tab: Districts ────────────────────────────────────────────────────────────
tab_districts = html.Div([
    section_card(
        dbc.Row([
            dbc.Col([
                html.Label("Select district", style={"fontSize": "0.78rem",
                                                      "color": C["muted"]}),
                dcc.Dropdown(
                    id="district-selector",
                    options=(
                        [{"label": "Compare All Districts", "value": "all"}] +
                        [{"label": d, "value": d} for d in DISTRICTS]
                    ),
                    value="all",
                    clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                ),
            ], md=4),
            dbc.Col([
                html.Label("Key events", style={"fontSize": "0.78rem",
                                                "color": C["muted"]}),
                dbc.Switch(id="district-events", value=True, label=""),
            ], md=2, className="d-flex flex-column justify-content-start"),
        ], className="mb-2"),
        _dl_btn("dl-districts-btn"),
        dcc.Download(id="dl-districts"),
        dcc.Graph(id="district-chart",
                  config={"displayModeBar": True,
                          "modeBarButtonsToRemove": ["lasso2d"],
                          "toImageButtonOptions": {"scale": 2}}),
    ),
    section_card(
        html.P("Latest values (end of selected date range)",
               style={"color": C["muted"], "fontSize": "0.78rem", "marginBottom": "10px"}),
        html.Div(id="district-price-table"),
    ),
])

# ── tab: Prime Areas ──────────────────────────────────────────────────────────
tab_prime = html.Div([
    section_card(
        dbc.Row([
            dbc.Col([
                html.Label("Select prime area", style={"fontSize": "0.78rem",
                                                       "color": C["muted"]}),
                dcc.Dropdown(
                    id="prime-selector",
                    options=(
                        [{"label": "Compare All Prime Areas", "value": "all"}] +
                        [{"label": a, "value": a} for a in PRIME_AREAS]
                    ),
                    value="all",
                    clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                ),
            ], md=5),
            dbc.Col([
                html.Label("Key events", style={"fontSize": "0.78rem",
                                                "color": C["muted"]}),
                dbc.Switch(id="prime-events", value=True, label=""),
            ], md=2, className="d-flex flex-column justify-content-start"),
        ], className="mb-2"),
        html.Div(
            "Prime areas are USD-indexed: sale prices and rents are routinely quoted in dollars. "
            "Their AHPI therefore reflects both USD price appreciation and GHS depreciation — "
            "producing significantly higher nominal GHS growth than mid-market districts.",
            style={"fontSize": "0.78rem", "color": C["muted"], "marginBottom": "10px"},
        ),
        _dl_btn("dl-prime-btn"),
        dcc.Download(id="dl-prime"),
        dcc.Graph(id="prime-chart",
                  config={"displayModeBar": True,
                          "modeBarButtonsToRemove": ["lasso2d"],
                          "toImageButtonOptions": {"scale": 2}}),
    ),
    section_card(
        html.P("Latest values (end of selected date range)",
               style={"color": C["muted"], "fontSize": "0.78rem", "marginBottom": "10px"}),
        html.Div(id="prime-price-table"),
    ),
])

# ── tab: Map ──────────────────────────────────────────────────────────────────
# ── GIS tile presets ──────────────────────────────────────────────────────────
_TILE_OPTS = [
    {"label": "Dark Matter",      "value": "dark"},
    {"label": "Satellite",        "value": "satellite"},
    {"label": "Positron",         "value": "positron"},
    {"label": "OSM (light)",      "value": "osm"},
]
_TILE_URLS = {
    "dark":      ("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                  "© OpenStreetMap contributors © CARTO"),
    "satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  "© Esri, Maxar, GeoEye, Earthstar Geographics, CNES/Airbus DS"),
    "positron":  ("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                  "© OpenStreetMap contributors © CARTO"),
    "osm":       ("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                  "© OpenStreetMap contributors"),
}
_MAP_CENTER = [5.610, -0.195]
_MAP_ZOOM   = 11

def _tile_layer(tile_key: str) -> dl.TileLayer:
    url, attr = _TILE_URLS.get(tile_key, _TILE_URLS["dark"])
    return dl.TileLayer(url=url, attribution=attr, maxZoom=19)


def _legend_strip(stops: list, lo: float, hi: float, label: str,
                  unit: str = "") -> html.Div:
    """Inline colour-bar legend as a horizontal gradient strip."""
    gradient = ", ".join(c for _, c in stops)
    ticks = [lo, (lo + hi) / 2, hi]
    tick_divs = html.Div([
        html.Span(f"{v:,.0f}{unit}", style={"position": "absolute",
                                             "left": f"{(v - lo) / (hi - lo + 1e-9) * 100:.0f}%",
                                             "transform": "translateX(-50%)",
                                             "fontSize": "0.65rem", "color": C["muted"],
                                             "whiteSpace": "nowrap"})
        for v in ticks
    ], style={"position": "relative", "height": "16px", "marginTop": "2px"})
    return html.Div([
        html.Div(label, style={"fontSize": "0.7rem", "color": C["muted"],
                               "marginBottom": "3px",
                               "textTransform": "uppercase",
                               "letterSpacing": "0.06em"}),
        html.Div(style={
            "background": f"linear-gradient(to right, {gradient})",
            "height": "10px", "borderRadius": "4px",
            "border": f"1px solid {C['border']}",
        }),
        tick_divs,
    ], style={"width": "260px"})


tab_map = html.Div([
    # ── Location dot map (existing Plotly / Scattermap) ─────────────────────
    html.Div(id="tab-map-location-panel", children=[
        section_card(
            dbc.Row([
                dbc.Col([
                    html.Label("Show locations", style={"fontSize": "0.78rem",
                                                        "color": C["muted"]}),
                    dbc.RadioItems(
                        id="map-segment",
                        options=[
                            {"label": "Mid-Market Districts", "value": "mid"},
                            {"label": "Prime Areas",          "value": "prime"},
                            {"label": "Both",                 "value": "both"},
                        ],
                        value="both",
                        inline=True,
                        className="mt-1",
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px", "fontSize": "0.82rem",
                                    "color": C["text"]},
                    ),
                ], md=7),
                dbc.Col([
                    html.Div(
                        "Marker size reflects Dec 2024 AHPI. Scroll or pinch to zoom.",
                        style={"fontSize": "0.72rem", "color": C["muted"],
                               "paddingTop": "10px"},
                    ),
                ], md=5),
            ], className="mb-2"),
            dcc.Graph(
                id="map-chart",
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {"scale": 2},
                },
                style={"height": "520px"},
            ),
        ),
    ]),

    # ── GIS Choropleth section ───────────────────────────────────────────────
    html.Hr(style={"borderColor": C["border"], "margin": "0 0 12px 0"}),

    section_card(
        # Controls row
        dbc.Row([
            dbc.Col([
                html.Div("Map layer", style={"fontSize": "0.75rem", "color": C["muted"],
                                              "marginBottom": "3px"}),
                dbc.RadioItems(
                    id="gis-layer",
                    options=[
                        {"label": "Price Heatmap",    "value": "price"},
                        {"label": "Forecast Growth",  "value": "forecast"},
                    ],
                    value="price",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "16px", "fontSize": "0.82rem",
                                "color": C["text"]},
                ),
            ], md=3),
            dbc.Col([
                html.Div("Price metric", style={"fontSize": "0.75rem", "color": C["muted"],
                                                 "marginBottom": "3px"}),
                dcc.Dropdown(
                    id="gis-price-metric",
                    options=[
                        {"label": "USD / sqm",    "value": "usd_sqm"},
                        {"label": "GHS / sqm",    "value": "ghs_sqm"},
                        {"label": "AHPI (index)", "value": "ahpi"},
                    ],
                    value="usd_sqm", clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Div("Forecast scenario", style={"fontSize": "0.75rem", "color": C["muted"],
                                                      "marginBottom": "3px"}),
                dcc.Dropdown(
                    id="gis-scenario",
                    options=[
                        {"label": "Bear  (GHS/USD → 20)", "value": "bear"},
                        {"label": "Base  (GHS/USD → 15)", "value": "base"},
                        {"label": "Bull  (GHS/USD → 12)", "value": "bull"},
                    ],
                    value="base", clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Div("Forecast year", style={"fontSize": "0.75rem", "color": C["muted"],
                                                  "marginBottom": "3px"}),
                dcc.Dropdown(
                    id="gis-fc-year",
                    options=FC_YEAR_OPTS, value=2027, clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Div("Base tiles", style={"fontSize": "0.75rem", "color": C["muted"],
                                               "marginBottom": "3px"}),
                dcc.Dropdown(
                    id="gis-tiles",
                    options=_TILE_OPTS, value="dark", clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Div("Export", style={"fontSize": "0.75rem", "color": C["muted"],
                                           "marginBottom": "3px"}),
                dbc.Button("⬇ GeoJSON", id="gis-dl-btn", size="sm", outline=True,
                           color="secondary",
                           style={"fontSize": "0.72rem", "padding": "2px 10px",
                                  "width": "100%"}),
                dcc.Download(id="gis-dl"),
            ], md=1),
        ], className="mb-2 g-2"),

        # Legend
        html.Div(id="gis-legend", style={"marginBottom": "8px"}),

        # Leaflet map
        dl.Map(
            id="gis-map",
            center=_MAP_CENTER,
            zoom=_MAP_ZOOM,
            style={"height": "520px", "width": "100%",
                   "borderRadius": "6px", "border": f"1px solid {C['border']}"},
            children=[
                dl.TileLayer(
                    url=_TILE_URLS["dark"][0],
                    attribution=_TILE_URLS["dark"][1],
                    maxZoom=19,
                    id="gis-tile-layer",
                ),
                dl.GeoJSON(
                    id="gis-geojson",
                    data=None,
                    style=_style_price,
                    onEachFeature=_on_each_feature_price,
                    zoomToBounds=False,
                    zoomToBoundsOnClick=True,
                    options=dict(preferCanvas=False),
                ),
                dl.ScaleControl(position="bottomleft"),
            ],
        ),
        html.Div(
            "Polygons are approximate representations of neighbourhood boundaries "
            "for visualisation purposes. Hover to see metrics. Click to zoom in.",
            style={"fontSize": "0.68rem", "color": C["muted"],
                   "fontStyle": "italic", "marginTop": "6px"},
        ),

        # ── Time-slider animation controls ───────────────────────────────────
        dcc.Interval(id="gis-anim-interval", interval=900,
                     n_intervals=0, disabled=True),
        dbc.Row([
            dbc.Col(
                dbc.Button("▶", id="gis-anim-btn", size="sm", color="secondary",
                           outline=True,
                           style={"width": "36px", "height": "34px",
                                  "padding": "0", "fontSize": "13px",
                                  "lineHeight": "1"}),
                width="auto", className="d-flex align-items-center pe-0",
            ),
            dbc.Col(
                dcc.Slider(
                    id="gis-anim-year",
                    min=2010, max=2029, step=1, value=2024,
                    marks={
                        **{y: {"label": str(y),
                               "style": {"color": C["muted"], "fontSize": "0.63rem"}}
                           for y in range(2010, 2025, 2)},
                        2024: {"label": "2024",
                               "style": {"color": C["gold"], "fontSize": "0.63rem",
                                         "fontWeight": "700"}},
                        2025: {"label": "2025 →",
                               "style": {"color": "#58a6ff", "fontSize": "0.63rem",
                                         "fontWeight": "700"}},
                        **{y: {"label": str(y),
                               "style": {"color": "#58a6ff", "fontSize": "0.63rem"}}
                           for y in range(2026, 2030, 2)},
                    },
                    tooltip={"placement": "top", "always_visible": False},
                    updatemode="drag",
                ),
            ),
            dbc.Col(
                html.Div(id="gis-anim-badge",
                         style={"fontSize": "0.7rem", "color": C["muted"],
                                "textAlign": "right", "whiteSpace": "nowrap"}),
                width="auto", className="d-flex align-items-center ps-2",
            ),
        ], className="g-1 mt-2 align-items-center"),
    ),
], style={"padding": "4px"})

# ── tab: Forecast ──────────────────────────────────────────────────────────────
def _stat_row(label, value, color=C["gold"]):
    return html.Div([
        html.Span(f"{label}: ", style={"color": C["muted"], "fontSize": "0.8rem"}),
        html.Span(value, style={"color": color, "fontWeight": "700", "fontSize": "0.9rem"}),
    ], className="mb-1")


_METRIC_GLOSSARY = html.Div(
    "MAE = avg error in pts · RMSE = penalises big errors · MAPE = error as % of actual.",
    style={"fontSize": "0.72rem", "color": C["muted"], "fontStyle": "italic", "marginTop": "4px"},
)

_forecast_metrics_div = html.Div([
    _stat_row("MAE",  f"{_TEST_MAE:.2f} index pts",  C["gold"]),
    _stat_row("RMSE", f"{_TEST_RMSE:.2f} index pts", C["orange"]),
    _stat_row("MAPE", f"{_TEST_MAPE:.1f}%",          C["red"]),
    html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
    html.Div(
        "Evaluation model trained 2010–2022 only, tested on 2023–2024 (n = 24 months).",
        style={"fontSize": "0.75rem", "color": C["muted"]},
    ),
    _METRIC_GLOSSARY,
])

def _build_fc_targets_div(year: int, fc_bear=None, fc_base=None, fc_bull=None) -> html.Div:
    """Scenario targets card for the mid-market composite forecast tab."""
    if fc_bear is None:
        fc_bear, fc_base, fc_bull = DF_FC_BEAR, DF_FC_BASE, DF_FC_BULL
    decs = {"bear": _fc_dec(fc_bear, year),
            "base": _fc_dec(fc_base, year),
            "bull": _fc_dec(fc_bull, year)}
    rows = [
        html.Div([
            html.Span("● ", style={"color": SCENARIO_STYLES[n][0], "fontSize": "1rem"}),
            html.Span(f"{SCENARIO_STYLES[n][2]}: ",
                      style={"color": C["muted"], "fontSize": "0.8rem"}),
            html.Span(f"{decs[n]['yhat']:.1f}",
                      style={"color": SCENARIO_STYLES[n][0],
                             "fontWeight": "700", "fontSize": "0.9rem"}),
            html.Span(f"  [{decs[n]['yhat_lower']:.1f} – {decs[n]['yhat_upper']:.1f}]",
                      style={"color": C["muted"], "fontSize": "0.78rem"}),
        ], className="mb-2")
        for n in ["bear", "base", "bull"]
    ]
    return html.Div([
        *rows,
        html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
        html.Div(f"Dec {year} AHPI under each scenario. 90% credible interval in brackets.",
                 style={"fontSize": "0.75rem", "color": C["muted"]}),
    ])

tab_forecast = html.Div([
    section_card(
        dbc.Row([
            dbc.Col([
                html.Label("90% Confidence Intervals",
                           style={"fontSize": "0.78rem", "color": C["muted"]}),
                dbc.Switch(id="forecast-ci", value=True, label=""),
            ], md=2, className="d-flex flex-column justify-content-start"),
            dbc.Col([
                html.Label("Target year",
                           style={"fontSize": "0.78rem", "color": C["muted"]}),
                dcc.Dropdown(
                    id="forecast-year", options=FC_YEAR_OPTS, value=2026,
                    clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Div(
                    "Prophet mid-market composite model · 180 months · 6 macro regressors. "
                    "Forecasts extend to Dec 2029 across Bear / Base / Bull scenarios.",
                    style={"fontSize": "0.75rem", "color": C["muted"], "paddingTop": "8px"},
                ),
            ], md=8),
        ], className="mb-2"),
        _dl_btn("dl-forecast-btn", "⬇  Download ZIP"),
        dcc.Download(id="dl-forecast"),
        dcc.Graph(
            id="forecast-chart",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d"],
                "toImageButtonOptions": {"scale": 2},
            },
        ),
    ),
    dbc.Row([
        dbc.Col(
            section_card(
                html.P("Test-set accuracy  (2023–2024  ·  n = 24)",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                _forecast_metrics_div,
            ), md=4,
        ),
        dbc.Col(
            section_card(
                html.P(id="forecast-targets-heading",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                html.Div(id="forecast-targets"),
            ), md=8,
        ),
    ]),
])

# ── tab: Prime Forecast ────────────────────────────────────────────────────────
tab_prime_forecast = html.Div([
    section_card(
        dbc.Row([
            dbc.Col([
                html.Label("Prime area", style={"fontSize": "0.78rem", "color": C["muted"]}),
                dcc.Dropdown(
                    id="prime-fc-area",
                    options=(
                        [{"label": "All Areas (average)", "value": "all"}] +
                        [{"label": a, "value": a} for a in PRIME_AREAS]
                    ),
                    value="all",
                    clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                ),
            ], md=5),
            dbc.Col([
                html.Label("90% Confidence Intervals",
                           style={"fontSize": "0.78rem", "color": C["muted"]}),
                dbc.Switch(id="prime-fc-ci", value=True, label=""),
            ], md=2, className="d-flex flex-column justify-content-start"),
            dbc.Col([
                html.Label("Target year",
                           style={"fontSize": "0.78rem", "color": C["muted"]}),
                dcc.Dropdown(
                    id="prime-fc-year", options=FC_YEAR_OPTS, value=2026,
                    clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Div(
                    "One Prophet model per prime area (6 models). "
                    "USD-indexed markets: exchange rate is the dominant regressor. "
                    "Forecasts extend to Dec 2029.",
                    style={"fontSize": "0.75rem", "color": C["muted"], "paddingTop": "8px"},
                ),
            ], md=3),
        ], className="mb-2"),
        _dl_btn("dl-prime-forecast-btn", "⬇  Download ZIP"),
        dcc.Download(id="dl-prime-forecast"),
        dcc.Graph(
            id="prime-forecast-chart",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d"],
                "toImageButtonOptions": {"scale": 2},
            },
        ),
    ),
    dbc.Row([
        dbc.Col(
            section_card(
                html.P("Test-set accuracy  (2023–2024  ·  n = 24)",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                html.Div(id="prime-fc-metrics"),
            ), md=4,
        ),
        dbc.Col(
            section_card(
                html.P(id="prime-fc-targets-heading",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                html.Div(id="prime-fc-targets"),
            ), md=8,
        ),
    ]),
])

# ── tab: District Forecast ─────────────────────────────────────────────────────
tab_district_forecast = html.Div([
    section_card(
        dbc.Row([
            dbc.Col([
                html.Label("District", style={"fontSize": "0.78rem", "color": C["muted"]}),
                dcc.Dropdown(
                    id="district-fc-area",
                    options=(
                        [{"label": "All Districts (average)", "value": "all"}] +
                        [{"label": d, "value": d} for d in DISTRICTS]
                    ),
                    value="all",
                    clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                ),
            ], md=5),
            dbc.Col([
                html.Label("90% Confidence Intervals",
                           style={"fontSize": "0.78rem", "color": C["muted"]}),
                dbc.Switch(id="district-fc-ci", value=True, label=""),
            ], md=2, className="d-flex flex-column justify-content-start"),
            dbc.Col([
                html.Label("Target year",
                           style={"fontSize": "0.78rem", "color": C["muted"]}),
                dcc.Dropdown(
                    id="district-fc-year", options=FC_YEAR_OPTS, value=2026,
                    clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Div(
                    "One Prophet model per mid-market district (5 models). "
                    "GHS-denominated; exchange rate and CPI are the primary regressors. "
                    "Forecasts extend to Dec 2029.",
                    style={"fontSize": "0.75rem", "color": C["muted"], "paddingTop": "8px"},
                ),
            ], md=3),
        ], className="mb-2"),
        _dl_btn("dl-district-forecast-btn", "⬇  Download ZIP"),
        dcc.Download(id="dl-district-forecast"),
        dcc.Graph(
            id="district-forecast-chart",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d"],
                "toImageButtonOptions": {"scale": 2},
            },
        ),
    ),
    dbc.Row([
        dbc.Col(
            section_card(
                html.P("Test-set accuracy  (2023–2024  ·  n = 24)",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                html.Div(id="district-fc-metrics"),
            ), md=4,
        ),
        dbc.Col(
            section_card(
                html.P(id="district-fc-targets-heading",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                html.Div(id="district-fc-targets"),
            ), md=8,
        ),
    ]),
])

# ── Phase 2: pre-computed historical December snapshots ───────────────────────
# Each dict: { year: pd.Series (row from the dataset at Dec of that year) }

_HIST_DEC_COMPOSITE: dict[int, pd.Series] = {
    int(row["ds"].year): row
    for _, row in DF[DF["ds"].dt.month == 12].iterrows()
}
_HIST_DEC_DISTRICT: dict[str, dict[int, pd.Series]] = {
    d: {int(row["ds"].year): row
        for _, row in grp[grp["ds"].dt.month == 12].iterrows()}
    for d, grp in DF_DISTRICT.groupby("district")
}
_HIST_DEC_PRIME: dict[str, dict[int, pd.Series]] = {
    a: {int(row["ds"].year): row
        for _, row in grp[grp["ds"].dt.month == 12].iterrows()}
    for a, grp in DF_PRIME.groupby("district")
}

# Median monthly household income proxy (GHS, 2024 estimate) for affordability
GHANA_MEDIAN_INCOME_GHS = 4_000

# Market label → lookup key (used by both calculator tabs)
_MARKET_OPTS = (
    [{"label": "⬛  Composite Mid-Market", "value": "composite"}] +
    [{"label": f"🔷  {d}", "value": d} for d in DISTRICTS] +
    [{"label": f"🔶  {a}", "value": a} for a in PRIME_AREAS]
)

def _get_hist_dec(market: str, year: int) -> pd.Series | None:
    if market == "composite":
        return _HIST_DEC_COMPOSITE.get(year)
    if market in DISTRICTS:
        return _HIST_DEC_DISTRICT.get(market, {}).get(year)
    return _HIST_DEC_PRIME.get(market, {}).get(year)

def _get_fc_dec(market: str, scenario: str, year: int) -> pd.Series | None:
    if market == "composite":
        df_fc = {"bear": DF_FC_BEAR, "base": DF_FC_BASE, "bull": DF_FC_BULL}[scenario]
    elif market in DISTRICTS:
        df_fc = _DISTRICT_FC.get((scenario, market))
    else:
        df_fc = _PRIME_FC.get((scenario, market))
    return None if df_fc is None else _fc_dec(df_fc, year)

def _result_card(label: str, value: str, sub: str = "", color: str = C["gold"]) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "0.7rem", "color": C["muted"],
                                "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        html.Div(value, style={"fontSize": "1.1rem", "fontWeight": "700", "color": color}),
        html.Div(sub,   style={"fontSize": "0.72rem", "color": C["muted"]}),
    ], style={"backgroundColor": C["hover"], "padding": "10px 14px", "borderRadius": "6px",
              "border": f"1px solid {C['border']}", "marginBottom": "8px"})

def _scenario_col(sc: str, label: str, color: str, content: list) -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div(label, style={"fontWeight": "700", "color": color,
                               "fontSize": "0.85rem", "marginBottom": "8px",
                               "borderBottom": f"2px solid {color}", "paddingBottom": "4px"}),
        *content,
    ], style={"backgroundColor": C["card"], "padding": "12px", "borderRadius": "6px",
              "border": f"1px solid {color}33"}))

# ── PDF report generation ─────────────────────────────────────────────────────

_PDF_BG       = rl_colors.HexColor("#0d1117")
_PDF_CARD     = rl_colors.HexColor("#161b22")
_PDF_GOLD     = rl_colors.HexColor("#d4a017")
_PDF_TEXT     = rl_colors.HexColor("#e6edf3")
_PDF_MUTED    = rl_colors.HexColor("#8b949e")
_PDF_GREEN    = rl_colors.HexColor("#3fb950")
_PDF_RED      = rl_colors.HexColor("#f85149")
_PDF_BLUE     = rl_colors.HexColor("#58a6ff")
_PDF_BORDER   = rl_colors.HexColor("#30363d")


def _chart_png(market: str, width_px: int = 900, height_px: int = 360) -> bytes | None:
    """Render the forecast figure for *market* as a PNG byte string."""
    try:
        if market == "composite":
            fig = build_forecast_fig(show_ci=True)
        elif market in DISTRICTS:
            fig = build_district_forecast_fig(market, show_ci=True)
        else:
            fig = build_prime_forecast_fig(market, show_ci=True)
        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            width=width_px, height=height_px,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        return pio.to_image(fig, format="png", engine="kaleido")
    except Exception:
        return None


def _pdf_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18,
                                textColor=_PDF_GOLD, alignment=TA_LEFT, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10,
                                   textColor=_PDF_MUTED, alignment=TA_LEFT, spaceAfter=10),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=11,
                                  textColor=_PDF_GOLD, alignment=TA_LEFT,
                                  spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8.5,
                               textColor=_PDF_TEXT, alignment=TA_LEFT, spaceAfter=3),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.5,
                                textColor=_PDF_MUTED, alignment=TA_LEFT),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8,
                               textColor=_PDF_TEXT),
        "cell_bold": ParagraphStyle("cell_bold", fontName="Helvetica-Bold", fontSize=8,
                                    textColor=_PDF_GOLD),
        "right": ParagraphStyle("right", fontName="Helvetica", fontSize=8,
                                textColor=_PDF_TEXT, alignment=TA_RIGHT),
        "disclaimer": ParagraphStyle("disclaimer", fontName="Helvetica-Oblique", fontSize=7,
                                     textColor=_PDF_MUTED, alignment=TA_CENTER, spaceAfter=4),
    }
    return styles


def generate_market_pdf(market: str, report_year: int) -> bytes:
    """Generate a 2-3 page PDF market report for the given market and forecast year."""
    buf = io.BytesIO()
    W, H = A4
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm,
    )

    styles = _pdf_styles()
    story  = []

    # Determine display label
    if market == "composite":
        label = "Composite Mid-Market"
        family = "mid-market"
    elif market in DISTRICTS:
        label = market
        family = "mid-market district"
    else:
        label = market
        family = "prime area"

    # ── Header ──────────────────────────────────────────────────────────────────
    story.append(Paragraph("ACCRA HOME PRICE INDEX", styles["title"]))
    story.append(Paragraph(
        f"Market Report — {label} ({family.title()}) · Forecast Horizon: Dec {report_year}",
        styles["subtitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.date.today().strftime('%d %B %Y')}  ·  "
        "Source: AHPI Prophet v2.1  ·  Base year: 2015 = 100",
        styles["small"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=_PDF_GOLD, spaceAfter=8))

    # ── Current snapshot (latest historical Dec) ─────────────────────────────
    story.append(Paragraph("Current Market Snapshot (Dec 2024)", styles["section"]))

    hist_2024 = _get_hist_dec(market, 2024)
    hist_2023 = _get_hist_dec(market, 2023)

    if hist_2024 is not None:
        ahpi_now  = float(hist_2024.get("y", "—"))
        ghs_sqm   = hist_2024.get("price_ghs_per_sqm")
        usd_sqm   = hist_2024.get("price_usd_per_sqm")
        if hist_2023 is not None:
            ahpi_prev = float(hist_2023.get("y", ahpi_now))
            yoy_pct   = (ahpi_now - ahpi_prev) / ahpi_prev * 100 if ahpi_prev else 0
            yoy_str   = f"{yoy_pct:+.1f}% YoY"
        else:
            yoy_str = "—"

        snap_data = [
            [Paragraph("Metric", styles["cell_bold"]),
             Paragraph("Value", styles["cell_bold"]),
             Paragraph("Note", styles["cell_bold"])],
            [Paragraph("AHPI (Dec 2024)", styles["cell"]),
             Paragraph(f"{ahpi_now:.1f}", styles["cell"]),
             Paragraph(yoy_str, styles["cell"])],
        ]
        if ghs_sqm is not None:
            snap_data.append([
                Paragraph("GHS / sqm", styles["cell"]),
                Paragraph(f"GHS {float(ghs_sqm):,.0f}", styles["cell"]),
                Paragraph("", styles["cell"]),
            ])
        if usd_sqm is not None:
            snap_data.append([
                Paragraph("USD / sqm", styles["cell"]),
                Paragraph(f"USD {float(usd_sqm):,.0f}", styles["cell"]),
                Paragraph("At 2024 GHS/USD rate", styles["cell"]),
            ])

        col_w = [(W - 36*mm) * f for f in (0.45, 0.25, 0.30)]
        snap_tbl = Table(snap_data, colWidths=col_w, repeatRows=1)
        snap_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _PDF_GOLD),
            ("TEXTCOLOR",  (0, 0), (-1, 0), _PDF_BG),
            ("BACKGROUND", (0, 1), (-1, -1), _PDF_CARD),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_PDF_CARD, _PDF_BG]),
            ("GRID",       (0, 0), (-1, -1), 0.3, _PDF_BORDER),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(snap_tbl)
        story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No historical data available for Dec 2024.", styles["body"]))

    # ── Scenario Forecast Table ──────────────────────────────────────────────
    story.append(Paragraph(f"Scenario Forecasts — Dec {report_year}", styles["section"]))
    story.append(Paragraph(
        "Three economic scenarios modelled with Facebook Prophet. "
        "Bear assumes continued GHS/USD depreciation (→ 20). "
        "Base assumes gradual stabilisation (→ 15). "
        "Bull assumes cedi recovery (→ 12). 90% credible interval shown.",
        styles["small"]))
    story.append(Spacer(1, 4))

    fc_rows = [
        [Paragraph("Scenario", styles["cell_bold"]),
         Paragraph(f"Dec {report_year} AHPI", styles["cell_bold"]),
         Paragraph("Lower (90%)", styles["cell_bold"]),
         Paragraph("Upper (90%)", styles["cell_bold"]),
         Paragraph("GHS/USD Assumption", styles["cell_bold"])],
    ]
    sc_label_map = {"bear": "Bear", "base": "Base", "bull": "Bull"}
    sc_color_map = {"bear": _PDF_RED, "base": _PDF_BLUE, "bull": _PDF_GREEN}
    for sc in ("bear", "base", "bull"):
        fc_row = _get_fc_dec(market, sc, report_year)
        if fc_row is not None:
            fc_rows.append([
                Paragraph(sc_label_map[sc], styles["cell"]),
                Paragraph(f"{float(fc_row['yhat']):.1f}", styles["cell"]),
                Paragraph(f"{float(fc_row['yhat_lower']):.1f}", styles["cell"]),
                Paragraph(f"{float(fc_row['yhat_upper']):.1f}", styles["cell"]),
                Paragraph(f"{SCENARIO_FX[sc]:.1f}", styles["cell"]),
            ])
        else:
            fc_rows.append([Paragraph(sc_label_map[sc], styles["cell"])] + [Paragraph("—", styles["cell"])] * 4)

    fc_col_w = [(W - 36*mm) * f for f in (0.18, 0.20, 0.20, 0.20, 0.22)]
    fc_tbl = Table(fc_rows, colWidths=fc_col_w, repeatRows=1)
    fc_style = [
        ("BACKGROUND", (0, 0), (-1, 0), _PDF_GOLD),
        ("TEXTCOLOR",  (0, 0), (-1, 0), _PDF_BG),
        ("GRID",       (0, 0), (-1, -1), 0.3, _PDF_BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i, sc in enumerate(("bear", "base", "bull"), start=1):
        fc_style.append(("BACKGROUND", (0, i), (0, i), sc_color_map[sc]))
        fc_style.append(("TEXTCOLOR",  (0, i), (0, i), _PDF_BG))
        row_bg = _PDF_CARD if i % 2 == 1 else _PDF_BG
        fc_style.append(("BACKGROUND", (1, i), (-1, i), row_bg))
    fc_tbl.setStyle(TableStyle(fc_style))
    story.append(fc_tbl)
    story.append(Spacer(1, 8))

    # ── Model accuracy metrics ───────────────────────────────────────────────
    story.append(Paragraph("Model Accuracy (2023–2024 Test Set)", styles["section"]))

    if market == "composite":
        mae  = _TEST_MAE
        rmse = _TEST_RMSE
        mape = _TEST_MAPE
    elif market in DISTRICTS and market in _DISTRICT_TEST_EVALS:
        te   = _DISTRICT_TEST_EVALS[market]
        mae  = (te["y"] - te["yhat"]).abs().mean()
        rmse = ((te["y"] - te["yhat"]) ** 2).mean() ** 0.5
        mape = ((te["y"] - te["yhat"]).abs() / te["y"]).mean() * 100
    elif market in _PRIME_TEST_EVALS:
        te   = _PRIME_TEST_EVALS[market]
        mae  = (te["y"] - te["yhat"]).abs().mean()
        rmse = ((te["y"] - te["yhat"]) ** 2).mean() ** 0.5
        mape = ((te["y"] - te["yhat"]).abs() / te["y"]).mean() * 100
    else:
        mae = rmse = mape = None

    if mae is not None:
        acc_data = [
            [Paragraph("MAE", styles["cell_bold"]),
             Paragraph("RMSE", styles["cell_bold"]),
             Paragraph("MAPE", styles["cell_bold"])],
            [Paragraph(f"{mae:.2f} pts", styles["cell"]),
             Paragraph(f"{rmse:.2f} pts", styles["cell"]),
             Paragraph(f"{mape:.1f}%", styles["cell"])],
        ]
        acc_col_w = [(W - 36*mm) / 3] * 3
        acc_tbl = Table(acc_data, colWidths=acc_col_w, repeatRows=1)
        acc_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _PDF_GOLD),
            ("TEXTCOLOR",  (0, 0), (-1, 0), _PDF_BG),
            ("BACKGROUND", (0, 1), (-1, -1), _PDF_CARD),
            ("GRID",       (0, 0), (-1, -1), 0.3, _PDF_BORDER),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(acc_tbl)
    else:
        story.append(Paragraph("Accuracy metrics unavailable.", styles["body"]))

    story.append(Spacer(1, 10))

    # ── Forecast chart ───────────────────────────────────────────────────────
    story.append(Paragraph("Historical AHPI & Scenario Forecasts", styles["section"]))

    png_bytes = _chart_png(market)
    if png_bytes:
        img_buf = io.BytesIO(png_bytes)
        chart_w = W - 36*mm
        chart_h = chart_w * 360 / 900
        story.append(RLImage(img_buf, width=chart_w, height=chart_h))
    else:
        story.append(Paragraph("Chart unavailable (kaleido not configured).", styles["small"]))

    story.append(Spacer(1, 10))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.4, color=_PDF_BORDER, spaceAfter=4))
    story.append(Paragraph(
        "This report is generated from the Accra Home Price Index (AHPI) model for informational purposes only. "
        "Forecasts are probabilistic outputs of a Facebook Prophet model trained on historical data. "
        "They do not constitute financial, investment, or legal advice. "
        "Past performance does not guarantee future results. "
        "Always conduct independent due diligence before making property investment decisions.",
        styles["disclaimer"],
    ))

    doc.build(story)
    return buf.getvalue()


# ── tab: Market Report (Phase 3.1) ────────────────────────────────────────────

_REPORT_YEAR_OPTS = [{"label": str(y), "value": y} for y in FC_YEARS]

tab_report = html.Div([
    section_card(
        html.P(
            "Generate a downloadable PDF market report for any district or prime area. "
            "Each report includes a current snapshot, scenario forecast table, model accuracy metrics, and a chart.",
            style={"color": C["muted"], "fontSize": "0.8rem", "marginBottom": "14px"},
        ),
        dbc.Row([
            dbc.Col([
                html.Div("Market / area", style={"fontSize": "0.75rem", "color": C["muted"],
                                                  "marginBottom": "3px"}),
                dcc.Dropdown(id="report-market", options=_MARKET_OPTS, value="composite",
                             clearable=False,
                             style={"backgroundColor": C["bg"], "color": C["text"],
                                    "fontSize": "0.82rem", "marginBottom": "10px"}),
                html.Div("Forecast target year", style={"fontSize": "0.75rem",
                                                         "color": C["muted"], "marginBottom": "3px"}),
                dcc.Dropdown(id="report-year", options=_REPORT_YEAR_OPTS, value=2027,
                             clearable=False,
                             style={"backgroundColor": C["bg"], "fontSize": "0.82rem",
                                    "marginBottom": "16px"}),
                dbc.Button(
                    "Generate & Download PDF Report",
                    id="report-pdf-btn", color="warning", outline=True, size="sm",
                    style={"fontWeight": "600", "fontSize": "0.82rem", "width": "100%"},
                ),
                dcc.Download(id="report-pdf-dl"),
            ], md=4),
            dbc.Col([
                html.Div(id="report-preview", style={"color": C["muted"],
                                                      "fontSize": "0.8rem", "lineHeight": "1.7"}),
            ], md=8),
        ], className="g-3"),
    )
], style={"padding": "4px"})


# ── tab: Market Snapshot Card (Phase 3.2) ─────────────────────────────────────

tab_snapshot = html.Div([
    section_card(
        html.P(
            "A single-page market briefing card — key numbers at a glance, optimised for sharing or printing.",
            style={"color": C["muted"], "fontSize": "0.8rem", "marginBottom": "14px"},
        ),
        dbc.Row([
            dbc.Col([
                html.Div("Market / area", style={"fontSize": "0.75rem", "color": C["muted"],
                                                  "marginBottom": "3px"}),
                dcc.Dropdown(id="snap-market", options=_MARKET_OPTS, value="composite",
                             clearable=False,
                             style={"backgroundColor": C["bg"], "color": C["text"],
                                    "fontSize": "0.82rem", "marginBottom": "10px"}),
                html.Div("Forecast target year", style={"fontSize": "0.75rem",
                                                         "color": C["muted"], "marginBottom": "3px"}),
                dcc.Dropdown(id="snap-year", options=_REPORT_YEAR_OPTS, value=2027,
                             clearable=False,
                             style={"backgroundColor": C["bg"], "fontSize": "0.82rem",
                                    "marginBottom": "16px"}),
            ], md=3),
            dbc.Col([
                html.Div(id="snap-card-container"),
            ], md=9),
        ], className="g-3"),
    )
], style={"padding": "4px"})


# ── tab: Investment Return Calculator ─────────────────────────────────────────
_INV_BUY_OPTS = [{"label": str(y), "value": y} for y in range(2010, 2025)]
_INV_SELL_OPTS = FC_YEAR_OPTS

tab_invest = html.Div([
    section_card(
        html.P(
            "Estimate the return on a residential property investment in Accra under each scenario. "
            "Historical prices are taken from the AHPI dataset; future values are Prophet model forecasts.",
            style={"color": C["muted"], "fontSize": "0.8rem", "marginBottom": "14px"},
        ),
        dbc.Row([
            # ── inputs ──────────────────────────────────────────────────────
            dbc.Col([
                html.Div("Market / area", style={"fontSize": "0.75rem", "color": C["muted"],
                                                  "marginBottom": "3px"}),
                dcc.Dropdown(id="inv-market", options=_MARKET_OPTS, value="composite",
                             clearable=False,
                             style={"backgroundColor": C["bg"], "color": C["text"],
                                    "fontSize": "0.82rem", "marginBottom": "10px"}),

                dbc.Row([
                    dbc.Col([
                        html.Div("Buy year", style={"fontSize": "0.75rem", "color": C["muted"],
                                                     "marginBottom": "3px"}),
                        dcc.Dropdown(id="inv-buy-year", options=_INV_BUY_OPTS, value=2020,
                                     clearable=False,
                                     style={"backgroundColor": C["bg"], "fontSize": "0.82rem"}),
                    ], md=6),
                    dbc.Col([
                        html.Div("Sell year", style={"fontSize": "0.75rem", "color": C["muted"],
                                                      "marginBottom": "3px"}),
                        dcc.Dropdown(id="inv-sell-year", options=_INV_SELL_OPTS, value=2027,
                                     clearable=False,
                                     style={"backgroundColor": C["bg"], "fontSize": "0.82rem"}),
                    ], md=6),
                ], className="mb-2"),

                html.Div("Property size (sqm)", style={"fontSize": "0.75rem",
                                                         "color": C["muted"], "marginBottom": "3px"}),
                dcc.Input(id="inv-sqm", type="number", value=100, min=10, max=10000, step=10,
                          debounce=True,
                          style={"backgroundColor": C["bg"], "color": C["text"],
                                 "border": f"1px solid {C['border']}", "borderRadius": "4px",
                                 "padding": "5px 10px", "width": "100%",
                                 "fontSize": "0.85rem", "marginBottom": "16px"}),

                html.Hr(style={"borderColor": C["border"]}),
                html.Div([
                    html.Div("Buy price (GHS/sqm)", style={"fontSize": "0.72rem",
                                                             "color": C["muted"]}),
                    html.Div(id="inv-buy-price-display",
                             style={"fontWeight": "600", "color": C["gold"],
                                    "fontSize": "0.9rem"}),
                ], className="mb-2"),
                html.Div([
                    html.Div("Buy price (USD/sqm)", style={"fontSize": "0.72rem",
                                                             "color": C["muted"]}),
                    html.Div(id="inv-buy-usd-display",
                             style={"fontWeight": "600", "color": C["blue"],
                                    "fontSize": "0.9rem"}),
                ], className="mb-2"),
                html.Div([
                    html.Div("GHS/USD at purchase", style={"fontSize": "0.72rem",
                                                             "color": C["muted"]}),
                    html.Div(id="inv-buy-fx-display",
                             style={"fontWeight": "600", "color": C["muted"],
                                    "fontSize": "0.9rem"}),
                ]),
            ], md=3, style={"borderRight": f"1px solid {C['border']}", "paddingRight": "16px"}),

            # ── results ─────────────────────────────────────────────────────
            dbc.Col([
                html.Div(id="inv-results"),
            ], md=9),
        ]),
    ),
])

# ── tab: Mortgage Stress Test ─────────────────────────────────────────────────
tab_mortgage = html.Div([
    section_card(
        html.P(
            "Estimate monthly repayments and stress-test collateral values against each "
            "scenario. Mid-market districts only (GHS-denominated mortgages).",
            style={"color": C["muted"], "fontSize": "0.8rem", "marginBottom": "14px"},
        ),
        dbc.Row([
            # ── inputs ──────────────────────────────────────────────────────
            dbc.Col([
                html.Div("District", style={"fontSize": "0.75rem", "color": C["muted"],
                                             "marginBottom": "3px"}),
                dcc.Dropdown(
                    id="mort-district",
                    options=[{"label": d, "value": d} for d in DISTRICTS],
                    value=DISTRICTS[0], clearable=False,
                    style={"backgroundColor": C["bg"], "color": C["text"],
                           "fontSize": "0.82rem", "marginBottom": "10px"},
                ),

                html.Div("Property size (sqm)", style={"fontSize": "0.75rem",
                                                         "color": C["muted"], "marginBottom": "3px"}),
                dcc.Input(id="mort-sqm", type="number", value=100, min=10, max=5000, step=10,
                          debounce=True,
                          style={"backgroundColor": C["bg"], "color": C["text"],
                                 "border": f"1px solid {C['border']}", "borderRadius": "4px",
                                 "padding": "5px 10px", "width": "100%",
                                 "fontSize": "0.85rem", "marginBottom": "10px"}),

                html.Div("Property value (GHS) — auto-filled, editable",
                         style={"fontSize": "0.75rem", "color": C["muted"], "marginBottom": "3px"}),
                dcc.Input(id="mort-value", type="number", min=1000, step=1000, debounce=True,
                          style={"backgroundColor": C["bg"], "color": C["text"],
                                 "border": f"1px solid {C['border']}", "borderRadius": "4px",
                                 "padding": "5px 10px", "width": "100%",
                                 "fontSize": "0.85rem", "marginBottom": "10px"}),

                html.Div("Loan-to-value (%)",
                         style={"fontSize": "0.75rem", "color": C["muted"], "marginBottom": "3px"}),
                dcc.Slider(id="mort-ltv", min=50, max=80, step=5, value=70,
                           marks={v: {"label": f"{v}%", "style": {"color": C["muted"],
                                                                    "fontSize": "0.7rem"}}
                                  for v in range(50, 85, 5)},
                           tooltip={"always_visible": False}),

                html.Div("Loan term (years)",
                         style={"fontSize": "0.75rem", "color": C["muted"],
                                "marginBottom": "3px", "marginTop": "10px"}),
                dcc.Dropdown(
                    id="mort-term",
                    options=[{"label": f"{y} years", "value": y} for y in [10, 15, 20, 25]],
                    value=20, clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem",
                           "marginBottom": "10px"},
                ),

                html.Div("Annual interest rate (%)",
                         style={"fontSize": "0.75rem", "color": C["muted"], "marginBottom": "3px"}),
                dcc.Input(id="mort-rate", type="number", value=28.0, min=1, max=60,
                          step=0.5, debounce=True,
                          style={"backgroundColor": C["bg"], "color": C["text"],
                                 "border": f"1px solid {C['border']}", "borderRadius": "4px",
                                 "padding": "5px 10px", "width": "100%",
                                 "fontSize": "0.85rem", "marginBottom": "10px"}),

                html.Div("Collateral check year",
                         style={"fontSize": "0.75rem", "color": C["muted"], "marginBottom": "3px"}),
                dcc.Dropdown(
                    id="mort-year", options=FC_YEAR_OPTS, value=2027, clearable=False,
                    style={"backgroundColor": C["bg"], "fontSize": "0.82rem"},
                ),
            ], md=3, style={"borderRight": f"1px solid {C['border']}", "paddingRight": "16px"}),

            # ── results ─────────────────────────────────────────────────────
            dbc.Col([
                html.Div(id="mort-results"),
            ], md=9),
        ]),
    ),
])

# ── stakeholder role configuration ────────────────────────────────────────────
# Each role maps to a landing-page card, a default dashboard tab, 3 headline KPIs,
# and up to 4 quick-jump tab shortcuts shown in the in-dashboard role banner.
ROLES: dict[str, dict] = {
    "diaspora": {
        "label":      "Diaspora Investor",
        "icon":       "💼",
        "tagline":    "Track USD returns across prime Accra neighbourhoods",
        "detail":     "Prime areas delivered +232% USD returns over 14 years — a category apart from mid-market's +34%.",
        "tab":        "tab-prime",
        "accent":     C["gold"],
        "kpis":       [("Prime USD/sqm", "USD 2,874"), ("East Legon gain", "+270% USD"), ("vs Mid-Market", "+232% vs +34%")],
        "quick_tabs": [("Prime Areas", "tab-prime"), ("Prime Forecast", "tab-prime-forecast"),
                       ("Invest. Calc", "tab-invest"), ("Map", "tab-map")],
    },
    "lender": {
        "label":      "Mortgage Lender",
        "icon":       "🏦",
        "tagline":    "Monitor collateral values and LTV risk across districts",
        "detail":     "Mid-market AHPI +1,303% GHS but only +34% USD — stress-test your book against FX scenarios.",
        "tab":        "tab-mortgage",
        "accent":     C["blue"],
        "kpis":       [("AHPI Dec 2024", "419.7"), ("Lending rate", "~27%"), ("FX–AHPI corr.", "0.991")],
        "quick_tabs": [("Mortgage Stress", "tab-mortgage"), ("Districts", "tab-districts"),
                       ("Dist. Forecast", "tab-district-forecast"), ("Overview", "tab-overview")],
    },
    "developer": {
        "label":      "Real Estate Developer",
        "icon":       "🏗",
        "tagline":    "Find the districts with the strongest margin and growth",
        "detail":     "Kasoa: lowest GHS/sqm (10,156) + fastest USD gain (+88%). Prime margins are 2.9× mid-market.",
        "tab":        "tab-districts",
        "accent":     C["orange"],
        "kpis":       [("Kasoa USD gain", "+88%"), ("Spintex USD/sqm", "USD 1,374"), ("Prime/Mid ratio", "2.9×")],
        "quick_tabs": [("Districts", "tab-districts"), ("Map", "tab-map"),
                       ("Prime Areas", "tab-prime"), ("Dist. Forecast", "tab-district-forecast")],
    },
    "institutional": {
        "label":      "Institutional Investor",
        "icon":       "📊",
        "tagline":    "Model Bear / Base / Bull return scenarios across 11 areas",
        "detail":     "Prime AHPI annualised ~8.8% USD over 14 years. Base scenario targets AHPI 854 by Dec 2026.",
        "tab":        "tab-prime-forecast",
        "accent":     C["purple"],
        "kpis":       [("Prime ann. USD", "~8.8%/yr"), ("Gross rental yield", "8–12%"), ("Base AHPI 2026", "~854 avg")],
        "quick_tabs": [("Prime Forecast", "tab-prime-forecast"), ("Dist. Forecast", "tab-district-forecast"),
                       ("Forecast", "tab-forecast"), ("Invest. Calc", "tab-invest")],
    },
    "policy": {
        "label":      "Policy Maker",
        "icon":       "🏛",
        "tagline":    "Assess housing affordability and macro transmission dynamics",
        "detail":     "FX–AHPI correlation is 0.991. Kasoa is the strongest target for mass-housing programmes.",
        "tab":        "tab-overview",
        "accent":     C["teal"],
        "kpis":       [("FX–AHPI corr.", "0.991"), ("Housing deficit", "~1.8 M units"), ("Kasoa GHS/sqm", "10,156")],
        "quick_tabs": [("Overview", "tab-overview"), ("Macro Drivers", "tab-macro"),
                       ("Districts", "tab-districts"), ("Forecast", "tab-forecast")],
    },
    "researcher": {
        "label":      "Researcher / Data Scientist",
        "icon":       "🔬",
        "tagline":    "Explore 20 macro regressors and Prophet model diagnostics",
        "detail":     "12 Prophet models trained across Bear/Base/Bull. Composite MAPE 7.8%; prime avg MAPE 2.8%.",
        "tab":        "tab-macro",
        "accent":     C["green"],
        "kpis":       [("Top ρ (FX rate)", "0.991"), ("Composite MAPE", "7.8%"), ("Prime avg MAPE", "2.8%")],
        "quick_tabs": [("Macro Drivers", "tab-macro"), ("Reg. Explorer", "tab-explorer"),
                       ("Forecast", "tab-forecast"), ("Prime Forecast", "tab-prime-forecast")],
    },
}

# ── app layout ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY,
                           "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"],
    title="Accra Home Price Index",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server

# ── landing page ───────────────────────────────────────────────────────────────
_HERO_STATS = [
    ("+1,303%", "GHS Growth",      "Mid-market nominal 2010 – 2024"),
    ("+232%",   "USD Return",      "Prime areas real USD appreciation"),
    ("11",      "Areas Tracked",   "5 mid-market districts + 6 prime"),
    ("2029",    "Forecast Horizon","Bear · Base · Bull Prophet scenarios"),
]

def _make_landing() -> html.Div:
    """Full-screen landing page with hero stats and role-selection cards."""

    _SHARED_FONT = "'Inter', 'Segoe UI', Arial, sans-serif"

    # ── 4 headline stat chips ────────────────────────────────────────────────
    stat_row = dbc.Row([
        dbc.Col(
            html.Div([
                html.Div(val,   style={"fontSize": "1.9rem", "fontWeight": "700",
                                       "color": C["gold"], "lineHeight": "1"}),
                html.Div(label, style={"fontSize": "0.88rem", "fontWeight": "600",
                                       "color": C["text"], "marginTop": "5px"}),
                html.Div(sub,   style={"fontSize": "0.68rem", "color": C["muted"],
                                       "marginTop": "3px"}),
            ], className="ahpi-stat-chip", style={
                "backgroundColor": C["card"],
                "border":          f"1px solid {C['border']}",
                "borderRadius":    "10px",
                "padding":         "18px 16px",
                "textAlign":       "center",
            }),
            md=3, xs=6, className="mb-3",
        )
        for val, label, sub in _HERO_STATS
    ], className="g-3 mb-5 justify-content-center ahpi-stats")

    # ── role cards (6 roles, 3-column grid) ─────────────────────────────────
    role_cards = dbc.Row([
        dbc.Col(
            html.Div(
                dbc.Button(
                    html.Div([
                        html.Div(info["icon"], style={"fontSize": "2.2rem",
                                                      "marginBottom": "10px",
                                                      "lineHeight": "1"}),
                        html.Div(info["label"],
                                 style={"fontWeight": "700", "fontSize": "0.92rem",
                                        "color": info["accent"], "marginBottom": "8px"}),
                        html.Div(info["tagline"],
                                 style={"fontSize": "0.73rem", "color": C["muted"],
                                        "lineHeight": "1.5"}),
                        html.Div("Enter  →",
                                 style={"marginTop": "16px", "fontSize": "0.78rem",
                                        "color": info["accent"], "fontWeight": "600",
                                        "letterSpacing": "0.04em"}),
                    ], style={"textAlign": "center"}),
                    id={"type": "role-btn", "role": role_key},
                    n_clicks=0, color="link",
                    style={
                        "backgroundColor": C["card"],
                        "border":          f"1px solid {C['border']}",
                        "borderRadius":    "12px",
                        "padding":         "28px 18px",
                        "width":           "100%",
                        "textDecoration":  "none",
                    },
                    className="ahpi-role-card",
                ),
                style={"height": "100%"},
            ),
            md=4, xs=12, className="mb-3",
        )
        for role_key, info in ROLES.items()
    ], className="g-3 mb-4 justify-content-center ahpi-roles")

    return html.Div([

        # ── top nav bar ──────────────────────────────────────────────────────
        html.Div(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Span("🏠 ", style={"fontSize": "1.2rem"}),
                        html.Span("AHPI", style={"fontSize": "1.05rem", "fontWeight": "700",
                                                  "color": C["gold"], "letterSpacing": "0.08em"}),
                        html.Span(" · Accra Home Price Index",
                                  style={"fontSize": "0.82rem", "color": C["muted"],
                                         "marginLeft": "8px"}),
                    ], className="d-flex align-items-center")),
                    dbc.Col(
                        dcc.Link(
                            dbc.Button("Enter Dashboard →", color="warning", outline=True,
                                       size="sm", style={"fontSize": "0.78rem"}),
                            href="/dashboard",
                        ),
                        className="d-flex justify-content-end",
                    ),
                ], align="center"),
            ], fluid=True),
            style={"backgroundColor": C["card"],
                   "borderBottom": f"1px solid {C['border']}",
                   "padding": "12px 0"},
        ),

        # ── hero ─────────────────────────────────────────────────────────────
        dbc.Container([
            html.Div([

                # title block
                html.Div([
                    html.Div("🏠", style={"fontSize": "3.2rem", "marginBottom": "14px",
                                          "lineHeight": "1"}),
                    html.H1("Accra Home Price Index",
                            className="ahpi-title-glow",
                            style={"fontSize": "2.7rem", "fontWeight": "700",
                                   "color": C["gold"], "marginBottom": "14px",
                                   "letterSpacing": "0.02em", "lineHeight": "1.15"}),
                    html.P("Ghana's only monthly residential property benchmark",
                           style={"fontSize": "1.1rem", "color": C["text"],
                                  "marginBottom": "6px"}),
                    html.P(
                        "Jan 2010 – Dec 2024  ·  Mid-market (5 districts)  ·  "
                        "Prime areas (6 locations)  ·  Prophet forecasts to 2029",
                        style={"fontSize": "0.83rem", "color": C["muted"],
                               "marginBottom": "52px"},
                    ),
                ], className="ahpi-hero"),

                # stat chips
                stat_row,

                html.Hr(className="ahpi-section-divider", style={"margin": "0 0 44px"}),

                # role selection
                html.H5("Choose your profile for a tailored experience",
                        style={"color": C["text"], "fontWeight": "600",
                               "marginBottom": "30px", "textAlign": "center"}),
                role_cards,

                # skip link
                html.Div(
                    dcc.Link("Continue without a profile  →",
                             href="/dashboard",
                             className="ahpi-skip-link"),
                    style={"textAlign": "center", "marginBottom": "56px"},
                ),

                # footer
                html.Hr(className="ahpi-section-divider", style={"margin": "0 0 18px"}),
                html.Div(
                    "Data sources: World Bank Open Data · Bank of Ghana · LBMA · "
                    "ICCO · EIA/Platts · Global Property Guide · Numbeo · "
                    "JLL Africa · Knight Frank Africa",
                    className="ahpi-footer-row",
                    style={"color": C["muted"], "fontSize": "0.68rem",
                           "textAlign": "center", "paddingBottom": "36px"},
                ),

            ], style={"textAlign": "center", "paddingTop": "80px"}),
        ], fluid=True, style={"maxWidth": "980px"}),

    ], style={
        "backgroundColor": C["bg"],
        "minHeight":        "100vh",
        "fontFamily":       _SHARED_FONT,
        "background":       (
            f"radial-gradient(ellipse 80% 50% at 50% -10%, "
            f"rgba(212,160,23,0.07) 0%, {C['bg']} 70%)"
        ),
    })


# ── in-dashboard role banner ──────────────────────────────────────────────────
def _make_role_banner(role_key: str | None) -> html.Div:
    """Slim contextual banner beneath the main header; empty if no role is set."""
    if not role_key or role_key not in ROLES:
        return html.Div()

    info   = ROLES[role_key]
    accent = info["accent"]

    kpi_chips = html.Div([
        html.Span([
            html.Span(lbl + " ", style={"color": C["muted"],
                                        "fontSize": "0.67rem"}),
            html.Span(val, style={"color": accent, "fontWeight": "700",
                                  "fontSize": "0.8rem"}),
        ], style={
            "backgroundColor": C["hover"],
            "border":          f"1px solid {C['border']}",
            "borderRadius":    "4px",
            "padding":         "3px 8px",
            "marginRight":     "6px",
            "whiteSpace":      "nowrap",
        })
        for lbl, val in info["kpis"]
    ], className="d-flex align-items-center flex-wrap gap-1")

    quick_btns = [
        dbc.Button(
            label,
            id={"type": "banner-tab-btn", "tab": tab_id},
            n_clicks=0, size="sm", color="link",
            className="ahpi-role-banner-btn",
            style={"fontSize": "0.7rem", "color": C["muted"],
                   "padding": "1px 5px", "textDecoration": "none"},
        )
        for label, tab_id in info["quick_tabs"]
    ]

    return html.Div(
        dbc.Container([
            dbc.Row([
                # identity
                dbc.Col(html.Div([
                    html.Span(info["icon"] + " ",
                              style={"fontSize": "1.05rem", "marginRight": "6px"}),
                    html.Span(info["label"].upper(),
                              style={"fontWeight": "700", "color": accent,
                                     "fontSize": "0.75rem", "letterSpacing": "0.07em",
                                     "marginRight": "10px"}),
                    html.Span(info["detail"],
                              style={"color": C["muted"], "fontSize": "0.73rem"}),
                ], className="d-flex align-items-center flex-wrap"), md=5),

                # KPI chips
                dbc.Col(kpi_chips, md=3),

                # quick-jump buttons
                dbc.Col(html.Div([
                    html.Span("Jump to: ",
                              style={"color": C["muted"], "fontSize": "0.68rem",
                                     "marginRight": "4px", "whiteSpace": "nowrap"}),
                    *quick_btns,
                ], className="d-flex align-items-center flex-wrap"), md=3),

                # switch profile
                dbc.Col(
                    dcc.Link(
                        dbc.Button("⇄ Switch Profile", size="sm", color="link",
                                   style={"fontSize": "0.68rem", "color": C["muted"],
                                          "padding": "2px 6px"}),
                        href="/",
                    ),
                    md=1, className="d-flex align-items-center justify-content-end",
                ),
            ], align="center", className="g-1"),
        ], fluid=True),
        style={
            "backgroundColor": C["hover"],
            "borderLeft":      f"3px solid {accent}",
            "borderBottom":    f"1px solid {C['border']}",
            "padding":         "8px 0",
            "marginBottom":    "10px",
        },
    )


_DASHBOARD_LAYOUT = html.Div(
    style={"backgroundColor": C["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', 'Segoe UI', Arial, sans-serif"},
    children=[

        # ── role banner (populated by callback from user-role store) ────────
        html.Div(id="role-banner"),

        # ── header ────────────────────────────────────────────────────────────
        html.Div(
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("🏠 ", style={"fontSize": "1.5rem"}),
                            html.Span("ACCRA HOME PRICE INDEX",
                                      style={"fontSize": "1.25rem", "fontWeight": "700",
                                             "color": C["gold"], "letterSpacing": "0.05em"}),
                        ], className="d-flex align-items-center"),
                        html.Div("Monthly dataset · Jan 2010 – Dec 2024  ·  Mid-Market (5 districts)  ·  Prime Areas (6 locations)  ·  Prophet-ready",
                                 style={"fontSize": "0.75rem", "color": C["muted"],
                                        "marginTop": "2px"}),
                    ], md=7),
                    dbc.Col([
                        html.Div([
                            html.Span("USD/sqm benchmarks → GHS conversion at BoG rates → normalised 2015 = 100 → monthly interpolation",
                                      style={"color": C["muted"], "fontSize": "0.72rem"}),
                        ]),
                    ], md=4, className="d-flex align-items-center"),
                    dbc.Col([
                        dbc.Button(
                            "ⓘ  Methodology",
                            id="methodology-btn",
                            size="sm", outline=True, color="warning",
                            style={"fontSize": "0.72rem", "padding": "3px 10px"},
                        ),
                    ], md=1, className="d-flex align-items-center justify-content-end"),
                ], align="center"),
            ], fluid=True),
            style={"backgroundColor": C["card"], "borderBottom": f"1px solid {C['border']}",
                   "padding": "12px 0", "marginBottom": "16px"},
        ),

        dbc.Container([

            # ── KPI row ───────────────────────────────────────────────────────
            dbc.Row([
                kpi_card("AHPI",             "kpi-ahpi",       "📈", C["gold"]),
                kpi_card("GHS / sqm",        "kpi-ghs",        "🏘", C["green"]),
                kpi_card("USD / sqm",        "kpi-usd",        "💵", C["blue"]),
                kpi_card("GHS / USD",        "kpi-fx",         "💱", C["orange"]),
                kpi_card("Inflation",        "kpi-infl",       "📊", C["red"]),
                kpi_card("Gold (USD/oz)",    "kpi-gold",       "🥇", C["gold"]),
            ], className="mb-3 g-2"),

            # ── date range slider ─────────────────────────────────────────────
            range_slider,

            # ── tabs ──────────────────────────────────────────────────────────
            dbc.Tabs([
                dbc.Tab(tab_overview,    label="Overview",            tab_id="tab-overview",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_macro,       label="Macro Drivers",       tab_id="tab-macro",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_commodities, label="Commodities",         tab_id="tab-commodities",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_explorer,    label="Regressor Explorer",  tab_id="tab-explorer",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_districts,  label="Districts",           tab_id="tab-districts",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_prime,      label="Prime Areas",         tab_id="tab-prime",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_map,        label="Map",                 tab_id="tab-map",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_forecast,       label="Forecast",            tab_id="tab-forecast",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_prime_forecast,    label="Prime Forecast",    tab_id="tab-prime-forecast",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_district_forecast, label="District Forecast", tab_id="tab-district-forecast",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_invest,   label="Investment Return", tab_id="tab-invest",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_mortgage, label="Mortgage Stress Test", tab_id="tab-mortgage",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_report,   label="Market Report PDF", tab_id="tab-report",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
                dbc.Tab(tab_snapshot, label="Snapshot Card",    tab_id="tab-snapshot",
                        label_style={"color": C["muted"], "fontSize": "0.85rem"},
                        active_label_style={"color": C["gold"], "fontWeight": "600"}),
            ], id="main-tabs", active_tab="tab-overview",
               style={"borderBottom": f"1px solid {C['border']}"},
               className="mb-3"),

        ], fluid=True),

        # ── methodology modal ──────────────────────────────────────────────────
        _METHODOLOGY_MODAL,

        # ── KPI tooltips ───────────────────────────────────────────────────────
        dbc.Tooltip("Accra Home Price Index · Base year 2015 = 100. An AHPI of 400 means property values are 4× their Jan 2015 level in GHS.",
                    target="kpi-ahpi-label", placement="bottom"),
        dbc.Tooltip("Average price in Ghanaian Cedis per square metre, across all tracked areas in the selected market segment.",
                    target="kpi-ghs-label", placement="bottom"),
        dbc.Tooltip("Average price in US Dollars per square metre. Useful for comparing across currencies.",
                    target="kpi-usd-label", placement="bottom"),
        dbc.Tooltip("GHS/USD exchange rate — how many cedis buy one dollar. Higher = weaker cedi. This is the single strongest driver of Accra property prices.",
                    target="kpi-fx-label", placement="bottom"),
        dbc.Tooltip("Annual Consumer Price Index (CPI) inflation rate in Ghana. High inflation nominally inflates GHS property values while eroding real purchasing power.",
                    target="kpi-infl-label", placement="bottom"),
        dbc.Tooltip("London Bullion Market gold spot price in USD per troy ounce. Ghana's gold exports support fiscal revenues, diaspora confidence, and cedi stability.",
                    target="kpi-gold-label", placement="bottom"),

        # ── footer ─────────────────────────────────────────────────────────────
        html.Div(
            dbc.Container(
                html.Small(
                    "Data sources: World Bank Open Data · Bank of Ghana · LBMA (gold) · "
                    "ICCO (cocoa) · EIA/Platts (oil) · Global Property Guide · Numbeo · "
                    "JLL Africa · Knight Frank Africa",
                    style={"color": C["muted"]},
                ), fluid=True,
            ),
            style={"borderTop": f"1px solid {C['border']}", "padding": "10px 0",
                   "marginTop": "24px", "textAlign": "center"},
        ),

        # ── hidden stores ───────────────────────────────────────────────────
        dcc.Store(id="gis-resize-store"),
        dcc.Store(id="gis-anim-playing", data=False),
    ],
)

# ── app.layout: routing container ─────────────────────────────────────────────
# dcc.Location + dcc.Store are always in the DOM; page-content receives either
# _make_landing() or _DASHBOARD_LAYOUT from the render_page callback.
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="user-role", storage_type="session"),
    html.Div(id="page-content"),
])


# ── callbacks ─────────────────────────────────────────────────────────────────

# ── Page routing ──────────────────────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname: str):
    """Render either the landing page or the full dashboard."""
    if pathname in (None, "/", ""):
        return _make_landing()
    return _DASHBOARD_LAYOUT


# ── Role selection (landing page cards) ───────────────────────────────────────
@app.callback(
    Output("user-role", "data"),
    Output("url",       "pathname"),
    Input({"type": "role-btn", "role": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_role(n_clicks_list):
    """Store chosen role and navigate to the dashboard."""
    ctx = dash.callback_context
    if not ctx.triggered or all((n or 0) == 0 for n in n_clicks_list):
        return dash.no_update, dash.no_update
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    role_key = json.loads(triggered_id)["role"]
    return role_key, "/dashboard"


# ── Role banner (dashboard) ────────────────────────────────────────────────────
@app.callback(
    Output("role-banner", "children"),
    Input("user-role", "data"),
    prevent_initial_call=False,
)
def update_role_banner(role):
    return _make_role_banner(role)


# ── Tab navigation: set active tab on role load + banner quick-jump buttons ───
@app.callback(
    Output("main-tabs", "active_tab"),
    Input("user-role",                       "data"),
    Input({"type": "banner-tab-btn", "tab": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_active_tab(role, _tab_btn_clicks):
    """Jump to the role's default tab on login; respect banner quick-jump clicks."""
    triggered = dash.callback_context.triggered[0]["prop_id"]
    if "banner-tab-btn" in triggered:
        tab_id = json.loads(triggered.split(".")[0])["tab"]
        return tab_id
    if role and role in ROLES:
        return ROLES[role]["tab"]
    return "tab-overview"


# ── Animation: play / pause toggle ───────────────────────────────────────────
# Flips the playing state, enables/disables the interval, and updates the button label.
app.clientside_callback(
    """
    function(n_clicks, playing) {
        var now_playing = !playing;
        return [now_playing, !now_playing, now_playing ? '\u23f8' : '\u25b6'];
    }
    """,
    Output("gis-anim-playing",  "data"),
    Output("gis-anim-interval", "disabled"),
    Output("gis-anim-btn",      "children"),
    Input("gis-anim-btn",  "n_clicks"),
    State("gis-anim-playing", "data"),
    prevent_initial_call=True,
)

# ── Animation: advance one frame per interval tick ───────────────────────────
app.clientside_callback(
    """
    function(n_intervals, year) {
        return year >= 2029 ? 2010 : year + 1;
    }
    """,
    Output("gis-anim-year", "value"),
    Input("gis-anim-interval", "n_intervals"),
    State("gis-anim-year",     "value"),
    prevent_initial_call=True,
)

# ── Animation: year badge (HISTORICAL / PROJECTED) ───────────────────────────
app.clientside_callback(
    """
    function(year) {
        if (year >= 2025) {
            return year + ' \u2014 PROJECTED';
        }
        return year + ' \u2014 HISTORICAL';
    }
    """,
    Output("gis-anim-badge", "children"),
    Input("gis-anim-year", "value"),
)


# Dispatch a window resize event whenever the Map tab becomes active.
# Leaflet listens for this and calls map.invalidateSize() internally,
# which fills any blank tile areas caused by rendering inside a hidden tab.
app.clientside_callback(
    """
    function(active_tab) {
        if (active_tab === 'tab-map') {
            setTimeout(function() {
                window.dispatchEvent(new Event('resize'));
            }, 150);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("gis-resize-store", "data"),
    Input("main-tabs", "active_tab"),
    prevent_initial_call=True,
)


@app.callback(
    Output("range-label", "children"),
    Input("year-range", "value"),
    prevent_initial_call=False,
)
def update_range_label(yr_range):
    return f"{yr_range[0]}  →  {yr_range[1]}"


@app.callback(
    Output("kpi-ahpi",      "children"),
    Output("kpi-ahpi-sub",  "children"),
    Output("kpi-ghs",       "children"),
    Output("kpi-ghs-sub",   "children"),
    Output("kpi-usd",       "children"),
    Output("kpi-usd-sub",   "children"),
    Output("kpi-fx",        "children"),
    Output("kpi-fx-sub",    "children"),
    Output("kpi-infl",      "children"),
    Output("kpi-infl-sub",  "children"),
    Output("kpi-gold",      "children"),
    Output("kpi-gold-sub",  "children"),
    Input("year-range",       "value"),
    Input("overview-segment", "value"),
    prevent_initial_call=False,
)
def update_kpis(yr_range, segment):
    # For the AHPI / price KPIs, switch to prime aggregate when prime is selected.
    # Macro KPIs (FX, inflation, gold) are national data — same for all segments.
    if segment == "prime":
        price_src = DF_PRIME_FULL[
            (DF_PRIME_FULL["ds"].dt.year >= yr_range[0]) &
            (DF_PRIME_FULL["ds"].dt.year <= yr_range[1])
        ].copy()
    else:
        price_src = filter_df(*yr_range)

    macro_src = filter_df(*yr_range)

    latest_p = price_src.iloc[-1]
    prev_p   = price_src.iloc[-13] if len(price_src) > 13 else price_src.iloc[0]
    latest_m = macro_src.iloc[-1]
    prev_m   = macro_src.iloc[-13] if len(macro_src) > 13 else macro_src.iloc[0]

    def yoy(latest, prev, col):
        chg = latest[col] - prev[col]
        pct = (chg / abs(prev[col]) * 100) if prev[col] != 0 else 0
        arrow = "▲" if chg >= 0 else "▼"
        color = C["green"] if chg >= 0 else C["red"]
        return html.Span(f"{arrow} {abs(pct):.1f}% YoY", style={"color": color})

    return (
        f"{latest_p['y']:.1f}",                           yoy(latest_p, prev_p, "y"),
        f"GHS {latest_p['price_ghs_per_sqm']:,.0f}",      yoy(latest_p, prev_p, "price_ghs_per_sqm"),
        f"${latest_p['price_usd_per_sqm']:,.0f}",         yoy(latest_p, prev_p, "price_usd_per_sqm"),
        f"{latest_m['exchange_rate_ghs_usd']:.2f}",       yoy(latest_m, prev_m, "exchange_rate_ghs_usd"),
        f"{latest_m['inflation_cpi_pct']:.1f}%",          yoy(latest_m, prev_m, "inflation_cpi_pct"),
        f"${latest_m['gold_price_usd']:,.0f}",            yoy(latest_m, prev_m, "gold_price_usd"),
    )


@app.callback(
    Output("ahpi-chart",   "figure"),
    Output("overlay-note", "style"),
    Input("year-range",        "value"),
    Input("ahpi-overlays",     "value"),
    Input("ahpi-events",       "value"),
    Input("overview-segment",  "value"),
    prevent_initial_call=False,
)
def update_ahpi(yr_range, overlays, events, segment):
    dff = filter_df(*yr_range)
    dff_prime_full = DF_PRIME_FULL[
        (DF_PRIME_FULL["ds"].dt.year >= yr_range[0]) &
        (DF_PRIME_FULL["ds"].dt.year <= yr_range[1])
    ].copy()
    note_style = ({"color": C["muted"], "fontSize": "0.75rem", "fontStyle": "italic"}
                  if segment == "both"
                  else {"display": "none"})
    return (
        build_ahpi_fig(dff, overlays or [], bool(events),
                       segment=segment or "mid",
                       dff_prime_full=dff_prime_full),
        note_style,
    )


@app.callback(
    Output("macro-chart", "figure"),
    Output("macro-grid",  "figure"),
    Input("year-range",       "value"),
    Input("macro-vars",       "value"),
    Input("macro-normalise",  "value"),
    prevent_initial_call=False,
)
def update_macro(yr_range, selected_vars, normalise):
    dff = filter_df(*yr_range)
    return (
        build_macro_fig(dff, selected_vars or [], normalise),
        build_macro_grid(dff),
    )


@app.callback(
    Output("commodity-chart", "figure"),
    Input("year-range", "value"),
    prevent_initial_call=False,
)
def update_commodities(yr_range):
    return build_commodity_fig(filter_df(*yr_range))


@app.callback(
    Output("scatter-chart", "figure"),
    Output("scatter-stats", "children"),
    Input("year-range", "value"),
    Input("scatter-x",  "value"),
    Input("scatter-y",  "value"),
    prevent_initial_call=False,
)
def update_scatter(yr_range, x_var, y_var):
    dff   = filter_df(*yr_range)
    fig   = build_scatter_fig(dff, x_var, y_var)
    x_arr = dff[x_var].values.astype(float)
    y_arr = dff[y_var].values.astype(float)
    slope, intercept, r2 = linreg(x_arr, y_arr)
    if r2 is not None:
        direction = "positive" if slope > 0 else "negative"
        strength  = ("strong" if abs(r2) > 0.7 else
                     "moderate" if abs(r2) > 0.4 else "weak")
        stats_div = html.Div([
            html.Div(f"R² = {r2:.3f}", style={"color": C["gold"], "fontWeight": "600",
                                               "fontSize": "1rem"}),
            html.Div(f"Slope: {slope:.4f}", style={"marginTop": "4px"}),
            html.Div(f"Intercept: {intercept:.2f}"),
            html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
            html.Div(f"{strength.title()} {direction} correlation",
                     style={"color": C["teal"]}),
            html.Div(f"n = {(~np.isnan(x_arr) & ~np.isnan(y_arr)).sum()} months",
                     style={"color": C["muted"], "marginTop": "4px", "fontSize": "0.75rem"}),
        ])
    else:
        stats_div = html.Div("Insufficient data", style={"color": C["muted"]})
    return fig, stats_div


@app.callback(
    Output("heatmap-chart", "figure"),
    Input("year-range", "value"),
    prevent_initial_call=False,
)
def update_heatmap(yr_range):
    return build_heatmap_fig(filter_df(*yr_range))


@app.callback(
    Output("district-chart",       "figure"),
    Output("district-price-table", "children"),
    Input("year-range",         "value"),
    Input("district-selector",  "value"),
    Input("district-events",    "value"),
    prevent_initial_call=False,
)
def update_district(yr_range, district, show_events):
    dff = filter_df_district(*yr_range)
    if district == "all":
        fig = build_district_comparison_fig(dff, show_events=bool(show_events))
    else:
        dff_single = dff[dff["district"] == district]
        fig = build_district_single_fig(dff_single, district)
    return fig, build_district_price_table(yr_range)


@app.callback(
    Output("prime-chart",       "figure"),
    Output("prime-price-table", "children"),
    Input("year-range",      "value"),
    Input("prime-selector",  "value"),
    Input("prime-events",    "value"),
    prevent_initial_call=False,
)
def update_prime(yr_range, area, show_events):
    dff = filter_df_prime(*yr_range)
    if area == "all":
        fig = build_prime_comparison_fig(dff, show_events=bool(show_events))
    else:
        dff_single = dff[dff["district"] == area]
        fig = build_prime_single_fig(dff_single, area)
    return fig, build_prime_price_table(yr_range)


@app.callback(
    Output("map-chart", "figure"),
    Input("map-segment", "value"),
    prevent_initial_call=False,
)
def update_map(segment):
    return build_map_fig(segment or "both")


@app.callback(
    Output("forecast-chart", "figure"),
    Input("forecast-ci", "value"),
    prevent_initial_call=False,
)
def update_forecast(show_ci):
    return build_forecast_fig(show_ci=bool(show_ci))


# ── methodology modal toggle ──────────────────────────────────────────────────
@app.callback(
    Output("methodology-modal", "is_open"),
    Input("methodology-btn",        "n_clicks"),
    Input("methodology-modal-close","n_clicks"),
    State("methodology-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_methodology_modal(open_clicks, close_clicks, is_open):
    return not is_open


# ── download: Overview ────────────────────────────────────────────────────────
@app.callback(
    Output("dl-overview", "data"),
    Input("dl-overview-btn", "n_clicks"),
    State("year-range",       "value"),
    State("overview-segment", "value"),
    prevent_initial_call=True,
)
def dl_overview(_, yr_range, segment):
    if segment == "prime":
        dff = filter_df_prime(yr_range[0], yr_range[1])
        cols = ["ds", "district", "y", "price_ghs_per_sqm", "price_usd_per_sqm"]
    else:
        dff  = filter_df(yr_range[0], yr_range[1])
        cols = ["ds", "y", "price_ghs_per_sqm", "price_usd_per_sqm",
                "exchange_rate_ghs_usd", "inflation_cpi_pct"]
    return dcc.send_data_frame(dff[cols].to_csv, "ahpi_overview.csv", index=False)


# ── download: Macro ───────────────────────────────────────────────────────────
@app.callback(
    Output("dl-macro", "data"),
    Input("dl-macro-btn", "n_clicks"),
    State("year-range",  "value"),
    State("macro-vars",  "value"),
    prevent_initial_call=True,
)
def dl_macro(_, yr_range, macro_vars):
    dff  = filter_df(yr_range[0], yr_range[1])
    cols = ["ds"] + [v for v in (macro_vars or []) if v in dff.columns]
    return dcc.send_data_frame(dff[cols].to_csv, "ahpi_macro.csv", index=False)


# ── download: Commodities ─────────────────────────────────────────────────────
@app.callback(
    Output("dl-commodities", "data"),
    Input("dl-commodities-btn", "n_clicks"),
    State("year-range", "value"),
    prevent_initial_call=True,
)
def dl_commodities(_, yr_range):
    dff  = filter_df(yr_range[0], yr_range[1])
    cols = ["ds", "gold_price_usd", "oil_brent_usd", "cocoa_price_usd"]
    return dcc.send_data_frame(dff[cols].to_csv, "ahpi_commodities.csv", index=False)


# ── download: Districts ───────────────────────────────────────────────────────
@app.callback(
    Output("dl-districts", "data"),
    Input("dl-districts-btn",  "n_clicks"),
    State("year-range",         "value"),
    State("district-selector",  "value"),
    prevent_initial_call=True,
)
def dl_districts(_, yr_range, district):
    dff  = filter_df_district(yr_range[0], yr_range[1], district or "all")
    cols = ["ds", "district", "y", "price_ghs_per_sqm", "price_usd_per_sqm"]
    return dcc.send_data_frame(dff[cols].to_csv, "ahpi_districts.csv", index=False)


# ── download: Prime Areas ─────────────────────────────────────────────────────
@app.callback(
    Output("dl-prime", "data"),
    Input("dl-prime-btn",    "n_clicks"),
    State("year-range",       "value"),
    State("prime-selector",   "value"),
    prevent_initial_call=True,
)
def dl_prime(_, yr_range, area):
    dff  = filter_df_prime(yr_range[0], yr_range[1], area or "all")
    cols = ["ds", "district", "y", "price_ghs_per_sqm", "price_usd_per_sqm"]
    return dcc.send_data_frame(dff[cols].to_csv, "ahpi_prime_areas.csv", index=False)


# ── download: Forecast ZIP (mid-market) ───────────────────────────────────────
@app.callback(
    Output("dl-forecast", "data"),
    Input("dl-forecast-btn", "n_clicks"),
    prevent_initial_call=True,
)
def dl_forecast(_):
    files = {
        "ahpi_test_eval.csv":        DF_TEST_EVAL,
        "ahpi_forecast_bear.csv":    DF_FC_BEAR,
        "ahpi_forecast_base.csv":    DF_FC_BASE,
        "ahpi_forecast_bull.csv":    DF_FC_BULL,
    }
    return dcc.send_bytes(_zip_dfs(files), "ahpi_midmarket_forecasts.zip")


# ── download: Prime Forecast ZIP ──────────────────────────────────────────────
@app.callback(
    Output("dl-prime-forecast", "data"),
    Input("dl-prime-forecast-btn", "n_clicks"),
    State("prime-fc-area", "value"),
    prevent_initial_call=True,
)
def dl_prime_forecast(_, area):
    area = area or "all"
    if area == "all":
        files = {f"prime_test_eval_{slug}.csv": _PRIME_TEST_EVALS[a]
                 for a, slug in PRIME_AREA_SLUGS.items()}
        for sc in ("bear", "base", "bull"):
            for a, slug in PRIME_AREA_SLUGS.items():
                files[f"prime_forecast_{sc}_{slug}.csv"] = _PRIME_FC[(sc, a)]
        files["prime_test_summary.csv"] = _PRIME_TEST_SUMMARY
    else:
        slug  = PRIME_AREA_SLUGS[area]
        files = {f"prime_test_eval_{slug}.csv": _PRIME_TEST_EVALS[area]}
        for sc in ("bear", "base", "bull"):
            files[f"prime_forecast_{sc}_{slug}.csv"] = _PRIME_FC[(sc, area)]
    label = "all_areas" if area == "all" else PRIME_AREA_SLUGS[area]
    return dcc.send_bytes(_zip_dfs(files), f"ahpi_prime_forecast_{label}.zip")


# ── download: District Forecast ZIP ──────────────────────────────────────────
@app.callback(
    Output("dl-district-forecast", "data"),
    Input("dl-district-forecast-btn", "n_clicks"),
    State("district-fc-area", "value"),
    prevent_initial_call=True,
)
def dl_district_forecast(_, district):
    district = district or "all"
    if district == "all":
        files = {f"district_test_eval_{slug}.csv": _DISTRICT_TEST_EVALS[d]
                 for d, slug in DISTRICT_SLUGS.items()}
        for sc in ("bear", "base", "bull"):
            for d, slug in DISTRICT_SLUGS.items():
                files[f"district_forecast_{sc}_{slug}.csv"] = _DISTRICT_FC[(sc, d)]
        files["district_test_summary.csv"] = _DISTRICT_TEST_SUMMARY
    else:
        slug  = DISTRICT_SLUGS[district]
        files = {f"district_test_eval_{slug}.csv": _DISTRICT_TEST_EVALS[district]}
        for sc in ("bear", "base", "bull"):
            files[f"district_forecast_{sc}_{slug}.csv"] = _DISTRICT_FC[(sc, district)]
    label = "all_districts" if district == "all" else DISTRICT_SLUGS[district]
    return dcc.send_bytes(_zip_dfs(files), f"ahpi_district_forecast_{label}.zip")


# ── forecast-tab year-selector callbacks ──────────────────────────────────────
@app.callback(
    Output("forecast-targets",         "children"),
    Output("forecast-targets-heading", "children"),
    Input("forecast-year", "value"),
    prevent_initial_call=False,
)
def update_forecast_targets(year):
    year = year or 2026
    return _build_fc_targets_div(year), f"Dec {year} AHPI targets by scenario"


@app.callback(
    Output("prime-forecast-chart",     "figure"),
    Output("prime-fc-metrics",         "children"),
    Output("prime-fc-targets",         "children"),
    Output("prime-fc-targets-heading", "children"),
    Input("prime-fc-ci",   "value"),
    Input("prime-fc-area", "value"),
    Input("prime-fc-year", "value"),
    prevent_initial_call=False,
)
def update_prime_forecast(show_ci, area, year):
    area = area or "all"
    year = year or 2026
    return (
        build_prime_forecast_fig(area, show_ci=bool(show_ci)),
        _build_prime_metrics_div(area),
        _build_prime_targets_div(area, year),
        f"Dec {year} AHPI targets by scenario",
    )


@app.callback(
    Output("district-forecast-chart",     "figure"),
    Output("district-fc-metrics",         "children"),
    Output("district-fc-targets",         "children"),
    Output("district-fc-targets-heading", "children"),
    Input("district-fc-ci",   "value"),
    Input("district-fc-area", "value"),
    Input("district-fc-year", "value"),
    prevent_initial_call=False,
)
def update_district_forecast(show_ci, district, year):
    district = district or "all"
    year     = year or 2026
    return (
        build_district_forecast_fig(district, show_ci=bool(show_ci)),
        _build_district_metrics_div(district),
        _build_district_targets_div(district, year),
        f"Dec {year} AHPI targets by scenario",
    )


# ── Investment Return Calculator callbacks ─────────────────────────────────────
@app.callback(
    Output("inv-buy-price-display", "children"),
    Output("inv-buy-usd-display",   "children"),
    Output("inv-buy-fx-display",    "children"),
    Input("inv-market",   "value"),
    Input("inv-buy-year", "value"),
    prevent_initial_call=False,
)
def update_inv_buy_info(market, buy_year):
    row = _get_hist_dec(market or "composite", buy_year or 2020)
    if row is None:
        return "—", "—", "—"
    return (
        f"GHS {row.get('price_ghs_per_sqm', 0):,.0f}",
        f"USD {row.get('price_usd_per_sqm', 0):,.0f}",
        f"{row.get('exchange_rate_ghs_usd', 0):.2f}",
    )


@app.callback(
    Output("inv-results", "children"),
    Input("inv-market",    "value"),
    Input("inv-buy-year",  "value"),
    Input("inv-sell-year", "value"),
    Input("inv-sqm",       "value"),
    prevent_initial_call=False,
)
def update_inv_results(market, buy_year, sell_year, sqm):
    market   = market   or "composite"
    buy_year = buy_year or 2020
    sell_year = sell_year or 2027
    sqm      = sqm or 100
    years    = sell_year - buy_year
    if years <= 0:
        return html.Div("Sell year must be after buy year.",
                        style={"color": C["red"], "padding": "20px"})

    buy_row = _get_hist_dec(market, buy_year)
    if buy_row is None:
        return html.Div("No historical data for this market / buy year.",
                        style={"color": C["muted"], "padding": "20px"})

    buy_ghs = float(buy_row.get("price_ghs_per_sqm", 0))
    buy_usd = float(buy_row.get("price_usd_per_sqm", 0))
    buy_fx  = float(buy_row.get("exchange_rate_ghs_usd", 1))
    buy_ahpi = float(buy_row.get("y", 100))

    cols = []
    for sc in ("bear", "base", "bull"):
        color, _, label = SCENARIO_STYLES[sc]
        fc_row = _get_fc_dec(market, sc, sell_year)
        if fc_row is None:
            cols.append(_scenario_col(sc, label, color,
                                      [html.Div("No forecast data", style={"color": C["muted"]})]))
            continue

        sell_ahpi = float(fc_row["yhat"])
        sell_ahpi_lo = float(fc_row["yhat_lower"])
        sell_ahpi_hi = float(fc_row["yhat_upper"])
        sell_fx   = SCENARIO_FX[sc]

        # Property values
        sell_ghs  = buy_ghs * (sell_ahpi / buy_ahpi) if buy_ahpi else 0
        sell_usd  = sell_ghs / sell_fx if sell_fx else 0

        total_buy_ghs  = buy_ghs  * sqm
        total_sell_ghs = sell_ghs * sqm
        total_buy_usd  = buy_usd  * sqm
        total_sell_usd = sell_usd * sqm

        ghs_ret  = (total_sell_ghs - total_buy_ghs) / total_buy_ghs * 100 if total_buy_ghs else 0
        usd_ret  = (total_sell_usd - total_buy_usd) / total_buy_usd * 100 if total_buy_usd else 0
        ghs_cagr = ((total_sell_ghs / total_buy_ghs) ** (1 / years) - 1) * 100 if total_buy_ghs else 0
        usd_cagr = ((total_sell_usd / total_buy_usd) ** (1 / years) - 1) * 100 if total_buy_usd else 0

        # Benchmarks
        tbill_rate  = SCENARIO_TBILL[sc] / 100
        tbill_final = total_buy_ghs * (1 + tbill_rate) ** years
        tbill_ret   = (tbill_final - total_buy_ghs) / total_buy_ghs * 100

        usd_dep_final = total_buy_usd * (1 + USD_DEPOSIT_RATE / 100) ** years
        usd_dep_ret   = (usd_dep_final - total_buy_usd) / total_buy_usd * 100

        # Colour helpers
        ghs_col = C["green"] if ghs_ret >= 0 else C["red"]
        usd_col = C["green"] if usd_ret >= 0 else C["red"]
        vs_tbill = "outperforms" if ghs_ret > tbill_ret else "underperforms"
        vs_usd   = "outperforms" if usd_ret > usd_dep_ret else "underperforms"

        cols.append(_scenario_col(sc, label, color, [
            _result_card("AHPI at sell",
                         f"{sell_ahpi:.1f}  [{sell_ahpi_lo:.1f}–{sell_ahpi_hi:.1f}]",
                         f"90% CI · was {buy_ahpi:.1f} at purchase", color),
            _result_card("Sell price (GHS/sqm)",
                         f"GHS {sell_ghs:,.0f}",
                         f"was GHS {buy_ghs:,.0f}", color),
            _result_card("Sell price (USD/sqm)",
                         f"USD {sell_usd:,.0f}",
                         f"was USD {buy_usd:,.0f}  ·  FX: {sell_fx} GHS/USD", C["blue"]),
            _result_card(f"GHS return  ({sqm} sqm · {years}y)",
                         f"{ghs_ret:+.1f}%",
                         f"CAGR {ghs_cagr:+.1f}% p.a.", ghs_col),
            _result_card(f"USD return  ({sqm} sqm · {years}y)",
                         f"{usd_ret:+.1f}%",
                         f"CAGR {usd_cagr:+.1f}% p.a.", usd_col),
            html.Div([
                html.Div("vs Benchmarks", style={"fontSize": "0.7rem", "color": C["muted"],
                                                   "textTransform": "uppercase",
                                                   "marginBottom": "4px"}),
                html.Div(f"🇬🇭 T-Bill ({SCENARIO_TBILL[sc]}% p.a.): {tbill_ret:+.1f}% total  "
                         f"→ property {vs_tbill}",
                         style={"fontSize": "0.75rem", "color": C["muted"]}),
                html.Div(f"💵 USD deposit ({USD_DEPOSIT_RATE}% p.a.): {usd_dep_ret:+.1f}% total  "
                         f"→ {vs_usd} in USD terms",
                         style={"fontSize": "0.75rem", "color": C["muted"]}),
            ], style={"backgroundColor": C["hover"], "padding": "8px 12px",
                      "borderRadius": "6px", "border": f"1px solid {C['border']}"}),
        ]))

    return dbc.Row([c for c in cols], className="g-2")


# ── Mortgage Stress Test callbacks ─────────────────────────────────────────────
@app.callback(
    Output("mort-value", "value"),
    Input("mort-district", "value"),
    Input("mort-sqm",      "value"),
    prevent_initial_call=False,
)
def prefill_mort_value(district, sqm):
    """Auto-fill property value from latest AHPI data."""
    sqm = sqm or 100
    district = district or DISTRICTS[0]
    row = _get_hist_dec(district, 2024)
    if row is None:
        return sqm * 3000
    return round(float(row.get("price_ghs_per_sqm", 3000)) * sqm, -3)


@app.callback(
    Output("mort-results", "children"),
    Input("mort-district", "value"),
    Input("mort-sqm",      "value"),
    Input("mort-value",    "value"),
    Input("mort-ltv",      "value"),
    Input("mort-term",     "value"),
    Input("mort-rate",     "value"),
    Input("mort-year",     "value"),
    prevent_initial_call=False,
)
def update_mort_results(district, sqm, prop_value, ltv, term, rate, check_year):
    district   = district   or DISTRICTS[0]
    sqm        = sqm        or 100
    prop_value = prop_value or 300_000
    ltv        = ltv        or 70
    term       = term       or 20
    rate       = rate       or 28.0
    check_year = check_year or 2027

    loan_amount  = prop_value * ltv / 100
    monthly_rate = rate / 100 / 12
    n_payments   = term * 12

    if monthly_rate > 0:
        monthly_pmt = (loan_amount * monthly_rate * (1 + monthly_rate) ** n_payments
                       / ((1 + monthly_rate) ** n_payments - 1))
    else:
        monthly_pmt = loan_amount / n_payments

    total_paid    = monthly_pmt * n_payments
    total_int     = total_paid - loan_amount
    pct_income    = monthly_pmt / GHANA_MEDIAN_INCOME_GHS * 100

    if pct_income <= 30:
        afford_color, afford_label = C["green"], "Affordable  (≤ 30% of median income)"
    elif pct_income <= 50:
        afford_color, afford_label = C["orange"], "Stretched  (30–50% of median income)"
    else:
        afford_color, afford_label = C["red"], "Unaffordable  (> 50% of median income)"

    # Repayment section
    repay_section = dbc.Col([
        html.Div("Monthly Repayment", style={"fontSize": "0.75rem", "color": C["muted"],
                                              "fontWeight": "600", "marginBottom": "8px"}),
        _result_card("Monthly payment", f"GHS {monthly_pmt:,.0f}",
                     f"on a {term}-year loan at {rate}% p.a.", C["gold"]),
        _result_card("Loan amount",   f"GHS {loan_amount:,.0f}",
                     f"{ltv}% of GHS {prop_value:,.0f}"),
        _result_card("Total repaid",  f"GHS {total_paid:,.0f}",
                     f"interest: GHS {total_int:,.0f}"),
        html.Div([
            html.Div("Affordability",
                     style={"fontSize": "0.7rem", "color": C["muted"],
                            "textTransform": "uppercase", "marginBottom": "4px"}),
            html.Div(f"{pct_income:.1f}% of Ghana median household income (GHS {GHANA_MEDIAN_INCOME_GHS:,}/mo)",
                     style={"fontSize": "0.8rem", "color": afford_color, "fontWeight": "600"}),
            html.Div(afford_label, style={"fontSize": "0.75rem", "color": afford_color}),
        ], style={"backgroundColor": C["hover"], "padding": "10px 14px", "borderRadius": "6px",
                  "border": f"2px solid {afford_color}"}),
    ], md=4)

    # Collateral stress test (3 scenarios)
    stress_cols = []
    for sc in ("bear", "base", "bull"):
        color, _, label = SCENARIO_STYLES[sc]
        fc_row = _get_fc_dec(district, sc, check_year)
        if fc_row is None:
            stress_cols.append(_scenario_col(sc, label, color,
                                             [html.Div("No data", style={"color": C["muted"]})]))
            continue

        hist_row  = _get_hist_dec(district, 2024)
        ahpi_2024 = float(hist_row["y"]) if hist_row is not None else 100
        ghs_2024  = float(hist_row.get("price_ghs_per_sqm", prop_value / sqm)) if hist_row is not None else prop_value / sqm

        sell_ahpi = float(fc_row["yhat"])
        collateral = ghs_2024 * (sell_ahpi / ahpi_2024) * sqm
        ltv_ratio  = loan_amount / collateral * 100 if collateral else 0
        change_pct = (collateral - prop_value) / prop_value * 100

        ltv_col = C["green"] if ltv_ratio < 80 else (C["orange"] if ltv_ratio < 100 else C["red"])
        ltv_warn = "" if ltv_ratio < 80 else (" — WATCH" if ltv_ratio < 100 else " — UNDERWATER")

        stress_cols.append(_scenario_col(sc, f"{label} · Dec {check_year}", color, [
            _result_card("Collateral value", f"GHS {collateral:,.0f}",
                         f"{change_pct:+.1f}% vs today", color),
            _result_card("LTV at check year", f"{ltv_ratio:.1f}%",
                         f"Loan: GHS {loan_amount:,.0f}{ltv_warn}", ltv_col),
        ]))

    stress_section = dbc.Col([
        html.Div(f"Collateral Stress Test — Dec {check_year}",
                 style={"fontSize": "0.75rem", "color": C["muted"],
                        "fontWeight": "600", "marginBottom": "8px"}),
        dbc.Row(stress_cols, className="g-2"),
        html.Div(
            f"LTV > 80%: watch list. LTV > 100%: collateral below outstanding loan. "
            f"Median income proxy: GHS {GHANA_MEDIAN_INCOME_GHS:,}/month.",
            style={"fontSize": "0.72rem", "color": C["muted"],
                   "marginTop": "8px", "fontStyle": "italic"},
        ),
    ], md=8)

    return dbc.Row([repay_section, stress_section], className="g-2")


# ── GIS choropleth callbacks ──────────────────────────────────────────────────
@app.callback(
    Output("gis-geojson",    "data"),
    Output("gis-geojson",    "style"),
    Output("gis-geojson",    "onEachFeature"),
    Output("gis-legend",     "children"),
    Output("gis-tile-layer", "url"),
    Output("gis-tile-layer", "attribution"),
    Input("gis-layer",        "value"),
    Input("gis-price-metric", "value"),
    Input("gis-scenario",     "value"),
    Input("gis-fc-year",      "value"),
    Input("gis-tiles",        "value"),
    Input("gis-anim-year",    "value"),
    prevent_initial_call=False,
)
def update_gis_map(layer, price_metric, scenario, fc_year, tile_key, anim_year):
    tile_key  = tile_key or "dark"
    tile_url, tile_attr = _TILE_URLS.get(tile_key, _TILE_URLS["dark"])
    scenario  = scenario or "base"
    anim_year = int(anim_year or 2024)

    _METRIC_MAP = {
        "usd_sqm": ("usd_sqm", "USD / sqm", ""),
        "ghs_sqm": ("ghs_sqm", "GHS / sqm", ""),
        "ahpi":    ("ahpi",    "AHPI",       " pts"),
    }
    snap_key, metric_label, unit = _METRIC_MAP.get(price_metric or "usd_sqm",
                                                    ("usd_sqm", "USD/sqm", ""))

    if layer == "price":
        # Slider drives the year; 2025+ yields AHPI-only with a projected label
        gj, lo, hi = _build_timeline_geojson(anim_year, snap_key, scenario)
        style_fn   = _style_price
        each_fn    = _on_each_feature_timeline
        if anim_year >= 2025:
            legend = _legend_strip(_CS_GOLD_TO_RED, lo, hi,
                                   f"AHPI (proj · {scenario.title()}) · Dec {anim_year}", " pts")
        else:
            legend = _legend_strip(_CS_GOLD_TO_RED, lo, hi,
                                   f"{metric_label} · Dec {anim_year}", unit)
    else:
        # Forecast-growth layer: clamp slider to valid forecast range
        fc_y = anim_year if anim_year >= 2025 else int(fc_year or 2027)
        gj, lo, hi = _build_forecast_geojson(scenario, fc_y)
        style_fn   = _style_forecast
        each_fn    = _on_each_feature_forecast
        legend     = _legend_strip(
            _CS_BLUE_TO_GREEN, lo, hi,
            f"AHPI Growth vs Dec 2024 — {scenario.title()} {fc_y}", "%")

    return gj, style_fn, each_fn, legend, tile_url, tile_attr


@app.callback(
    Output("gis-dl", "data"),
    Input("gis-dl-btn",       "n_clicks"),
    State("gis-layer",        "value"),
    State("gis-price-metric", "value"),
    State("gis-scenario",     "value"),
    State("gis-fc-year",      "value"),
    prevent_initial_call=True,
)
def download_geojson(n_clicks, layer, price_metric, scenario, fc_year):
    fc_year  = int(fc_year or 2027)
    scenario = scenario or "base"
    _METRIC_MAP = {
        "usd_sqm": "usd_sqm",
        "ghs_sqm": "ghs_sqm",
        "ahpi":    "ahpi",
    }
    snap_key = _METRIC_MAP.get(price_metric or "usd_sqm", "usd_sqm")

    if layer == "price":
        gj, _, _ = _build_price_geojson(snap_key)
        fname    = f"accra_price_heatmap_{snap_key}.geojson"
    else:
        gj, _, _ = _build_forecast_geojson(scenario, fc_year)
        fname    = f"accra_forecast_growth_{scenario}_{fc_year}.geojson"

    return dcc.send_bytes(
        json.dumps(gj, indent=2).encode(),
        fname,
    )


# ── Market Report PDF callback ────────────────────────────────────────────────
@app.callback(
    Output("report-pdf-dl",      "data"),
    Output("report-preview",     "children"),
    Input("report-pdf-btn",      "n_clicks"),
    State("report-market",       "value"),
    State("report-year",         "value"),
    prevent_initial_call=True,
)
def generate_pdf_report(n_clicks, market, report_year):
    market      = market or "composite"
    report_year = int(report_year or 2027)
    label       = "Composite Mid-Market" if market == "composite" else market

    try:
        pdf_bytes = generate_market_pdf(market, report_year)
        filename  = f"AHPI_Report_{label.replace(' ', '_').replace('/', '-')}_{report_year}.pdf"
        preview   = html.Div([
            html.Div("Report generated successfully.", style={"color": C["green"],
                                                               "fontWeight": "600",
                                                               "marginBottom": "8px"}),
            html.Div(f"Market: {label}", style={"color": C["text"]}),
            html.Div(f"Forecast horizon: Dec {report_year}", style={"color": C["text"]}),
            html.Div(f"File: {filename}", style={"color": C["muted"], "fontSize": "0.75rem",
                                                  "marginTop": "6px"}),
        ])
        return dcc.send_bytes(pdf_bytes, filename), preview
    except Exception as exc:
        err = html.Div(f"Error generating PDF: {exc}",
                       style={"color": C["red"], "fontSize": "0.8rem"})
        return dash.no_update, err


# ── Snapshot card callback ─────────────────────────────────────────────────────
@app.callback(
    Output("snap-card-container", "children"),
    Input("snap-market",          "value"),
    Input("snap-year",            "value"),
    prevent_initial_call=False,
)
def update_snapshot_card(market, snap_year):
    market    = market or "composite"
    snap_year = int(snap_year or 2027)
    label     = "Composite Mid-Market" if market == "composite" else market
    family    = ("Mid-Market" if market in DISTRICTS else
                 ("Prime" if market in PRIME_AREAS else "Composite"))

    hist_2024 = _get_hist_dec(market, 2024)
    hist_2023 = _get_hist_dec(market, 2023)

    # Current AHPI
    if hist_2024 is not None:
        ahpi_now = float(hist_2024.get("y", 0))
        ghs_sqm  = hist_2024.get("price_ghs_per_sqm")
        usd_sqm  = hist_2024.get("price_usd_per_sqm")
        ahpi_prev = float(hist_2023.get("y", ahpi_now)) if hist_2023 is not None else ahpi_now
        yoy_pct  = (ahpi_now - ahpi_prev) / ahpi_prev * 100 if ahpi_prev else 0
        yoy_color = C["green"] if yoy_pct >= 0 else C["red"]
        yoy_str   = f"{yoy_pct:+.1f}% YoY"
    else:
        ahpi_now, ghs_sqm, usd_sqm, yoy_str, yoy_color = None, None, None, "—", C["muted"]

    # Forecast rows
    fc_rows = []
    for sc in ("bear", "base", "bull"):
        color, _, sc_label = SCENARIO_STYLES[sc]
        fc_row = _get_fc_dec(market, sc, snap_year)
        if fc_row is not None:
            yhat = float(fc_row["yhat"])
            lo   = float(fc_row["yhat_lower"])
            hi   = float(fc_row["yhat_upper"])
            chg  = (yhat - ahpi_now) / ahpi_now * 100 if ahpi_now else 0
            chg_c = C["green"] if chg >= 0 else C["red"]
            fc_rows.append(html.Tr([
                html.Td(sc_label.split()[0],
                        style={"color": color, "fontWeight": "700",
                               "padding": "5px 10px", "fontSize": "0.8rem"}),
                html.Td(f"{yhat:.1f}",
                        style={"color": C["text"], "padding": "5px 10px",
                               "fontWeight": "600", "fontSize": "0.95rem"}),
                html.Td(f"[{lo:.1f} – {hi:.1f}]",
                        style={"color": C["muted"], "padding": "5px 10px",
                               "fontSize": "0.75rem"}),
                html.Td(f"{chg:+.1f}%",
                        style={"color": chg_c, "padding": "5px 10px",
                               "fontSize": "0.8rem", "fontWeight": "600"}),
                html.Td(f"GHS/USD → {SCENARIO_FX[sc]:.0f}",
                        style={"color": C["muted"], "padding": "5px 10px",
                               "fontSize": "0.75rem"}),
            ], style={"borderBottom": f"1px solid {C['border']}"}))

    # Build the card
    stat_items = []
    if ahpi_now is not None:
        stat_items.append(
            dbc.Col(html.Div([
                html.Div("AHPI Dec 2024", style={"fontSize": "0.65rem", "color": C["muted"],
                                                  "textTransform": "uppercase"}),
                html.Div(f"{ahpi_now:.1f}",
                         style={"fontSize": "1.6rem", "fontWeight": "700", "color": C["gold"]}),
                html.Div(yoy_str, style={"fontSize": "0.75rem", "color": yoy_color}),
            ], style={"textAlign": "center"}), md=3)
        )
    if ghs_sqm is not None:
        stat_items.append(
            dbc.Col(html.Div([
                html.Div("GHS / sqm", style={"fontSize": "0.65rem", "color": C["muted"],
                                              "textTransform": "uppercase"}),
                html.Div(f"{float(ghs_sqm):,.0f}",
                         style={"fontSize": "1.4rem", "fontWeight": "700", "color": C["text"]}),
            ], style={"textAlign": "center"}), md=3)
        )
    if usd_sqm is not None:
        stat_items.append(
            dbc.Col(html.Div([
                html.Div("USD / sqm", style={"fontSize": "0.65rem", "color": C["muted"],
                                              "textTransform": "uppercase"}),
                html.Div(f"{float(usd_sqm):,.0f}",
                         style={"fontSize": "1.4rem", "fontWeight": "700", "color": C["text"]}),
            ], style={"textAlign": "center"}), md=3)
        )

    card = html.Div([
        # ── Card header ─────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("ACCRA HOME PRICE INDEX",
                          style={"fontSize": "0.65rem", "color": C["gold"],
                                 "letterSpacing": "0.12em", "fontWeight": "700"}),
                html.Div(label, style={"fontSize": "1.1rem", "fontWeight": "700",
                                       "color": C["text"]}),
                html.Div(f"{family} · Forecast: Dec {snap_year}  ·  "
                         f"Report date: {datetime.date.today().strftime('%d %b %Y')}",
                         style={"fontSize": "0.72rem", "color": C["muted"]}),
            ], style={"flex": "1"}),
            html.Div("MARKET BRIEFING", style={"fontSize": "0.75rem", "color": C["gold"],
                                               "fontWeight": "700",
                                               "border": f"1px solid {C['gold']}",
                                               "padding": "4px 10px", "borderRadius": "4px",
                                               "alignSelf": "center"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "borderBottom": f"2px solid {C['gold']}",
                  "paddingBottom": "8px", "marginBottom": "14px"}),

        # ── KPI row ─────────────────────────────────────────────────────────
        dbc.Row(stat_items, className="g-2 mb-3"),

        # ── Scenario forecast table ──────────────────────────────────────────
        html.Div(f"Scenario Forecasts — Dec {snap_year}",
                 style={"fontSize": "0.75rem", "color": C["muted"],
                        "textTransform": "uppercase", "letterSpacing": "0.06em",
                        "marginBottom": "6px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Scenario", style={"padding": "5px 10px", "fontSize": "0.75rem",
                                            "color": C["gold"], "borderBottom": f"1px solid {C['border']}"}),
                html.Th(f"AHPI Dec {snap_year}",
                        style={"padding": "5px 10px", "fontSize": "0.75rem", "color": C["gold"],
                               "borderBottom": f"1px solid {C['border']}"}),
                html.Th("90% CI", style={"padding": "5px 10px", "fontSize": "0.75rem",
                                          "color": C["gold"],
                                          "borderBottom": f"1px solid {C['border']}"}),
                html.Th("vs 2024", style={"padding": "5px 10px", "fontSize": "0.75rem",
                                           "color": C["gold"],
                                           "borderBottom": f"1px solid {C['border']}"}),
                html.Th("FX Assumption", style={"padding": "5px 10px", "fontSize": "0.75rem",
                                                 "color": C["gold"],
                                                 "borderBottom": f"1px solid {C['border']}"}),
            ])),
            html.Tbody(fc_rows),
        ], style={"width": "100%", "borderCollapse": "collapse",
                  "backgroundColor": C["card"], "borderRadius": "6px",
                  "marginBottom": "14px"}),

        # ── Footer disclaimer ────────────────────────────────────────────────
        html.Div(
            "For informational purposes only. Not financial advice. "
            "Forecasts generated by Facebook Prophet; 90% credible intervals shown. "
            "Source: AHPI Prophet v2.1 · Base year 2015 = 100.",
            style={"fontSize": "0.68rem", "color": C["muted"], "fontStyle": "italic",
                   "borderTop": f"1px solid {C['border']}", "paddingTop": "8px",
                   "textAlign": "center"},
        ),
    ], style={
        "backgroundColor": C["card"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "8px",
        "padding": "18px 20px",
        "fontFamily": "monospace",
    })

    return card


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8050
    print(f"\n  AHPI Dashboard → http://localhost:{port}\n")
    app.run(debug=False, port=port, host="0.0.0.0")
