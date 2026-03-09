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

import os
import sys
import warnings
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    ])


def _build_prime_targets_div(area: str) -> html.Div:
    rows = []
    for sc in ("bear", "base", "bull"):
        color, _, label = SCENARIO_STYLES[sc]
        fc_df = _PRIME_FC_AGG[sc] if area == "all" else _PRIME_FC[(sc, area)]
        dec26 = fc_df.iloc[-1]
        rows.append(html.Div([
            html.Span("● ", style={"color": color, "fontSize": "1rem"}),
            html.Span(f"{label}: ", style={"color": C["muted"], "fontSize": "0.8rem"}),
            html.Span(f"{dec26['yhat']:.1f}",
                      style={"color": color, "fontWeight": "700", "fontSize": "0.9rem"}),
            html.Span(f"  [{dec26['yhat_lower']:.1f} – {dec26['yhat_upper']:.1f}]",
                      style={"color": C["muted"], "fontSize": "0.78rem"}),
        ], className="mb-2"))
    return html.Div([
        *rows,
        html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
        html.Div(
            "Dec 2026 AHPI targets under each scenario. 90% credible interval in brackets.",
            style={"fontSize": "0.75rem", "color": C["muted"]},
        ),
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
    ])


def _build_district_targets_div(district: str) -> html.Div:
    rows = []
    for sc in ("bear", "base", "bull"):
        color, _, label = SCENARIO_STYLES[sc]
        fc_df = _DISTRICT_FC_AGG[sc] if district == "all" else _DISTRICT_FC[(sc, district)]
        dec26 = fc_df.iloc[-1]
        rows.append(html.Div([
            html.Span("● ", style={"color": color, "fontSize": "1rem"}),
            html.Span(f"{label}: ", style={"color": C["muted"], "fontSize": "0.8rem"}),
            html.Span(f"{dec26['yhat']:.1f}",
                      style={"color": color, "fontWeight": "700", "fontSize": "0.9rem"}),
            html.Span(f"  [{dec26['yhat_lower']:.1f} – {dec26['yhat_upper']:.1f}]",
                      style={"color": C["muted"], "fontSize": "0.78rem"}),
        ], className="mb-2"))
    return html.Div([
        *rows,
        html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
        html.Div(
            "Dec 2026 AHPI targets under each scenario. 90% credible interval in brackets.",
            style={"fontSize": "0.75rem", "color": C["muted"]},
        ),
    ])


# ── reusable layout pieces ────────────────────────────────────────────────────
def kpi_card(label, val_id, icon, color=C["gold"]):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Span(icon, style={"fontSize": "1.2rem", "marginRight": "6px"}),
                    html.Span(label,
                              style={"fontSize": "0.68rem", "textTransform": "uppercase",
                                     "letterSpacing": "0.07em", "color": C["muted"]}),
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
tab_map = html.Div([
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
            style={"height": "560px"},
        ),
    ),
])

# ── tab: Forecast ──────────────────────────────────────────────────────────────
def _stat_row(label, value, color=C["gold"]):
    return html.Div([
        html.Span(f"{label}: ", style={"color": C["muted"], "fontSize": "0.8rem"}),
        html.Span(value, style={"color": color, "fontWeight": "700", "fontSize": "0.9rem"}),
    ], className="mb-1")


_forecast_metrics_div = html.Div([
    _stat_row("MAE",  f"{_TEST_MAE:.2f} index pts",  C["gold"]),
    _stat_row("RMSE", f"{_TEST_RMSE:.2f} index pts", C["orange"]),
    _stat_row("MAPE", f"{_TEST_MAPE:.1f}%",          C["red"]),
    html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
    html.Div(
        "Evaluation model trained 2010–2022 only, tested on 2023–2024 (n = 24 months).",
        style={"fontSize": "0.75rem", "color": C["muted"]},
    ),
])

_dec26 = {
    "bear": DF_FC_BEAR.iloc[-1],
    "base": DF_FC_BASE.iloc[-1],
    "bull": DF_FC_BULL.iloc[-1],
}

_scenario_targets_div = html.Div([
    *[
        html.Div([
            html.Span("● ", style={"color": SCENARIO_STYLES[n][0], "fontSize": "1rem"}),
            html.Span(f"{SCENARIO_STYLES[n][2]}: ",
                      style={"color": C["muted"], "fontSize": "0.8rem"}),
            html.Span(f"{_dec26[n]['yhat']:.1f}",
                      style={"color": SCENARIO_STYLES[n][0],
                             "fontWeight": "700", "fontSize": "0.9rem"}),
            html.Span(f"  [{_dec26[n]['yhat_lower']:.1f} – {_dec26[n]['yhat_upper']:.1f}]",
                      style={"color": C["muted"], "fontSize": "0.78rem"}),
        ], className="mb-2")
        for n in ["bear", "base", "bull"]
    ],
    html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
    html.Div(
        "Dec 2026 AHPI targets under each scenario. 90% credible interval in brackets.",
        style={"fontSize": "0.75rem", "color": C["muted"]},
    ),
])

tab_forecast = html.Div([
    section_card(
        dbc.Row([
            dbc.Col([
                html.Label("90% Confidence Intervals",
                           style={"fontSize": "0.78rem", "color": C["muted"]}),
                dbc.Switch(id="forecast-ci", value=True, label=""),
            ], md=3, className="d-flex flex-column justify-content-start"),
            dbc.Col([
                html.Div(
                    "Prophet mid-market model trained on 180 months (Jan 2010 – Dec 2024) "
                    "with 6 macro regressors. "
                    "Test accuracy uses an evaluation model trained on 2010–2022 only.",
                    style={"fontSize": "0.75rem", "color": C["muted"], "paddingTop": "8px"},
                ),
            ], md=9),
        ], className="mb-2"),
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
                html.P("Dec 2026 AHPI targets by scenario",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                _scenario_targets_div,
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
            ], md=3, className="d-flex flex-column justify-content-start"),
            dbc.Col([
                html.Div(
                    "One Prophet model per prime area (6 models). "
                    "USD-indexed markets: exchange rate is the dominant regressor. "
                    "Bear/Base/Bull assume GHS/USD → 20 / 15 / 12 by end-2026.",
                    style={"fontSize": "0.75rem", "color": C["muted"], "paddingTop": "8px"},
                ),
            ], md=4),
        ], className="mb-2"),
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
                html.P("Dec 2026 AHPI targets by scenario",
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
            ], md=3, className="d-flex flex-column justify-content-start"),
            dbc.Col([
                html.Div(
                    "One Prophet model per mid-market district (5 models). "
                    "GHS-denominated markets; exchange rate and CPI are the primary regressors. "
                    "Bear/Base/Bull assume GHS/USD → 20 / 15 / 12 by end-2026.",
                    style={"fontSize": "0.75rem", "color": C["muted"], "paddingTop": "8px"},
                ),
            ], md=4),
        ], className="mb-2"),
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
                html.P("Dec 2026 AHPI targets by scenario",
                       style={"color": C["muted"], "fontSize": "0.78rem",
                              "fontWeight": "600", "marginBottom": "10px"}),
                html.Div(id="district-fc-targets"),
            ), md=8,
        ),
    ]),
])

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

app.layout = html.Div(
    style={"backgroundColor": C["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', 'Segoe UI', Arial, sans-serif"},
    children=[

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
                            html.Span("AHPI Methodology: ",
                                      style={"color": C["muted"], "fontSize": "0.72rem"}),
                            html.Span("USD/sqm anchors (GPG · Numbeo · JLL · Knight Frank) → GHS conversion "
                                      "at Bank of Ghana rates → normalised 2015 = 100 → monthly interpolation",
                                      style={"color": C["text"], "fontSize": "0.72rem"}),
                        ]),
                    ], md=5, className="d-flex align-items-center"),
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
            ], id="main-tabs", active_tab="tab-overview",
               style={"borderBottom": f"1px solid {C['border']}"},
               className="mb-3"),

        ], fluid=True),

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
    ],
)


# ── callbacks ─────────────────────────────────────────────────────────────────

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


@app.callback(
    Output("prime-forecast-chart", "figure"),
    Output("prime-fc-metrics",     "children"),
    Output("prime-fc-targets",     "children"),
    Input("prime-fc-ci",   "value"),
    Input("prime-fc-area", "value"),
    prevent_initial_call=False,
)
def update_prime_forecast(show_ci, area):
    area = area or "all"
    return (
        build_prime_forecast_fig(area, show_ci=bool(show_ci)),
        _build_prime_metrics_div(area),
        _build_prime_targets_div(area),
    )


@app.callback(
    Output("district-forecast-chart", "figure"),
    Output("district-fc-metrics",     "children"),
    Output("district-fc-targets",     "children"),
    Input("district-fc-ci",   "value"),
    Input("district-fc-area", "value"),
    prevent_initial_call=False,
)
def update_district_forecast(show_ci, district):
    district = district or "all"
    return (
        build_district_forecast_fig(district, show_ci=bool(show_ci)),
        _build_district_metrics_div(district),
        _build_district_targets_div(district),
    )


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8050
    print(f"\n  AHPI Dashboard → http://localhost:{port}\n")
    app.run(debug=False, port=port, host="0.0.0.0")
