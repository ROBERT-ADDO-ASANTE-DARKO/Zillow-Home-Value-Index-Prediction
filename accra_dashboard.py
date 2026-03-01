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
DF = pd.read_csv(DATA_PATH, parse_dates=["ds"])
YEARS = list(range(DF["ds"].dt.year.min(), DF["ds"].dt.year.max() + 1))

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
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                borderwidth=1, font_size=10),
    xaxis=dict(showgrid=False, linecolor=C["border"],
               tickcolor=C["muted"], tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor=C["border"],
               linecolor=C["border"], tickfont=dict(size=10)),
)

# ── helpers ───────────────────────────────────────────────────────────────────
def filter_df(start_yr, end_yr):
    return DF[(DF["ds"].dt.year >= start_yr) & (DF["ds"].dt.year <= end_yr)].copy()


def add_event_lines(fig, dff, row=1, col=1):
    """Add vertical event annotation lines to a figure."""
    for date_str, label, pos in KEY_EVENTS:
        dt = pd.Timestamp(date_str)
        if dt < dff["ds"].min() or dt > dff["ds"].max():
            continue
        idx   = dff["ds"].searchsorted(dt)
        y_val = float(dff["y"].iloc[max(0, min(idx, len(dff) - 1))])
        ypos  = 1.02 if pos == "above" else -0.06
        fig.add_vline(
            x=dt.timestamp() * 1000,
            line_width=1, line_dash="dot",
            line_color=C["muted"],
            opacity=0.5,
            row=row, col=col,
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
            row=row, col=col,
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
def build_ahpi_fig(dff, overlays, show_events):
    fig = go.Figure()

    # area fill
    fig.add_trace(go.Scatter(
        x=dff["ds"], y=dff["y"],
        fill="tozeroy", fillcolor="rgba(212,160,23,0.12)",
        line=dict(color=C["gold"], width=2.5),
        name="AHPI (2015 = 100)",
        hovertemplate="%{y:.1f}<extra>AHPI</extra>",
    ))

    # baseline
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"],
                  line_width=0.9, opacity=0.5,
                  annotation_text="2015 baseline",
                  annotation_font_color=C["muted"],
                  annotation_font_size=9)

    if "ghs" in overlays:
        fig.add_trace(go.Scatter(
            x=dff["ds"], y=dff["price_ghs_per_sqm"],
            line=dict(color=C["green"], width=1.6, dash="dash"),
            name="GHS / sqm",
            yaxis="y2",
            hovertemplate="GHS %{y:,.0f}<extra>GHS/sqm</extra>",
        ))

    if "usd" in overlays:
        fig.add_trace(go.Scatter(
            x=dff["ds"], y=dff["price_usd_per_sqm"],
            line=dict(color=C["blue"], width=1.6, dash="dot"),
            name="USD / sqm",
            yaxis="y3",
            hovertemplate="$%{y:,.0f}<extra>USD/sqm</extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Accra Home Price Index</b>  (GHS-denominated, base 2015 = 100)",
                   font_size=14, x=0.01),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title="Index (2015 = 100)"),
        yaxis2=dict(title="GHS / sqm", overlaying="y", side="right",
                    showgrid=False, tickformat=",",
                    titlefont=dict(color=C["green"]),
                    tickfont=dict(color=C["green"], size=10)),
        yaxis3=dict(title="USD / sqm", overlaying="y", side="right",
                    anchor="free", position=0.97,
                    showgrid=False,
                    titlefont=dict(color=C["blue"]),
                    tickfont=dict(color=C["blue"], size=10)),
        legend=dict(**BASE_LAYOUT["legend"], orientation="h",
                    x=0.01, y=1.06),
        height=440,
    )
    if show_events:
        add_event_lines(fig, dff)
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
        **BASE_LAYOUT,
        title=dict(text="<b>Macroeconomic Regressors</b>", font_size=14, x=0.01),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title=y_title),
        legend=dict(**BASE_LAYOUT["legend"], orientation="h", x=0.01, y=1.08),
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
        **BASE_LAYOUT,
        title=dict(text="<b>Commodity Prices</b>  –  Key drivers of the Ghanaian economy",
                   font_size=14, x=0.01),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title="Gold (USD/oz)"),
        yaxis2=dict(title="Brent (USD/bbl)", overlaying="y", side="right",
                    showgrid=False,
                    titlefont=dict(color=C["muted"]),
                    tickfont=dict(color=C["muted"], size=10)),
        yaxis3=dict(**BASE_LAYOUT["yaxis"], title="Cocoa (USD/MT)"),
        legend=dict(**BASE_LAYOUT["legend"], orientation="h", x=0.01, y=1.08),
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
        **BASE_LAYOUT,
        title=dict(
            text=f"<b>{y_label}</b>  vs  <b>{x_label}</b>"
                 + (f"   <span style='color:{C['muted']}; font-size:12px'>R² = {r2:.3f}</span>"
                    if r2 is not None else ""),
            font_size=13, x=0.01,
        ),
        xaxis=dict(**BASE_LAYOUT["xaxis"], title=x_label),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title=y_label),
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
                        html.Div("Monthly dataset · Jan 2010 – Dec 2024  ·  20 regressors  ·  Prophet-ready",
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
    Input("year-range", "value"),
    prevent_initial_call=False,
)
def update_kpis(yr_range):
    dff    = filter_df(*yr_range)
    latest = dff.iloc[-1]
    prev   = dff.iloc[-13] if len(dff) > 13 else dff.iloc[0]

    def yoy(col):
        chg = latest[col] - prev[col]
        pct = (chg / abs(prev[col]) * 100) if prev[col] != 0 else 0
        arrow = "▲" if chg >= 0 else "▼"
        color = C["green"] if chg >= 0 else C["red"]
        return html.Span(f"{arrow} {abs(pct):.1f}% YoY",
                         style={"color": color})

    return (
        f"{latest['y']:.1f}",             yoy("y"),
        f"GHS {latest['price_ghs_per_sqm']:,.0f}",  yoy("price_ghs_per_sqm"),
        f"${latest['price_usd_per_sqm']:,.0f}",      yoy("price_usd_per_sqm"),
        f"{latest['exchange_rate_ghs_usd']:.2f}",    yoy("exchange_rate_ghs_usd"),
        f"{latest['inflation_cpi_pct']:.1f}%",        yoy("inflation_cpi_pct"),
        f"${latest['gold_price_usd']:,.0f}",          yoy("gold_price_usd"),
    )


@app.callback(
    Output("ahpi-chart", "figure"),
    Input("year-range",    "value"),
    Input("ahpi-overlays", "value"),
    Input("ahpi-events",   "value"),
    prevent_initial_call=False,
)
def update_ahpi(yr_range, overlays, events):
    dff = filter_df(*yr_range)
    return build_ahpi_fig(dff, overlays or [], bool(events))


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


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8050
    print(f"\n  AHPI Dashboard → http://localhost:{port}\n")
    app.run(debug=False, port=port, host="0.0.0.0")
