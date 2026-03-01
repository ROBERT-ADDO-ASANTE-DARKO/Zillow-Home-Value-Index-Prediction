#!/usr/bin/env python3
"""
Accra Home Price Index – Time-Series Visualisation
===================================================
Produces one high-resolution PNG (accra_home_price_index_plot.png) that
contains four coordinated panels covering every variable in the dataset.

Usage:
    python accra_plot_timeseries.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "accra_home_price_index.csv"
)
OUT_PATH  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "accra_home_price_index_plot.png"
)

# ── palette ──────────────────────────────────────────────────────────────────
GOLD   = "#C8963E"
GREEN  = "#2E7D5B"
RED    = "#C0392B"
BLUE   = "#1A5276"
PURPLE = "#6C3483"
TEAL   = "#117A8B"
GREY   = "#566573"
LGREY  = "#BFC9CA"
BG     = "#FAFAFA"

# Key events to annotate on the AHPI panel
EVENTS = [
    ("2014-01", "Cedi crisis\n(−30 % vs USD)", "down"),
    ("2016-01", "IMF bailout\n(USD 918 m)", "up"),
    ("2020-04", "COVID-19\npandemic", "down"),
    ("2022-01", "Debt crisis /\ncurrency collapse", "down"),
    ("2023-07", "IMF programme\n(USD 3 bn)", "up"),
    ("2024-01", "Cocoa price\nsurge (+117 %)", "up"),
]


def shade_recessions(ax, df):
    """Light grey bands for the two main economic contractions."""
    contractions = [("2014-06", "2016-12"), ("2020-01", "2021-06")]
    for s, e in contractions:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=LGREY, alpha=0.25, zorder=0)


def fmt_year(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", which="major", labelsize=8)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    t  = df["ds"]

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22), facecolor=BG)
    fig.suptitle(
        "Accra Home Price Index  |  Jan 2010 – Dec 2024\n"
        "Monthly time-series dataset for Facebook Prophet forecasting",
        fontsize=16, fontweight="bold", y=0.995, color="#1C1C1C"
    )

    gs = GridSpec(
        4, 2,
        figure=fig,
        hspace=0.52, wspace=0.30,
        left=0.07, right=0.97,
        top=0.965, bottom=0.04,
    )

    ax_ahpi  = fig.add_subplot(gs[0, :])    # full-width – target variable
    ax_macro = fig.add_subplot(gs[1, :])    # full-width – macro block
    ax_comm  = fig.add_subplot(gs[2, 0])    # left  – commodity prices
    ax_comm2 = fig.add_subplot(gs[2, 1])    # right – cocoa (different scale)
    ax_prop  = fig.add_subplot(gs[3, 0])    # left  – property in GHS
    ax_prop2 = fig.add_subplot(gs[3, 1])    # right – property in USD vs FX

    for ax in fig.get_axes():
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(LGREY)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color=LGREY)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 – Accra Home Price Index (target variable y)
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_ahpi
    ax.fill_between(t, df["y"], alpha=0.15, color=GOLD)
    ax.plot(t, df["y"], color=GOLD, linewidth=2.2, label="AHPI (base 2015 = 100)")

    # base-year reference line
    ax.axhline(100, color=GREY, linewidth=0.9, linestyle="--", alpha=0.7)
    ax.text(t.iloc[0], 103, " 2015 baseline (100)", color=GREY, fontsize=7.5)

    # annotate key events
    arrowprops = dict(arrowstyle="-|>", color=GREY, lw=0.8)
    for date_str, label, direction in EVENTS:
        x = pd.Timestamp(date_str)
        if x < t.min() or x > t.max():
            continue
        idx  = df["ds"].searchsorted(x)
        yval = df["y"].iloc[max(0, min(idx, len(df) - 1))]
        offset = 60 if direction == "up" else -60
        ax.annotate(
            label,
            xy=(x, yval),
            xytext=(x, yval + offset),
            fontsize=7,
            ha="center",
            color=GREY,
            arrowprops=arrowprops,
            bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=LGREY, alpha=0.9),
        )

    shade_recessions(ax, df)
    fmt_year(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.set_ylim(0, df["y"].max() * 1.25)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_title(
        "● TARGET VARIABLE  –  Accra Home Price Index (AHPI, GHS-denominated, 2015 = 100)",
        fontsize=11, fontweight="bold", loc="left", pad=6, color=BLUE
    )
    ax.set_ylabel("Index (2015 = 100)", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)

    # secondary axis: GHS/sqm
    ax2 = ax.twinx()
    ax2.plot(t, df["price_ghs_per_sqm"], color=GREEN, linewidth=1.4,
             linestyle="--", alpha=0.85, label="GHS / sqm")
    ax2.set_ylabel("GHS per sqm", fontsize=9, color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN, labelsize=8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREEN)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"
    ))
    ax2.legend(loc="upper center", fontsize=8)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 – Macroeconomic regressors (multi-line, dual axis)
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_macro
    shade_recessions(ax, df)

    # left axis: rates (%)
    l1, = ax.plot(t, df["inflation_cpi_pct"],  color=RED,    linewidth=1.8, label="Inflation CPI %")
    l2, = ax.plot(t, df["lending_rate_pct"],   color=PURPLE, linewidth=1.8, label="Lending rate %")
    l3, = ax.plot(t, df["gdp_growth_pct"],     color=GREEN,  linewidth=1.8, label="GDP growth %",    linestyle="--")
    l4, = ax.plot(t, df["unemployment_pct"],   color=TEAL,   linewidth=1.4, label="Unemployment %",  linestyle=":")
    ax.axhline(0, color=GREY, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylabel("Rate / Growth (%)", fontsize=9)
    fmt_year(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.set_title(
        "● MACROECONOMIC REGRESSORS  –  Rates, growth, and structural indicators",
        fontsize=11, fontweight="bold", loc="left", pad=6, color=BLUE
    )

    # right axis: exchange rate
    ax_r = ax.twinx()
    l5, = ax_r.plot(t, df["exchange_rate_ghs_usd"], color=GOLD,
                    linewidth=2.0, linestyle="-.", label="GHS/USD exchange rate")
    ax_r.set_ylabel("GHS per 1 USD", fontsize=9, color=GOLD)
    ax_r.tick_params(axis="y", labelcolor=GOLD, labelsize=8)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_color(GOLD)
    ax_r.spines["top"].set_visible(False)

    # combined legend
    handles = [l1, l2, l3, l4, l5]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, ncol=5,
              framealpha=0.9, edgecolor=LGREY)

    # ─ second right axis: govt debt % GDP (stacked twin trick)
    ax_r2 = ax.twinx()
    ax_r2.spines["right"].set_position(("outward", 55))
    ax_r2.spines["right"].set_visible(True)
    ax_r2.spines["right"].set_color(GREY)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.set_facecolor("none")
    ax_r2.fill_between(t, df["govt_debt_pct_gdp"], alpha=0.08, color=RED)
    ax_r2.plot(t, df["govt_debt_pct_gdp"], color=RED, linewidth=0.9,
               linestyle=":", alpha=0.7, label="Govt debt % GDP")
    ax_r2.set_ylabel("Govt debt % GDP", fontsize=8, color=RED)
    ax_r2.tick_params(axis="y", labelcolor=RED, labelsize=8)
    ax_r2.legend(loc="upper right", fontsize=8)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 3a – Gold & Oil prices
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_comm
    shade_recessions(ax, df)
    ax.plot(t, df["gold_price_usd"], color=GOLD, linewidth=1.8, label="Gold (USD/troy oz)")
    ax.fill_between(t, df["gold_price_usd"], alpha=0.10, color=GOLD)
    ax.set_ylabel("Gold (USD / troy oz)", fontsize=9, color=GOLD)
    ax.tick_params(axis="y", labelcolor=GOLD, labelsize=8)
    fmt_year(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.set_title("● COMMODITY PRICES  –  Gold & Brent crude",
                 fontsize=10, fontweight="bold", loc="left", pad=5, color=BLUE)

    ax_o = ax.twinx()
    ax_o.plot(t, df["oil_brent_usd"], color=GREY, linewidth=1.4,
              linestyle="--", label="Brent crude (USD/bbl)")
    ax_o.set_ylabel("Oil (USD / bbl)", fontsize=9, color=GREY)
    ax_o.tick_params(axis="y", labelcolor=GREY, labelsize=8)
    ax_o.spines["right"].set_visible(True)
    ax_o.spines["right"].set_color(GREY)
    ax_o.spines["top"].set_visible(False)

    lines_g = [
        Line2D([0], [0], color=GOLD, linewidth=1.8, label="Gold (USD/troy oz)"),
        Line2D([0], [0], color=GREY, linewidth=1.4, linestyle="--", label="Brent crude (USD/bbl)"),
    ]
    ax.legend(handles=lines_g, loc="upper left", fontsize=8)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 3b – Cocoa prices (Ghana = world's 2nd-largest producer)
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_comm2
    shade_recessions(ax, df)
    ax.fill_between(t, df["cocoa_price_usd"], alpha=0.15, color="#7B3F00")
    ax.plot(t, df["cocoa_price_usd"], color="#7B3F00", linewidth=1.8,
            label="Cocoa (USD/MT)")
    fmt_year(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.set_ylabel("Cocoa (USD / MT)", fontsize=9, color="#7B3F00")
    ax.tick_params(axis="y", labelcolor="#7B3F00", labelsize=8)
    ax.set_title("● COMMODITY PRICES  –  Cocoa\n(Ghana = 2nd-largest global producer)",
                 fontsize=10, fontweight="bold", loc="left", pad=5, color=BLUE)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}")
    )
    ax.legend(loc="upper left", fontsize=8)

    # annotate 2024 cocoa spike
    spike_x = pd.Timestamp("2024-06")
    spike_y = df.loc[df["ds"] == "2024-06-01", "cocoa_price_usd"].values
    if len(spike_y):
        ax.annotate(
            f"2024 surge\n{spike_y[0]/1000:.1f}k USD/MT\n(crop disease +\nWest Africa shortage)",
            xy=(spike_x, spike_y[0]),
            xytext=(pd.Timestamp("2022-01"), spike_y[0] * 0.85),
            fontsize=7, color=GREY,
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=LGREY),
        )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 4a – Property price in GHS/sqm vs CPI
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_prop
    shade_recessions(ax, df)
    ax.fill_between(t, df["price_ghs_per_sqm"] / 1000, alpha=0.15, color=GREEN)
    ax.plot(t, df["price_ghs_per_sqm"] / 1000, color=GREEN, linewidth=1.8,
            label="Residential price (GHS/sqm ÷1000)")
    ax.set_ylabel("GHS / sqm (thousands)", fontsize=9, color=GREEN)
    ax.tick_params(axis="y", labelcolor=GREEN, labelsize=8)
    fmt_year(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.set_title("● PROPERTY PRICES  –  Local currency (GHS)\nvs Consumer Price Index",
                 fontsize=10, fontweight="bold", loc="left", pad=5, color=BLUE)

    ax_c = ax.twinx()
    ax_c.plot(t, df["cpi_index"], color=RED, linewidth=1.3,
              linestyle="--", alpha=0.8, label="CPI index (2010 = 100)")
    ax_c.set_ylabel("CPI index (2010 = 100)", fontsize=9, color=RED)
    ax_c.tick_params(axis="y", labelcolor=RED, labelsize=8)
    ax_c.spines["right"].set_visible(True)
    ax_c.spines["right"].set_color(RED)
    ax_c.spines["top"].set_visible(False)

    lines_p = [
        Line2D([0], [0], color=GREEN, linewidth=1.8, label="GHS/sqm (÷1000)"),
        Line2D([0], [0], color=RED,   linewidth=1.3, linestyle="--", label="CPI index"),
    ]
    ax.legend(handles=lines_p, loc="upper left", fontsize=8)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 4b – Property price in USD/sqm vs remittances & FDI
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_prop2
    shade_recessions(ax, df)
    ax.plot(t, df["price_usd_per_sqm"], color=BLUE, linewidth=1.8,
            label="Residential price (USD/sqm)")
    ax.fill_between(t, df["price_usd_per_sqm"], alpha=0.12, color=BLUE)
    ax.set_ylabel("USD / sqm", fontsize=9, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
    fmt_year(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.set_title("● PROPERTY PRICES  –  USD/sqm\nvs Remittances & FDI (% GDP)",
                 fontsize=10, fontweight="bold", loc="left", pad=5, color=BLUE)

    ax_rem = ax.twinx()
    ax_rem.fill_between(t, df["remittances_pct_gdp"], alpha=0.10, color=PURPLE)
    ax_rem.plot(t, df["remittances_pct_gdp"], color=PURPLE, linewidth=1.2,
                linestyle="--", label="Remittances % GDP")
    ax_rem.fill_between(t, df["fdi_pct_gdp"], alpha=0.07, color=TEAL)
    ax_rem.plot(t, df["fdi_pct_gdp"], color=TEAL, linewidth=1.2,
                linestyle=":", label="FDI % GDP")
    ax_rem.set_ylabel("% of GDP", fontsize=9)
    ax_rem.tick_params(labelsize=8)
    ax_rem.spines["right"].set_visible(True)
    ax_rem.spines["right"].set_color(GREY)
    ax_rem.spines["top"].set_visible(False)

    lines_u = [
        Line2D([0], [0], color=BLUE,   linewidth=1.8, label="USD/sqm"),
        Line2D([0], [0], color=PURPLE, linewidth=1.2, linestyle="--", label="Remittances % GDP"),
        Line2D([0], [0], color=TEAL,   linewidth=1.2, linestyle=":",  label="FDI % GDP"),
    ]
    ax.legend(handles=lines_u, loc="upper right", fontsize=8)

    # ── global caption ────────────────────────────────────────────────────
    caption = (
        "Grey bands = economic contractions (2014–16 cedi crisis; 2020 COVID).  "
        "AHPI methodology: annual USD/sqm price anchors (Global Property Guide, Numbeo, JLL, Knight Frank) "
        "converted to GHS at Bank-of-Ghana rates, normalised to 2015 = 100 and interpolated to monthly frequency.  "
        "Macroeconomic data: World Bank Open Data.  Commodity prices: LBMA (gold), ICCO (cocoa), EIA/Platts (oil)."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7,
             color=GREY, wrap=True, style="italic")

    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Plot saved → {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
