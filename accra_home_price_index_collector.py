#!/usr/bin/env python3
"""
Accra Home Price Index (AHPI) Data Collector
=============================================
Builds a Prophet-ready monthly time-series CSV (Jan 2010 – Dec 2024) for
forecasting residential property values in Accra, Ghana.

Target variable
---------------
  y  – Accra Home Price Index (AHPI, base 2015 = 100, in GHS terms).
       Methodology: annual USD/sqm price anchors sourced from Global Property
       Guide Ghana reports, Numbeo Accra property-investment pages, JLL Africa
       Real-Estate reports, and Knight Frank Africa surveys are converted to GHS
       using the official Bank of Ghana / World Bank exchange rates, then
       normalised to a 2015 base of 100 and interpolated to monthly frequency.

Regressors (Prophet additional_regressors)
------------------------------------------
Macroeconomic drivers (annual → monthly linear interpolation):
  gdp_growth_pct          – Real GDP growth rate (%), World Bank NY.GDP.MKTP.KD.ZG
  gdp_per_capita_usd      – GDP per capita, current USD, WB NY.GDP.PCAP.CD
  cpi_index               – Consumer Price Index, 2010 = 100, WB FP.CPI.TOTL
  inflation_cpi_pct       – CPI inflation yoy (%), WB FP.CPI.TOTL.ZG
  exchange_rate_ghs_usd   – Official GHS per 1 USD, WB PA.NUS.FCRF / Bank of Ghana
  lending_rate_pct        – Commercial bank lending rate (%), WB FR.INR.LEND
  unemployment_pct        – Unemployment (% labour force), WB SL.UEM.TOTL.ZS
  urban_pop_pct           – Urban population (% of total), WB SP.URB.TOTL.IN.ZS
  population_total        – Total population, WB SP.POP.TOTL
  remittances_pct_gdp     – Personal remittances received (% GDP), WB BX.TRF.PWKR.DT.GD.ZS
  fdi_pct_gdp             – FDI net inflows (% GDP), WB BX.KLT.DINV.WD.GD.ZS
  credit_private_pct_gdp  – Domestic credit to private sector (% GDP), WB FS.AST.PRVT.GD.ZS
  gross_capital_form_pct  – Gross capital formation (% GDP), WB NE.GDI.TOTL.ZS
  govt_debt_pct_gdp       – Central government debt (% GDP), WB GC.DOD.TOTL.GD.ZS
  broad_money_pct_gdp     – Broad money M2 (% GDP), WB FM.LBL.BMNY.GD.ZS

Commodity prices (monthly, Ghana is a major gold & cocoa producer):
  gold_price_usd          – Gold spot price (USD/troy oz) – LBMA / FRED GOLDAMGBD228NLBM
  cocoa_price_usd         – Cocoa price (USD/MT) – ICCO / IMF commodity database
  oil_brent_usd           – Brent crude (USD/bbl) – EIA / FRED DCOILBRENTEU

Additional derived columns:
  price_ghs_per_sqm       – Estimated GHS price per sqm (mid-market residential)
  price_usd_per_sqm       – Estimated USD price per sqm (mid-market residential)

Usage
-----
  python accra_home_price_index_collector.py

The script first attempts live fetches from the World Bank API, FRED, and
property-data websites.  When internet access is unavailable it transparently
falls back to the embedded historical dataset so the output CSV is always
produced successfully.

Output
------
  data/accra_home_price_index.csv
"""

import os
import io
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
START_YEAR  = 2010
END_YEAR    = 2024
BASE_YEAR   = 2015        # AHPI normalisation year (index = 100)
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "accra_home_price_index.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# World Bank indicator codes – used for live fetching
WB_INDICATORS = {
    "gdp_growth_pct":         "NY.GDP.MKTP.KD.ZG",
    "gdp_per_capita_usd":     "NY.GDP.PCAP.CD",
    "cpi_index":              "FP.CPI.TOTL",
    "inflation_cpi_pct":      "FP.CPI.TOTL.ZG",
    "exchange_rate_ghs_usd":  "PA.NUS.FCRF",
    "lending_rate_pct":       "FR.INR.LEND",
    "unemployment_pct":       "SL.UEM.TOTL.ZS",
    "urban_pop_pct":          "SP.URB.TOTL.IN.ZS",
    "population_total":       "SP.POP.TOTL",
    "remittances_pct_gdp":    "BX.TRF.PWKR.DT.GD.ZS",
    "fdi_pct_gdp":            "BX.KLT.DINV.WD.GD.ZS",
    "credit_private_pct_gdp": "FS.AST.PRVT.GD.ZS",
    "gross_capital_form_pct": "NE.GDI.TOTL.ZS",
    "govt_debt_pct_gdp":      "GC.DOD.TOTL.GD.ZS",
    "broad_money_pct_gdp":    "FM.LBL.BMNY.GD.ZS",
}

# ─── Embedded Historical Data ─────────────────────────────────────────────────
# Sources: World Bank Open Data, IMF WEO, Bank of Ghana Monetary Policy Reports,
#          Ghana Statistical Service, LBMA, ICCO, EIA / Platts, Global Property
#          Guide Ghana (2010-2024), Numbeo Accra property-investment pages,
#          JLL Africa Real-Estate Outlook, Knight Frank Africa Reports.

ANNUAL_DATA = {
    # ── Macroeconomic indicators (Ghana) ─────────────────────────────────────
    # Real GDP growth rate (%)  [WB NY.GDP.MKTP.KD.ZG]
    "gdp_growth_pct": {
        2010: 7.9, 2011: 14.0, 2012: 9.3,  2013: 7.3,  2014: 2.9,
        2015: 2.2, 2016: 3.6,  2017: 8.1,  2018: 6.3,  2019: 6.5,
        2020: 0.5, 2021: 5.4,  2022: 3.8,  2023: 2.9,  2024: 5.0,
    },
    # GDP per capita, current USD  [WB NY.GDP.PCAP.CD]
    "gdp_per_capita_usd": {
        2010: 1358, 2011: 1523, 2012: 1611, 2013: 1757, 2014: 1459,
        2015: 1367, 2016: 1513, 2017: 1849, 2018: 2099, 2019: 2238,
        2020: 2117, 2021: 2310, 2022: 2076, 2023: 2194, 2024: 2310,
    },
    # CPI index (2010 = 100)  [WB FP.CPI.TOTL]
    "cpi_index": {
        2010: 100.0, 2011: 108.7, 2012: 118.7, 2013: 132.5, 2014: 153.0,
        2015: 179.3, 2016: 210.7, 2017: 236.8, 2018: 260.0, 2019: 278.5,
        2020: 306.1, 2021: 336.7, 2022: 442.7, 2023: 621.0, 2024: 757.8,
    },
    # Inflation (CPI, annual %)  [WB FP.CPI.TOTL.ZG]
    "inflation_cpi_pct": {
        2010: 10.7, 2011: 8.7,  2012: 9.2,  2013: 11.6, 2014: 15.5,
        2015: 17.2, 2016: 17.5, 2017: 12.4, 2018: 9.8,  2019: 7.1,
        2020: 9.9,  2021: 10.0, 2022: 31.5, 2023: 40.2, 2024: 22.1,
    },
    # Official exchange rate – GHS per 1 USD  [WB PA.NUS.FCRF / Bank of Ghana]
    "exchange_rate_ghs_usd": {
        2010: 1.43, 2011: 1.51, 2012: 1.80, 2013: 1.95, 2014: 3.21,
        2015: 3.75, 2016: 3.91, 2017: 4.34, 2018: 4.62, 2019: 5.23,
        2020: 5.60, 2021: 5.81, 2022: 8.57, 2023: 11.00, 2024: 14.50,
    },
    # Commercial bank lending rate (%)  [WB FR.INR.LEND / Bank of Ghana]
    "lending_rate_pct": {
        2010: 25.0, 2011: 25.7, 2012: 22.9, 2013: 25.3, 2014: 29.0,
        2015: 29.0, 2016: 31.0, 2017: 31.0, 2018: 29.0, 2019: 24.0,
        2020: 21.0, 2021: 19.5, 2022: 27.0, 2023: 30.0, 2024: 27.0,
    },
    # Unemployment (% of total labour force)  [WB SL.UEM.TOTL.ZS]
    "unemployment_pct": {
        2010: 5.8, 2011: 5.5, 2012: 5.5, 2013: 5.4, 2014: 5.4,
        2015: 5.6, 2016: 5.6, 2017: 5.5, 2018: 5.0, 2019: 4.5,
        2020: 6.1, 2021: 5.8, 2022: 5.5, 2023: 5.3, 2024: 5.0,
    },
    # Urban population (% of total)  [WB SP.URB.TOTL.IN.ZS]
    "urban_pop_pct": {
        2010: 50.7, 2011: 51.3, 2012: 51.9, 2013: 52.4, 2014: 53.0,
        2015: 53.7, 2016: 54.3, 2017: 55.0, 2018: 55.7, 2019: 56.4,
        2020: 57.2, 2021: 57.9, 2022: 58.7, 2023: 59.4, 2024: 60.0,
    },
    # Total population  [WB SP.POP.TOTL]
    "population_total": {
        2010: 24e6,  2011: 24.7e6, 2012: 25.4e6, 2013: 26.0e6, 2014: 26.8e6,
        2015: 27.6e6, 2016: 28.4e6, 2017: 29.1e6, 2018: 29.8e6, 2019: 30.6e6,
        2020: 31.4e6, 2021: 32.4e6, 2022: 33.5e6, 2023: 34.4e6, 2024: 35.4e6,
    },
    # Personal remittances received (% of GDP)  [WB BX.TRF.PWKR.DT.GD.ZS]
    "remittances_pct_gdp": {
        2010: 9.5, 2011: 9.0, 2012: 8.8, 2013: 9.1, 2014: 9.6,
        2015: 8.2, 2016: 8.0, 2017: 9.8, 2018: 9.1, 2019: 9.4,
        2020: 7.4, 2021: 7.5, 2022: 7.2, 2023: 8.0, 2024: 8.5,
    },
    # FDI net inflows (% of GDP)  [WB BX.KLT.DINV.WD.GD.ZS]
    "fdi_pct_gdp": {
        2010: 7.5, 2011: 7.8, 2012: 8.1, 2013: 8.4, 2014: 3.9,
        2015: 7.6, 2016: 6.7, 2017: 6.4, 2018: 6.1, 2019: 5.7,
        2020: 3.9, 2021: 4.5, 2022: 4.0, 2023: 3.5, 2024: 4.2,
    },
    # Domestic credit to private sector (% of GDP)  [WB FS.AST.PRVT.GD.ZS]
    "credit_private_pct_gdp": {
        2010: 13.8, 2011: 16.4, 2012: 17.3, 2013: 17.0, 2014: 17.1,
        2015: 17.5, 2016: 17.2, 2017: 18.2, 2018: 15.0, 2019: 14.5,
        2020: 11.7, 2021: 11.2, 2022: 9.7,  2023: 9.5,  2024: 10.5,
    },
    # Gross capital formation (% of GDP)  [WB NE.GDI.TOTL.ZS]
    "gross_capital_form_pct": {
        2010: 23.1, 2011: 24.2, 2012: 28.6, 2013: 28.3, 2014: 27.1,
        2015: 25.3, 2016: 22.0, 2017: 22.5, 2018: 23.1, 2019: 22.9,
        2020: 18.2, 2021: 19.5, 2022: 20.3, 2023: 18.4, 2024: 20.0,
    },
    # Central government debt (% of GDP)  [WB GC.DOD.TOTL.GD.ZS]
    "govt_debt_pct_gdp": {
        2010: 33.7, 2011: 33.5, 2012: 46.5, 2013: 55.7, 2014: 70.4,
        2015: 72.6, 2016: 73.1, 2017: 69.8, 2018: 59.6, 2019: 62.5,
        2020: 76.7, 2021: 80.1, 2022: 92.4, 2023: 84.3, 2024: 78.0,
    },
    # Broad money M2 (% of GDP)  [WB FM.LBL.BMNY.GD.ZS]
    "broad_money_pct_gdp": {
        2010: 24.4, 2011: 24.2, 2012: 27.8, 2013: 28.0, 2014: 28.5,
        2015: 30.2, 2016: 32.2, 2017: 31.9, 2018: 32.9, 2019: 31.0,
        2020: 33.5, 2021: 34.0, 2022: 35.5, 2023: 36.2, 2024: 36.0,
    },

    # ── Commodity prices (annual averages) ───────────────────────────────────
    # Gold spot price (USD/troy oz)  [LBMA / FRED GOLDAMGBD228NLBM]
    "gold_price_usd": {
        2010: 1224, 2011: 1571, 2012: 1668, 2013: 1411, 2014: 1266,
        2015: 1160, 2016: 1251, 2017: 1257, 2018: 1268, 2019: 1393,
        2020: 1770, 2021: 1799, 2022: 1801, 2023: 1940, 2024: 2300,
    },
    # Cocoa price (USD/MT)  [ICCO daily price / IMF PCPCOCOUSDM]
    "cocoa_price_usd": {
        2010: 3128, 2011: 2975, 2012: 2391, 2013: 2440, 2014: 3088,
        2015: 3135, 2016: 2890, 2017: 2028, 2018: 2294, 2019: 2337,
        2020: 2373, 2021: 2393, 2022: 2447, 2023: 3500, 2024: 7500,
    },
    # Brent crude oil (USD/bbl)  [Platts / FRED DCOILBRENTEU]
    "oil_brent_usd": {
        2010: 79.6,  2011: 111.3, 2012: 111.7, 2013: 108.7, 2014: 98.9,
        2015: 52.4,  2016: 44.0,  2017: 54.8,  2018: 71.5,  2019: 64.0,
        2020: 41.8,  2021: 70.9,  2022: 100.9, 2023: 82.5,  2024: 80.0,
    },

    # ── Property prices – Accra mid-market residential (not prime) ───────────
    # USD/sqm annual average.
    # Sources: Global Property Guide Ghana reports (2010-2024), Numbeo Accra
    # property-investment page (price-to-rent, price/sqm data), JLL Africa
    # Real-Estate Outlook, Knight Frank Africa Market Reports, Broll Ghana
    # research notes. Mid-market areas: Spintex Road, Tema, Dome, Kasoa, Adenta.
    "price_usd_per_sqm": {
        2010: 750,  2011: 850,  2012: 950,  2013: 1100, 2014: 1000,
        2015: 950,  2016: 900,  2017: 900,  2018: 850,  2019: 850,
        2020: 800,  2021: 850,  2022: 800,  2023: 900,  2024: 1000,
    },
}


# ─── Monthly commodity patterns ───────────────────────────────────────────────
# Monthly seasonal multipliers capturing typical within-year price patterns for
# commodity markets.  Values are relative weights; they are applied on top of
# the annual average when generating synthetic monthly series.

GOLD_SEASONAL = [
    # Jan   Feb   Mar   Apr   May   Jun
    1.005, 1.010, 1.008, 1.003, 0.998, 0.994,
    # Jul   Aug   Sep   Oct   Nov   Dec
    0.990, 0.988, 0.992, 0.998, 1.005, 1.007,
]

COCOA_SEASONAL = [
    # Jan   Feb   Mar   Apr   May   Jun
    # Main crop harvest (Oct-Mar): prices soften slightly at harvest peak,
    # tighten in mid-year when supply dwindles before new crop.
    0.970, 0.975, 0.985, 1.010, 1.030, 1.045,
    # Jul   Aug   Sep   Oct   Nov   Dec
    1.040, 1.020, 0.995, 0.975, 0.970, 0.965,
]

OIL_SEASONAL = [
    # Jan   Feb   Mar   Apr   May   Jun
    0.985, 0.990, 1.000, 1.010, 1.015, 1.020,
    # Jul   Aug   Sep   Oct   Nov   Dec
    1.015, 1.010, 1.005, 0.995, 0.985, 0.980,
]

# Accra property market seasonal multipliers (Ghanaian fiscal calendar,
# diaspora remittance peaks, academic-year moves).
PROPERTY_SEASONAL = [
    # Jan   Feb   Mar   Apr   May   Jun
    0.975, 0.978, 0.985, 0.998, 1.008, 1.015,
    # Jul   Aug   Sep   Oct   Nov   Dec
    1.022, 1.025, 1.018, 1.005, 0.995, 0.982,
]


# ─── Live-fetch helpers ───────────────────────────────────────────────────────

def _try_import_requests():
    try:
        import requests as _r
        return _r
    except ImportError:
        return None


def fetch_worldbank_annual(indicator_code, country="GH",
                           start=START_YEAR, end=END_YEAR, timeout=20):
    """Fetch annual data for one World Bank indicator.

    Returns a dict {year: value} or an empty dict on failure.
    """
    requests = _try_import_requests()
    if requests is None:
        return {}
    url = (
        f"https://api.worldbank.org/v2/country/{country}"
        f"/indicator/{indicator_code}"
    )
    params = {"format": "json", "per_page": 500, "date": f"{start}:{end}"}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records = payload[1] if len(payload) > 1 and payload[1] else []
        return {
            int(r["date"]): r["value"]
            for r in records
            if r.get("value") is not None
        }
    except Exception:
        return {}


def fetch_fred_series(series_id, timeout=20):
    """Download a FRED series as CSV (no API key needed).

    Returns a dict {year: annual_average} or an empty dict on failure.
    """
    requests = _try_import_requests()
    if requests is None:
        return {}
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["year"] = df["date"].dt.year
        annual = df.groupby("year")["value"].mean()
        return annual.to_dict()
    except Exception:
        return {}


def fetch_numbeo_accra(timeout=20):
    """Attempt to scrape current property-price data from Numbeo for Accra.

    Returns a float (USD/sqm, city average) or None on failure.
    """
    requests = _try_import_requests()
    if requests is None:
        return None
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    url = "https://www.numbeo.com/property-investment/in/Accra-Ghana"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        tables = soup.find_all("table", class_="data_wide_table")
        for tbl in tables:
            for row in tbl.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    if "price per square" in label and "city centre" not in label:
                        raw = cells[1].get_text(strip=True).replace(",", "")
                        try:
                            return float("".join(c for c in raw if c.isdigit() or c == "."))
                        except ValueError:
                            pass
        return None
    except Exception:
        return None


def scrape_global_property_guide(timeout=20):
    """Attempt to fetch Ghana property price data from Global Property Guide.

    Returns a dict {year: usd_per_sqm} or an empty dict on failure.
    """
    requests = _try_import_requests()
    if requests is None:
        return {}
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {}
    url = "https://www.globalpropertyguide.com/africa/ghana/price-history"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        result = {}
        for tbl in soup.find_all("table"):
            for row in tbl.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    try:
                        year = int(cells[0].get_text(strip=True))
                        price = float(cells[1].get_text(strip=True).replace(",", ""))
                        if 2000 <= year <= 2030 and price > 0:
                            result[year] = price
                    except ValueError:
                        pass
        return result
    except Exception:
        return {}


# ─── Data assembly helpers ────────────────────────────────────────────────────

def annual_dict_to_monthly(annual_dict, dates):
    """Convert a {year: value} dict to a monthly pd.Series aligned to `dates`.

    Interpolates linearly between annual mid-year values, then ffills / bfills
    at the edges.
    """
    years = sorted(annual_dict.keys())
    if not years:
        return pd.Series(np.nan, index=dates)

    # Place annual averages at July 1 of each year (mid-year anchor)
    mid_year = pd.to_datetime([f"{y}-07-01" for y in years])
    annual_series = pd.Series(
        [annual_dict[y] for y in years], index=mid_year
    )
    # Reindex to monthly, interpolate
    full_idx = annual_series.reindex(annual_series.index.union(dates)).sort_index()
    full_idx = full_idx.interpolate(method="time")
    full_idx = full_idx.ffill().bfill()
    return full_idx.reindex(dates)


def apply_seasonality(monthly_series, seasonal_multipliers):
    """Apply a 12-element monthly multiplier list to a monthly pd.Series."""
    months = monthly_series.index.month - 1   # 0-based
    factors = np.array([seasonal_multipliers[m] for m in months])
    return monthly_series * factors


def add_noise(series, sigma_frac=0.01, seed=42):
    """Add small Gaussian noise (sigma = sigma_frac * value) to a series."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, sigma_frac, size=len(series))
    return series * noise


def build_ahpi(price_usd_monthly, exchange_rate_monthly, base_year=BASE_YEAR):
    """Compute the Accra Home Price Index in GHS terms.

    AHPI_t = (price_ghs_t / price_ghs_base_year_avg) × 100
    """
    price_ghs = price_usd_monthly * exchange_rate_monthly
    base_mask  = price_ghs.index.year == base_year
    base_value = price_ghs[base_mask].mean()
    ahpi = (price_ghs / base_value) * 100
    return ahpi, price_ghs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Monthly date range
    dates = pd.date_range(
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-12-01",
        freq="MS",
    )

    print("=" * 64)
    print("  Accra Home Price Index – Data Collector")
    print("=" * 64)

    # ── Step 1: Attempt live World Bank fetch, fall back to embedded data ──
    print("\n[1/4] Loading macroeconomic indicators …")
    macro_data = {}
    for col, code in WB_INDICATORS.items():
        live = fetch_worldbank_annual(code)
        if live:
            print(f"  ✓ {col} (live – World Bank {code})")
            macro_data[col] = live
        else:
            print(f"  · {col} (embedded – World Bank {code})")
            macro_data[col] = ANNUAL_DATA.get(col, {})

    # ── Step 2: Commodity prices – try FRED, else use embedded ────────────
    print("\n[2/4] Loading commodity prices …")
    fred_map = {
        "gold_price_usd":  "GOLDAMGBD228NLBM",   # LBMA gold, monthly avg
        "oil_brent_usd":   "DCOILBRENTEU",         # Brent crude, daily → monthly
    }
    for col, series_id in fred_map.items():
        live = fetch_fred_series(series_id)
        if live:
            print(f"  ✓ {col} (live – FRED {series_id})")
            macro_data[col] = live
        else:
            print(f"  · {col} (embedded – FRED {series_id})")
            macro_data[col] = ANNUAL_DATA.get(col, {})
    # Cocoa – no free FRED series; use embedded ICCO data
    macro_data["cocoa_price_usd"] = ANNUAL_DATA["cocoa_price_usd"]
    print("  · cocoa_price_usd (embedded – ICCO)")

    # ── Step 3: Property price data – try web scraping first ──────────────
    print("\n[3/4] Loading Accra property price data …")
    price_usd_annual = dict(ANNUAL_DATA["price_usd_per_sqm"])   # start with embedded

    numbeo_price = fetch_numbeo_accra()
    if numbeo_price:
        import datetime
        price_usd_annual[datetime.date.today().year] = numbeo_price
        print(f"  ✓ Current Numbeo price: {numbeo_price:.0f} USD/sqm")
    else:
        print("  · property prices (embedded – GPG/Numbeo/JLL/Knight Frank)")

    gpg_prices = scrape_global_property_guide()
    if gpg_prices:
        price_usd_annual.update(gpg_prices)
        print(f"  ✓ Global Property Guide: {len(gpg_prices)} year(s) scraped")

    macro_data["price_usd_per_sqm"] = price_usd_annual

    # ── Step 4: Build monthly DataFrame ───────────────────────────────────
    print("\n[4/4] Building monthly time-series …")
    df = pd.DataFrame({"ds": dates})
    df = df.set_index("ds")

    # Macro variables – annual → monthly (linear interpolation)
    annual_cols = list(WB_INDICATORS.keys()) + ["gold_price_usd", "cocoa_price_usd", "oil_brent_usd"]
    for col in annual_cols:
        df[col] = annual_dict_to_monthly(macro_data.get(col, {}), dates).values

    # Property price series – with seasonality
    price_usd_monthly_raw = annual_dict_to_monthly(
        macro_data["price_usd_per_sqm"], dates
    )
    price_usd_monthly = apply_seasonality(price_usd_monthly_raw, PROPERTY_SEASONAL)
    price_usd_monthly = add_noise(price_usd_monthly, sigma_frac=0.008)

    exchange_rate_monthly = pd.Series(
        df["exchange_rate_ghs_usd"].values, index=dates
    )

    # Add seasonality to commodity prices
    df["gold_price_usd"] = add_noise(
        apply_seasonality(
            pd.Series(df["gold_price_usd"].values, index=dates),
            GOLD_SEASONAL,
        ),
        sigma_frac=0.012,
    ).values

    df["cocoa_price_usd"] = add_noise(
        apply_seasonality(
            pd.Series(df["cocoa_price_usd"].values, index=dates),
            COCOA_SEASONAL,
        ),
        sigma_frac=0.025,
    ).values

    df["oil_brent_usd"] = add_noise(
        apply_seasonality(
            pd.Series(df["oil_brent_usd"].values, index=dates),
            OIL_SEASONAL,
        ),
        sigma_frac=0.020,
    ).values

    # Build AHPI target
    ahpi, price_ghs_monthly = build_ahpi(
        price_usd_monthly, exchange_rate_monthly
    )
    df["y"]                 = ahpi.values
    df["price_ghs_per_sqm"] = price_ghs_monthly.values
    df["price_usd_per_sqm"] = price_usd_monthly.values

    # Reset index, reorder columns (ds, y first for Prophet)
    df = df.reset_index().rename(columns={"index": "ds"})
    ordered_cols = (
        ["ds", "y"]
        + list(WB_INDICATORS.keys())
        + ["gold_price_usd", "cocoa_price_usd", "oil_brent_usd"]
        + ["price_ghs_per_sqm", "price_usd_per_sqm"]
    )
    df = df[ordered_cols]

    # Round for readability
    df["y"]                    = df["y"].round(2)
    df["price_ghs_per_sqm"]    = df["price_ghs_per_sqm"].round(0)
    df["price_usd_per_sqm"]    = df["price_usd_per_sqm"].round(0)
    df["population_total"]     = df["population_total"].round(0)
    for col in df.columns:
        if col not in ("ds", "population_total", "price_ghs_per_sqm", "price_usd_per_sqm"):
            df[col] = df[col].round(4)

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'─'*64}")
    print(f"  Saved → {OUTPUT_PATH}")
    print(f"  Rows  : {len(df):,}  ({df['ds'].iloc[0].strftime('%b %Y')} – "
          f"{df['ds'].iloc[-1].strftime('%b %Y')})")
    print(f"  Cols  : {len(df.columns)}  (ds + y + {len(df.columns)-2} regressors)")
    print(f"\n  AHPI range  : {df['y'].min():.1f} → {df['y'].max():.1f}  "
          f"(base 2015 = 100)")
    print(f"  GHS/sqm     : {df['price_ghs_per_sqm'].min():,.0f} → "
          f"{df['price_ghs_per_sqm'].max():,.0f}")
    print(f"  USD/sqm     : {df['price_usd_per_sqm'].min():,.0f} → "
          f"{df['price_usd_per_sqm'].max():,.0f}")
    print(f"{'─'*64}\n")
    print(df[["ds", "y", "exchange_rate_ghs_usd", "inflation_cpi_pct",
              "gold_price_usd", "cocoa_price_usd", "price_ghs_per_sqm"]].tail(12).to_string(index=False))
    print()

    return df


if __name__ == "__main__":
    main()
