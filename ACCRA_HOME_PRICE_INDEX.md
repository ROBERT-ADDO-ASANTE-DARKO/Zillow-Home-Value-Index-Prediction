# Accra Home Price Index (AHPI)
### A Monthly Residential Property Index for Accra, Ghana · Jan 2010 – Dec 2024

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What is the AHPI?](#2-what-is-the-ahpi)
3. [Methodology](#3-methodology)
4. [Geographic Coverage](#4-geographic-coverage)
5. [Dataset Structure](#5-dataset-structure)
6. [Key Findings & Trends](#6-key-findings--trends)
7. [Regressor Correlations](#7-regressor-correlations)
8. [Stakeholders & Implications](#8-stakeholders--implications)
9. [Limitations & Caveats](#9-limitations--caveats)
10. [Using AHPI with Facebook Prophet](#10-using-ahpi-with-facebook-prophet)
11. [Data Sources](#11-data-sources)

---

## 1. Executive Summary

The **Accra Home Price Index (AHPI)** is a monthly time-series index that tracks the nominal value of mid-market residential property in Greater Accra, Ghana, expressed in **Ghanaian Cedis (GHS)** and normalised to a base of **100 in 2015**.

Between January 2010 and December 2024, the AHPI rose from **29.9 to 406.9** — an increase of **+1,261%** in cedi terms. In US Dollar terms, the same properties rose only **+34%** (from ~USD 733/sqm to ~USD 984/sqm), revealing that the overwhelming driver of nominal cedi price growth is **currency depreciation and inflation**, not a fundamental appreciation in real property values.

The dataset contains **180 monthly observations across 22 columns** — one target variable (`y`) and **20 regressors** drawn from macroeconomic, demographic, fiscal, and commodity-price data — making it directly suitable for training a [Facebook Prophet](https://facebook.github.io/prophet/) forecasting model with additional regressors.

---

## 2. What is the AHPI?

The AHPI is modelled conceptually after the [Zillow Home Value Index (ZHVI)](https://www.zillow.com/research/data/) and the US [Case-Shiller Home Price Index](https://www.spglobal.com/spdji/en/index-family/real-estate/sp-corelogic-case-shiller/), adapted to the data realities of Ghana's residential property market, where no official government or central-bank index exists.

| Attribute | Value |
|-----------|-------|
| **Target variable** | `y` — AHPI index level |
| **Currency** | Ghanaian Cedi (GHS) |
| **Base year** | 2015 = 100 |
| **Frequency** | Monthly (first of each month) |
| **Coverage** | Jan 2010 – Dec 2024 (180 months) |
| **Market segment** | Mid-market residential (not prime/luxury) |
| **Geographic scope** | Greater Accra metropolitan area |

Unlike the ZHVI, which aggregates millions of automated valuations, the AHPI is constructed from **annual price-anchor data points** (USD/sqm) sourced from published real estate research, converted to GHS, and interpolated to monthly frequency. It is an **estimated index**, not a transaction-based one, and should be interpreted accordingly.

---

## 3. Methodology

### 3.1 Price Anchors

Annual median transaction prices in **USD per square metre** for mid-market residential property in Accra were sourced from:

- **Global Property Guide** — Ghana country reports (2010–2024)
- **Numbeo** — Accra property investment page (price/sqm, price-to-rent ratio)
- **JLL Africa Real Estate Outlook** — annual Accra market reports
- **Knight Frank Africa** — residential market surveys
- **Broll Ghana** — research notes on Greater Accra

These anchors are placed at the **mid-year (July 1)** of each year and represent the average across the five mid-market districts listed in [Section 4](#4-geographic-coverage).

### 3.2 GHS Conversion

USD anchors are converted to **Ghanaian Cedis** using the **official annual average exchange rate** sourced from the World Bank (`PA.NUS.FCRF`) and the Bank of Ghana's monetary policy reports.

```
GHS_price_t = USD_price_t × exchange_rate_GHS_per_USD_t
```

This step is critical: Ghana's cedi depreciated **914%** against the dollar between 2010 and 2024 (from 1.43 to 14.50 GHS/USD), which is the single largest driver of AHPI growth in nominal cedi terms.

### 3.3 Index Normalisation

The AHPI is normalised so that the **2015 annual average GHS price = 100**:

```
AHPI_t = (GHS_price_t / mean(GHS_price_2015)) × 100
```

2015 was chosen because it sits in the middle of the series, after the first major cedi crisis (2014) had stabilised but before the second wave (2022).

### 3.4 Monthly Interpolation

Annual mid-year anchors are interpolated to monthly frequency using **linear time interpolation** via `pandas.Series.interpolate(method="time")`. Edge months (Jan 2010 and Dec 2024) are forward/backward filled from the nearest anchor.

### 3.5 Seasonal Adjustment

A **12-month multiplicative seasonal factor** is applied to the property price series to capture within-year buying patterns observed in Accra's market:

| Period | Pattern | Driver |
|--------|---------|--------|
| Jul–Sep | **+2.0–2.5% above trend** | Peak buying season; professionals relocating before the school year |
| Jan–Mar | **−2.0–2.5% below trend** | Post-holiday spending fatigue; tighter liquidity |
| Oct–Dec | Near trend | Moderate activity; year-end decisions |

Similar seasonal factors are applied to gold, cocoa, and oil prices to reflect known commodity-market seasonality.

---

## 4. Geographic Coverage

### 4.1 Mid-Market Districts Covered by the AHPI

The index represents the **mid-market residential segment** across five districts of Greater Accra:

| District | Location | Character | Typical Price Range (2024) |
|----------|----------|-----------|---------------------------|
| **Spintex Road** | East Accra | Fast-growing corridor; gated estates, mixed residential-commercial | GHS 10,000–18,000/sqm |
| **Tema** | 30 km east | Ghana's planned industrial port city; large self-contained communities | GHS 8,000–14,000/sqm |
| **Dome** | North Accra | Dense residential suburb on the Accra–Kumasi road | GHS 7,000–12,000/sqm |
| **Adenta** | East Accra | Established municipality; good road access, family estates | GHS 9,000–15,000/sqm |
| **Kasoa** | Peri-urban west | Rapid expansion beyond Central Region border; most affordable | GHS 5,000–10,000/sqm |

### 4.2 Prime Districts Excluded from the AHPI

The following **prime and upper-income areas** are explicitly outside the index scope. Their pricing dynamics are dominated by USD-denominated leases, expatriate demand, and embassy tenancies, which would distort a general residential benchmark:

| District | Why Excluded | Estimated Price (2024) |
|----------|-------------|----------------------|
| **Cantonments** | Diplomatic quarter; embassy compounds | USD 2,500–4,500/sqm |
| **East Legon** | Accra's most prestigious suburb | USD 2,000–4,000/sqm |
| **Airport Residential** | Proximity premium to Kotoka International Airport | USD 1,800–3,500/sqm |
| **Labone / Roman Ridge** | Historic upscale enclaves; very limited supply | USD 1,500–3,000/sqm |
| **Dzorwulu / Abelenkpe** | Professional-class suburb; corporate housing | USD 1,500–2,800/sqm |
| **Trasacco Valley** | Luxury gated estates | USD 3,000–6,000/sqm |

> **Note:** A separate prime-market index for these districts would behave very differently — more correlated with gold prices and FDI inflows, less sensitive to domestic CPI — and would require its own anchors and methodology.

---

## 5. Dataset Structure

**File:** `data/accra_home_price_index.csv`
**Shape:** 180 rows × 22 columns · Zero null values

### 5.1 Column Reference

| Column | Unit | Role | Source |
|--------|------|------|--------|
| `ds` | Date (YYYY-MM-DD) | Prophet date column | — |
| `y` | Index (2015 = 100) | **Target variable** | Constructed (see §3) |
| `gdp_growth_pct` | % | Regressor | World Bank `NY.GDP.MKTP.KD.ZG` |
| `gdp_per_capita_usd` | USD | Regressor | World Bank `NY.GDP.PCAP.CD` |
| `cpi_index` | Index (2010 = 100) | Regressor | World Bank `FP.CPI.TOTL` |
| `inflation_cpi_pct` | % YoY | Regressor | World Bank `FP.CPI.TOTL.ZG` |
| `exchange_rate_ghs_usd` | GHS per USD | Regressor | World Bank `PA.NUS.FCRF` / Bank of Ghana |
| `lending_rate_pct` | % | Regressor | World Bank `FR.INR.LEND` |
| `unemployment_pct` | % | Regressor | World Bank `SL.UEM.TOTL.ZS` |
| `urban_pop_pct` | % of total | Regressor | World Bank `SP.URB.TOTL.IN.ZS` |
| `population_total` | Persons | Regressor | World Bank `SP.POP.TOTL` |
| `remittances_pct_gdp` | % of GDP | Regressor | World Bank `BX.TRF.PWKR.DT.GD.ZS` |
| `fdi_pct_gdp` | % of GDP | Regressor | World Bank `BX.KLT.DINV.WD.GD.ZS` |
| `credit_private_pct_gdp` | % of GDP | Regressor | World Bank `FS.AST.PRVT.GD.ZS` |
| `gross_capital_form_pct` | % of GDP | Regressor | World Bank `NE.GDI.TOTL.ZS` |
| `govt_debt_pct_gdp` | % of GDP | Regressor | World Bank `GC.DOD.TOTL.GD.ZS` |
| `broad_money_pct_gdp` | % of GDP | Regressor | World Bank `FM.LBL.BMNY.GD.ZS` |
| `gold_price_usd` | USD/troy oz | Regressor | LBMA / FRED `GOLDAMGBD228NLBM` |
| `cocoa_price_usd` | USD/MT | Regressor | ICCO / IMF commodity database |
| `oil_brent_usd` | USD/bbl | Regressor | EIA/Platts / FRED `DCOILBRENTEU` |
| `price_ghs_per_sqm` | GHS | Derived | Constructed |
| `price_usd_per_sqm` | USD | Derived | Constructed |

### 5.2 Summary Statistics

| Metric | Min | Mean | Median | Max |
|--------|-----|------|--------|-----|
| **AHPI (y)** | 29.7 | 130.9 | 108.5 | 426.2 |
| **GHS/sqm** | 1,040 | 4,592 | 3,804 | 14,947 |
| **USD/sqm** | 727 | 896 | 882 | 1,121 |
| **GHS/USD rate** | 1.43 | 5.11 | 4.32 | 14.50 |
| **CPI inflation %** | 7.1 | 15.5 | 11.9 | 40.2 |
| **Gold (USD/oz)** | 1,144 | 1,535 | 1,437 | 2,325 |
| **Cocoa (USD/MT)** | 1,967 | 2,974 | 2,447 | 7,757 |

---

## 6. Key Findings & Trends

### 6.1 The Dual-Currency Story

The most important insight from the AHPI is the **divergence between GHS and USD prices**:

| Metric | Jan 2010 | Dec 2024 | Change |
|--------|----------|----------|--------|
| AHPI (GHS index) | 29.9 | 406.9 | **+1,261%** |
| GHS/sqm | 1,048 | 14,272 | **+1,262%** |
| USD/sqm | 733 | 984 | **+34%** |
| GHS/USD exchange rate | 1.43 | 14.50 | **+914%** |
| CPI index (2010 = 100) | 100 | 757.8 | **+658%** |

In hard currency, Accra's mid-market property appreciated by only **34% over 14 years** (~2.1% per annum in USD) — a modest return that barely keeps pace with US inflation. In local currency, the same properties appear to have gained 12× in value, but this is almost entirely attributable to the cedi's structural depreciation.

### 6.2 Four Distinct Eras

**Era 1 — Oil Boom Optimism (2010–2013):**
Ghana's first offshore oil came online (Jubilee field, 2010), triggering 14% GDP growth in 2011. AHPI climbed from 30 to 65. USD prices rose steadily as foreign investors and returning diaspora increased demand in Spintex and Tema.

**Era 2 — The First Cedi Crisis (2014–2016):**
The cedi lost ~30% of its value in a single year (2014), inflation surged to 15–17%, and lending rates reached 31%. AHPI breached 100 in 2015 purely through GHS conversion effects. USD prices actually declined slightly as buyers pulled back. Ghana entered an IMF Extended Credit Facility in 2015 (USD 918 million).

**Era 3 — Stabilisation & Steady Growth (2017–2021):**
Improved fiscal discipline, the Petroleum Revenue Management Act, and strong cocoa and gold export revenues supported the cedi. GDP growth recovered to 8.1% in 2017. AHPI growth was moderate (100 → 140), and USD prices held in the USD 850–900/sqm range.

**Era 4 — Debt Crisis & Currency Collapse (2022–2024):**
Ghana's public debt reached 92.4% of GDP by 2022. The government defaulted on Eurobond coupon payments in December 2022, triggering an IMF bailout of USD 3 billion in May 2023. The cedi collapsed from 5.81 GHS/USD (2021) to 14.50 GHS/USD (2024). Annual CPI peaked at **54% in December 2022**. The AHPI surged from 145 to 426 in two years — an index move driven almost entirely by depreciation, not real demand.

### 6.3 Commodity-Cycle Linkages

Ghana's fiscal health is tightly coupled to commodity export revenues, which in turn affect the cedi, government spending, and purchasing power:

- **Gold:** Ghana is Africa's largest gold producer. The gold price rally from USD 1,200/oz (2010) to USD 2,300/oz (2024) boosted government revenues, strengthened reserves, and supported diaspora purchasing power — all positive for property demand.
- **Cocoa:** Ghana is the world's 2nd-largest cocoa producer. The 2024 supply shock (crop disease across West Africa) pushed cocoa to USD 7,500+/MT — a USD 2+ billion annual revenue windfall that should stabilise the cedi and support purchasing power in the medium term.
- **Oil:** The 2014–16 oil price crash (from USD 111/bbl to USD 44/bbl) directly precipitated the first cedi crisis, as oil accounted for ~30% of Ghana's export revenues at the time.

---

## 7. Regressor Correlations

Pearson correlations between each regressor and the AHPI (`y`), computed over all 180 monthly observations:

### Positive correlations (AHPI rises with these)

| Regressor | ρ | Interpretation |
|-----------|---|----------------|
| `exchange_rate_ghs_usd` | **+0.991** | Exchange rate depreciation is the primary driver of nominal GHS price growth |
| `cpi_index` | **+0.986** | Inflation passes through directly into construction costs and asking prices |
| `urban_pop_pct` | **+0.871** | Long-run urbanisation structurally raises housing demand in Accra |
| `broad_money_pct_gdp` | **+0.813** | Monetary expansion fuels inflation and cedi depreciation |
| `gold_price_usd` | **+0.786** | Higher gold revenues support the fiscal position and diaspora remittances |
| `cocoa_price_usd` | **+0.764** | Cocoa export revenues drive fiscal space and cedi stability |
| `inflation_cpi_pct` | **+0.721** | Annual inflation rate co-moves with property price changes |
| `gdp_per_capita_usd` | **+0.699** | Rising incomes (in USD) support real demand |
| `govt_debt_pct_gdp` | **+0.684** | Debt crises have coincided with currency depreciation episodes |

### Negative correlations (AHPI falls when these are high)

| Regressor | ρ | Interpretation |
|-----------|---|----------------|
| `fdi_pct_gdp` | **−0.744** | FDI flows weaken during macro crises — the same periods that AHPI nominally surges |
| `credit_private_pct_gdp` | **−0.723** | Credit to the private sector contracts as lending rates rise in crisis |
| `gross_capital_form_pct` | **−0.648** | Investment falls as the economy deteriorates; same periods AHPI nominally spikes |
| `gdp_growth_pct` | **−0.462** | Real growth tends to be low in the depreciation episodes that inflate the index |
| `remittances_pct_gdp` | **−0.445** | Remittances as a share of GDP fall in boom years when GDP grows faster |

> **Key insight:** The negative correlations do not mean FDI or credit growth lowers property prices in real life. Rather, they reflect the **regime effect**: the index spikes during macro crises driven by depreciation, and these are precisely the periods when FDI, credit, and real growth are weakest. A model using these regressors should account for this structural non-linearity.

---

## 8. Stakeholders & Implications

### 8.1 Real Estate Developers & Construction Companies

**What the AHPI tells them:**
The 14× nominal GHS rise in prices masks near-flat real (USD) appreciation. Construction material costs are largely dollar-linked (cement, steel, glass), which means **building in Ghana today is more expensive in GHS terms than ever**, yet the USD sales price has not kept pace. Developers face a cost-price squeeze in hard-currency terms.

**Actionable insight:**
- Sell and receive payments in USD or index contracts to the exchange rate
- Time new projects during cedi stability windows (periods of strong gold/cocoa prices)
- Prioritise affordable mid-market segments in Kasoa and Dome where land costs are lower

---

### 8.2 Commercial and Retail Banks / Mortgage Lenders

**What the AHPI tells them:**
With lending rates at 25–31% and AHPI growing at a similar nominal rate, **mortgage affordability is structurally broken** for most Ghanaian households. The 2022–2024 AHPI spike was driven by depreciation, not income growth — meaning collateral values have risen nominally but buyers' repayment capacity has not.

**Actionable insight:**
- Stress-test mortgage books against further cedi depreciation scenarios
- Develop foreign-currency mortgage products for diaspora buyers
- Use the AHPI forecast as a leading indicator for non-performing loan risk in the housing portfolio

---

### 8.3 Diaspora Investors (Ghanaians Abroad)

**What the AHPI tells them:**
In USD terms, Accra mid-market property has appreciated only **+34% over 14 years** — a weak return compared to equities or even savings rates in developed markets. However, **purchasing in USD while rents and resale values are denominated in GHS** gives diaspora buyers natural exposure to the cedi's depreciation recovery. If the cedi stabilises (as cocoa revenues suggest it may), USD-denominated buyers will benefit disproportionately.

**Actionable insight:**
- Buy during cedi weakness (high AHPI phases) for best entry in USD terms
- Spintex Road and Adenta offer the best combination of liquidity and yield
- Monitor `remittances_pct_gdp` and `gold_price_usd` as forward indicators

---

### 8.4 Institutional Investors & Private Equity (Real Estate Funds)

**What the AHPI tells them:**
The near-zero real return (+34% USD in 14 years) suggests Accra mid-market residential is not a strong capital appreciation play for USD-denominated funds. The value proposition lies in **rental yield** (typically 8–12% gross in prime areas, 6–9% in mid-market) rather than price appreciation, and in **development margins** when construction costs are managed.

**Actionable insight:**
- Use the AHPI forecast to plan fund-entry and exit timing relative to macro cycles
- Hedge GHS exposure through currency swaps or USD-indexed leases
- Track the `fdi_pct_gdp` and `credit_private_pct_gdp` regressors as leading indicators of investment climate

---

### 8.5 Government & Housing Policy Makers (Ministry of Works & Housing, NHC)

**What the AHPI tells them:**
Ghana faces a **housing deficit of approximately 1.8 million units** (Ghana Statistical Service), with affordability worsening each year as nominal prices outpace wage growth. The AHPI shows that **the problem is not insufficient construction activity — it is currency instability making construction inputs unaffordable** for low- and middle-income households.

**Actionable insight:**
- Index social housing subsidy amounts to the AHPI to prevent real-value erosion
- Use AHPI forecasts to calibrate the National Housing Fund's mortgage subsidy rates
- Target Kasoa and Tema for mass housing programmes where land costs are lowest
- Implement exchange-rate stabilisation as a housing affordability measure

---

### 8.6 Bank of Ghana (Monetary Policy Committee)

**What the AHPI tells them:**
The 0.991 correlation between the exchange rate and the AHPI is a direct transmission mechanism: **cedi depreciation immediately inflates the GHS price of housing**, which feeds into shelter components of the CPI, creating a self-reinforcing inflation spiral. Housing is not a passive bystander in Ghana's inflation problem.

**Actionable insight:**
- Incorporate AHPI trends into the monetary policy transmission analysis
- Use the AHPI as a near-real-time proxy for shelter inflation between formal CPI releases
- Monitor `broad_money_pct_gdp` (+0.813 correlation) as a leading indicator of AHPI acceleration

---

### 8.7 Researchers & Data Scientists (Prophet Forecasting)

**What the AHPI tells them:**
The dataset is structured for immediate use with **Facebook Prophet** (`ds`, `y`, and 20 additional regressors). The dominant regressors by correlation are `exchange_rate_ghs_usd`, `cpi_index`, and `urban_pop_pct`. The series contains **two structural breaks** (2014 cedi crisis; 2022 debt crisis) that Prophet's changepoint detection should identify automatically, but manual changepoints at `2014-01-01` and `2022-01-01` are recommended for robustness.

**Actionable insight:**
- Standardise regressors before fitting (wide variance ranges across columns)
- Add Prophet changepoints: `changepoints=['2014-01-01', '2022-01-01']`
- Use `add_regressor('exchange_rate_ghs_usd')` as the primary external driver
- Consider a log-transformation of `y` to stabilise variance in the 2022–2024 spike

---

### 8.8 Insurance Companies (Property & Casualty Underwriters)

**What the AHPI tells them:**
Sum-insured values for property policies denominated in GHS will significantly understate replacement cost if not updated annually against the AHPI. The 1,261% GHS appreciation over 14 years means a policy written in 2010 at a GHS-denominated value would cover only **8% of today's replacement cost** without adjustment.

**Actionable insight:**
- Embed automatic AHPI-linked indexation clauses in long-term property policies
- Use AHPI forecasts to anticipate claims inflation in major depreciation years
- Require USD-denominated reinstatement values for commercial and high-value residential policies

---

## 9. Limitations & Caveats

| Limitation | Detail |
|------------|--------|
| **No transaction database** | The AHPI is based on surveyed price estimates, not actual recorded transactions. Ghana has no mandatory public deed price register. |
| **Sparse anchor data** | Price anchors are annual point-in-time estimates, not monthly observations. Monthly values between anchors are interpolated, not observed. |
| **Mid-market only** | The index does not represent prime areas (East Legon, Cantonments) or very low-income areas (Ashaiman, Nima). Separate indices are needed for those segments. |
| **Annual macro data** | Most World Bank indicators are reported annually. Monthly values are linearly interpolated, which smooths within-year volatility that undoubtedly exists. |
| **Informal market excluded** | A substantial fraction of Accra's housing transactions are informal (no title, no agent). These transactions are not captured. |
| **Single city** | The AHPI covers Greater Accra only. Kumasi, Takoradi, and other Ghanaian cities would require their own indices. |
| **GHS nominal, not real** | The index is not inflation-adjusted. An AHPI deflated by CPI would show near-zero real growth, which is economically accurate but less useful for nominal forecasting. |

---

## 10. Using AHPI with Facebook Prophet

### Minimal working example

```python
from prophet import Prophet
import pandas as pd

df = pd.read_csv("data/accra_home_price_index.csv", parse_dates=["ds"])

# Recommended: add changepoints at structural break dates
m = Prophet(
    changepoints=["2014-01-01", "2022-01-01"],
    changepoint_prior_scale=0.5,
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
)

# Primary regressors (highest correlation with y)
for reg in ["exchange_rate_ghs_usd", "cpi_index", "gold_price_usd",
            "cocoa_price_usd", "urban_pop_pct", "broad_money_pct_gdp"]:
    m.add_regressor(reg)

m.fit(df)

# Future dataframe requires future regressor values (scenario-based)
future = m.make_future_dataframe(periods=24, freq="MS")
# ... populate future regressors with economic scenarios ...
forecast = m.predict(future)
```

### Recommended preprocessing

```python
from sklearn.preprocessing import StandardScaler

regressors = [c for c in df.columns if c not in ("ds", "y")]
scaler = StandardScaler()
df[regressors] = scaler.fit_transform(df[regressors])
```

### Suggested scenario regressors for 2025–2026 forecasting

| Scenario | `exchange_rate_ghs_usd` | `inflation_cpi_pct` | `gold_price_usd` |
|----------|------------------------|--------------------|--------------------|
| **Bear** (continued depreciation) | 18–22 | 28–35% | 1,800–2,000 |
| **Base** (gradual stabilisation) | 14–16 | 18–22% | 2,100–2,400 |
| **Bull** (cedi recovery on cocoa windfall) | 11–13 | 12–16% | 2,300–2,600 |

---

## 11. Data Sources

| Source | Data Provided | Access |
|--------|--------------|--------|
| **World Bank Open Data** | 15 Ghana macroeconomic indicators (2010–2024) | Free API · `api.worldbank.org/v2` |
| **Bank of Ghana** | Exchange rate, monetary policy rate, lending rates | `bog.gov.gh` (PDF reports) |
| **Ghana Statistical Service** | Population, urban statistics | `statsghana.gov.gh` |
| **LBMA (London Bullion Market Association)** | Monthly gold spot prices | FRED `GOLDAMGBD228NLBM` |
| **ICCO (Int'l Cocoa Organization)** | Monthly cocoa prices | `icco.org` / IMF commodity database |
| **EIA / Platts** | Monthly Brent crude oil prices | FRED `DCOILBRENTEU` |
| **Global Property Guide** | Ghana residential price history (USD/sqm) | `globalpropertyguide.com/africa/ghana` |
| **Numbeo** | Accra property price/sqm, price-to-rent ratio | `numbeo.com/property-investment/in/Accra-Ghana` |
| **JLL Africa** | Annual Accra real estate outlook reports | `jll.co.za` |
| **Knight Frank Africa** | Residential market surveys | `knightfrank.com/research` |
| **Broll Ghana** | Greater Accra research notes | `broll.com/ghana` |

---

*Document generated: March 2026 · Dataset version: 1.0 · Index base year: 2015 = 100*
*AHPI is an estimated index for research and forecasting purposes. It should not be used as the sole basis for individual investment decisions.*
