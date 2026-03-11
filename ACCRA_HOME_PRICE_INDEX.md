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
7. [Per-District Analysis](#7-per-district-analysis)
8. [Prime Areas Index](#8-prime-areas-index)
9. [Regressor Correlations](#9-regressor-correlations)
10. [Stakeholders & Implications](#10-stakeholders--implications)
11. [Limitations & Caveats](#11-limitations--caveats)
12. [Prophet Forecasting: Implementation & Results](#12-prophet-forecasting-implementation--results)
13. [Data Sources](#13-data-sources)

---

## 1. Executive Summary

The **Accra Home Price Index (AHPI)** is a suite of monthly time-series indices that track the nominal value of residential property in Greater Accra, Ghana, expressed in **Ghanaian Cedis (GHS)** and normalised to a base of **100 in 2015**. The suite comprises three datasets:

| Dataset | File | Scope | Rows |
|---------|------|-------|------|
| **Aggregate mid-market** | `accra_home_price_index.csv` | 5-district composite + 20 macro regressors | 180 |
| **Per-district mid-market** | `accra_district_prices.csv` | 5 districts individually | 900 |
| **Prime areas** | `accra_prime_prices.csv` | 6 prime/luxury locations | 1,080 |

### Mid-Market (Aggregate)
Between January 2010 and December 2024, the composite AHPI rose from **29.9 to 419.7** — an increase of **+1,303%** in cedi terms. In US Dollar terms, the same properties rose only **+34%** (from ~USD 733/sqm to ~USD 985/sqm), revealing that the overwhelming driver of nominal cedi price growth is **currency depreciation and inflation**, not a fundamental appreciation in real property values.

### Per-District Divergence
District-level indices reveal meaningful within-market heterogeneity. **Kasoa** recorded the strongest nominal GHS growth (+1,762%) and the highest USD appreciation (+88%), reflecting rapid peri-urban expansion. **Dome** was the most stable (+1,145% GHS, +26% USD). December 2024 AHPI levels range from **389.6** (Dome) to **489.1** (Kasoa).

### Prime Areas
The six prime areas — East Legon, Cantonments, Airport Residential, Labone/Roman Ridge, Dzorwulu/Abelenkpe, and Trasacco Valley — tell a fundamentally different story. These markets are **USD-indexed**, meaning real (USD) appreciation is substantial: average USD/sqm grew from **USD 866** (Jan 2010) to **USD 2,874** (Dec 2024), a **+232% real gain** compared to mid-market's +34%. December 2024 prime AHPI averaged **776.3** (range: 655.2 to 851.5), approximately **1.8× the mid-market composite**.

The aggregate mid-market dataset contains **180 monthly observations across 22 columns** — one target variable (`y`) and **20 regressors** drawn from macroeconomic, demographic, fiscal, and commodity-price data — making it directly suitable for training a [Facebook Prophet](https://facebook.github.io/prophet/) forecasting model with additional regressors.

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
| **Market segment** | Mid-market residential (composite); per-district; prime/luxury |
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

### 3.6 Per-District Methodology

District-level indices are derived from the composite aggregate using a **time-varying multiplier** applied to the shared composite USD/sqm anchor:

```
district_usd_sqm(t) = composite_usd_sqm(t) × mult(t)

mult(t) = base_mult + (year(t) − 2010 + month_fraction(t)) × annual_drift
```

Each district's `(base_mult, annual_drift)` pair captures both its relative price level in 2010 and the trajectory of that premium or discount over time:

| District | `base_mult` | `annual_drift` | Interpretation |
|----------|-------------|----------------|----------------|
| Spintex Road | 1.28 | +0.008 | Above composite and widening — rising corridor premium |
| Adenta | 1.15 | +0.001 | Above composite, nearly flat premium — stable upper-mid suburb |
| Tema | 0.98 | −0.002 | Near composite, slight discount widening — industrial city moderating |
| Dome | 0.86 | −0.004 | Below composite, discount widening — densifying northern suburb |
| Kasoa | 0.53 | +0.013 | Deep discount in 2010, fastest closing — peri-urban catch-up story |

The resulting USD/sqm series for each district is then passed through the same **seasonality**, **noise**, **GHS conversion**, and **AHPI normalisation** (each district normalised to its own 2015 average = 100) steps as the aggregate index.

### 3.7 Prime Areas Methodology

Prime areas are **independently anchored** rather than multiplier-derived. Each area has its own annual USD/sqm time series sourced from published luxury market research (JLL, Knight Frank, Global Property Guide premium-segment reports). These anchors reflect the USD-denominated nature of prime Accra real estate, where asking prices, leases, and transactions are routinely quoted in US Dollars.

```python
# Linear interpolation of annual anchors to monthly frequency (July 1 anchor placement)
monthly_usd = annual_dict_to_monthly(annual_anchors_dict)
```

A flatter seasonal pattern (±1.0% peak-to-trough vs ±2.5% for mid-market) is applied, reflecting the more liquid and internationally connected nature of prime-area demand. Each area is normalised to its own 2015 average = 100.

---

## 4. Geographic Coverage

### 4.1 Mid-Market Districts

The aggregate AHPI represents the **mid-market residential segment** across five districts of Greater Accra. Individual district indices are available in `accra_district_prices.csv`.

| District | Location | Character | Dec 2024 AHPI | Dec 2024 GHS/sqm | Dec 2024 USD/sqm | USD growth 2010–24 |
|----------|----------|-----------|:-------------:|:----------------:|:----------------:|:-----------------:|
| **Spintex Road** | East Accra | Fast-growing corridor; gated estates, mixed residential-commercial | 429.1 | 19,482 | 1,374 | +48% |
| **Adenta** | East Accra | Established municipality; good road access, family estates | 398.1 | 15,888 | 1,120 | +34% |
| **Tema** | 30 km east | Ghana's planned industrial port city; large self-contained communities | 392.6 | 13,090 | 923 | +29% |
| **Dome** | North Accra | Dense residential suburb on the Accra–Kumasi road | 389.6 | 11,211 | 791 | +26% |
| **Kasoa** | Peri-urban west | Rapid expansion beyond Central Region border; most affordable | 489.1 | 10,156 | 716 | +88% |

> **Kasoa** stands out as both the most affordable (lowest absolute GHS/sqm) and the fastest-growing in real USD terms (+88%), driven by rapid peri-urban expansion and infrastructure investment closing the discount gap with the established suburbs.

### 4.2 Prime Areas

The six prime and upper-income areas were originally outside the mid-market index scope. They now have their own dedicated index (`accra_prime_prices.csv`), as their pricing dynamics — USD-denominated leases, expatriate demand, and embassy tenancies — differ fundamentally from the mid-market composite.

| Area | Character | Dec 2024 AHPI | Dec 2024 GHS/sqm | Dec 2024 USD/sqm | USD growth 2010–24 |
|------|-----------|:-------------:|:----------------:|:----------------:|:-----------------:|
| **East Legon** | Accra's most prestigious suburb; gated communities, embassies | 851.5 | 42,264 | 2,980 | +270% |
| **Cantonments** | Diplomatic quarter; embassy compounds and ministerial residences | 698.2 | 45,351 | 3,198 | +218% |
| **Airport Residential** | Proximity premium to Kotoka International Airport | 826.3 | 35,662 | 2,515 | +258% |
| **Labone / Roman Ridge** | Historic upscale enclave; very limited supply | 778.5 | 27,837 | 1,963 | +222% |
| **Dzorwulu / Abelenkpe** | Professional-class suburb; corporate and NGO housing | 847.9 | 29,753 | 2,098 | +264% |
| **Trasacco Valley** | Ultra-luxury gated estates; highest absolute price per sqm | 655.2 | 63,657 | 4,489 | +200% |

> **Prime vs mid-market:** The average prime AHPI (776.3) is **1.8× the mid-market composite** (419.7). In USD/sqm terms, the premium is even larger: prime areas average USD 2,874/sqm versus USD 985/sqm for mid-market — a **2.9× premium**. Unlike mid-market where USD appreciation was modest (+34%), prime areas delivered genuine real returns averaging **+232% in USD** over 14 years, driven by structural undersupply relative to expatriate and diplomatic demand.

---

## 5. Dataset Structure

### 5.0 Files Overview

| File | Shape | Description |
|------|-------|-------------|
| `data/accra_home_price_index.csv` | 180 × 22 | Aggregate mid-market composite with all 20 macro regressors. Primary Prophet training file. |
| `data/accra_district_prices.csv` | 900 × 5 | Per-district long-format. 5 districts × 180 months. Columns: `ds, district, y, price_ghs_per_sqm, price_usd_per_sqm` |
| `data/accra_prime_prices.csv` | 1,080 × 5 | Per-prime-area long-format. 6 areas × 180 months. Same schema as district file. |

### 5.1 Column Reference (Aggregate File)

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
| AHPI composite (GHS index) | 29.9 | 419.7 | **+1,303%** |
| GHS/sqm (mid-market avg) | 1,048 | 14,272 | **+1,262%** |
| USD/sqm (mid-market avg) | 733 | 985 | **+34%** |
| AHPI prime avg (GHS index) | 23.1 | 776.3 | **+3,264%** |
| USD/sqm (prime avg) | 866 | 2,874 | **+232%** |
| GHS/USD exchange rate | 1.43 | 14.50 | **+914%** |
| CPI index (2010 = 100) | 100 | 757.8 | **+658%** |

In hard currency, Accra's **mid-market** property appreciated by only **34% over 14 years** (~2.1% per annum in USD) — a modest return that barely keeps pace with US inflation. In local currency, the same properties appear to have gained 14× in value, but this is almost entirely attributable to the cedi's structural depreciation. **Prime areas** are the exception: genuine USD appreciation of +232% reflects true real demand from a structurally supply-constrained, internationally connected market.

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

## 7. Per-District Analysis

### 7.1 District AHPI Summary (Jan 2010 → Dec 2024)

| District | AHPI Jan 2010 | AHPI Dec 2024 | GHS change | USD/sqm Jan 2010 | USD/sqm Dec 2024 | USD change |
|----------|:-------------:|:-------------:|:----------:|:----------------:|:----------------:|:----------:|
| Spintex Road | 29.3 | 429.1 | +1,364% | 929 | 1,374 | +48% |
| Adenta | 29.9 | 398.1 | +1,230% | 834 | 1,120 | +34% |
| Tema | 30.8 | 392.6 | +1,176% | 713 | 923 | +29% |
| Dome | 31.3 | 389.6 | +1,145% | 628 | 791 | +26% |
| Kasoa | 26.3 | 489.1 | +1,762% | 382 | 716 | +88% |

### 7.2 District Divergence Themes

**Kasoa — The Peri-Urban Catch-Up Story**
Kasoa had the lowest absolute price level in 2010 (USD 382/sqm, a 48% discount to the composite) and the fastest USD appreciation (+88%). This reflects the district's transformation from a peri-urban fringe into a fully developed residential corridor as the Greater Accra Metropolitan Area expanded westward. The widening base multiplier (+0.013/year) captures this structural convergence. Affordable entry point combined with strong appreciation makes Kasoa the highest-return mid-market district over the period.

**Spintex Road — The Corridor Premium**
Spintex entered 2010 at a 28% premium to the composite (USD 929/sqm) and widened that premium further (+0.008/year). The East Accra corridor's proximity to the Airport City commercial hub, good road infrastructure, and concentration of gated communities sustained its premium positioning. At USD 1,374/sqm (Dec 2024), it is the most expensive mid-market district.

**Dome — The Stable Discount**
Dome's discount to the composite widened steadily (−0.004/year), reflecting its dense urban character with limited new supply of high-quality stock and increasing competition from newer developments in Adenta and Spintex. At USD 791/sqm, it offers the lowest entry price among the established suburbs.

**Tema — Industrial Moderation**
Tema's proximity to the port and industrial zone maintained its position near the composite average, but a slight downward drift (−0.002/year) reflects the area's industrial character moderating residential demand relative to the purely residential suburbs.

### 7.3 Within-Market Spread

The spread between the highest-AHPI (Kasoa, 489.1) and lowest-AHPI (Dome, 389.6) district at December 2024 is **99.5 index points** — approximately **25% of the composite**. This spread has widened over time: in January 2010, the range was only 5.0 index points (31.3 to 26.3). The increasing within-market dispersion reflects neighbourhood differentiation as the Accra metropolitan area matures.

---

## 8. Prime Areas Index

### 8.1 Prime AHPI Summary (Jan 2010 → Dec 2024)

All six prime areas share one defining characteristic absent from the mid-market: their prices are quoted, negotiated, and paid in **US Dollars**. GHS appreciation in these areas is primarily a mechanical product of cedi depreciation applied to a USD-anchored series, but the USD appreciation itself is genuine and substantial.

| Area | AHPI Jan 2010 | AHPI Dec 2024 | GHS change | USD/sqm Jan 2010 | USD/sqm Dec 2024 | USD change |
|------|:-------------:|:-------------:|:----------:|:----------------:|:----------------:|:----------:|
| East Legon | 23.2 | 851.5 | +3,573% | 805 | 2,980 | +270% |
| Cantonments | 22.1 | 698.2 | +3,055% | 1,005 | 3,198 | +218% |
| Airport Residential | 23.3 | 826.3 | +3,446% | 703 | 2,515 | +258% |
| Labone / Roman Ridge | 24.3 | 778.5 | +3,098% | 609 | 1,963 | +222% |
| Dzorwulu / Abelenkpe | 23.5 | 847.9 | +3,510% | 576 | 2,098 | +264% |
| Trasacco Valley | 22.0 | 655.2 | +2,877% | 1,496 | 4,489 | +200% |
| **Prime average** | **23.1** | **776.3** | **+3,264%** | **866** | **2,874** | **+232%** |

### 8.2 Area-by-Area Insights

**East Legon (AHPI 851.5 · USD 2,980/sqm)**
Accra's highest-profile residential address and the strongest index performer. The area has transformed from a prestige suburb into a fully mixed-use luxury zone with high-end retail, offices, and diplomatic missions. Demand is driven by senior government officials, business elites, and returning diaspora. The +270% USD appreciation is the highest among the six areas.

**Cantonments (AHPI 698.2 · USD 3,198/sqm)**
Despite the lowest AHPI among the six (reflecting a flatter percentage gain from a high 2010 base of USD 1,005/sqm), Cantonments commands the second-highest absolute USD price. Its character as Accra's diplomatic and ministerial quarter limits supply to an extremely small number of large-plot estates, sustaining an elevated price floor.

**Airport Residential (AHPI 826.3 · USD 2,515/sqm)**
Airport proximity creates consistent demand from international business travellers, airline crews, and multinational tenants seeking short-term furnished accommodation. The +258% USD appreciation reflects the area's capture of the Accra corporate-stay premium.

**Labone / Roman Ridge (AHPI 778.5 · USD 1,963/sqm)**
Among the more affordable prime areas in absolute USD terms, Labone and Roman Ridge are established enclaves valued for their mature street trees, spacious plots, and proximity to the Cantonments cluster without the full diplomatic premium. Moderate USD appreciation (+222%) reflects steady organic demand rather than rapid commercial development.

**Dzorwulu / Abelenkpe (AHPI 847.9 · USD 2,098/sqm)**
Second only to East Legon in AHPI terms (+264% USD), Dzorwulu serves a concentrated professional-class and NGO community. Its location between the major employment centres of Airport City and Cantonments sustains strong rental demand and a consistent price premium.

**Trasacco Valley (AHPI 655.2 · USD 4,489/sqm)**
Trasacco is in a category of its own: the highest absolute USD price per sqm of any area tracked (USD 4,489 vs USD 3,198 for second-place Cantonments), but the lowest AHPI among prime areas. This reflects the **exceptionally high 2010 base** (USD 1,496/sqm, already 1.7× the composite prime price) and the ultra-luxury, gated estate model that limits both supply and the number of comparable transactions. The +200% USD gain is "modest" only relative to the other prime areas — it still far outpaces mid-market.

### 8.3 Prime vs Mid-Market: Structural Comparison

| Dimension | Mid-Market (composite) | Prime (average) | Ratio |
|-----------|:---------------------:|:---------------:|:-----:|
| AHPI Jan 2010 | 29.9 | 23.1 | — |
| AHPI Dec 2024 | 419.7 | 776.3 | 1.8× |
| USD/sqm Jan 2010 | 733 | 866 | 1.2× |
| USD/sqm Dec 2024 | 985 | 2,874 | 2.9× |
| USD appreciation 2010–24 | +34% | +232% | 6.8× |
| GHS appreciation 2010–24 | +1,303% | +3,264% | 2.5× |
| Seasonal peak-to-trough | ±2.5% | ±1.0% | — |

The key structural difference is that **prime areas function as USD-denominated assets** held in a GHS-denominated country. Mid-market buyers transact in GHS and bear the full cedi depreciation risk. Prime buyers and sellers transact in USD; their GHS index appreciation is largely mechanical. Their **real** return (+232% USD) is genuine and reflects structural undersupply relative to international demand — a fundamentally different investment thesis.

---

## 9. Regressor Correlations

Pearson correlations below are computed against the **aggregate mid-market AHPI** (`y`) over all 180 monthly observations. Prime areas follow a broadly similar macro pattern but with stronger gold and FDI correlations and weaker CPI pass-through.

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

## 10. Stakeholders & Implications

### 10.1 Real Estate Developers & Construction Companies

**Mid-market (AHPI composite / district indices):**
The 14× nominal GHS rise in prices masks near-flat real (USD) appreciation (+34%). Construction material costs are largely dollar-linked (cement, steel, glass), which means **building in Ghana today is more expensive in GHS terms than ever**, yet the mid-market USD sale price has not kept pace. Developers face a cost-price squeeze in hard-currency terms.

**Prime areas (prime AHPI):**
The picture is more attractive: prime USD/sqm grew +232% over the same period, meaning sale prices have more than doubled in real terms. Developers who can control land in East Legon, Dzorwulu, or Airport Residential and sell at prime USD rates benefit from a margin structure unavailable in the mid-market.

**Actionable insight:**
- Mid-market: sell in USD or index contracts to the exchange rate; time projects during cedi stability windows; prioritise Kasoa and Dome where land costs remain low
- Prime: East Legon and Dzorwulu/Abelenkpe offer the best combination of appreciation (+264–270% USD) and transaction liquidity; Trasacco Valley's ultra-luxury model requires patient capital but commands the highest absolute price (USD 4,489/sqm)
- Use the per-district index (`accra_district_prices.csv`) to benchmark which district offers the strongest current margin relative to construction cost

---

### 10.2 Commercial and Retail Banks / Mortgage Lenders

**Mid-market:**
With lending rates at 25–31% and AHPI growing at a similar nominal rate, **mortgage affordability is structurally broken** for most Ghanaian households. The 2022–2024 AHPI spike was driven by depreciation, not income growth — meaning collateral values have risen nominally but buyers' repayment capacity has not.

**Prime areas:**
Prime property collateral is effectively USD-denominated. A GHS mortgage secured on a Cantonments property priced at USD 3,198/sqm carries a fundamentally different risk profile: cedi depreciation inflates the collateral's GHS value, cushioning LTV ratios even as it erodes the borrower's GHS repayment capacity. Banks should model these two segments separately.

**Actionable insight:**
- Stress-test mid-market mortgage books against further cedi depreciation scenarios using the composite AHPI forecast
- For prime-area lending, consider USD-indexed mortgage products — the prime AHPI's high gold correlation (+0.786 for mid-market; likely higher for prime) can inform hedging strategies
- Use the district AHPI to set location-specific LTV caps: Kasoa's strong USD appreciation (+88%) supports higher LTVs than Dome (+26%)
- Use AHPI forecasts as a leading indicator for non-performing loan risk in the housing portfolio

---

### 10.3 Diaspora Investors (Ghanaians Abroad)

**Mid-market:**
In USD terms, mid-market property appreciated only **+34% over 14 years** — a weak return. However, **purchasing in USD while rents and resale values are denominated in GHS** provides natural exposure to a cedi recovery. If the cedi stabilises on the back of the 2024 cocoa windfall, USD-based buyers benefit disproportionately.

**Prime areas:**
The prime index reframes the conversation entirely. East Legon (+270% USD), Airport Residential (+258%), and Dzorwulu (+264%) delivered returns competitive with developed-market equities. These are the segments diaspora investors with USD capital should be comparing to offshore alternatives — not the mid-market composite.

**Actionable insight:**
- For capital appreciation: target prime areas, particularly East Legon and Dzorwulu, where USD appreciation has been strongest and supply remains structurally constrained
- For yield + affordability: Spintex Road and Adenta offer the best mid-market combination of liquidity and rental income
- Entry timing: buy mid-market during high-AHPI (weak cedi) periods for best USD entry price; prime-market entry is less sensitive to the cedi cycle since pricing is USD-anchored
- Monitor `gold_price_usd` and `remittances_pct_gdp` as forward indicators for both segments

---

### 10.4 Institutional Investors & Private Equity (Real Estate Funds)

**Mid-market:**
The near-flat real return (+34% USD in 14 years) means mid-market residential is not a capital appreciation play for USD funds. Value lies in **development yield** (affordable housing gap) and **rental income** rather than price appreciation.

**Prime areas:**
The prime index changes the calculus for USD-denominated funds. A +232% real USD return over 14 years (~8.8% annualised in USD) is a compelling long-term capital appreciation story, especially for funds with 7–10 year horizons. Rental yields in prime areas (typically 8–12% gross) add further return on top of appreciation.

**Actionable insight:**
- Allocate capital appreciation mandates to prime areas (East Legon, Airport Residential, Dzorwulu); allocate yield mandates to mid-market (Spintex, Adenta)
- The 2.9× USD/sqm premium of prime over mid-market (USD 2,874 vs USD 985) implies a wide margin for mixed-income fund structures that can develop at mid-market cost and sell at prime prices with location repositioning
- Hedge GHS exposure through USD-indexed leases (standard practice in prime Accra)
- Use `fdi_pct_gdp` and `credit_private_pct_gdp` as leading indicators of the investment climate cycle

---

### 10.5 Government & Housing Policy Makers (Ministry of Works & Housing, NHC)

**Mid-market / district indices:**
Ghana faces a **housing deficit of approximately 1.8 million units** (Ghana Statistical Service), with affordability worsening as nominal prices outpace wage growth. The AHPI shows that the affordability crisis is driven by **currency instability inflating construction input costs**, not a shortage of developer interest. The per-district data reveals that Kasoa offers the most affordable entry point (GHS 10,156/sqm) with the fastest real appreciation — making it the strongest candidate for mass-housing programme targeting.

**Prime areas:**
The prime index illustrates the widening two-tier market: prime USD/sqm (USD 2,874) is now **2.9× mid-market** (USD 985), up from a 1.2× ratio in 2010. Without policy intervention, the market is bifurcating into a USD-denominated enclave for elites and a structurally depreciating GHS market for everyone else.

**Actionable insight:**
- Index social housing subsidy amounts to the mid-market AHPI to prevent real-value erosion
- Use AHPI district forecasts to calibrate National Housing Fund mortgage subsidy rates by location — Kasoa rates should differ from Spintex rates
- Target Kasoa (lowest GHS/sqm at 10,156) and Tema (well-serviced industrial satellite) for mass housing programmes
- Consider a **prime-area uplift levy** on transactions above USD 2,000/sqm to cross-subsidise affordable housing development
- Implement exchange-rate stabilisation as the single most impactful housing affordability measure (given the 0.991 FX–AHPI correlation)

---

### 10.6 Bank of Ghana (Monetary Policy Committee)

**Mid-market:**
The 0.991 correlation between the exchange rate and the mid-market AHPI is a direct transmission mechanism: **cedi depreciation immediately inflates the GHS price of housing**, which feeds into shelter components of the CPI, creating a self-reinforcing inflation spiral.

**Prime areas:**
The prime market adds a second transmission channel. As the GHS/USD rate falls, prime-area GHS prices spike mechanically (+3,264% vs +1,303% for mid-market over 2010–2024). Wealthy households who transact in both markets use prime property as an inflation hedge, which increases USD demand, further pressuring the cedi. The two-way reinforcement between cedi weakness, AHPI spikes, and USD hoarding is visible in both index series.

**Actionable insight:**
- Incorporate both mid-market and prime AHPI trends into monetary policy transmission analysis
- Use the mid-market AHPI as a near-real-time proxy for shelter inflation between formal CPI releases
- The divergence between prime and mid-market AHPI growth rates (3,264% vs 1,303%) is itself a measure of **wealth inequality amplified by monetary instability** — a secondary input to social stability assessments
- Monitor `broad_money_pct_gdp` (+0.813 mid-market correlation) as a leading indicator of AHPI acceleration

---

### 10.7 Researchers & Data Scientists (Prophet Forecasting)

**Mid-market aggregate:**
The dataset is structured for immediate use with **Facebook Prophet** (`ds`, `y`, and 20 regressors). The dominant regressors by correlation are `exchange_rate_ghs_usd` (+0.991), `cpi_index` (+0.986), and `urban_pop_pct` (+0.871). Two structural breaks (2014 cedi crisis; 2022 debt crisis) are recommended as manual changepoints.

**Per-district models:**
Each district is normalised to its own 2015=100 base, so district AHPI levels are directly comparable cross-sectionally. The district series carry no macro regressors — join on `ds` to the aggregate file for regressor-augmented modelling. Kasoa's high `annual_drift` (+0.013/year) creates a stronger trend component; consider a higher `changepoint_prior_scale` for that district.

**Prime areas:**
Prime series have a flatter seasonal pattern (±1.0% vs ±2.5%) and stronger USD-appreciation signal. Modelling in USD/sqm before GHS conversion decouples the trend from exchange-rate noise and is recommended when the forecast horizon extends beyond 12 months. Prime models benefit from lower `changepoint_prior_scale` (suggested: 0.3 vs 0.5 for mid-market) given smoother underlying dynamics.

**Actionable insight:**
- Standardise all regressors before fitting (wide variance ranges across columns)
- Add Prophet changepoints: `changepoints=['2014-01-01', '2022-01-01']`
- Use `add_regressor('exchange_rate_ghs_usd')` as the primary external driver for mid-market
- For prime areas, consider `add_regressor('gold_price_usd')` as the primary driver (stronger FDI and gold linkage)
- Consider a log-transformation of `y` to stabilise variance in the 2022–2024 spike (both mid-market and prime)
- Three datasets × three scenario assumptions = nine forecast paths for comprehensive scenario analysis

---

### 10.8 Insurance Companies (Property & Casualty Underwriters)

**Mid-market:**
Sum-insured values denominated in GHS will significantly understate replacement cost if not updated annually. The composite AHPI grew +1,303% over 14 years — a policy written in 2010 at its then-GHS value would cover only **7% of today's replacement cost** without indexation.

**Prime areas:**
The prime AHPI grew +3,264% over the same period — more than double the mid-market rate. A prime property insured in GHS terms without indexation faces even more severe underinsurance. However, since prime construction and finishing costs are largely USD-linked (imported materials, international contractors), the correct sum-insured benchmark for prime properties is the **USD replacement cost**, not the GHS AHPI.

**Actionable insight:**
- Mid-market: embed automatic mid-market AHPI-linked indexation clauses in long-term GHS-denominated property policies
- Prime: require USD-denominated reinstatement values; use the prime AHPI as a secondary cross-check on reasonableness
- Use AHPI forecasts to anticipate claims inflation in major cedi depreciation years — the 2022 spike from AHPI 145 → 250 in 18 months represents the kind of claims inflation event that should be in catastrophe models
- Segment underwriting books by district (using the per-district index) to set location-specific premium rates that reflect actual price appreciation rates

---

## 11. Limitations & Caveats

| Limitation | Detail |
|------------|--------|
| **No transaction database** | The AHPI is based on surveyed price estimates, not actual recorded transactions. Ghana has no mandatory public deed price register. |
| **Sparse anchor data** | Price anchors are annual point-in-time estimates, not monthly observations. Monthly values between anchors are interpolated, not observed. |
| **Low-income areas excluded** | The index does not represent very low-income areas (Ashaiman, Nima). A separate index would be needed for those segments. Prime areas now have their own dedicated index (`accra_prime_prices.csv`). |
| **Annual macro data** | Most World Bank indicators are reported annually. Monthly values are linearly interpolated, which smooths within-year volatility that undoubtedly exists. |
| **Informal market excluded** | A substantial fraction of Accra's housing transactions are informal (no title, no agent). These transactions are not captured. |
| **Single city** | The AHPI covers Greater Accra only. Kumasi, Takoradi, and other Ghanaian cities would require their own indices. |
| **GHS nominal, not real** | The index is not inflation-adjusted. An AHPI deflated by CPI would show near-zero real growth, which is economically accurate but less useful for nominal forecasting. |

---

## 12. Prophet Forecasting: Implementation & Results

Three production-ready Prophet training scripts have been built and trained on the AHPI datasets. All models share the same six-regressor macro set, `changepoint_prior_scale=0.5`, `seasonality_mode="multiplicative"`, and manual changepoints at the 2014 cedi crisis and 2022 debt crisis. Regressors are standardised with a `StandardScaler` fitted on training data only (2010–2022) to prevent leakage. Separate production models are then re-fitted on the full 2010–2024 dataset for forecasting.

### 12.1 Training Scripts

| Script | Models trained | Output models | Output forecasts |
|--------|:-------------:|:-------------:|:----------------:|
| `ahpi_prophet.py` | 1 (composite mid-market) | `models/ahpi_prophet_model.json` + `ahpi_scaler.pkl` | Bear / Base / Bull CSVs |
| `ahpi_prime_prophet.py` | 6 (one per prime area) | `models/prime_prophet_{slug}.json` + `prime_scaler.pkl` | 18 scenario CSVs + summary |
| `ahpi_district_prophet.py` | 5 (one per mid-market district) | `models/district_prophet_{slug}.json` + `district_scaler.pkl` | 15 scenario CSVs + summary |

Run any script directly:

```bash
python ahpi_prophet.py
python ahpi_prime_prophet.py
python ahpi_district_prophet.py
```

### 12.2 Model Configuration

All three scripts use the same core setup:

```python
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

REGRESSORS = [
    "exchange_rate_ghs_usd",   # rho = +0.991 — primary driver
    "cpi_index",               # rho = +0.986 — accumulated inflation
    "urban_pop_pct",           # rho = +0.871 — structural demand
    "broad_money_pct_gdp",     # rho = +0.813 — monetary conditions
    "gold_price_usd",          # rho = +0.786 — fiscal / remittance proxy
    "cocoa_price_usd",         # rho = +0.764 — export revenue proxy
]

m = Prophet(
    changepoints=["2014-01-01", "2022-01-01"],
    changepoint_prior_scale=0.5,
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.90,
)
for reg in REGRESSORS:
    m.add_regressor(reg)

# Fit scaler on training data only — prevents leakage into 2023-2024 test period
scaler = StandardScaler()
scaler.fit(df_train[REGRESSORS])
df_scaled = df.copy()
df_scaled[REGRESSORS] = scaler.transform(df[REGRESSORS])

m.fit(df_scaled[df_scaled["ds"] <= "2022-12-01"][["ds", "y"] + REGRESSORS])
```

### 12.3 Scenario Assumptions (2025–2026 Forecasts)

Three forward scenarios are applied to all 12 models (composite, 6 prime, 5 district):

| Scenario | `exchange_rate_ghs_usd` | `inflation_cpi_pct` | `gold_price_usd` | `cocoa_price_usd` |
|----------|:-----------------------:|:-------------------:|:----------------:|:-----------------:|
| **Bear** — continued cedi depreciation | 20.0 | 31% | 1,900 | 4,000 |
| **Base** — gradual stabilisation | 15.0 | 20% | 2,250 | 5,500 |
| **Bull** — cedi recovery on cocoa windfall | 12.0 | 14% | 2,500 | 6,500 |

`cpi_index` is compounded monthly from the last observed value using the scenario's `inflation_cpi_pct`. `urban_pop_pct` and `broad_money_pct_gdp` are forward-extrapolated via OLS from the trailing 36-month trend.

### 12.4 Test-Set Accuracy (2023–2024 · 24 months held out)

#### Composite Mid-Market

| Metric | Value |
|--------|-------|
| MAE | **28.49** index pts |
| RMSE | **32.73** index pts |
| MAPE | **7.8%** |

#### Prime Areas (per-area models)

| Area | MAE | RMSE | MAPE |
|------|:---:|:----:|:----:|
| East Legon | 15.33 | 22.55 | 1.9% |
| Cantonments | 25.02 | 32.32 | 3.8% |
| Airport Residential | 8.00 | 9.62 | 1.2% |
| Labone / Roman Ridge | 8.75 | 11.14 | 1.2% |
| Dzorwulu / Abelenkpe | 7.87 | 9.63 | 1.1% |
| Trasacco Valley | 46.87 | 58.86 | 7.6% |
| **Average** | **18.64** | **24.02** | **2.8%** |

> Trasacco Valley's higher error reflects its ultra-thin transaction market and the greater noisiness of its price anchors. All other prime areas achieve MAPE ≤ 4%.

#### Mid-Market Districts (per-district models)

| District | MAE | RMSE | MAPE |
|----------|:---:|:----:|:----:|
| Spintex Road | 47.93 | 55.44 | 12.3% |
| Adenta | 34.80 | 39.23 | 9.6% |
| Tema | 30.76 | 35.58 | 8.5% |
| Dome | 32.77 | 37.33 | 9.3% |
| Kasoa | 39.48 | 45.28 | 9.1% |
| **Average** | **37.15** | **42.57** | **9.8%** |

> District MAPEs are higher than the composite (9.8% vs 7.8%) because individual districts carry more idiosyncratic noise that partially cancels in the aggregate.

### 12.5 Dec 2026 Scenario Forecasts

#### Composite Mid-Market AHPI

| Scenario | Dec 2026 AHPI | 90% CI |
|----------|:-------------:|:------:|
| Bear | 663.6 | [634.6 – 692.7] |
| Base | 550.6 | [530.5 – 571.7] |
| Bull | 489.8 | [469.0 – 501.4] |

#### Prime Areas — Base Scenario Dec 2026

| Area | Dec 2026 AHPI |
|------|:-------------:|
| Airport Residential | 953.8 |
| East Legon | 938.2 |
| Dzorwulu / Abelenkpe | 925.0 |
| Labone / Roman Ridge | 814.4 |
| Trasacco Valley | 754.1 |
| Cantonments | 737.8 |

#### Mid-Market Districts — Base Scenario Dec 2026

| District | Dec 2026 AHPI |
|----------|:-------------:|
| Kasoa | 657.2 |
| Spintex Road | 568.9 |
| Adenta | 550.1 |
| Dome | 539.0 |
| Tema | 515.4 |

### 12.6 Interactive Dashboard (`accra_dashboard.py`)

A full-featured Dash dashboard visualises all three datasets and all three Prophet model families across **10 tabs**:

| Tab | Content |
|-----|---------|
| **Overview** | AHPI composite KPIs, trend chart, YoY change |
| **Districts** | Per-district AHPI lines, spread chart, price tables |
| **Prime Areas** | Per-area AHPI lines, USD comparison, heatmap |
| **Macro** | All 20 regressors with correlation rankings |
| **Correlation** | Regressor–AHPI Pearson heatmap and scatter |
| **Seasonality** | Monthly seasonal factors for all markets |
| **Map** | Location dot map (Mapbox) + GIS Leaflet choropleth with animated time slider (2010–2029); play/pause animation across historical actuals and Bear/Base/Bull scenario forecasts |
| **Forecast** | Mid-market Prophet: historical + test eval + Bear/Base/Bull scenarios + residuals |
| **Prime Forecast** | Per-area Prophet: area selector, CI toggle, 2-panel figure, accuracy card, Dec 2026 targets |
| **District Forecast** | Per-district Prophet: district selector, CI toggle, 2-panel figure, accuracy card, Dec 2026 targets |

Launch with:

```bash
python accra_dashboard.py
# or with a custom port:
python accra_dashboard.py --port 8080
```

The **Map tab** contains two sections:
- **Location dot map:** Mapbox scatter map showing all 11 districts/areas with bubble size scaled to AHPI; segmented by Mid-Market / Prime / Both
- **GIS choropleth:** Leaflet polygon map with per-neighbourhood price or forecast-growth shading, metric/scenario/tile controls, GeoJSON export, and an **animated time slider** spanning 2010–2029. Drag the slider or press ▶ to play — historical frames (2010–2024) show actual USD/sqm, GHS/sqm, or AHPI values; projected frames (2025–2029) show Prophet AHPI forecasts for the selected scenario. The colour scale is globally normalised so frames are directly comparable. A badge labels each frame as HISTORICAL or PROJECTED.

The **Forecast, Prime Forecast, and District Forecast** tabs share a common two-panel layout:
- **Upper panel (72%):** historical actuals (area fill), test-period predicted vs actual with 90% CI band, three scenario lines (2025–2026) each with their own CI shading
- **Lower panel (28%):** monthly residual bar chart over the 2023–2024 test period
- **Info cards:** dynamic MAE / RMSE / MAPE card and Dec 2026 scenario targets, both updating when the area/district selector changes

```
models/
├── ahpi_prophet_model.json       ← mid-market production model
├── ahpi_scaler.pkl               ← mid-market StandardScaler
├── prime_prophet_{slug}.json     ← 6 prime-area models
├── prime_scaler.pkl              ← prime StandardScaler (shared)
├── district_prophet_{slug}.json  ← 5 district models
└── district_scaler.pkl           ← district StandardScaler (shared)

forecasts/
├── ahpi_test_eval.csv
├── ahpi_forecast_{bear,base,bull}.csv
├── prime_test_eval_{slug}.csv          (×6)
├── prime_forecast_{scen}_{slug}.csv    (×18)
├── prime_test_summary.csv
├── district_test_eval_{slug}.csv       (×5)
├── district_forecast_{scen}_{slug}.csv (×15)
└── district_test_summary.csv
```

---

## 13. Data Sources

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

*Document updated: March 2026 · Dataset version: 2.2 · Index base year: 2015 = 100*
*Three datasets: aggregate mid-market (180 × 22), per-district (900 × 5), prime areas (1,080 × 5)*
*Prophet models: 12 trained (composite + 6 prime + 5 district) · Scenarios: Bear / Base / Bull (2025–2026) · Dashboard: 10-tab interactive Dash app (`accra_dashboard.py`)*
*Map tab: GIS Leaflet choropleth with animated 2010–2029 time slider; globally normalised colour scale spans historical actuals and scenario forecasts.*
*AHPI is an estimated index for research and forecasting purposes. It should not be used as the sole basis for individual investment decisions.*
