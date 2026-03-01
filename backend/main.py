"""
ZHVI Prediction FastAPI Backend
Exposes REST endpoints for the Flutter web frontend.
"""

from __future__ import annotations

import os
from typing import Optional

import gdown
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet
from pydantic import BaseModel

app = FastAPI(
    title="ZHVI Prediction API",
    description="API for Zillow Home Value Index prediction and analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── City coordinates ──────────────────────────────────────────────────────────

CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "New York": (40.7128, -74.0060),
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "Seattle": (47.6062, -122.3321),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
    "Austin": (30.2672, -97.7431),
    "Jacksonville": (30.3322, -81.6557),
    "Fort Worth": (32.7555, -97.3331),
    "Columbus": (39.9612, -82.9988),
    "Indianapolis": (39.7684, -86.1581),
    "Charlotte": (35.2271, -80.8431),
    "Denver": (39.7392, -104.9903),
    "Washington": (38.8951, -77.0369),
    "Boston": (42.3601, -71.0589),
    "El Paso": (31.7619, -106.4850),
    "Detroit": (42.3314, -83.0458),
    "Nashville": (36.1627, -86.7816),
    "Baltimore": (39.2904, -76.6122),
    "Oklahoma City": (35.4676, -97.5164),
    "Las Vegas": (36.1699, -115.1398),
    "Louisville": (38.2527, -85.7585),
    "Milwaukee": (43.0389, -87.9065),
    "Albuquerque": (35.0844, -106.6504),
    "Tucson": (32.2226, -110.9747),
    "Fresno": (36.7378, -119.7871),
    "Sacramento": (38.58, -121.49),
    "Kansas City": (39.0997, -94.5786),
    "Mesa": (33.4152, -111.8315),
    "Virginia Beach": (36.8529, -75.9780),
    "Atlanta": (33.7490, -84.3880),
    "Colorado Springs": (38.8339, -104.8214),
    "Omaha": (41.2565, -95.9345),
    "Raleigh": (35.7796, -78.6382),
    "Miami": (25.7617, -80.1918),
    "Cleveland": (41.4993, -81.6944),
    "Tulsa": (36.1540, -95.9928),
    "Oakland": (37.8049, -122.2711),
    "Minneapolis": (44.9778, -93.2650),
    "Wichita": (37.6872, -97.3301),
    "Arlington": (32.7357, -97.1081),
    "Bakersfield": (35.3733, -119.0187),
    "New Orleans": (29.9511, -90.0715),
    "Honolulu": (21.3069, -157.8583),
    "Anaheim": (33.8366, -117.9143),
    "Tampa": (27.9506, -82.4572),
}

# ─── Data cache ────────────────────────────────────────────────────────────────

_data: Optional[pd.DataFrame] = None
_DATA_PATH = "/tmp/zillow_data.csv"
_FILE_ID = "1wcabOuayxwGUzj_cd5k5fIboKFBce4yj"
_ID_COLS = ["RegionID", "Zipcode", "City", "State", "Metro", "CountyName", "SizeRank"]


def get_data() -> pd.DataFrame:
    global _data
    if _data is None:
        if not os.path.exists(_DATA_PATH):
            gdown.download(
                f"https://drive.google.com/uc?id={_FILE_ID}", _DATA_PATH, quiet=False
            )
        df = pd.read_csv(_DATA_PATH)
        df.rename({"RegionName": "Zipcode"}, axis="columns", inplace=True)
        _data = df
    return _data


def melt_data(df: pd.DataFrame) -> pd.DataFrame:
    id_vars = [c for c in _ID_COLS if c in df.columns]
    melted = pd.melt(df, id_vars=id_vars, var_name="time")
    melted["time"] = pd.to_datetime(melted["time"], infer_datetime_format=True)
    melted = melted.dropna(subset=["value"])
    return melted.groupby("time").aggregate({"value": "mean"}).reset_index()


# ─── Startup ───────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load the dataset so the first request is fast."""
    try:
        get_data()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not pre-load data during startup: {exc}")


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/cities")
async def get_cities() -> dict:
    data = get_data()
    cities = sorted(data["City"].dropna().unique().tolist())
    return {"cities": cities}


@app.get("/zipcodes/{city}")
async def get_zipcodes(city: str) -> dict:
    data = get_data()
    city_data = data[data["City"] == city]
    if city_data.empty:
        raise HTTPException(status_code=404, detail="City not found")
    zipcodes = [str(z) for z in city_data["Zipcode"].dropna().unique().tolist()]
    return {"zipcodes": zipcodes}


@app.get("/market-metrics")
async def get_market_metrics(
    city: str = Query(...),
    zipcode: str = Query(...),
) -> dict:
    data = get_data()
    city_data = data[data["City"] == city]
    if city_data.empty:
        raise HTTPException(status_code=404, detail="City not found")

    city_melted = melt_data(city_data)
    if len(city_melted) < 2:
        raise HTTPException(status_code=422, detail="Insufficient data for metrics")

    volatility = float(city_melted["value"].std())
    first_val = float(city_melted["value"].iloc[0])
    last_val = float(city_melted["value"].iloc[-1])
    roi = (last_val - first_val) / first_val if first_val != 0 else 0.0
    risk_score = volatility * 0.6 + roi * 0.4
    return {"volatility": volatility, "roi": roi, "risk_score": risk_score}


@app.get("/price-history")
async def get_price_history(
    city: str = Query(...),
    zipcode: str = Query(...),
) -> dict:
    data = get_data()
    city_data = data[data["City"] == city]
    zipcode_data = data[data["Zipcode"].astype(str) == str(zipcode)]

    city_melted = melt_data(city_data)
    zip_melted = melt_data(zipcode_data)

    city_history = [
        {"date": row["time"].strftime("%Y-%m-%d"), "value": float(row["value"])}
        for _, row in city_melted.iterrows()
    ]
    zip_history = [
        {"date": row["time"].strftime("%Y-%m-%d"), "value": float(row["value"])}
        for _, row in zip_melted.iterrows()
    ]
    return {"city_history": city_history, "zipcode_history": zip_history}


class ForecastRequest(BaseModel):
    zipcode: str
    years: int = 5
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    event_impact: Optional[int] = 0


@app.post("/forecast")
async def get_forecast(request: ForecastRequest) -> dict:
    data = get_data()
    zipcode_data = data[data["Zipcode"].astype(str) == str(request.zipcode)]
    if zipcode_data.empty:
        raise HTTPException(status_code=404, detail="Zipcode not found")

    melted = melt_data(zipcode_data)
    df_train = melted[["time", "value"]].rename(columns={"time": "ds", "value": "y"})
    period = request.years * 365

    use_event = bool(
        request.event_name
        and request.event_date
        and request.event_impact is not None
    )

    model = Prophet(daily_seasonality=False)

    if use_event:
        event_date = pd.to_datetime(request.event_date)
        if df_train["ds"].min() <= event_date <= df_train["ds"].max():
            model.add_regressor(request.event_name)  # type: ignore[arg-type]
            df_train = df_train.copy()
            df_train[request.event_name] = 0  # type: ignore[index]
            df_train.loc[df_train["ds"] >= event_date, request.event_name] = (  # type: ignore[index]
                request.event_impact
            )
        else:
            use_event = False

    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)

    if use_event:
        event_date = pd.to_datetime(request.event_date)
        future[request.event_name] = 0  # type: ignore[index]
        future.loc[future["ds"] >= event_date, request.event_name] = (  # type: ignore[index]
            request.event_impact
        )

    forecast = model.predict(future)

    # Sample every 30 days to keep response size reasonable
    sampled = forecast[::30]
    return {
        "dates": sampled["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "forecast": [float(v) for v in sampled["yhat"].tolist()],
        "lower": [float(v) for v in sampled["yhat_lower"].tolist()],
        "upper": [float(v) for v in sampled["yhat_upper"].tolist()],
    }


@app.get("/city-coordinates/{city}")
async def get_city_coordinates(city: str) -> dict:
    coords = CITY_COORDINATES.get(city)
    if not coords:
        raise HTTPException(status_code=404, detail="Coordinates not available for city")
    return {"lat": coords[0], "lon": coords[1]}


@app.get("/city-markers/{city}")
async def get_city_markers(city: str) -> list:
    data = get_data()
    city_data = data[data["City"] == city]
    coords = CITY_COORDINATES.get(city)
    if not coords or city_data.empty:
        return []

    id_cols_present = [c for c in _ID_COLS if c in city_data.columns]
    date_cols = [c for c in city_data.columns if c not in id_cols_present]

    markers = []
    for _, row in city_data.iterrows():
        latest_val: Optional[float] = None
        for col in reversed(date_cols):
            if pd.notna(row[col]):
                latest_val = float(row[col])
                break
        if latest_val is not None:
            markers.append(
                {
                    "zipcode": str(row["Zipcode"]),
                    "lat": coords[0],
                    "lon": coords[1],
                    "latest_value": latest_val,
                }
            )

    return markers
