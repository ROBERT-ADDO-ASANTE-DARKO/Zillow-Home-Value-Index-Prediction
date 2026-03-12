"""
District-level AHPI endpoints.

Serves per-district mid-market price indices for Accra's 5 districts:
Spintex Road, Adenta, Tema, Dome, Kasoa.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..auth import AuthUser
from ..config import Settings, get_settings
from ..rate_limiter import sliding_window_rate_limit

router = APIRouter(
    prefix="/districts",
    tags=["Districts"],
    dependencies=[Depends(sliding_window_rate_limit)],
)

DISTRICTS = ["Spintex Road", "Adenta", "Tema", "Dome", "Kasoa"]


def _data_path(settings: Settings) -> Path:
    base = Path(__file__).resolve().parents[3]
    return base / "data" / "accra_district_prices.csv"


@lru_cache(maxsize=1)
def _load_districts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.sort_values(["district", "ds"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _get_df(settings: Annotated[Settings, Depends(get_settings)]) -> pd.DataFrame:
    path = _data_path(settings)
    if not path.exists():
        raise HTTPException(status_code=503, detail="District data unavailable")
    return _load_districts(str(path))


@router.get("")
async def list_districts(_user: AuthUser):
    """Return the list of available districts."""
    return {"districts": DISTRICTS}


@router.get("/index")
async def get_district_index(
    _user: AuthUser,
    df: Annotated[pd.DataFrame, Depends(_get_df)],
    district: str | None = None,
):
    """
    Return district AHPI time series.

    Pass ?district=Kasoa to filter by one district, or omit for all.
    """
    if district and district not in DISTRICTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown district '{district}'. Available: {DISTRICTS}",
        )
    filtered = df[df["district"] == district] if district else df
    return {
        "data": filtered.assign(ds=filtered["ds"].dt.strftime("%Y-%m")).to_dict(
            orient="records"
        )
    }


@router.get("/summary")
async def get_district_summary(
    _user: AuthUser,
    df: Annotated[pd.DataFrame, Depends(_get_df)],
):
    """Return latest AHPI value and total % change for each district."""
    results = []
    for district, grp in df.groupby("district"):
        grp = grp.sort_values("ds")
        start_val = float(grp.iloc[0]["y"])
        end_val = float(grp.iloc[-1]["y"])
        pct = ((end_val - start_val) / start_val) * 100
        results.append(
            {
                "district": district,
                "latest_ahpi": round(end_val, 2),
                "start_ahpi": round(start_val, 2),
                "pct_change": round(pct, 1),
                "latest_date": grp.iloc[-1]["ds"].strftime("%Y-%m"),
            }
        )
    return {"summary": sorted(results, key=lambda x: x["latest_ahpi"], reverse=True)}
