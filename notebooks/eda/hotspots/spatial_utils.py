"""
Utility functions for spatial hotspot analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from lib.db import init_db
from lib.data_validator import validate_and_clean


def load_clean_booking_locations(
    rooms_to_exclude: Iterable[str] | None = ("reception_hall",),
    exclude_missing_location_bookings: bool = True,
) -> pd.DataFrame:
    """
    Load booking-level latitude/longitude points from a freshly
    initialized DuckDB connection that has been cleaned by the
    data validator.
    """
    con: duckdb.DuckDBPyConnection = validate_and_clean(
        init_db(),
        verbose=False,
        rooms_to_exclude=list(rooms_to_exclude) if rooms_to_exclude else None,
        exclude_missing_location_bookings=exclude_missing_location_bookings,
    )

    query = """
        SELECT
            b.id AS booking_id,
            b.total_price,
            CAST(b.arrival_date AS DATE) AS arrival_date,
            CAST(b.departure_date AS DATE) AS departure_date,
            hl.city,
            hl.country,
            hl.latitude,
            hl.longitude
        FROM bookings b
        JOIN hotel_location hl
          ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('confirmed', 'Booked')
          AND hl.latitude IS NOT NULL
          AND hl.longitude IS NOT NULL
    """
    df = con.execute(query).fetchdf()
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    return df


def summarize_bbox(df: pd.DataFrame) -> dict[str, float]:
    """
    Return the bounding box of a lat/lon dataframe.
    """
    if df.empty:
        return {"min_lat": np.nan, "max_lat": np.nan, "min_lon": np.nan, "max_lon": np.nan}
    return {
        "min_lat": float(df["latitude"].min()),
        "max_lat": float(df["latitude"].max()),
        "min_lon": float(df["longitude"].min()),
        "max_lon": float(df["longitude"].max()),
    }


def ensure_output_dir(path: str | Path) -> Path:
    """
    Ensure that the output directory exists and return it as a Path.
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


