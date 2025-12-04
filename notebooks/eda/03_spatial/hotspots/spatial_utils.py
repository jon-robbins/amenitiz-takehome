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
from lib.data_validator import CleaningConfig, DataCleaner
from lib.sql_loader import load_sql_file


def load_clean_booking_locations(
    rooms_to_exclude: Iterable[str] | None = ("reception_hall",),
    exclude_missing_location_bookings: bool = True,
) -> pd.DataFrame:
    """
    Load booking-level latitude/longitude points from a freshly
    initialized DuckDB connection that has been cleaned by the
    data validator.
    
    SQL Query: QUERY_LOAD_BOOKING_LOCATIONS (defined below)
    
    Parameters
    ----------
    rooms_to_exclude : Iterable[str] | None
        Room types to exclude from analysis.
    exclude_missing_location_bookings : bool
        If True, exclude bookings without location data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with booking_id, coordinates, and booking details.
    """
    # Initialize database
    con: duckdb.DuckDBPyConnection = init_db()
    
    # Clean data
    config = CleaningConfig(
        exclude_reception_halls=(rooms_to_exclude is not None),
        exclude_missing_location=exclude_missing_location_bookings
    )
    cleaner = DataCleaner(config)
    con = cleaner.clean(con)
    
    # Load SQL query from file
    query = load_sql_file('QUERY_LOAD_BOOKING_LOCATIONS.sql', __file__)
    
    # Execute query
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


