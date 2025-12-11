"""
Distance feature calculation for hotel locations.

Provides reusable functions for calculating:
- Distance from Madrid (capital, major hub)
- Distance from nearest coastline

Features are cached to CSV for efficiency.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Import haversine from engineering.py to avoid duplication
from src.features.engineering import haversine_distance, MADRID_LAT, MADRID_LON


# =============================================================================
# CONSTANTS
# =============================================================================

# Cache path for distance features
CACHE_PATH = Path("outputs/data/hotel_distance_features.csv")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_madrid_coords() -> Tuple[float, float]:
    """
    Get Madrid coordinates from cities500.json.
    
    Returns:
        Tuple of (latitude, longitude).
    """
    try:
        from src.data.cities_downloader import load_cities500
        cities_data = load_cities500()
        
        for city in cities_data:
            name = city.get("name", "").lower()
            if name == "madrid" and city.get("country") == "ES":
                return float(city["lat"]), float(city["lon"])
    except Exception:
        pass
    
    # Fallback to hardcoded coordinates
    return MADRID_LAT, MADRID_LON


def calculate_distance_from_madrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance_from_madrid column to DataFrame.
    
    Args:
        df: DataFrame with 'latitude' and 'longitude' columns
        
    Returns:
        DataFrame with added 'distance_from_madrid' column (km)
    """
    df = df.copy()
    madrid_lat, madrid_lon = get_madrid_coords()
    
    df["distance_from_madrid"] = haversine_distance(
        df["latitude"].values,
        df["longitude"].values,
        madrid_lat,
        madrid_lon,
    )
    return df


def calculate_distance_from_coast(
    df: pd.DataFrame,
    coastline_gdf: Optional["gpd.GeoDataFrame"] = None
) -> pd.DataFrame:
    """
    Add distance_from_coast column to DataFrame.
    
    Uses GSHHS shapefile for accurate coastline distance.
    
    Args:
        df: DataFrame with 'latitude' and 'longitude' columns
        coastline_gdf: Optional pre-loaded coastline GeoDataFrame
        
    Returns:
        DataFrame with added 'distance_from_coast' column (km)
    """
    
    df = df.copy()
    
    # Load coastline if not provided
    if coastline_gdf is None:
        from src.data.shapefile_downloader import get_coastline_path
        shp_path = get_coastline_path()
        if shp_path.exists():
            coastline_gdf = gpd.read_file(shp_path, bbox=(-20, 25, 10, 50))
            return df
    
    # Create GeoDataFrame from hotel locations
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    hotels_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Convert to projected CRS for accurate distance (UTM zone 30N for Spain)
    hotels_projected = hotels_gdf.to_crs("EPSG:32630")
    coastline_projected = coastline_gdf.to_crs("EPSG:32630")
    
    # Build spatial index
    print("Building spatial index for coastline...")
    sindex = coastline_projected.sindex
    
    # Calculate distances
    print(f"Calculating coastal distances for {len(hotels_projected)} hotels...")
    distances = []
    
    for i, hotel_point in enumerate(hotels_projected.geometry):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i:,}/{len(hotels_projected):,} hotels...")
        
        # Use spatial index for efficiency (500km buffer)
        buffer_distance = 500000  # meters
        possible_matches = list(
            sindex.query(hotel_point.buffer(buffer_distance), predicate="intersects")
        )
        
        if possible_matches:
            nearby = coastline_projected.iloc[possible_matches]
            min_distance_m = nearby.geometry.distance(hotel_point).min()
        else:
            min_distance_m = coastline_projected.geometry.distance(hotel_point).min()
        
        distances.append(min_distance_m / 1000.0)  # Convert to km
    
    df["distance_from_coast"] = distances
    return df


def calculate_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all distance features for hotel locations.
    
    Adds:
    - distance_from_madrid (km)
    - distance_from_coast (km)
    
    Args:
        df: DataFrame with 'hotel_id', 'latitude', 'longitude' columns
        
    Returns:
        DataFrame with distance features added
    """
    print("Calculating distance features...")
    
    # Distance from Madrid
    print("  Computing distance from Madrid...")
    df = calculate_distance_from_madrid(df)
    
    # Distance from coast
    print("  Computing distance from coast (this may take a few minutes)...")
    df = calculate_distance_from_coast(df)
    
    return df


def ensure_distance_features(
    con: Optional["duckdb.DuckDBPyConnection"] = None,
    force_recalculate: bool = False
) -> pd.DataFrame:
    """
    Load distance features from cache or calculate them.
    
    Args:
        con: Optional database connection for loading hotel locations
        force_recalculate: If True, recalculate even if cache exists
        
    Returns:
        DataFrame with hotel_id, latitude, longitude, distance_from_madrid, distance_from_coast
    """
    cache_path = Path(__file__).parent.parent.parent / CACHE_PATH
    
    # Try loading from cache
    if not force_recalculate and cache_path.exists():
        print(f"Loading distance features from cache: {cache_path}")
        return pd.read_csv(cache_path)
    
    # Need to calculate - get hotel locations
    if con is None:
        from src.data.loader import init_db
        con = init_db()
    
    query = """
    SELECT DISTINCT
        hotel_id,
        latitude,
        longitude,
        city
    FROM hotel_location
    WHERE latitude IS NOT NULL 
      AND longitude IS NOT NULL
    """
    
    hotels_df = con.execute(query).fetchdf()
    print(f"Loaded {len(hotels_df):,} hotel locations")
    
    # Calculate features
    hotels_df = calculate_distance_features(hotels_df)
    
    # Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hotels_df.to_csv(cache_path, index=False)
    print(f"Cached distance features to: {cache_path}")
    
    return hotels_df


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Calculate and cache distance features
    df = ensure_distance_features(force_recalculate=True)
    
    print("\n" + "=" * 60)
    print("DISTANCE FEATURES SUMMARY")
    print("=" * 60)
    
    print(f"\nHotels processed: {len(df):,}")
    
    print("\nDistance from Madrid (km):")
    print(df["distance_from_madrid"].describe())
    
    print("\nDistance from Coast (km):")
    print(df["distance_from_coast"].describe())
    
    print("\nClosest hotels to Madrid:")
    print(df.nsmallest(5, "distance_from_madrid")[
        ["hotel_id", "city", "distance_from_madrid"]
    ].to_string(index=False))
    
    print("\nClosest hotels to coast:")
    print(df.nsmallest(5, "distance_from_coast")[
        ["hotel_id", "city", "distance_from_coast"]
    ].to_string(index=False))

