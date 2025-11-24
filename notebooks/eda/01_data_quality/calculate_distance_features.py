"""
Calculate distance features for hotel locations.

Creates two new features:
- distance_from_madrid: Distance in km from Madrid center
- distance_from_coast: Distance in km from nearest coastline
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

from lib.db import init_db
from lib.data_validator import validate_and_clean
from lib.eda_utils import load_coastline_shapefile

# %%
# Constants
MADRID_LAT = 40.4165
MADRID_LON = -3.70256


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great circle distance between two points on Earth in km.
    Uses the Haversine formula.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


def calculate_distance_from_madrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distance from Madrid for each hotel location.
    Adds 'distance_from_madrid' column in km.
    """
    df = df.copy()
    df["distance_from_madrid"] = haversine_distance(
        df["latitude"].values,
        df["longitude"].values,
        MADRID_LAT,
        MADRID_LON,
    )
    return df


def calculate_distance_from_coast(
    df: pd.DataFrame, coastline_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Calculate distance from nearest coastline for each hotel location.
    Adds 'distance_from_coast' column in km.
    Uses spatial index for efficient nearest neighbor search.
    """
    df = df.copy()

    # Create GeoDataFrame from hotel locations
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    hotels_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Convert to projected CRS for accurate distance calculations (meters)
    # Use UTM zone 30N which covers Spain
    hotels_projected = hotels_gdf.to_crs("EPSG:32630")
    coastline_projected = coastline_gdf.to_crs("EPSG:32630")

    # Build spatial index for coastline
    print("Building spatial index...")
    sindex = coastline_projected.sindex
    
    # Calculate minimum distance to coastline for each hotel
    print(f"Calculating distances for {len(hotels_projected)} hotels...")
    distances = []
    
    for i, hotel_point in enumerate(hotels_projected.geometry):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i:,}/{len(hotels_projected):,} hotels...")
        
        # Use spatial index to find nearby coastlines (within 500km buffer)
        buffer_distance = 500000  # 500 km in meters
        possible_matches_idx = list(
            sindex.query(hotel_point.buffer(buffer_distance), predicate="intersects")
        )
        
        if possible_matches_idx:
            # Calculate distance only to nearby coastlines
            nearby_coastlines = coastline_projected.iloc[possible_matches_idx]
            min_distance_m = nearby_coastlines.geometry.distance(hotel_point).min()
        else:
            # Fallback: calculate to all coastlines if nothing within buffer
            min_distance_m = coastline_projected.geometry.distance(hotel_point).min()
        
        distance_km = min_distance_m / 1000.0
        distances.append(distance_km)

    df["distance_from_coast"] = distances
    return df


def load_hotel_locations() -> pd.DataFrame:
    """
    Load unique hotel locations from the database.
    Returns DataFrame with hotel_id, latitude, longitude, city, country.
    Filters out any NaN or invalid coordinates.
    """
    con = validate_and_clean(
        init_db(),
        verbose=False,
        rooms_to_exclude=["reception_hall"],
        exclude_missing_location_bookings=True,
    )

    query = """
        SELECT DISTINCT
            hotel_id,
            latitude,
            longitude,
            city,
            country
        FROM hotel_location
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
    """
    df = con.execute(query).fetchdf()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    
    # Drop any rows with NaN coordinates
    initial_count = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    if len(df) < initial_count:
        print(f"Filtered out {initial_count - len(df)} hotels with invalid coordinates.")
    
    return df


def main() -> None:
    """Calculate and save distance features."""
    print("=" * 80)
    print("CALCULATING DISTANCE FEATURES")
    print("=" * 80)

    # Load hotel locations (unique hotels only)
    print("\nLoading unique hotel locations...")
    hotels_df = load_hotel_locations()
    print(f"Loaded {len(hotels_df):,} unique hotel locations.")

    # Calculate distance from Madrid
    print("\nCalculating distance from Madrid...")
    print(f"Madrid coordinates: ({MADRID_LAT}, {MADRID_LON})")
    hotels_df = calculate_distance_from_madrid(hotels_df)
    print(f"Distance from Madrid - Min: {hotels_df['distance_from_madrid'].min():.2f} km")
    print(f"Distance from Madrid - Max: {hotels_df['distance_from_madrid'].max():.2f} km")
    print(
        f"Distance from Madrid - Median: {hotels_df['distance_from_madrid'].median():.2f} km"
    )

    # Load coastline shapefile
    print("\nLoading coastline shapefile...")
    coastline_gdf = load_coastline_shapefile(use_local_zip=True)
    print(f"Loaded {len(coastline_gdf):,} coastline features.")

    # Calculate distance from coast
    print("\nCalculating distance from coast...")
    print("(This may take several minutes for ~20K hotels...)")
    hotels_df = calculate_distance_from_coast(hotels_df, coastline_gdf)
    print(f"Distance from coast - Min: {hotels_df['distance_from_coast'].min():.2f} km")
    print(f"Distance from coast - Max: {hotels_df['distance_from_coast'].max():.2f} km")
    print(
        f"Distance from coast - Median: {hotels_df['distance_from_coast'].median():.2f} km"
    )

    # Save hotel-level results
    output_path = Path("../../../outputs/eda/spatial/data/hotel_distance_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hotels_df.to_csv(output_path, index=False)
    print(f"\nSaved hotel-level distance features to {output_path}")
    print("(Join this to bookings using hotel_id)")

    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print("\nDistance from Madrid (km):")
    print(hotels_df["distance_from_madrid"].describe())
    print("\nDistance from Coast (km):")
    print(hotels_df["distance_from_coast"].describe())

    # Display top/bottom hotels
    print("\n" + "=" * 80)
    print("CLOSEST/FARTHEST HOTELS")
    print("=" * 80)

    print("\nTop 5 closest hotels to Madrid:")
    print(
        hotels_df.nsmallest(5, "distance_from_madrid")[
            ["hotel_id", "city", "distance_from_madrid"]
        ].to_string(index=False)
    )

    print("\nTop 5 farthest hotels from Madrid:")
    print(
        hotels_df.nlargest(5, "distance_from_madrid")[
            ["hotel_id", "city", "distance_from_madrid"]
        ].to_string(index=False)
    )

    print("\nTop 5 closest hotels to coast:")
    print(
        hotels_df.nsmallest(5, "distance_from_coast")[
            ["hotel_id", "city", "distance_from_coast"]
        ].to_string(index=False)
    )

    print("\nTop 5 farthest hotels from coast:")
    print(
        hotels_df.nlargest(5, "distance_from_coast")[
            ["hotel_id", "city", "distance_from_coast"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("DONE - Use the CSV to join distance features to your bookings data")
    print("=" * 80)


if __name__ == "__main__":
    main()

