"""
Debug distance feature calculations.
Test with smaller dataset and fix geometry issues.
"""

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

from lib.db import init_db
from lib.data_validator import validate_and_clean
from lib.eda_utils import load_coastline_shapefile

# %%
# Constants
MADRID_LAT = 40.4165
MADRID_LON = -3.70256

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points on Earth in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371  # Earth radius in km

# %%
# Load unique hotel locations
print("Loading unique hotel locations...")
con = validate_and_clean(
    init_db(),
    verbose=False,
    rooms_to_exclude=["reception_hall"],
    exclude_missing_location_bookings=True,
)

hotel_locations = con.execute("""
    SELECT DISTINCT
        hotel_id,
        latitude,
        longitude,
        city,
        country
    FROM hotel_location
    WHERE latitude IS NOT NULL
      AND longitude IS NOT NULL
""").fetchdf()

# Convert to numeric and drop NaN
hotel_locations['latitude'] = pd.to_numeric(hotel_locations['latitude'], errors='coerce')
hotel_locations['longitude'] = pd.to_numeric(hotel_locations['longitude'], errors='coerce')
hotel_locations = hotel_locations.dropna(subset=['latitude', 'longitude'])

print(f"Loaded {len(hotel_locations):,} unique hotel locations")
print(f"Lat range: [{hotel_locations['latitude'].min():.2f}, {hotel_locations['latitude'].max():.2f}]")
print(f"Lon range: [{hotel_locations['longitude'].min():.2f}, {hotel_locations['longitude'].max():.2f}]")

# %%
# Filter to Spain only (reasonable range)
spain_bounds = {
    'lat_min': 27.0,  # Canary Islands
    'lat_max': 44.0,  # Northern Spain
    'lon_min': -18.5,  # Canary Islands
    'lon_max': 5.0     # Eastern Spain
}

hotel_locations_spain = hotel_locations[
    (hotel_locations['latitude'] >= spain_bounds['lat_min']) &
    (hotel_locations['latitude'] <= spain_bounds['lat_max']) &
    (hotel_locations['longitude'] >= spain_bounds['lon_min']) &
    (hotel_locations['longitude'] <= spain_bounds['lon_max'])
].copy()

print(f"\nFiltered to Spain bounds: {len(hotel_locations_spain):,} hotels")

# %%
# Calculate distance from Madrid (fast)
print("\nCalculating distance from Madrid...")
hotel_locations_spain['distance_from_madrid'] = haversine_distance(
    hotel_locations_spain['latitude'].values,
    hotel_locations_spain['longitude'].values,
    MADRID_LAT,
    MADRID_LON
)

print(f"Distance from Madrid - Min: {hotel_locations_spain['distance_from_madrid'].min():.2f} km")
print(f"Distance from Madrid - Max: {hotel_locations_spain['distance_from_madrid'].max():.2f} km")
print(f"Distance from Madrid - Median: {hotel_locations_spain['distance_from_madrid'].median():.2f} km")

# %%
# Load coastline shapefile
print("\nLoading coastline shapefile...")
coastline_gdf = load_coastline_shapefile(use_local_zip=True)
print(f"Loaded {len(coastline_gdf):,} coastline features")

# %%
# Test with small sample first
print("\n=== TESTING WITH SMALL SAMPLE (10 hotels) ===")
sample_hotels = hotel_locations_spain.sample(min(10, len(hotel_locations_spain)), random_state=42)

# Create GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(sample_hotels['longitude'], sample_hotels['latitude'])]
sample_gdf = gpd.GeoDataFrame(sample_hotels, geometry=geometry, crs="EPSG:4326")

print(f"\nSample hotels:")
print(sample_gdf[['hotel_id', 'city', 'latitude', 'longitude']].to_string(index=False))

# %%
# Project to UTM Zone 30N
print("\nProjecting to UTM Zone 30N...")
sample_projected = sample_gdf.to_crs("EPSG:32630")
coastline_projected = coastline_gdf.to_crs("EPSG:32630")

print(f"Sample hotels projected: {len(sample_projected)}")
print(f"Coastline features projected: {len(coastline_projected)}")

# Check for invalid geometries
print(f"\nChecking for invalid geometries...")
print(f"Hotels with invalid geometry: {(~sample_projected.geometry.is_valid).sum()}")
print(f"Coastline with invalid geometry: {(~coastline_projected.geometry.is_valid).sum()}")

# Check for NaN coordinates in projected data
print(f"\nHotels with NaN in bounds: {sample_projected.geometry.bounds.isna().any().any()}")
print(f"Coastline with NaN in bounds: {coastline_projected.geometry.bounds.isna().any().any()}")

# %%
# Simple approach: Calculate distance without spatial index
print("\n=== APPROACH 1: Direct distance calculation (slow but safe) ===")
distances_simple = []

for i, hotel_point in enumerate(sample_projected.geometry):
    print(f"Hotel {i+1}: {sample_hotels.iloc[i]['city']}")
    
    # Check if hotel point is valid
    if not hotel_point.is_valid or hotel_point.is_empty:
        print(f"  WARNING: Invalid hotel point")
        distances_simple.append(np.nan)
        continue
    
    # Calculate distance to all coastlines (this is slow but works)
    try:
        min_distance_m = coastline_projected.geometry.distance(hotel_point).min()
        distance_km = min_distance_m / 1000.0
        distances_simple.append(distance_km)
        print(f"  Distance to coast: {distance_km:.2f} km")
    except Exception as e:
        print(f"  ERROR: {e}")
        distances_simple.append(np.nan)

sample_hotels_result = sample_hotels.copy()
sample_hotels_result['distance_from_coast'] = distances_simple

print("\nResults:")
print(sample_hotels_result[['hotel_id', 'city', 'distance_from_coast']].to_string(index=False))

# %%
# If sample works, run on all Spain hotels
if len([d for d in distances_simple if not np.isnan(d)]) > 0:
    print("\n=== SUCCESS! Running on all Spain hotels ===")
    
    # Create full GeoDataFrame
    geometry_all = [Point(lon, lat) for lon, lat in zip(hotel_locations_spain['longitude'], hotel_locations_spain['latitude'])]
    hotels_gdf_all = gpd.GeoDataFrame(hotel_locations_spain, geometry=geometry_all, crs="EPSG:4326")
    hotels_projected_all = hotels_gdf_all.to_crs("EPSG:32630")
    
    # Calculate distances for all
    distances_all = []
    print(f"Processing {len(hotels_projected_all)} hotels...")
    
    for i, hotel_point in enumerate(hotels_projected_all.geometry):
        if i % 500 == 0:
            print(f"  Processed {i:,}/{len(hotels_projected_all):,} hotels...")
        
        if not hotel_point.is_valid or hotel_point.is_empty:
            distances_all.append(np.nan)
            continue
        
        try:
            min_distance_m = coastline_projected.geometry.distance(hotel_point).min()
            distances_all.append(min_distance_m / 1000.0)
        except Exception:
            distances_all.append(np.nan)
    
    hotel_locations_spain['distance_from_coast'] = distances_all
    
    print(f"\nCompleted!")
    print(f"Distance from coast - Min: {hotel_locations_spain['distance_from_coast'].min():.2f} km")
    print(f"Distance from coast - Max: {hotel_locations_spain['distance_from_coast'].max():.2f} km")
    print(f"Distance from coast - Median: {hotel_locations_spain['distance_from_coast'].median():.2f} km")
    print(f"Missing values: {hotel_locations_spain['distance_from_coast'].isna().sum()}")
    
    # Save results
    output_path = Path("outputs/hotel_distance_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hotel_locations_spain[['hotel_id', 'distance_from_madrid', 'distance_from_coast']].to_csv(
        output_path, index=False
    )
    print(f"\nSaved to {output_path}")
else:
    print("\n=== FAILED on sample - check geometry issues ===")

# %%

