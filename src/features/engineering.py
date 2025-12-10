"""
Feature engineering for price recommendation.

Core transformations validated by XGBoost (R² = 0.71) in elasticity analysis.

Validated features:
- Geographic: dist_center_km (to CITY center), is_madrid_metro, dist_coast_log, is_coastal
- Product: log_room_size, room_capacity_pax, amenities_score, view_quality_ordinal
- Temporal: month_sin, month_cos, is_summer, is_winter
"""

import numpy as np
import pandas as pd
import re
from typing import Optional, Tuple, Dict


# =============================================================================
# CONSTANTS
# =============================================================================

# Validated market elasticity from matched pairs analysis
# 735 pairs, bootstrap CI: [-0.41, -0.37]
MARKET_ELASTICITY = -0.39

# View quality mapping (ordinal 0-3)
VIEW_QUALITY_MAP = {
    'ocean_view': 3, 'sea_view': 3,
    'lake_view': 2, 'mountain_view': 2,
    'pool_view': 1, 'garden_view': 1,
    'city_view': 0, 'no_view': 0
}

# Top 5 cities by revenue (canonical mapping)
TOP_5_CITIES = {
    'madrid': 'madrid',
    'barcelona': 'barcelona',
    'sevilla': 'sevilla',
    'malaga': 'malaga',
    'málaga': 'malaga',
    'toledo': 'toledo'
}

# Market segment thresholds (km)
COASTAL_THRESHOLD_KM = 20.0
MADRID_THRESHOLD_KM = 50.0

# Madrid city center coordinates
MADRID_LAT = 40.4168
MADRID_LON = -3.7038

# Coastline shapefile for distance calculation
COASTLINE_SHAPEFILE = None  # Lazy loaded
COASTLINE_SHAPEFILE_PATH = "lib/data/GSHHS_shp/i/GSHHS_i_L1.shp"  # Intermediate resolution for speed


# =============================================================================
# GEOGRAPHIC UTILITIES
# =============================================================================

def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray
) -> np.ndarray:
    """
    Calculate haversine distance between two sets of coordinates in km.
    
    Vectorized for efficient batch processing.
    
    Args:
        lat1, lon1: First coordinate set (arrays or scalars)
        lat2, lon2: Second coordinate set (arrays or scalars)
    
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def calculate_city_centers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate city centers from hotel coordinates.
    
    City center = mean latitude/longitude of all hotels in that city.
    This is used when cities500.json doesn't have the city.
    
    Args:
        df: DataFrame with city_standardized, latitude, longitude
    
    Returns:
        DataFrame with city_standardized, city_lat, city_lon
    """
    city_centers = df.groupby('city_standardized').agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    city_centers.columns = ['city_standardized', 'city_lat', 'city_lon']
    return city_centers


def add_geographic_features(
    df: pd.DataFrame,
    city_centers: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add validated geographic features to DataFrame.
    
    Features added:
    - dist_center_km: Distance from hotel to its OWN city center (not Madrid!)
    - is_madrid_metro: Whether hotel is within 50km of Madrid (categorical)
    - is_coastal: Whether hotel is within 20km of coast (if distance_from_coast exists)
    - dist_coast_log: Log of distance to coast (if distance_from_coast exists)
    
    IMPORTANT: dist_center_km measures how central a hotel is within its city.
    This is different from distance to Madrid!
    
    Args:
        df: DataFrame with latitude, longitude, city_standardized
        city_centers: Optional pre-computed city centers. If None, computed from df.
    
    Returns:
        DataFrame with geographic features added
    """
    df = df.copy()
    
    # Ensure city_standardized exists
    if 'city_standardized' not in df.columns:
        if 'city' in df.columns:
            df['city_standardized'] = df['city'].apply(clean_city_name)
        else:
            df['city_standardized'] = 'other'
    
    # Calculate city centers if not provided
    if city_centers is None:
        city_centers = calculate_city_centers(df)
    
    # Merge city centers
    if 'city_lat' not in df.columns:
        df = df.merge(city_centers, on='city_standardized', how='left')
    
    # Fill missing city centers with hotel's own coordinates (fallback)
    df['city_lat'] = df['city_lat'].fillna(df['latitude'])
    df['city_lon'] = df['city_lon'].fillna(df['longitude'])
    
    # Distance to hotel's OWN city center (not Madrid!)
    df['dist_center_km'] = haversine_distance(
        df['latitude'].values,
        df['longitude'].values,
        df['city_lat'].values,
        df['city_lon'].values
    )
    
    # Is Madrid Metro: within 50km of Madrid city center
    df['dist_from_madrid'] = haversine_distance(
        df['latitude'].values,
        df['longitude'].values,
        MADRID_LAT,
        MADRID_LON
    )
    df['is_madrid_metro'] = (df['dist_from_madrid'] <= MADRID_THRESHOLD_KM).astype(int)
    
    # Coastal features (if distance_from_coast available)
    if 'distance_from_coast' in df.columns:
        df['is_coastal'] = (df['distance_from_coast'] < COASTAL_THRESHOLD_KM).astype(int)
        df['dist_coast_log'] = np.log1p(df['distance_from_coast'].fillna(100))
    
    return df


# =============================================================================
# PRODUCT FEATURES
# =============================================================================

def add_amenities_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate amenities score from boolean amenity flags.
    
    Amenities considered:
    - children_allowed
    - pets_allowed
    - events_allowed
    - smoking_allowed
    
    Args:
        df: DataFrame with amenity columns
    
    Returns:
        DataFrame with amenities_score column
    """
    df = df.copy()
    
    amenity_cols = ['children_allowed', 'pets_allowed', 'events_allowed', 'smoking_allowed']
    existing_cols = [c for c in amenity_cols if c in df.columns]
    
    if existing_cols:
        df['amenities_score'] = df[existing_cols].fillna(0).sum(axis=1)
    else:
        df['amenities_score'] = 0
    
    return df


# =============================================================================
# COLD-START PEER FEATURES (10km Radius)
# =============================================================================

# Default radius for peer search
PEER_RADIUS_KM = 10.0


def calculate_peer_price_features(
    target_lat: float,
    target_lon: float,
    target_room_type: str,
    peer_df: pd.DataFrame,
    radius_km: float = PEER_RADIUS_KM
) -> Dict[str, float]:
    """
    Calculate peer pricing features for cold-start hotels.
    
    For a new hotel without pricing history, we look at nearby hotels
    (within 10km radius) to derive pricing benchmarks.
    
    Features computed:
    - peer_price_mean: Average price of peers within radius
    - peer_price_median: Median price of peers
    - peer_price_p25: 25th percentile (budget benchmark)
    - peer_price_p75: 75th percentile (premium benchmark)
    - peer_price_std: Price variation in market
    - peer_occupancy_mean: Average occupancy of peers
    - peer_revpar_mean: Average RevPAR of peers
    - n_peers_10km: Number of comparable hotels
    - peer_price_same_type: Price of same room type peers
    
    Args:
        target_lat: Hotel latitude
        target_lon: Hotel longitude
        target_room_type: Hotel's primary room type
        peer_df: DataFrame with peer hotel data (must have lat, lon, price columns)
        radius_km: Search radius in km (default 10km)
    
    Returns:
        Dict with peer pricing features
    """
    # Calculate distances to all peers
    if len(peer_df) == 0:
        return _empty_peer_features()
    
    distances = haversine_distance(
        target_lat, target_lon,
        peer_df['latitude'].values, peer_df['longitude'].values
    )
    
    # Filter to radius
    mask = distances <= radius_km
    peers_in_radius = peer_df[mask].copy()
    peers_in_radius['distance_km'] = distances[mask]
    
    if len(peers_in_radius) == 0:
        return _empty_peer_features()
    
    # Price column detection
    price_col = None
    for col in ['actual_price', 'avg_adr', 'price', 'total_price']:
        if col in peers_in_radius.columns:
            price_col = col
            break
    
    if price_col is None:
        return _empty_peer_features()
    
    prices = peers_in_radius[price_col].dropna()
    
    # Calculate features
    features = {
        'peer_price_mean': prices.mean() if len(prices) > 0 else np.nan,
        'peer_price_median': prices.median() if len(prices) > 0 else np.nan,
        'peer_price_p25': prices.quantile(0.25) if len(prices) > 0 else np.nan,
        'peer_price_p75': prices.quantile(0.75) if len(prices) > 0 else np.nan,
        'peer_price_std': prices.std() if len(prices) > 1 else 0.0,
        'n_peers_10km': len(peers_in_radius),
    }
    
    # Occupancy features (if available)
    occ_col = None
    for col in ['actual_occupancy', 'occupancy_rate', 'occupancy']:
        if col in peers_in_radius.columns:
            occ_col = col
            break
    
    if occ_col:
        occupancies = peers_in_radius[occ_col].dropna()
        features['peer_occupancy_mean'] = occupancies.mean() if len(occupancies) > 0 else np.nan
    else:
        features['peer_occupancy_mean'] = np.nan
    
    # RevPAR features (if available)
    revpar_col = None
    for col in ['actual_revpar', 'revpar']:
        if col in peers_in_radius.columns:
            revpar_col = col
            break
    
    if revpar_col:
        revpars = peers_in_radius[revpar_col].dropna()
        features['peer_revpar_mean'] = revpars.mean() if len(revpars) > 0 else np.nan
    elif price_col and occ_col:
        # Calculate RevPAR
        valid = peers_in_radius[[price_col, occ_col]].dropna()
        if len(valid) > 0:
            features['peer_revpar_mean'] = (valid[price_col] * valid[occ_col]).mean()
        else:
            features['peer_revpar_mean'] = np.nan
    else:
        features['peer_revpar_mean'] = np.nan
    
    # Same room type peers
    if 'room_type' in peers_in_radius.columns:
        same_type = peers_in_radius[peers_in_radius['room_type'] == target_room_type]
        if len(same_type) > 0:
            features['peer_price_same_type'] = same_type[price_col].mean()
            features['n_peers_same_type'] = len(same_type)
        else:
            features['peer_price_same_type'] = features['peer_price_mean']
            features['n_peers_same_type'] = 0
    else:
        features['peer_price_same_type'] = features['peer_price_mean']
        features['n_peers_same_type'] = 0
    
    # Distance-weighted price (closer hotels weighted more)
    if len(peers_in_radius) > 0:
        weights = 1.0 / (peers_in_radius['distance_km'] + 0.1)  # Add 0.1 to avoid div by zero
        features['peer_price_distance_weighted'] = np.average(
            peers_in_radius[price_col].fillna(features['peer_price_mean']),
            weights=weights
        )
    else:
        features['peer_price_distance_weighted'] = np.nan
    
    return features


def _empty_peer_features() -> Dict[str, float]:
    """Return empty peer features when no peers found."""
    return {
        'peer_price_mean': np.nan,
        'peer_price_median': np.nan,
        'peer_price_p25': np.nan,
        'peer_price_p75': np.nan,
        'peer_price_std': 0.0,
        'peer_occupancy_mean': np.nan,
        'peer_revpar_mean': np.nan,
        'n_peers_10km': 0,
        'peer_price_same_type': np.nan,
        'n_peers_same_type': 0,
        'peer_price_distance_weighted': np.nan,
    }


def add_peer_price_features(
    df: pd.DataFrame,
    peer_df: Optional[pd.DataFrame] = None,
    radius_km: float = PEER_RADIUS_KM
) -> pd.DataFrame:
    """
    Add peer pricing features to a DataFrame of hotels.
    
    For each hotel, calculates pricing statistics from nearby hotels
    within the specified radius. This is particularly useful for
    cold-start scenarios where a hotel has no pricing history.
    
    Args:
        df: DataFrame with hotel data (must have latitude, longitude)
        peer_df: DataFrame with peer pricing data. If None, uses df itself.
        radius_km: Search radius in km (default 10km)
    
    Returns:
        DataFrame with peer pricing features added
    """
    df = df.copy()
    
    if peer_df is None:
        peer_df = df
    
    # Get room type column
    room_type_col = None
    for col in ['room_type', 'primary_room_type']:
        if col in df.columns:
            room_type_col = col
            break
    
    # Calculate peer features for each hotel
    peer_features_list = []
    for idx, row in df.iterrows():
        # Exclude self from peer search
        peers = peer_df[peer_df.index != idx] if peer_df is df else peer_df
        
        room_type = row.get(room_type_col, 'Standard') if room_type_col else 'Standard'
        
        features = calculate_peer_price_features(
            row['latitude'],
            row['longitude'],
            room_type,
            peers,
            radius_km
        )
        peer_features_list.append(features)
    
    # Add features to dataframe
    peer_features_df = pd.DataFrame(peer_features_list, index=df.index)
    
    for col in peer_features_df.columns:
        df[col] = peer_features_df[col]
    
    return df


# =============================================================================
# CITY STANDARDIZATION
# =============================================================================

def clean_city_name(name: str) -> str:
    """
    Clean city name for standardization.
    
    Removes punctuation, converts to lowercase, normalizes whitespace.
    """
    if pd.isna(name):
        return ''
    cleaned = re.sub(r'[^\w\s]', '', str(name).lower().strip())
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def standardize_city(city_str: str) -> str:
    """
    Standardize city to one of top 5 or 'other'.
    
    Uses fuzzy matching to handle variations like 'Málaga' vs 'malaga'.
    """
    if pd.isna(city_str):
        return 'other'
    
    city_clean = clean_city_name(city_str)
    
    if city_clean in TOP_5_CITIES:
        return TOP_5_CITIES[city_clean]
    
    # Check for partial matches
    for key, canonical in TOP_5_CITIES.items():
        if key in city_clean or city_clean in key:
            return canonical
    
    return 'other'


# =============================================================================
# MARKET SEGMENTATION
# =============================================================================

def _load_coastline():
    """Lazy load coastline shapefile."""
    global COASTLINE_SHAPEFILE
    if COASTLINE_SHAPEFILE is None:
        try:
            import geopandas as gpd
            from pathlib import Path
            
            project_root = Path(__file__).parent.parent.parent
            shp_path = project_root / COASTLINE_SHAPEFILE_PATH
            
            if shp_path.exists():
                # Load and clip to Spain bounding box for speed
                COASTLINE_SHAPEFILE = gpd.read_file(shp_path, bbox=(-20, 25, 10, 50))
            else:
                COASTLINE_SHAPEFILE = False  # Mark as unavailable
        except Exception as e:
            print(f"Warning: Could not load coastline shapefile: {e}")
            COASTLINE_SHAPEFILE = False
    return COASTLINE_SHAPEFILE


def compute_distance_to_coast(lat: float, lon: float) -> float:
    """
    Compute distance to coast using GSHHS shapefile.
    
    Uses the actual coastline geometry for accurate distance calculation.
    Falls back to approximate calculation if shapefile unavailable.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Distance to nearest coastline in km
    """
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    
    coastline = _load_coastline()
    
    if coastline is False or coastline is None:
        # Fallback: very rough estimate based on longitude (Spain context)
        # If near Mediterranean/Atlantic longitudes, assume closer to coast
        return 50.0  # Default to 50km if no shapefile
    
    try:
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        
        point = Point(lon, lat)
        
        # Find nearest point on coastline (polygon BOUNDARY, not interior)
        min_dist = float('inf')
        for geom in coastline.geometry:
            if geom is not None:
                # Use .boundary to get the coastline (edge of polygon)
                boundary = geom.boundary
                nearest = nearest_points(point, boundary)[1]
                # Approximate km using haversine
                dist = haversine_distance(
                    np.array([lat]), np.array([lon]),
                    np.array([nearest.y]), np.array([nearest.x])
                )[0]
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 50.0
        
    except Exception as e:
        print(f"Warning: Coast distance calculation failed: {e}")
        return 50.0  # Default fallback


def compute_distance_to_madrid(lat: float, lon: float) -> float:
    """Compute distance to Madrid city center."""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    
    return haversine_distance(
        np.array([lat]), np.array([lon]),
        np.array([MADRID_LAT]), np.array([MADRID_LON])
    )[0]


def get_market_segment(
    distance_coast_km: float = None,
    distance_madrid_km: float = None,
    latitude: float = None,
    longitude: float = None
) -> str:
    """
    Classify hotel into market segment based on geographic location.
    
    Market segments (matching elasticity EDA methodology):
    - coastal: within 20km of coast (resort market)
    - madrid_metro: within 50km of Madrid AND not coastal (urban market)
    - provincial: everything else (regional market)
    
    Args:
        distance_coast_km: Distance from nearest coastline in km.
        distance_madrid_km: Distance from Madrid city center in km.
        latitude: Hotel latitude (used if distances not provided)
        longitude: Hotel longitude (used if distances not provided)
    
    Returns:
        Market segment: 'coastal', 'madrid_metro', or 'provincial'.
    """
    # If distances not provided, compute from lat/lon
    if pd.isna(distance_coast_km) and latitude is not None:
        distance_coast_km = compute_distance_to_coast(latitude, longitude)
    if pd.isna(distance_madrid_km) and latitude is not None:
        distance_madrid_km = compute_distance_to_madrid(latitude, longitude)
    
    # Still missing? Return unknown
    if pd.isna(distance_coast_km) or pd.isna(distance_madrid_km):
        return 'unknown'
    
    is_coastal = distance_coast_km <= COASTAL_THRESHOLD_KM
    is_madrid_metro = distance_madrid_km <= MADRID_THRESHOLD_KM
    
    if is_coastal:
        return 'coastal'
    elif is_madrid_metro:
        return 'madrid_metro'
    else:
        return 'provincial'


# =============================================================================
# GRANULAR MARKET SEGMENTS - VECTORIZED
# =============================================================================

# Cache for cities data with KD-trees
_CITIES_CACHE = None

def _load_cities_with_kdtrees():
    """Load cities500 data and build KD-trees for fast lookups."""
    global _CITIES_CACHE
    if _CITIES_CACHE is not None:
        return _CITIES_CACHE
    
    import json
    from pathlib import Path
    from scipy.spatial import cKDTree
    
    cities_path = Path(__file__).parent.parent.parent / 'data' / 'cities500.json'
    if not cities_path.exists():
        return None
    
    with open(cities_path, 'r') as f:
        cities = json.load(f)
    
    spain_cities = pd.DataFrame([c for c in cities if c.get('country') == 'ES'])
    spain_cities['pop'] = spain_cities['pop'].fillna(0)
    
    # Pre-compute city tiers with KD-trees
    # Convert lat/lon to approximate km (for Spain: 1 deg lat ≈ 111km, 1 deg lon ≈ 85km)
    def build_tree(tier_df):
        if len(tier_df) == 0:
            return None, None
        coords = np.column_stack([
            tier_df['lat'].values * 111,  # Convert to km
            tier_df['lon'].values * 85
        ])
        return cKDTree(coords), tier_df
    
    _CITIES_CACHE = {
        'major_metros': build_tree(spain_cities[spain_cities['pop'] >= 500000]),
        'large_cities': build_tree(spain_cities[(spain_cities['pop'] >= 100000) & (spain_cities['pop'] < 500000)]),
        'medium_cities': build_tree(spain_cities[(spain_cities['pop'] >= 50000) & (spain_cities['pop'] < 100000)]),
        'small_cities': build_tree(spain_cities[(spain_cities['pop'] >= 10000) & (spain_cities['pop'] < 50000)]),
    }
    return _CITIES_CACHE


# Known resort regions (bounding boxes)
RESORT_REGIONS = {
    'Balearic Islands': {'lat_range': (38.5, 40.5), 'lon_range': (1.0, 4.5)},
    'Canary Islands': {'lat_range': (27.5, 29.5), 'lon_range': (-18.5, -13.0)},
    'Costa del Sol': {'lat_range': (36.0, 36.9), 'lon_range': (-5.5, -3.5)},
    'Costa Brava': {'lat_range': (41.5, 42.5), 'lon_range': (2.5, 3.5)},
    'Costa Blanca': {'lat_range': (37.5, 39.0), 'lon_range': (-1.0, 0.5)},
}


def _vectorized_min_distance(lats: np.ndarray, lons: np.ndarray, tree_tuple) -> np.ndarray:
    """Vectorized minimum distance to any city in a tier using KD-tree."""
    tree, tier_df = tree_tuple
    n = len(lats)
    
    if tree is None:
        return np.full(n, np.inf)
    
    # Handle NaN values - KD-tree can't handle them
    valid = ~(np.isnan(lats) | np.isnan(lons))
    distances = np.full(n, np.inf)
    
    if np.any(valid):
        # Convert hotel coords to km for valid entries only
        coords = np.column_stack([lats[valid] * 111, lons[valid] * 85])
        dist_valid, _ = tree.query(coords, k=1)
        distances[valid] = dist_valid
    
    return distances


def _vectorized_resort_check(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized check if coordinates are in a known resort region."""
    # Handle NaN values
    valid = ~(np.isnan(lats) | np.isnan(lons))
    is_resort = np.zeros(len(lats), dtype=bool)
    
    for bounds in RESORT_REGIONS.values():
        in_region = (
            valid &
            (lats >= bounds['lat_range'][0]) & (lats <= bounds['lat_range'][1]) &
            (lons >= bounds['lon_range'][0]) & (lons <= bounds['lon_range'][1])
        )
        is_resort |= in_region
    return is_resort


def get_market_segments_vectorized(
    lats: np.ndarray,
    lons: np.ndarray,
    dist_coast: np.ndarray = None
) -> np.ndarray:
    """
    Vectorized classification of hotels into granular market segments.
    
    Segments (in priority order):
    1. major_metro: Within 30km of Madrid/Barcelona/Valencia/Sevilla/Málaga/Zaragoza
    2. urban_core: Within 10km of city with pop 100k-500k
    3. urban_fringe: 10-30km from city with pop >100k
    4. resort_coastal: In known resort area + near coast (<30km)
    5. coastal_town: Near coast (<20km) but not in resort region
    6. provincial_city: Within 15km of city with pop 50k-100k
    7. small_town: Within 15km of city with pop 10k-50k
    8. rural: Everything else
    
    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        dist_coast: Array of distances to coast in km (optional)
    
    Returns:
        Array of segment strings
    """
    n = len(lats)
    segments = np.full(n, 'rural', dtype=object)
    
    # Handle missing data
    valid = ~(np.isnan(lats) | np.isnan(lons))
    if not np.any(valid):
        segments[:] = 'unknown'
        return segments
    
    # Load cities with KD-trees
    cities = _load_cities_with_kdtrees()
    if cities is None:
        segments[~valid] = 'unknown'
        return segments
    
    # Compute distances to each city tier (vectorized)
    dist_to_metro = _vectorized_min_distance(lats, lons, cities['major_metros'])
    dist_to_large = _vectorized_min_distance(lats, lons, cities['large_cities'])
    dist_to_medium = _vectorized_min_distance(lats, lons, cities['medium_cities'])
    dist_to_small = _vectorized_min_distance(lats, lons, cities['small_cities'])
    
    # Resort region check (vectorized)
    is_resort = _vectorized_resort_check(lats, lons)
    
    # Distance to coast (use provided or default)
    if dist_coast is None:
        dist_coast = np.full(n, 100.0)  # Default far from coast
    dist_coast = np.where(np.isnan(dist_coast), 100.0, dist_coast)
    
    # Apply priority rules (vectorized, in reverse order so higher priority overwrites)
    # 8. Rural (default)
    # 7. Small Town
    segments = np.where(dist_to_small <= 15, 'small_town', segments)
    # 6. Provincial City
    segments = np.where(dist_to_medium <= 15, 'provincial_city', segments)
    # 5. Coastal Town
    segments = np.where(dist_coast <= 20, 'coastal_town', segments)
    # 4. Resort Coastal
    segments = np.where(is_resort & (dist_coast <= 30), 'resort_coastal', segments)
    # 3. Urban Fringe
    segments = np.where((dist_to_large > 10) & (dist_to_large <= 30), 'urban_fringe', segments)
    # 2. Urban Core
    segments = np.where(dist_to_large <= 10, 'urban_core', segments)
    # 1. Major Metro (highest priority)
    segments = np.where(dist_to_metro <= 30, 'major_metro', segments)
    
    # Mark invalid as unknown
    segments = np.where(~valid, 'unknown', segments)
    
    return segments


def add_granular_segment(df: pd.DataFrame, column_name: str = 'market_segment') -> pd.DataFrame:
    """
    Add granular market segment to DataFrame (VECTORIZED).
    
    Requires 'latitude', 'longitude' columns.
    Optionally uses 'distance_from_coast' if available.
    
    Args:
        df: DataFrame with latitude/longitude columns
        column_name: Name of output column (default: 'market_segment')
    
    Returns:
        DataFrame with market_segment column added
    """
    df = df.copy()
    
    lats = df['latitude'].values
    lons = df['longitude'].values
    dist_coast = df.get('distance_from_coast', pd.Series([None] * len(df))).values
    
    df[column_name] = get_market_segments_vectorized(lats, lons, dist_coast)
    # Backwards compatibility alias
    if column_name == 'market_segment':
        df['market_segment_v2'] = df['market_segment']
    return df


# Alias for backwards compatibility
def get_market_segment_v2(
    latitude: float,
    longitude: float,
    distance_coast_km: float = None
) -> str:
    """Single-hotel version of granular segment (calls vectorized version)."""
    result = get_market_segments_vectorized(
        np.array([latitude]),
        np.array([longitude]),
        np.array([distance_coast_km]) if distance_coast_km else None
    )
    return result[0]


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def add_temporal_features(df: pd.DataFrame, date_col: str = 'month') -> pd.DataFrame:
    """
    Add temporal features to DataFrame.
    
    Args:
        df: DataFrame with a date column
        date_col: Name of the date column
    
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Month number
    df['month_number'] = df[date_col].dt.month
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month_number'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_number'] / 12)
    
    # Season flags
    df['is_summer'] = df['month_number'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month_number'].isin([12, 1, 2]).astype(int)
    df['is_july_august'] = df['month_number'].isin([7, 8]).astype(int)  # Peak summer
    
    # Day of week (if date has day info)
    if df[date_col].dt.day.notna().all():
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def add_view_quality(df: pd.DataFrame, view_col: str = 'room_view') -> pd.DataFrame:
    """
    Add ordinal view quality score.
    
    Args:
        df: DataFrame with room_view column
        view_col: Name of the view column
    
    Returns:
        DataFrame with view_quality_ordinal column
    """
    df = df.copy()
    df['view_quality_ordinal'] = df[view_col].map(VIEW_QUALITY_MAP).fillna(0)
    return df


def engineer_features(
    df: pd.DataFrame,
    city_centers: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Apply all standard feature engineering.
    
    This applies the validated feature transformations from XGBoost analysis:
    - Geographic: dist_center_km (to city center), is_madrid_metro, is_coastal
    - Product: log_room_size, amenities_score, view_quality_ordinal
    - Temporal: month_sin, month_cos, is_summer, is_winter
    
    Args:
        df: Raw hotel data with lat/lon, city, room info
        city_centers: Optional pre-computed city centers for dist_center_km
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Standardize city first (needed for geographic features)
    if 'city' in df.columns:
        df['city_standardized'] = df['city'].apply(standardize_city)
    
    # Geographic features (dist_center_km, is_madrid_metro, is_coastal)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = add_geographic_features(df, city_centers)
    
    # Temporal features
    if 'month' in df.columns:
        df = add_temporal_features(df, 'month')
    
    # View quality
    if 'room_view' in df.columns:
        df = add_view_quality(df)
    
    # Product features
    df = add_amenities_score(df)
    
    # Log transforms
    if 'room_size' in df.columns:
        df['log_room_size'] = np.log1p(df['room_size'].fillna(20))
    elif 'avg_room_size' in df.columns:
        df['log_room_size'] = np.log1p(df['avg_room_size'].fillna(20))
    
    if 'total_rooms' in df.columns:
        df['log_total_rooms'] = np.log1p(df['total_rooms'].fillna(10))
        df['total_capacity_log'] = df['log_total_rooms']  # Alias for compatibility
    
    if 'max_occupancy' in df.columns:
        df['room_capacity_pax'] = df['max_occupancy']
    
    return df


def engineer_validated_features(
    df: pd.DataFrame,
    city_centers: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Engineer the exact features validated by XGBoost (R² = 0.71).
    
    This is the canonical function for feature engineering.
    Call this to ensure you're using the validated feature set.
    
    Validated features:
    - Geographic: dist_center_km, is_madrid_metro, dist_coast_log, is_coastal
    - Product: log_room_size, room_capacity_pax, amenities_score, view_quality_ordinal
    - Temporal: month_sin, month_cos, is_summer, is_winter
    - City: city_standardized
    
    Args:
        df: Raw hotel data
        city_centers: Pre-computed city centers (optional)
    
    Returns:
        DataFrame with validated features
    """
    return engineer_features(df, city_centers)

