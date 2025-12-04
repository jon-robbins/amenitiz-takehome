"""
Feature engineering module for price recommendation.

Extracts validated features for any (hotel_id, date) pair.
Features validated by XGBoost (R² = 0.71) in elasticity analysis.
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from lib.holiday_features import (
    map_hotels_to_admin1,
    add_date_holiday_features,
    is_holiday,
    is_near_holiday
)


# Feature groups for documentation and validation
GEOGRAPHIC_FEATURES = [
    'dist_center_km',
    'dist_coast_log', 
    'is_coastal',
    'city_standardized'
]

PRODUCT_FEATURES = [
    'log_room_size',
    'room_capacity_pax',
    'amenities_score',
    'view_quality_ordinal',
    'room_type',
    'room_view',
    'children_allowed'
]

TEMPORAL_FEATURES = [
    'month_sin',
    'month_cos',
    'day_of_week',
    'is_weekend',
    'is_summer',
    'is_winter',
    'is_holiday',
    'is_holiday_pm1',
    'is_holiday_pm2'
]

CAPACITY_FEATURES = [
    'total_capacity',
    'total_capacity_log'
]

# All numeric features for scaling
NUMERIC_FEATURES = [
    'dist_center_km', 'dist_coast_log', 'log_room_size', 'room_capacity_pax',
    'amenities_score', 'view_quality_ordinal', 'total_capacity', 'total_capacity_log',
    'month_sin', 'month_cos', 'weekend_ratio'
]

# Categorical features for encoding
CATEGORICAL_FEATURES = ['room_type', 'room_view', 'city_standardized']

# Boolean features
BOOLEAN_FEATURES = [
    'is_coastal', 'is_summer', 'is_winter', 'children_allowed',
    'is_holiday', 'is_holiday_pm1', 'is_holiday_pm2'
]

# Top 5 cities by revenue (canonical mapping)
TOP_5_CITIES = {
    'madrid': 'madrid',
    'barcelona': 'barcelona',
    'sevilla': 'sevilla',
    'malaga': 'malaga',
    'málaga': 'malaga',
    'toledo': 'toledo'
}

# View quality mapping (ordinal 0-3)
VIEW_QUALITY_MAP = {
    'ocean_view': 3, 'sea_view': 3,
    'lake_view': 2, 'mountain_view': 2,
    'pool_view': 1, 'garden_view': 1,
    'city_view': 0, 'no_view': 0
}


def clean_city_name(name: str) -> str:
    """
    Cleans city name for standardization.
    
    Removes punctuation, converts to lowercase, normalizes whitespace.
    """
    if pd.isna(name):
        return ''
    cleaned = re.sub(r'[^\w\s]', '', str(name).lower().strip())
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def standardize_city(city_str: str) -> str:
    """
    Standardizes city to one of top 5 or 'other'.
    
    Uses fuzzy matching to handle variations like 'Málaga' vs 'malaga'.
    """
    if pd.isna(city_str):
        return 'other'
    
    city_clean = clean_city_name(city_str)
    
    if city_clean in TOP_5_CITIES:
        return TOP_5_CITIES[city_clean]
    
    for canonical_key in TOP_5_CITIES.keys():
        if canonical_key in city_clean:
            return TOP_5_CITIES[canonical_key]
    
    return 'other'


def engineer_temporal_features(df: pd.DataFrame, date_col: str = 'month') -> pd.DataFrame:
    """
    Engineers temporal features from date column.
    
    Creates cyclical month encoding and seasonal flags.
    """
    df = df.copy()
    
    # Extract month number if needed
    if df[date_col].dtype == 'datetime64[ns]' or 'datetime' in str(df[date_col].dtype):
        df['month_number'] = pd.to_datetime(df[date_col]).dt.month
    elif 'month_number' not in df.columns:
        df['month_number'] = pd.to_datetime(df[date_col]).dt.month
    
    # Cyclical encoding (captures December-January continuity)
    df['month_sin'] = np.sin(2 * np.pi * df['month_number'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_number'] / 12)
    
    # Seasonal flags
    df['is_summer'] = df['month_number'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month_number'].isin([12, 1, 2]).astype(int)
    
    return df


def engineer_holiday_features(
    df: pd.DataFrame,
    cities500_path: Optional[Path] = None,
    date_col: str = 'date',
    hotel_id_col: str = 'hotel_id'
) -> pd.DataFrame:
    """
    Engineers holiday-related features based on date and hotel location.
    
    Creates boolean features:
    - is_holiday: True if date is exactly a Spanish regional holiday
    - is_holiday_pm1: True if date is within ±1 day of a holiday
    - is_holiday_pm2: True if date is within ±2 days of a holiday
    
    Args:
        df: DataFrame with hotel_id and date columns
        cities500_path: Path to cities500.json for hotel-to-region mapping
        date_col: Name of date column
        hotel_id_col: Name of hotel ID column
    
    Returns:
        DataFrame with holiday boolean features added
    """
    df = df.copy()
    
    # Check if we have the required columns
    if date_col not in df.columns:
        # Try common alternatives
        for alt_col in ['arrival_date', 'month', 'booking_date']:
            if alt_col in df.columns:
                date_col = alt_col
                break
        else:
            # No date column found, set defaults to 0
            df['is_holiday'] = 0
            df['is_holiday_pm1'] = 0
            df['is_holiday_pm2'] = 0
            return df
    
    # If cities500_path provided, map hotels to regions
    if cities500_path is not None and 'latitude' in df.columns and 'longitude' in df.columns:
        # Get unique hotels with coordinates
        hotels = df[[hotel_id_col, 'latitude', 'longitude']].drop_duplicates()
        hotel_admin1 = map_hotels_to_admin1(
            hotels,
            cities500_path,
            lat_col='latitude',
            lon_col='longitude',
            hotel_id_col=hotel_id_col
        )
        
        # Add holiday features
        df = add_date_holiday_features(
            df,
            hotel_admin1,
            date_col=date_col,
            hotel_id_col=hotel_id_col
        )
    else:
        # No location data - use national holidays (MD as proxy)
        df['subdiv_code'] = 'MD'
        df = add_date_holiday_features(
            df,
            pd.DataFrame({hotel_id_col: df[hotel_id_col].unique(), 'subdiv_code': 'MD'}),
            date_col=date_col,
            hotel_id_col=hotel_id_col
        )
    
    return df


def engineer_geographic_features(
    df: pd.DataFrame, 
    distance_features: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Engineers geographic features.
    
    Requires distance_features DataFrame with columns:
    - hotel_id, distance_from_coast, distance_from_madrid
    """
    df = df.copy()
    
    # Merge distance features if provided
    if distance_features is not None:
        df = df.merge(distance_features, on='hotel_id', how='left')
    
    # Coastal flag (within 20km of coast)
    if 'distance_from_coast' in df.columns:
        df['is_coastal'] = (df['distance_from_coast'] < 20).astype(int)
        df['dist_coast_log'] = np.log1p(df['distance_from_coast'])
    
    # Distance to city center (requires lat/lon and city centroids)
    if all(col in df.columns for col in ['latitude', 'longitude', 'city_lat', 'city_lon']):
        df['dist_center_km'] = np.sqrt(
            (df['latitude'] - df['city_lat'])**2 + 
            (df['longitude'] - df['city_lon'])**2
        ) * 111  # Rough conversion to km
    
    # Standardize city names
    if 'city' in df.columns:
        df['city_standardized'] = df['city'].apply(standardize_city)
    
    return df


def engineer_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers product features from room and hotel attributes.
    """
    df = df.copy()
    
    # Log-transform room size
    if 'avg_room_size' in df.columns:
        df['log_room_size'] = np.log1p(df['avg_room_size'])
    elif 'room_size' in df.columns:
        df['log_room_size'] = np.log1p(df['room_size'])
    
    # Capacity features
    if 'total_capacity' in df.columns:
        df['total_capacity_log'] = np.log1p(df['total_capacity'])
    
    # View quality ordinal
    if 'room_view' in df.columns:
        df['view_quality_ordinal'] = df['room_view'].map(VIEW_QUALITY_MAP).fillna(0)
    
    # Amenities score (if component columns exist)
    amenity_cols = ['events_allowed', 'pets_allowed', 'smoking_allowed', 'children_allowed']
    if all(col in df.columns for col in amenity_cols):
        df['amenities_score'] = sum(df[col].astype(int) for col in amenity_cols)
    
    return df


def add_capacity_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds capacity quartile bins based on total_capacity (number of rooms).
    
    This replaces revenue_quartile because:
    1. New hotels have no revenue history
    2. Capacity is available at listing time
    3. Capacity is a strong proxy for business scale
    """
    if 'total_capacity' not in df.columns:
        df['capacity_quartile'] = 'Q2'  # Default to middle quartile
        return df
    
    hotel_capacity = df.groupby('hotel_id')['total_capacity'].first().reset_index()
    hotel_capacity.columns = ['hotel_id', 'hotel_capacity']
    
    try:
        # Try to create quartiles
        hotel_capacity['capacity_quartile'] = pd.qcut(
            hotel_capacity['hotel_capacity'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'
        )
    except ValueError:
        # If quartiles fail (e.g., too many duplicates), use manual bins
        try:
            # Use percentile-based binning without labels first
            hotel_capacity['capacity_quartile'] = pd.qcut(
                hotel_capacity['hotel_capacity'].rank(method='first'),
                q=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
        except ValueError:
            # Last resort: just assign all to Q2
            hotel_capacity['capacity_quartile'] = 'Q2'
    
    return df.merge(hotel_capacity[['hotel_id', 'capacity_quartile']], on='hotel_id', how='left')


def engineer_all_features(
    df: pd.DataFrame,
    distance_features: Optional[pd.DataFrame] = None,
    cities500_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Engineers all validated features for price prediction.
    
    Args:
        df: DataFrame with booking/hotel data
        distance_features: Optional DataFrame with hotel distance features
        cities500_path: Optional path to cities500.json for holiday features
    
    Returns:
        DataFrame with all feature columns populated.
    """
    df = engineer_temporal_features(df)
    df = engineer_geographic_features(df, distance_features)
    df = engineer_product_features(df)
    df = add_capacity_quartiles(df)
    df = engineer_holiday_features(df, cities500_path=cities500_path)
    
    return df


def get_feature_columns() -> Dict[str, List[str]]:
    """
    Returns dictionary of feature column groups.
    
    Useful for feature selection and documentation.
    """
    return {
        'numeric': NUMERIC_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'boolean': BOOLEAN_FEATURES,
        'geographic': GEOGRAPHIC_FEATURES,
        'product': PRODUCT_FEATURES,
        'temporal': TEMPORAL_FEATURES,
        'capacity': CAPACITY_FEATURES
    }


def validate_features(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that required features are present and non-null.
    
    Returns:
        Tuple of (is_valid, missing_features)
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        return False, missing
    
    # Check for excessive nulls
    null_pct = df[required].isnull().mean()
    high_null = null_pct[null_pct > 0.5].index.tolist()
    
    if high_null:
        return False, [f"{col} (>50% null)" for col in high_null]
    
    return True, []

