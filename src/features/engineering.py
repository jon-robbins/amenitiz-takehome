"""
Feature engineering for price recommendation.

Core transformations validated by XGBoost (R² = 0.71) in elasticity analysis.
"""

import numpy as np
import pandas as pd
import re
from typing import Optional


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

def get_market_segment(
    distance_coast_km: float,
    distance_madrid_km: float
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
    
    Returns:
        Market segment: 'coastal', 'madrid_metro', or 'provincial'.
    """
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all standard feature engineering.
    
    Args:
        df: Raw hotel data
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Standardize city
    if 'city' in df.columns:
        df['city_standardized'] = df['city'].apply(standardize_city)
    
    # Temporal features
    if 'month' in df.columns:
        df = add_temporal_features(df, 'month')
    
    # View quality
    if 'room_view' in df.columns:
        df = add_view_quality(df)
    
    # Log transforms
    if 'room_size' in df.columns:
        df['log_room_size'] = np.log1p(df['room_size'].fillna(20))
    
    if 'total_rooms' in df.columns:
        df['log_total_rooms'] = np.log1p(df['total_rooms'].fillna(10))
    
    return df

