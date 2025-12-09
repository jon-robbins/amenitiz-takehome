"""
Holiday feature engineering for Spanish hotels.

Maps hotels to autonomous communities using geospatial coordinates and computes
holiday proximity features using the holidays library.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import holidays
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# Mapping from cities500.json admin1 names to holidays library subdivision codes
ADMIN1_TO_SUBDIV = {
    'Andalusia': 'AN',
    'Aragon': 'AR',
    'Asturias': 'AS',
    'Balearic Islands': 'IB',
    'Basque Country': 'PV',
    'Canary Islands': 'CN',
    'Cantabria': 'CB',
    'Castille and León': 'CL',
    'Castille-La Mancha': 'CM',
    'Catalonia': 'CT',
    'Ceuta': 'CE',
    'Extremadura': 'EX',
    'Galicia': 'GA',
    'La Rioja': 'RI',
    'Madrid': 'MD',
    'Melilla': 'ML',
    'Murcia': 'MC',
    'Navarre': 'NC',
    'Valencia': 'VC'
}


def load_spanish_cities(cities500_path: str | Path) -> pd.DataFrame:
    """
    Loads Spanish cities from cities500.json with their coordinates and admin1.
    
    Returns DataFrame with columns: name, lat, lon, admin1, subdiv_code
    """
    with open(cities500_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter to Spanish cities only
    es_cities = [c for c in data if c.get('country') == 'ES' and c.get('admin1')]
    
    df = pd.DataFrame(es_cities)
    df = df[['name', 'lat', 'lon', 'admin1']].copy()
    
    # Map admin1 to subdivision codes
    df['subdiv_code'] = df['admin1'].map(ADMIN1_TO_SUBDIV)
    
    # Drop rows where we couldn't map admin1 (shouldn't happen, but safety)
    df = df.dropna(subset=['subdiv_code'])
    
    return df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes Haversine distance between two points in kilometers.
    """
    R = 6371  # Earth's radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def map_hotels_to_admin1(
    hotels_df: pd.DataFrame,
    cities500_path: str | Path,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    hotel_id_col: str = 'hotel_id'
) -> pd.DataFrame:
    """
    Maps hotels to their nearest Spanish city's admin1 region using coordinates.
    
    Uses a KDTree for efficient nearest-neighbor lookup.
    
    Args:
        hotels_df: DataFrame with hotel_id, latitude, longitude columns
        cities500_path: Path to cities500.json
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        hotel_id_col: Name of hotel ID column
    
    Returns:
        DataFrame with hotel_id, admin1, subdiv_code columns
    """
    # Load Spanish cities
    cities_df = load_spanish_cities(cities500_path)
    
    # Build KDTree from city coordinates
    # Note: Using Euclidean approximation on lat/lon is acceptable for nearby points
    # in the same country (errors are small within Spain's extent)
    city_coords = cities_df[['lat', 'lon']].values
    tree = cKDTree(city_coords)
    
    # Filter hotels with valid coordinates
    valid_hotels = hotels_df[
        hotels_df[lat_col].notna() & 
        hotels_df[lon_col].notna()
    ].copy()
    
    if len(valid_hotels) == 0:
        return pd.DataFrame(columns=[hotel_id_col, 'admin1', 'subdiv_code'])
    
    # Query nearest city for each hotel
    hotel_coords = valid_hotels[[lat_col, lon_col]].values
    distances, indices = tree.query(hotel_coords)
    
    # Build result DataFrame
    result = pd.DataFrame({
        hotel_id_col: valid_hotels[hotel_id_col].values,
        'admin1': cities_df.iloc[indices]['admin1'].values,
        'subdiv_code': cities_df.iloc[indices]['subdiv_code'].values,
        'nearest_city': cities_df.iloc[indices]['name'].values,
        'distance_to_city_km': [
            haversine_distance(
                hotel_coords[i, 0], hotel_coords[i, 1],
                city_coords[indices[i], 0], city_coords[indices[i], 1]
            )
            for i in range(len(hotel_coords))
        ]
    })
    
    return result


def get_holiday_dates(year: int, subdiv: str, buffer_days: int = 2) -> set:
    """
    Returns set of all dates within buffer_days of a holiday for given year/subdivision.
    
    Args:
        year: Year to get holidays for
        subdiv: Spanish subdivision code (e.g., 'MD' for Madrid)
        buffer_days: Number of days before/after holiday to include
    
    Returns:
        Set of date objects that are within buffer_days of a holiday
    """
    try:
        es_holidays = holidays.ES(years=year, subdiv=subdiv)
    except Exception:
        # Fallback to national holidays if subdivision fails
        es_holidays = holidays.ES(years=year)
    
    holiday_adjacent_dates = set()
    
    for holiday_date in es_holidays.keys():
        for delta in range(-buffer_days, buffer_days + 1):
            adjacent_date = holiday_date + timedelta(days=delta)
            # Only include dates from the target year
            if adjacent_date.year == year:
                holiday_adjacent_dates.add(adjacent_date)
    
    return holiday_adjacent_dates


def compute_holiday_ratio(
    month_start: date,
    subdiv: str,
    buffer_days: int = 2
) -> float:
    """
    Computes the proportion of days in a month within buffer_days of a regional holiday.
    
    Args:
        month_start: First day of the month (any date in the month works too)
        subdiv: Spanish subdivision code (e.g., 'MD' for Madrid)
        buffer_days: Number of days before/after holiday to consider
    
    Returns:
        Float between 0.0 and 1.0 representing proportion of holiday-adjacent days
    """
    # Normalize to first of month
    year = month_start.year
    month = month_start.month
    
    # Get all days in this month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    
    first_day = date(year, month, 1)
    days_in_month = (next_month - first_day).days
    
    # Get holiday-adjacent dates for this year (and adjacent year if needed for buffer)
    holiday_dates = get_holiday_dates(year, subdiv, buffer_days)
    
    # Also check previous/next year holidays that might affect this month's edge days
    if month == 1:
        holiday_dates |= get_holiday_dates(year - 1, subdiv, buffer_days)
    if month == 12:
        holiday_dates |= get_holiday_dates(year + 1, subdiv, buffer_days)
    
    # Count days in month that are holiday-adjacent
    holiday_days = 0
    current_day = first_day
    while current_day < next_month:
        if current_day in holiday_dates:
            holiday_days += 1
        current_day += timedelta(days=1)
    
    return holiday_days / days_in_month


def add_holiday_features(
    df: pd.DataFrame,
    hotel_admin1_map: pd.DataFrame,
    month_col: str = 'month',
    hotel_id_col: str = 'hotel_id',
    buffer_days: int = 2
) -> pd.DataFrame:
    """
    Adds holiday_ratio feature to a hotel-month DataFrame.
    
    Args:
        df: DataFrame with hotel_id and month columns
        hotel_admin1_map: DataFrame from map_hotels_to_admin1()
        month_col: Name of month column (datetime)
        hotel_id_col: Name of hotel ID column
        buffer_days: Days before/after holiday to consider
    
    Returns:
        DataFrame with added 'holiday_ratio' column
    """
    df = df.copy()
    
    # Merge subdivision codes
    df = df.merge(
        hotel_admin1_map[[hotel_id_col, 'subdiv_code']],
        on=hotel_id_col,
        how='left'
    )
    
    # Compute holiday ratio for each row
    def calc_ratio(row):
        if pd.isna(row['subdiv_code']):
            return np.nan
        
        month_date = row[month_col]
        if isinstance(month_date, pd.Timestamp):
            month_date = month_date.date()
        
        return compute_holiday_ratio(month_date, row['subdiv_code'], buffer_days)
    
    df['holiday_ratio'] = df.apply(calc_ratio, axis=1)
    
    # Fill NaN with national average (using MD as proxy, or 0)
    if df['holiday_ratio'].isna().any():
        # For hotels without region mapping, use Madrid as fallback
        df['holiday_ratio'] = df['holiday_ratio'].fillna(
            df.apply(
                lambda row: compute_holiday_ratio(
                    row[month_col].date() if isinstance(row[month_col], pd.Timestamp) 
                    else row[month_col],
                    'MD',
                    buffer_days
                ) if pd.isna(row['holiday_ratio']) else row['holiday_ratio'],
                axis=1
            )
        )
    
    return df


# ============================================================================
# Date-level holiday features (for ml_pipeline)
# ============================================================================

def is_holiday(check_date: date, subdiv: str) -> bool:
    """
    Checks if a specific date is a holiday in the given subdivision.
    
    Args:
        check_date: Date to check
        subdiv: Spanish subdivision code (e.g., 'MD' for Madrid)
    
    Returns:
        True if the date is a holiday, False otherwise
    """
    try:
        es_holidays = holidays.ES(years=check_date.year, subdiv=subdiv)
    except Exception:
        es_holidays = holidays.ES(years=check_date.year)
    
    return check_date in es_holidays


def is_near_holiday(check_date: date, subdiv: str, buffer_days: int = 1) -> bool:
    """
    Checks if a date is within buffer_days of a holiday.
    
    Args:
        check_date: Date to check
        subdiv: Spanish subdivision code (e.g., 'MD' for Madrid)
        buffer_days: Number of days before/after holiday to consider
    
    Returns:
        True if within buffer_days of a holiday, False otherwise
    """
    holiday_dates = get_holiday_dates(check_date.year, subdiv, buffer_days)
    
    # Also check adjacent years if near year boundary
    if check_date.month == 1:
        holiday_dates |= get_holiday_dates(check_date.year - 1, subdiv, buffer_days)
    if check_date.month == 12:
        holiday_dates |= get_holiday_dates(check_date.year + 1, subdiv, buffer_days)
    
    return check_date in holiday_dates


def get_date_holiday_features(
    check_date: date, 
    subdiv: str
) -> Dict[str, bool]:
    """
    Returns all holiday-related boolean features for a specific date.
    
    Args:
        check_date: Date to check
        subdiv: Spanish subdivision code (e.g., 'MD' for Madrid)
    
    Returns:
        Dict with keys: is_holiday, is_holiday_pm1, is_holiday_pm2
    """
    return {
        'is_holiday': is_holiday(check_date, subdiv),
        'is_holiday_pm1': is_near_holiday(check_date, subdiv, buffer_days=1),
        'is_holiday_pm2': is_near_holiday(check_date, subdiv, buffer_days=2)
    }


def build_holiday_lookup(years: list, subdivs: list) -> Dict[tuple, Dict[str, bool]]:
    """
    Pre-computes a lookup dict of (date, subdiv) -> holiday flags.
    
    This is MUCH faster than computing per-row.
    """
    lookup = {}
    
    for subdiv in subdivs:
        for year in years:
            try:
                es_holidays = holidays.ES(years=year, subdiv=subdiv)
            except Exception:
                es_holidays = holidays.ES(years=year)
            
            holiday_dates = set(es_holidays.keys())
            
            # Generate all dates in year
            start = date(year, 1, 1)
            end = date(year, 12, 31)
            current = start
            
            while current <= end:
                is_hol = current in holiday_dates
                
                # Check ±1 day
                is_pm1 = any(
                    (current + timedelta(days=d)) in holiday_dates 
                    for d in [-1, 0, 1]
                )
                
                # Check ±2 days
                is_pm2 = any(
                    (current + timedelta(days=d)) in holiday_dates 
                    for d in [-2, -1, 0, 1, 2]
                )
                
                lookup[(current, subdiv)] = {
                    'is_holiday': is_hol,
                    'is_holiday_pm1': is_pm1,
                    'is_holiday_pm2': is_pm2
                }
                
                current += timedelta(days=1)
    
    return lookup


def add_date_holiday_features(
    df: pd.DataFrame,
    hotel_admin1_map: pd.DataFrame,
    date_col: str = 'date',
    hotel_id_col: str = 'hotel_id'
) -> pd.DataFrame:
    """
    Adds date-level holiday boolean features to a DataFrame.
    
    Uses vectorized lookup for speed (handles 700k+ rows efficiently).
    
    Args:
        df: DataFrame with hotel_id and date columns
        hotel_admin1_map: DataFrame from map_hotels_to_admin1()
        date_col: Name of date column
        hotel_id_col: Name of hotel ID column
    
    Returns:
        DataFrame with added is_holiday, is_holiday_pm1, is_holiday_pm2 columns
    """
    df = df.copy()
    
    # Merge subdivision codes
    if 'subdiv_code' not in df.columns:
        df = df.merge(
            hotel_admin1_map[[hotel_id_col, 'subdiv_code']],
            on=hotel_id_col,
            how='left'
        )
    
    # Default to Madrid (MD) for hotels without mapping
    df['subdiv_code'] = df['subdiv_code'].fillna('MD')
    
    # Convert dates to date objects
    dates = pd.to_datetime(df[date_col]).dt.date
    
    # Get unique years and subdivisions
    years = list(dates.apply(lambda d: d.year).unique())
    subdivs = list(df['subdiv_code'].unique())
    
    print(f"   Building holiday lookup for {len(years)} years, {len(subdivs)} regions...", flush=True)
    
    # Build lookup table (one-time cost)
    lookup = build_holiday_lookup(years, subdivs)
    
    print(f"   Mapping {len(df):,} rows...", flush=True)
    
    # Vectorized lookup using tuple keys
    keys = list(zip(dates, df['subdiv_code']))
    
    # Map to holiday features
    df['is_holiday'] = [lookup.get((d, s), {}).get('is_holiday', False) for d, s in keys]
    df['is_holiday_pm1'] = [lookup.get((d, s), {}).get('is_holiday_pm1', False) for d, s in keys]
    df['is_holiday_pm2'] = [lookup.get((d, s), {}).get('is_holiday_pm2', False) for d, s in keys]
    
    # Convert to int
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['is_holiday_pm1'] = df['is_holiday_pm1'].astype(int)
    df['is_holiday_pm2'] = df['is_holiday_pm2'].astype(int)
    
    return df


# ============================================================================
# Convenience function for feature_importance_validation.py integration
# ============================================================================

def get_hotel_holiday_features(
    hotel_month_df: pd.DataFrame,
    cities500_path: str | Path,
    hotel_id_col: str = 'hotel_id',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    month_col: str = 'month',
    buffer_days: int = 2
) -> pd.DataFrame:
    """
    One-stop function to add holiday features to a hotel-month DataFrame.
    
    Combines hotel-to-admin1 mapping and holiday ratio computation.
    
    Args:
        hotel_month_df: DataFrame with hotel_id, latitude, longitude, month
        cities500_path: Path to cities500.json
        hotel_id_col: Name of hotel ID column
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        month_col: Name of month column
        buffer_days: Days before/after holiday to consider
    
    Returns:
        DataFrame with added 'holiday_ratio' and 'subdiv_code' columns
    """
    # Extract unique hotels with coordinates
    hotels = hotel_month_df[[hotel_id_col, lat_col, lon_col]].drop_duplicates()
    
    # Map to admin1 regions
    hotel_admin1 = map_hotels_to_admin1(
        hotels,
        cities500_path,
        lat_col=lat_col,
        lon_col=lon_col,
        hotel_id_col=hotel_id_col
    )
    
    # Add holiday features
    result = add_holiday_features(
        hotel_month_df,
        hotel_admin1,
        month_col=month_col,
        hotel_id_col=hotel_id_col,
        buffer_days=buffer_days
    )
    
    return result


# ============================================================================
# Testing / Demo
# ============================================================================

if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    
    cities_path = Path(__file__).parent.parent / "data" / "cities500.json"
    
    # Test loading cities
    cities = load_spanish_cities(cities_path)
    print(f"Loaded {len(cities)} Spanish cities")
    print(f"Sample cities:\n{cities.head()}")
    
    # Test holiday ratio calculation
    test_month = date(2023, 8, 1)  # August - peak summer
    test_subdiv = 'MD'  # Madrid
    ratio = compute_holiday_ratio(test_month, test_subdiv, buffer_days=2)
    print(f"\nHoliday ratio for Madrid, August 2023: {ratio:.3f}")
    
    # Test December (more holidays expected)
    test_month = date(2023, 12, 1)
    ratio = compute_holiday_ratio(test_month, test_subdiv, buffer_days=2)
    print(f"Holiday ratio for Madrid, December 2023: {ratio:.3f}")

