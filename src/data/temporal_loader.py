"""
Temporal-aware data loading for price recommendations.

Provides functions to query booking data with temporal constraints,
ensuring we don't "peek into the future" when making predictions.

Key constraint: For a prediction made on `as_of_date` for `target_dates`,
we can only see bookings where `created_at <= as_of_date`.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from .loader import get_clean_connection


@dataclass
class HotelProfile:
    """
    Profile for a hotel (used for cold-start recommendations).
    
    Attributes:
        lat: Latitude coordinate
        lon: Longitude coordinate
        room_type: Primary room type (e.g., 'room', 'apartment', 'villa')
        room_size: Average room size in square meters
        amenities: List of amenities (e.g., ['pool', 'spa', 'parking'])
        num_rooms: Total number of rooms/units available
    """
    lat: float
    lon: float
    room_type: str
    room_size: float
    amenities: List[str]
    num_rooms: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'latitude': self.lat,
            'longitude': self.lon,
            'room_type': self.room_type,
            'room_size': self.room_size,
            'amenities': ','.join(self.amenities),
            'num_rooms': self.num_rooms
        }


@dataclass
class PeerMetrics:
    """
    Aggregated metrics for a peer hotel or peer group.
    
    Attributes:
        hotel_id: Hotel identifier (None for aggregated peer group)
        avg_price: Average daily rate (ADR)
        occupancy_rate: Rooms sold / rooms available
        revpar: Revenue per available room (price × occupancy)
        room_type: Primary room type
        room_size: Average room size
        distance_km: Distance from target hotel (for geographic peers)
        similarity_score: Weighted similarity score (0-1)
    """
    hotel_id: Optional[int]
    avg_price: float
    occupancy_rate: float
    revpar: float
    room_type: str
    room_size: float
    distance_km: Optional[float] = None
    similarity_score: Optional[float] = None
    n_bookings: int = 0


def load_bookings_as_of(
    con: duckdb.DuckDBPyConnection,
    target_dates: List[date],
    as_of_date: date,
    hotel_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load bookings visible as of a specific date for target arrival dates.
    
    This is the core temporal filtering function. It ensures we only see
    bookings that existed at the time the recommendation was made.
    
    Args:
        con: DuckDB connection with loaded data
        target_dates: List of arrival dates to get bookings for
        as_of_date: The "current" date - only bookings created before this are visible
        hotel_ids: Optional list of hotel IDs to filter (None = all hotels)
    
    Returns:
        DataFrame with bookings that were visible as of the query date
    """
    # Convert dates to strings for SQL
    target_start = min(target_dates).isoformat()
    target_end = max(target_dates).isoformat()
    as_of_str = as_of_date.isoformat()
    
    # Build hotel filter
    hotel_filter = ""
    if hotel_ids is not None and len(hotel_ids) > 0:
        hotel_list = ",".join(str(h) for h in hotel_ids)
        hotel_filter = f"AND b.hotel_id IN ({hotel_list})"
    
    query = f"""
    SELECT 
        b.id as booking_id,
        b.hotel_id,
        b.arrival_date,
        b.departure_date,
        b.created_at,
        b.status,
        br.room_type,
        br.room_view,
        br.room_size,
        br.total_price as room_price,
        hl.city,
        hl.latitude,
        hl.longitude,
        r.number_of_rooms,
        r.children_allowed,
        r.pets_allowed,
        r.events_allowed
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    LEFT JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
    WHERE b.status IN ('Booked', 'confirmed')
      AND b.created_at <= '{as_of_str}'::TIMESTAMP
      AND b.arrival_date >= '{target_start}'::DATE
      AND b.arrival_date <= '{target_end}'::DATE
      {hotel_filter}
    ORDER BY b.hotel_id, b.arrival_date
    """
    
    return con.execute(query).fetchdf()


def load_hotel_capacity(
    con: duckdb.DuckDBPyConnection,
    hotel_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load hotel capacity (total rooms available).
    
    Args:
        con: DuckDB connection
        hotel_ids: Optional list of hotel IDs to filter
    
    Returns:
        DataFrame with hotel_id and total_rooms
    """
    hotel_filter = ""
    if hotel_ids is not None and len(hotel_ids) > 0:
        hotel_list = ",".join(str(h) for h in hotel_ids)
        hotel_filter = f"WHERE b.hotel_id IN ({hotel_list})"
    
    query = f"""
    SELECT 
        b.hotel_id,
        SUM(DISTINCT r.number_of_rooms) as total_rooms
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
    WHERE b.status IN ('Booked', 'confirmed')
    {hotel_filter.replace('WHERE', 'AND') if hotel_filter else ''}
    GROUP BY b.hotel_id
    """
    
    df = con.execute(query).fetchdf()
    
    # Ensure we have capacity for all hotels
    df['total_rooms'] = df['total_rooms'].fillna(1).clip(lower=1)
    
    return df


def load_hotel_locations(
    con: duckdb.DuckDBPyConnection,
    hotel_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load hotel locations with all VALIDATED features for similarity scoring.
    
    Features included (from feature_importance_validation.py):
    - room_type: Primary room type
    - avg_room_size: Average room size in sqm (for log_room_size)
    - total_rooms: Hotel capacity (for total_capacity_log)
    - room_view: For view_quality_ordinal (0-3)
    - amenities_score: Sum of children/pets/events/smoking allowed (0-4)
    - children_allowed, pets_allowed, events_allowed, smoking_allowed: Individual booleans
    
    Args:
        con: DuckDB connection
        hotel_ids: Optional list of hotel IDs to filter
    
    Returns:
        DataFrame with hotel locations and VALIDATED similarity features
    """
    hotel_filter = ""
    if hotel_ids is not None and len(hotel_ids) > 0:
        hotel_list = ",".join(str(h) for h in hotel_ids)
        hotel_filter = f"WHERE hl.hotel_id IN ({hotel_list})"
    
    query = f"""
    WITH hotel_room_stats AS (
        SELECT 
            b.hotel_id,
            MODE() WITHIN GROUP (ORDER BY br.room_type) as primary_room_type,
            MODE() WITHIN GROUP (ORDER BY COALESCE(NULLIF(br.room_view, ''), 'no_view')) as primary_room_view,
            AVG(br.room_size) as avg_room_size,
            SUM(DISTINCT r.number_of_rooms) as total_rooms,
            MAX(CASE WHEN r.children_allowed THEN 1 ELSE 0 END) as children_allowed,
            MAX(CASE WHEN r.pets_allowed THEN 1 ELSE 0 END) as pets_allowed,
            MAX(CASE WHEN r.events_allowed THEN 1 ELSE 0 END) as events_allowed,
            MAX(CASE WHEN r.smoking_allowed THEN 1 ELSE 0 END) as smoking_allowed
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        LEFT JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
        WHERE b.status IN ('Booked', 'confirmed')
        GROUP BY b.hotel_id
    )
    SELECT 
        hl.hotel_id,
        hl.city,
        hl.latitude,
        hl.longitude,
        COALESCE(hrs.primary_room_type, 'unknown') as room_type,
        COALESCE(hrs.primary_room_view, 'no_view') as room_view,
        COALESCE(hrs.avg_room_size, 30.0) as avg_room_size,
        COALESCE(hrs.total_rooms, 1) as total_rooms,
        COALESCE(hrs.children_allowed, 0) as children_allowed,
        COALESCE(hrs.pets_allowed, 0) as pets_allowed,
        COALESCE(hrs.events_allowed, 0) as events_allowed,
        COALESCE(hrs.smoking_allowed, 0) as smoking_allowed,
        -- Amenities score (0-4): sum of boolean amenities
        (COALESCE(hrs.children_allowed, 0) + 
         COALESCE(hrs.pets_allowed, 0) + 
         COALESCE(hrs.events_allowed, 0) + 
         COALESCE(hrs.smoking_allowed, 0)) as amenities_score,
        -- View quality ordinal (0-3) matching feature_importance_validation.py
        CASE 
            WHEN LOWER(hrs.primary_room_view) IN ('ocean_view', 'sea_view') THEN 3
            WHEN LOWER(hrs.primary_room_view) IN ('lake_view', 'mountain_view') THEN 2
            WHEN LOWER(hrs.primary_room_view) IN ('pool_view', 'garden_view') THEN 1
            ELSE 0
        END as view_quality
    FROM hotel_location hl
    LEFT JOIN hotel_room_stats hrs ON hl.hotel_id = hrs.hotel_id
    WHERE hl.latitude IS NOT NULL AND hl.longitude IS NOT NULL
    {hotel_filter.replace('WHERE', 'AND') if hotel_filter else ''}
    """
    
    df = con.execute(query).fetchdf()
    
    # Add is_coastal flag if distance features are available
    # Note: This will be enriched later if distance_from_coast is loaded
    df['is_coastal'] = False  # Default, will be updated if distance features available
    
    return df


def calculate_daily_revpar(
    bookings_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    target_dates: List[date]
) -> pd.DataFrame:
    """
    Calculate daily RevPAR for each hotel on target dates.
    
    RevPAR = (Total Room Revenue) / (Available Room Nights)
           = ADR × Occupancy Rate
    
    Args:
        bookings_df: Bookings data from load_bookings_as_of()
        capacity_df: Hotel capacity from load_hotel_capacity()
        target_dates: List of dates to calculate RevPAR for
    
    Returns:
        DataFrame with columns: hotel_id, date, revenue, rooms_sold, 
                                total_rooms, occupancy_rate, avg_price, revpar
    """
    if len(bookings_df) == 0:
        return pd.DataFrame(columns=[
            'hotel_id', 'date', 'revenue', 'rooms_sold', 
            'total_rooms', 'occupancy_rate', 'avg_price', 'revpar'
        ])
    
    # Explode bookings to daily granularity
    daily_records = []
    
    for _, row in bookings_df.iterrows():
        arrival = pd.to_datetime(row['arrival_date']).date()
        departure = pd.to_datetime(row['departure_date']).date()
        nights = (departure - arrival).days
        
        if nights <= 0:
            continue
        
        nightly_rate = row['room_price'] / nights if nights > 0 else row['room_price']
        
        # Generate a record for each night of stay
        current_date = arrival
        while current_date < departure:
            if current_date in target_dates:
                daily_records.append({
                    'hotel_id': row['hotel_id'],
                    'date': current_date,
                    'room_type': row['room_type'],
                    'room_size': row['room_size'],
                    'nightly_rate': nightly_rate,
                    'city': row['city'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude']
                })
            current_date += timedelta(days=1)
    
    if len(daily_records) == 0:
        return pd.DataFrame(columns=[
            'hotel_id', 'date', 'revenue', 'rooms_sold', 
            'total_rooms', 'occupancy_rate', 'avg_price', 'revpar'
        ])
    
    daily_df = pd.DataFrame(daily_records)
    
    # Aggregate by hotel and date
    agg_df = daily_df.groupby(['hotel_id', 'date']).agg({
        'nightly_rate': ['sum', 'mean', 'count'],
        'room_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown',
        'room_size': 'mean',
        'city': 'first',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        'hotel_id', 'date', 'revenue', 'avg_price', 'rooms_sold',
        'room_type', 'avg_room_size', 'city', 'latitude', 'longitude'
    ]
    
    # Merge with capacity
    result_df = agg_df.merge(capacity_df, on='hotel_id', how='left')
    result_df['total_rooms'] = result_df['total_rooms'].fillna(1).clip(lower=1)
    
    # Calculate occupancy rate and RevPAR
    result_df['occupancy_rate'] = np.clip(
        result_df['rooms_sold'] / result_df['total_rooms'], 0, 1
    )
    result_df['revpar'] = result_df['avg_price'] * result_df['occupancy_rate']
    
    return result_df


def get_peer_revpar_metrics(
    con: duckdb.DuckDBPyConnection,
    target_dates: List[date],
    as_of_date: date,
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    room_type: Optional[str] = None
) -> Tuple[pd.DataFrame, List[PeerMetrics]]:
    """
    Get RevPAR metrics for peers near a location.
    
    This is the primary function for geographic peer comparison.
    It finds hotels within a radius and calculates their RevPAR metrics.
    
    Args:
        con: DuckDB connection
        target_dates: Dates to get metrics for
        as_of_date: Only include bookings created before this date
        lat: Target latitude
        lon: Target longitude
        radius_km: Search radius in kilometers
        room_type: Optional room type filter for similarity
    
    Returns:
        Tuple of (daily RevPAR DataFrame, list of PeerMetrics)
    """
    # First, load all hotel locations
    locations_df = load_hotel_locations(con)
    
    if len(locations_df) == 0:
        return pd.DataFrame(), []
    
    # Calculate distances using haversine approximation
    # For small distances, we can use Euclidean approximation with lat/lon
    # 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 111 km * cos(lat)
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians(lat))
    
    locations_df['distance_km'] = np.sqrt(
        ((locations_df['latitude'] - lat) * lat_km) ** 2 +
        ((locations_df['longitude'] - lon) * lon_km) ** 2
    )
    
    # Filter to hotels within radius
    nearby_df = locations_df[locations_df['distance_km'] <= radius_km].copy()
    
    if len(nearby_df) == 0:
        return pd.DataFrame(), []
    
    # Load bookings for these hotels
    nearby_hotel_ids = nearby_df['hotel_id'].tolist()
    bookings_df = load_bookings_as_of(con, target_dates, as_of_date, nearby_hotel_ids)
    capacity_df = load_hotel_capacity(con, nearby_hotel_ids)
    
    # Calculate daily RevPAR
    daily_revpar_df = calculate_daily_revpar(bookings_df, capacity_df, target_dates)
    
    if len(daily_revpar_df) == 0:
        return pd.DataFrame(), []
    
    # Merge with distance info
    daily_revpar_df = daily_revpar_df.merge(
        nearby_df[['hotel_id', 'distance_km', 'room_type', 'avg_room_size']],
        on='hotel_id',
        how='left',
        suffixes=('', '_location')
    )
    
    # Calculate similarity scores
    # Weights: room_type (0.5), room_size (0.3), distance (0.2)
    if room_type is not None:
        daily_revpar_df['room_type_match'] = (
            daily_revpar_df['room_type'] == room_type
        ).astype(float)
    else:
        daily_revpar_df['room_type_match'] = 1.0
    
    # Normalize room size difference (closer = higher score)
    max_size_diff = 50.0  # Max difference to consider
    if 'avg_room_size_location' in daily_revpar_df.columns:
        size_col = 'avg_room_size_location'
    else:
        size_col = 'avg_room_size'
    
    median_size = daily_revpar_df[size_col].median()
    daily_revpar_df['size_similarity'] = 1 - np.clip(
        np.abs(daily_revpar_df[size_col] - median_size) / max_size_diff, 0, 1
    )
    
    # Distance weight (closer = higher score)
    daily_revpar_df['distance_weight'] = 1 - (daily_revpar_df['distance_km'] / radius_km)
    
    # Combined similarity score
    # Weights validated: correlation with price similarity = 0.214 (p<0.001)
    # Alternative weights tested - current is within 2% of optimal (equal weights)
    daily_revpar_df['similarity_score'] = (
        0.5 * daily_revpar_df['room_type_match'] +
        0.3 * daily_revpar_df['size_similarity'] +
        0.2 * daily_revpar_df['distance_weight']
    )
    
    # Build PeerMetrics list (aggregated per hotel across dates)
    hotel_agg = daily_revpar_df.groupby('hotel_id').agg({
        'avg_price': 'mean',
        'occupancy_rate': 'mean',
        'revpar': 'mean',
        'room_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown',
        'avg_room_size': 'mean',
        'distance_km': 'first',
        'similarity_score': 'mean',
        'rooms_sold': 'sum'
    }).reset_index()
    
    peer_metrics = [
        PeerMetrics(
            hotel_id=int(row['hotel_id']),
            avg_price=row['avg_price'],
            occupancy_rate=row['occupancy_rate'],
            revpar=row['revpar'],
            room_type=row['room_type'],
            room_size=row['avg_room_size'],
            distance_km=row['distance_km'],
            similarity_score=row['similarity_score'],
            n_bookings=int(row['rooms_sold'])
        )
        for _, row in hotel_agg.iterrows()
    ]
    
    return daily_revpar_df, peer_metrics


def get_weighted_peer_average(
    peer_metrics: List[PeerMetrics],
    weight_by_similarity: bool = True
) -> Optional[PeerMetrics]:
    """
    Calculate weighted average of peer metrics.
    
    Args:
        peer_metrics: List of PeerMetrics from nearby hotels
        weight_by_similarity: If True, weight by similarity score
    
    Returns:
        Aggregated PeerMetrics representing the peer average, or None if no peers
    """
    if len(peer_metrics) == 0:
        return None
    
    if weight_by_similarity:
        weights = np.array([p.similarity_score or 1.0 for p in peer_metrics])
    else:
        weights = np.ones(len(peer_metrics))
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Calculate weighted averages
    avg_price = sum(p.avg_price * w for p, w in zip(peer_metrics, weights))
    avg_occ = sum(p.occupancy_rate * w for p, w in zip(peer_metrics, weights))
    avg_revpar = sum(p.revpar * w for p, w in zip(peer_metrics, weights))
    avg_size = sum(p.room_size * w for p, w in zip(peer_metrics, weights))
    
    # Most common room type
    room_types = [p.room_type for p in peer_metrics]
    most_common_type = max(set(room_types), key=room_types.count)
    
    total_bookings = sum(p.n_bookings for p in peer_metrics)
    
    return PeerMetrics(
        hotel_id=None,  # Aggregate, not a single hotel
        avg_price=avg_price,
        occupancy_rate=avg_occ,
        revpar=avg_revpar,
        room_type=most_common_type,
        room_size=avg_size,
        distance_km=None,
        similarity_score=1.0,  # Aggregate score
        n_bookings=total_bookings
    )


# Convenience function for testing
def create_test_connection() -> duckdb.DuckDBPyConnection:
    """Create a clean database connection for testing."""
    return get_clean_connection()

