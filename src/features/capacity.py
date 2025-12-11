"""
Hotel room capacity estimation.

Estimates actual room capacity from booking patterns, since the rooms.number_of_rooms
field is often incomplete or defaults to 1.

Method: Calculate maximum simultaneous bookings per hotel across all dates.
This gives a lower bound on capacity (can't book more rooms than you have).
"""

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


CACHE_PATH = Path("outputs/data/hotel_room_capacity.csv")


def calculate_hotel_capacity(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Calculate room capacity for each hotel based on max simultaneous bookings.
    
    Uses the overlap of arrival_date to departure_date ranges to find the
    maximum number of rooms booked on any single day.
    
    Args:
        con: DuckDB connection with bookings table loaded
        
    Returns:
        DataFrame with columns: hotel_id, max_simultaneous_rooms
    """
    # Get all confirmed bookings
    query = """
    SELECT 
        hotel_id,
        id as booking_id,
        arrival_date,
        departure_date
    FROM bookings
    WHERE status IN ('confirmed', 'Booked')
      AND arrival_date >= '2023-01-01'
      AND departure_date > arrival_date
      AND DATE_DIFF('day', arrival_date, departure_date) <= 30
    """
    
    bookings = con.execute(query).fetchdf()
    
    def get_max_simultaneous(group: pd.DataFrame) -> int:
        """Find max bookings active on any single day using sweep line."""
        if len(group) == 0:
            return 0
        
        events = []
        for _, row in group.iterrows():
            events.append((row['arrival_date'], 1))
            events.append((row['departure_date'], -1))
        
        events.sort(key=lambda x: (x[0], -x[1]))
        
        current = 0
        max_occupancy = 0
        for _, delta in events:
            current += delta
            max_occupancy = max(max_occupancy, current)
        
        return max_occupancy
    
    # Calculate per hotel
    capacity = bookings.groupby('hotel_id', group_keys=False).apply(
        lambda g: pd.Series({'max_simultaneous_rooms': get_max_simultaneous(g)})
    ).reset_index()
    
    return capacity


def get_hotel_capacity(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Get hotel room capacity, using cache if available.
    
    Args:
        con: DuckDB connection (required if cache doesn't exist)
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame with hotel_id and max_simultaneous_rooms
    """
    if use_cache and CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)
    
    if con is None:
        from src.data.loader import init_db
        con = init_db()
    
    capacity = calculate_hotel_capacity(con)
    
    # Cache for future use
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    capacity.to_csv(CACHE_PATH, index=False)
    
    return capacity


def get_capacity_stats() -> dict:
    """Get summary statistics for hotel capacity."""
    capacity = get_hotel_capacity()
    
    return {
        'total_hotels': len(capacity),
        'total_rooms': int(capacity['max_simultaneous_rooms'].sum()),
        'avg_rooms_per_hotel': round(capacity['max_simultaneous_rooms'].mean(), 1),
        'median_rooms': int(capacity['max_simultaneous_rooms'].median()),
        'max_rooms': int(capacity['max_simultaneous_rooms'].max()),
    }


if __name__ == "__main__":
    # Regenerate capacity cache
    from src.data.loader import init_db
    
    con = init_db()
    capacity = calculate_hotel_capacity(con)
    
    print("Hotel Capacity Statistics:")
    print(f"  Hotels: {len(capacity)}")
    print(f"  Total rooms: {capacity['max_simultaneous_rooms'].sum():,}")
    print(f"  Avg rooms/hotel: {capacity['max_simultaneous_rooms'].mean():.1f}")
    print(f"  Median: {capacity['max_simultaneous_rooms'].median():.0f}")
    
    # Save
    capacity.to_csv(CACHE_PATH, index=False)
    print(f"\nSaved to {CACHE_PATH}")

