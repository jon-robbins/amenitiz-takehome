"""
Data loading utilities for the price recommender system.

Loads data from CSV files into DuckDB for fast SQL queries.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .validator import CleaningConfig, DataCleaner


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def init_db(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Load raw data from CSV files into DuckDB.
    
    Returns a connection with all tables loaded and properly typed.
    """
    con = duckdb.connect(database=db_path, read_only=False)
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    csv_files = {
        "ds_booked_rooms.csv": "booked_rooms",
        "ds_bookings.csv": "bookings",
        "ds_hotel_location.csv": "hotel_location",
        "ds_rooms.csv": "rooms"
    }
    
    for filename, table_name in csv_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            con.execute(f"""
                CREATE TEMP TABLE temp_{table_name} AS 
                SELECT * FROM read_csv_auto('{file_path}', all_varchar=True, nullstr='NULL')
            """)
            
            if table_name == "booked_rooms":
                con.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT
                        id,
                        TRY_CAST(NULLIF(booking_id,'NULL') AS BIGINT) AS booking_id,
                        TRY_CAST(NULLIF(total_adult,'NULL') AS INTEGER) AS total_adult,
                        TRY_CAST(NULLIF(total_children,'NULL') AS INTEGER) AS total_children,
                        TRY_CAST(NULLIF(room_id,'NULL') AS BIGINT) AS room_id,
                        TRY_CAST(NULLIF(room_size,'NULL') AS DOUBLE) AS room_size,
                        room_view,
                        room_type,
                        TRY_CAST(NULLIF(total_price,'NULL') AS DOUBLE) AS total_price
                    FROM temp_{table_name}
                """)
            elif table_name == "bookings":
                con.execute(f"""
                    CREATE TABLE {table_name} AS
                    SELECT
                        TRY_CAST(NULLIF(id,'NULL') AS BIGINT) AS id,
                        status,
                        TRY_CAST(NULLIF(total_price,'NULL') AS DOUBLE) AS total_price,
                        TRY_CAST(NULLIF(created_at,'NULL') AS TIMESTAMP) AS created_at,
                        TRY_CAST(NULLIF(cancelled_date,'NULL') AS DATE) AS cancelled_date,
                        source,
                        TRY_CAST(NULLIF(arrival_date,'NULL') AS DATE) AS arrival_date,
                        TRY_CAST(NULLIF(departure_date,'NULL') AS DATE) AS departure_date,
                        payment_method,
                        cancelled_by,
                        TRY_CAST(NULLIF(hotel_id,'NULL') AS BIGINT) AS hotel_id
                    FROM temp_{table_name}
                """)
            elif table_name == "hotel_location":
                con.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT
                        TRY_CAST(NULLIF(id,'NULL') AS BIGINT) AS id,
                        TRY_CAST(NULLIF(hotel_id,'NULL') AS BIGINT) AS hotel_id,
                        address,
                        city,
                        zip,
                        country,
                        TRY_CAST(NULLIF(latitude,'NULL') AS DOUBLE) AS latitude,
                        TRY_CAST(NULLIF(longitude,'NULL') AS DOUBLE) AS longitude
                    FROM temp_{table_name}
                """)
            elif table_name == "rooms":
                con.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT
                        TRY_CAST(NULLIF(id,'NULL') AS BIGINT) AS id,
                        TRY_CAST(NULLIF(number_of_rooms,'NULL') AS BIGINT) AS number_of_rooms,
                        TRY_CAST(NULLIF(max_occupancy,'NULL') AS BIGINT) AS max_occupancy,
                        TRY_CAST(NULLIF(max_adults,'NULL') AS BIGINT) AS max_adults,
                        TRY_CAST(NULLIF(pricing_per_person_activated,'NULL') AS BOOLEAN) AS pricing_per_person_activated,
                        TRY_CAST(NULLIF(events_allowed,'NULL') AS BOOLEAN) AS events_allowed,
                        TRY_CAST(NULLIF(pets_allowed,'NULL') AS BOOLEAN) AS pets_allowed,
                        TRY_CAST(NULLIF(smoking_allowed,'NULL') AS BOOLEAN) AS smoking_allowed,
                        TRY_CAST(NULLIF(children_allowed,'NULL') AS BOOLEAN) AS children_allowed
                    FROM temp_{table_name}
                """)
            else:
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_{table_name}")
            
            con.execute(f"DROP TABLE temp_{table_name}")
            print(f"Loaded {filename} into table '{table_name}' with type casting")
        else:
            print(f"Warning: {file_path} not found")
    
    return con


def get_clean_connection(
    exclude_reception_halls: bool = True,
    exclude_missing_location: bool = True,
    fix_empty_strings: bool = True,
    verbose: bool = False
) -> duckdb.DuckDBPyConnection:
    """
    Initialize database with standard cleaning rules applied.
    
    Args:
        exclude_reception_halls: Remove reception halls from rooms
        exclude_missing_location: Remove hotels without coordinates
        fix_empty_strings: Convert empty strings to NULL
        verbose: Print cleaning statistics
    
    Returns:
        Cleaned DuckDB connection
    """
    config = CleaningConfig(
        exclude_reception_halls=exclude_reception_halls,
        exclude_missing_location=exclude_missing_location,
        fix_empty_strings=fix_empty_strings,
        verbose=verbose
    )
    cleaner = DataCleaner(config)
    return cleaner.clean(init_db())


def load_hotel_month_data(con: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    """
    Load hotel-month aggregated data for modeling.
    
    Returns DataFrame with:
    - hotel_id, month
    - avg_price, total_bookings
    - occupancy_rate
    - city, latitude, longitude
    - room_type, room_size
    
    Args:
        con: Optional DuckDB connection. If None, creates a clean connection.
    
    Returns:
        DataFrame with hotel-month records
    """
    if con is None:
        con = get_clean_connection()
    
    query = """
    WITH hotel_month_stats AS (
        SELECT 
            b.hotel_id,
            DATE_TRUNC('month', b.arrival_date::DATE) as month,
            COUNT(DISTINCT br.id) as total_bookings,
            AVG(br.total_price) as avg_price,
            AVG(br.room_size) as avg_room_size,
            MODE() WITHIN GROUP (ORDER BY br.room_type) as room_type,
            hl.city,
            hl.latitude,
            hl.longitude
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('Booked', 'confirmed')
          AND b.arrival_date >= '2023-01-01'
        GROUP BY b.hotel_id, DATE_TRUNC('month', b.arrival_date::DATE), 
                 hl.city, hl.latitude, hl.longitude
    ),
    hotel_capacity AS (
        SELECT 
            b.hotel_id,
            SUM(DISTINCT r.number_of_rooms) as total_rooms
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
        WHERE b.status IN ('Booked', 'confirmed')
        GROUP BY b.hotel_id
    )
    SELECT 
        hms.*,
        COALESCE(hc.total_rooms, 10) as total_rooms,
        EXTRACT(MONTH FROM hms.month) as month_number
    FROM hotel_month_stats hms
    LEFT JOIN hotel_capacity hc ON hms.hotel_id = hc.hotel_id
    WHERE hms.avg_price > 0 AND hms.avg_price < 1000
    ORDER BY hms.hotel_id, hms.month
    """
    
    df = con.execute(query).fetchdf()
    
    # Compute occupancy rate
    days_in_month = 30  # Approximation
    df['occupancy_rate'] = np.clip(
        df['total_bookings'] / (df['total_rooms'] * days_in_month),
        0, 1
    )
    
    print(f"Loaded {len(df):,} hotel-month records for {df['hotel_id'].nunique():,} hotels")
    
    return df


def load_distance_features() -> Optional[pd.DataFrame]:
    """
    Load pre-computed distance features (distance from coast, Madrid, etc.).
    
    Returns:
        DataFrame with hotel_id, distance_from_coast, distance_from_madrid
        or None if file doesn't exist
    """
    project_root = get_project_root()
    distance_path = project_root / 'outputs' / 'hotel_distance_features.csv'
    
    if distance_path.exists():
        df = pd.read_csv(distance_path)
        print(f"Loaded distance features for {len(df):,} hotels")
        return df
    else:
        print(f"Warning: Distance features not found at {distance_path}")
        return None

