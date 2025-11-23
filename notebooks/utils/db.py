import duckdb
from pathlib import Path


def get_connection(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Load raw data from CSV files into DuckDB.
    Returns a connection with all tables loaded.
    """
    return init_db(db_path)


def init_db(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=db_path, read_only=False)
    project_root = Path(__file__).parent.parent.parent
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
