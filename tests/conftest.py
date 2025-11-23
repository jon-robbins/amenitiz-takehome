"""
Shared pytest fixtures for testing lib/db.py and lib/data_validator.py.
"""

import pytest
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def raw_connection():
    """Raw database connection with minimal test data."""
    con = duckdb.connect(":memory:")
    return con


@pytest.fixture
def sample_booked_rooms():
    """Sample booked_rooms data for testing."""
    return pd.DataFrame({
        'id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'booking_id': ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109'],
        'total_price': ['100.0', '-50.0', '0.0', 'NULL', '10000.0', '50.0', '75.0', '200.0', '150.0', '300.0'],
        'room_id': ['1', '2', '3', 'NULL', '5', '1', '2', '3', '4', '5'],
        'total_adult': ['2', '2', '1', '2', '4', '2', '2', '3', '2', '2'],
        'total_children': ['0', '1', '0', '0', '2', '0', '0', '1', '0', '0'],
        'room_size': ['20.0', '25.0', '100.0', '15.0', '50.0', '20.0', '25.0', '30.0', '35.0', '40.0'],
        'room_view': ['sea', 'mountain', '', 'city', 'sea', 'sea', '', 'mountain', 'city', 'sea'],
        'room_type': ['room', 'room', 'reception_hall', 'room', 'villa', 'room', 'room', 'apartment', 'room', 'cottage']
    })


@pytest.fixture
def sample_bookings():
    """Sample bookings data for testing."""
    return pd.DataFrame({
        'id': ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111'],
        'status': ['confirmed', 'Booked', 'confirmed', 'confirmed', 'Booked', 'confirmed', 'confirmed', 'Booked', 'confirmed', 'confirmed', 'confirmed', 'Booked'],
        'total_price': ['100.0', '50.0', '0.0', '75.0', '10000.0', '50.0', '75.0', '200.0', '150.0', '300.0', '100.0', '50.0'],
        'created_at': ['2024-01-01', '2024-01-02', '2024-01-03', 'NULL', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12'],
        'cancelled_date': ['NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', '2024-01-15', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'],
        'source': ['web', 'web', 'web', 'web', 'web', 'web', 'web', 'web', 'web', 'web', 'web', 'web'],
        'arrival_date': ['2024-02-01', '2024-02-02', 'NULL', '2024-02-04', '2024-02-05', '2024-02-06', '2024-02-07', '2024-02-08', '2024-02-09', '2024-02-10', '2024-01-01', '2024-02-12'],
        'departure_date': ['2024-02-05', '2024-02-06', '2024-02-07', '2024-02-08', '2024-02-06', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-01-01', '2024-02-16'],
        'payment_method': ['card', 'card', 'card', 'card', 'card', 'card', 'card', 'card', 'card', 'card', 'card', 'card'],
        'cancelled_by': ['NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'],
        'hotel_id': ['1', '2', '3', '4', '5', '1', '2', '3', '4', '5', '6', '7']
    })


@pytest.fixture
def sample_rooms():
    """Sample rooms data for testing."""
    return pd.DataFrame({
        'id': ['1', '2', '3', '4', '5'],
        'number_of_rooms': ['1', '2', '1', '3', '1'],
        'max_occupancy': ['2', '4', '10', '2', '6'],
        'max_adults': ['2', '4', '10', '2', '6'],
        'pricing_per_person_activated': ['false', 'false', 'false', 'false', 'false'],
        'events_allowed': ['false', 'false', 'false', 'false', 'false'],
        'pets_allowed': ['false', 'false', 'false', 'false', 'false'],
        'smoking_allowed': ['false', 'false', 'false', 'false', 'false'],
        'children_allowed': ['false', 'false', 'false', 'false', 'false']
    })


@pytest.fixture
def sample_hotel_location():
    """Sample hotel_location data for testing."""
    return pd.DataFrame({
        'id': ['1', '2', '3', '4', '5', '6', '7', '8'],
        'hotel_id': ['1', '2', '3', '4', '5', '6', '7', '8'],
        'address': ['123 Main St', '456 Oak Ave', '', '789 Pine Rd', '321 Elm St', '654 Maple Dr', '987 Cedar Ln', '147 Birch Way'],
        'city': ['Barcelona', 'Madrid', '', 'Valencia', 'Seville', 'NULL', '', 'Bilbao'],
        'zip': ['08001', '28001', '46001', '41001', '41001', '48001', '50001', '48001'],
        'country': ['Spain', 'Spain', 'Spain', 'Spain', 'Spain', 'Spain', 'Spain', 'Spain'],
        'latitude': ['41.3851', '40.4168', '39.4699', '37.3891', '37.3891', 'NULL', '', '43.2630'],
        'longitude': ['2.1734', '-3.7038', '-0.3763', '-5.9845', '-5.9845', 'NULL', '', '-2.9350']
    })


@pytest.fixture
def test_db_with_data(raw_connection, sample_booked_rooms, sample_bookings, sample_rooms, sample_hotel_location):
    """Database connection with test data loaded."""
    con = raw_connection
    
    # Create booked_rooms table
    con.execute("CREATE TEMP TABLE temp_booked_rooms AS SELECT * FROM sample_booked_rooms")
    con.execute("""
        CREATE TABLE booked_rooms AS 
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
        FROM temp_booked_rooms
    """)
    con.execute("DROP TABLE temp_booked_rooms")
    
    # Create bookings table
    con.execute("CREATE TEMP TABLE temp_bookings AS SELECT * FROM sample_bookings")
    con.execute("""
        CREATE TABLE bookings AS
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
        FROM temp_bookings
    """)
    con.execute("DROP TABLE temp_bookings")
    
    # Create rooms table
    con.execute("CREATE TEMP TABLE temp_rooms AS SELECT * FROM sample_rooms")
    con.execute("""
        CREATE TABLE rooms AS 
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
        FROM temp_rooms
    """)
    con.execute("DROP TABLE temp_rooms")
    
    # Create hotel_location table
    con.execute("CREATE TEMP TABLE temp_hotel_location AS SELECT * FROM sample_hotel_location")
    con.execute("""
        CREATE TABLE hotel_location AS 
        SELECT
            TRY_CAST(NULLIF(id,'NULL') AS BIGINT) AS id,
            TRY_CAST(NULLIF(hotel_id,'NULL') AS BIGINT) AS hotel_id,
            address,
            city,
            zip,
            country,
            TRY_CAST(NULLIF(latitude,'NULL') AS DOUBLE) AS latitude,
            TRY_CAST(NULLIF(longitude,'NULL') AS DOUBLE) AS longitude
        FROM temp_hotel_location
    """)
    con.execute("DROP TABLE temp_hotel_location")
    
    return con


@pytest.fixture
def expected_test_data_stats():
    """Expected statistics for test data."""
    return {
        'total_booked_rooms': 10,
        'total_bookings': 12,
        'total_rooms': 5,
        'total_hotels': 8,
        'negative_prices': 1,
        'zero_prices': 1,
        'null_prices': 1,
        'extreme_prices': 1,  # >5000/night
        'null_room_ids': 1,
        'null_dates': 1,
        'negative_stay': 1,  # arrival == departure
        'negative_lead_time': 1,  # created > arrival
        'overcrowded_rooms': 1,  # 4 adults + 2 children in room with max_occupancy 6
        'reception_halls': 1,
        'missing_location_hotels': 2,  # Hotels 6 and 7
        'empty_string_cities': 2,  # Hotels 3 and 7
        'empty_string_views': 2,
        'bookings_with_children': 2,
        'cancelled_but_active': 1  # Booking 106
    }

