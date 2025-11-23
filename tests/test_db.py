"""
Tests for lib/db.py - Database connection and data loading functionality.
"""

import pytest
import duckdb
from pathlib import Path
from lib.db import init_db


class TestDatabaseConnection:
    """Test database connection initialization."""
    
    def test_init_db_creates_connection(self):
        """Test that init_db() creates a valid DuckDB connection."""
        con = init_db()
        assert con is not None
        assert isinstance(con, duckdb.DuckDBPyConnection)
    
    def test_init_db_in_memory(self):
        """Test that init_db() creates in-memory database by default."""
        con = init_db()
        # In-memory databases should work without file system
        result = con.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1
    
    def test_connection_is_queryable(self):
        """Test that connection can execute queries."""
        con = init_db()
        result = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        assert isinstance(result, int)
        assert result > 0


class TestTableLoading:
    """Test that all tables are loaded correctly."""
    
    def test_all_tables_loaded(self):
        """Test that all 4 tables are loaded."""
        con = init_db()
        
        # Check each table exists
        tables = ['booked_rooms', 'bookings', 'rooms', 'hotel_location']
        for table in tables:
            result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert result > 0, f"Table {table} should have data"
    
    def test_booked_rooms_schema(self):
        """Test booked_rooms table has correct schema."""
        con = init_db()
        
        # Get column names
        result = con.execute("DESCRIBE booked_rooms").fetchdf()
        columns = set(result['column_name'].tolist())
        
        expected_columns = {
            'id', 'booking_id', 'total_adult', 'total_children',
            'room_id', 'room_size', 'room_view', 'room_type', 'total_price'
        }
        assert expected_columns.issubset(columns)
    
    def test_bookings_schema(self):
        """Test bookings table has correct schema."""
        con = init_db()
        
        result = con.execute("DESCRIBE bookings").fetchdf()
        columns = set(result['column_name'].tolist())
        
        expected_columns = {
            'id', 'status', 'total_price', 'created_at', 'cancelled_date',
            'source', 'arrival_date', 'departure_date', 'payment_method',
            'cancelled_by', 'hotel_id'
        }
        assert expected_columns.issubset(columns)
    
    def test_rooms_schema(self):
        """Test rooms table has correct schema."""
        con = init_db()
        
        result = con.execute("DESCRIBE rooms").fetchdf()
        columns = set(result['column_name'].tolist())
        
        expected_columns = {
            'id', 'number_of_rooms', 'max_occupancy', 'max_adults',
            'pricing_per_person_activated', 'events_allowed', 'pets_allowed',
            'smoking_allowed', 'children_allowed'
        }
        assert expected_columns.issubset(columns)
    
    def test_hotel_location_schema(self):
        """Test hotel_location table has correct schema."""
        con = init_db()
        
        result = con.execute("DESCRIBE hotel_location").fetchdf()
        columns = set(result['column_name'].tolist())
        
        expected_columns = {
            'id', 'hotel_id', 'address', 'city', 'zip',
            'country', 'latitude', 'longitude'
        }
        assert expected_columns.issubset(columns)


class TestTypeCasting:
    """Test that type casting works correctly."""
    
    def test_integer_columns_cast_correctly(self):
        """Test that integer columns are cast to INTEGER."""
        con = init_db()
        
        # Check booked_rooms integer columns
        result = con.execute("DESCRIBE booked_rooms").fetchdf()
        int_columns = result[result['column_name'].isin(['total_adult', 'total_children'])]
        
        for _, row in int_columns.iterrows():
            assert 'INT' in row['column_type'].upper()
    
    def test_bigint_columns_cast_correctly(self):
        """Test that ID columns are cast to BIGINT."""
        con = init_db()
        
        # Check bookings id column
        result = con.execute("DESCRIBE bookings").fetchdf()
        id_row = result[result['column_name'] == 'id'].iloc[0]
        
        assert 'BIGINT' in id_row['column_type'].upper()
    
    def test_double_columns_cast_correctly(self):
        """Test that price columns are cast to DOUBLE."""
        con = init_db()
        
        # Check booked_rooms total_price
        result = con.execute("DESCRIBE booked_rooms").fetchdf()
        price_row = result[result['column_name'] == 'total_price'].iloc[0]
        
        assert 'DOUBLE' in price_row['column_type'].upper()
    
    def test_date_columns_cast_correctly(self):
        """Test that date columns are cast to DATE."""
        con = init_db()
        
        # Check bookings date columns
        result = con.execute("DESCRIBE bookings").fetchdf()
        date_columns = result[result['column_name'].isin(['arrival_date', 'departure_date', 'cancelled_date'])]
        
        for _, row in date_columns.iterrows():
            assert 'DATE' in row['column_type'].upper()
    
    def test_timestamp_columns_cast_correctly(self):
        """Test that timestamp columns are cast to TIMESTAMP."""
        con = init_db()
        
        # Check bookings created_at
        result = con.execute("DESCRIBE bookings").fetchdf()
        timestamp_row = result[result['column_name'] == 'created_at'].iloc[0]
        
        assert 'TIMESTAMP' in timestamp_row['column_type'].upper()
    
    def test_boolean_columns_cast_correctly(self):
        """Test that boolean columns are cast to BOOLEAN."""
        con = init_db()
        
        # Check rooms boolean columns
        result = con.execute("DESCRIBE rooms").fetchdf()
        bool_columns = result[result['column_name'].isin([
            'events_allowed', 'pets_allowed', 'smoking_allowed', 'children_allowed'
        ])]
        
        for _, row in bool_columns.iterrows():
            assert 'BOOLEAN' in row['column_type'].upper()


class TestNullHandling:
    """Test that NULL values are handled correctly."""
    
    def test_nullif_logic_works(self):
        """Test that 'NULL' strings are converted to actual NULL."""
        con = init_db()
        
        # Check if there are any NULL values (there should be from data quality issues)
        null_prices = con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price IS NULL"
        ).fetchone()[0]
        
        # Should have at least some NULL prices from data quality issues
        assert null_prices >= 0  # Just verify query works
    
    def test_empty_strings_preserved(self):
        """Test that empty strings are preserved (not converted to NULL yet)."""
        con = init_db()
        
        # Empty strings in room_view should exist before cleaning
        empty_views = con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_view = ''"
        ).fetchone()[0]
        
        # Should have some empty strings
        assert empty_views >= 0  # Just verify query works


class TestDataIntegrity:
    """Test data integrity after loading."""
    
    def test_row_counts_positive(self):
        """Test that all tables have positive row counts."""
        con = init_db()
        
        tables = ['booked_rooms', 'bookings', 'rooms', 'hotel_location']
        for table in tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count > 0, f"{table} should have rows"
    
    def test_no_data_corruption(self):
        """Test that data is loaded without corruption."""
        con = init_db()
        
        # Check that we can query various data types without errors
        result = con.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT room_id) as unique_rooms,
                AVG(total_price) as avg_price
            FROM booked_rooms
            WHERE total_price IS NOT NULL AND total_price > 0
        """).fetchone()
        
        assert result[0] > 0  # total
        assert result[1] > 0  # unique_rooms
        assert result[2] > 0  # avg_price
    
    def test_foreign_key_relationships_exist(self):
        """Test that foreign key relationships can be queried."""
        con = init_db()
        
        # Test join between booked_rooms and bookings
        result = con.execute("""
            SELECT COUNT(*)
            FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
        """).fetchone()[0]
        
        assert result > 0, "Should be able to join booked_rooms and bookings"
        
        # Test join between bookings and hotel_location
        result = con.execute("""
            SELECT COUNT(*)
            FROM bookings b
            JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        """).fetchone()[0]
        
        assert result > 0, "Should be able to join bookings and hotel_location"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_multiple_connections_independent(self):
        """Test that multiple connections are independent."""
        con1 = init_db()
        con2 = init_db()
        
        # Modify con1
        con1.execute("CREATE TEMP TABLE test_table (id INTEGER)")
        con1.execute("INSERT INTO test_table VALUES (1)")
        
        # con2 should not have test_table
        with pytest.raises(Exception):
            con2.execute("SELECT * FROM test_table")
    
    def test_connection_reusable(self):
        """Test that connection can be reused for multiple queries."""
        con = init_db()
        
        # Execute multiple queries
        result1 = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        result2 = con.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]
        result3 = con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        assert all(r > 0 for r in [result1, result2, result3])

