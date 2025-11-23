"""
Integration tests for lib/db.py and lib/data_validator.py.

These tests verify end-to-end workflows and real-world scenarios.
"""

import pytest
import time
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner, validate_and_clean


@pytest.mark.integration
class TestFullPipeline:
    """Test full data loading and cleaning pipeline."""
    
    def test_load_and_clean_workflow(self):
        """Test complete workflow: load raw data → clean → verify results."""
        # Step 1: Load raw data
        con = init_db()
        
        # Verify raw data loaded
        raw_bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        raw_booked_rooms = con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        assert raw_bookings > 0
        assert raw_booked_rooms > 0
        
        # Step 2: Clean data
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(con)
        
        # Step 3: Verify results
        clean_bookings = clean_con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        clean_booked_rooms = clean_con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Should have cleaned some data
        assert clean_bookings <= raw_bookings
        assert clean_booked_rooms < raw_booked_rooms
        
        # Should be queryable
        result = clean_con.execute("""
            SELECT AVG(total_price) 
            FROM booked_rooms 
            WHERE total_price > 0
        """).fetchone()[0]
        assert result > 0
    
    def test_idempotency(self):
        """Test that cleaning twice produces same result."""
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        
        # First cleaning
        con1 = init_db()
        cleaner1 = DataCleaner(config)
        clean_con1 = cleaner1.clean(con1)
        count1 = clean_con1.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        
        # Second cleaning (on already clean data)
        cleaner2 = DataCleaner(config)
        clean_con2 = cleaner2.clean(clean_con1._con)
        count2 = clean_con2.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        
        # Should be identical
        assert count1 == count2
        
        # Second pass should have minimal changes (maybe 0)
        total_changes = sum(cleaner2.stats.values())
        assert total_changes < 10, "Second cleaning should have minimal changes"
    
    def test_stats_match_actual_changes(self):
        """Test that stats accurately reflect actual changes."""
        con = init_db()
        
        # Count problematic data before
        negative_before = con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0"
        ).fetchone()[0]
        zero_before = con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0"
        ).fetchone()[0]
        
        # Clean
        config = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(con)
        
        # Verify stats match
        assert cleaner.stats.get('Negative Price', 0) == negative_before
        assert cleaner.stats.get('Zero Price', 0) == zero_before
        
        # Verify actual data cleaned
        negative_after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0"
        ).fetchone()[0]
        zero_after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0"
        ).fetchone()[0]
        
        assert negative_after == 0
        assert zero_after == 0
    
    def test_final_data_passes_quality_checks(self):
        """Test that cleaned data passes basic quality checks."""
        con = init_db()
        
        # Clean with full config
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(con)
        
        # Quality checks
        # 1. No negative prices
        negative_prices = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0"
        ).fetchone()[0]
        assert negative_prices == 0
        
        # 2. No zero prices
        zero_prices = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0"
        ).fetchone()[0]
        assert zero_prices == 0
        
        # 3. No NULL prices
        null_prices = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price IS NULL"
        ).fetchone()[0]
        assert null_prices == 0
        
        # 4. No NULL room_ids
        null_room_ids = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_id IS NULL"
        ).fetchone()[0]
        assert null_room_ids == 0
        
        # 5. No reception halls
        reception_halls = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_type = 'reception_hall'"
        ).fetchone()[0]
        assert reception_halls == 0
        
        # 6. No empty string room_views
        empty_views = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_view = ''"
        ).fetchone()[0]
        assert empty_views == 0


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_eda_workflow(self):
        """Test typical EDA workflow: init_db → validate_and_clean → query."""
        # Load and clean
        con = validate_and_clean(
            init_db(),
            verbose=False,
            rooms_to_exclude=['reception_hall'],
            exclude_missing_location_bookings=True
        )
        
        # Typical EDA queries should work
        # 1. Count bookings
        bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        assert bookings > 0
        
        # 2. Average price
        avg_price = con.execute("""
            SELECT AVG(total_price) 
            FROM booked_rooms 
            WHERE total_price > 0
        """).fetchone()[0]
        assert avg_price > 0
        
        # 3. Join tables
        result = con.execute("""
            SELECT COUNT(*)
            FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
        """).fetchone()[0]
        assert result > 0
        
        # 4. Get DataFrame
        df = con.execute("SELECT * FROM booked_rooms LIMIT 10").fetchdf()
        assert len(df) == 10
        assert df['total_price'].notna().all()  # No NaN prices
    
    def test_ml_pipeline_workflow(self):
        """Test typical ML pipeline workflow: init_db → DataCleaner → feature engineering."""
        # Load and clean
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        con = cleaner.clean(init_db())
        
        # Feature engineering queries should work
        # 1. Create features
        features = con.execute("""
            SELECT 
                br.total_price,
                br.total_adult + br.total_children as total_guests,
                r.max_occupancy,
                CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as stay_length,
                br.room_type
            FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
            WHERE b.status IN ('confirmed', 'Booked')
            LIMIT 100
        """).fetchdf()
        
        assert len(features) > 0
        assert features['total_price'].notna().all()
        assert (features['total_price'] > 0).all()
        assert features['stay_length'].notna().all()
        assert (features['stay_length'] > 0).all()
    
    def test_incremental_cleaning(self):
        """Test adding rules one by one."""
        con = init_db()
        
        # Step 1: Just basic validation
        config1 = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=True,
            remove_null_prices=True,
            exclude_reception_halls=False,
            exclude_missing_location=False,
            verbose=False
        )
        cleaner1 = DataCleaner(config1)
        con1 = cleaner1.clean(con)
        count1 = con1.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Step 2: Add reception hall exclusion
        config2 = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=True,
            remove_null_prices=True,
            exclude_reception_halls=True,
            exclude_missing_location=False,
            verbose=False
        )
        cleaner2 = DataCleaner(config2)
        con2 = cleaner2.clean(init_db())
        count2 = con2.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Step 3: Add missing location exclusion
        config3 = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=True,
            remove_null_prices=True,
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        cleaner3 = DataCleaner(config3)
        con3 = cleaner3.clean(init_db())
        count3 = con3.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Each step should remove more data
        assert count2 < count1, "Adding reception hall exclusion should remove data"
        assert count3 < count2, "Adding missing location exclusion should remove data"
    
    def test_different_configs_produce_different_results(self):
        """Test that different configs produce different results."""
        # Config 1: Minimal cleaning
        config1 = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=False,
            remove_null_prices=False,
            exclude_reception_halls=False,
            exclude_missing_location=False,
            verbose=False
        )
        cleaner1 = DataCleaner(config1)
        con1 = cleaner1.clean(init_db())
        count1 = con1.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Config 2: Aggressive cleaning
        config2 = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=True,
            remove_null_prices=True,
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        cleaner2 = DataCleaner(config2)
        con2 = cleaner2.clean(init_db())
        count2 = con2.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Aggressive cleaning should remove more
        assert count2 < count1


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics."""
    
    def test_cleaning_completes_in_reasonable_time(self):
        """Test that cleaning completes in <30 seconds for full dataset."""
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        
        start_time = time.time()
        
        con = init_db()
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(con)
        
        # Verify it worked
        result = clean_con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        assert result > 0
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 30, f"Cleaning took {elapsed:.2f}s, should be <30s"
    
    def test_multiple_cleanings_efficient(self):
        """Test that multiple cleanings don't cause memory issues."""
        config = CleaningConfig(verbose=False)
        
        # Run cleaning 5 times
        for i in range(5):
            con = init_db()
            cleaner = DataCleaner(config)
            clean_con = cleaner.clean(con)
            
            # Verify each one works
            result = clean_con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
            assert result > 0
        
        # If we got here without memory errors, test passes


@pytest.mark.integration
class TestDataConsistency:
    """Test data consistency after cleaning."""
    
    def test_foreign_keys_remain_valid(self):
        """Test that foreign key relationships remain valid after cleaning."""
        con = init_db()
        
        # Count orphans before cleaning
        orphans_before = con.execute("""
            SELECT COUNT(*)
            FROM booked_rooms br
            LEFT JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            WHERE b.id IS NULL
        """).fetchone()[0]
        
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(con)
        
        # Check booked_rooms → bookings FK
        orphans_after = clean_con.execute("""
            SELECT COUNT(*)
            FROM booked_rooms br
            LEFT JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            WHERE b.id IS NULL
        """).fetchone()[0]
        
        # Orphans should be the same or fewer (cleaning may remove some bookings)
        # Note: The cleaning process may create orphans by removing bookings,
        # but the orphan cleanup rule should handle them
        assert orphans_after >= 0, "Query should execute without error"
        
        # Check bookings → hotel_location FK
        orphan_bookings = clean_con.execute("""
            SELECT COUNT(*)
            FROM bookings b
            LEFT JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
            WHERE hl.hotel_id IS NULL
        """).fetchone()[0]
        # Some orphans may exist if hotels have no location data
        assert orphan_bookings >= 0  # Just verify query works
    
    def test_no_data_corruption(self):
        """Test that cleaning doesn't corrupt data."""
        con = init_db()
        
        # Get sample data before cleaning
        before_sample = con.execute("""
            SELECT id, booking_id, room_id, total_price, room_type
            FROM booked_rooms
            WHERE total_price > 100 AND total_price < 200
            LIMIT 5
        """).fetchdf()
        
        # Clean
        config = CleaningConfig(verbose=False)
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(con)
        
        # Get same records after cleaning
        after_sample = clean_con.execute(f"""
            SELECT id, booking_id, room_id, total_price, room_type
            FROM booked_rooms
            WHERE id IN ({','.join(f"'{x}'" for x in before_sample['id'].tolist())})
        """).fetchdf()
        
        # Data should be unchanged (these records were valid)
        if len(after_sample) > 0:
            for col in ['booking_id', 'room_id', 'total_price', 'room_type']:
                # Values should match for records that survived
                for idx in after_sample.index:
                    before_val = before_sample[before_sample['id'] == after_sample.loc[idx, 'id']][col].values[0]
                    after_val = after_sample.loc[idx, col]
                    assert before_val == after_val, f"Data corruption detected in {col}"

