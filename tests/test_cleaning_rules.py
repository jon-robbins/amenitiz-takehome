"""
Tests for individual cleaning rules in lib/data_validator.py.

Each test verifies that a specific cleaning rule works correctly in isolation.
"""

import pytest
import pandas as pd
from lib.data_validator import CleaningConfig, DataCleaner


class TestBasicValidationRules:
    """Test basic validation rules (negative prices, NULL values, etc.)."""
    
    def test_remove_negative_prices(self, test_db_with_data):
        """Test that negative prices are removed."""
        # Count negative prices before
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0"
        ).fetchone()[0]
        assert before > 0, "Test data should have negative prices"
        
        # Apply cleaning with only this rule enabled
        config = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        # Count negative prices after
        after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0"
        ).fetchone()[0]
        
        assert after == 0, "All negative prices should be removed"
        assert cleaner.stats.get('Negative Price', 0) == before
    
    def test_remove_zero_prices(self, test_db_with_data):
        """Test that zero prices are removed."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0"
        ).fetchone()[0]
        assert before > 0, "Test data should have zero prices"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=True,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0"
        ).fetchone()[0]
        
        assert after == 0, "All zero prices should be removed"
        assert cleaner.stats.get('Zero Price', 0) == before
    
    def test_remove_null_prices(self, test_db_with_data):
        """Test that NULL prices are removed."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price IS NULL"
        ).fetchone()[0]
        assert before > 0, "Test data should have NULL prices"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=True,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price IS NULL"
        ).fetchone()[0]
        
        assert after == 0, "All NULL prices should be removed"
        assert cleaner.stats.get('NULL Price', 0) == before
    
    def test_remove_null_room_id(self, test_db_with_data):
        """Test that NULL room_ids are removed."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_id IS NULL"
        ).fetchone()[0]
        assert before > 0, "Test data should have NULL room_ids"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=True,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_id IS NULL"
        ).fetchone()[0]
        
        assert after == 0, "All NULL room_ids should be removed"
        assert cleaner.stats.get('NULL Room ID', 0) == before
    
    def test_remove_null_dates(self, test_db_with_data):
        """Test that NULL dates are removed."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM bookings WHERE arrival_date IS NULL OR departure_date IS NULL"
        ).fetchone()[0]
        assert before > 0, "Test data should have NULL dates"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=True,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM bookings WHERE arrival_date IS NULL OR departure_date IS NULL"
        ).fetchone()[0]
        
        assert after == 0, "All NULL dates should be removed"
        assert cleaner.stats.get('NULL Dates', 0) == before
    
    def test_remove_negative_stay(self, test_db_with_data):
        """Test that bookings with negative stay (arrival >= departure) are removed."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)"
        ).fetchone()[0]
        assert before > 0, "Test data should have negative stay bookings"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=True,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)"
        ).fetchone()[0]
        
        assert after == 0, "All negative stay bookings should be removed"
    
    def test_remove_cancelled_but_active(self, test_db_with_data):
        """Test that bookings marked as active but with cancelled_date are removed."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM bookings WHERE status IN ('confirmed', 'Booked') AND cancelled_date IS NOT NULL"
        ).fetchone()[0]
        assert before > 0, "Test data should have cancelled but active bookings"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=True,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM bookings WHERE status IN ('confirmed', 'Booked') AND cancelled_date IS NOT NULL"
        ).fetchone()[0]
        
        assert after == 0, "All cancelled but active bookings should be removed"
        assert cleaner.stats.get('Cancelled but Active', 0) == before


class TestExclusionRules:
    """Test exclusion rules (reception halls, missing location)."""
    
    def test_exclude_reception_halls(self, test_db_with_data):
        """Test that reception halls are excluded."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_type = 'reception_hall'"
        ).fetchone()[0]
        assert before > 0, "Test data should have reception halls"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            exclude_reception_halls=True,
            exclude_missing_location=False,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_type = 'reception_hall'"
        ).fetchone()[0]
        
        assert after == 0, "All reception halls should be excluded"
        assert cleaner.stats.get('Exclude Reception Halls', 0) == before


class TestFixRules:
    """Test fix rules (empty string conversion)."""
    
    def test_fix_empty_strings_in_room_view(self, test_db_with_data):
        """Test that empty strings in room_view are converted to NULL."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_view = ''"
        ).fetchone()[0]
        assert before > 0, "Test data should have empty string room_views"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=True,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_view = ''"
        ).fetchone()[0]
        
        assert after == 0, "All empty string room_views should be converted to NULL"
        assert cleaner.stats.get('Fix Empty room_view', 0) == before
    
    def test_fix_empty_strings_in_city(self, test_db_with_data):
        """Test that empty strings in city are converted to NULL."""
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM hotel_location WHERE city = ''"
        ).fetchone()[0]
        assert before > 0, "Test data should have empty string cities"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=True,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        after = clean_con.execute(
            "SELECT COUNT(*) FROM hotel_location WHERE city = ''"
        ).fetchone()[0]
        
        assert after == 0, "All empty string cities should be converted to NULL"
        assert cleaner.stats.get('Fix Empty city', 0) == before


class TestImputationRules:
    """Test imputation rules (children_allowed, events_allowed)."""
    
    def test_impute_children_allowed(self, test_db_with_data):
        """Test that children_allowed is imputed from bookings with children."""
        # Count rooms with children_allowed=FALSE before
        before_false = test_db_with_data.execute(
            "SELECT COUNT(*) FROM rooms WHERE children_allowed = FALSE"
        ).fetchone()[0]
        
        # Count rooms that should be imputed (have bookings with children)
        should_impute = test_db_with_data.execute("""
            SELECT COUNT(DISTINCT r.id)
            FROM rooms r
            JOIN booked_rooms br ON CAST(br.room_id AS BIGINT) = r.id
            WHERE br.total_children > 0 AND r.children_allowed = FALSE
        """).fetchone()[0]
        assert should_impute > 0, "Test data should have rooms to impute"
        
        config = CleaningConfig(
            remove_negative_prices=False,
            remove_zero_prices=False,
            remove_null_prices=False,
            remove_extreme_prices=False,
            remove_null_dates=False,
            remove_null_created_at=False,
            remove_negative_stay=False,
            remove_negative_lead_time=False,
            remove_null_occupancy=False,
            remove_overcrowded_rooms=False,
            remove_null_room_id=False,
            remove_null_booking_id=False,
            remove_null_hotel_id=False,
            remove_orphan_bookings=False,
            remove_null_status=False,
            remove_cancelled_but_active=False,
            fix_empty_strings=False,
            impute_children_allowed=True,
            impute_events_allowed=False,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        # Count rooms with children_allowed=FALSE after
        after_false = clean_con.execute(
            "SELECT COUNT(*) FROM rooms WHERE children_allowed = FALSE"
        ).fetchone()[0]
        
        # Should have fewer FALSE values
        assert after_false < before_false, "Some rooms should be imputed to TRUE"
        assert cleaner.stats.get('Impute children_allowed', 0) == should_impute


class TestRuleOrdering:
    """Test that rules execute in the correct order."""
    
    def test_missing_location_phase_ordering(self, test_db_with_data):
        """Test that Phase 1 runs before empty string fixes, Phase 2 after."""
        config = CleaningConfig(
            exclude_missing_location=True,
            fix_empty_strings=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        
        # Check that rules are in correct order
        rule_names = [rule.name for rule in cleaner.rules]
        
        phase1_idx = rule_names.index('Exclude Missing Location (Phase 1)')
        fix_empty_idx = rule_names.index('Fix Empty city')
        phase2_idx = rule_names.index('Exclude Missing Location (Phase 2)')
        
        assert phase1_idx < fix_empty_idx, "Phase 1 should run before empty string fixes"
        assert fix_empty_idx < phase2_idx, "Empty string fixes should run before Phase 2"
    
    def test_orphan_cleanup_after_exclusions(self, test_db_with_data):
        """Test that orphan cleanup runs after exclusions."""
        config = CleaningConfig(
            exclude_reception_halls=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        
        # Check that rules are in correct order
        rule_names = [rule.name for rule in cleaner.rules]
        
        exclude_idx = rule_names.index('Exclude Reception Halls')
        orphan_idx = rule_names.index('Orphan Bookings (after exclusions)')
        
        assert exclude_idx < orphan_idx, "Orphan cleanup should run after exclusions"


class TestCombinedRules:
    """Test that multiple rules work together correctly."""
    
    def test_multiple_price_rules(self, test_db_with_data):
        """Test that multiple price rules work together."""
        # Count all problematic prices before
        before = test_db_with_data.execute("""
            SELECT COUNT(*) FROM booked_rooms 
            WHERE total_price < 0 OR total_price = 0 OR total_price IS NULL
        """).fetchone()[0]
        assert before > 0, "Test data should have problematic prices"
        
        config = CleaningConfig(
            remove_negative_prices=True,
            remove_zero_prices=True,
            remove_null_prices=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        # Count all problematic prices after
        after = clean_con.execute("""
            SELECT COUNT(*) FROM booked_rooms 
            WHERE total_price < 0 OR total_price = 0 OR total_price IS NULL
        """).fetchone()[0]
        
        assert after == 0, "All problematic prices should be removed"
    
    def test_full_cleaning_pipeline(self, test_db_with_data):
        """Test that full cleaning pipeline with all rules works."""
        # Get initial counts
        before_bookings = test_db_with_data.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        before_booked_rooms = test_db_with_data.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Apply full cleaning
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(test_db_with_data)
        
        # Get final counts
        after_bookings = clean_con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        after_booked_rooms = clean_con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        # Should have fewer rows (some were cleaned)
        assert after_bookings <= before_bookings
        assert after_booked_rooms < before_booked_rooms  # Definitely should remove some
        
        # Should have stats for multiple rules
        assert len(cleaner.stats) > 0

