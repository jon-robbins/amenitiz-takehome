"""
Tests for lib/data_validator.py - Core components of data validation and cleaning.
"""

import pytest
import duckdb
import pandas as pd
import numpy as np
from lib.db import init_db
from lib.data_validator import (
    Rule,
    CleaningConfig,
    DataCleaner,
    DuckDBConnectionWrapper,
    _DuckDBResultWrapper,
    validate_and_clean,
    check_data_quality
)


class TestRule:
    """Test Rule dataclass."""
    
    def test_rule_creation(self):
        """Test that Rule can be created successfully."""
        rule = Rule(
            name="Test Rule",
            check_query="SELECT COUNT(*) FROM test",
            action_query="DELETE FROM test WHERE id = 1"
        )
        assert rule.name == "Test Rule"
        assert rule.enabled is True
    
    def test_rule_with_disabled_flag(self):
        """Test that Rule can be created with enabled=False."""
        rule = Rule(
            name="Disabled Rule",
            check_query="SELECT 1",
            action_query="SELECT 1",
            enabled=False
        )
        assert rule.enabled is False


class TestCleaningConfig:
    """Test CleaningConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test that default config creates successfully."""
        config = CleaningConfig()
        assert isinstance(config, CleaningConfig)
    
    def test_default_config_values(self):
        """Test default config values."""
        config = CleaningConfig()
        
        # Basic validation should default to True
        assert config.remove_negative_prices is True
        assert config.remove_zero_prices is True
        assert config.remove_null_prices is True
        
        # Exclusions should default to False
        assert config.exclude_reception_halls is False
        assert config.exclude_missing_location is False
        
        # Imputations should default to True
        assert config.impute_children_allowed is True
        assert config.impute_events_allowed is True
        
        # Verbose should default to False
        assert config.verbose is False
    
    def test_config_with_custom_values(self):
        """Test config with custom values."""
        config = CleaningConfig(
            remove_negative_prices=False,
            exclude_reception_halls=True,
            verbose=True
        )
        assert config.remove_negative_prices is False
        assert config.exclude_reception_halls is True
        assert config.verbose is True
    
    def test_all_boolean_flags_work(self):
        """Test that all boolean flags can be set."""
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
            exclude_missing_location=True,
            fix_empty_strings=False,
            impute_children_allowed=False,
            impute_events_allowed=False,
            verbose=True
        )
        # Just verify it creates without error
        assert isinstance(config, CleaningConfig)


class TestDataCleaner:
    """Test DataCleaner class."""
    
    def test_data_cleaner_initialization(self):
        """Test that DataCleaner initializes with config."""
        config = CleaningConfig()
        cleaner = DataCleaner(config)
        
        assert cleaner.config == config
        assert isinstance(cleaner.rules, list)
        assert isinstance(cleaner.stats, dict)
    
    def test_builds_correct_number_of_rules(self):
        """Test that DataCleaner builds correct number of rules based on config."""
        # Default config
        config = CleaningConfig()
        cleaner = DataCleaner(config)
        
        # Should have basic validation + fixes + imputations
        assert len(cleaner.rules) > 0
    
    def test_builds_more_rules_with_exclusions(self):
        """Test that enabling exclusions adds more rules."""
        config_basic = CleaningConfig(
            exclude_reception_halls=False,
            exclude_missing_location=False
        )
        cleaner_basic = DataCleaner(config_basic)
        basic_count = len(cleaner_basic.rules)
        
        config_with_exclusions = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True
        )
        cleaner_with_exclusions = DataCleaner(config_with_exclusions)
        exclusions_count = len(cleaner_with_exclusions.rules)
        
        assert exclusions_count > basic_count
    
    def test_clean_returns_wrapped_connection(self, test_db_with_data):
        """Test that clean() returns DuckDBConnectionWrapper."""
        config = CleaningConfig(verbose=False)
        cleaner = DataCleaner(config)
        
        result = cleaner.clean(test_db_with_data)
        
        assert isinstance(result, DuckDBConnectionWrapper)
    
    def test_stats_tracking_works(self, test_db_with_data):
        """Test that stats are tracked correctly."""
        config = CleaningConfig(verbose=False)
        cleaner = DataCleaner(config)
        
        cleaner.clean(test_db_with_data)
        
        # Stats should be populated
        assert isinstance(cleaner.stats, dict)
        # Should have at least some rules that affected data
        assert len(cleaner.stats) > 0
    
    def test_verbose_mode(self, test_db_with_data):
        """Test that verbose mode doesn't cause errors."""
        config = CleaningConfig(verbose=True)
        cleaner = DataCleaner(config)
        
        # Should not raise any errors
        result = cleaner.clean(test_db_with_data)
        
        # Should still return wrapped connection
        assert isinstance(result, DuckDBConnectionWrapper)


class TestDuckDBConnectionWrapper:
    """Test DuckDBConnectionWrapper class."""
    
    def test_wrapper_creation(self, raw_connection):
        """Test that wrapper wraps connection correctly."""
        wrapped = DuckDBConnectionWrapper(raw_connection)
        assert isinstance(wrapped, DuckDBConnectionWrapper)
        assert wrapped._con == raw_connection
    
    def test_execute_returns_result_wrapper(self, raw_connection):
        """Test that execute() returns _DuckDBResultWrapper."""
        raw_connection.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        raw_connection.execute("INSERT INTO test VALUES (1, 'test')")
        
        wrapped = DuckDBConnectionWrapper(raw_connection)
        result = wrapped.execute("SELECT * FROM test")
        
        assert isinstance(result, _DuckDBResultWrapper)
    
    def test_fetchdf_converts_none_to_nan(self, raw_connection):
        """Test that fetchdf() converts None to NaN."""
        raw_connection.execute("CREATE TABLE test (id INTEGER, city VARCHAR)")
        raw_connection.execute("INSERT INTO test VALUES (1, 'Barcelona'), (2, NULL)")
        
        wrapped = DuckDBConnectionWrapper(raw_connection)
        df = wrapped.execute("SELECT * FROM test").fetchdf()
        
        # Check NaN conversion
        assert df['city'].isna().sum() == 1
        assert pd.isna(df.loc[1, 'city'])
    
    def test_fetchdf_converts_empty_strings_to_nan(self, raw_connection):
        """Test that fetchdf() converts empty strings to NaN."""
        raw_connection.execute("CREATE TABLE test (id INTEGER, city VARCHAR)")
        raw_connection.execute("INSERT INTO test VALUES (1, 'Barcelona'), (2, '')")
        
        wrapped = DuckDBConnectionWrapper(raw_connection)
        df = wrapped.execute("SELECT * FROM test").fetchdf()
        
        # Check empty string conversion
        assert df['city'].isna().sum() == 1
        assert pd.isna(df.loc[1, 'city'])
    
    def test_fetchdf_handles_both_none_and_empty_strings(self, raw_connection):
        """Test that fetchdf() handles both None and empty strings."""
        raw_connection.execute("CREATE TABLE test (id INTEGER, city VARCHAR, country VARCHAR)")
        raw_connection.execute("""
            INSERT INTO test VALUES 
            (1, 'Barcelona', 'Spain'),
            (2, NULL, 'Spain'),
            (3, '', 'Spain'),
            (4, 'Madrid', NULL)
        """)
        
        wrapped = DuckDBConnectionWrapper(raw_connection)
        df = wrapped.execute("SELECT * FROM test").fetchdf()
        
        # Check conversions
        assert df['city'].isna().sum() == 2  # NULL and empty string
        assert df['country'].isna().sum() == 1  # NULL only
    
    def test_other_methods_delegate(self, raw_connection):
        """Test that other methods delegate to underlying connection."""
        raw_connection.execute("CREATE TABLE test (id INTEGER)")
        raw_connection.execute("INSERT INTO test VALUES (1), (2), (3)")
        
        wrapped = DuckDBConnectionWrapper(raw_connection)
        
        # fetchone should work
        result = wrapped.execute("SELECT COUNT(*) FROM test").fetchone()
        assert result[0] == 3


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""
    
    def test_validate_and_clean_works(self):
        """Test that validate_and_clean() still works."""
        con = init_db()
        
        clean_con = validate_and_clean(con, verbose=False)
        
        assert isinstance(clean_con, DuckDBConnectionWrapper)
    
    def test_validate_and_clean_with_parameters(self):
        """Test that old parameters map to new config correctly."""
        con = init_db()
        
        clean_con = validate_and_clean(
            con,
            verbose=False,
            rooms_to_exclude=['reception_hall'],
            exclude_missing_location_bookings=True
        )
        
        assert isinstance(clean_con, DuckDBConnectionWrapper)
        
        # Should be queryable
        result = clean_con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        assert result > 0
    
    def test_validate_and_clean_excludes_reception_halls(self):
        """Test that rooms_to_exclude parameter works."""
        con = init_db()
        
        # Count reception halls before
        before = con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_type = 'reception_hall'"
        ).fetchone()[0]
        
        # Clean with exclusion
        clean_con = validate_and_clean(
            con,
            verbose=False,
            rooms_to_exclude=['reception_hall']
        )
        
        # Count reception halls after
        after = clean_con.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE room_type = 'reception_hall'"
        ).fetchone()[0]
        
        # Should have fewer (or zero) reception halls
        assert after < before
    
    def test_check_data_quality_works(self):
        """Test that check_data_quality() works."""
        con = init_db()
        
        results = check_data_quality(con)
        
        assert isinstance(results, dict)
        assert 'rules' in results
        assert 'total_failed' in results
        assert 'checks_passed' in results
        assert 'total_checks' in results
    
    def test_check_data_quality_returns_expected_structure(self):
        """Test that check_data_quality() returns expected structure."""
        con = init_db()
        
        results = check_data_quality(con)
        
        # Check structure
        assert isinstance(results['rules'], list)
        assert isinstance(results['total_failed'], int)
        assert isinstance(results['checks_passed'], int)
        assert isinstance(results['total_checks'], int)
        
        # Check that rules have expected fields
        if len(results['rules']) > 0:
            rule = results['rules'][0]
            assert 'name' in rule
            assert 'failed' in rule
            assert 'total' in rule
            assert 'pct' in rule


class TestRuleExecution:
    """Test that rules execute correctly."""
    
    def test_check_query_executes(self, test_db_with_data):
        """Test that check query executes without error."""
        rule = Rule(
            name="Test Check",
            check_query="SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0",
            action_query="DELETE FROM booked_rooms WHERE total_price < 0"
        )
        
        # Should not raise error
        result = test_db_with_data.execute(rule.check_query).fetchone()[0]
        assert isinstance(result, int)
    
    def test_action_query_executes(self, test_db_with_data):
        """Test that action query executes without error."""
        rule = Rule(
            name="Test Action",
            check_query="SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0",
            action_query="DELETE FROM booked_rooms WHERE total_price < 0"
        )
        
        # Should not raise error
        test_db_with_data.execute(rule.action_query)
    
    def test_enabled_flag_works(self, test_db_with_data):
        """Test that enabled/disabled flag works."""
        # Count negative prices before
        before = test_db_with_data.execute(
            "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0"
        ).fetchone()[0]
        
        # Create config with rule disabled
        config = CleaningConfig(
            remove_negative_prices=False,  # Disabled
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
        
        # Should be the same (rule was disabled)
        assert after == before

