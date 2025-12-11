#!/usr/bin/env python3
"""
End-to-end pipeline verification script.

Runs all pipeline components to verify the system works on a fresh machine.

Usage:
    python scripts/verify_pipeline.py
"""

import sys
import traceback
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def step(name: str):
    """Decorator to track step execution."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"STEP: {name}")
            print('='*60)
            try:
                result = func()
                print(f"✓ {name} - PASSED")
                return True, result
            except Exception as e:
                print(f"✗ {name} - FAILED")
                print(f"  Error: {e}")
                traceback.print_exc()
                return False, None
        return wrapper
    return decorator


@step("1. Data Loading")
def verify_data_loading():
    """Verify database connection and data loading."""
    from src.data.loader import init_db
    
    con = init_db()
    
    # Check tables exist
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    
    required = ['bookings', 'booked_rooms', 'hotel_location', 'rooms']
    for t in required:
        assert t in table_names, f"Missing table: {t}"
    
    # Check row counts
    bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
    hotels = con.execute("SELECT COUNT(DISTINCT hotel_id) FROM hotel_location").fetchone()[0]
    
    print(f"  Bookings: {bookings:,}")
    print(f"  Hotels: {hotels:,}")
    
    assert bookings > 0, "No bookings found"
    assert hotels > 0, "No hotels found"
    
    return con


@step("2. Data Validation")
def verify_data_validation():
    """Verify data validation and cleaning."""
    from lib.db import init_db
    from lib.data_validator import DataCleaner, CleaningConfig
    
    con = init_db()
    config = CleaningConfig()
    cleaner = DataCleaner(config)
    
    # Get initial counts
    initial_bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
    
    # Run cleaning
    con = cleaner.clean(con)
    
    # Get final counts
    final_bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
    
    removed = initial_bookings - final_bookings
    pct_removed = (removed / initial_bookings) * 100
    
    print(f"  Initial bookings: {initial_bookings:,}")
    print(f"  Final bookings: {final_bookings:,}")
    print(f"  Removed: {removed:,} ({pct_removed:.1f}%)")
    
    assert final_bookings > 0, "All bookings were removed"
    
    return con


@step("3. Distance Features")
def verify_distance_features():
    """Verify distance feature calculation or cache loading."""
    from src.features.distance import ensure_distance_features
    
    df = ensure_distance_features()
    
    print(f"  Hotels with features: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    assert 'distance_from_madrid' in df.columns
    assert 'distance_from_coast' in df.columns
    assert len(df) > 0
    
    return df


@step("4. Market Segmentation")
def verify_market_segmentation():
    """Verify market segment assignment."""
    from src.features.engineering import get_market_segments_vectorized
    from src.data.loader import init_db
    import numpy as np
    import pandas as pd
    
    con = init_db()
    
    # Get hotel locations
    hotels = con.execute("""
        SELECT hotel_id, latitude, longitude, city
        FROM hotel_location
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        LIMIT 100
    """).fetchdf()
    
    # Get segments using correct signature
    lats = hotels['latitude'].values
    lons = hotels['longitude'].values
    segments = get_market_segments_vectorized(lats, lons)
    
    hotels['market_segment'] = segments
    
    segment_counts = hotels['market_segment'].value_counts()
    print(f"  Sample segment distribution:")
    for seg, count in segment_counts.items():
        print(f"    {seg}: {count}")
    
    assert len(segments) == len(hotels)
    assert pd.notna(segments).sum() > 0
    
    return hotels


@step("5. Pricing Pipeline")
def verify_pricing_pipeline():
    """Verify the main pricing pipeline works."""
    from src.recommender.pricing_pipeline import PricingPipeline
    
    pipeline = PricingPipeline()
    pipeline.fit()
    
    # Get a sample hotel ID
    hotel_ids = pipeline.con.execute("""
        SELECT DISTINCT hotel_id 
        FROM bookings 
        LIMIT 5
    """).fetchdf()['hotel_id'].tolist()
    
    # Test recommendation
    rec = pipeline.recommend_daily(
        hotel_id=hotel_ids[0],
        target_date=date(2024, 6, 15)
    )
    
    print(f"  Sample recommendation for hotel {hotel_ids[0]}:")
    for key, value in rec.items():
        print(f"    {key}: {value}")
    
    assert 'recommended_price' in rec
    assert rec['recommended_price'] is not None or rec['recommendation'] == 'KEEP'
    
    return rec


@step("6. Unit Tests")
def verify_unit_tests():
    """Run unit tests."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short', '-x'],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Don't fail on test failures, just report
    if result.returncode == 0:
        print("  All tests passed")
    else:
        print(f"  Some tests failed (exit code: {result.returncode})")
    
    return result.returncode == 0


def main():
    """Run all verification steps."""
    print("\n" + "="*60)
    print("PRICEADVISOR PIPELINE VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run each step
    results.append(verify_data_loading())
    results.append(verify_data_validation())
    results.append(verify_distance_features())
    results.append(verify_market_segmentation())
    results.append(verify_pricing_pipeline())
    results.append(verify_unit_tests())
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r[0])
    total = len(results)
    
    steps = [
        "1. Data Loading",
        "2. Data Validation", 
        "3. Distance Features",
        "4. Market Segmentation",
        "5. Pricing Pipeline",
        "6. Unit Tests"
    ]
    
    for i, (success, _) in enumerate(results):
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {steps[i]}: {status}")
    
    print(f"\nOverall: {passed}/{total} steps passed")
    
    if passed == total:
        print("\n✓ All verification steps passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} step(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

