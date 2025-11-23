"""
Example usage of the new CleaningConfig and DataCleaner API.

This shows how to use the refactored data validator with a self-documenting
configuration approach.
"""

import sys
sys.path.insert(0, '../..')

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner, check_data_quality

print("=" * 80)
print("DATA CLEANING - NEW API EXAMPLE")
print("=" * 80)

# ============================================================================
# STEP 1: Define Cleaning Configuration
# ============================================================================

print("\n--- STEP 1: Define Configuration ---")
print("Creating CleaningConfig with EDA-discovered issues...")

config = CleaningConfig(
    # Basic validation (defaults to True)
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_extreme_prices=True,
    remove_null_dates=True,
    remove_negative_stay=True,
    remove_negative_lead_time=True,
    remove_null_occupancy=True,
    remove_overcrowded_rooms=True,
    remove_null_room_id=True,
    remove_null_booking_id=True,
    remove_null_hotel_id=True,
    remove_orphan_bookings=True,
    remove_null_status=True,
    remove_cancelled_but_active=True,
    
    # Exclusions (found during EDA)
    exclude_reception_halls=True,      # Section 2.2: Not accommodation
    exclude_missing_location=True,     # Section 3.1: Can't analyze location
    
    # Data quality fixes
    fix_empty_strings=True,            # Convert '' to NULL
    
    # Imputations (found during EDA)
    impute_children_allowed=True,      # Section 2.2: From booking behavior
    impute_events_allowed=True,        # Section 2.2: From room_type
    
    # Logging
    verbose=True
)

print("\nConfiguration created:")
print(f"  exclude_reception_halls: {config.exclude_reception_halls}")
print(f"  exclude_missing_location: {config.exclude_missing_location}")
print(f"  impute_children_allowed: {config.impute_children_allowed}")
print(f"  impute_events_allowed: {config.impute_events_allowed}")

# ============================================================================
# STEP 2: Apply Cleaning
# ============================================================================

print("\n--- STEP 2: Apply Cleaning ---")

cleaner = DataCleaner(config)
print(f"DataCleaner created with {len(cleaner.rules)} rules")

con = cleaner.clean(init_db())

# ============================================================================
# STEP 3: Review Results
# ============================================================================

print("\n" + "=" * 80)
print("CLEANING SUMMARY")
print("=" * 80)

print(f"\nTotal rules applied: {len(cleaner.stats)}")
print("\nRules that affected data:")
for rule_name, rows_affected in cleaner.stats.items():
    print(f"  • {rule_name}: {rows_affected:,} rows")

# ============================================================================
# STEP 4: Verify Clean Data
# ============================================================================

print("\n--- STEP 4: Verify Clean Data ---")

bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
booked_rooms = con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
rooms = con.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]

print(f"\nFinal counts:")
print(f"  Bookings: {bookings:,}")
print(f"  Booked rooms: {booked_rooms:,}")
print(f"  Room configurations: {rooms:,}")

# Check imputation results
children_allowed_true = con.execute(
    "SELECT COUNT(*) FROM rooms WHERE children_allowed = TRUE"
).fetchone()[0]
events_allowed_true = con.execute(
    "SELECT COUNT(*) FROM rooms WHERE events_allowed = TRUE"
).fetchone()[0]

print(f"\nImputation results:")
print(f"  children_allowed=TRUE: {children_allowed_true:,} rooms")
print(f"  events_allowed=TRUE: {events_allowed_true:,} rooms")

print("\n" + "=" * 80)
print("✓ DATA CLEANING COMPLETE")
print("=" * 80)

# ============================================================================
# EXAMPLE: Adding New Rule During EDA
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE: How to Add New Rule During EDA")
print("=" * 80)

print("""
When you find a new data quality issue during EDA:

1. Add field to CleaningConfig in lib/data_validator.py:
   
   @dataclass
   class CleaningConfig:
       # ... existing fields
       remove_duplicate_bookings: bool = False  # NEW: Section X.Y

2. Add rule to DataCleaner._build_rules():
   
   if self.config.remove_duplicate_bookings:
       rules.append(Rule(
           "Remove Duplicate Bookings",
           "SELECT COUNT(*) - COUNT(DISTINCT ...) FROM bookings",
           "DELETE FROM bookings WHERE ..."
       ))

3. Use it in your notebook:
   
   config = CleaningConfig(
       remove_duplicate_bookings=True,  # ← New field
       verbose=True
   )
   cleaner = DataCleaner(config)
   con = cleaner.clean(init_db())

That's it! The config field documents when/why it was added.
""")

