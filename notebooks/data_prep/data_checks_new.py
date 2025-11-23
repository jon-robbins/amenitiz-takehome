# %%
"""
Data Quality Assessment
Shows the percentage of dirty data in the raw database.
"""
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0, '../..')

from lib.db import init_db
from lib.data_validator import check_data_quality, CleaningConfig, DataCleaner


# %%
# Load raw data
con = init_db()


# %% [markdown]
# We're gonna check some basic logical tests about the data. 

# %%
# Check data quality
results = check_data_quality(con)

print(f"\n{'='*60}")
print("DATA QUALITY REPORT")
print(f"{'='*60}\n")

print(f"Total Checks: {results['total_checks']}")
print(f"Checks Passed: {results['checks_passed']}")
print(f"Checks Failed: {results['total_checks'] - results['checks_passed']}")
print(f"\nTotal Problematic Rows: {results['total_failed']:,}\n")

print(f"{'Rule':<30} {'Failed':<12} {'Total':<12} {'%':<8}")
print("-" * 60)

for r in results['rules']:
    if r['failed'] > 0:
        print(f"{r['name']:<30} {r['failed']:<12,} {r['total']:<12,} {r['pct']:<8.2f}")


# %% [markdown]
#  We can use about 95% of the data for our analysis.
# 
#  In a real world scenario I would want to dig more in depth into the 5% to see what we can save, but for the purposes of this I'm going to make the assumption that it's all due to data quality issues.

# %%
booked_rooms = con.execute("SELECT * FROM booked_rooms").fetchdf()
rooms = con.execute("SELECT * FROM rooms").fetchdf()
bookings = con.execute("SELECT * FROM bookings").fetchdf()
hotel_location = con.execute("SELECT * FROM hotel_location").fetchdf()

# %%
# check nulls or empty strings per column in each df
print("Booked Rooms Null Pct: ", booked_rooms.isnull().mean() * 100)
print("Booked Rooms Empty String Pct: ", booked_rooms.map(lambda x: 1 if x == '' else 0).mean() * 100)
print("-"*50)
print("Rooms Null Pct: ", rooms.isnull().mean() * 100)
print("Rooms Empty String Pct: ", rooms.map(lambda x: 1 if x == '' else 0).mean() * 100)
print("-"*50)
print("Bookings Null Pct: ", bookings.isnull().mean() * 100)
print("Bookings Empty String Pct: ", bookings.map(lambda x: 1 if x == '' else 0).mean() * 100)
print("-"*50)
print("Hotel Location Null Pct: ", hotel_location.isnull().mean() * 100)
print("Hotel Location Empty String Pct: ", hotel_location.map(lambda x: 1 if x == '' else 0).mean() * 100)
print("-"*50)

# %% [markdown]
#  Takeaways:
# 
#  - Room view is empty when there's no view. We'll replace empty strings with NULL.
# 
#  - When there's no lat/long, it's an empty string. We'll replace them with nulls.
# 
#  - Hotel location has empty strings for some values, we'll replace them with nulls.
# 
#  I'll make the changes in the data validator class and move on to the EDA.

# %%
# Let's also check to see if there are any columns that only have one value, we can drop them. 
print("Booked Rooms Unique Values: ", booked_rooms.nunique())
print("Rooms Unique Values: ", rooms.nunique())
print("Bookings Unique Values: ", bookings.nunique())
print("Hotel Location Unique Values: ", hotel_location.nunique())

# %% [markdown]
# Takeaways:
# - pets_allowed and smoking_allowed only have one value. We can drop these columns entirely.
# - However, we do have booked_rooms.total_children. If there is >=1 booking of a room_id that has >=1 child, then we can impute "TRUE" for the rooms.children_allowed column.
# - We can also do the same for reception halls. Events are definitely allowed for reception halls, so we can fix that feature. 

# %% [markdown]
# ## Step-by-Step Data Cleaning
# 
# Now let's apply the cleaning rules we discovered and see the improvements.

# %%
# Step 1: Start with basic validation only
print("="*80)
print("STEP 1: Basic Validation Only")
print("="*80)

config_basic = CleaningConfig(
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
    
    # Don't apply any other fixes yet
    fix_empty_strings=False,
    impute_children_allowed=False,
    impute_events_allowed=False,
    exclude_reception_halls=False,
    exclude_missing_location=False,
    
    verbose=True
)

cleaner_basic = DataCleaner(config_basic)
con_basic = cleaner_basic.clean(init_db())

print(f"\nRules applied: {len(cleaner_basic.stats)}")
print(f"Total rows cleaned: {sum(cleaner_basic.stats.values()):,}")

# %%
# Step 2: Add empty string fixes
print("\n" + "="*80)
print("STEP 2: Add Empty String Fixes")
print("="*80)
print("Issue: Empty strings should be NULL for proper analysis")

config_with_fixes = CleaningConfig(
    # Basic validation
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
    
    # Add empty string fixes
    fix_empty_strings=True,  # NEW
    
    # Don't apply other fixes yet
    impute_children_allowed=False,
    impute_events_allowed=False,
    exclude_reception_halls=False,
    exclude_missing_location=False,
    
    verbose=True
)

cleaner_with_fixes = DataCleaner(config_with_fixes)
con_with_fixes = cleaner_with_fixes.clean(init_db())

print(f"\nAdditional rules applied: {len(cleaner_with_fixes.stats) - len(cleaner_basic.stats)}")
empty_string_fixes = [k for k in cleaner_with_fixes.stats.keys() if 'Fix Empty' in k]
print(f"Empty strings converted to NULL: {sum(cleaner_with_fixes.stats[k] for k in empty_string_fixes):,}")

# %%
# Step 3: Add policy flag imputations
print("\n" + "="*80)
print("STEP 3: Add Policy Flag Imputations")
print("="*80)
print("Issue: children_allowed and events_allowed are all FALSE")
print("Solution: Impute TRUE based on booking behavior")

# Check before imputation
rooms_before = con_with_fixes.execute("SELECT * FROM rooms").fetchdf()
print(f"\nBefore imputation:")
print(f"  children_allowed=TRUE: {(rooms_before['children_allowed'] == True).sum():,} rooms")
print(f"  events_allowed=TRUE: {(rooms_before['events_allowed'] == True).sum():,} rooms")

config_with_imputation = CleaningConfig(
    # Basic validation
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
    
    # Empty string fixes
    fix_empty_strings=True,
    
    # Add imputations
    impute_children_allowed=True,  # NEW: From bookings with children
    impute_events_allowed=True,    # NEW: From reception_hall room_type
    
    # Don't apply exclusions yet
    exclude_reception_halls=False,
    exclude_missing_location=False,
    
    verbose=True
)

cleaner_with_imputation = DataCleaner(config_with_imputation)
con_with_imputation = cleaner_with_imputation.clean(init_db())

# Check after imputation
rooms_after = con_with_imputation.execute("SELECT * FROM rooms").fetchdf()
print(f"\nAfter imputation:")
print(f"  children_allowed=TRUE: {(rooms_after['children_allowed'] == True).sum():,} rooms")
print(f"  events_allowed=TRUE: {(rooms_after['events_allowed'] == True).sum():,} rooms")

print(f"\nImprovement:")
print(f"  children_allowed: +{cleaner_with_imputation.stats.get('Impute children_allowed', 0):,} rooms")
print(f"  events_allowed: +{cleaner_with_imputation.stats.get('Impute events_allowed', 0):,} rooms")

# %%
# Step 4: Add exclusions (found during EDA)
print("\n" + "="*80)
print("STEP 4: Add Exclusions (Found During EDA)")
print("="*80)
print("Issue 1 (Section 2.2): Reception halls are event spaces, not accommodation")
print("Issue 2 (Section 3.1): Hotels with missing location can't be analyzed")

config_final = CleaningConfig(
    # Basic validation
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
    
    # Empty string fixes
    fix_empty_strings=True,
    
    # Imputations
    impute_children_allowed=True,
    impute_events_allowed=True,
    
    # Add exclusions
    exclude_reception_halls=True,      # NEW: Section 2.2
    exclude_missing_location=True,     # NEW: Section 3.1
    
    verbose=True
)

cleaner_final = DataCleaner(config_final)
con_final = cleaner_final.clean(init_db())

print(f"\nExclusions applied:")
if 'Exclude Reception Halls' in cleaner_final.stats:
    print(f"  Reception halls: {cleaner_final.stats['Exclude Reception Halls']:,} bookings")
missing_loc_rules = [k for k in cleaner_final.stats.keys() if 'Missing Location' in k]
total_missing_loc = sum(cleaner_final.stats[k] for k in missing_loc_rules)
print(f"  Missing location: {total_missing_loc:,} bookings")

# %%
# Final Summary
print("\n" + "="*80)
print("FINAL CLEANING SUMMARY")
print("="*80)

bookings_raw = init_db().execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
bookings_clean = con_final.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
booked_rooms_raw = init_db().execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
booked_rooms_clean = con_final.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]

print(f"\nData Reduction:")
print(f"  Bookings: {bookings_raw:,} → {bookings_clean:,} ({(1 - bookings_clean/bookings_raw)*100:.1f}% removed)")
print(f"  Booked Rooms: {booked_rooms_raw:,} → {booked_rooms_clean:,} ({(1 - booked_rooms_clean/booked_rooms_raw)*100:.1f}% removed)")

print(f"\nTotal Rules Applied: {len(cleaner_final.stats)}")
print(f"Total Rows Affected: {sum(cleaner_final.stats.values()):,}")

print("\nTop 10 Rules by Impact:")
sorted_stats = sorted(cleaner_final.stats.items(), key=lambda x: x[1], reverse=True)[:10]
for rule_name, count in sorted_stats:
    print(f"  {rule_name}: {count:,} rows")

# %%
# Verify improvements
print("\n" + "="*80)
print("DATA QUALITY VERIFICATION")
print("="*80)

# Check that empty strings are gone
df_rooms_clean = con_final.execute("SELECT * FROM rooms").fetchdf()
df_booked_rooms_clean = con_final.execute("SELECT * FROM booked_rooms").fetchdf()
df_bookings_clean = con_final.execute("SELECT * FROM bookings").fetchdf()
df_hotel_location_clean = con_final.execute("SELECT * FROM hotel_location").fetchdf()

print("\nEmpty String Check (should all be 0):")
print(f"  Booked Rooms: {df_booked_rooms_clean.map(lambda x: 1 if x == '' else 0).sum().sum()}")
print(f"  Rooms: {df_rooms_clean.map(lambda x: 1 if x == '' else 0).sum().sum()}")
print(f"  Bookings: {df_bookings_clean.map(lambda x: 1 if x == '' else 0).sum().sum()}")
print(f"  Hotel Location: {df_hotel_location_clean.map(lambda x: 1 if x == '' else 0).sum().sum()}")

print("\nPolicy Flag Distribution:")
print(f"  children_allowed=TRUE: {(df_rooms_clean['children_allowed'] == True).sum():,} rooms")
print(f"  events_allowed=TRUE: {(df_rooms_clean['events_allowed'] == True).sum():,} rooms")

print("\n✓ Data cleaning complete and verified!")

# %% [markdown]
# ## Configuration Documentation
# 
# The final `CleaningConfig` serves as documentation for all cleaning steps:
# 
# ```python
# config = CleaningConfig(
#     # Basic validation (15 rules)
#     remove_negative_prices=True,
#     remove_zero_prices=True,
#     # ... etc
#     
#     # Fixes
#     fix_empty_strings=True,             # Convert '' to NULL
#     
#     # Imputations (found during EDA)
#     impute_children_allowed=True,       # Section 2.2: From booking behavior
#     impute_events_allowed=True,         # Section 2.2: From room_type
#     
#     # Exclusions (found during EDA)
#     exclude_reception_halls=True,       # Section 2.2: Not accommodation
#     exclude_missing_location=True,      # Section 3.1: Can't analyze location
# )
# ```
# 
# The config itself documents:
# - WHAT was done (field names)
# - WHY it was done (comments)
# - WHEN it was discovered (section references)
