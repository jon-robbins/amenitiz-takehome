# %%
"""
Data Quality Assessment
Shows the percentage of dirty data in the raw database.
"""

import sys
sys.path.append('../..')

from notebooks.utils.db import get_connection
from notebooks.utils.data_validator import check_data_quality
from notebooks.utils.db import init_db
from notebooks.utils.data_validator import validate_and_clean

# %%
# Load raw data
con = get_connection()

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
# Show overall quality percentage
total_rows = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
quality_pct = ((total_rows - results['total_failed']) / total_rows * 100) if total_rows > 0 else 0

print(f"\n{'='*60}")
print(f"OVERALL DATA QUALITY: {quality_pct:.2f}%")
print(f"{'='*60}")

# %% [markdown]
# We can use about 95% of the data for our analysis. 
# In a real world scenario I would want to dig more in depth into the 5% to see what we can save, but for the purposes of this I'm going to make the assumption that it's all due to data quality issues.
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
# Takeaways:
# - Room view is empty when there's no view. We'll replace empty strings with 'No view' and then convert to categorical.
# - When there's no lat/long, it's an empty string. We'll replace them with nulls.
# - Hotel location has empty strings for some values, we'll replace them with nulls.

# I'll make the changes in the data validator class and move on to the EDA.

# %%
# Let's also check to see if there are any columns that only have one value, we can drop them. 
print("Booked Rooms Unique Values: ", booked_rooms.nunique())
print("Rooms Unique Values: ", rooms.nunique())
print("Bookings Unique Values: ", bookings.nunique())
print("Hotel Location Unique Values: ", hotel_location.nunique())
# %%
# Takeaways:
# - events_allowed, pets_allowed, smoking_allowed, and children_allowed only have one value. We can drop these columns entirely. 
# However, we do have booked_rooms.total_children. If there is >=1 booking of a room_id that has >=1 child, then we can impute "TRUE" for the rooms.children_allowed column.
# We can also do the same for reception halls. Events are definitely allowed for reception halls, so we can fix that feature. 

con_raw = init_db()
con = validate_and_clean(con_raw)

df_rooms_clean = con.execute("SELECT * FROM rooms").fetchdf()
df_booked_rooms_clean = con.execute("SELECT * FROM booked_rooms").fetchdf()
df_bookings_clean = con.execute("SELECT * FROM bookings").fetchdf()
df_hotel_location_clean = con.execute("SELECT * FROM hotel_location").fetchdf()

print("Original df children distribution: ",rooms['children_allowed'].value_counts())
print("New df children distribution: ",df_rooms_clean['children_allowed'].value_counts())
print("-"*50)
print("Original df events distribution: ",rooms['events_allowed'].value_counts())
print("New df events distribution: ",df_rooms_clean['events_allowed'].value_counts())
print("-"*50)

# %% [markdown]

#Let's check to see if there are any bookings that are missing location data.

con = init_db()

query = "select * from hotel_location where city is null and (latitude is null or longitude is null)"
con.execute(query).fetchdf()

# %% [markdown]
#6800 of them. Do they correspond to bookings?
query = """
SELECT 
    br.*,
    b.id as booking_id,
    b.hotel_id,
    b.status,
    b.arrival_date,
    b.departure_date,
    b.total_price as booking_total_price,
    hl.city,
    hl.country,
    hl.latitude,
    hl.longitude
FROM booked_rooms br
JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
WHERE b.status IN ('confirmed', 'Booked')
  AND hl.city IS NULL 
  AND (hl.latitude IS NULL OR hl.longitude IS NULL)
"""
con.execute(query).fetchdf()

# %% [markdown]
# Not that many. We'll add a parameter to the data_validator to exclude bookings from hotels with missing location data.
con = validate_and_clean(init_db(), exclude_missing_location_bookings=True)
con.execute(query).fetchdf()
# %%
