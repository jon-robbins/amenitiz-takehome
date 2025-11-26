"""
Schema Exploration: Understanding Daily Hotel Occupancy

Goal: Figure out which hotels are full vs. have availability on any given day.
This is the foundation for understanding pricing power.

Schema:
    bookings (1 row = 1 booking for a stay)
    ├── hotel_id, arrival_date, departure_date, status
    └── Links to: booked_rooms via booking_id

    booked_rooms (1 row = 1 room booked)
    ├── booking_id, room_id, total_adult, total_children, room_view, room_type, total_price
    └── Links to: rooms via room_id

    rooms (1 row = 1 room TYPE definition)
    ├── id, number_of_rooms (count of physical rooms of this type)
    ├── max_occupancy, max_adults, children_allowed, etc.
    └── NO hotel_id! Only linked via booked_rooms → bookings
"""

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lib.db import init_db
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# %%
print("=" * 80)
print("PART 1: SINGLE HOTEL DEEP DIVE")
print("=" * 80)

con = init_db()

# Find a small hotel with manageable number of bookings
print("\n1.1 Finding a small hotel to analyze...")
small_hotels = con.execute("""
    SELECT 
        b.hotel_id,
        COUNT(DISTINCT b.id) as num_bookings,
        COUNT(DISTINCT br.room_id) as num_room_types,
        MIN(b.arrival_date) as first_booking,
        MAX(b.arrival_date) as last_booking
    FROM bookings b
    JOIN booked_rooms br ON b.id = br.booking_id
    WHERE b.status = 'confirmed'
    GROUP BY b.hotel_id
    HAVING COUNT(DISTINCT b.id) BETWEEN 20 AND 50
    ORDER BY num_room_types ASC, num_bookings ASC
    LIMIT 5
""").fetchdf()
print(small_hotels.to_string())

# Pick the first one
SAMPLE_HOTEL_ID = small_hotels.iloc[0]['hotel_id']
print(f"\n>>> Selected hotel_id = {SAMPLE_HOTEL_ID}")

# %%
print("\n1.2 All bookings for this hotel...")
hotel_bookings = con.execute(f"""
    SELECT 
        b.id as booking_id,
        b.arrival_date,
        b.departure_date,
        (b.departure_date - b.arrival_date) as nights,
        b.status,
        b.total_price as booking_total
    FROM bookings b
    WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
      AND b.status = 'confirmed'
    ORDER BY b.arrival_date
    LIMIT 10
""").fetchdf()
print(hotel_bookings.to_string())

# %%
print("\n1.3 Room types used by this hotel (via booked_rooms)...")
hotel_room_types = con.execute(f"""
    SELECT DISTINCT
        r.id as room_type_id,
        r.number_of_rooms,
        r.max_occupancy,
        br.room_type,
        br.room_view
    FROM bookings b
    JOIN booked_rooms br ON b.id = br.booking_id
    JOIN rooms r ON br.room_id = r.id
    WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
      AND b.status = 'confirmed'
    ORDER BY r.id
""").fetchdf()
print(hotel_room_types.to_string())
print(f"\n>>> This hotel has {len(hotel_room_types)} distinct room types")
print(f">>> Total capacity: {hotel_room_types['number_of_rooms'].sum()} rooms")

# %%
print("\n1.4 Sample booked_rooms entries for this hotel...")
sample_booked_rooms = con.execute(f"""
    SELECT 
        br.id as booked_room_id,
        br.booking_id,
        br.room_id,
        br.room_type,
        br.room_view,
        br.total_adult,
        br.total_children,
        br.total_price,
        b.arrival_date,
        b.departure_date,
        (b.departure_date - b.arrival_date) as nights,
        r.number_of_rooms as room_type_capacity
    FROM bookings b
    JOIN booked_rooms br ON b.id = br.booking_id
    JOIN rooms r ON br.room_id = r.id
    WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
      AND b.status = 'confirmed'
    ORDER BY b.arrival_date
    LIMIT 15
""").fetchdf()
print(sample_booked_rooms.to_string())

# %%
print("\n" + "=" * 80)
print("PART 2: EXPLODE TO DAILY GRANULARITY")
print("=" * 80)

print("\n2.1 Exploding bookings into individual nights...")
# Each booking spans multiple days. We need to create a row for each night.

daily_bookings = con.execute(f"""
    WITH booking_nights AS (
        SELECT 
            b.hotel_id,
            b.id as booking_id,
            br.id as booked_room_id,
            br.room_id as room_type_id,
            br.room_type,
            br.room_view,
            br.total_adult,
            br.total_children,
            br.total_price as total_booking_price,
            b.arrival_date,
            b.departure_date,
            (b.departure_date - b.arrival_date) as nights,
            r.number_of_rooms as room_type_capacity,
            r.max_occupancy
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN rooms r ON br.room_id = r.id
        WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
          AND b.status = 'confirmed'
          AND (b.departure_date - b.arrival_date) > 0
    )
    SELECT 
        hotel_id,
        -- Generate each night of the stay
        CAST(arrival_date + (n * INTERVAL '1 day') AS DATE) as stay_date,
        booking_id,
        booked_room_id,
        room_type_id,
        room_type,
        room_view,
        total_adult,
        total_children,
        room_type_capacity,
        max_occupancy,
        -- Nightly rate = total price / nights
        total_booking_price / nights as nightly_rate,
        nights as total_nights_in_booking
    FROM booking_nights
    -- Cross join with numbers to explode dates
    CROSS JOIN generate_series(0, nights - 1) as t(n)
    ORDER BY stay_date, room_type_id
""").fetchdf()

print(f"Exploded {len(sample_booked_rooms)} bookings into {len(daily_bookings)} room-nights")
print("\nSample of daily data:")
print(daily_bookings.head(20).to_string())

# %%
print("\n" + "=" * 80)
print("PART 3: CALCULATE DAILY CAPACITY AND BOOKED ROOMS")
print("=" * 80)

print("\n3.1 Daily occupancy by room type...")
# Group by date and room type to see how many rooms are booked

daily_by_room_type = con.execute(f"""
    WITH booking_nights AS (
        SELECT 
            b.hotel_id,
            b.id as booking_id,
            br.room_id as room_type_id,
            br.room_type,
            b.arrival_date,
            b.departure_date,
            (b.departure_date - b.arrival_date) as nights,
            r.number_of_rooms as room_type_capacity,
            br.total_price
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN rooms r ON br.room_id = r.id
        WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
          AND b.status = 'confirmed'
          AND (b.departure_date - b.arrival_date) > 0
    ),
    daily_exploded AS (
        SELECT 
            hotel_id,
            CAST(arrival_date + (n * INTERVAL '1 day') AS DATE) as stay_date,
            room_type_id,
            room_type,
            room_type_capacity,
            total_price / nights as nightly_revenue
        FROM booking_nights
        CROSS JOIN generate_series(0, nights - 1) as t(n)
    )
    SELECT 
        stay_date,
        room_type_id,
        room_type,
        room_type_capacity,
        COUNT(*) as rooms_booked,
        SUM(nightly_revenue) as daily_revenue,
        COUNT(*)::FLOAT / room_type_capacity as occupancy_rate
    FROM daily_exploded
    GROUP BY stay_date, room_type_id, room_type, room_type_capacity
    ORDER BY stay_date, room_type_id
""").fetchdf()

print(daily_by_room_type.head(30).to_string())

# %%
print("\n3.2 Check for overbookings (rooms_booked > capacity)...")
overbookings = daily_by_room_type[daily_by_room_type['rooms_booked'] > daily_by_room_type['room_type_capacity']]
if len(overbookings) > 0:
    print(f"WARNING: Found {len(overbookings)} overbooking instances!")
    print(overbookings.head(10).to_string())
else:
    print("No overbookings found - data looks consistent.")

# %%
print("\n" + "=" * 80)
print("PART 4: CALCULATE DAILY HOTEL OCCUPANCY")
print("=" * 80)

print("\n4.1 Aggregate to hotel-day level...")
daily_hotel_occupancy = con.execute(f"""
    WITH booking_nights AS (
        SELECT 
            b.hotel_id,
            CAST(arrival_date + (n * INTERVAL '1 day') AS DATE) as stay_date,
            br.room_id as room_type_id,
            r.number_of_rooms as room_type_capacity,
            br.total_price / (b.departure_date - b.arrival_date) as nightly_revenue
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN rooms r ON br.room_id = r.id
        CROSS JOIN generate_series(0, (b.departure_date - b.arrival_date) - 1) as t(n)
        WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
          AND b.status = 'confirmed'
          AND (b.departure_date - b.arrival_date) > 0
    ),
    hotel_capacity AS (
        -- Get total hotel capacity (sum of all room types)
        SELECT DISTINCT
            b.hotel_id,
            r.id as room_type_id,
            r.number_of_rooms
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN rooms r ON br.room_id = r.id
        WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
    ),
    hotel_total_capacity AS (
        SELECT hotel_id, SUM(number_of_rooms) as total_rooms
        FROM hotel_capacity
        GROUP BY hotel_id
    )
    SELECT 
        bn.stay_date,
        bn.hotel_id,
        htc.total_rooms as hotel_capacity,
        COUNT(*) as rooms_booked,
        SUM(bn.nightly_revenue) as daily_revenue,
        COUNT(*)::FLOAT / htc.total_rooms as occupancy_rate,
        CASE 
            WHEN COUNT(*) >= htc.total_rooms THEN 'FULL'
            WHEN COUNT(*)::FLOAT / htc.total_rooms >= 0.8 THEN 'NEARLY_FULL'
            WHEN COUNT(*)::FLOAT / htc.total_rooms >= 0.5 THEN 'MODERATE'
            ELSE 'LOW'
        END as occupancy_status
    FROM booking_nights bn
    JOIN hotel_total_capacity htc ON bn.hotel_id = htc.hotel_id
    GROUP BY bn.stay_date, bn.hotel_id, htc.total_rooms
    ORDER BY bn.stay_date
""").fetchdf()

print(f"Daily occupancy for hotel {SAMPLE_HOTEL_ID}:")
print(daily_hotel_occupancy.head(30).to_string())

# %%
print("\n4.2 Summary statistics...")
print(f"\nOccupancy distribution:")
print(daily_hotel_occupancy['occupancy_rate'].describe())

print(f"\nOccupancy status counts:")
print(daily_hotel_occupancy['occupancy_status'].value_counts())

print(f"\nDays at 100%+ occupancy: {(daily_hotel_occupancy['occupancy_rate'] >= 1.0).sum()}")
print(f"Days at 80%+ occupancy: {(daily_hotel_occupancy['occupancy_rate'] >= 0.8).sum()}")
print(f"Days at 50%+ occupancy: {(daily_hotel_occupancy['occupancy_rate'] >= 0.5).sum()}")

# %%
print("\n" + "=" * 80)
print("PART 5: AGGREGATE TO MONTHLY AND VALIDATE")
print("=" * 80)

print("\n5.1 Monthly aggregation for sample hotel...")
monthly_occupancy = con.execute(f"""
    WITH booking_nights AS (
        SELECT 
            b.hotel_id,
            CAST(arrival_date + (n * INTERVAL '1 day') AS DATE) as stay_date,
            br.room_id as room_type_id,
            r.number_of_rooms as room_type_capacity,
            br.total_price / (b.departure_date - b.arrival_date) as nightly_revenue
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN rooms r ON br.room_id = r.id
        CROSS JOIN generate_series(0, (b.departure_date - b.arrival_date) - 1) as t(n)
        WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
          AND b.status = 'confirmed'
          AND (b.departure_date - b.arrival_date) > 0
    ),
    hotel_capacity AS (
        SELECT DISTINCT
            b.hotel_id,
            r.id as room_type_id,
            r.number_of_rooms
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN rooms r ON br.room_id = r.id
        WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
    ),
    hotel_total_capacity AS (
        SELECT hotel_id, SUM(number_of_rooms) as total_rooms
        FROM hotel_capacity
        GROUP BY hotel_id
    ),
    daily_stats AS (
        SELECT 
            bn.stay_date,
            bn.hotel_id,
            htc.total_rooms as hotel_capacity,
            COUNT(*) as rooms_booked,
            SUM(bn.nightly_revenue) as daily_revenue
        FROM booking_nights bn
        JOIN hotel_total_capacity htc ON bn.hotel_id = htc.hotel_id
        GROUP BY bn.stay_date, bn.hotel_id, htc.total_rooms
    )
    SELECT 
        DATE_TRUNC('month', stay_date) as month,
        hotel_id,
        hotel_capacity,
        COUNT(*) as days_with_bookings,
        SUM(rooms_booked) as total_room_nights_sold,
        hotel_capacity * COUNT(*) as total_room_nights_available,
        SUM(rooms_booked)::FLOAT / (hotel_capacity * COUNT(*)) as monthly_occupancy,
        SUM(daily_revenue) as monthly_revenue,
        SUM(daily_revenue) / NULLIF(SUM(rooms_booked), 0) as avg_daily_rate,
        SUM(daily_revenue) / (hotel_capacity * COUNT(*)) as revpar
    FROM daily_stats
    GROUP BY month, hotel_id, hotel_capacity
    ORDER BY month
""").fetchdf()

print(monthly_occupancy.to_string())

# %%
print("\n5.2 Validate against old (buggy) calculation...")
old_calculation = con.execute(f"""
    SELECT 
        DATE_TRUNC('month', b.arrival_date) as month,
        b.hotel_id,
        COUNT(*) as booking_count,
        SUM(r.number_of_rooms) as sum_number_of_rooms,
        SUM(br.total_price) as total_revenue,
        COUNT(*)::FLOAT / NULLIF(SUM(r.number_of_rooms) * 30, 0) as old_occupancy
    FROM bookings b
    JOIN booked_rooms br ON b.id = br.booking_id
    JOIN rooms r ON br.room_id = r.id
    WHERE b.hotel_id = {SAMPLE_HOTEL_ID}
      AND b.status = 'confirmed'
    GROUP BY month, b.hotel_id
    ORDER BY month
""").fetchdf()

print("\nOLD (buggy) calculation:")
print(old_calculation.to_string())

print("\n" + "=" * 80)
print("COMPARISON: OLD vs NEW")
print("=" * 80)
comparison = monthly_occupancy[['month', 'monthly_occupancy', 'avg_daily_rate']].merge(
    old_calculation[['month', 'old_occupancy']], 
    on='month', 
    how='outer'
)
comparison['occupancy_ratio'] = comparison['monthly_occupancy'] / comparison['old_occupancy']
print(comparison.to_string())

# %%
print("\n" + "=" * 80)
print("FINAL: REUSABLE QUERY FOR ALL HOTELS")
print("=" * 80)

print("\n6.1 Daily occupancy for ALL hotels (sample)...")
all_hotels_daily = con.execute("""
    WITH hotel_capacity AS (
        -- Get each hotel's room types and capacity
        SELECT DISTINCT
            b.hotel_id,
            r.id as room_type_id,
            r.number_of_rooms
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN rooms r ON br.room_id = r.id
        WHERE b.status = 'confirmed'
    ),
    hotel_total_capacity AS (
        SELECT hotel_id, SUM(number_of_rooms) as total_rooms
        FROM hotel_capacity
        GROUP BY hotel_id
    ),
    booking_nights AS (
        -- Explode each booking into individual nights
        SELECT 
            b.hotel_id,
            CAST(b.arrival_date + (n * INTERVAL '1 day') AS DATE) as stay_date,
            br.total_price / (b.departure_date - b.arrival_date) as nightly_revenue
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        CROSS JOIN generate_series(0, (b.departure_date - b.arrival_date) - 1) as t(n)
        WHERE b.status = 'confirmed'
          AND (b.departure_date - b.arrival_date) > 0
          AND b.arrival_date >= '2024-01-01'  -- Limit for performance
    )
    SELECT 
        bn.stay_date,
        bn.hotel_id,
        htc.total_rooms as hotel_capacity,
        COUNT(*) as rooms_booked,
        SUM(bn.nightly_revenue) as daily_revenue,
        COUNT(*)::FLOAT / htc.total_rooms as occupancy_rate,
        SUM(bn.nightly_revenue) / COUNT(*) as adr,
        SUM(bn.nightly_revenue) / htc.total_rooms as revpar
    FROM booking_nights bn
    JOIN hotel_total_capacity htc ON bn.hotel_id = htc.hotel_id
    GROUP BY bn.stay_date, bn.hotel_id, htc.total_rooms
    ORDER BY bn.stay_date, bn.hotel_id
    LIMIT 100
""").fetchdf()

print(all_hotels_daily.to_string())

print("\n6.2 Occupancy distribution across all hotel-days:")
print(all_hotels_daily['occupancy_rate'].describe())

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
print("""
KEY FINDINGS:
1. Each booking spans multiple nights - must explode to daily level
2. ADR = total_price / nights (not per booking)
3. Hotel capacity = SUM(number_of_rooms) for all room types linked to that hotel
4. Daily occupancy = rooms_booked_that_day / hotel_capacity
5. Monthly occupancy = total_room_nights_sold / (capacity × days)

CORRECT FORMULA:
    occupancy = SUM(rooms booked per day) / (hotel_total_rooms × days_in_period)
    
NOT:
    occupancy = COUNT(bookings) / (SUM(number_of_rooms per booking) × days)
""")
