# %% [markdown]
#   # Exploring Hotel Data 
# 
# 
#   This notebook demonstrates how to use the database initialization function to load CSV data into DuckDB and explore it with SQL queries.

# %%
from notebooks.utils.db import init_db
# %load_ext autoreload
# %autoreload 2

# Initialize the database with CSV data
con = init_db()
print("Database initialized successfully!")
tables = con.execute("SHOW TABLES").fetchall()
print("Available tables:")
for table in tables:
    print(f"- {table[0]}")



# %%
booked_rooms = con.execute("select * from booked_rooms").fetchdf()
bookings = con.execute("select * from bookings").fetchdf()
hotel_location = con.execute("select * from hotel_location").fetchdf()
rooms = con.execute("select * from rooms").fetchdf()

print("Booked Rooms:")
booked_rooms.head(15)



# %%
print("Rooms:")
rooms.head(15)



# %%
print("Bookings:")
bookings.head(15)



# %%
print("Hotel location:")
hotel_location.head(15)



# %% [markdown]
#   # Problem framing
# 
# 
# 
#   Within this dataset we have multiple tables:
# 
# 
# 
#   ## `bookings` ( Transaction log)
# 
# 
# 
#   Transaction log. One row is one reservation, which might have multiple days or multiple rooms.
# 
# 
# 
#   Key Data:
# 
# 
# 
#   - Dates: created_at (when they booked), arrival_date, departure_date. These are critical for calculating Lead Time and Length of Stay.
# 
# 
# 
#   - Financials: total_price (total cost of the stay), status (confirmed/cancelled).
# 
# 
# 
#   - Context: source (Booking.com, Airbnb, etc.)—useful for understanding channel costs or customer types.
# 
# 
# 
#   - Join Key: id (joins to booked_rooms.booking_id), hotel_id.
# 
# 
# 
#   ## `booked_rooms` (Order details)
# 
# 
# 
#   The order details. This links a booking to a specific inventory. If someone books 1 suite and one standard room in a booking, then this table will have two rows for this single booking ID.
# 
# 
# 
#   Key Data:
# 
# 
# 
#   - Price Breakdown: total_price (price for this specific room, separate from the booking total).
# 
# 
# 
#   - Room Specifics: room_view, room_size, room_type.
# 
# 
# 
#   - Guest Details: total_adult, total_children.
# 
# 
# 
#   - Join Key: booking_id (links back to bookings), room_id (links to rooms).
# 
# 
# 
#   ## `rooms` (Inventory definitions)
# 
# 
# 
#   This table defines the types of products the hotels are selling. It describes the static attributes of a room category.
# 
# 
# 
#   Key Data:
# 
# 
# 
#   - Capacity: number_of_rooms (Crucial! This is the total stock/inventory count for this room type), max_occupancy.
# 
# 
# 
#   - Amenities: events_allowed, pets_allowed, etc.
# 
# 
# 
#   - Pricing Rules: pricing_per_person_activated.
# 
# 
# 
#   - Join Key: id (links to booked_rooms.room_id).
# 
# 
# 
#   ## `hotel_location` (Property metadata)
# 
#   Key Data:
# 
# 
# 
#  - Location: city, country, latitude, longitude.
# 
# 
# 
#  - Usage: Essential for clustering hotels. A hotel in "Paris" has a different demand curve than a hotel in "Rimini."
# 
# 
# 
#  - Join Key: hotel_id (links to bookings.hotel_id).
# 
# 

# %% [markdown]
#  # Initial EDA
# 
# ## Sanity checks
# 
# We're going to run some sanity checks to get a sense of the data.
# We're going to check for:
# - Negative prices (booked_rooms.total_price < 0)
# - Zero prices (booked_rooms.total_price = 0)
# - Extreme prices (booked_rooms.total_price > 5000)
# - Negative stay (bookings.departure_date <= bookings.arrival_date)
# - Negative lead time (bookings.created_at > bookings.arrival_date)
# - Overcrowded room (booked_rooms.total_adult + booked_rooms.total_children > rooms.max_occupancy)
# - Impossible occupancy (booked_rooms.rooms_sold > rooms.number_of_rooms)
# - Orphan bookings (bookings.id not in booked_rooms.booking_id)
# 
from notebooks.utils.sanity_checks import  *


check_negative_price(con)
check_zero_price(con)
check_extreme_price(con)
check_negative_stay(con)
check_negative_lead_time(con)
check_overcrowded_room(con)
check_impossible_occupancy(con)
check_orphan_bookings(con)


# %% [markdown]
# There are failures, but most of them are around 1% of the data. For efficiency we're going to drop them and assume that they are edge cases that can be ignored. 
# %%
