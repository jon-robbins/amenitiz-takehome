import duckdb

def clean_data(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    print("Dropping dirty data...")

    # Group A: Financial Sanity - Drop from booked_rooms
    # 1. Negative prices
    con.execute("DELETE FROM booked_rooms WHERE total_price < 0")
    print("  Dropped negative prices")

    # 2. Zero prices
    con.execute("DELETE FROM booked_rooms WHERE total_price = 0")
    print("  Dropped zero prices")

    # 3. Extreme prices (>5000/night) - need to join with bookings to calculate daily rate
    con.execute("""
        DELETE FROM booked_rooms 
        WHERE id IN (
            SELECT br.id
            FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            WHERE (b.status = 'confirmed' OR b.status = 'Booked')
            AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
            AND (br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) > 5000
        )
    """)
    print("  Dropped extreme prices (>5k/night)")

    # Group B: Temporal Sanity - Drop from bookings
    # 4. Negative stays
    con.execute("DELETE FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)")
    print("  Dropped negative stays")

    # 5. Negative lead times
    con.execute("DELETE FROM bookings WHERE CAST(created_at AS DATE) > CAST(arrival_date AS DATE)")
    print("  Dropped negative lead times")

    # Group C: Physics Sanity - Drop from booked_rooms
    # 6. Overcrowded rooms
    con.execute("""
        DELETE FROM booked_rooms 
        WHERE id IN (
            SELECT br.id
            FROM booked_rooms br
            JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
            WHERE (COALESCE(br.total_adult, 0) + COALESCE(br.total_children, 0)) > r.max_occupancy
        )
    """)
    print("  Dropped overcrowded rooms")

    # 7. Impossible occupancy - This is complex, we'll drop the bookings that contribute to overbooking
    # We'll identify bookings that have at least one day where occupancy > 100%
    con.execute("""
        DELETE FROM bookings
        WHERE id IN (
            WITH daily_usage AS (
                SELECT 
                    CAST(br.booking_id AS BIGINT) as booking_id,
                    CAST(br.room_id AS BIGINT) as room_id,
                    UNNEST(GENERATE_SERIES(
                        CAST(b.arrival_date AS DATE), 
                        CAST(b.departure_date AS DATE) - INTERVAL 1 DAY, 
                        INTERVAL 1 DAY
                    )) as stay_date
                FROM bookings b
                JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
                WHERE (b.status = 'confirmed' OR b.status = 'Booked')
                AND CAST(b.arrival_date AS DATE) < CAST(b.departure_date AS DATE)
            ),
            daily_counts AS (
                SELECT 
                    room_id,
                    stay_date,
                    COUNT(*) as rooms_sold
                FROM daily_usage
                GROUP BY 1, 2
            ),
            overbooked_days AS (
                SELECT DISTINCT du.booking_id
                FROM daily_counts dc
                JOIN rooms r ON CAST(dc.room_id AS BIGINT) = r.id
                JOIN daily_usage du ON dc.room_id = du.room_id AND dc.stay_date = du.stay_date
                WHERE dc.rooms_sold > r.number_of_rooms
            )
            SELECT booking_id FROM overbooked_days
        )
    """)
    print("  Dropped bookings with impossible occupancy")

    # Group D: Linkage Sanity - Drop from bookings
    # 8. Orphan bookings
    con.execute("""
        DELETE FROM bookings 
        WHERE id NOT IN (SELECT DISTINCT CAST(booking_id AS BIGINT) FROM booked_rooms)
    """)
    print("  Dropped orphan bookings")

    return con