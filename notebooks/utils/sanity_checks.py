import duckdb

def run_check(con, query_fail, query_total, check_name):
    """
    Execute a sanity check and display results.
    query_fail: SQL query that counts the number of FAILED rows.
    query_total: SQL query that counts the TOTAL rows for the denominator.
    """
    try:
        failed_count = con.execute(query_fail).fetchone()[0]
        total_count = con.execute(query_total).fetchone()[0]
        
        if total_count == 0:
            pct = 0.0
        else:
            pct = (failed_count / total_count) * 100
            
        # Recommendation Logic
        if pct < 1:
            rec = "DROP (Safe)"
        elif pct < 5:
            rec = "INVESTIGATE"
        else:
            rec = "CRITICAL DATA ISSUE"
        
        # Display results
        print(f"\n{check_name}:")
        print(f"  Failed Rows: {failed_count:,}")
        print(f"  Total Rows: {total_count:,}")
        print(f"  Failure %: {pct:.2f}%")
        print(f"  Recommendation: {rec}")
        
    except Exception as e:
        print(f"Error running check '{check_name}': {e}")

def check_negative_price(con):
    """Group A: Check for negative prices in booked_rooms"""
    run_check(
        con,
        "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0",
        "SELECT COUNT(*) FROM booked_rooms",
        "Negative Price"
    )

def check_zero_price(con):
    """Group A: Check for zero prices in booked_rooms"""
    run_check(
        con,
        "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0",
        "SELECT COUNT(*) FROM booked_rooms",
        "Zero Price"
    )

def check_extreme_price(con):
    """Group A: Check for extreme prices (>5000/night) in confirmed bookings"""
    run_check(
        con,
        """
        SELECT COUNT(*) 
        FROM booked_rooms br
        JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
        WHERE (b.status = 'confirmed' OR b.status = 'Booked')
          AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
          AND (br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) > 5000
        """,
        """
        SELECT COUNT(*) 
        FROM booked_rooms br
        JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
        WHERE (b.status = 'confirmed' OR b.status = 'Booked')
          AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
        """,
        "Extreme Price (>5k/night)"
    )

def check_negative_stay(con):
    """Group B: Check for bookings where departure <= arrival"""
    run_check(
        con,
        "SELECT COUNT(*) FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)",
        "SELECT COUNT(*) FROM bookings",
        "Negative Stay (Dep <= Arr)"
    )

def check_negative_lead_time(con):
    """Group B: Check for bookings created after arrival date"""
    run_check(
        con,
        "SELECT COUNT(*) FROM bookings WHERE CAST(created_at AS DATE) > CAST(arrival_date AS DATE)",
        "SELECT COUNT(*) FROM bookings",
        "Negative Lead Time"
    )

def check_overcrowded_room(con):
    """Group C: Check for rooms with more occupants than max_occupancy"""
    run_check(
        con,
        """
        SELECT COUNT(*)
        FROM booked_rooms br
        JOIN rooms r ON br.room_id = r.id
        WHERE (COALESCE(br.total_adult, 0) + COALESCE(br.total_children, 0)) > r.max_occupancy
        """,
        "SELECT COUNT(*) FROM booked_rooms",
        "Overcrowded Room"
    )

def check_impossible_occupancy(con):
    """Group C: Check for days where rooms_sold > capacity"""
    query_overbook_fail = """
    WITH daily_usage AS (
        SELECT 
            br.room_id,
            UNNEST(GENERATE_SERIES(
                CAST(b.arrival_date AS DATE), 
                CAST(b.departure_date AS DATE) - INTERVAL 1 DAY, 
                INTERVAL 1 DAY
            )) as stay_date
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        WHERE b.status = 'confirmed' OR b.status = 'Booked'
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
    capacity_check AS (
        SELECT 
            dc.room_id,
            dc.stay_date,
            dc.rooms_sold,
            r.number_of_rooms as capacity
        FROM daily_counts dc
        JOIN rooms r ON dc.room_id = r.id
    )
    SELECT COUNT(*) FROM capacity_check WHERE rooms_sold > capacity
    """
    
    query_overbook_total = """
    WITH daily_usage AS (
        SELECT 
            br.room_id,
            UNNEST(GENERATE_SERIES(
                CAST(b.arrival_date AS DATE), 
                CAST(b.departure_date AS DATE) - INTERVAL 1 DAY, 
                INTERVAL 1 DAY
            )) as stay_date
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        WHERE b.status = 'confirmed' OR b.status = 'Booked'
          AND CAST(b.arrival_date AS DATE) < CAST(b.departure_date AS DATE)
    )
    SELECT COUNT(*) FROM daily_usage
    """
    
    run_check(
        con,
        query_overbook_fail,
        query_overbook_total,
        "Impossible Occupancy (>100%)"
    )

def check_orphan_bookings(con):
    """Group D: Check for bookings with no matching booked_rooms"""
    run_check(
        con,
        """
        SELECT COUNT(*) 
        FROM bookings b 
        LEFT JOIN booked_rooms br ON b.id = br.booking_id 
        WHERE br.booking_id IS NULL
        """,
        "SELECT COUNT(*) FROM bookings",
        "Orphan Bookings (No Items)"
    )


