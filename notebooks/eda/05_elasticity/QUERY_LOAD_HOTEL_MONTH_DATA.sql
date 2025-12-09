-- Step 1: Get TRUE hotel capacity (sum of distinct room types per hotel)
WITH hotel_capacity AS (
    SELECT DISTINCT
        b.hotel_id,
        r.id as room_type_id,
        r.number_of_rooms
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN rooms r ON br.room_id = r.id
    WHERE b.status IN ('confirmed', 'Booked')
),
hotel_total_capacity AS (
    SELECT hotel_id, SUM(number_of_rooms) as hotel_rooms
    FROM hotel_capacity
    GROUP BY hotel_id
),

hotel_partner_size AS (
    SELECT 
        hotel_id, 
        COUNT(DISTINCT id) as locations_count 
    FROM hotel_location
    GROUP BY hotel_id
),

-- Step 2: Explode bookings to daily granularity
daily_bookings AS (
    SELECT 
        b.hotel_id,
        CAST(b.arrival_date + (n * INTERVAL '1 day') AS DATE) as stay_date,
        br.room_type,
        COALESCE(NULLIF(br.room_view, ''), 'no_view') AS room_view,
        r.children_allowed,
        hl.city,
        hl.latitude,
        hl.longitude,
        br.total_price / (b.departure_date - b.arrival_date) as nightly_rate,
        br.room_size,
        r.max_occupancy as room_capacity_pax,
        r.events_allowed,
        r.pets_allowed,
        r.smoking_allowed
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    JOIN rooms r ON br.room_id = r.id
    CROSS JOIN generate_series(0, (b.departure_date - b.arrival_date) - 1) as t(n)
    WHERE b.status IN ('confirmed', 'Booked')
      AND CAST(b.arrival_date AS DATE) BETWEEN '2023-01-01' AND '2024-12-31'
      AND hl.city IS NOT NULL
      AND (b.departure_date - b.arrival_date) > 0
),

-- Step 3: FIRST aggregate to HOTEL-MONTH level to get correct total occupancy
hotel_month_totals AS (
    SELECT 
        db.hotel_id,
        DATE_TRUNC('month', db.stay_date) AS month,
        MAX(db.city) as city,
        MAX(db.latitude) as latitude,
        MAX(db.longitude) as longitude,
        
        -- TOTAL revenue and room-nights for the ENTIRE hotel
        SUM(db.nightly_rate) AS total_revenue,
        COUNT(*) AS total_room_nights_sold,
        AVG(db.nightly_rate) AS avg_adr,
        
        -- Temporal
        EXTRACT(MONTH FROM MAX(db.stay_date)) AS month_number,
        EXTRACT(DAY FROM LAST_DAY(MAX(db.stay_date))) AS days_in_month,
        SUM(CASE WHEN EXTRACT(ISODOW FROM db.stay_date) >= 6 THEN 1 ELSE 0 END)::FLOAT / 
            NULLIF(COUNT(*), 0) AS weekend_ratio
    FROM daily_bookings db
    GROUP BY db.hotel_id, month
),

-- Step 4: Get room-type features (most common per hotel-month)
hotel_month_room_features AS (
    SELECT 
        db.hotel_id,
        DATE_TRUNC('month', db.stay_date) AS month,
        -- Use the most common room type/view for this hotel-month
        MODE() WITHIN GROUP (ORDER BY db.room_type) as room_type,
        MODE() WITHIN GROUP (ORDER BY db.room_view) as room_view,
        MAX(db.children_allowed) as children_allowed,
        AVG(db.room_size) AS avg_room_size,
        MAX(db.room_capacity_pax) AS room_capacity_pax,
        (CAST(MAX(db.events_allowed) AS INT) + 
         CAST(MAX(db.pets_allowed) AS INT) + 
         CAST(MAX(db.smoking_allowed) AS INT) + 
         CAST(MAX(db.children_allowed) AS INT)) AS amenities_score
    FROM daily_bookings db
    GROUP BY db.hotel_id, month
)

-- Step 5: Join everything and calculate CORRECT occupancy
SELECT 
    hmt.hotel_id,
    hmt.month,
    hmrf.room_type,
    hmrf.room_view,
    hmrf.children_allowed,
    hmt.city,
    hmt.latitude,
    hmt.longitude,
    hmt.total_revenue,
    hmt.total_room_nights_sold as room_nights_sold,
    hmt.avg_adr,
    hmrf.avg_room_size,
    hmrf.room_capacity_pax,
    hmt.month_number,
    hmt.days_in_month,
    hmt.weekend_ratio,
    hmrf.amenities_score,
    -- View quality (ordinal 0-3)
    CASE 
        WHEN hmrf.room_view IN ('ocean_view', 'sea_view') THEN 3
        WHEN hmrf.room_view IN ('lake_view', 'mountain_view') THEN 2
        WHEN hmrf.room_view IN ('pool_view', 'garden_view') THEN 1
        ELSE 0
    END AS view_quality_ordinal,
    htc.hotel_rooms AS total_capacity,
    -- CORRECT occupancy: TOTAL room_nights / (hotel_capacity × days)
    (hmt.total_room_nights_sold::FLOAT / NULLIF(htc.hotel_rooms * hmt.days_in_month, 0)) AS occupancy_rate,
    COALESCE(hps.locations_count, 1) as partner_size
FROM hotel_month_totals hmt
JOIN hotel_total_capacity htc ON hmt.hotel_id = htc.hotel_id
LEFT JOIN hotel_partner_size hps ON hmt.hotel_id = hps.hotel_id
JOIN hotel_month_room_features hmrf ON hmt.hotel_id = hmrf.hotel_id AND hmt.month = hmrf.month
WHERE htc.hotel_rooms > 0 AND hmt.total_room_nights_sold > 0 AND hmt.avg_adr > 0

