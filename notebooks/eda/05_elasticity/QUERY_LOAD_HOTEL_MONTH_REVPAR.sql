WITH hotel_month AS (
    SELECT 
        b.hotel_id,
        DATE_TRUNC('month', CAST(b.arrival_date AS DATE)) AS month,
        br.room_type,
        r.children_allowed,
        hl.city,
        -- Metrics
        SUM(br.total_price) AS total_revenue,
        COUNT(*) AS room_nights_sold,
        SUM(br.total_price) / NULLIF(COUNT(*), 0) AS avg_adr,
        AVG(br.room_size) AS avg_room_size,
        SUM(r.number_of_rooms) AS total_capacity,
        MAX(r.max_occupancy) AS room_capacity_pax,
        -- Controls
        SUM(CASE WHEN EXTRACT(ISODOW FROM CAST(b.arrival_date AS DATE)) >= 6 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS weekend_ratio
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    JOIN rooms r ON br.room_id = r.id
    WHERE b.status = 'confirmed'
    GROUP BY 1, 2, 3, 4, 5
)
SELECT 
    *,
    (total_revenue / NULLIF(total_capacity * 30, 0)) AS revpar, -- Approx days
    (room_nights_sold::FLOAT / NULLIF(total_capacity * 30, 0)) AS occupancy_rate
FROM hotel_month
WHERE total_capacity > 0 AND room_nights_sold > 0

