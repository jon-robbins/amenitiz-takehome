SELECT 
    b.hotel_id,
    EXTRACT(YEAR FROM CAST(b.arrival_date AS DATE)) AS year,
    EXTRACT(MONTH FROM CAST(b.arrival_date AS DATE)) AS month,
    br.room_type,
    COALESCE(NULLIF(br.room_view, ''), 'no_view') AS room_view,
    r.children_allowed,
    AVG(br.total_price) AS avg_adr,
    SUM(br.total_price) AS total_revenue,
    COUNT(*) AS room_nights,
    SUM(r.number_of_rooms) AS total_capacity
FROM bookings b
JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
JOIN rooms r ON br.room_id = r.id
WHERE b.status IN ('confirmed', 'Booked')
  AND EXTRACT(YEAR FROM CAST(b.arrival_date AS DATE)) IN (2023, 2024)
GROUP BY 
    b.hotel_id, 
    year, 
    month, 
    br.room_type, 
    room_view, 
    r.children_allowed
HAVING COUNT(*) >= 5  -- Minimum sample size per product-month

