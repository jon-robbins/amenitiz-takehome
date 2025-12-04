SELECT 
    b.id as booking_id,
    b.hotel_id,
    b.arrival_date,
    b.departure_date,
    CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
    br.total_price as room_price,
    br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price,
    br.room_type
FROM bookings b
JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
WHERE b.status IN ('confirmed', 'Booked')
  AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
  AND br.total_price > 0
  AND b.arrival_date IS NOT NULL
  AND b.departure_date IS NOT NULL

