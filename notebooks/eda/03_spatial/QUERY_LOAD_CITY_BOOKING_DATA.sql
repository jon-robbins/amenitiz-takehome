SELECT 
    hl.city,
    b.id as booking_id,
    hl.hotel_id,
    br.room_id,
    br.total_price,
    (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as nights
FROM bookings b
JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
WHERE b.status IN ('confirmed', 'Booked')
  AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
  AND hl.city IS NOT NULL

