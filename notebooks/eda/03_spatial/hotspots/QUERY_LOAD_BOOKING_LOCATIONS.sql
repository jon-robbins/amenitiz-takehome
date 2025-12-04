SELECT
    b.id AS booking_id,
    b.total_price,
    CAST(b.arrival_date AS DATE) AS arrival_date,
    CAST(b.departure_date AS DATE) AS departure_date,
    hl.city,
    hl.country,
    hl.latitude,
    hl.longitude
FROM bookings b
JOIN hotel_location hl
  ON b.hotel_id = hl.hotel_id
WHERE b.status IN ('confirmed', 'Booked')
  AND hl.latitude IS NOT NULL
  AND hl.longitude IS NOT NULL

