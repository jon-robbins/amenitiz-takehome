SELECT 
    b.id as booking_id,
    b.hotel_id,
    b.arrival_date,
    b.departure_date,
    CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
    br.total_price as room_price,
    br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price,
    br.room_type,
    hl.city,
    hl.country,
    -- Extract temporal features from arrival date
    EXTRACT(MONTH FROM CAST(b.arrival_date AS DATE)) as arrival_month,
    EXTRACT(DOW FROM CAST(b.arrival_date AS DATE)) as arrival_dow,
    EXTRACT(YEAR FROM CAST(b.arrival_date AS DATE)) as arrival_year,
    -- Extract from departure date
    EXTRACT(MONTH FROM CAST(b.departure_date AS DATE)) as departure_month,
    EXTRACT(DOW FROM CAST(b.departure_date AS DATE)) as departure_dow
FROM bookings b
JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
WHERE b.status IN ('confirmed', 'Booked')
  AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
  AND br.total_price > 0
  AND b.arrival_date IS NOT NULL
  AND b.departure_date IS NOT NULL

