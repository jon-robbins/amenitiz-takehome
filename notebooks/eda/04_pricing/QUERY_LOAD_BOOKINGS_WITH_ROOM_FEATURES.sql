SELECT 
    b.id as booking_id,
    b.hotel_id,
    b.arrival_date,
    b.departure_date,
    CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
    br.total_price as room_price,
    br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price,
    br.room_id,
    br.room_type,
    br.room_size,
    br.room_view,
    br.total_adult,
    br.total_children,
    r.max_occupancy,
    r.max_adults,
    r.pricing_per_person_activated as pricing_per_person,
    r.events_allowed,
    r.pets_allowed,
    r.smoking_allowed,
    r.children_allowed,
    hl.city,
    hl.country
FROM bookings b
JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
JOIN rooms r ON br.room_id = r.id
JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
WHERE b.status IN ('confirmed', 'Booked')
  AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
  AND br.total_price > 0

