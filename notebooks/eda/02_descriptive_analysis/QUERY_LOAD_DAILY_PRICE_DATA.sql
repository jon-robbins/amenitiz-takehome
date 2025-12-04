SELECT 
    br.id as booked_room_id,
    br.booking_id,
    br.room_id,
    br.room_type,
    br.room_size,
    br.total_price,
    b.arrival_date,
    b.departure_date,
    DATE_DIFF('day', b.arrival_date, b.departure_date) as stay_length_days,
    br.total_price / NULLIF(DATE_DIFF('day', b.arrival_date, b.departure_date), 0) as daily_price,
    br.total_adult + br.total_children as total_guests,
    b.hotel_id
FROM booked_rooms br
JOIN bookings b ON b.id = br.booking_id
WHERE b.arrival_date IS NOT NULL 
  AND b.departure_date IS NOT NULL
  AND DATE_DIFF('day', b.arrival_date, b.departure_date) > 0
  AND br.total_price > 0
  AND br.room_type IS NOT NULL

