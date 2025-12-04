SELECT 
    b.hotel_id,
    COUNT(DISTINCT br.room_id) as num_configurations,
    SUM(r.number_of_rooms) as total_units,
    COUNT(DISTINCT br.room_type) as num_categories,
    STRING_AGG(DISTINCT br.room_type, ', ') as categories
FROM bookings b
JOIN booked_rooms br ON br.booking_id = b.id
JOIN rooms r ON r.id = br.room_id
WHERE b.hotel_id IS NOT NULL AND br.room_type IS NOT NULL
GROUP BY b.hotel_id

