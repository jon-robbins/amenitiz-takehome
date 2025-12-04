SELECT 
    br.room_id,
    ANY_VALUE(br.room_type) as room_type,
    r.number_of_rooms,
    COUNT(*) as total_bookings,
    COUNT(*) * 1.0 / NULLIF(r.number_of_rooms, 0) as bookings_per_individual_room
FROM booked_rooms br
JOIN rooms r ON br.room_id = r.id
GROUP BY br.room_id, r.number_of_rooms
ORDER BY total_bookings DESC

