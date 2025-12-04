SELECT 
    br.room_type,
    COUNT(DISTINCT r.id) as num_configurations,
    SUM(r.number_of_rooms) as total_units,
    AVG(r.number_of_rooms) as avg_units_per_config,
    MEDIAN(r.number_of_rooms) as median_units_per_config
FROM rooms r
JOIN booked_rooms br ON br.room_id = r.id
WHERE br.room_type IS NOT NULL
GROUP BY br.room_type
ORDER BY total_units DESC

