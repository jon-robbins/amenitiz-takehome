SELECT DISTINCT
    hotel_id,
    latitude,
    longitude,
    city,
    country
FROM hotel_location
WHERE latitude IS NOT NULL
  AND longitude IS NOT NULL

