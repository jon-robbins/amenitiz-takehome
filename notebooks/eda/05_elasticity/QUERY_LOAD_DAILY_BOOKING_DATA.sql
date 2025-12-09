-- Daily booking data for price prediction with holiday features
-- Each row = one booking on a specific arrival_date
-- This enables learning date-specific pricing patterns (holidays, weekends, etc.)

WITH booking_details AS (
    SELECT 
        b.id as booking_id,
        b.hotel_id,
        b.arrival_date,
        b.departure_date,
        b.created_at,
        b.total_price as booking_total_price,
        (b.departure_date - b.arrival_date) as stay_nights,
        -- Lead time: days between booking and arrival
        (b.arrival_date - CAST(b.created_at AS DATE)) as lead_time_days,
        EXTRACT(DOW FROM b.arrival_date) as day_of_week,
        EXTRACT(MONTH FROM b.arrival_date) as month_number,
        CASE WHEN EXTRACT(DOW FROM b.arrival_date) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
        
        -- Room details (aggregate if multiple rooms per booking)
        AVG(br.room_size) as avg_room_size,
        SUM(br.total_adult + br.total_children) as total_guests,
        MODE() WITHIN GROUP (ORDER BY br.room_type) as room_type,
        MODE() WITHIN GROUP (ORDER BY COALESCE(NULLIF(br.room_view, ''), 'no_view')) as room_view,
        COUNT(DISTINCT br.room_id) as rooms_booked
        
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    WHERE b.status IN ('Booked', 'confirmed')
      AND b.arrival_date >= '2023-01-01'
      AND b.arrival_date <= '2024-12-31'
      AND b.total_price > 0
      AND (b.departure_date - b.arrival_date) >= 1
    GROUP BY b.id, b.hotel_id, b.arrival_date, b.departure_date, b.created_at, b.total_price
),

hotel_features AS (
    SELECT 
        hl.hotel_id,
        hl.city,
        hl.latitude,
        hl.longitude,
        
        -- Room configuration
        MAX(r.max_occupancy) as room_capacity_pax,
        SUM(r.number_of_rooms) as total_capacity,
        MAX(r.children_allowed::INT) as children_allowed,
        (MAX(r.events_allowed::INT) + MAX(r.pets_allowed::INT) + 
         MAX(r.smoking_allowed::INT) + MAX(r.children_allowed::INT)) as amenities_score
         
    FROM hotel_location hl
    JOIN bookings b ON hl.hotel_id = b.hotel_id
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN rooms r ON br.room_id = r.id
    GROUP BY hl.hotel_id, hl.city, hl.latitude, hl.longitude
)

SELECT 
    bd.booking_id,
    bd.hotel_id,
    bd.arrival_date,
    bd.created_at,  -- For competitor occupancy calculation
    bd.stay_nights,
    bd.lead_time_days,
    bd.day_of_week,
    bd.month_number,
    bd.is_weekend,
    
    -- Target: daily price (total / nights)
    bd.booking_total_price / NULLIF(bd.stay_nights, 0) as daily_price,
    
    -- Room features
    bd.avg_room_size,
    bd.total_guests,
    bd.room_type,
    bd.room_view,
    bd.rooms_booked,
    
    -- Hotel features
    hf.city,
    hf.latitude,
    hf.longitude,
    hf.room_capacity_pax,
    hf.total_capacity,
    hf.children_allowed,
    hf.amenities_score

FROM booking_details bd
JOIN hotel_features hf ON bd.hotel_id = hf.hotel_id
WHERE bd.booking_total_price / NULLIF(bd.stay_nights, 0) BETWEEN 20 AND 500  -- Filter outliers
ORDER BY bd.arrival_date

