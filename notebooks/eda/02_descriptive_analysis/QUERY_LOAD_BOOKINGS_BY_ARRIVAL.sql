SELECT 
    b.id as booking_id,
    b.arrival_date,
    b.total_price,
    b.created_at,
    EXTRACT(YEAR FROM CAST(b.arrival_date AS DATE)) as arrival_year,
    EXTRACT(MONTH FROM CAST(b.arrival_date AS DATE)) as arrival_month
FROM bookings b
WHERE b.status IN ('confirmed', 'Booked')
  AND b.arrival_date IS NOT NULL
  AND b.total_price > 0

