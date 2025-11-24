# %%
"""
Section 5.2: Occupancy-Based Pricing Analysis

Question: How do same-day bookings differ from other bookings, and where are hotels
underpricing due to high occupancy + high last-minute demand?

Insight: Last-minute discounts are rational for low-occupancy dates (inventory clearing).
However, if a hotel has high occupancy AND high last-minute booking volume,
they're leaving money on the table - they could charge more.

Approach:
- Compare same-day vs advance bookings
- Identify dates with high occupancy (≥80%) + high last-minute bookings (≥20%)
- Calculate potential revenue gain from better dynamic pricing
"""

# %%
import sys
sys.path.insert(0, '../../..')
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
from lib.eda_utils import (
    analyze_same_day_bookings,
    identify_underpricing_opportunities,
    plot_occupancy_pricing_analysis,
    print_occupancy_pricing_summary
)
import pandas as pd
from pathlib import Path

# %%
# Initialize database with FULL cleaning configuration
print("Initializing database with full data cleaning...")

config = CleaningConfig(
    # Enable ALL cleaning rules
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_low_prices=True,
    remove_null_prices=True,
    remove_extreme_prices=True,
    remove_null_dates=True,
    remove_null_created_at=True,
    remove_negative_stay=True,
    remove_negative_lead_time=True,
    remove_null_occupancy=True,
    remove_overcrowded_rooms=True,
    remove_null_room_id=True,
    remove_null_booking_id=True,
    remove_null_hotel_id=True,
    remove_orphan_bookings=True,
    remove_null_status=True,
    remove_cancelled_but_active=True,
    remove_bookings_before_2023=True,
    remove_bookings_after_2024=True,
    exclude_reception_halls=True,
    exclude_missing_location=True,
    fix_empty_strings=True,
    impute_children_allowed=True,
    impute_events_allowed=True,
    verbose=True
)

cleaner = DataCleaner(config)
con = cleaner.clean(init_db())

# %%
print("=" * 80)
print("SECTION 5.2: OCCUPANCY-BASED PRICING ANALYSIS")
print("=" * 80)

# %%
# Load booking data with lead time
print("\nLoading booking data...")
bookings_lead_time = con.execute("""
    SELECT 
        b.id as booking_id,
        b.arrival_date,
        b.departure_date,
        b.created_at,
        b.hotel_id,
        CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
        br.total_price as room_price,
        br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price,
        br.room_type,
        DATE_DIFF('day', CAST(b.created_at AS DATE), CAST(b.arrival_date AS DATE)) as lead_time_days
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
      AND br.total_price > 0
      AND b.arrival_date IS NOT NULL
      AND b.created_at IS NOT NULL
      AND DATE_DIFF('day', CAST(b.created_at AS DATE), CAST(b.arrival_date AS DATE)) >= 0
""").fetchdf()

print(f"Loaded {len(bookings_lead_time):,} bookings")

# %%
# Analyze same-day vs advance bookings
print("\nAnalyzing same-day vs advance bookings...")
same_day_stats = analyze_same_day_bookings(bookings_lead_time)

# %%
# Identify underpricing opportunities
print("\nIdentifying underpricing opportunities...")
print("Criteria: Occupancy ≥80% AND Last-minute bookings ≥20%")
underpriced_dates = identify_underpricing_opportunities(
    bookings_lead_time,
    min_occupancy=80.0,
    min_last_minute_pct=20.0
)

# %%
# Print comprehensive summary
print_occupancy_pricing_summary(same_day_stats, underpriced_dates, bookings_lead_time)

# %%
# Create visualization
print("\nCreating visualizations...")
output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "section_5_2_occupancy_pricing.png"

plot_occupancy_pricing_analysis(bookings_lead_time, underpriced_dates, same_day_stats, str(output_path))
print(f"Saved visualization to {output_path}")

# %%
print("\n" + "=" * 80)
print("SECTION 5.2: KEY FINDINGS SUMMARY")
print("=" * 80)

print(f"""
UNDERPRICING DETECTION INSIGHTS:

1. SAME-DAY BOOKING BEHAVIOR:
   - {same_day_stats['same_day_pct']:.1f}% of bookings are same-day
   - Average same-day price: €{same_day_stats['same_day_avg_price']:.2f}
   - Discount vs advance: {same_day_stats['price_difference_pct']:.1f}%
   → This is RATIONAL for low-occupancy dates (better €65 than €0)

2. UNDERPRICING OPPORTUNITIES IDENTIFIED:
   - {len(underpriced_dates)} dates with high occupancy + high last-minute volume
   - Total potential revenue: €{underpriced_dates['potential_revenue_gain'].sum() if len(underpriced_dates) > 0 else 0:,.2f}
   - These dates had demand but still offered last-minute discounts

3. THE UNDERPRICING SIGNAL:
   When a hotel has BOTH:
   - High occupancy (≥80%) = Strong demand
   - Many last-minute bookings (≥20%) = People still booking at discount
   → Hotel could have charged MORE for those last-minute bookings

4. REVENUE MANAGEMENT IMPLICATION:
   {"✓ Significant underpricing detected!" if len(underpriced_dates) > 10 else "→ Limited underpricing (good yield management)" if len(underpriced_dates) > 0 else "→ No clear underpricing detected"}
   {"  Focus on implementing dynamic pricing for high-demand dates" if len(underpriced_dates) > 10 else "  Monitor high-occupancy dates for optimization" if len(underpriced_dates) > 0 else "  Current pricing strategy appears well-optimized"}

5. BUSINESS STRATEGY:
   - Last-minute discounts are CORRECT when occupancy is low
   - But on high-occupancy dates, maintain/increase prices
   - Use occupancy forecasts to adjust pricing dynamically
   - The €{underpriced_dates['potential_revenue_gain'].sum() if len(underpriced_dates) > 0 else 0:,.2f} represents pure opportunity cost
""")

print("=" * 80)

# %%
print("\n✓ Section 5.2 completed successfully!")

