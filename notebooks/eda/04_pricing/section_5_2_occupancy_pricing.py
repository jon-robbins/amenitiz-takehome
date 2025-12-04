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
sys.path.insert(0, '../../../..')
from lib.db import init_db
from lib.sql_loader import load_sql_file
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

# Initialize database
con = init_db()

# Clean data
config = CleaningConfig(
    exclude_reception_halls=True,
    exclude_missing_location=True,
    verbose=True
)
cleaner = DataCleaner(config)
con = cleaner.clean(con)

# %%
print("=" * 80)
print("SECTION 5.2: OCCUPANCY-BASED PRICING ANALYSIS")
print("=" * 80)

# %%
# Load SQL query from file
query = load_sql_file('QUERY_LOAD_BOOKINGS_WITH_LEAD_TIME_OCCUPANCY.sql', __file__)

# Execute query
print("\nLoading booking data...")
bookings_lead_time = con.execute(query).fetchdf()

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
output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "eda" / "pricing" / "figures"
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

