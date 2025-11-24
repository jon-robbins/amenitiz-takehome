# %%
"""
Section 7.2: RevPAR and Relationship with Occupancy

Question: What is RevPAR (Revenue per Available Room) per hotel and date,
and how does it relate to occupancy levels?

RevPAR = Total Revenue / Total Available Rooms
It's a key hotel performance metric that combines occupancy and pricing.

Approach:
- Calculate RevPAR per hotel per date
- Bin by occupancy ranges (<50%, 50-70%, 70-85%, 85-95%, ≥95%)
- Analyze RevPAR patterns and optimization opportunities
"""

# %%
import sys
sys.path.insert(0, '../../..')
from lib.db import init_db
from lib.data_validator import validate_and_clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
# Initialize database connection
print("Initializing database...")
con = validate_and_clean(
    init_db(),
    verbose=False,
    rooms_to_exclude=['reception_hall'],
    exclude_missing_location_bookings=True,
    
)

# %%
print("=" * 80)
print("SECTION 7.2: RevPAR ANALYSIS")
print("=" * 80)

# %%
# Get hotel capacities (reuse from 7.1)
print("\nCalculating hotel capacities...")
hotel_capacity = con.execute("""
    SELECT 
        b.hotel_id,
        SUM(r.number_of_rooms) as total_capacity
    FROM (
        SELECT DISTINCT hotel_id, room_id
        FROM booked_rooms br
        JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
    ) b
    JOIN rooms r ON b.room_id = r.id
    GROUP BY b.hotel_id
""").fetchdf()

print(f"Analyzed {len(hotel_capacity)} hotels")

# %%
# Calculate revenue and occupancy per hotel-date
print("\nCalculating RevPAR by hotel and date...")
revpar_data = con.execute("""
    WITH RECURSIVE date_series AS (
        SELECT MIN(CAST(arrival_date AS DATE)) as stay_date,
               MAX(CAST(departure_date AS DATE)) as max_date
        FROM bookings
        WHERE status IN ('confirmed', 'Booked')
        
        UNION ALL
        
        SELECT stay_date + INTERVAL 1 DAY, max_date
        FROM date_series
        WHERE stay_date < max_date
    ),
    daily_revenue AS (
        SELECT 
            b.hotel_id,
            ds.stay_date,
            COUNT(*) as occupied_rooms,
            SUM(br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) as total_revenue,
            AVG(br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) as avg_daily_price
        FROM date_series ds
        CROSS JOIN bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        WHERE ds.stay_date >= CAST(b.arrival_date AS DATE)
          AND ds.stay_date < CAST(b.departure_date AS DATE)
          AND b.status IN ('confirmed', 'Booked')
        GROUP BY b.hotel_id, ds.stay_date
    )
    SELECT 
        hotel_id,
        stay_date,
        occupied_rooms,
        total_revenue,
        avg_daily_price
    FROM daily_revenue
""").fetchdf()

print(f"Calculated revenue for {len(revpar_data):,} hotel-date combinations")

# %%
# Merge with capacity and calculate RevPAR
revpar_analysis = revpar_data.merge(hotel_capacity, on='hotel_id', how='left')
revpar_analysis['occupancy_rate'] = (revpar_analysis['occupied_rooms'] / revpar_analysis['total_capacity']) * 100
revpar_analysis['occupancy_rate'] = revpar_analysis['occupancy_rate'].clip(upper=100)
revpar_analysis['revpar'] = revpar_analysis['total_revenue'] / revpar_analysis['total_capacity']

print(f"\nRevPAR calculated for {len(revpar_analysis):,} hotel-nights")

# %%
# Create occupancy bins
bins = [0, 50, 70, 85, 95, 100]
labels = ['<50%', '50-70%', '70-85%', '85-95%', '≥95%']
revpar_analysis['occupancy_bin'] = pd.cut(
    revpar_analysis['occupancy_rate'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# %%
# Summary statistics
print("\n" + "=" * 80)
print("RevPAR STATISTICS")
print("=" * 80)

print(f"\n1. OVERALL RevPAR DISTRIBUTION:")
print(f"   Mean RevPAR: €{revpar_analysis['revpar'].mean():.2f}")
print(f"   Median RevPAR: €{revpar_analysis['revpar'].median():.2f}")
print(f"   Std dev: €{revpar_analysis['revpar'].std():.2f}")
print(f"   25th percentile: €{revpar_analysis['revpar'].quantile(0.25):.2f}")
print(f"   75th percentile: €{revpar_analysis['revpar'].quantile(0.75):.2f}")

# RevPAR by occupancy bin
print(f"\n2. RevPAR BY OCCUPANCY LEVEL:")
occupancy_revpar = revpar_analysis.groupby('occupancy_bin', observed=True).agg(
    count=('revpar', 'count'),
    mean_revpar=('revpar', 'mean'),
    median_revpar=('revpar', 'median'),
    mean_occupancy=('occupancy_rate', 'mean'),
    mean_price=('avg_daily_price', 'mean')
).round(2)

print(occupancy_revpar.to_string())

# Calculate RevPAR efficiency
print(f"\n3. RevPAR EFFICIENCY ANALYSIS:")
for idx, row in occupancy_revpar.iterrows():
    pct_of_total = (row['count'] / len(revpar_analysis)) * 100
    print(f"   {idx}: €{row['mean_revpar']:.2f} RevPAR, "
          f"{row['mean_occupancy']:.1f}% occupancy, "
          f"€{row['mean_price']:.2f} ADR ({pct_of_total:.1f}% of nights)")

# %%
# Per-hotel RevPAR performance
print(f"\n4. HOTEL-LEVEL RevPAR PERFORMANCE:")
hotel_revpar = revpar_analysis.groupby('hotel_id').agg(
    avg_revpar=('revpar', 'mean'),
    avg_occupancy=('occupancy_rate', 'mean'),
    avg_price=('avg_daily_price', 'mean'),
    capacity=('total_capacity', 'first'),
    num_nights=('stay_date', 'count')
).reset_index()

print(f"\n   Top 10 hotels by average RevPAR:")
top_revpar_hotels = hotel_revpar.nlargest(10, 'avg_revpar')
for idx, row in top_revpar_hotels.iterrows():
    print(f"   Hotel {row['hotel_id']}: €{row['avg_revpar']:.2f} RevPAR, "
          f"{row['avg_occupancy']:.1f}% occupancy, "
          f"€{row['avg_price']:.2f} ADR, "
          f"{int(row['capacity'])} rooms")

# %%
# RevPAR optimization opportunity
print(f"\n5. RevPAR OPTIMIZATION OPPORTUNITIES:")

# Compare high occupancy RevPAR to theoretical maximum
high_occ = revpar_analysis[revpar_analysis['occupancy_rate'] >= 90]
if len(high_occ) > 0:
    actual_revpar_high = high_occ['revpar'].mean()
    avg_price_high = high_occ['avg_daily_price'].mean()
    avg_capacity = high_occ['total_capacity'].mean()
    
    # Theoretical max RevPAR at 100% occupancy with current prices
    theoretical_max = avg_price_high * 1.0  # 100% occupancy
    
    print(f"   High occupancy (≥90%):")
    print(f"   - Actual RevPAR: €{actual_revpar_high:.2f}")
    print(f"   - Theoretical max (100% at current prices): €{theoretical_max:.2f}")
    print(f"   - Gap: €{theoretical_max - actual_revpar_high:.2f}")

# %%
# Create visualization
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. RevPAR distribution
ax1 = axes[0, 0]
ax1.hist(revpar_analysis['revpar'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(revpar_analysis['revpar'].mean(), color='red', 
            linestyle='--', linewidth=2, label=f"Mean: €{revpar_analysis['revpar'].mean():.2f}")
ax1.set_xlabel('RevPAR (€)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of RevPAR')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(0, revpar_analysis['revpar'].quantile(0.95))

# 2. RevPAR by occupancy bin
ax2 = axes[0, 1]
occupancy_revpar['mean_revpar'].plot(kind='bar', ax=ax2, color='coral', alpha=0.7)
ax2.set_xlabel('Occupancy Range')
ax2.set_ylabel('Average RevPAR (€)')
ax2.set_title('RevPAR by Occupancy Level')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(occupancy_revpar['mean_revpar']):
    ax2.text(i, v + 2, f'€{v:.0f}', ha='center', va='bottom', fontsize=9)

# 3. RevPAR vs Occupancy scatter
ax3 = axes[0, 2]
sample_size = min(10000, len(revpar_analysis))
sample = revpar_analysis.sample(n=sample_size, random_state=42)
scatter = ax3.scatter(sample['occupancy_rate'], sample['revpar'], 
                     alpha=0.3, s=10, c=sample['avg_daily_price'], cmap='viridis')
ax3.set_xlabel('Occupancy Rate (%)')
ax3.set_ylabel('RevPAR (€)')
ax3.set_title(f'RevPAR vs Occupancy (sample of {sample_size:,})')
ax3.set_ylim(0, revpar_analysis['revpar'].quantile(0.95))
plt.colorbar(scatter, ax=ax3, label='Avg Daily Rate (€)')
ax3.grid(True, alpha=0.3)

# Calculate correlation
corr = revpar_analysis[['occupancy_rate', 'revpar']].corr().iloc[0, 1]
ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}',
        transform=ax3.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. RevPAR over time
ax4 = axes[1, 0]
daily_revpar = revpar_analysis.groupby('stay_date')['revpar'].mean().reset_index()
ax4.plot(daily_revpar['stay_date'], daily_revpar['revpar'], linewidth=0.8, alpha=0.7)
ax4.set_xlabel('Date')
ax4.set_ylabel('Average RevPAR (€)')
ax4.set_title('Average RevPAR Over Time')
ax4.grid(True, alpha=0.3)

# 5. Hotel RevPAR distribution
ax5 = axes[1, 1]
ax5.hist(hotel_revpar['avg_revpar'], bins=30, color='green', alpha=0.7, edgecolor='black')
ax5.axvline(hotel_revpar['avg_revpar'].mean(), color='red', 
            linestyle='--', linewidth=2, label=f"Mean: €{hotel_revpar['avg_revpar'].mean():.2f}")
ax5.set_xlabel('Average RevPAR (€)')
ax5.set_ylabel('Number of Hotels')
ax5.set_title('Distribution of Hotel Average RevPAR')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. RevPAR components (occupancy vs price contribution)
ax6 = axes[1, 2]
# Plot showing how occupancy and price contribute to RevPAR
occupancy_bins_pct = occupancy_revpar['mean_occupancy'] / 100
price_contribution = occupancy_revpar['mean_price']

x = np.arange(len(occupancy_revpar))
width = 0.35

ax6.bar(x, occupancy_bins_pct * 100, width, label='Occupancy %', alpha=0.7, color='steelblue')
ax6_twin = ax6.twinx()
ax6_twin.bar(x + width, price_contribution, width, label='Avg Daily Rate (€)', alpha=0.7, color='coral')

ax6.set_xlabel('Occupancy Range')
ax6.set_ylabel('Occupancy Rate (%)', color='steelblue')
ax6_twin.set_ylabel('Average Daily Rate (€)', color='coral')
ax6.set_title('RevPAR Components by Occupancy Level')
ax6.set_xticks(x + width/2)
ax6.set_xticklabels(occupancy_revpar.index, rotation=45, ha='right')
ax6.tick_params(axis='y', labelcolor='steelblue')
ax6_twin.tick_params(axis='y', labelcolor='coral')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "section_7_2_revpar.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved visualization to {output_path}")

# %%
print("\n" + "=" * 80)
print("SECTION 7.2: KEY FINDINGS SUMMARY")
print("=" * 80)

mean_revpar = revpar_analysis['revpar'].mean()
median_revpar = revpar_analysis['revpar'].median()

# RevPAR by bin
revpar_low = occupancy_revpar.loc['<50%', 'mean_revpar'] if '<50%' in occupancy_revpar.index else 0
revpar_high = occupancy_revpar.loc['≥95%', 'mean_revpar'] if '≥95%' in occupancy_revpar.index else 0
revpar_range = revpar_high - revpar_low
revpar_multiplier = revpar_high / revpar_low if revpar_low > 0 else 0

print(f"""
RevPAR ANALYSIS INSIGHTS:

1. OVERALL RevPAR PERFORMANCE:
   - Mean RevPAR: €{mean_revpar:.2f}
   - Median RevPAR: €{median_revpar:.2f}
   → RevPAR = Occupancy × ADR, so it captures total performance

2. RevPAR BY OCCUPANCY LEVEL:
   - <50% occupancy: €{revpar_low:.2f} RevPAR
   - ≥95% occupancy: €{revpar_high:.2f} RevPAR
   - Range: €{revpar_range:.2f} ({revpar_multiplier:.1f}x multiplier)
   → Clear progression: higher occupancy = higher RevPAR

3. RevPAR-OCCUPANCY CORRELATION:
   - Correlation: {corr:.3f}
   → {"Strong" if corr > 0.8 else "Moderate" if corr > 0.5 else "Weak"} relationship
   → RevPAR increases with occupancy (as expected)

4. OPTIMIZATION OPPORTUNITY:
   - RevPAR peaks at highest occupancy levels
   - But Section 5.2 showed hotels discount at high occupancy!
   - This suppresses RevPAR growth potential

5. CONNECTION TO PREVIOUS SECTIONS:
   - Section 5.2: €2.25M underpricing on high-occupancy dates
   - Section 7.1: 16.6% of nights at ≥95% occupancy
   - Section 7.2: These high-occupancy nights have highest RevPAR potential
   → Fixing underpricing would boost RevPAR significantly

BUSINESS IMPACT:
- RevPAR is THE key hotel performance metric
- Current mean: €{mean_revpar:.2f} across all hotel-nights
- High-occupancy nights (≥95%) achieve €{revpar_high:.2f} RevPAR
- Eliminating last-minute discounts on these nights would push RevPAR higher
- The €2.25M opportunity = direct RevPAR improvement

STRATEGIC RECOMMENDATIONS:
1. Track RevPAR as primary KPI (not just occupancy or ADR alone)
2. Implement occupancy-based pricing to maximize RevPAR
3. Focus on nights currently at <85% occupancy (room for growth)
4. Protect high-occupancy nights from discounting
5. Use RevPAR benchmarking across properties for performance management
""")

print("=" * 80)

# %%
print("\n✓ Section 7.2 completed successfully!")

# %%
"""
## Section 7.2: RevPAR Analysis - Critical Reflection

### What This Section Actually Tells Us:

**RevPAR = Occupancy × ADR by definition**, so finding that RevPAR increases with occupancy 
is somewhat tautological. We're essentially confirming: "when hotels are fuller and charge 
reasonable prices, they make more revenue per room." Not exactly groundbreaking! 😄

### The REAL Insights Came From Earlier Sections:

**1. Section 5.2 (Occupancy-based underpricing) - THE KEY FINDING:**
- **€2.25M opportunity** from high-occupancy dates with last-minute discounts
- This IS surprising: hotels discount when they're FULL
- Counter-intuitive behavior that's costing real money

**2. Section 7.1 (Occupancy validation):**
- **Weak correlation (0.143)** between occupancy and price
- Hotels DON'T dynamically price based on demand
- They could but don't - this is the actual problem

**3. Section 6.1 (Room features):**
- **Children-allowed premium (+€39)** bigger than most room features
- Room type matters 3x more than size
- Actionable pricing insights independent of occupancy

**4. Section 4.3 (Prophet forecasting):**
- **Linear regression completely wrong** (R²=0.03 vs Prophet R²=0.71)
- Reveals +20% growth hidden by seasonality
- Shows importance of proper time series modeling

**5. Section 5.1 (Lead time):**
- **39% last-minute bookings at -35% discount**
- Inverted pricing vs airlines (who charge premium for last-minute)
- Suggests inventory clearing vs surge pricing strategy

### What Section 7.2 Actually Added (Honestly, Not Much):

- ✗ RevPAR range (5.4x) just restates occupancy range from 7.1
- ✗ Correlation (0.567) is occupancy by mathematical definition
- ✓ Validates that RevPAR formula works as expected (sanity check)
- ✓ Shows €168 RevPAR at high occupancy could be €200 (connects to 5.2)

### The Real Insight:

**RevPAR is a REPORTING metric, not a DISCOVERY metric.**

It's useful for:
- Benchmarking hotel performance
- Tracking overall business health
- Communicating to stakeholders

It's NOT useful for:
- Finding pricing opportunities (use Section 5.2's occupancy-based analysis)
- Understanding demand patterns (use Section 4.3's Prophet model)
- Feature-based pricing (use Section 6.1's room analysis)

### Conclusion:

Section 7.2 confirms our math works correctly but doesn't add novel insights beyond 
what we learned from:
- **5.2**: €2.25M underpricing opportunity (THE actionable finding)
- **7.1**: Weak dynamic pricing (correlation 0.143) despite frequent high occupancy
- **4.3**: Prophet forecasting for demand prediction

**Bottom line**: RevPAR is a good KPI for dashboards, but the actionable insights 
came from analyzing its components (occupancy and pricing) separately, not together.
"""

