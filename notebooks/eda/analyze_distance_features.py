"""
Analyze distance features vs price correlation.
Uses pre-calculated distance features from outputs/hotel_distance_features.csv
"""

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib.db import init_db
from lib.data_validator import validate_and_clean

# %%
# Load distance features
print("Loading distance features...")
distance_features = pd.read_csv('outputs/hotel_distance_features.csv')
print(f"Loaded distance features for {len(distance_features):,} hotels")
print(f"\nDistance from Madrid - Min: {distance_features['distance_from_madrid'].min():.2f} km")
print(f"Distance from Madrid - Max: {distance_features['distance_from_madrid'].max():.2f} km")
print(f"Distance from Madrid - Median: {distance_features['distance_from_madrid'].median():.2f} km")
print(f"\nDistance from Coast - Min: {distance_features['distance_from_coast'].min():.2f} km")
print(f"Distance from Coast - Max: {distance_features['distance_from_coast'].max():.2f} km")
print(f"Distance from Coast - Median: {distance_features['distance_from_coast'].median():.2f} km")

# %%
# Load bookings with pricing data
print("\nLoading bookings data...")
con = validate_and_clean(
    init_db(),
    verbose=False,
    rooms_to_exclude=["reception_hall"],
    exclude_missing_location_bookings=True,
)

bookings = con.execute("""
    SELECT 
        b.id,
        b.hotel_id,
        b.total_price,
        b.arrival_date,
        b.departure_date,
        CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
        br.total_price as room_price,
        br.room_type,
        hl.city,
        hl.country
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
""").fetchdf()

print(f"Loaded {len(bookings):,} bookings")

# %%
# Merge distance features with bookings
bookings_with_distances = bookings.merge(
    distance_features,
    on='hotel_id',
    how='left'
)

# Calculate daily price
bookings_with_distances['daily_price'] = bookings_with_distances['room_price'] / bookings_with_distances['nights']

print(f"\nBookings with distance features: {len(bookings_with_distances):,}")
print(f"Missing distance_from_madrid: {bookings_with_distances['distance_from_madrid'].isna().sum()}")
print(f"Missing distance_from_coast: {bookings_with_distances['distance_from_coast'].isna().sum()}")

# Filter to bookings with distance features
bookings_analysis = bookings_with_distances.dropna(subset=['distance_from_madrid', 'distance_from_coast'])
print(f"Bookings for analysis (with distances): {len(bookings_analysis):,}")

# %%
# Correlation analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS: DISTANCE FEATURES VS PRICE")
print("="*80)

corr_madrid = bookings_analysis[['distance_from_madrid', 'daily_price']].corr().iloc[0, 1]
corr_coast = bookings_analysis[['distance_from_coast', 'daily_price']].corr().iloc[0, 1]

print(f"\nCorrelation between distance_from_madrid and daily_price: {corr_madrid:.4f}")
print(f"Correlation between distance_from_coast and daily_price: {corr_coast:.4f}")

# %%
# Visualize relationships
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sample for faster plotting
sample_size = min(10000, len(bookings_analysis))
sample_data = bookings_analysis.sample(sample_size, random_state=42)

# Distance from Madrid vs Price (scatter)
ax1 = axes[0, 0]
ax1.scatter(sample_data['distance_from_madrid'], sample_data['daily_price'], alpha=0.3, s=10)
ax1.set_xlabel('Distance from Madrid (km)')
ax1.set_ylabel('Daily Price (€)')
ax1.set_title(f'Distance from Madrid vs Daily Price\n(Correlation: {corr_madrid:.4f}, n={len(bookings_analysis):,})')
ax1.set_ylim(0, 500)
ax1.grid(True, alpha=0.3)

# Distance from Coast vs Price (scatter)
ax2 = axes[0, 1]
ax2.scatter(sample_data['distance_from_coast'], sample_data['daily_price'], alpha=0.3, s=10)
ax2.set_xlabel('Distance from Coast (km)')
ax2.set_ylabel('Daily Price (€)')
ax2.set_title(f'Distance from Coast vs Daily Price\n(Correlation: {corr_coast:.4f}, n={len(bookings_analysis):,})')
ax2.set_ylim(0, 500)
ax2.grid(True, alpha=0.3)

# Binned analysis: Distance from Madrid
ax3 = axes[1, 0]
madrid_bins = pd.cut(bookings_analysis['distance_from_madrid'], bins=10)
madrid_binned = bookings_analysis.groupby(madrid_bins, observed=True)['daily_price'].agg(['mean', 'median', 'count'])
madrid_binned['bin_center'] = madrid_binned.index.map(lambda x: x.mid)

x_pos = range(len(madrid_binned))
ax3.bar(x_pos, madrid_binned['mean'], alpha=0.7, label='Mean', color='steelblue')
ax3.plot(x_pos, madrid_binned['median'], 'r-o', label='Median', linewidth=2)
ax3.set_xlabel('Distance Bin from Madrid (km)')
ax3.set_ylabel('Daily Price (€)')
ax3.set_title('Average Daily Price by Distance from Madrid')
ax3.legend()
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{int(x.mid)}' for x in madrid_binned.index], rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Binned analysis: Distance from Coast
ax4 = axes[1, 1]
coast_bins = pd.cut(bookings_analysis['distance_from_coast'], bins=10)
coast_binned = bookings_analysis.groupby(coast_bins, observed=True)['daily_price'].agg(['mean', 'median', 'count'])
coast_binned['bin_center'] = coast_binned.index.map(lambda x: x.mid)

x_pos = range(len(coast_binned))
ax4.bar(x_pos, coast_binned['mean'], alpha=0.7, label='Mean', color='steelblue')
ax4.plot(x_pos, coast_binned['median'], 'r-o', label='Median', linewidth=2)
ax4.set_xlabel('Distance Bin from Coast (km)')
ax4.set_ylabel('Daily Price (€)')
ax4.set_title('Average Daily Price by Distance from Coast')
ax4.legend()
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{x.mid:.2f}' for x in coast_binned.index], rotation=45)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
plt.savefig('outputs/figures/distance_features_vs_price.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nSaved visualization to outputs/figures/distance_features_vs_price.png")

# %%
# Statistical summary by distance bins
print("\n" + "="*80)
print("PRICING BY DISTANCE FROM MADRID (BINNED)")
print("="*80)
print(madrid_binned[['mean', 'median', 'count']].to_string())

print("\n" + "="*80)
print("PRICING BY DISTANCE FROM COAST (BINNED)")
print("="*80)
print(coast_binned[['mean', 'median', 'count']].to_string())

# %%
# Summary insights
print("\n" + "="*80)
print("SUMMARY: DISTANCE FEATURES AS PRICING SIGNALS")
print("="*80)

print(f"\n1. Distance from Madrid:")
print(f"   - Correlation with price: {corr_madrid:.4f}")
if abs(corr_madrid) > 0.1:
    direction = "Higher" if corr_madrid > 0 else "Lower"
    print(f"   - {direction} prices for hotels farther from Madrid")
    print(f"   - MODERATE pricing signal (|r| > 0.1)")
elif abs(corr_madrid) > 0.05:
    direction = "Higher" if corr_madrid > 0 else "Lower"
    print(f"   - {direction} prices for hotels farther from Madrid")
    print(f"   - WEAK pricing signal (|r| > 0.05)")
else:
    print(f"   - VERY WEAK pricing signal (|r| < 0.05)")

print(f"\n2. Distance from Coast:")
print(f"   - Correlation with price: {corr_coast:.4f}")
print(f"   - Note: {(bookings_analysis['distance_from_coast'] == 0).sum():,} bookings ({(bookings_analysis['distance_from_coast'] == 0).mean()*100:.1f}%) are at coast (0km)")
if abs(corr_coast) > 0.1:
    direction = "Higher" if corr_coast > 0 else "Lower"
    print(f"   - {direction} prices for hotels farther from coast")
    print(f"   - MODERATE pricing signal (|r| > 0.1)")
elif abs(corr_coast) > 0.05:
    direction = "Higher" if corr_coast > 0 else "Lower"
    print(f"   - {direction} prices for hotels farther from coast")
    print(f"   - WEAK pricing signal (|r| > 0.05)")
else:
    print(f"   - VERY WEAK pricing signal (|r| < 0.05)")

print("\n3. Recommendation for Pricing Model:")
if abs(corr_madrid) > 0.05 or abs(corr_coast) > 0.05:
    print("   ✓ INCLUDE distance features in pricing model")
    print("   ✓ Consider non-linear transformations (binned analysis shows variation)")
    if abs(corr_madrid) > abs(corr_coast):
        print("   ✓ Distance from Madrid appears stronger than distance from coast")
    else:
        print("   ✓ Distance from coast appears stronger than distance from Madrid")
else:
    print("   ✗ Distance features are WEAK pricing signals")
    print("   - May still be useful in interaction with other features (city, room_type, etc.)")
    print("   - Consider location-specific analysis (Madrid vs coastal cities)")

print("\n4. Data Insights:")
print(f"   - Most hotels are coastal: {(bookings_analysis['distance_from_coast'] < 1).mean()*100:.1f}% within 1km of coast")
print(f"   - Distance from Madrid range: {bookings_analysis['distance_from_madrid'].min():.0f}-{bookings_analysis['distance_from_madrid'].max():.0f} km")
print(f"   - High concentration near coast may limit signal strength for coast distance")

print("="*80)

# %%

