# %%
"""
Section 1.2: Hotel Supply Structure

Question: What does a "room" represent, and how is hotel inventory structured?

Key Concepts:
- room_id = A room configuration (unique combo of hotel + type + size + view + occupancy)
- room_type = Category (room/apartment/villa/cottage/reception_hall)
- number_of_rooms = Count of identical physical units for that configuration
"""

# %%
import sys
sys.path.insert(0, '../../../..')
from lib.db import init_db
from lib.sql_loader import load_sql_file
from lib.data_validator import CleaningConfig, DataCleaner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# Load SQL queries from files
query_hotel_config = load_sql_file('QUERY_HOTEL_CONFIG_DISTRIBUTION.sql', __file__)
query_room_type = load_sql_file('QUERY_ROOM_TYPE_SUMMARY.sql', __file__)
query_bookings_per_room = load_sql_file('QUERY_BOOKINGS_PER_ROOM.sql', __file__)

# Execute queries
# Get data - HOTEL-LEVEL ANALYSIS
# Distribution of configurations per hotel
hotel_config_distribution = con.execute(query_hotel_config).fetchdf()

# Summary stats by room type (for category understanding)
room_type_summary = con.execute(query_room_type).fetchdf()

# Bookings per room (for utilization analysis)
bookings_per_room = con.execute(query_bookings_per_room).fetchdf()

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Distribution of configurations per hotel
ax1 = axes[0, 0]
config_bins = [0, 1, 2, 5, 10, 20, 100]
config_labels = ['1', '2', '3-5', '6-10', '11-20', '20+']
hotel_config_distribution['config_bin'] = pd.cut(
    hotel_config_distribution['num_configurations'], 
    bins=config_bins, 
    labels=config_labels,
    include_lowest=True
)
config_counts = hotel_config_distribution['config_bin'].value_counts().sort_index()
bars = ax1.bar(range(len(config_counts)), config_counts.values, color='steelblue', edgecolor='black', alpha=0.8)
ax1.set_xticks(range(len(config_counts)))
ax1.set_xticklabels(config_counts.index, fontsize=11)
ax1.set_xlabel('Number of Configurations per Hotel', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Hotels', fontsize=12, fontweight='bold')
ax1.set_title('Hotel Inventory Diversity: Configurations per Hotel', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
# Add percentage labels
total_hotels = len(hotel_config_distribution)
for bar, count in zip(bars, config_counts.values):
    pct = count / total_hotels * 100
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Distribution of total units per hotel
ax2 = axes[0, 1]
unit_bins = [0, 1, 5, 10, 20, 50, 1000]
unit_labels = ['1', '2-5', '6-10', '11-20', '21-50', '50+']
hotel_config_distribution['unit_bin'] = pd.cut(
    hotel_config_distribution['total_units'], 
    bins=unit_bins, 
    labels=unit_labels,
    include_lowest=True
)
unit_counts = hotel_config_distribution['unit_bin'].value_counts().sort_index()
bars2 = ax2.bar(range(len(unit_counts)), unit_counts.values, color='coral', edgecolor='black', alpha=0.8)
ax2.set_xticks(range(len(unit_counts)))
ax2.set_xticklabels(unit_counts.index, fontsize=11)
ax2.set_xlabel('Total Units per Hotel', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Hotels', fontsize=12, fontweight='bold')
ax2.set_title('Hotel Size: Total Units per Hotel', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
# Add percentage labels
for bar, count in zip(bars2, unit_counts.values):
    pct = count / total_hotels * 100
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Utilization by Room Type Category (Boxplot)
ax3 = axes[1, 0]
sns.boxplot(data=bookings_per_room, x='room_type', y='bookings_per_individual_room', 
            ax=ax3, palette='Set2', hue='room_type', legend=False)
ax3.set_xlabel('Room Type Category', fontsize=12, fontweight='bold')
ax3.set_ylabel('Bookings per Individual Unit', fontsize=12, fontweight='bold')
ax3.set_title('Utilization: Bookings per Unit by Category', fontsize=14, fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

# 4. Category distribution across hotels
ax4 = axes[1, 1]
category_counts = hotel_config_distribution['num_categories'].value_counts().sort_index()
bars4 = ax4.bar(category_counts.index, category_counts.values, color='seagreen', edgecolor='black', alpha=0.8)
ax4.set_xlabel('Number of Room Categories per Hotel', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Hotels', fontsize=12, fontweight='bold')
ax4.set_title('Category Diversity: How Many Categories per Hotel?', fontsize=14, fontweight='bold')
ax4.set_xticks(category_counts.index)
ax4.grid(axis='y', alpha=0.3)
# Add percentage labels
for bar, cat, count in zip(bars4, category_counts.index, category_counts.values):
    pct = count / total_hotels * 100
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Print comprehensive insights
print("\n" + "="*80)
print("KEY INSIGHTS: Hotel-Level Supply Structure")
print("="*80)

print(f"\n--- DATASET OVERVIEW ---")
print(f"Total hotels: {total_hotels:,}")
print(f"Total unique room configurations: {len(bookings_per_room):,}")
print(f"Total individual units: {hotel_config_distribution['total_units'].sum():,.0f}")

print("\n--- HOTEL INVENTORY DIVERSITY ---")
config_dist_summary = hotel_config_distribution['num_configurations'].describe()
print(f"Configurations per hotel:")
print(f"  - Mean: {config_dist_summary['mean']:.1f}")
print(f"  - Median: {config_dist_summary['50%']:.0f}")
print(f"  - 25th percentile: {config_dist_summary['25%']:.0f}")
print(f"  - 75th percentile: {config_dist_summary['75%']:.0f}")

single_config = (hotel_config_distribution['num_configurations'] == 1).sum()
small_config = ((hotel_config_distribution['num_configurations'] >= 2) & 
                (hotel_config_distribution['num_configurations'] <= 5)).sum()
medium_config = ((hotel_config_distribution['num_configurations'] >= 6) & 
                 (hotel_config_distribution['num_configurations'] <= 10)).sum()
large_config = (hotel_config_distribution['num_configurations'] > 10).sum()

print(f"\n  - {single_config} hotels ({single_config/total_hotels*100:.1f}%) have 1 configuration")
print(f"  - {small_config} hotels ({small_config/total_hotels*100:.1f}%) have 2-5 configurations")
print(f"  - {medium_config} hotels ({medium_config/total_hotels*100:.1f}%) have 6-10 configurations")
print(f"  - {large_config} hotels ({large_config/total_hotels*100:.1f}%) have 10+ configurations")

print("\n--- HOTEL SIZE (Total Units) ---")
unit_dist_summary = hotel_config_distribution['total_units'].describe()
print(f"Total units per hotel:")
print(f"  - Mean: {unit_dist_summary['mean']:.1f}")
print(f"  - Median: {unit_dist_summary['50%']:.0f}")
print(f"  - 25th percentile: {unit_dist_summary['25%']:.0f}")
print(f"  - 75th percentile: {unit_dist_summary['75%']:.0f}")

print("\n--- CATEGORY DIVERSITY ---")
cat_dist = hotel_config_distribution['num_categories'].value_counts().sort_index()
print("Number of categories per hotel:")
for cat, count in cat_dist.items():
    pct = count / total_hotels * 100
    print(f"  - {int(cat)} category: {count} hotels ({pct:.1f}%)")

print("\n--- ROOM TYPE CATEGORY SUMMARY ---")
print(room_type_summary[['room_type', 'num_configurations', 'avg_units_per_config', 'median_units_per_config']].to_string(index=False))

print("\n--- UTILIZATION BY CATEGORY ---")
util_by_type = bookings_per_room.groupby('room_type')['bookings_per_individual_room'].median().sort_values(ascending=False)
print("Median bookings per unit:")
for cat, util in util_by_type.items():
    print(f"  - {cat}: {util:.1f} bookings/unit")

print("\n" + "="*80)
print("SUMMARY FOR PRICING MODEL")
print("="*80)
print("1. Market is diverse: 25% single-config (apartments/villas), 75% multi-config hotels")
print("2. Most hotels (50%) have 2-5 room configurations - small properties")
print("3. Category specialization: 74% of hotels offer only 1 category")
print("4. Utilization varies by category: rooms (53) > apartments (31) > villas (21)")
print("5. For pricing: Model at HOTEL level + room attributes (category, size, occupancy)")
print("="*80)

# %%
"""
## Section 1.2: Key Takeaways & Business Insights

### Data Quality Impact
After applying full data cleaning (all validation rules enabled):
- Removed invalid bookings (negative prices, null dates, cancelled records)
- Excluded reception_hall (not accommodation)
- Excluded hotels with missing location data
- Clean dataset ensures accurate hotel capacity calculations

### Market Structure Findings

**1. HOTEL INVENTORY COMPLEXITY**
- Most hotels (75%) operate with MULTIPLE room configurations
- Median hotel has 2-5 different room types/sizes/views
- This complexity requires configuration-level pricing (not just hotel-level)

**2. PROPERTY SIZE DISTRIBUTION**
- SMALL properties dominate: Median ~5 units per hotel
- This is a BOUTIQUE hotel market, not chain hotels
- Smaller properties = higher occupancy volatility (one booking = 20% occupancy jump)

**3. CATEGORY SPECIALIZATION**
- 74% of hotels offer ONLY ONE category (room, apartment, villa, or cottage)
- Single-category hotels = simpler operations, clearer positioning
- Multi-category hotels = more complex inventory management

**4. UTILIZATION PATTERNS**
- Rooms: Highest utilization (53 bookings/unit median)
- Apartments: Medium utilization (31 bookings/unit)
- Villas/Cottages: Lower utilization (21 bookings/unit)
- Suggests rooms are easier to sell (shorter stays, business travel)

### Implications for Revenue Management

**1. PRICING STRATEGY**
- Cannot use simple "hotel average" price
- Need CONFIGURATION-LEVEL pricing: 
  - Base price by room_type (category)
  - Adjustments for size, view, occupancy
  - Hotel-level multipliers for location, brand

**2. INVENTORY MANAGEMENT**
- Small hotel size = high sensitivity to single bookings
- At 5-unit hotel: Each booking = 20% occupancy change
- Dynamic pricing is CRITICAL (unlike large chains with 100+ rooms)

**3. SEGMENTATION FOR MODELING**
- Segment 1: Single-config properties (25%) - simpler pricing
- Segment 2: Multi-config properties (75%) - complex optimization
- Different strategies needed for each segment

**4. CAPACITY CONSTRAINTS**
- Small properties hit capacity constraints frequently
- Section 7.1 showed 16.6% of nights at ≥95% occupancy
- This validates the €2.25M underpricing opportunity from Section 5.2

### Actionable Recommendations

1. **Immediate:** Implement occupancy-based pricing with hotel size adjustment
   - Small hotels (<5 rooms): More aggressive surge pricing (±30%)
   - Medium hotels (5-20 rooms): Moderate surge pricing (±20%)
   - Large hotels (20+ rooms): Conservative surge pricing (±15%)

2. **Short-term:** Build configuration-level price optimization
   - Each room_id gets own price model
   - Trained on historical bookings for that configuration
   - Hotel-level constraints ensure internal consistency

3. **Long-term:** Portfolio optimization across configurations
   - For multi-config hotels: Optimize WHICH room to sell at what price
   - Cannibalization risk: Don't undercut premium rooms with cheap rooms
   - Revenue management: Save premium rooms for high-demand dates

### Connection to Other Sections

- **Section 5.2 (Underpricing):** Small hotel size explains why occupancy-based pricing is so critical
- **Section 7.1 (Occupancy):** 778 hotels frequently at 90%+ occupancy = capacity-constrained
- **Section 7.2 (RevPAR):** Small properties can achieve HIGH RevPAR with proper pricing

**Bottom Line:** This is a boutique hotel market with complex inventory. Success requires 
sophisticated, configuration-level dynamic pricing, NOT simple hotel-wide rates.
"""

