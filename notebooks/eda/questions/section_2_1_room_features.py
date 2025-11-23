# 2.1 Room type, size, and view distributions
# Analyze supply-side features: room_type, room_size, room_view

import sys
sys.path.insert(0, '../../..')
from lib.db import init_db
from lib.data_validator import validate_and_clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con_raw = init_db()
con = validate_and_clean(con_raw)

print("="*80)
print("SECTION 2.1: Room Features Analysis (Size, View)")
print("="*80)

# Get room features with pricing
room_features = con.execute("""
    SELECT 
        br.room_id,
        br.room_type,
        br.room_size,
        br.room_view,
        br.total_price,
        b.arrival_date,
        b.departure_date,
        DATE_DIFF('day', b.arrival_date, b.departure_date) as stay_length,
        br.total_price / NULLIF(DATE_DIFF('day', b.arrival_date, b.departure_date), 0) as daily_price,
        r.number_of_rooms,
        r.max_occupancy,
        b.hotel_id
    FROM booked_rooms br
    JOIN bookings b ON b.id = br.booking_id
    JOIN rooms r ON r.id = br.room_id
    WHERE b.arrival_date IS NOT NULL 
      AND b.departure_date IS NOT NULL
      AND DATE_DIFF('day', b.arrival_date, b.departure_date) > 0
      AND br.total_price > 0
      AND br.room_type IS NOT NULL
""").fetchdf()

print(f"\nTotal bookings with feature data: {len(room_features):,}")

# 1. ROOM SIZE ANALYSIS
print("\n" + "="*80)
print("1. ROOM SIZE ANALYSIS")
print("="*80)

# Check data quality
size_available = room_features[room_features['room_size'] > 0]
print(f"Bookings with room_size data: {len(size_available):,} ({len(size_available)/len(room_features)*100:.1f}%)")

if len(size_available) > 0:
    print("\nRoom size statistics:")
    print(size_available['room_size'].describe())
    
    # Size by category
    size_by_category = size_available.groupby('room_type')['room_size'].agg([
        'count', 'mean', 'median', 'std',
        ('p25', lambda x: x.quantile(0.25)),
        ('p75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    print("\nRoom size by category:")
    print(size_by_category)
    
    # Price per sqm by category
    size_available['price_per_sqm'] = size_available['daily_price'] / size_available['room_size']
    price_per_sqm = size_available.groupby('room_type')['price_per_sqm'].agg([
        'median', 'mean'
    ]).round(2)
    
    print("\nPrice per sqm by category:")
    print(price_per_sqm)

# 2. ROOM VIEW ANALYSIS
print("\n" + "="*80)
print("2. ROOM VIEW ANALYSIS")
print("="*80)

# Check data quality
view_available = room_features[room_features['room_view'].notna() & (room_features['room_view'] != '')]
print(f"Bookings with room_view data: {len(view_available):,} ({len(view_available)/len(room_features)*100:.1f}%)")

if len(view_available) > 0:
    # View distribution
    view_counts = view_available['room_view'].value_counts()
    print(f"\nTop 10 room views:")
    print(view_counts.head(10))
    
    # Price by view
    price_by_view = view_available.groupby('room_view')['daily_price'].agg([
        'count', 'median', 'mean'
    ]).sort_values('median', ascending=False).head(15)
    
    print("\nTop 15 views by median price:")
    print(price_by_view.round(2))

# 3. COMBINED ANALYSIS: Category + Size + View
print("\n" + "="*80)
print("3. FEATURE INTERACTIONS")
print("="*80)

# Size bins
if len(size_available) > 0:
    size_available['size_bin'] = pd.cut(
        size_available['room_size'],
        bins=[0, 20, 30, 40, 60, 100, 500],
        labels=['<20 sqm', '20-30 sqm', '30-40 sqm', '40-60 sqm', '60-100 sqm', '100+ sqm']
    )
    
    price_by_size_category = size_available.groupby(['room_type', 'size_bin'])['daily_price'].agg([
        'count', 'median'
    ]).reset_index()
    
    print("\nMedian price by category and size:")
    pivot_price = price_by_size_category.pivot(index='size_bin', columns='room_type', values='median')
    print(pivot_price.round(2))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Room size distribution by category
ax1 = axes[0, 0]
if len(size_available) > 0:
    size_filtered = size_available[size_available['room_size'] <= 150]
    sns.boxplot(data=size_filtered, x='room_type', y='room_size', ax=ax1, palette='Set2', hue='room_type', legend=False)
    ax1.set_xlabel('Room Type Category', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Room Size (sqm)', fontsize=11, fontweight='bold')
    ax1.set_title('Room Size Distribution by Category', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

# 2. Price vs Size scatter
ax2 = axes[0, 1]
if len(size_available) > 0:
    size_subset = size_available[(size_available['room_size'] > 0) & 
                                  (size_available['room_size'] <= 150) &
                                  (size_available['daily_price'] <= 500)]
    for cat in size_subset['room_type'].unique():
        cat_data = size_subset[size_subset['room_type'] == cat]
        ax2.scatter(cat_data['room_size'], cat_data['daily_price'], 
                   alpha=0.3, s=10, label=cat)
    ax2.set_xlabel('Room Size (sqm)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Daily Price (€)', fontsize=11, fontweight='bold')
    ax2.set_title('Price vs Size by Category', fontsize=12, fontweight='bold')
    ax2.legend(title='Category', fontsize=8)
    ax2.grid(alpha=0.3)

# 3. Price per sqm by category
ax3 = axes[0, 2]
if len(size_available) > 0:
    price_per_sqm_filtered = size_available[size_available['price_per_sqm'] <= 20]
    sns.boxplot(data=price_per_sqm_filtered, x='room_type', y='price_per_sqm', 
                ax=ax3, palette='Set2', hue='room_type', legend=False)
    ax3.set_xlabel('Room Type Category', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Price per sqm (€/sqm)', fontsize=11, fontweight='bold')
    ax3.set_title('Price Efficiency by Category', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

# 4. View distribution (top 10)
ax4 = axes[1, 0]
if len(view_available) > 0:
    top_views = view_counts.head(10)
    bars = ax4.barh(range(len(top_views)), top_views.values, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.set_yticks(range(len(top_views)))
    ax4.set_yticklabels(top_views.index, fontsize=9)
    ax4.set_xlabel('Number of Bookings', fontsize=11, fontweight='bold')
    ax4.set_title('Top 10 Room Views by Frequency', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)

# 5. Price by view (top 10 by price)
ax5 = axes[1, 1]
if len(view_available) > 0:
    top_price_views = price_by_view.head(10).sort_values('median')
    bars5 = ax5.barh(range(len(top_price_views)), top_price_views['median'], 
                     color='coral', edgecolor='black', alpha=0.7)
    ax5.set_yticks(range(len(top_price_views)))
    ax5.set_yticklabels(top_price_views.index, fontsize=9)
    ax5.set_xlabel('Median Daily Price (€)', fontsize=11, fontweight='bold')
    ax5.set_title('Top 10 Most Expensive Views', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)

# 6. Heatmap: Price by category and size
ax6 = axes[1, 2]
if len(size_available) > 0 and len(pivot_price) > 0:
    sns.heatmap(pivot_price, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Daily Price (€)'})
    ax6.set_xlabel('Room Type Category', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Room Size Range', fontsize=11, fontweight='bold')
    ax6.set_title('Median Price Heatmap: Category × Size', fontsize=12, fontweight='bold')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS: Room Features")
print("="*80)

print("\n--- ROOM SIZE ---")
if len(size_available) > 0:
    print(f"Data availability: {len(size_available)/len(room_features)*100:.1f}% of bookings have size data")
    print(f"Median room size: {size_available['room_size'].median():.0f} sqm")
    print(f"Size range (IQR): {size_available['room_size'].quantile(0.25):.0f} - {size_available['room_size'].quantile(0.75):.0f} sqm")
    print("\nSize by category:")
    for cat in size_by_category.index:
        print(f"  {cat}: {size_by_category.loc[cat, 'median']:.0f} sqm median")

print("\n--- ROOM VIEW ---")
if len(view_available) > 0:
    print(f"Data availability: {len(view_available)/len(room_features)*100:.1f}% of bookings have view data")
    print(f"Unique views: {view_available['room_view'].nunique()}")
    print(f"Most common view: {view_counts.index[0]} ({view_counts.values[0]:,} bookings)")
    print(f"Most expensive view: {price_by_view.index[0]} (€{price_by_view.iloc[0]['median']:.2f} median)")

print("\n--- PRICE RELATIONSHIPS ---")
if len(size_available) > 0:
    # Correlation
    corr = size_available[['room_size', 'daily_price']].corr().iloc[0, 1]
    print(f"Size-Price correlation: {corr:.3f}")
    print("\nPrice per sqm by category:")
    for cat in price_per_sqm.index:
        print(f"  {cat}: €{price_per_sqm.loc[cat, 'median']:.2f}/sqm")

print("\n" + "="*80)
print("SUMMARY FOR PRICING MODEL")
print("="*80)
print("1. Room size is available for ~50% of bookings - important feature when available")
print("2. Size varies by category: villas/cottages larger, rooms smaller")
print("3. Price per sqm varies: villas most expensive per sqm, rooms most efficient")
print("4. View data is sparse but valuable - premium views command higher prices")
print("5. Size × Category interaction matters - large rooms in premium categories")
print("="*80)

