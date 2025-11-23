"""
Section 3.1: Supply and Demand by City (Spain only)

Question: How are supply and demand distributed by city 
(number of hotels, rooms, bookings, and average daily price)?
"""
import sys
sys.path.insert(0, '../../..')
from lib.db import init_db
from lib.data_validator import validate_and_clean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize with reception halls excluded
con = validate_and_clean(init_db(), verbose=False, rooms_to_exclude=['reception_hall'])

print("="*80)
print("SECTION 3.1: SUPPLY AND DEMAND BY CITY")
print("="*80)

# %%
# Query 1: City-level analysis
print("\n1. CITY-LEVEL ANALYSIS")
print("-" * 80)

city_analysis = con.execute("""
    SELECT 
        hl.city,
        COUNT(DISTINCT hl.hotel_id) as num_hotels,
        COUNT(DISTINCT br.room_id) as num_room_configs,
        COUNT(DISTINCT b.id) as num_bookings,
        COUNT(*) as num_booked_rooms,
        SUM(br.total_price) as total_revenue,
        AVG(br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) as avg_daily_price,
        MEDIAN(br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) as median_daily_price
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
      AND hl.city IS NOT NULL
    GROUP BY hl.city
    ORDER BY num_bookings DESC
""").fetchdf()

print(f"\nTotal cities: {len(city_analysis)}")
print("\nTop 20 cities by bookings:")
print(city_analysis.head(20).to_string(index=False))

# %%
# Query 2: Room type distribution by top cities
print("\n2. ROOM TYPE DISTRIBUTION BY TOP 10 CITIES")
print("-" * 80)

room_type_by_city = con.execute("""
    SELECT 
        hl.city,
        br.room_type,
        COUNT(*) as num_bookings,
        AVG(br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) as avg_daily_price
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
      AND hl.city IS NOT NULL
    GROUP BY hl.city, br.room_type
    ORDER BY hl.city, num_bookings DESC
""").fetchdf()

# Show for top 5 cities
top_cities = city_analysis.head(5)['city'].tolist()
for city in top_cities:
    subset = room_type_by_city[room_type_by_city['city'] == city]
    print(f"\n{city}:")
    print(subset[['room_type', 'num_bookings', 'avg_daily_price']].to_string(index=False))

# %%
# Visualizations
print("\n3. CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Section 3.1: Supply and Demand by City (Spain)', fontsize=16, fontweight='bold', y=0.995)

# 1. Top 15 cities by bookings
ax1 = axes[0, 0]
top_15_cities = city_analysis.head(15)
bars1 = ax1.barh(range(len(top_15_cities)), top_15_cities['num_bookings'], 
                 color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_yticks(range(len(top_15_cities)))
ax1.set_yticklabels(top_15_cities['city'], fontsize=9)
ax1.set_xlabel('Number of Bookings', fontsize=11, fontweight='bold')
ax1.set_title('Top 15 Cities by Bookings', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)
# Add value labels
for i, (bar, val) in enumerate(zip(bars1, top_15_cities['num_bookings'])):
    ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:,}', 
             ha='left', va='center', fontsize=8, fontweight='bold')

# 2. Top 15 cities by median price
ax2 = axes[0, 1]
top_15_price = city_analysis.nlargest(15, 'median_daily_price')
bars2 = ax2.barh(range(len(top_15_price)), top_15_price['median_daily_price'], 
                 color='coral', edgecolor='black', alpha=0.7)
ax2.set_yticks(range(len(top_15_price)))
ax2.set_yticklabels(top_15_price['city'], fontsize=9)
ax2.set_xlabel('Median Daily Price (€)', fontsize=11, fontweight='bold')
ax2.set_title('Top 15 Cities by Median Price', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
# Add value labels
for i, (bar, val) in enumerate(zip(bars2, top_15_price['median_daily_price'])):
    ax2.text(val, bar.get_y() + bar.get_height()/2, f'€{val:.0f}', 
             ha='left', va='center', fontsize=8, fontweight='bold')

# 3. Hotels vs Bookings scatter
ax3 = axes[0, 2]
scatter = ax3.scatter(city_analysis['num_hotels'], 
                     city_analysis['num_bookings'],
                     c=city_analysis['median_daily_price'],
                     s=100, alpha=0.6, cmap='viridis', edgecolors='black')
ax3.set_xlabel('Number of Hotels', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Bookings', fontsize=11, fontweight='bold')
ax3.set_title('Hotels vs Bookings by City', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Median Daily Price (€)', fontsize=9, fontweight='bold')
# Label top cities
for _, row in city_analysis.head(5).iterrows():
    ax3.annotate(row['city'], (row['num_hotels'], row['num_bookings']),
                fontsize=8, ha='right', va='bottom')

# 4. Revenue by top 15 cities
ax4 = axes[1, 0]
top_15_revenue = city_analysis.nlargest(15, 'total_revenue')
bars4 = ax4.barh(range(len(top_15_revenue)), top_15_revenue['total_revenue'] / 1_000_000, 
                 color='mediumseagreen', edgecolor='black', alpha=0.7)
ax4.set_yticks(range(len(top_15_revenue)))
ax4.set_yticklabels(top_15_revenue['city'], fontsize=9)
ax4.set_xlabel('Total Revenue (€ millions)', fontsize=11, fontweight='bold')
ax4.set_title('Top 15 Cities by Total Revenue', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)
# Add value labels
for i, (bar, val) in enumerate(zip(bars4, top_15_revenue['total_revenue'] / 1_000_000)):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'€{val:.1f}M', 
             ha='left', va='center', fontsize=8, fontweight='bold')

# 5. Price distribution by top 10 cities
ax5 = axes[1, 1]
top_10_cities = city_analysis.head(10)
price_data_by_city = []
city_labels_short = []
for _, row in top_10_cities.iterrows():
    prices = con.execute("""
        SELECT br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('confirmed', 'Booked')
          AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
          AND hl.city = ?
    """, [row['city']]).fetchdf()['daily_price'].tolist()
    price_data_by_city.append(prices)
    city_labels_short.append(row['city'])

bp = ax5.boxplot(price_data_by_city, tick_labels=city_labels_short, vert=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
ax5.set_xlabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax5.set_ylabel('City', fontsize=11, fontweight='bold')
ax5.set_title('Price Distribution by Top 10 Cities', fontsize=12, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)
ax5.set_xlim(0, 500)
ax5.tick_params(axis='y', labelsize=8)

# 6. Room type mix by top 5 cities
ax6 = axes[1, 2]
top_5_cities = city_analysis.head(5)['city'].tolist()
room_type_pivot = room_type_by_city[room_type_by_city['city'].isin(top_5_cities)].pivot_table(
    index='city', columns='room_type', values='num_bookings', fill_value=0
)
# Normalize to percentages
room_type_pct = room_type_pivot.div(room_type_pivot.sum(axis=1), axis=0) * 100

room_type_pct.plot(kind='barh', stacked=True, ax=ax6, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                   edgecolor='black', linewidth=0.5)
ax6.set_xlabel('Percentage of Bookings (%)', fontsize=11, fontweight='bold')
ax6.set_ylabel('City', fontsize=11, fontweight='bold')
ax6.set_title('Room Type Mix by Top 5 Cities', fontsize=12, fontweight='bold')
ax6.legend(title='Room Type', fontsize=8, title_fontsize=9, loc='lower right')
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS: City-Level Supply & Demand (Spain)")
print("="*80)

print("\n--- MARKET SIZE ---")
print(f"Total cities with bookings: {len(city_analysis)}")
print(f"Total hotels: {city_analysis['num_hotels'].sum():,}")
print(f"Total bookings: {city_analysis['num_bookings'].sum():,}")
print(f"Total revenue: €{city_analysis['total_revenue'].sum() / 1_000_000:.1f}M")

print("\n--- TOP CITY: {0} ---".format(city_analysis.iloc[0]['city']))
top_city = city_analysis.iloc[0]
print(f"  Hotels: {top_city['num_hotels']:,}")
print(f"  Bookings: {top_city['num_bookings']:,} ({top_city['num_bookings']/city_analysis['num_bookings'].sum()*100:.1f}% of total)")
print(f"  Revenue: €{top_city['total_revenue'] / 1_000_000:.1f}M")
print(f"  Median price: €{top_city['median_daily_price']:.2f}")

print("\n--- MARKET CONCENTRATION ---")
city_analysis['market_share'] = city_analysis['num_bookings'] / city_analysis['num_bookings'].sum()
herfindahl_city = (city_analysis['market_share'] ** 2).sum()
print(f"Herfindahl index: {herfindahl_city:.4f}")
if herfindahl_city < 0.15:
    print("  → Highly competitive market (bookings spread across many cities)")
elif herfindahl_city < 0.25:
    print("  → Moderately concentrated market")
else:
    print("  → Highly concentrated market (few dominant cities)")

top_3_cities = city_analysis.head(3)['city'].tolist()
print(f"\nTop 3 cities ({', '.join(top_3_cities)}):")
print(f"  Represent {city_analysis.head(3)['num_bookings'].sum()/city_analysis['num_bookings'].sum()*100:.1f}% of bookings")
print(f"\nTop 10 cities represent {city_analysis.head(10)['num_bookings'].sum()/city_analysis['num_bookings'].sum()*100:.1f}% of bookings")

print("\n--- PRICE VARIATION ---")
highest_price_city = city_analysis.nlargest(1, 'median_daily_price').iloc[0]
lowest_price_city = city_analysis.nsmallest(1, 'median_daily_price').iloc[0]
print(f"Highest median price: {highest_price_city['city']} (€{highest_price_city['median_daily_price']:.2f})")
print(f"Lowest median price: {lowest_price_city['city']} (€{lowest_price_city['median_daily_price']:.2f})")
print(f"Price range: €{city_analysis['median_daily_price'].min():.2f} - €{city_analysis['median_daily_price'].max():.2f}")
print(f"Median price across all cities: €{city_analysis['median_daily_price'].median():.2f}")
print(f"Price variation (std dev): €{city_analysis['median_daily_price'].std():.2f}")

print("\n--- CITY TIERS ---")
print(f"Large markets (>5,000 bookings): {(city_analysis['num_bookings'] > 5000).sum()} cities")
print(f"Medium markets (1,000-5,000 bookings): {((city_analysis['num_bookings'] >= 1000) & (city_analysis['num_bookings'] <= 5000)).sum()} cities")
print(f"Small markets (<1,000 bookings): {(city_analysis['num_bookings'] < 1000).sum()} cities")

print("\n--- PREMIUM MARKETS ---")
print(f"Cities with median price >€100: {(city_analysis['median_daily_price'] > 100).sum()}")
print(f"Cities with median price >€150: {(city_analysis['median_daily_price'] > 150).sum()}")

print("="*80)

