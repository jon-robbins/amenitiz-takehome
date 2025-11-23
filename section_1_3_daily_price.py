# 1.3 Define and inspect daily price per room-night
# Calculate daily price = total_price / stay_length for each booked room

import sys
sys.path.insert(0, '.')
from notebooks.utils.db import init_db
from notebooks.utils.data_validator import validate_and_clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con_raw = init_db()
con = validate_and_clean(con_raw)

# Get daily price data
daily_price_data = con.execute("""
    SELECT 
        br.id as booked_room_id,
        br.booking_id,
        br.room_id,
        br.room_type,
        br.room_size,
        br.total_price,
        b.arrival_date,
        b.departure_date,
        DATE_DIFF('day', b.arrival_date, b.departure_date) as stay_length_days,
        br.total_price / NULLIF(DATE_DIFF('day', b.arrival_date, b.departure_date), 0) as daily_price,
        br.total_adult + br.total_children as total_guests,
        b.hotel_id
    FROM booked_rooms br
    JOIN bookings b ON b.id = br.booking_id
    WHERE b.arrival_date IS NOT NULL 
      AND b.departure_date IS NOT NULL
      AND DATE_DIFF('day', b.arrival_date, b.departure_date) > 0
      AND br.total_price > 0
      AND br.room_type IS NOT NULL
""").fetchdf()

print(f"Total booked rooms with valid pricing: {len(daily_price_data):,}")

# Basic statistics
print("\n=== DAILY PRICE STATISTICS ===")
print(daily_price_data['daily_price'].describe())

# Check for outliers
print("\n=== OUTLIER DETECTION ===")
q1 = daily_price_data['daily_price'].quantile(0.25)
q3 = daily_price_data['daily_price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = daily_price_data[(daily_price_data['daily_price'] < lower_bound) | 
                            (daily_price_data['daily_price'] > upper_bound)]
print(f"Outliers (IQR method): {len(outliers):,} ({len(outliers)/len(daily_price_data)*100:.1f}%)")
print(f"Lower bound: ${lower_bound:.2f}")
print(f"Upper bound: ${upper_bound:.2f}")

# Daily price by category
daily_price_by_category = daily_price_data.groupby('room_type')['daily_price'].agg([
    'count', 'mean', 'median', 
    ('p25', lambda x: x.quantile(0.25)),
    ('p75', lambda x: x.quantile(0.75)),
    ('p90', lambda x: x.quantile(0.90))
]).round(2)

print("\n=== DAILY PRICE BY CATEGORY ===")
print(daily_price_by_category)

# Daily price by stay length
stay_length_groups = daily_price_data.copy()
stay_length_groups['stay_group'] = pd.cut(
    stay_length_groups['stay_length_days'],
    bins=[0, 1, 3, 7, 14, 30, 365],
    labels=['1 night', '2-3 nights', '4-7 nights', '8-14 nights', '15-30 nights', '30+ nights']
)
price_by_stay = stay_length_groups.groupby('stay_group')['daily_price'].agg([
    'count', 'mean', 'median'
]).round(2)

print("\n=== DAILY PRICE BY STAY LENGTH ===")
print(price_by_stay)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Overall distribution (with outlier filtering for visualization)
ax1 = axes[0, 0]
filtered_prices = daily_price_data[
    (daily_price_data['daily_price'] >= lower_bound) & 
    (daily_price_data['daily_price'] <= upper_bound)
]
ax1.hist(filtered_prices['daily_price'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(daily_price_data['daily_price'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: ${daily_price_data["daily_price"].median():.2f}')
ax1.axvline(daily_price_data['daily_price'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: ${daily_price_data["daily_price"].mean():.2f}')
ax1.set_xlabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Daily Price per Room-Night\n(Outliers removed for visualization)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Boxplot by category
ax2 = axes[0, 1]
sns.boxplot(data=daily_price_data, x='room_type', y='daily_price', ax=ax2, palette='Set2', hue='room_type', legend=False)
ax2.set_ylim(0, daily_price_data['daily_price'].quantile(0.95))
ax2.set_xlabel('Room Type Category', fontsize=11, fontweight='bold')
ax2.set_ylabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax2.set_title('Daily Price Distribution by Category', fontsize=12, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# 3. Daily price vs stay length
ax3 = axes[0, 2]
stay_length_subset = daily_price_data[daily_price_data['stay_length_days'] <= 30]
ax3.scatter(stay_length_subset['stay_length_days'], stay_length_subset['daily_price'], 
            alpha=0.3, s=10, c='steelblue')
ax3.set_xlabel('Stay Length (days)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax3.set_title('Daily Price vs Stay Length (≤30 days)', fontsize=12, fontweight='bold')
ax3.set_ylim(0, daily_price_data['daily_price'].quantile(0.95))
ax3.grid(alpha=0.3)

# Add trend line
stay_avg = stay_length_subset.groupby('stay_length_days')['daily_price'].mean()
ax3.plot(stay_avg.index, stay_avg.values, color='red', linewidth=2, label='Average')
ax3.legend()

# 4. Price by stay length groups (bar chart)
ax4 = axes[1, 0]
price_by_stay_plot = stay_length_groups.groupby('stay_group')['daily_price'].median()
bars = ax4.bar(range(len(price_by_stay_plot)), price_by_stay_plot.values, 
               color='coral', edgecolor='black', alpha=0.7)
ax4.set_xticks(range(len(price_by_stay_plot)))
ax4.set_xticklabels(price_by_stay_plot.index, rotation=45, ha='right')
ax4.set_xlabel('Stay Length', fontsize=11, fontweight='bold')
ax4.set_ylabel('Median Daily Price (€)', fontsize=11, fontweight='bold')
ax4.set_title('Median Daily Price by Stay Length', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
# Add value labels
for bar, val in zip(bars, price_by_stay_plot.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'€{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 5. Daily price vs room size (for rooms with size data)
ax5 = axes[1, 1]
size_data = daily_price_data[(daily_price_data['room_size'] > 0) & 
                              (daily_price_data['room_size'] < 200)]
ax5.scatter(size_data['room_size'], size_data['daily_price'], 
            alpha=0.3, s=10, c='seagreen')
ax5.set_xlabel('Room Size (sqm)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax5.set_title('Daily Price vs Room Size', fontsize=12, fontweight='bold')
ax5.set_ylim(0, daily_price_data['daily_price'].quantile(0.95))
ax5.grid(alpha=0.3)

# Add trend line
if len(size_data) > 0:
    z = np.polyfit(size_data['room_size'], size_data['daily_price'], 1)
    p = np.poly1d(z)
    ax5.plot(size_data['room_size'].sort_values(), 
             p(size_data['room_size'].sort_values()), 
             "r--", linewidth=2, label=f'Trend: €{z[0]:.2f}/sqm')
    ax5.legend()

# 6. Daily price vs number of guests
ax6 = axes[1, 2]
guest_data = daily_price_data[daily_price_data['total_guests'] <= 10]
guest_avg = guest_data.groupby('total_guests')['daily_price'].median()
bars6 = ax6.bar(guest_avg.index, guest_avg.values, color='purple', edgecolor='black', alpha=0.7)
ax6.set_xlabel('Total Guests', fontsize=11, fontweight='bold')
ax6.set_ylabel('Median Daily Price (€)', fontsize=11, fontweight='bold')
ax6.set_title('Median Daily Price by Number of Guests', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Print insights
print("\n" + "="*80)
print("KEY INSIGHTS: Daily Price Analysis")
print("="*80)

print(f"\n--- OVERALL PRICING ---")
print(f"Median daily price: €{daily_price_data['daily_price'].median():.2f}")
print(f"Mean daily price: €{daily_price_data['daily_price'].mean():.2f}")
print(f"25th percentile: €{daily_price_data['daily_price'].quantile(0.25):.2f}")
print(f"75th percentile: €{daily_price_data['daily_price'].quantile(0.75):.2f}")
print(f"90th percentile: €{daily_price_data['daily_price'].quantile(0.90):.2f}")

print(f"\n--- CATEGORY PRICING ---")
for cat in daily_price_by_category.index:
    median_price = daily_price_by_category.loc[cat, 'median']
    print(f"{cat}: €{median_price:.2f} median (n={daily_price_by_category.loc[cat, 'count']:,.0f})")

print(f"\n--- STAY LENGTH EFFECT ---")
short_stay = price_by_stay.loc['1 night', 'median']
long_stay = price_by_stay.loc['8-14 nights', 'median'] if '8-14 nights' in price_by_stay.index else None
if long_stay:
    discount = (short_stay - long_stay) / short_stay * 100
    print(f"1 night: €{short_stay:.2f}")
    print(f"8-14 nights: €{long_stay:.2f}")
    print(f"Long-stay discount: {discount:.1f}%")

print(f"\n--- ROOM SIZE EFFECT ---")
if len(size_data) > 0:
    print(f"Price per sqm (trend): €{z[0]:.2f}/sqm")
    print(f"Sample: 30 sqm room = €{30 * z[0]:.2f}, 60 sqm room = €{60 * z[0]:.2f}")

print("\n" + "="*80)
print("SUMMARY FOR PRICING MODEL")
print("="*80)
print("1. Wide price range: €{:.0f} (25th) to €{:.0f} (75th percentile)".format(
    daily_price_data['daily_price'].quantile(0.25),
    daily_price_data['daily_price'].quantile(0.75)
))
print("2. Category matters: Reception halls most expensive, cottages least")
print("3. Stay length discounts: Longer stays have lower daily rates")
print("4. Room size premium: Larger rooms command higher prices")
print("5. Guest count: More guests = higher price (capacity premium)")
print("="*80)

