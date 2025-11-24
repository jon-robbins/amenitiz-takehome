# %%
# %load_ext autoreload
# %autoreload 2

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.stats import mstats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#suppress plt warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import seaborn as sns
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner

# Initialize database connection
con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_low_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    verbose=False
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)

# %% [markdown]
# # EDA

# %%
import os, sys
from pathlib import Path

parent_dir = Path(os.getcwd()).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner



config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    verbose=True
)
cleaner = DataCleaner(config)
con =cleaner.clean(init_db())

con_raw = init_db()
query = """
"""
print("Raw data counts:")
print("Bookings count: ",con_raw.execute("select count(*) as cnt from bookings").fetchdf()['cnt'][0])
print("Booked rooms count: ",con_raw.execute("select count(*) as cnt from booked_rooms").fetchdf()['cnt'][0])
print("Rooms count: ",con_raw.execute("select count(*) as cnt from rooms").fetchdf()['cnt'][0])
print("Hotel location count: ",con_raw.execute("select count(*) as cnt from hotel_location").fetchdf()['cnt'][0])
print("-"*50)
print("Cleaned data counts:")
print("Bookings count: ",con.execute("select count(*) as cnt from bookings").fetchdf()['cnt'][0])
print("Booked rooms count: ",con.execute("select count(*) as cnt from booked_rooms").fetchdf()['cnt'][0])
print("Rooms count: ",con.execute("select count(*) as cnt from rooms").fetchdf()['cnt'][0])
print("Hotel location count: ",con.execute("select count(*) as cnt from hotel_location").fetchdf()['cnt'][0])
print("-"*50)
# %% [markdown]
# ### 1. Data sanity and target definition
# 
# #### 1.1 Validate core joins between tables
# 
# **Question**  
# Are the key relationships between tables clean (e.g., every `booked_rooms.room_id` exists in `rooms.id`, and every `bookings.hotel_id` exists in `hotel_location.hotel_id`)? 
# 
# **SQL direction**  
# - Do `LEFT JOIN` checks: from `booked_rooms` to `rooms` on `room_id`, and from `bookings` to `hotel_location` on `hotel_id`, and count rows where the right side is `NULL` to find orphans.   
# - Save a note on whether there are missing links, because this affects how confidently you can build a single modeling table later. 

# %%
# 2. Validate booked_rooms.room_id -> rooms.id

print("=== booked_rooms.room_id -> rooms.id ===")

# Total booked_rooms rows
total_booked = con.execute("""
    SELECT COUNT(*) AS total_booked_rooms
    FROM booked_rooms
""").fetchdf()
print(total_booked)

# Count of booked_rooms rows whose room_id does NOT exist in rooms.id
orphan_booked_to_rooms = con.execute("""
    SELECT COUNT(*) AS missing_room_fk_rows
    FROM booked_rooms br
    LEFT JOIN rooms r
        ON br.room_id = r.id
    WHERE r.id IS NULL
""").fetchdf()
print(orphan_booked_to_rooms)

# Optional: show a few distinct missing room_ids (if any)
sample_missing_rooms = con.execute("""
    SELECT br.room_id, COUNT(*) AS n_bookings
    FROM booked_rooms br
    LEFT JOIN rooms r
        ON br.room_id = r.id
    WHERE r.id IS NULL
    GROUP BY br.room_id
    ORDER BY n_bookings DESC
    LIMIT 10
""").fetchdf()
print(sample_missing_rooms)


# %% [markdown]
# Great, so there is a 1:1 relationship between `booked_rooms.room_id` and `rooms.id`, except for when there's no room_id in the booked_rooms. Let's look at the bookings where there are null room_ids. 

# %% [markdown]
# 
# ---
# 
# #### 1.2 Understand what a "room" actually represents
# 
# **Question**  
# Is `rooms.id` a "room type" with `number_of_rooms` representing identical units, and how does that compare to the way `booked_rooms` references `room_id`? 
# 
# **SQL direction**  
# - Summarize `rooms` with `GROUP BY id` and inspect `number_of_rooms` to see if there are duplicates or oddities.   
# - Join `rooms` to `booked_rooms` on `room_id` and check how many bookings each `rooms.id` receives, and whether that scale roughly matches `number_of_rooms * days` intuition. 
# 
# 

# %%
# 1.2 Understand what a "room" actually represents
# Note: room_id = a room configuration (unique combo of hotel + type + size + view + occupancy)
#       room_type = category (room/apartment/villa/cottage/reception_hall)
#       number_of_rooms = count of identical physical units for that configuration

# Get data - HOTEL-LEVEL ANALYSIS
# Distribution of configurations per hotel
hotel_config_distribution = con.execute("""
    SELECT 
        b.hotel_id,
        COUNT(DISTINCT br.room_id) as num_configurations,
        SUM(r.number_of_rooms) as total_units,
        COUNT(DISTINCT br.room_type) as num_categories,
        STRING_AGG(DISTINCT br.room_type, ', ') as categories
    FROM bookings b
    JOIN booked_rooms br ON br.booking_id = b.id
    JOIN rooms r ON r.id = br.room_id
    WHERE b.hotel_id IS NOT NULL AND br.room_type IS NOT NULL
    GROUP BY b.hotel_id
""").fetchdf()

# Summary stats by room type (for category understanding)
room_type_summary = con.execute("""
    SELECT 
        br.room_type,
        COUNT(DISTINCT r.id) as num_configurations,
        SUM(r.number_of_rooms) as total_units,
        AVG(r.number_of_rooms) as avg_units_per_config,
        MEDIAN(r.number_of_rooms) as median_units_per_config
    FROM rooms r
    JOIN booked_rooms br ON br.room_id = r.id
    WHERE br.room_type IS NOT NULL
    GROUP BY br.room_type
    ORDER BY total_units DESC
""").fetchdf()

# Bookings per room (for utilization analysis)
bookings_per_room = con.execute("""
    SELECT 
        br.room_id,
        ANY_VALUE(br.room_type) as room_type,
        r.number_of_rooms,
        COUNT(*) as total_bookings,
        COUNT(*) * 1.0 / NULLIF(r.number_of_rooms, 0) as bookings_per_individual_room
    FROM booked_rooms br
    JOIN rooms r ON br.room_id = r.id
    GROUP BY br.room_id, r.number_of_rooms
    ORDER BY total_bookings DESC
""").fetchdf()

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



# %% [markdown]
# ## Section 1.2 Summary: Understanding Supply Structure
# 
# ### Data Model ✅
# - **`rooms.id`**: Room configuration (hotel + category + size + view + occupancy)
# - **`room_type`**: 5 categories (room, apartment, villa, cottage, reception_hall)
# - **`number_of_rooms`**: Count of identical physical units per configuration
# - **Data quality**: ~90% consistency in attributes per room_id
# 
# ### Market Structure
# **2,277 hotels** with **10,124 unique room configurations** and **5.8M total units**
# 
# **Hotel Inventory Diversity:**
# - **25% single-config hotels** (1 configuration) - apartments, villas, small B&Bs
# - **50% small hotels** (2-5 configurations) - typical small properties
# - **20% medium hotels** (6-10 configurations)
# - **6% large hotels** (10+ configurations) - diverse inventory
# 
# **Hotel Size (median: 156 units):**
# - Highly skewed distribution (mean: 2,568 units)
# - 75% of hotels have ≤893 units
# - Suggests mix of boutique properties and large hotel chains
# 
# ### Category Insights
# 
# **Specialization is the norm:**
# - **75% of hotels offer only 1 category** (specialized)
# - **21% offer 2 categories** (some diversity)
# - **4% offer 3+ categories** (full-service properties)
# 
# **Category Characteristics:**
# | Category | Configs | Avg Units/Config | Median Units | Utilization (bookings/unit) |
# |----------|---------|------------------|--------------|------------------------------|
# | **room** | 4,681 | 6.0 | 4 | **54** (highest demand) |
# | **apartment** | 4,370 | 2.3 | 1 | 31 |
# | **villa** | 758 | 1.3 | 1 | 20 |
# | **cottage** | 499 | 1.0 | 1 | 14 |
# | **reception_hall** | 29 | 14.8 | 17 | 5 (specialized/seasonal) |
# 
# ### Key Takeaways for Pricing
# 
# 1. **Hotel-level features matter most**: Location, size, brand drive base pricing
# 2. **Category drives utilization**: Rooms have 2x utilization vs apartments
# 3. **Most properties are small & specialized**: 75% have 2-5 configs in 1 category
# 4. **Room attributes for differentiation**: Within a hotel, use size, occupancy, view
# 5. **Avoid cross-hotel config comparisons**: A "room" in Paris ≠ "room" in rural France
# 
# ### Implication
# Model pricing as: **Hotel baseline** (location, size, market) + **Room adjustments** (category, size, occupancy, amenities)

# %% [markdown]
# ## 1.3 Define and inspect daily price per room-night
# 
# **Question:** What is the distribution of daily price per occupied room-night across all bookings?
# 
# **SQL direction:**
# - From `booked_rooms` join to `bookings` to get `arrival_date` and `departure_date`, compute stay length in days and derive `daily_price = total_price / stay_length` at the room level.
# - Optionally expand to one row per `room_id` per calendar date using a date series and then compute summary stats (mean, median, percentiles) of `daily_price`.

# %%
# 1.3 Define and inspect daily price per room-night
# Calculate daily price = total_price / stay_length for each booked room

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    verbose=False
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)

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
price_by_stay = stay_length_groups.groupby('stay_group', observed=True)['daily_price'].agg([
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
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# 2. Boxplot by category
ax2 = axes[0, 1]
sns.boxplot(data=daily_price_data, x='room_type', y='daily_price', ax=ax2, palette='Set2', hue='room_type', legend=False)
ax2.set_ylim(0, daily_price_data['daily_price'].quantile(0.95))
ax2.set_xlabel('Room Type Category', fontsize=11, fontweight='bold')
ax2.set_ylabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax2.set_title('Daily Price Distribution by Category', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
for label in ax2.get_xticklabels():
    label.set_ha('right')
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
ax3.legend(loc='upper right')

# 4. Price by stay length groups (bar chart)
ax4 = axes[1, 0]
price_by_stay_plot = stay_length_groups.groupby('stay_group', observed=True)['daily_price'].median()
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
    ax5.legend(loc='upper left')

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



# %% [markdown]
# 
# ### Overall Pricing Distribution
# - **Median**: €75/night
# - **Mean**: €92/night (right-skewed distribution)
# - **IQR**: €51 (25th) to €111 (75th percentile)
# - **90th percentile**: €160/night
# - **5.3% outliers** (prices >€200 or <€0)
# 
# ### Pricing by Category
# | Category | Median Price | Mean Price | Sample Size |
# |----------|--------------|------------|-------------|
# | **Villa** | €182 | €250 | 24,606 |
# | **Cottage** | €180 | €250 | 10,766 |
# | **Apartment** | €100 | €120 | 269,251 |
# | **Room** | €67 | €77 | 861,631 |
# | **Reception Hall** | €23 | €52 | 2,120 |
# 
# **Key Insight**: Villas/cottages command **2.7x premium** over standard rooms. Reception halls are cheapest (event pricing model).
# 
# ### Stay Length Effect ⚠️ **UNEXPECTED FINDING**
# | Stay Length | Median Daily Price |
# |-------------|-------------------|
# | 1 night | €64 |
# | 2-3 nights | €89 |
# | 4-7 nights | €93 |
# | 8-14 nights | €97 |
# | 15-30 nights | €68 |
# | 30+ nights | €45 |
# 
# **Paradox**: Longer stays (2-14 nights) have **HIGHER** daily prices, not discounts!
# - Possible reasons: Vacation bookings (peak season) vs long-term rentals (off-season)
# - Only 30+ night stays show true long-term discounts
# 
# ### Room Size Premium
# - **€0.88 per sqm** price increase
# - Example: 60 sqm room = €53 base, 30 sqm room = €27 base
# - **Room size explains significant price variation**
# 
# ### Guest Count Effect
# - More guests → higher prices (capacity premium)
# - Suggests per-person pricing or larger room selection
# 
# ### Data Quality
# - **1.17M valid bookings** with pricing data
# - Filtered out: zero prices, missing dates, negative stay lengths
# - Clean data for modeling
# 
# ### Implications for Pricing Model
# 1. **Category is primary driver**: 2.7x range between categories
# 2. **Stay length is complex**: Not a simple discount curve - need seasonality interaction
# 3. **Room size matters**: €0.88/sqm premium - include as feature
# 4. **Guest capacity**: Price scales with occupancy
# 5. **Wide variance within categories**: Need hotel-level and location features

# %% [markdown]
# # 2. Supply: rooms and features
# 
# ## 2.1 Room type, size, and view distributions
# 
# **Question:** On the supply side, which room types, room sizes, and room views are most and least common?
# 
# **SQL direction:**
# - Use `rooms` and (optionally) `booked_rooms` to `GROUP BY room_type`, binned `room_size`, and `room_view`, counting distinct `room_id` and summing `number_of_rooms`.
# - Produce simple frequency tables (e.g., top 10 room types by supply) to see what you can realistically model and what's too rare.

# %%
# 2.1 Room type, size, and view distributions
# Analyze supply-side features: room_type, room_size, room_view

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    verbose=False
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)

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
size_available = room_features[room_features['room_size'] > 0].copy()
print(f"Bookings with room_size data: {len(size_available):,} ({len(size_available)/len(room_features)*100:.1f}%)")

if len(size_available) > 0:
    print("\nRoom size statistics:")
    print(size_available['room_size'].describe())
    
    # Size by category
    size_by_category = size_available.groupby('room_type', observed=True)['room_size'].agg([
        'count', 'mean', 'median', 'std',
        ('p25', lambda x: x.quantile(0.25)),
        ('p75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    print("\nRoom size by category:")
    print(size_by_category)
    
    # Price per sqm by category
    size_available['price_per_sqm'] = size_available['daily_price'] / size_available['room_size']
    price_per_sqm = size_available.groupby('room_type', observed=True)['price_per_sqm'].agg([
        'median', 'mean'
    ]).round(2)
    
    print("\nPrice per sqm by category:")
    print(price_per_sqm)

# 2. ROOM VIEW ANALYSIS
print("\n" + "="*80)
print("2. ROOM VIEW ANALYSIS")
print("="*80)

# Check data quality
view_available = room_features[room_features['room_view'].notna() & (room_features['room_view'] != '')].copy()
print(f"Bookings with room_view data: {len(view_available):,} ({len(view_available)/len(room_features)*100:.1f}%)")

if len(view_available) > 0:
    # View distribution
    view_counts = view_available['room_view'].value_counts()
    print(f"\nTop 10 room views:")
    print(view_counts.head(10))
    
    # Price by view
    price_by_view = view_available.groupby('room_view', observed=True)['daily_price'].agg([
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
    
    price_by_size_category = size_available.groupby(['room_type', 'size_bin'], observed=True)['daily_price'].agg([
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
    ax1.tick_params(axis='x', rotation=45)
    for label in ax1.get_xticklabels():
        label.set_ha('right')
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
    ax2.legend(title='Category', fontsize=8, loc='upper left')
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
    ax3.tick_params(axis='x', rotation=45)
    for label in ax3.get_xticklabels():
        label.set_ha('right')
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
    ax6.tick_params(axis='x', rotation=45)
    for label in ax6.get_xticklabels():
        label.set_ha('right')

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

# %% [markdown]
# ## 2.2 Capacity and policy flags
# 
# **Question:** What is the distribution of capacity and policy flags (`max_occupancy`, `max_adults`, `pricing_per_person`, `events_allowed`, `pets_allowed`, `smoking_allowed`, `children_allowed`)?
# 
# **SQL direction:**
# - In `rooms`, compute histograms or grouped counts for numeric capacities (e.g., `max_occupancy`) and counts of `TRUE`/`FALSE` for each boolean flag.
# - Optionally cross-tab some of these with `room_type` (e.g., how many suites allow pets) to anticipate interaction terms for the model.

# %%
# 2.2 Capacity and policy flags
# Analyze capacity (max_occupancy, max_adults) and policy flags

import sys
sys.path.insert(0, '.')
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    # Impute policy flags based on data analysis findings
    impute_children_allowed=True,
    impute_events_allowed=True,
    verbose=False
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)

print("="*80)
print("SECTION 2.2: Capacity and Policy Flags Analysis")
print("="*80)

# Get room capacity and policy data with pricing
capacity_data = con.execute("""
    SELECT 
        r.id as room_id,
        r.max_occupancy,
        r.max_adults,
        r.pricing_per_person_activated,
        r.events_allowed,
        r.pets_allowed,
        r.smoking_allowed,
        r.children_allowed,
        r.number_of_rooms,
        br.room_type,
        br.room_size,
        br.total_adult,
        br.total_children,
        br.total_price,
        b.arrival_date,
        b.departure_date,
        DATE_DIFF('day', b.arrival_date, b.departure_date) as stay_length,
        br.total_price / NULLIF(DATE_DIFF('day', b.arrival_date, b.departure_date), 0) as daily_price
    FROM rooms r
    JOIN booked_rooms br ON br.room_id = r.id
    JOIN bookings b ON b.id = br.booking_id
    WHERE b.arrival_date IS NOT NULL 
      AND b.departure_date IS NOT NULL
      AND DATE_DIFF('day', b.arrival_date, b.departure_date) > 0
      AND br.total_price > 0
      AND br.room_type IS NOT NULL
""").fetchdf()

print(f"\nTotal bookings with capacity/policy data: {len(capacity_data):,}")

# 1. CAPACITY ANALYSIS
print("\n" + "="*80)
print("1. CAPACITY ANALYSIS")
print("="*80)

print("\n--- Max Occupancy Distribution ---")
print(capacity_data['max_occupancy'].describe())

occupancy_by_category = capacity_data.groupby('room_type')['max_occupancy'].agg([
    'count', 'mean', 'median', 'min', 'max'
]).round(2)
print("\nMax occupancy by category:")
print(occupancy_by_category)

print("\n--- Max Adults Distribution ---")
print(capacity_data['max_adults'].describe())

adults_by_category = capacity_data.groupby('room_type')['max_adults'].agg([
    'count', 'mean', 'median', 'min', 'max'
]).round(2)
print("\nMax adults by category:")
print(adults_by_category)

# Actual occupancy vs capacity
capacity_data['total_guests'] = capacity_data['total_adult'] + capacity_data['total_children']
capacity_data['occupancy_rate'] = capacity_data['total_guests'] / capacity_data['max_occupancy']

print("\n--- Actual Occupancy vs Capacity ---")
print(f"Average occupancy rate: {capacity_data['occupancy_rate'].mean():.2%}")
print(f"Median occupancy rate: {capacity_data['occupancy_rate'].median():.2%}")

# 2. POLICY FLAGS ANALYSIS
print("\n" + "="*80)
print("2. POLICY FLAGS ANALYSIS")
print("="*80)

policy_flags = ['pricing_per_person_activated', 'events_allowed', 'pets_allowed', 
                'smoking_allowed', 'children_allowed']

print("\n--- Overall Policy Distribution ---")
for flag in policy_flags:
    true_count = capacity_data[flag].sum()
    total = len(capacity_data)
    pct = true_count / total * 100
    print(f"{flag}: {true_count:,} / {total:,} ({pct:.1f}%) allow")

# Policy by category
print("\n--- Policy Flags by Category ---")
policy_by_category = capacity_data.groupby('room_type')[policy_flags].mean() * 100
print(policy_by_category.round(1))

# 3. PRICING RELATIONSHIPS
print("\n" + "="*80)
print("3. PRICING RELATIONSHIPS")
print("="*80)

# Price by capacity
print("\n--- Price by Max Occupancy ---")
price_by_occupancy = capacity_data.groupby('max_occupancy')['daily_price'].agg([
    'count', 'median', 'mean'
]).head(10)
print(price_by_occupancy.round(2))

# Price by policy flags
print("\n--- Price Impact of Policy Flags ---")
for flag in policy_flags:
    allowed = capacity_data[capacity_data[flag] == True]['daily_price'].median()
    not_allowed = capacity_data[capacity_data[flag] == False]['daily_price'].median()
    diff_pct = (allowed - not_allowed) / not_allowed * 100
    print(f"{flag}:")
    print(f"  Allowed: €{allowed:.2f} median")
    print(f"  Not allowed: €{not_allowed:.2f} median")
    print(f"  Difference: {diff_pct:+.1f}%")

# Occupancy rate vs price
print("\n--- Occupancy Rate vs Price ---")
capacity_data['occupancy_bin'] = pd.cut(
    capacity_data['occupancy_rate'],
    bins=[0, 0.5, 0.75, 1.0, 2.0],
    labels=['<50%', '50-75%', '75-100%', '>100%']
)
price_by_occ_rate = capacity_data.groupby('occupancy_bin')['daily_price'].agg([
    'count', 'median'
]).round(2)
print(price_by_occ_rate)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Max occupancy distribution by category
ax1 = axes[0, 0]
sns.boxplot(data=capacity_data, x='room_type', y='max_occupancy', ax=ax1, 
            palette='Set2', hue='room_type', legend=False)
ax1.set_xlabel('Room Type Category', fontsize=11, fontweight='bold')
ax1.set_ylabel('Max Occupancy', fontsize=11, fontweight='bold')
ax1.set_title('Maximum Occupancy by Category', fontsize=12, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_ylim(0, 15)
ax1.grid(axis='y', alpha=0.3)

# 2. Policy flags distribution
ax2 = axes[0, 1]
policy_summary = capacity_data[policy_flags].mean() * 100
bars = ax2.barh(range(len(policy_summary)), policy_summary.values, 
                color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_yticks(range(len(policy_summary)))
ax2.set_yticklabels([p.replace('_', ' ').title() for p in policy_summary.index], fontsize=9)
ax2.set_xlabel('% of Bookings Allowing', fontsize=11, fontweight='bold')
ax2.set_title('Policy Flags: Overall Distribution', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
# Add percentage labels
for bar, val in zip(bars, policy_summary.values):
    ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')

# 3. Policy flags by category (heatmap)
ax3 = axes[0, 2]
sns.heatmap(policy_by_category.T, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax3, 
            cbar_kws={'label': '% Allowing'})
ax3.set_xlabel('Room Type Category', fontsize=11, fontweight='bold')
ax3.set_ylabel('Policy Flag', fontsize=11, fontweight='bold')
ax3.set_title('Policy Flags by Category (%)', fontsize=12, fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.set_yticklabels([p.replace('_', ' ').title() for p in policy_flags], rotation=0)

# 4. Price by max occupancy
ax4 = axes[1, 0]
price_occ_plot = capacity_data[capacity_data['max_occupancy'] <= 10].groupby('max_occupancy')['daily_price'].median()
bars4 = ax4.bar(price_occ_plot.index, price_occ_plot.values, 
                color='coral', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Max Occupancy', fontsize=11, fontweight='bold')
ax4.set_ylabel('Median Daily Price (€)', fontsize=11, fontweight='bold')
ax4.set_title('Price by Maximum Occupancy', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Occupancy rate distribution
ax5 = axes[1, 1]
capacity_filtered = capacity_data[capacity_data['occupancy_rate'] <= 1.5]
ax5.hist(capacity_filtered['occupancy_rate'], bins=30, color='seagreen', edgecolor='black', alpha=0.7)
ax5.axvline(capacity_filtered['occupancy_rate'].median(), color='red', linestyle='--', 
            linewidth=2, label=f'Median: {capacity_filtered["occupancy_rate"].median():.2%}')
ax5.set_xlabel('Occupancy Rate (Actual / Max)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Actual Occupancy Rate Distribution', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Price impact of policies (comparison)
ax6 = axes[1, 2]
price_impacts = []
labels = []
for flag in policy_flags:
    allowed = capacity_data[capacity_data[flag] == True]['daily_price'].median()
    not_allowed = capacity_data[capacity_data[flag] == False]['daily_price'].median()
    diff_pct = (allowed - not_allowed) / not_allowed * 100
    price_impacts.append(diff_pct)
    labels.append(flag.replace('_', ' ').title())

colors = ['green' if x > 0 else 'red' for x in price_impacts]
bars6 = ax6.barh(range(len(price_impacts)), price_impacts, color=colors, edgecolor='black', alpha=0.7)
ax6.set_yticks(range(len(labels)))
ax6.set_yticklabels(labels, fontsize=9)
ax6.set_xlabel('Price Impact (%)', fontsize=11, fontweight='bold')
ax6.set_title('Price Impact of Policy Flags', fontsize=12, fontweight='bold')
ax6.axvline(0, color='black', linestyle='-', linewidth=1)
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)
# Add percentage labels
for bar, val in zip(bars6, price_impacts):
    ax6.text(val, bar.get_y() + bar.get_height()/2, 
             f'{val:+.1f}%', ha='left' if val > 0 else 'right', va='center', 
             fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS: Capacity and Policy Flags")
print("="*80)

print("\n--- CAPACITY ---")
print(f"Median max occupancy: {capacity_data['max_occupancy'].median():.0f} guests")
print(f"Median max adults: {capacity_data['max_adults'].median():.0f} adults")
print(f"Actual occupancy rate: {capacity_data['occupancy_rate'].median():.2%} (median)")
print("\nOccupancy by category:")
for cat in occupancy_by_category.index:
    print(f"  {cat}: {occupancy_by_category.loc[cat, 'median']:.0f} guests median")

print("\n--- POLICY FLAGS ---")
print("Most restrictive to most permissive:")
policy_pcts = capacity_data[policy_flags].mean() * 100
for flag in policy_pcts.sort_values().index:
    print(f"  {flag}: {policy_pcts[flag]:.1f}% allow")

print("\n--- PRICING IMPACT ---")
print("Policies with positive price impact:")
for flag in policy_flags:
    allowed = capacity_data[capacity_data[flag] == True]['daily_price'].median()
    not_allowed = capacity_data[capacity_data[flag] == False]['daily_price'].median()
    diff_pct = (allowed - not_allowed) / not_allowed * 100
    if diff_pct > 0:
        print(f"  {flag}: +{diff_pct:.1f}%")

print("\nCapacity-price relationship:")
corr = capacity_data[['max_occupancy', 'daily_price']].corr().iloc[0, 1]
print(f"  Max occupancy vs price correlation: {corr:.3f}")

print("\n" + "="*80)
print("SUMMARY FOR PRICING MODEL")
print("="*80)
print(f"1. Capacity matters: Larger occupancy = higher prices (correlation: {corr:.2f})")
print("2. Most properties are restrictive: <50% allow pets, smoking, events")
print(f"3. Children allowed is most common ({policy_pcts['children_allowed']:.1f}%)")
print(f"4. Pricing per person is rare ({policy_pcts['pricing_per_person_activated']:.1f}%) - most charge per room")
print("5. Policy flags have mixed price impact - some positive, some negative")
print(f"6. Actual occupancy rate ~{capacity_data['occupancy_rate'].median()*100:.0f}% - rooms not always fully utilized")
print("="*80)

# %% [markdown]
# Event halls are giving significantly lower prices. This is probably due to different pricing strategies and purposes. We'll narrow the purpose of our model to solely focus on accommodations, not event halls. That would make our model too complex. 
# 
# I'll add a parameter in our clean_and_validate function to remove reception_halls entirely. 

# %% [markdown]
# # 3. Location and market structure
# 
# ## 3.1 Supply and demand by city and country
# 
# **Question:** How are supply and demand distributed by city and country (number of hotels, rooms, bookings, and average daily price)?
# 
# **SQL direction:**
# - Join `bookings` to `hotel_location` on `hotel_id` and compute, by `city` and `country`, counts of distinct hotels, counts of bookings, and total revenue.
# - Join in your per-night `daily_price` table to compute average daily price per city and country.

# %%
"""
Section 3.1: Supply and Demand by City (Spain only)

Question: How are supply and demand distributed by city 
(number of hotels, rooms, bookings, and average daily price)?
"""
import sys
sys.path.insert(0, '.')
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize with reception halls excluded and missing locations excluded
con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_low_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    impute_children_allowed=False,
    impute_events_allowed=False,
    exclude_reception_halls=True,  # Exclude event spaces from analysis
    exclude_missing_location=True,  # Exclude hotels with no location data
    verbose=False
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)

print("="*80)
print("SECTION 3.1: SUPPLY AND DEMAND BY CITY")
print("="*80)

# Query 1: City-level analysis BEFORE consolidation
print("\n1. CITY-LEVEL ANALYSIS (BEFORE CONSOLIDATION)")
print("-" * 80)

city_analysis_raw = con.execute("""
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

print(f"\nTotal cities (before consolidation): {len(city_analysis_raw)}")
print("\nTop 20 cities by bookings:")
print(city_analysis_raw.head(20).to_string(index=False))

# City consolidation using TF-IDF + cosine similarity
print("\n2. CITY NAME CONSOLIDATION")
print("-" * 80)
print("Using TF-IDF with cosine similarity >= 0.95 to consolidate city names...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get all unique cities
cities = city_analysis_raw['city'].unique().tolist()
print(f"Original unique cities: {len(cities)}")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
tfidf_matrix = vectorizer.fit_transform(cities)

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Create mapping based on similarity threshold
city_mapping = {}
processed = set()

for i, city1 in enumerate(cities):
    if city1 in processed:
        continue
    
    # Find all similar cities (>= 0.95 similarity)
    similar_indices = [j for j, sim in enumerate(cosine_sim[i]) if sim >= 0.95 and i != j]
    
    if similar_indices:
        # Get the similar cities and their booking counts
        similar_cities = [cities[j] for j in similar_indices] + [city1]
        booking_counts = city_analysis_raw[city_analysis_raw['city'].isin(similar_cities)].groupby('city')['num_bookings'].sum()
        
        # Choose the city with most bookings as canonical
        canonical = booking_counts.idxmax()
        
        # Map all similar cities to the canonical one
        for city in similar_cities:
            city_mapping[city] = canonical
            processed.add(city)
    else:
        city_mapping[city1] = city1
        processed.add(city1)

print(f"Consolidated to: {len(set(city_mapping.values()))} unique cities")
print(f"Consolidated {len(cities) - len(set(city_mapping.values()))} cities")

# Show some examples of consolidation
print("\nExample consolidations:")
examples = {}
for orig, canon in city_mapping.items():
    if orig != canon:
        if canon not in examples:
            examples[canon] = []
        examples[canon].append(orig)

for canon, variants in list(examples.items())[:10]:
    print(f"  '{canon}' ← {variants}")

# Apply consolidation
city_analysis_raw['city_consolidated'] = city_analysis_raw['city'].map(city_mapping)

# Re-aggregate after consolidation
city_analysis = city_analysis_raw.groupby('city_consolidated').agg({
    'num_hotels': 'sum',
    'num_room_configs': 'sum',
    'num_bookings': 'sum',
    'num_booked_rooms': 'sum',
    'total_revenue': 'sum'
}).reset_index()
city_analysis = city_analysis.rename(columns={'city_consolidated': 'city'})

# Recalculate weighted averages for prices
price_data = con.execute("""
    SELECT 
        hl.city,
        br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
      AND hl.city IS NOT NULL
""").fetchdf()
price_data['city_consolidated'] = price_data['city'].map(city_mapping)

price_stats = price_data.groupby('city_consolidated')['daily_price'].agg(['mean', 'median']).reset_index()
price_stats = price_stats.rename(columns={
    'city_consolidated': 'city',
    'mean': 'avg_daily_price',
    'median': 'median_daily_price'
})

city_analysis = city_analysis.merge(price_stats, on='city', how='left')
city_analysis = city_analysis.sort_values('num_bookings', ascending=False)

print("\n3. CONSOLIDATED CITY-LEVEL ANALYSIS")
print("-" * 80)
print(f"\nTotal cities (after consolidation): {len(city_analysis)}")
print("\nTop 20 cities by bookings:")
print(city_analysis.head(20).to_string(index=False))

# Query 2: Room type distribution by top cities
print("\n4. ROOM TYPE DISTRIBUTION BY TOP 10 CITIES")
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
room_type_by_city['city_consolidated'] = room_type_by_city['city'].map(city_mapping)

# Show for top 5 cities
top_cities = city_analysis.head(5)['city'].tolist()
for city in top_cities:
    subset = room_type_by_city[room_type_by_city['city_consolidated'] == city]
    subset_agg = subset.groupby('room_type').agg({
        'num_bookings': 'sum',
        'avg_daily_price': 'mean'
    }).sort_values('num_bookings', ascending=False)
    print(f"\n{city}:")
    print(subset_agg.to_string())

# Visualizations
print("\n5. CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Section 3.1: Supply and Demand by City (Spain - Consolidated)', fontsize=16, fontweight='bold', y=0.995)

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

# 5. Consolidation impact: before vs after
ax5 = axes[1, 1]
consolidation_data = pd.DataFrame({
    'Stage': ['Before\nConsolidation', 'After\nConsolidation'],
    'Cities': [len(city_analysis_raw), len(city_analysis)]
})
bars5 = ax5.bar(consolidation_data['Stage'], consolidation_data['Cities'], 
                color=['lightcoral', 'lightgreen'], edgecolor='black', alpha=0.7)
ax5.set_ylabel('Number of Unique Cities', fontsize=11, fontweight='bold')
ax5.set_title('City Consolidation Impact', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for bar, val in zip(bars5, consolidation_data['Cities']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. Room type mix by top 5 cities
ax6 = axes[1, 2]
top_5_cities = city_analysis.head(5)['city'].tolist()
room_type_pivot = room_type_by_city[room_type_by_city['city_consolidated'].isin(top_5_cities)].groupby(
    ['city_consolidated', 'room_type']
)['num_bookings'].sum().reset_index().pivot_table(
    index='city_consolidated', columns='room_type', values='num_bookings', fill_value=0
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

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS: City-Level Supply & Demand (Spain - Consolidated)")
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

# Interactive heatmap visualization
print("\n6. INTERACTIVE HEATMAP VISUALIZATION")
print("-" * 80)
print("Creating interactive temporal heatmap...")

from lib.interactive_heatmap import create_interactive_heatmap
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path('outputs/city_consolidation')
output_dir.mkdir(parents=True, exist_ok=True)

# Generate interactive heatmap
# Generate interactive heatmap (will display inline in Jupyter)
heatmap = create_interactive_heatmap(
    con=con,
    output_path=output_dir / 'spain_bookings_heatmap.html',  # Also save to file
    title='Spain Bookings - Weekly Patterns'
)

print("\n✓ Interactive heatmap created!")
print(f"  - Saved to: {output_dir / 'spain_bookings_heatmap.html'}")
print("  - Displaying below (interactive in Jupyter):")
print("\nFeatures:")
print("  • Weekly booking patterns throughout the year")
print("  • Use the week slider to explore different weeks")
print("  • Click 'Play' to animate through the year")
print("  • Adjust radius slider to change heatmap detail level")
print("  • Zoom and pan to explore regions")

# Display the map inline in Jupyter
heatmap

# %% [markdown]
# Key takeaways:
# - Successfully consolidated city names using TF-IDF and cosine similarity (threshold >= 0.95)
# - Created an interactive heatmap showing weekly booking patterns across Spain
# - Two key location signals for pricing: distance from water and distance from Madrid

# %% [markdown]
# # 4. Time, seasonality, and booking intensity
# 
# ## 4.1 Seasonality in price
# 
# **Question:** How does daily price vary by month and day of week across the year?
# 
# **SQL direction:**
# - On the per-night `daily_price` table, derive `month = EXTRACT(MONTH FROM stay_date)` and `dow = EXTRACT(DOW FROM stay_date)`.
# - `GROUP BY month, dow` and compute mean and median `daily_price` plus counts of room-nights to see patterns like weekend or high-season premiums.

# %%
# 4.1 Seasonality in price
# Analyze how price varies by month and day of week

import sys
sys.path.insert(0, '.')
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    exclude_reception_halls=True,  # Focus on accommodations only
    exclude_missing_location=True,
    verbose=False
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)

print("="*80)
print("SECTION 4.1: SEASONALITY IN PRICE")
print("="*80)

# Create a per-night expansion to analyze daily patterns
print("\n1. CREATING PER-NIGHT EXPANSION")
print("-" * 80)

# Use DuckDB's generate_series to expand each booking to individual nights
per_night_data = con.execute("""
    WITH date_range AS (
        SELECT 
            b.id as booking_id,
            br.id as booked_room_id,
            b.arrival_date,
            b.departure_date,
            CAST(b.arrival_date AS DATE) + INTERVAL (n) DAY as stay_date,
            br.total_price,
            DATE_DIFF('day', b.arrival_date, b.departure_date) as stay_length,
            br.room_type,
            b.hotel_id,
            hl.city
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        CROSS JOIN generate_series(0, DATE_DIFF('day', b.arrival_date, b.departure_date) - 1) as t(n)
        WHERE b.status IN ('confirmed', 'Booked')
          AND DATE_DIFF('day', b.arrival_date, b.departure_date) > 0
          AND br.total_price > 0
          AND hl.city IS NOT NULL
    )
    SELECT 
        stay_date,
        EXTRACT(MONTH FROM stay_date) as month,
        EXTRACT(DOW FROM stay_date) as dow,
        EXTRACT(WEEK FROM stay_date) as week,
        EXTRACT(YEAR FROM stay_date) as year,
        total_price / stay_length as daily_price,
        room_type,
        city
    FROM date_range
""").fetchdf()

print(f"Total room-nights: {len(per_night_data):,}")
print(f"Date range: {per_night_data['stay_date'].min()} to {per_night_data['stay_date'].max()}")
print(f"Average daily price: €{per_night_data['daily_price'].mean():.2f}")
print(f"Median daily price: €{per_night_data['daily_price'].median():.2f}")

# 2. MONTHLY SEASONALITY
print("\n2. MONTHLY SEASONALITY")
print("-" * 80)

monthly_stats = per_night_data.groupby('month')['daily_price'].agg([
    'count', 'mean', 'median', 'std',
    ('p25', lambda x: x.quantile(0.25)),
    ('p75', lambda x: x.quantile(0.75))
]).round(2)

# Add month names for readability
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_stats['month_name'] = [month_names[int(m)-1] for m in monthly_stats.index]

print("\nMonthly price statistics:")
print(monthly_stats.to_string())

# Identify peak and low seasons
peak_month = monthly_stats['median'].idxmax()
low_month = monthly_stats['median'].idxmin()
print(f"\nPeak season: {month_names[int(peak_month)-1]} (€{monthly_stats.loc[peak_month, 'median']:.2f} median)")
print(f"Low season: {month_names[int(low_month)-1]} (€{monthly_stats.loc[low_month, 'median']:.2f} median)")
print(f"Seasonal price variation: {(monthly_stats['median'].max() - monthly_stats['median'].min()) / monthly_stats['median'].min() * 100:.1f}%")

# 3. DAY OF WEEK PATTERNS
print("\n3. DAY OF WEEK PATTERNS")
print("-" * 80)

dow_stats = per_night_data.groupby('dow')['daily_price'].agg([
    'count', 'mean', 'median', 'std'
]).round(2)

# Add day names
dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
dow_stats['day_name'] = [dow_names[int(d)] for d in dow_stats.index]

print("\nDay of week price statistics:")
print(dow_stats.to_string())

# Weekend vs weekday
weekend_mask = per_night_data['dow'].isin([0, 6])  # Sunday=0, Saturday=6
weekend_price = per_night_data[weekend_mask]['daily_price'].median()
weekday_price = per_night_data[~weekend_mask]['daily_price'].median()
weekend_premium = (weekend_price - weekday_price) / weekday_price * 100

print(f"\nWeekend vs Weekday:")
print(f"  Weekend median: €{weekend_price:.2f}")
print(f"  Weekday median: €{weekday_price:.2f}")
print(f"  Weekend premium: {weekend_premium:+.1f}%")

# 4. COMBINED SEASONALITY (Month × Day of Week)
print("\n4. COMBINED SEASONALITY (Month × Day of Week)")
print("-" * 80)

month_dow_pivot = per_night_data.groupby(['month', 'dow'])['daily_price'].median().unstack(fill_value=0)
month_dow_pivot.index = [month_names[int(m)-1] for m in month_dow_pivot.index]
month_dow_pivot.columns = [dow_names[int(d)] for d in month_dow_pivot.columns]

print("\nMedian daily price by month and day of week:")
print(month_dow_pivot.round(2).to_string())

# Visualizations
print("\n5. CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Section 4.1: Seasonality in Price', fontsize=16, fontweight='bold', y=0.995)

# 1. Monthly price trend (line + confidence band)
ax1 = axes[0, 0]
monthly_for_plot = monthly_stats.reset_index()
ax1.plot(monthly_for_plot['month'], monthly_for_plot['median'], 
         marker='o', linewidth=2, markersize=8, color='steelblue', label='Median')
ax1.fill_between(monthly_for_plot['month'], 
                 monthly_for_plot['p25'], 
                 monthly_for_plot['p75'],
                 alpha=0.3, color='steelblue', label='IQR (25th-75th)')
ax1.set_xlabel('Month', fontsize=11, fontweight='bold')
ax1.set_ylabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax1.set_title('Seasonal Price Trend (Monthly)', fontsize=12, fontweight='bold')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_names, rotation=45, ha='right')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 2. Monthly volume (bar chart)
ax2 = axes[0, 1]
bars2 = ax2.bar(monthly_for_plot['month'], monthly_for_plot['count'] / 1000, 
                color='mediumseagreen', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
ax2.set_ylabel('Room-Nights (thousands)', fontsize=11, fontweight='bold')
ax2.set_title('Booking Volume by Month', fontsize=12, fontweight='bold')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# 3. Day of week pattern
ax3 = axes[0, 2]
dow_for_plot = dow_stats.reset_index()
colors = ['coral' if d in [0, 6] else 'steelblue' for d in dow_for_plot['dow']]
bars3 = ax3.bar(dow_for_plot['dow'], dow_for_plot['median'], 
                color=colors, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
ax3.set_ylabel('Median Daily Price (€)', fontsize=11, fontweight='bold')
ax3.set_title('Price by Day of Week', fontsize=12, fontweight='bold')
ax3.set_xticks(range(7))
ax3.set_xticklabels([d[:3] for d in dow_names], rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)
# Add value labels
for bar, val in zip(bars3, dow_for_plot['median']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'€{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 4. Heatmap: Month × Day of Week
ax4 = axes[1, 0]
sns.heatmap(month_dow_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4, 
            cbar_kws={'label': 'Median Daily Price (€)'})
ax4.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
ax4.set_ylabel('Month', fontsize=11, fontweight='bold')
ax4.set_title('Price Heatmap: Month × Day of Week', fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)

# 5. Price distribution by season
ax5 = axes[1, 1]
# Define seasons
spring_months = [3, 4, 5]
summer_months = [6, 7, 8]
fall_months = [9, 10, 11]
winter_months = [12, 1, 2]

season_data = []
season_labels = []
for season_name, months in [('Spring', spring_months), ('Summer', summer_months), 
                            ('Fall', fall_months), ('Winter', winter_months)]:
    season_prices = per_night_data[per_night_data['month'].isin(months)]['daily_price']
    season_data.append(season_prices[season_prices <= 300].tolist())  # Cap for visualization
    season_labels.append(season_name)

bp = ax5.boxplot(season_data, labels=season_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightgreen', 'coral', 'orange', 'lightblue']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_ylabel('Daily Price (€)', fontsize=11, fontweight='bold')
ax5.set_xlabel('Season', fontsize=11, fontweight='bold')
ax5.set_title('Price Distribution by Season', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. Weekly pattern throughout the year
ax6 = axes[1, 2]
weekly_median = per_night_data.groupby('week')['daily_price'].median()
ax6.plot(weekly_median.index, weekly_median.values, 
         linewidth=2, color='steelblue', alpha=0.8)
ax6.axhline(per_night_data['daily_price'].median(), 
            color='red', linestyle='--', linewidth=1, label='Overall Median')
ax6.set_xlabel('Week of Year', fontsize=11, fontweight='bold')
ax6.set_ylabel('Median Daily Price (€)', fontsize=11, fontweight='bold')
ax6.set_title('Weekly Price Trend Throughout Year', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS: Seasonality in Price")
print("="*80)

print("\n--- MONTHLY SEASONALITY ---")
print(f"Peak pricing month: {month_names[int(peak_month)-1]} (€{monthly_stats.loc[peak_month, 'median']:.2f})")
print(f"Lowest pricing month: {month_names[int(low_month)-1]} (€{monthly_stats.loc[low_month, 'median']:.2f})")
print(f"Seasonal variation: {(monthly_stats['median'].max() - monthly_stats['median'].min()):.2f} (€{(monthly_stats['median'].max() - monthly_stats['median'].min()) / monthly_stats['median'].min() * 100:.1f}%)")

# Identify high/low seasons
summer_median = per_night_data[per_night_data['month'].isin(summer_months)]['daily_price'].median()
winter_median = per_night_data[per_night_data['month'].isin(winter_months)]['daily_price'].median()
print(f"\nSummer (Jun-Aug): €{summer_median:.2f} median")
print(f"Winter (Dec-Feb): €{winter_median:.2f} median")
print(f"Summer premium: {(summer_median - winter_median) / winter_median * 100:+.1f}%")

print("\n--- DAY OF WEEK PATTERNS ---")
peak_dow = dow_stats['median'].idxmax()
print(f"Highest price day: {dow_names[int(peak_dow)]} (€{dow_stats.loc[peak_dow, 'median']:.2f})")
print(f"Weekend premium: {weekend_premium:+.1f}%")

print("\n--- VOLUME PATTERNS ---")
peak_volume_month = monthly_stats['count'].idxmax()
print(f"Busiest month: {month_names[int(peak_volume_month)-1]} ({monthly_stats.loc[peak_volume_month, 'count']:,} room-nights)")
print(f"Volume concentration: Top 3 months account for {monthly_stats.nlargest(3, 'count')['count'].sum() / monthly_stats['count'].sum() * 100:.1f}% of stays")

print("\n" + "="*80)
print("SUMMARY FOR PRICING MODEL")
print("="*80)
print("1. Strong monthly seasonality - summer months command premium prices")
print(f"2. Weekend premium exists but modest ({weekend_premium:+.1f}%)")
print("3. Price peaks align with volume peaks - high demand = high prices")
print("4. Consider month and day-of-week as important time features in model")
print("5. May want to create 'season' categorical variable (Spring/Summer/Fall/Winter)")
print("="*80)

# %%
# [markdown]
# I think we're still seeing a lot of underlying trends due to highly desirable locations. I can notice two patterns:
# - Areas close to Madrid are in high demand
# - Areas close to the ocean are in high demand

# - For each room, let's get these two continuous variables, and then see the correlation between price and distance from water/madrid. 
# %%
# # Distance Features: Madrid and Coast
# 
# **Pre-calculated features** for pricing analysis:
# - distance_from_madrid: Distance in km from Madrid center (Haversine)
# - distance_from_coast: Distance in km from nearest coastline boundary (accurate geometric calculation)
#
# **NOTE:** Features calculated in `debug_distance_coastline_v2.py`
# Run that script first if `outputs/hotel_distance_features.csv` doesn't exist.

# %%
from lib.eda_utils import (
    load_distance_features,
    calculate_distance_correlations,
    plot_distance_vs_price,
    print_distance_feature_summary
)

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner

con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    exclude_reception_halls=True,
    exclude_missing_location=True,
    impute_children_allowed=True,
    impute_events_allowed=True,
    verbose=True
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)
# %%
# Load pre-calculated distance features
distance_features = load_distance_features()
print(f"Loaded distance features for {len(distance_features):,} hotels")

# %%
# Join distance features to bookings for pricing analysis
bookings_with_distances = con.execute("""
    SELECT 
        b.id,
        b.hotel_id,
        b.total_price,
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

# Merge distance features
bookings_with_distances = bookings_with_distances.merge(
    distance_features,
    on='hotel_id',
    how='left'
)

# Calculate daily price
bookings_with_distances['daily_price'] = bookings_with_distances['room_price'] / bookings_with_distances['nights']

# Filter to bookings with distance features
bookings_analysis = bookings_with_distances.dropna(subset=['distance_from_madrid', 'distance_from_coast'])
print(f"Bookings for analysis: {len(bookings_analysis):,}")

# %%
# Calculate correlations between distance features and price
corr_madrid, corr_coast = calculate_distance_correlations(bookings_analysis)

# %%
# Visualize distance vs price relationships
plot_distance_vs_price(bookings_analysis, corr_madrid, corr_coast)

# %%
# Print comprehensive summary
print_distance_feature_summary(distance_features, bookings_analysis, corr_madrid, corr_coast)

# %%

# %%
import sys
sys.path.insert(0, '../../..')
from lib.db import init_db
from lib.data_validator import validate_and_clean
from lib.eda_utils import (
    plot_seasonality_analysis,
    calculate_seasonality_stats,
    print_seasonality_summary
)
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
    exclude_missing_location_bookings=False
)

# %%
print("=" * 80)
print("SECTION 4.1: SEASONALITY IN PRICE")
print("=" * 80)

# %%
# Query to get per-night pricing with temporal features
print("\nLoading booking data with temporal features...")
seasonality_data = con.execute("""
    SELECT 
        b.id as booking_id,
        b.hotel_id,
        b.arrival_date,
        b.departure_date,
        CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
        br.total_price as room_price,
        br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price,
        br.room_type,
        hl.city,
        hl.country,
        -- Extract temporal features from arrival date
        EXTRACT(MONTH FROM CAST(b.arrival_date AS DATE)) as arrival_month,
        EXTRACT(DOW FROM CAST(b.arrival_date AS DATE)) as arrival_dow,
        EXTRACT(YEAR FROM CAST(b.arrival_date AS DATE)) as arrival_year,
        -- Extract from departure date
        EXTRACT(MONTH FROM CAST(b.departure_date AS DATE)) as departure_month,
        EXTRACT(DOW FROM CAST(b.departure_date AS DATE)) as departure_dow
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
      AND br.total_price > 0
      AND b.arrival_date IS NOT NULL
      AND b.departure_date IS NOT NULL
""").fetchdf()

print(f"Loaded {len(seasonality_data):,} bookings")
print(f"Date range: {seasonality_data['arrival_date'].min()} to {seasonality_data['arrival_date'].max()}")
print(f"Years covered: {sorted(seasonality_data['arrival_year'].unique())}")

# %%
# Basic statistics by month
print("\n" + "=" * 80)
print("PRICING BY MONTH (ARRIVAL DATE)")
print("=" * 80)

monthly_stats = seasonality_data.groupby('arrival_month')['daily_price'].agg([
    'count', 'mean', 'median', 'std',
    ('q25', lambda x: x.quantile(0.25)),
    ('q75', lambda x: x.quantile(0.75))
]).round(2)

# Add month names
month_names = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}
monthly_stats['month_name'] = monthly_stats.index.map(month_names)
monthly_stats = monthly_stats[['month_name', 'count', 'mean', 'median', 'std', 'q25', 'q75']]

print("\n" + monthly_stats.to_string())

# %%
# Statistics by day of week
print("\n" + "=" * 80)
print("PRICING BY DAY OF WEEK (ARRIVAL DATE)")
print("=" * 80)

dow_stats = seasonality_data.groupby('arrival_dow')['daily_price'].agg([
    'count', 'mean', 'median', 'std',
    ('q25', lambda x: x.quantile(0.25)),
    ('q75', lambda x: x.quantile(0.75))
]).round(2)

# Add day names (0=Sunday in DuckDB)
day_names = {
    0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed',
    4: 'Thu', 5: 'Fri', 6: 'Sat'
}
dow_stats['day_name'] = dow_stats.index.map(day_names)
dow_stats = dow_stats[['day_name', 'count', 'mean', 'median', 'std', 'q25', 'q75']]

print("\n" + dow_stats.to_string())

# %%
# Calculate seasonality metrics
seasonality_metrics = calculate_seasonality_stats(seasonality_data)

# %%
# Create comprehensive visualization
print("\nCreating visualizations...")
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "section_4_1_seasonality.png"

plot_seasonality_analysis(
    seasonality_data,
    monthly_stats,
    dow_stats,
    output_path=str(output_path)
)

print(f"Saved visualization to {output_path}")

# %%
# Print comprehensive summary
print_seasonality_summary(seasonality_data, monthly_stats, dow_stats, seasonality_metrics)

# %%
# Additional analysis: Month x Day of Week interaction
print("\n" + "=" * 80)
print("INTERACTION: MONTH x DAY OF WEEK")
print("=" * 80)

# Create heatmap data
month_dow_stats = seasonality_data.groupby(['arrival_month', 'arrival_dow'])['daily_price'].agg(['mean', 'count'])
month_dow_pivot = month_dow_stats['mean'].unstack(fill_value=np.nan)

# Only show if we have data
if not month_dow_pivot.empty:
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        month_dow_pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        center=month_dow_pivot.median().median(),
        cbar_kws={'label': 'Average Daily Price (€)'},
        ax=ax
    )
    ax.set_xlabel('Day of Week (0=Sun)')
    ax.set_ylabel('Month')
    ax.set_title('Average Daily Price by Month and Day of Week')
    
    # Add month names
    month_labels = [month_names.get(i, str(i)) for i in month_dow_pivot.index]
    ax.set_yticklabels(month_labels, rotation=0)
    
    # Add day names
    day_labels = [day_names.get(int(i), str(i)) for i in month_dow_pivot.columns]
    ax.set_xticklabels(day_labels, rotation=0)
    
    plt.tight_layout()
    heatmap_path = output_dir / "section_4_1_month_dow_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved month x day-of-week heatmap to {heatmap_path}")

# %%
print("=" * 80)
print("SECTION 4.2: POPULAR AND EXPENSIVE STAY DATES")
print("=" * 80)

# %%
from lib.eda_utils import (
    expand_bookings_to_stay_nights,
    analyze_popular_expensive_dates,
    plot_popular_expensive_analysis,
    print_popular_expensive_summary
)

# %%
# Load booking data (reuse query structure from 4.1 or create fresh)
print("\nLoading booking data for stay-night expansion...")
bookings_for_expansion = con.execute("""
    SELECT 
        b.id as booking_id,
        b.hotel_id,
        b.arrival_date,
        b.departure_date,
        CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
        br.total_price as room_price,
        br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price,
        br.room_type
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
      AND br.total_price > 0
      AND b.arrival_date IS NOT NULL
      AND b.departure_date IS NOT NULL
""").fetchdf()

print(f"Loaded {len(bookings_for_expansion):,} bookings")
print(f"Total room-nights: {bookings_for_expansion['nights'].sum():,.0f}")

# %%
# Expand bookings to per-night level
print("\nExpanding bookings to per-night level...")
print("(This may take a moment for large datasets)")
stay_nights = expand_bookings_to_stay_nights(bookings_for_expansion)
print(f"Expanded to {len(stay_nights):,} stay-night records")
print(f"Date range: {stay_nights['stay_date'].min()} to {stay_nights['stay_date'].max()}")

# %%
# Analyze most popular and expensive dates
most_popular, most_expensive, daily_stats = analyze_popular_expensive_dates(
    stay_nights,
    top_n=20
)

# %%
# Print summary
print_popular_expensive_summary(most_popular, most_expensive, daily_stats)

# %%
# Create visualization
print("\nCreating visualizations...")
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "section_4_2_popular_expensive.png"
plot_popular_expensive_analysis(daily_stats, most_popular, most_expensive, str(output_path))
print(f"Saved visualization to {output_path}")

# %%
print("=" * 80)
print("SECTION 4.3: BOOKING COUNTS BY ARRIVAL DATE")
print("=" * 80)

# %%
from lib.eda_utils import (
    analyze_booking_counts_by_arrival,
    fit_prophet_model,
    analyze_seasonal_patterns,
    plot_booking_counts_analysis,
    print_booking_counts_summary
)

# %%
# Load booking-level data (not expanded by nights)
print("\nLoading booking data...")
bookings_arrival = con.execute("""
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
""").fetchdf()

print(f"Loaded {len(bookings_arrival):,} bookings")
print(f"Date range: {bookings_arrival['arrival_date'].min()} to {bookings_arrival['arrival_date'].max()}")

# %%
# Analyze at different aggregation levels
print("\n" + "=" * 80)
print("DAILY LEVEL ANALYSIS")
print("=" * 80)

daily_stats = analyze_booking_counts_by_arrival(bookings_arrival, aggregate_by='day')
daily_prophet = fit_prophet_model(daily_stats, y_col='num_bookings')
daily_seasonal = analyze_seasonal_patterns(daily_stats)

print_booking_counts_summary(daily_stats, daily_prophet, daily_seasonal, aggregate_by='day')

# %%
# Weekly analysis
print("\n" + "=" * 80)
print("WEEKLY LEVEL ANALYSIS")
print("=" * 80)

weekly_stats = analyze_booking_counts_by_arrival(bookings_arrival, aggregate_by='week')
weekly_prophet = fit_prophet_model(weekly_stats, y_col='num_bookings')
weekly_seasonal = analyze_seasonal_patterns(weekly_stats)

print_booking_counts_summary(weekly_stats, weekly_prophet, weekly_seasonal, aggregate_by='week')

# %%
# Monthly analysis
print("\n" + "=" * 80)
print("MONTHLY LEVEL ANALYSIS")
print("=" * 80)

monthly_stats = analyze_booking_counts_by_arrival(bookings_arrival, aggregate_by='month')
monthly_prophet = fit_prophet_model(monthly_stats, y_col='num_bookings')
monthly_seasonal = analyze_seasonal_patterns(monthly_stats)

print_booking_counts_summary(monthly_stats, monthly_prophet, monthly_seasonal, aggregate_by='month')

# %%
# Create visualizations for each level
print("\nCreating visualizations...")

# Daily
output_path_daily = Path("outputs/figures/section_4_3_bookings_daily.png")
plot_booking_counts_analysis(
    daily_stats, daily_prophet, daily_seasonal, 
    aggregate_by='day', output_path=str(output_path_daily)
)

# Weekly
output_path_weekly = Path("outputs/figures/section_4_3_bookings_weekly.png")
plot_booking_counts_analysis(
    weekly_stats, weekly_prophet, weekly_seasonal,
    aggregate_by='week', output_path=str(output_path_weekly)
)

# Monthly
output_path_monthly = Path("outputs/figures/section_4_3_bookings_monthly.png")
plot_booking_counts_analysis(
    monthly_stats, monthly_prophet, monthly_seasonal,
    aggregate_by='month', output_path=str(output_path_monthly)
)

print(f"\nSaved visualizations:")
print(f"  - Daily: {output_path_daily}")
print(f"  - Weekly: {output_path_weekly}")
print(f"  - Monthly: {output_path_monthly}")

# %%
# Summary of key findings
print("\n" + "=" * 80)
print("SECTION 4.3: KEY FINDINGS")
print("=" * 80)
print("""
Prophet model reveals:
- R² = 0.71 (explains 71% of variance vs 0.03 with linear regression)
- Weekly trend: +20% GROWTH (most reliable indicator)
- Strong seasonality: Peak May (1,523/day), Trough Nov (480/day)
- 3.2x variation Q2 vs Q4 → Dynamic pricing opportunity
- Year-over-year consistency confirms healthy, predictable business
""")

# %%
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
con_raw = init_db()
config = CleaningConfig(
    remove_negative_prices=True,
    remove_zero_prices=True,
    remove_null_prices=True,
    remove_negative_stay=True,
    remove_null_dates=True,
    fix_empty_strings=True,
    verbose=True
)
cleaner = DataCleaner(config)
con = cleaner.clean(con_raw)

query = """
select arrival_date, count(*) as cnt from bookings
group by arrival_date
order by arrival_date;
"""
times = con.execute(query).fetchdf()

# %%
print("=" * 80)
print("SECTION 5.1: LEAD TIME DISTRIBUTION AND PRICE")
print("=" * 80)

# %%
from lib.eda_utils import (
    analyze_lead_time_distribution,
    plot_lead_time_analysis,
    print_lead_time_summary
)

# %%
# Load booking data with lead time calculation
print("\nLoading booking data and calculating lead times...")
bookings_lead_time = con.execute("""
    SELECT 
        b.id as booking_id,
        b.arrival_date,
        b.departure_date,
        b.created_at,
        b.total_price,
        CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE) as nights,
        br.total_price as room_price,
        br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as daily_price,
        br.room_type,
        -- Calculate lead time in days
        DATE_DIFF('day', CAST(b.created_at AS DATE), CAST(b.arrival_date AS DATE)) as lead_time_days
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    WHERE b.status IN ('confirmed', 'Booked')
      AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
      AND br.total_price > 0
      AND b.arrival_date IS NOT NULL
      AND b.created_at IS NOT NULL
      -- Filter to non-negative lead times (created before arrival)
      AND DATE_DIFF('day', CAST(b.created_at AS DATE), CAST(b.arrival_date AS DATE)) >= 0
""").fetchdf()

print(f"Loaded {len(bookings_lead_time):,} bookings with valid lead times")
print(f"Lead time range: {bookings_lead_time['lead_time_days'].min():.0f} to {bookings_lead_time['lead_time_days'].max():.0f} days")

# %%
# Analyze lead time distribution
print("\nAnalyzing lead time distribution by buckets...")
lead_time_stats = analyze_lead_time_distribution(
    bookings_lead_time,
    buckets=[0, 1, 7, 30, 60, 90, 180, 365, float('inf')]
)

# %%
# Print summary
print_lead_time_summary(lead_time_stats, bookings_lead_time)

# %%
# Create visualization
print("\nCreating visualizations...")
output_path = Path("outputs/figures/section_5_1_lead_time.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plot_lead_time_analysis(lead_time_stats, bookings_lead_time, str(output_path))
print(f"Saved visualization to {output_path}")

# %%

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
from lib.data_validator import validate_and_clean
from lib.eda_utils import (
    analyze_same_day_bookings,
    identify_underpricing_opportunities,
    plot_occupancy_pricing_analysis,
    print_occupancy_pricing_summary
)
import pandas as pd
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


# %%
