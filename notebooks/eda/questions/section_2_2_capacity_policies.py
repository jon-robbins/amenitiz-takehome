# 2.2 Capacity and policy flags
# Analyze capacity (max_occupancy, max_adults) and policy flags

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

