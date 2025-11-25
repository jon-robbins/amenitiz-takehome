# %%
import sys
sys.path.insert(0, '../../..')
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import seaborn as sns
# 1. Setup
print("Loading database...")
config = CleaningConfig(match_city_names_with_tfidf=True, verbose=False)
cleaner = DataCleaner(config)
con = cleaner.clean(init_db())

# Load distance features
root_dir = Path(__file__).parent.parent.parent.parent
distance_features = pd.read_csv(root_dir / 'outputs' / 'eda' / 'spatial' / 'data' / 'hotel_distance_features.csv')

# 2. Create Hotel-Month Aggregation (With RevPAR)
print("Creating aggregation...")
query = """
WITH hotel_month AS (
    SELECT 
        b.hotel_id,
        DATE_TRUNC('month', CAST(b.arrival_date AS DATE)) AS month,
        br.room_type,
        r.children_allowed,
        hl.city,
        -- Metrics
        SUM(br.total_price) AS total_revenue,
        COUNT(*) AS room_nights_sold,
        SUM(br.total_price) / NULLIF(COUNT(*), 0) AS avg_adr,
        AVG(br.room_size) AS avg_room_size,
        SUM(r.number_of_rooms) AS total_capacity,
        MAX(r.max_occupancy) AS room_capacity_pax,
        -- Controls
        SUM(CASE WHEN EXTRACT(ISODOW FROM CAST(b.arrival_date AS DATE)) >= 6 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS weekend_ratio
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    JOIN rooms r ON br.room_id = r.id
    WHERE b.status = 'confirmed'
    GROUP BY 1, 2, 3, 4, 5
)
SELECT 
    *,
    (total_revenue / NULLIF(total_capacity * 30, 0)) AS revpar, -- Approx days
    (room_nights_sold::FLOAT / NULLIF(total_capacity * 30, 0)) AS occupancy_rate
FROM hotel_month
WHERE total_capacity > 0 AND room_nights_sold > 0
"""
df = con.execute(query).fetchdf()
df = df.merge(distance_features, on='hotel_id', how='inner')

# Add Geographic Segments
df['market_segment'] = (df['distance_from_coast'] <= 20).astype(str) + '_' + (df['distance_from_madrid'] <= 50).astype(str)

# 3. Matching Logic (Identifying Winners AND Losers)
print("Matching pairs to identify Overpricing...")

# Exact Blocks
df['block_id'] = df.groupby(['market_segment', 'room_type', 'month', 'children_allowed']).ngroup()
valid_blocks = df.groupby('block_id')['hotel_id'].nunique()
df_blocked = df[df['block_id'].isin(valid_blocks[valid_blocks >= 2].index)]

matched_pairs = []

for _, block in df_blocked.groupby('block_id'):
    if len(block) < 2 or len(block) > 100: continue
    
    # Normalize continuous features
    features = ['avg_room_size', 'total_capacity', 'weekend_ratio']
    scaler = StandardScaler()
    try:
        feat_norm = scaler.fit_transform(block[features].fillna(0))
    except: continue
    
    prices = block['avg_adr'].values
    revpars = block['revpar'].values
    ids = block['hotel_id'].values
    
    dist_matrix = cdist(feat_norm, feat_norm)
    
    for i in range(len(block)):
        for j in range(i+1, len(block)):
            # Filter for identical twins
            if dist_matrix[i,j] > 1.5: continue
            
            # Filter for price difference (>10%)
            price_diff = abs(prices[i] - prices[j]) / min(prices[i], prices[j])
            if price_diff < 0.10: continue
            
            # Identify High vs Low
            high_idx, low_idx = (i, j) if prices[i] > prices[j] else (j, i)
            
            # THE KEY METRIC: Did High Price = High RevPAR?
            revpar_impact = revpars[high_idx] - revpars[low_idx]
            is_winner = revpar_impact > 0
            
            matched_pairs.append({
                'high_price': prices[high_idx],
                'low_price': prices[low_idx],
                'price_premium_pct': price_diff,
                'revpar_impact': revpar_impact,
                'is_winner': is_winner,
                'market_segment': block.iloc[i]['market_segment']
            })

results = pd.DataFrame(matched_pairs)

# 4. Analysis & Visualization
print("\n" + "="*60)
print("THE REVPAR OPTIMIZATION CURVE")
print("="*60)

# Bin by Price Premium to see where the "Tipping Point" is
results['premium_bin'] = pd.cut(results['price_premium_pct'], bins=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0], labels=['10-20%', '20-30%', '30-40%', '40-50%', '50%+'])
summary = results.groupby('premium_bin')['is_winner'].mean()

print("\nProbability of RevPAR Gain by Price Hike:")
print(summary)

print("\nInterpretation:")
print("- If 'is_winner' drops below 0.5, it means raising prices hurts RevPAR more often than it helps.")
print("- This threshold is our 'Safety Cap' for the algorithm.")

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x=summary.index, y=summary.values, color='steelblue')
plt.axhline(0.5, color='red', linestyle='--', label='Breakeven Probability')
plt.title("Risk of Overpricing: Success Rate of Price Hikes")
plt.ylabel("Probability that High Price Twin made MORE Money")
plt.xlabel("Price Premium vs. Twin")
plt.legend()
plt.savefig(root_dir / 'outputs' / 'eda' / 'elasticity' / 'figures' / 'revpar_tipping_point.png')
print("Saved plot to outputs/figures/revpar_tipping_point.png")