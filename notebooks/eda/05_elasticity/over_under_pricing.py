# %%
import sys
sys.path.insert(0, '../../..')
from lib.db import init_db
from lib.sql_loader import load_sql_file
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
# Initialize database
con = init_db()

# Clean data
config = CleaningConfig(match_city_names_with_tfidf=True)
cleaner = DataCleaner(config)
con = cleaner.clean(con)

# Load distance features
root_dir = Path(__file__).parent.parent.parent.parent
distance_features = pd.read_csv(root_dir / 'outputs' / 'eda' / 'spatial' / 'data' / 'hotel_distance_features.csv')

# 2. Create Hotel-Month Aggregation (With RevPAR)
print("Creating aggregation...")
# Load SQL query from file
query = load_sql_file('QUERY_LOAD_HOTEL_MONTH_REVPAR.sql', __file__)

# Execute query
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
            
            # Identify Premium vs Discount
            high_idx, low_idx = (i, j) if prices[i] > prices[j] else (j, i)
            
            # THE KEY METRIC: Did Premium Pricing = Higher RevPAR?
            # Calculate Relative Lift (%)
            revpar_impact = revpars[high_idx] - revpars[low_idx]
            revpar_lift_pct = (revpar_impact / revpars[low_idx]) * 100
            
            is_winner = revpar_impact > 0
            
            matched_pairs.append({
                'high_price': prices[high_idx],
                'low_price': prices[low_idx],
                'price_premium_pct': price_diff,
                'revpar_impact': revpar_impact,
                'revpar_lift_pct': revpar_lift_pct,
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
plt.title("Premium Strategy Success Rate: RevPAR Gain Probability")
plt.ylabel("Probability that Premium Strategy yielded higher RevPAR")
plt.xlabel("Price Premium vs. Twin")
plt.ylim(0, 1.0)

# Add relative lift annotation
median_lift = results.groupby('premium_bin')['revpar_lift_pct'].median()
for i, lift in enumerate(median_lift):
    plt.text(i, 0.1, f"Median Lift:\n+{lift:.0f}%", ha='center', color='white', fontweight='bold')

plt.legend()
plt.savefig(root_dir / 'outputs' / 'eda' / 'elasticity' / 'figures' / 'revpar_tipping_point.png')
print("Saved plot to outputs/figures/revpar_tipping_point.png")