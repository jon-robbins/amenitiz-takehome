# %%
"""
Matched Pair Analysis - WITH VALIDATED FEATURES

Integrates the validated feature engineering from feature_importance_validation.py
into the geographic matching methodology.

Key Updates:
1. Uses validated feature set (17 features from XGBoost validation)
2. Applies same preprocessing (log transforms, cyclical encoding, etc.)
3. Maintains geographic matching methodology (coastal/inland)
4. Enhanced matching on validated features only
"""

# %%
import sys
sys.path.insert(0, '../../../..')

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from matplotlib.ticker import FuncFormatter
import re

# %%


def load_hotel_month_data(con) -> pd.DataFrame:
    """Loads hotel-month aggregation with all features needed for validation."""
    query = """
    WITH hotel_month_room AS (
        SELECT 
            b.hotel_id,
            DATE_TRUNC('month', CAST(b.arrival_date AS DATE)) AS month,
            br.room_type,
            COALESCE(NULLIF(br.room_view, ''), 'no_view') AS room_view,
            r.children_allowed,
            hl.city,
            hl.latitude,
            hl.longitude,
            
            -- Revenue metrics
            SUM(br.total_price) AS total_revenue,
            COUNT(*) AS room_nights_sold,
            SUM(br.total_price) / NULLIF(COUNT(*), 0) AS avg_adr,
            
            -- Room features
            AVG(br.room_size) AS avg_room_size,
            SUM(r.number_of_rooms) AS total_capacity,
            MAX(r.max_occupancy) AS room_capacity_pax,
            
            -- Temporal features
            EXTRACT(MONTH FROM MAX(CAST(b.arrival_date AS DATE))) AS month_number,
            EXTRACT(DAY FROM LAST_DAY(MAX(CAST(b.arrival_date AS DATE)))) AS days_in_month,
            SUM(CASE WHEN EXTRACT(ISODOW FROM CAST(b.arrival_date AS DATE)) >= 6 THEN 1 ELSE 0 END)::FLOAT / 
                NULLIF(COUNT(*), 0) AS weekend_ratio,
            
            -- Amenities score (0-4)
            (CAST(MAX(r.events_allowed) AS INT) + 
             CAST(MAX(r.pets_allowed) AS INT) + 
             CAST(MAX(r.smoking_allowed) AS INT) + 
             CAST(MAX(r.children_allowed) AS INT)) AS amenities_score,
            
            -- View quality (ordinal 0-3)
            CASE 
                WHEN COALESCE(NULLIF(br.room_view, ''), 'no_view') IN ('ocean_view', 'sea_view') THEN 3
                WHEN COALESCE(NULLIF(br.room_view, ''), 'no_view') IN ('lake_view', 'mountain_view') THEN 2
                WHEN COALESCE(NULLIF(br.room_view, ''), 'no_view') IN ('pool_view', 'garden_view') THEN 1
                ELSE 0
            END AS view_quality_ordinal
            
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        JOIN rooms r ON br.room_id = r.id
        WHERE b.status IN ('confirmed', 'Booked')
          AND CAST(b.arrival_date AS DATE) BETWEEN '2023-01-01' AND '2024-12-31'
          AND hl.city IS NOT NULL
        GROUP BY b.hotel_id, month, br.room_type, room_view, 
                 r.children_allowed, hl.city, hl.latitude, hl.longitude
    )
    SELECT 
        *,
        (room_nights_sold::FLOAT / NULLIF(total_capacity * days_in_month, 0)) AS occupancy_rate
    FROM hotel_month_room
    WHERE total_capacity > 0 AND room_nights_sold > 0 AND avg_adr > 0
    """
    return con.execute(query).fetchdf()


def engineer_validated_features(df: pd.DataFrame, distance_features: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the VALIDATED feature set from feature_importance_validation.py.
    
    This uses the exact same feature engineering that achieved R² = 0.71.
    """
    df = df.copy()
    
    # Merge distance features
    df = df.merge(distance_features, on='hotel_id', how='left')
    
    # Top 5 cities by revenue with canonical names
    top_5_canonical = {
        'madrid': 'madrid',
        'barcelona': 'barcelona',
        'sevilla': 'sevilla',
        'malaga': 'malaga',
        'málaga': 'malaga',
        'toledo': 'toledo'
    }
    
    def clean_city_name(name):
        if pd.isna(name):
            return ''
        cleaned = re.sub(r'[^\w\s]', '', str(name).lower().strip())
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned
    
    def standardize_city(city_str):
        if pd.isna(city_str):
            return 'other'
        
        city_clean = clean_city_name(city_str)
        
        if city_clean in top_5_canonical:
            return top_5_canonical[city_clean]
        
        for canonical_key in top_5_canonical.keys():
            if canonical_key in city_clean:
                return top_5_canonical[canonical_key]
        
        return 'other'
    
    df['city_standardized'] = df['city'].apply(standardize_city)
    
    # Calculate city centroids (booking-weighted mean) using standardized cities
    city_centroids = df.groupby('city_standardized').apply(
        lambda x: pd.Series({
            'city_lat': np.average(x['latitude'], weights=x['room_nights_sold']),
            'city_lon': np.average(x['longitude'], weights=x['room_nights_sold'])
        }), include_groups=False
    ).reset_index()
    
    df = df.merge(city_centroids, on='city_standardized', how='left')
    
    # Geographic features
    df['dist_center_km'] = np.sqrt(
        (df['latitude'] - df['city_lat'])**2 + 
        (df['longitude'] - df['city_lon'])**2
    ) * 111  # Rough conversion to km
    
    df['is_coastal'] = (df['distance_from_coast'] < 20).astype(int)
    df['dist_coast_log'] = np.log1p(df['distance_from_coast'])
    
    # Product features
    df['log_room_size'] = np.log1p(df['avg_room_size'])
    df['total_capacity_log'] = np.log1p(df['total_capacity'])
    
    # Temporal features (cyclical encoding)
    df['month_sin'] = np.sin(2 * np.pi * df['month_number'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_number'] / 12)
    df['is_summer'] = df['month_number'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month_number'].isin([12, 1, 2]).astype(int)
    
    # Fill NaN values in city centroids
    df['city_lat'] = df['city_lat'].fillna(df['latitude'])
    df['city_lon'] = df['city_lon'].fillna(df['longitude'])
    df['dist_center_km'] = df['dist_center_km'].fillna(0)
    
    return df


def add_revenue_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Adds annual revenue quartile bins to dataframe."""
    hotel_annual_revenue = df.groupby('hotel_id')['total_revenue'].sum().reset_index()
    hotel_annual_revenue.columns = ['hotel_id', 'annual_revenue']
    hotel_annual_revenue['revenue_quartile'] = pd.qcut(
        hotel_annual_revenue['annual_revenue'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    return df.merge(hotel_annual_revenue[['hotel_id', 'revenue_quartile']], on='hotel_id', how='left')


def create_match_blocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates exact match blocks for geographic matching.
    
    Uses validated categorical features:
    - is_coastal (geographic market)
    - room_type (product category)
    - room_view (product quality)
    - month (seasonality)
    - children_allowed (market segment)
    - revenue_quartile (business scale)
    - city_standardized (top 5 cities + other)
    """
    df = df.copy()
    df['block_id'] = df.groupby([
        'is_coastal',
        'room_type',
        'room_view',
        'month',
        'children_allowed',
        'revenue_quartile',
        'city_standardized'
    ], observed=True).ngroup()
    
    block_hotel_counts = df.groupby('block_id', observed=True)['hotel_id'].nunique()
    valid_blocks = block_hotel_counts[block_hotel_counts >= 2].index
    df_filtered = df[df['block_id'].isin(valid_blocks)]
    
    return df, df_filtered


def find_matched_pairs(df_blocked: pd.DataFrame, max_block_size: int = 100) -> pd.DataFrame:
    """
    Finds matched pairs within blocks using KNN on VALIDATED continuous features.
    
    Matching features (from validated model):
    - dist_center_km
    - dist_coast_log  
    - log_room_size
    - room_capacity_pax
    - amenities_score
    - total_capacity_log
    - view_quality_ordinal
    - weekend_ratio
    """
    # Validated continuous matching features
    match_features = [
        'dist_center_km',
        'dist_coast_log',
        'log_room_size',
        'room_capacity_pax',
        'amenities_score',
        'total_capacity_log',
        'view_quality_ordinal',
        'weekend_ratio'
    ]
    
    matched_pairs = []
    blocks_processed = 0
    
    for block_vars, block in df_blocked.groupby([
        'is_coastal', 'room_type', 'room_view', 'month', 
        'children_allowed', 'revenue_quartile', 'city_standardized'
    ]):
        if len(block) < 2 or len(block) > max_block_size:
            continue
        
        blocks_processed += 1
        
        # Check all required features exist
        if not all(f in block.columns for f in match_features):
            continue
        
        # Prepare feature matrix
        block_features = block[match_features].fillna(0)
        
        try:
            features_norm = StandardScaler().fit_transform(block_features)
        except:
            continue
        
        prices = block['avg_adr'].values
        occ = block['occupancy_rate'].values
        ids = block['hotel_id'].values
        n = len(block)
        
        dist_matrix = cdist(features_norm, features_norm, metric='euclidean')
        price_matrix = np.abs(prices[:, None] - prices[None, :]) / np.minimum(prices[:, None], prices[None, :])
        
        # Match threshold
        for i in range(n):
            for j in range(i+1, n):
                if ids[i] == ids[j] or price_matrix[i, j] < 0.10 or dist_matrix[i, j] > 3.0:
                    continue
                
                high_idx = i if prices[i] > prices[j] else j
                low_idx = j if prices[i] > prices[j] else i
                
                matched_pairs.append({
                    'is_coastal': block.iloc[i]['is_coastal'],
                    'city_standardized': block.iloc[i]['city_standardized'],
                    'room_type': block.iloc[i]['room_type'],
                    'room_view': block.iloc[i]['room_view'],
                    'high_price_hotel': ids[high_idx],
                    'low_price_hotel': ids[low_idx],
                    'high_price': prices[high_idx],
                    'low_price': prices[low_idx],
                    'high_occupancy': occ[high_idx],
                    'low_occupancy': occ[low_idx],
                    'price_diff_pct': price_matrix[i, j],
                    'match_distance': dist_matrix[i, j],
                    'month': str(block.iloc[i]['month']),
                    'capacity': block.iloc[low_idx]['total_capacity'],
                    'days_in_month': block.iloc[low_idx]['days_in_month'],
                    'dist_coast_diff': abs(block.iloc[i]['distance_from_coast'] - block.iloc[j]['distance_from_coast'])
                })
        
        if blocks_processed % 100 == 0:
            print(f"  Processed {blocks_processed} blocks, found {len(matched_pairs)} pairs...")
    
    return pd.DataFrame(matched_pairs)


def calculate_elasticity_and_opportunity(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates arc elasticity and counterfactual revenue opportunity."""
    df = pairs_df.copy()
    
    # Arc elasticity (midpoint method)
    df['price_avg'] = (df['high_price'] + df['low_price']) / 2
    df['occ_avg'] = (df['high_occupancy'] + df['low_occupancy']) / 2
    df['price_pct_change'] = (df['high_price'] - df['low_price']) / df['price_avg']
    df['occ_pct_change'] = (df['high_occupancy'] - df['low_occupancy']) / df['occ_avg']
    df['arc_elasticity'] = df['occ_pct_change'] / df['price_pct_change']
    
    # Filter valid pairs
    df_valid = df[
        (df['arc_elasticity'] < 0) &
        (df['arc_elasticity'] > -5) &
        (df['occ_avg'] > 0.01) &
        (df['match_distance'] < 3.0)
    ]
    
    # Counterfactual opportunity
    df_valid['current_revenue'] = (
        df_valid['low_price'] * df_valid['low_occupancy'] * 
        df_valid['capacity'] * df_valid['days_in_month']
    )
    df_valid['counterfactual_occ'] = (
        df_valid['low_occupancy'] * (1 + df_valid['arc_elasticity'] * df_valid['price_pct_change'])
    ).clip(0, 1.0)
    df_valid['counterfactual_revenue'] = (
        df_valid['high_price'] * df_valid['counterfactual_occ'] * 
        df_valid['capacity'] * df_valid['days_in_month']
    )
    df_valid['opportunity'] = df_valid['counterfactual_revenue'] - df_valid['current_revenue']
    
    return df_valid[df_valid['opportunity'] > 0]


def print_results(opp_positive: pd.DataFrame) -> None:
    """Prints comprehensive results summary."""
    print("\n" + "=" * 80)
    print("VALIDATED MATCHED PAIRS RESULTS")
    print("=" * 80)
    
    print(f"\n1. SAMPLE SIZE:")
    print(f"   Pairs with positive opportunity: {len(opp_positive):,}")
    print(f"   Unique low-price hotels: {opp_positive['low_price_hotel'].nunique():,}")
    
    print(f"\n2. ELASTICITY ESTIMATE:")
    print(f"   Median: {opp_positive['arc_elasticity'].median():.4f}")
    print(f"   Mean: {opp_positive['arc_elasticity'].mean():.4f}")
    print(f"   Std: {opp_positive['arc_elasticity'].std():.4f}")
    print(f"   95% CI: [{opp_positive['arc_elasticity'].quantile(0.025):.4f}, {opp_positive['arc_elasticity'].quantile(0.975):.4f}]")
    
    print(f"\n3. MATCH QUALITY:")
    print(f"   Avg match distance (normalized): {opp_positive['match_distance'].mean():.3f}")
    print(f"   Avg coast distance difference: {opp_positive['dist_coast_diff'].mean():.1f} km")
    
    print(f"\n4. OPPORTUNITY SIZING:")
    print(f"   Total opportunity: €{opp_positive['opportunity'].sum():,.0f}")
    print(f"   Average per hotel-month: €{opp_positive['opportunity'].mean():,.0f}")
    
    print(f"\n5. BY MARKET SEGMENT:")
    segment_stats = opp_positive.groupby('is_coastal').agg({
        'arc_elasticity': 'median',
        'opportunity': 'sum',
        'low_price_hotel': 'count'
    })
    segment_stats.index = ['Inland', 'Coastal']
    segment_stats.columns = ['median_elasticity', 'total_opportunity', 'n_pairs']
    print(segment_stats.to_string())
    
    print(f"\n6. BY CITY:")
    city_stats = opp_positive.groupby('city_standardized').agg({
        'arc_elasticity': 'median',
        'opportunity': 'sum',
        'low_price_hotel': 'count'
    })
    city_stats.columns = ['median_elasticity', 'total_opportunity', 'n_pairs']
    city_stats = city_stats.sort_values('n_pairs', ascending=False)
    print(city_stats.to_string())


# %%
# Main execution
print("=" * 80)
print("VALIDATED MATCHED PAIRS ANALYSIS")
print("Using validated features from feature_importance_validation.py")
print("=" * 80)

print("\nLoading database...")
config = CleaningConfig(
    exclude_reception_halls=True,
    exclude_missing_location=True,
    match_city_names_with_tfidf=True
)
cleaner = DataCleaner(config)
# Initialize database
con = init_db()

# Clean data
con = cleaner.clean(con)

# %%
print("\nLoading distance features...")
script_dir = Path(__file__).parent
distance_features_path = script_dir / '../../../outputs/eda/spatial/data/hotel_distance_features.csv'
distance_features = pd.read_csv(distance_features_path.resolve())
print(f"Loaded distance features for {len(distance_features):,} hotels")

# %%
print("\nCreating hotel-month aggregation...")
df_geo = load_hotel_month_data(con)
print(f"Hotel-months: {len(df_geo):,}")

# %%
print("\nEngineering VALIDATED features...")
df_geo = engineer_validated_features(df_geo, distance_features)
df_geo = df_geo.dropna(subset=['distance_from_coast'])
print(f"After feature engineering: {len(df_geo):,} hotel-months")

# %%
print("\nCalculating annual revenue quartiles...")
df_geo = add_revenue_quartiles(df_geo)
print(f"Revenue quartile distribution:")
print(df_geo['revenue_quartile'].value_counts().sort_index())

# %%
print("\n" + "=" * 80)
print("CREATING MATCH BLOCKS WITH VALIDATED FEATURES")
print("=" * 80)

print("\nExact matching variables:")
print("  - is_coastal (geographic market)")
print("  - room_type (product category)")
print("  - room_view (product quality)")
print("  - month (seasonality)")
print("  - children_allowed (market segment)")
print("  - revenue_quartile (business scale)")
print("  - city_standardized (top 5 cities + other)")

print("\nContinuous matching features (KNN):")
print("  - dist_center_km, dist_coast_log")
print("  - log_room_size, room_capacity_pax")
print("  - amenities_score, total_capacity_log")
print("  - view_quality_ordinal, weekend_ratio")

df_geo_with_blocks, df_blocked = create_match_blocks(df_geo)

block_hotel_counts = df_geo_with_blocks.groupby('block_id', observed=True)['hotel_id'].nunique()
valid_blocks = block_hotel_counts[block_hotel_counts >= 2].index

print(f"\nBlocking results:")
print(f"  Total blocks created: {df_geo_with_blocks['block_id'].nunique():,}")
print(f"  Blocks with ≥2 hotels: {len(valid_blocks):,}")
print(f"  Hotel-months retained: {len(df_blocked):,}")
print(f"  Avg hotels per block: {block_hotel_counts[valid_blocks].mean():.1f}")

# %%
print("\nFinding matched pairs...")
pairs_geo = find_matched_pairs(df_blocked)
print(f"\nMatched pairs found: {len(pairs_geo):,}")

# %%
print("\nCalculating arc elasticity and opportunity...")
opp_positive = calculate_elasticity_and_opportunity(pairs_geo)
print(f"Valid pairs with positive opportunity: {len(opp_positive):,}")

# %%
print_results(opp_positive)

# %%
pairs_path = (script_dir / '../../../outputs/eda/elasticity/data/matched_pairs_validated.csv').resolve()
pairs_path.parent.mkdir(parents=True, exist_ok=True)
opp_positive.to_csv(pairs_path, index=False)
print(f"\n✓ Saved results to: {pairs_path}")

print("\n" + "=" * 80)
print("✓ VALIDATED MATCHED PAIRS ANALYSIS COMPLETE")
print("=" * 80)
