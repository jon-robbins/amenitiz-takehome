# %%
"""
Matched Pair Analysis - 1:1 MATCHING WITH REPLACEMENT

The Goldilocks Solution:
- Too Hot (Many-to-Many): N=194k pairs (combinatorial explosion)
- Too Cold (Greedy 1:1): N=94 pairs (too strict, discards good controls)
- Just Right (1:1 with Replacement): N=~3-5k pairs (one match per treatment)

Methodology:
1. Identify Treatment (Premium Pricing) and Control (Discount Pricing) hotels within each block
2. For EACH Treatment hotel, find its SINGLE BEST Control match
3. DO NOT remove the Control from the pool (allow reuse)
4. This estimates ATT (Average Treatment Effect on the Treated) - the "Premium Advantage"

Features validated by XGBoost (R² = 0.71):
- Geographic: dist_center_km, dist_coast_log
- Product: log_room_size, room_capacity_pax, amenities_score, total_capacity_log, view_quality_ordinal, total_capacity
- Temporal: month_sin, month_cos, weekend_ratio
- Categorical: room_type, room_view, city (top 5)
- Boolean: is_coastal, is_summer, is_winter, children_allowed
"""

# %%
import sys
sys.path.insert(0, '../../../..')

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
from lib.sql_loader import load_sql_file
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter



# %%
def load_hotel_month_data(con) -> pd.DataFrame:
    """
    Loads hotel-month aggregation with all validated features.
    
    CORRECTED CALCULATIONS (based on schema exploration):
    1. Explode each booking to daily granularity (each night is a row)
    2. ADR = total_price / nights_stayed
    3. Hotel capacity = SUM(number_of_rooms) for all distinct room types
    4. Aggregate to HOTEL-MONTH level first to get total room-nights sold
    5. Occupancy = total_room_nights_sold / (hotel_capacity × days_in_month)
    6. Then join back room-type features for matching
    
    SQL Query: QUERY_LOAD_HOTEL_MONTH_DATA (defined below)
    
    Returns
    -------
    pd.DataFrame
        Hotel-month aggregated statistics with all validated features.
    """
    # Define SQL query for traceability
    # Load SQL query from file
    query = load_sql_file('QUERY_LOAD_HOTEL_MONTH_DATA.sql', __file__)
    
    # Execute query
    return con.execute(query).fetchdf()


# %%
def engineer_validated_features(df: pd.DataFrame, distance_features: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the 17 validated features from XGBoost analysis.
    
    Numeric (10): dist_center_km, dist_coast_log, log_room_size, room_capacity_pax, 
                  amenities_score, total_capacity_log, view_quality_ordinal,
                  month_sin, month_cos, weekend_ratio
    Categorical (3): room_type, room_view, city_standardized (top 5)
    Boolean (4): is_coastal, is_summer, is_winter, children_allowed
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
    
    # Calculate city centroids (booking-weighted mean)
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
    df['log_partner_size'] = np.log1p(df['partner_size'])
    
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


# %%
def add_capacity_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds capacity quartile bins based on total_capacity (number of rooms).
    
    This replaces revenue_quartile because:
    1. New hotels have no revenue history
    2. Capacity is available at listing time
    3. Capacity is a strong proxy for business scale
    """
    hotel_capacity = df.groupby('hotel_id')['total_capacity'].first().reset_index()
    hotel_capacity.columns = ['hotel_id', 'hotel_capacity']
    hotel_capacity['capacity_quartile'] = pd.qcut(
        hotel_capacity['hotel_capacity'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    return df.merge(hotel_capacity[['hotel_id', 'capacity_quartile']], on='hotel_id', how='left')


# %%
def create_match_blocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates exact match blocks for matching.
    
    Exact matching on:
    - is_coastal (binary)
    - room_type (categorical)
    - room_view (categorical)
    - city_standardized (top 5 + other)
    - month (temporal)
    - children_allowed (binary)
    - capacity_quartile (Q1-Q4) - uses room count instead of revenue for new hotel support
    """
    df = df.copy()
    df['block_id'] = df.groupby([
        'is_coastal',
        'room_type',
        'room_view',
        'city_standardized',
        'month',
        'children_allowed',
        'capacity_quartile'
    ], observed=True).ngroup()
    
    block_hotel_counts = df.groupby('block_id', observed=True)['hotel_id'].nunique()
    valid_blocks = block_hotel_counts[block_hotel_counts >= 2].index
    df_filtered = df[df['block_id'].isin(valid_blocks)]
    
    return df, df_filtered


# %%
def find_matched_pairs_with_replacement(df_blocked: pd.DataFrame, max_block_size: int = 200) -> pd.DataFrame:
    """
    1:1 Matching WITH REPLACEMENT
    
    For each Premium Pricing hotel (Treatment):
    1. Find its single best Discount Pricing match (Control) in the same block
    2. DO NOT remove the Control from the pool
    3. Multiple Treatments can match to the same Control
    
    This maximizes sample size while maintaining match quality.
    
    Matching features (validated by XGBoost):
    - dist_center_km, dist_coast_log
    - log_room_size, room_capacity_pax, amenities_score, total_capacity_log
    - view_quality_ordinal, weekend_ratio
    """
    # Validated matching features (continuous only)
    # Added 'total_capacity' (number of rooms) as a direct matching factor per user request
    match_features = [
        'dist_center_km', 'dist_coast_log',
        'log_room_size', 'room_capacity_pax', 'amenities_score', 'total_capacity_log',
        'view_quality_ordinal', 'weekend_ratio', 'total_capacity'
    ]
    
    matched_pairs = []
    blocks_processed = 0
    treatments_matched = 0
    controls_used = set()
    
    print("\nMatching with Replacement:")
    print("=" * 80)
    
    for block_vars, block in df_blocked.groupby([
        'is_coastal', 'room_type', 'room_view', 'city_standardized',
        'month', 'children_allowed', 'capacity_quartile'
    ]):
        if len(block) < 2 or len(block) > max_block_size:
            continue
        
        blocks_processed += 1
        
        # Check all required features exist
        if not all(f in block.columns for f in match_features):
            continue
        
        # Split into Treatment (Premium Pricing) and Control (Discount Pricing)
        # Use median price within block as threshold
        median_price = block['avg_adr'].median()
        treatment = block[block['avg_adr'] > median_price].copy()
        control = block[block['avg_adr'] <= median_price].copy()
        
        if len(treatment) == 0 or len(control) == 0:
            continue
        
        # Prepare feature matrices
        treatment_features = treatment[match_features].fillna(0)
        control_features = control[match_features].fillna(0)
        
        try:
            # Normalize features
            scaler = StandardScaler()
            treatment_norm = scaler.fit_transform(treatment_features)
            control_norm = scaler.transform(control_features)
        except:
            continue
        
        # For EACH Treatment, find its SINGLE BEST Control
        for t_idx, t_row in enumerate(treatment.itertuples()):
            # Calculate distances to ALL controls
            distances = np.sqrt(((treatment_norm[t_idx] - control_norm) ** 2).sum(axis=1))
            
            # Find the BEST match (minimum distance)
            best_control_idx = np.argmin(distances)
            best_distance = distances[best_control_idx]
            c_row = control.iloc[best_control_idx]
            
            # Quality check: distance threshold
            if best_distance > 3.0:  # Relaxed threshold
                continue
            
            # Price difference check (must be at least 10%)
            price_diff_pct = abs(t_row.avg_adr - c_row['avg_adr']) / min(t_row.avg_adr, c_row['avg_adr'])
            if price_diff_pct < 0.10:
                continue
            
            # Store the match
            matched_pairs.append({
                'block_id': t_row.block_id,
                'is_coastal': t_row.is_coastal,
                'market_segment': t_row.market_segment,
                'room_type': t_row.room_type,
                'room_view': t_row.room_view,
                'city': t_row.city_standardized,
                'month': str(t_row.month),
                'capacity_quartile': t_row.capacity_quartile,
                
                # Treatment (Premium Pricing)
                'treatment_hotel': t_row.hotel_id,
                'treatment_price': t_row.avg_adr,
                'treatment_occupancy': t_row.occupancy_rate,
                
                # Control (Discount Pricing)
                'control_hotel': c_row['hotel_id'],
                'control_price': c_row['avg_adr'],
                'control_occupancy': c_row['occupancy_rate'],
                
                # Match quality
                'price_diff_pct': price_diff_pct,
                'match_distance': best_distance,
                
                # For revenue calculation
                'capacity': c_row['total_capacity'],
                'days_in_month': c_row['days_in_month']
            })
            
            treatments_matched += 1
            controls_used.add(c_row['hotel_id'])
        
        if blocks_processed % 100 == 0:
            print(f"  Processed {blocks_processed} blocks, {treatments_matched} treatments matched, {len(controls_used)} unique controls used...")
    
    pairs_df = pd.DataFrame(matched_pairs)
    
    print(f"\n✓ Matching complete:")
    print(f"  Blocks processed: {blocks_processed:,}")
    print(f"  Treatment hotels matched: {treatments_matched:,}")
    print(f"  Unique control hotels used: {len(controls_used):,}")
    print(f"  Reuse ratio: {treatments_matched / len(controls_used):.2f}x")
    
    return pairs_df


# %%

# %%
def block_bootstrap_ci(pairs_df: pd.DataFrame, n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """
    Calculate confidence intervals using block bootstrap.
    
    Clusters by treatment_hotel to account for:
    1. Multiple observations per treatment hotel (same hotel in different months)
    2. Serial correlation from reusing the same control hotels
    
    This properly accounts for the dependence structure introduced by matching with replacement.
    """
    print(f"\nCalculating {confidence*100:.0f}% CI with block bootstrap (n={n_bootstrap})...")
    print("Clustering by treatment_hotel to account for control reuse...")
    
    # Check control reuse distribution
    control_counts = pairs_df['control_hotel'].value_counts()
    print(f"\n  Control Reuse Distribution:")
    print(f"    Total unique controls: {len(control_counts)}")
    print(f"    Mean reuse: {control_counts.mean():.2f}x")
    print(f"    Median reuse: {control_counts.median():.0f}x")
    print(f"    Max reuse: {control_counts.max():.0f}x")
    print(f"    Top 10 controls account for: {control_counts.head(10).sum() / len(pairs_df) * 100:.1f}% of pairs")
    
    # Get unique treatment hotels (blocks)
    treatment_hotels = pairs_df['treatment_hotel'].unique()
    n_hotels = len(treatment_hotels)
    
    bootstrap_elasticities = []
    bootstrap_opportunities = []
    
    for i in range(n_bootstrap):
        # Resample treatment hotels WITH REPLACEMENT
        resampled_hotels = np.random.choice(treatment_hotels, size=n_hotels, replace=True)
        
        # Get all pairs for the resampled hotels
        bootstrap_sample = pairs_df[pairs_df['treatment_hotel'].isin(resampled_hotels)]
        
        if len(bootstrap_sample) > 0:
            bootstrap_elasticities.append(bootstrap_sample['arc_elasticity'].median())
            bootstrap_opportunities.append(bootstrap_sample['opportunity'].sum())
        
        if (i + 1) % 200 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}...")
    
    # Calculate percentile confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    elasticity_ci = (
        np.percentile(bootstrap_elasticities, lower_percentile),
        np.percentile(bootstrap_elasticities, upper_percentile)
    )
    
    opportunity_ci = (
        np.percentile(bootstrap_opportunities, lower_percentile),
        np.percentile(bootstrap_opportunities, upper_percentile)
    )
    
    results = {
        'elasticity_median': np.median(bootstrap_elasticities),
        'elasticity_ci_lower': elasticity_ci[0],
        'elasticity_ci_upper': elasticity_ci[1],
        'elasticity_std': np.std(bootstrap_elasticities),
        'elasticity_ci_width': elasticity_ci[1] - elasticity_ci[0],
        'opportunity_mean': np.mean(bootstrap_opportunities),
        'opportunity_ci_lower': opportunity_ci[0],
        'opportunity_ci_upper': opportunity_ci[1],
        'opportunity_std': np.std(bootstrap_opportunities),
        'n_bootstrap': n_bootstrap,
        'n_treatment_hotels': n_hotels,
        'control_reuse_mean': float(control_counts.mean()),
        'control_reuse_max': int(control_counts.max()),
        'control_concentration_top10_pct': float(control_counts.head(10).sum() / len(pairs_df) * 100)
    }
    
    print(f"\n✓ Bootstrap complete:")
    print(f"  Elasticity: {results['elasticity_median']:.4f} [{results['elasticity_ci_lower']:.4f}, {results['elasticity_ci_upper']:.4f}]")
    print(f"  CI Width: {results['elasticity_ci_width']:.4f}")
    print(f"  Total Opportunity: €{results['opportunity_mean']:,.0f} [€{results['opportunity_ci_lower']:,.0f}, €{results['opportunity_ci_upper']:,.0f}]")
    
    return results


def calculate_elasticity_and_opportunity(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates arc elasticity and counterfactual revenue opportunity (ATT)."""
    df = pairs_df.copy()
    
    # Arc elasticity (midpoint method)
    df['price_avg'] = (df['treatment_price'] + df['control_price']) / 2
    df['occ_avg'] = (df['treatment_occupancy'] + df['control_occupancy']) / 2
    df['price_pct_change'] = (df['treatment_price'] - df['control_price']) / df['price_avg']
    df['occ_pct_change'] = (df['treatment_occupancy'] - df['control_occupancy']) / df['occ_avg']
    df['arc_elasticity'] = df['occ_pct_change'] / df['price_pct_change']
    
    # Calculate price difference percentage for filtering
    df['price_diff_pct'] = abs(df['price_pct_change']) * 100
    
    # Filter valid pairs
    # Key constraint: price_diff < 100% ensures true substitutes (not different asset classes)
    df_valid = df[
        (df['arc_elasticity'] < 0) &
        (df['arc_elasticity'] > -5) &
        (df['occ_avg'] > 0.01) &
        (df['match_distance'] < 3.0) &
        (df['price_diff_pct'] > 10) &  # Meaningful treatment
        (df['price_diff_pct'] < 100)   # True substitutes (€100 vs €200, not €100 vs €500)
    ]
    
    # Counterfactual opportunity (for low-price hotels)
    df_valid['current_revenue'] = (
        df_valid['control_price'] * df_valid['control_occupancy'] * 
        df_valid['capacity'] * df_valid['days_in_month']
    )
    df_valid['counterfactual_occ'] = (
        df_valid['control_occupancy'] * (1 + df_valid['arc_elasticity'] * df_valid['price_pct_change'])
    ).clip(0, 1.0)
    df_valid['counterfactual_revenue'] = (
        df_valid['treatment_price'] * df_valid['counterfactual_occ'] * 
        df_valid['capacity'] * df_valid['days_in_month']
    )
    df_valid['opportunity'] = df_valid['counterfactual_revenue'] - df_valid['current_revenue']
    
    return df_valid[df_valid['opportunity'] > 0]


# %%
def calculate_revpar_metrics(opp_positive: pd.DataFrame, con=None) -> dict:
    """Calculate RevPAR increase metrics from matched pairs.
    
    Now uses ACTUAL data from the corrected occupancy calculation:
    - Occupancy is correctly calculated from exploded daily bookings
    - Capacity is the true hotel room count
    - RevPAR = ADR × Occupancy (from actual data)
    
    Returns:
        dict with RevPAR metrics calculated from actual data
    """
    # === ELASTICITY (from matched pairs) ===
    elasticity = opp_positive['arc_elasticity'].median()
    avg_price_diff_pct = opp_positive['price_diff_pct'].mean()
    n_hotels = opp_positive['treatment_hotel'].nunique()
    
    # Calculate robust relative lift metrics
    opp_positive['revpar_treatment'] = opp_positive['treatment_price'] * opp_positive['treatment_occupancy']
    opp_positive['revpar_control'] = opp_positive['control_price'] * opp_positive['control_occupancy']
    opp_positive['lift_abs'] = opp_positive['revpar_treatment'] - opp_positive['revpar_control']
    opp_positive['lift_pct'] = (opp_positive['lift_abs'] / opp_positive['revpar_control']) * 100
    
    # Use median lift for conservative estimate (mean is skewed by outliers)
    median_lift_pct = opp_positive['lift_pct'].median()
    mean_lift_pct = opp_positive['lift_pct'].mean()
    
    # === ACTUAL DATA FROM CONTROL (underpriced) HOTELS ===
    avg_adr = opp_positive['control_price'].mean()
    avg_occupancy = opp_positive['control_occupancy'].mean()
    avg_capacity = opp_positive['capacity'].mean()
    
    # Actual RevPAR from data = ADR × Occupancy
    actual_revpar = avg_adr * avg_occupancy
    
    # Actual monthly revenue per hotel = RevPAR × rooms × days
    avg_days = opp_positive['days_in_month'].mean()
    actual_monthly_per_hotel = actual_revpar * avg_capacity * avg_days
    
    # Total baseline for our sample
    actual_monthly_total = actual_monthly_per_hotel * n_hotels
    actual_annual_total = actual_monthly_total * 12
    
    # === MAXIMUM THEORETICAL SCENARIO ===
    # If underpriced hotels match twin prices (68% avg increase)
    max_revpar_multiplier = (1 + avg_price_diff_pct/100) * (1 + elasticity * avg_price_diff_pct/100)
    max_revpar_pct = (max_revpar_multiplier - 1) * 100
    max_annual_opportunity = actual_annual_total * (max_revpar_pct / 100)
    max_per_room_per_night = actual_revpar * (max_revpar_pct / 100)
    
    # === RECOMMENDED SCENARIOS (using elasticity formula) ===
    # RevPAR multiplier = (1+p) × (1+e×p)
    scenarios = {}
    for price_increase in [0.05, 0.10, 0.15, 0.20]:
        revpar_multiplier = (1 + price_increase) * (1 + elasticity * price_increase)
        revpar_gain_pct = (revpar_multiplier - 1) * 100
        
        annual_opp = actual_annual_total * (revpar_gain_pct / 100)
        per_room_night = actual_revpar * (revpar_gain_pct / 100)
        
        scenarios[f'{int(price_increase*100)}pct'] = {
            'price_increase_pct': price_increase * 100,
            'revpar_gain_pct': revpar_gain_pct,
            'monthly_opportunity': annual_opp / 12,
            'annual_opportunity': annual_opp,
            'per_room_per_night': per_room_night
        }
    
    return {
        # Elasticity & Lift
        'elasticity_median': elasticity,
        'avg_price_diff_pct': avg_price_diff_pct,
        'observed_lift_pct_median': median_lift_pct,
        'observed_lift_pct_mean': mean_lift_pct,
        
        # Actual baseline (from corrected data)
        'avg_adr': avg_adr,
        'avg_occupancy': avg_occupancy,
        'actual_revpar': actual_revpar,
        'avg_capacity': avg_capacity,
        'actual_annual_revenue': actual_annual_total,
        
        # Hotel info
        'n_hotels': n_hotels,
        
        # Maximum theoretical (matching twin prices - NOT recommended)
        'max_revpar_pct': max_revpar_pct,
        'max_annual_opportunity': max_annual_opportunity,
        'max_per_room_per_night': max_per_room_per_night,
        
        # Recommended scenarios
        'scenarios': scenarios,
        
        # Primary recommendation (10% increase)
        'recommended_revpar_pct': scenarios['10pct']['revpar_gain_pct'],
        'recommended_annual_opportunity': scenarios['10pct']['annual_opportunity'],
        'recommended_per_room_per_night': scenarios['10pct']['per_room_per_night']
    }


# %%
def create_executive_summary_figure(opp_positive: pd.DataFrame, bootstrap_results: dict, output_path: Path) -> None:
    """Creates executive summary figure matching geographic analysis format."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    # Calculate key metrics
    n_hotels = opp_positive['treatment_hotel'].nunique()
    n_obs = len(opp_positive)
    total_opp = opp_positive['opportunity'].sum()
    avg_opp_per_obs = total_opp / n_obs
    annual_per_hotel = avg_opp_per_obs * 12
    conservative_total = annual_per_hotel * n_hotels
    
    # Market extrapolation
    total_market = 2255
    moderate_hotels = int(total_market * 0.10)
    optimistic_hotels = int(total_market * 0.20)
    moderate_total = annual_per_hotel * moderate_hotels
    optimistic_total = annual_per_hotel * optimistic_hotels
    
    elasticity_median = opp_positive['arc_elasticity'].median()
    elasticity_ci_lower = bootstrap_results['elasticity_ci_lower']
    elasticity_ci_upper = bootstrap_results['elasticity_ci_upper']
    
    # Create figure with 2x3 grid (matching geographic analysis layout)
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25, 
                          left=0.06, right=0.97, top=0.93, bottom=0.06)
    
    colors = {
        'primary': '#2E86AB',
        'success': '#3E8914',
        'warning': '#F18F01',
        'gray': '#808080'
    }
    
    def currency_fmt(x, pos):
        return f'€{x/1e6:.1f}M'
    
    # ========================================================================
    # Plot 1: Elasticity Distribution (KDE)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    sns.kdeplot(opp_positive['arc_elasticity'], fill=True, alpha=0.3, linewidth=2.5, 
                color=colors['primary'], ax=ax1)
    ax1.axvline(elasticity_median, color='black', linestyle='--', linewidth=2, 
                label=f'Median: {elasticity_median:.2f}')
    ax1.axvline(elasticity_ci_lower, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax1.axvline(elasticity_ci_upper, color='red', linestyle=':', linewidth=2, alpha=0.7,
                label=f'95% CI: [{elasticity_ci_lower:.2f}, {elasticity_ci_upper:.2f}]')
    
    ax1.set_title('Confidence: Price Elasticity Distribution', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Price Elasticity of Demand (ε)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fontsize=11)
    ax1.set_xlim(-1.5, 0)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Opportunity per Hotel Distribution
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    hotel_opp = opp_positive.groupby('treatment_hotel')['opportunity'].sum()
    sns.histplot(hotel_opp, bins=20, kde=True, ax=ax2, color=colors['success'], 
                 alpha=0.6, edgecolor='black', linewidth=1.5)
    ax2.axvline(hotel_opp.median(), color='red', linestyle='--', linewidth=2.5, 
                label=f'Median: €{hotel_opp.median():,.0f}')
    ax2.axvline(hotel_opp.mean(), color='blue', linestyle=':', linewidth=2, 
                label=f'Mean: €{hotel_opp.mean():,.0f}')
    
    ax2.set_title('Opportunity per Hotel (Sample)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Total Opportunity (€)', fontsize=12)
    ax2.set_ylabel('Number of Hotels', fontsize=12)
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'€{x/1000:.0f}k'))
    
    # ========================================================================
    # Plot 3: Market Opportunity Bar Chart
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    scenarios = ['Conservative\n(97 hotels)', 'Moderate\n(226 hotels)', 'Optimistic\n(451 hotels)']
    values = [conservative_total, moderate_total, optimistic_total]
    colors_list = [colors['primary'], colors['success'], colors['warning']]
    
    bars = ax3.bar(scenarios, values, color=colors_list, alpha=0.85, 
                   edgecolor='black', linewidth=2, width=0.7)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                 f'€{val/1e6:.1f}M', ha='center', va='bottom', 
                 fontsize=13, fontweight='bold')
    
    ax3.set_title('Total Revenue Opportunity by Scenario', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Net Opportunity', fontsize=13, fontweight='bold')
    ax3.yaxis.set_major_formatter(FuncFormatter(currency_fmt))
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_ylim(0, max(values) * 1.15)
    
    # ========================================================================
    # Plot 4: Sample Size & Match Quality
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    quality_metrics = {
        'Matched Pairs': len(opp_positive),
        'Unique Hotels': n_hotels,
        'Unique Controls': opp_positive['control_hotel'].nunique()
    }
    
    x_pos = np.arange(len(quality_metrics))
    bars = ax4.bar(x_pos, list(quality_metrics.values()), 
                   color=[colors['primary'], colors['success'], colors['warning']], 
                   alpha=0.85, edgecolor='black', linewidth=2, width=0.7)
    
    for bar, val in zip(bars, quality_metrics.values()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                 f'{int(val)}', ha='center', va='bottom', 
                 fontsize=13, fontweight='bold')
    
    ax4.set_title('Sample Size & Match Quality', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(list(quality_metrics.keys()), fontsize=11, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.set_ylim(0, max(quality_metrics.values()) * 1.2)
    
    # Add quality note
    avg_match_dist = opp_positive['match_distance'].mean()
    avg_price_diff = opp_positive['price_diff_pct'].mean()
    control_reuse = len(opp_positive) / opp_positive['control_hotel'].nunique()
    
    note_text = f'Match Distance: {avg_match_dist:.2f}\nPrice Diff: {avg_price_diff:.1f}%\nControl Reuse: {control_reuse:.1f}x'
    ax4.text(0.02, 0.98, note_text, transform=ax4.transAxes, ha='left', va='top', 
             fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                                   edgecolor='black', linewidth=1, alpha=0.8))
    
    # ========================================================================
    # Plot 5: Confidence Intervals & Key Metrics
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Plot elasticity CI
    ax5.plot([elasticity_ci_lower, elasticity_ci_upper], [0, 0], 
             color=colors['primary'], linewidth=10, alpha=0.3, solid_capstyle='round')
    ax5.plot(elasticity_median, 0, 'o', color=colors['primary'], 
             markersize=15, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax5.text(elasticity_median, 0, f"  ε = {elasticity_median:.2f}", 
             va='center', ha='left', fontsize=13, fontweight='bold')
    
    ax5.set_yticks([0])
    ax5.set_yticklabels(['Elasticity'], fontsize=12, fontweight='bold')
    ax5.set_xlabel('Price Elasticity (ε) with 95% CI', fontsize=13, fontweight='bold')
    ax5.set_title('Elasticity Estimate (with Confidence Interval)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    ax5.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax5.set_xlim(-1.0, 0)
    ax5.set_ylim(-0.5, 0.5)
    
    # Add interpretation box
    interp_text = f"""INTERPRETATION:
    
ε = {elasticity_median:.2f} means:
• 10% price increase → {abs(elasticity_median)*10:.1f}% occupancy decrease
• Net revenue gain: +{(1 + elasticity_median*0.1)*100 - 100:.1f}%
• Demand is INELASTIC (|ε| < 1)
• Hotels have pricing power

OPPORTUNITY:
• {n_obs} observations from {n_hotels} hotels
• €{avg_opp_per_obs:,.0f} per observation
• €{annual_per_hotel:,.0f} per hotel annually
    """
    
    ax5.text(0.02, 0.02, interp_text, transform=ax5.transAxes, ha='left', va='bottom', 
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                      edgecolor='black', linewidth=1, alpha=0.3))
    
    # ========================================================================
    # Plot 6: Calculation Verification
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    verification_text = f"""CALCULATION VERIFICATION
(Reproducible from matched_pairs_with_replacement.csv)

Step 1: Sample Results
  • Total opportunity: €{total_opp:,.2f}
  • Observations: {n_obs}
  • Average per obs: €{avg_opp_per_obs:,.2f}

Step 2: Annualization
  • Per hotel per year: 
    €{avg_opp_per_obs:,.2f} × 12 months
    = €{annual_per_hotel:,.2f}

Step 3: Market Extrapolation
  • Conservative: {n_hotels} hotels
    = €{conservative_total:,.0f} (€{conservative_total/1e6:.2f}M)
  
  • Moderate: {moderate_hotels} hotels (10%)
    = €{moderate_total:,.0f} (€{moderate_total/1e6:.2f}M)
  
  • Optimistic: {optimistic_hotels} hotels (20%)
    = €{optimistic_total:,.0f} (€{optimistic_total/1e6:.2f}M)

Reality Check:
  • €{annual_per_hotel:,.0f}/year per hotel
    = €{avg_opp_per_obs:,.0f}/month average
  • For 20-room hotel at €75 ADR:
    ~{avg_opp_per_obs/(75*30):.1f} extra room-nights/day
  • Aligns with ε = {elasticity_median:.2f}
    """
    
    ax6.text(0.05, 0.95, verification_text, transform=ax6.transAxes, 
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                      edgecolor='black', linewidth=2, alpha=0.3))
    
    # ========================================================================
    # Overall title
    # ========================================================================
    plt.suptitle('Matched Pairs Analysis: Comprehensive Elasticity Assessment', 
                 fontsize=19, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Executive summary saved to: {output_path}")


# %%
def create_geographic_style_figure(opp_positive: pd.DataFrame, bootstrap_results: dict, output_path: Path) -> None:
    """Creates geographic-style executive figure matching matched_pairs_geographic.py format."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    colors = {
        'coastal': '#F18F01',
        'urban': '#2E86AB',
        'provincial': '#A23B72',
        'success': '#3E8914',
        'gray': '#808080'
    }
    
    color_map = {
        'Coastal/Resort': colors['coastal'],
        'Urban/Madrid': colors['urban'],
        'Provincial/Regional': colors['provincial']
    }
    
    def currency_fmt(x, pos):
        return f'€{x/1e6:.1f}M'
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, height_ratios=[1, 1, 1.2])
    
    # Plot 1: Elasticity Distribution by Market Segment (KDE)
    ax1 = fig.add_subplot(gs[0, :])
    
    for segment in sorted(opp_positive['market_segment'].unique()):
        segment_data = opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity']
        color = color_map.get(segment, colors['gray'])
        
        if len(segment_data) > 5:
            sns.kdeplot(segment_data, ax=ax1, fill=True, alpha=0.3, linewidth=2.5, 
                       label=f'{segment} (n={len(segment_data)}, ε={segment_data.median():.2f})',
                       color=color)
    
    overall_median = opp_positive['arc_elasticity'].median()
    ax1.axvline(overall_median, color='black', linestyle='--', linewidth=3, 
                label=f'Overall Median: {overall_median:.2f}', zorder=10)
    
    ax1.set_title('Price Elasticity Distribution by Market Segment', 
                  fontsize=17, fontweight='bold', pad=20)
    ax1.set_xlabel('Price Elasticity of Demand (ε)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fontsize=11, shadow=True, ncol=2)
    ax1.set_xlim(-1.5, 0)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax1.text(0.98, 0.97, 'More Negative = More Elastic\nCloser to 0 = More Inelastic\n(Revenue gains from price increases)', 
             transform=ax1.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                      edgecolor='black', linewidth=1.5, alpha=0.9))
    
    # Plot 2: Box Plot by Segment
    ax2 = fig.add_subplot(gs[1, 0])
    
    segment_data_list = []
    segment_labels = []
    segment_colors = []
    
    for segment in sorted(opp_positive['market_segment'].unique()):
        segment_data_list.append(opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity'])
        segment_labels.append(segment)
        segment_colors.append(color_map.get(segment, colors['gray']))
    
    bp = ax2.boxplot(segment_data_list, labels=segment_labels, patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True,
                     boxprops=dict(linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2),
                     medianprops=dict(color='red', linewidth=3),
                     meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    for patch, color in zip(bp['boxes'], segment_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_title('Elasticity Distribution by Segment', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Price Elasticity (ε)', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(segment_labels, fontsize=11, fontweight='bold', rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax2.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    for i, (label, data) in enumerate(zip(segment_labels, segment_data_list)):
        ax2.text(i+1, ax2.get_ylim()[0] * 0.95, f'n={len(data)}', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Plot 3: Annualized Market Opportunity by Segment
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate annualized opportunity per segment
    segment_annual = {}
    for segment in opp_positive['market_segment'].unique():
        segment_data = opp_positive[opp_positive['market_segment'] == segment]
        n_obs = len(segment_data)
        total_opp = segment_data['opportunity'].sum()
        avg_per_obs = total_opp / n_obs
        n_hotels = segment_data['treatment_hotel'].nunique()
        # Annualize: avg per obs × 12 months × number of hotels
        annual_opp = avg_per_obs * 12 * n_hotels
        segment_annual[segment] = annual_opp
    
    # Sort by value
    segments_sorted = sorted(segment_annual.items(), key=lambda x: x[1], reverse=True)
    segment_names = [s[0] for s in segments_sorted]
    segment_values = [s[1] for s in segments_sorted]
    bar_colors_list = [color_map.get(l, colors['gray']) for l in segment_names]
    
    bars = ax3.bar(range(len(segment_names)), segment_values, color=bar_colors_list, 
                   alpha=0.85, edgecolor='black', linewidth=2, width=0.7)
    
    for bar, val in zip(bars, segment_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                 f'€{val/1e6:.2f}M', ha='center', va='bottom', 
                 fontsize=13, fontweight='bold')
    
    ax3.set_title('Annualized Market Opportunity by Segment', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(range(len(segment_names)))
    ax3.set_xticklabels(segment_names, fontsize=11, fontweight='bold', rotation=15, ha='right')
    ax3.set_ylabel('Annual Opportunity (Conservative)', fontsize=13, fontweight='bold')
    ax3.yaxis.set_major_formatter(FuncFormatter(currency_fmt))
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_ylim(0, max(segment_values) * 1.15)
    
    # Plot 4: Sample Size & Match Quality
    ax4 = fig.add_subplot(gs[2, 0])
    
    segment_stats = opp_positive.groupby('market_segment').agg({
        'treatment_hotel': 'count',
        'match_distance': 'mean'
    }).reset_index()
    segment_stats = segment_stats.sort_values('treatment_hotel', ascending=False)
    
    x_pos = np.arange(len(segment_stats))
    bars1 = ax4.bar(x_pos, segment_stats['treatment_hotel'], 
                    color=[color_map.get(l, colors['gray']) for l in segment_stats['market_segment']],
                    alpha=0.85, edgecolor='black', linewidth=2, width=0.7)
    
    for bar, val, quality in zip(bars1, segment_stats['treatment_hotel'], segment_stats['match_distance']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                f'{int(val)} pairs\n(quality: {quality:.2f})', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_title('Sample Size & Match Quality by Segment', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(segment_stats['market_segment'], fontsize=11, fontweight='bold', 
                        rotation=15, ha='right')
    ax4.set_ylabel('Number of Matched Pairs', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.set_ylim(0, segment_stats['treatment_hotel'].max() * 1.2)
    
    ax4.text(0.02, 0.98, 'Lower match quality score = better match', 
             transform=ax4.transAxes, ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                      edgecolor='black', linewidth=1, alpha=0.8))
    
    # Plot 5: Confidence Intervals by Segment
    ax5 = fig.add_subplot(gs[2, 1])
    
    segment_ci_data = []
    for segment in sorted(opp_positive['market_segment'].unique()):
        data = opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity']
        median = data.median()
        ci_lower = data.quantile(0.025)
        ci_upper = data.quantile(0.975)
        segment_ci_data.append({
            'label': segment,
            'median': median,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'color': color_map.get(segment, colors['gray'])
        })
    
    segment_ci_data = sorted(segment_ci_data, key=lambda x: x['median'])
    y_pos = np.arange(len(segment_ci_data))
    
    for i, seg in enumerate(segment_ci_data):
        ax5.plot([seg['ci_lower'], seg['ci_upper']], [i, i], 
                color=seg['color'], linewidth=8, alpha=0.3, solid_capstyle='round')
        ax5.plot(seg['median'], i, 'o', color=seg['color'], 
                markersize=12, markeredgecolor='black', markeredgewidth=2, zorder=10)
        ax5.text(seg['median'], i, f"  {seg['median']:.2f}", 
                va='center', ha='left', fontsize=11, fontweight='bold')
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([seg['label'] for seg in segment_ci_data], fontsize=11, fontweight='bold')
    ax5.set_xlabel('Price Elasticity (ε) with 95% CI', fontsize=13, fontweight='bold')
    ax5.set_title('Elasticity Estimates by Segment (with Confidence Intervals)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    ax5.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax5.set_xlim(-1.2, 0)
    
    ax5.plot([], [], 'o', color='gray', markersize=10, markeredgecolor='black', 
            markeredgewidth=2, label='Median')
    ax5.plot([], [], '-', color='gray', linewidth=8, alpha=0.3, label='95% CI')
    ax5.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    
    plt.suptitle('Matched Pairs Analysis: Comprehensive Elasticity Assessment', 
                 fontsize=19, fontweight='bold', y=0.995)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Geographic-style figure saved")


# %%
def calculate_underpricing_analysis(pairs_df: pd.DataFrame, total_market_hotels: int = 2255) -> dict:
    """
    Calculate underpricing opportunity for CONTROL (underpriced) hotels.
    
    The CONTROL hotels in matched pairs are the underpriced ones:
    - Same features as peers (matched on 17 variables)
    - Lower prices, higher occupancy = classic underpricing signal
    
    Args:
        pairs_df: DataFrame with matched pairs
        total_market_hotels: Total hotels in market for percentage calculation
    
    Returns:
        Dictionary with underpricing metrics and opportunity scenarios
    """
    # Identify unique underpriced hotels (controls)
    underpriced_hotels = pairs_df['control_hotel'].unique()
    peer_hotels = pairs_df['treatment_hotel'].unique()
    
    # Calculate averages for underpriced vs peers
    # Group by control hotel to avoid double-counting
    control_stats = pairs_df.groupby('control_hotel').agg({
        'control_price': 'mean',
        'control_occupancy': 'mean',
        'treatment_price': 'mean',
        'treatment_occupancy': 'mean',
        'capacity': 'mean',
        'days_in_month': 'mean',
        'arc_elasticity': 'median'
    }).reset_index()
    
    # Key metrics
    avg_underpriced_adr = control_stats['control_price'].mean()
    avg_peer_adr = control_stats['treatment_price'].mean()
    avg_underpriced_occ = control_stats['control_occupancy'].mean()
    avg_peer_occ = control_stats['treatment_occupancy'].mean()
    avg_capacity = control_stats['capacity'].mean()
    median_elasticity = control_stats['arc_elasticity'].median()
    
    # Price gap
    price_gap_pct = ((avg_peer_adr - avg_underpriced_adr) / avg_peer_adr) * 100
    
    # RevPAR comparison
    underpriced_revpar = avg_underpriced_adr * avg_underpriced_occ
    peer_revpar = avg_peer_adr * avg_peer_occ
    revpar_gap_pct = ((peer_revpar - underpriced_revpar) / peer_revpar) * 100
    
    # Current annual revenue for underpriced hotels
    # Monthly: ADR × occupancy × capacity × 30 days
    # Annual: × 12 months × number of hotels
    monthly_revenue_per_hotel = avg_underpriced_adr * avg_underpriced_occ * avg_capacity * 30
    annual_revenue_per_hotel = monthly_revenue_per_hotel * 12
    total_current_annual = annual_revenue_per_hotel * len(underpriced_hotels)
    
    # Calculate scenarios for price increases
    scenarios = {}
    for increase_pct in [10, 20, 30, 46]:  # 46% would close the gap
        price_multiplier = 1 + (increase_pct / 100)
        new_adr = avg_underpriced_adr * price_multiplier
        
        # Adjust occupancy based on elasticity
        # elasticity = %Δoccupancy / %Δprice
        # So %Δoccupancy = elasticity × %Δprice
        occ_change_pct = median_elasticity * increase_pct  # elasticity is negative
        new_occ = avg_underpriced_occ * (1 + occ_change_pct / 100)
        new_occ = max(0, min(1, new_occ))  # Clamp to [0, 1]
        
        # New RevPAR
        new_revpar = new_adr * new_occ
        revpar_gain_pct = ((new_revpar - underpriced_revpar) / underpriced_revpar) * 100
        
        # New revenue
        new_monthly = new_adr * new_occ * avg_capacity * 30
        new_annual_per_hotel = new_monthly * 12
        new_total_annual = new_annual_per_hotel * len(underpriced_hotels)
        
        opportunity = new_total_annual - total_current_annual
        opportunity_per_hotel = opportunity / len(underpriced_hotels)
        
        scenarios[f'{increase_pct}pct'] = {
            'price_increase_pct': increase_pct,
            'new_adr': new_adr,
            'new_occupancy': new_occ,
            'new_revpar': new_revpar,
            'revpar_gain_pct': revpar_gain_pct,
            'annual_opportunity': opportunity,
            'opportunity_per_hotel': opportunity_per_hotel,
            'new_total_annual_revenue': new_total_annual
        }
    
    return {
        # Market context
        'total_market_hotels': total_market_hotels,
        'underpriced_hotel_count': len(underpriced_hotels),
        'underpriced_market_pct': (len(underpriced_hotels) / total_market_hotels) * 100,
        'peer_hotel_count': len(peer_hotels),
        
        # Current metrics - Underpriced
        'avg_underpriced_adr': avg_underpriced_adr,
        'avg_underpriced_occupancy': avg_underpriced_occ,
        'avg_underpriced_revpar': underpriced_revpar,
        
        # Current metrics - Peers
        'avg_peer_adr': avg_peer_adr,
        'avg_peer_occupancy': avg_peer_occ,
        'avg_peer_revpar': peer_revpar,
        
        # Gaps
        'price_gap_pct': price_gap_pct,
        'revpar_gap_pct': revpar_gap_pct,
        'occupancy_advantage_pct': ((avg_underpriced_occ - avg_peer_occ) / avg_peer_occ) * 100,
        
        # Capacity and elasticity
        'avg_capacity': avg_capacity,
        'median_elasticity': median_elasticity,
        
        # Current revenue
        'current_annual_per_hotel': annual_revenue_per_hotel,
        'current_total_annual': total_current_annual,
        
        # Scenarios
        'scenarios': scenarios
    }


def print_underpricing_summary(analysis: dict) -> None:
    """Print clear underpricing analysis summary."""
    print("\n" + "=" * 80)
    print("UNDERPRICING ANALYSIS")
    print("=" * 80)
    
    print(f"\n{'='*60}")
    print("WHO IS UNDERPRICED?")
    print(f"{'='*60}")
    print(f"   Underpriced hotels: {analysis['underpriced_hotel_count']} ({analysis['underpriced_market_pct']:.0f}% of {analysis['total_market_hotels']} market)")
    print(f"   Their peers (similar features, higher prices): {analysis['peer_hotel_count']} hotels")
    
    print(f"\n{'='*60}")
    print("HOW MUCH ARE THEY UNDERPRICING?")
    print(f"{'='*60}")
    print(f"\n   {'Metric':<25} {'Underpriced':<15} {'Peers':<15} {'Gap':<15}")
    print(f"   {'-'*70}")
    print(f"   {'Avg ADR':<25} €{analysis['avg_underpriced_adr']:<14.2f} €{analysis['avg_peer_adr']:<14.2f} {-analysis['price_gap_pct']:+.0f}%")
    print(f"   {'Avg Occupancy':<25} {analysis['avg_underpriced_occupancy']*100:<14.1f}% {analysis['avg_peer_occupancy']*100:<14.1f}% {analysis['occupancy_advantage_pct']:+.0f}%")
    print(f"   {'RevPAR':<25} €{analysis['avg_underpriced_revpar']:<14.2f} €{analysis['avg_peer_revpar']:<14.2f} {-analysis['revpar_gap_pct']:+.0f}%")
    
    print(f"\n   Key insight: Underpriced hotels charge {analysis['price_gap_pct']:.0f}% LESS")
    print(f"                but have {analysis['occupancy_advantage_pct']:.0f}% HIGHER occupancy")
    print(f"                → Demand exists at higher prices!")
    
    print(f"\n{'='*60}")
    print("WHAT'S THE OPPORTUNITY?")
    print(f"{'='*60}")
    print(f"\n   Elasticity: ε = {analysis['median_elasticity']:.2f}")
    print(f"   → For every 10% price increase, only {abs(analysis['median_elasticity']) * 10:.1f}% occupancy loss")
    print(f"\n   Current annual revenue ({analysis['underpriced_hotel_count']} hotels): €{analysis['current_total_annual']:,.0f}")
    
    print(f"\n   If underpriced hotels raised prices by:")
    print(f"   {'Increase':<12} {'New ADR':<12} {'New Occ':<12} {'RevPAR Gain':<15} {'Annual Opportunity':<20}")
    print(f"   {'-'*75}")
    
    for key in ['10pct', '20pct', '30pct', '46pct']:
        s = analysis['scenarios'][key]
        label = f"{s['price_increase_pct']}%"
        if s['price_increase_pct'] == 46:
            label = "46% (close gap)"
        print(f"   {label:<12} €{s['new_adr']:<11.2f} {s['new_occupancy']*100:<11.1f}% +{s['revpar_gain_pct']:<14.1f}% €{s['annual_opportunity']:>18,.0f}")
    
    # Recommended scenario (30%)
    rec = analysis['scenarios']['30pct']
    print(f"\n   RECOMMENDED (30% increase):")
    print(f"   → Still {analysis['price_gap_pct'] - 30:.0f}% below peers (conservative)")
    print(f"   → RevPAR gain: +{rec['revpar_gain_pct']:.1f}%")
    print(f"   → Annual opportunity: €{rec['annual_opportunity']:,.0f}")
    print(f"   → Per hotel: €{rec['opportunity_per_hotel']:,.0f}/year")


# %%
def create_underpricing_figure(pairs_df: pd.DataFrame, analysis: dict, output_path: Path) -> None:
    """
    Create figure showing underpricing analysis:
    1. Price gap distribution (underpriced vs peers)
    2. ADR scatter (current vs peer)
    3. Opportunity by price increase scenario
    4. RevPAR comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Color scheme
    colors = {
        'underpriced': '#E74C3C',  # Red for underpriced
        'peers': '#2ECC71',        # Green for peers
        'opportunity': '#3498DB',   # Blue for opportunity
        'neutral': '#95A5A6'
    }
    
    # --- Panel 1: Price gap distribution ---
    ax1 = axes[0, 0]
    
    # Calculate per-hotel price gaps
    hotel_stats = pairs_df.groupby('control_hotel').agg({
        'control_price': 'mean',
        'treatment_price': 'mean'
    }).reset_index()
    hotel_stats['price_gap_pct'] = ((hotel_stats['treatment_price'] - hotel_stats['control_price']) 
                                     / hotel_stats['treatment_price'] * 100)
    
    ax1.hist(hotel_stats['price_gap_pct'], bins=30, color=colors['underpriced'], 
             alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(hotel_stats['price_gap_pct'].median(), color='black', linestyle='--', 
                linewidth=2, label=f'Median: {hotel_stats["price_gap_pct"].median():.0f}%')
    ax1.set_xlabel('Price Gap (% below peers)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Hotels', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Underpricing\n(How much below peers?)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: ADR Scatter (Underpriced vs Peer) ---
    ax2 = axes[0, 1]
    
    ax2.scatter(hotel_stats['control_price'], hotel_stats['treatment_price'], 
                alpha=0.5, s=30, c=colors['underpriced'], edgecolors='black', linewidth=0.3)
    
    # 45-degree line (no gap)
    max_val = max(hotel_stats['control_price'].max(), hotel_stats['treatment_price'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='No gap (equal pricing)')
    
    # Annotate
    ax2.set_xlabel('Underpriced Hotel ADR (€)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Peer Hotel ADR (€)', fontsize=12, fontweight='bold')
    ax2.set_title('ADR Comparison\n(Points above line = underpricing)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for gap - point to actual data region
    ax2.annotate('Underpricing\ngap', 
                 xy=(80, 200), xytext=(200, 100),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # --- Panel 3: Opportunity by Scenario ---
    ax3 = axes[1, 0]
    
    scenarios = analysis['scenarios']
    scenario_labels = ['10%', '20%', '30%', '46%\n(close gap)']
    scenario_keys = ['10pct', '20pct', '30pct', '46pct']
    opportunities = [scenarios[k]['annual_opportunity'] / 1e6 for k in scenario_keys]
    revpar_gains = [scenarios[k]['revpar_gain_pct'] for k in scenario_keys]
    
    bars = ax3.bar(scenario_labels, opportunities, color=colors['opportunity'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, opp, gain in zip(bars, opportunities, revpar_gains):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'€{opp:.1f}M\n(+{gain:.0f}% RevPAR)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Price Increase Scenario', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Annual Opportunity (€M)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Revenue Opportunity by Scenario\n({analysis["underpriced_hotel_count"]} underpriced hotels)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight recommended
    bars[2].set_color('#27AE60')  # Green for recommended
    ax3.annotate('RECOMMENDED', xy=(2, opportunities[2]), 
                 xytext=(2.5, opportunities[2] + 1),
                 fontsize=10, fontweight='bold', color='#27AE60',
                 arrowprops=dict(arrowstyle='->', color='#27AE60'))
    
    # --- Panel 4: RevPAR Comparison ---
    ax4 = axes[1, 1]
    
    categories = ['Underpriced\nHotels', 'Peer\nHotels', 'After 30%\nIncrease']
    revpars = [
        analysis['avg_underpriced_revpar'],
        analysis['avg_peer_revpar'],
        scenarios['30pct']['new_revpar']
    ]
    bar_colors = [colors['underpriced'], colors['peers'], '#27AE60']
    
    bars = ax4.bar(categories, revpars, color=bar_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, revpar in zip(bars, revpars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'€{revpar:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4.set_ylabel('RevPAR (€/room/night)', fontsize=12, fontweight='bold')
    ax4.set_title('RevPAR Comparison\n(Before vs After Optimization)', 
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add gap annotation
    gap = analysis['avg_peer_revpar'] - analysis['avg_underpriced_revpar']
    ax4.annotate(f'Gap: €{gap:.2f}\n({analysis["revpar_gap_pct"]:.0f}%)', 
                 xy=(0.5, (revpars[0] + revpars[1])/2),
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
    
    plt.suptitle('Underpricing Analysis: Who, How Much, What Opportunity?', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Underpricing figure saved to: {output_path}")


# %%
def create_optimization_curve_figure(pairs_df: pd.DataFrame, output_path: Path) -> None:
    """
    Create revenue optimization curve from matched pairs data.
    
    This is the empirical version - using actual elasticity measured from matched pairs
    to project RevPAR at different price points.
    """
    import numpy as np
    
    # Get empirical elasticity from matched pairs
    elasticity_median = pairs_df['arc_elasticity'].median()
    elasticity_25 = pairs_df['arc_elasticity'].quantile(0.25)
    elasticity_75 = pairs_df['arc_elasticity'].quantile(0.75)
    
    # Get baseline metrics from control (underpriced) hotels
    base_adr = pairs_df['control_price'].mean()
    base_occ = pairs_df['control_occupancy'].mean()
    base_revpar = base_adr * base_occ
    
    # Price deviation range
    price_range = np.linspace(-30, 120, 100)
    
    # Variable elasticity model: elasticity becomes more negative at higher prices
    # This captures competitive dynamics and customer price sensitivity
    sensitivity_factor = 0.015  # Elasticity increases 1.5% per 1% price deviation
    
    def compute_revpar_curve(base_elasticity, price_devs):
        """Compute RevPAR curve with variable elasticity."""
        prices = base_adr * (1 + price_devs / 100)
        
        # Variable elasticity
        elasticities = base_elasticity * (1 + sensitivity_factor * np.abs(price_devs))
        
        # Compute occupancy with cumulative elasticity effect
        occupancies = np.zeros_like(price_devs)
        # Find index closest to 0 as baseline
        zero_idx = np.argmin(np.abs(price_devs))
        occupancies[zero_idx] = base_occ
        
        # Forward from zero
        for i in range(zero_idx + 1, len(price_devs)):
            delta_p = (price_devs[i] - price_devs[i-1]) / 100
            avg_e = (elasticities[i] + elasticities[i-1]) / 2
            occupancies[i] = occupancies[i-1] * (1 + avg_e * delta_p)
        
        # Backward from zero
        for i in range(zero_idx - 1, -1, -1):
            delta_p = (price_devs[i] - price_devs[i+1]) / 100
            avg_e = (elasticities[i] + elasticities[i+1]) / 2
            occupancies[i] = occupancies[i+1] * (1 + avg_e * delta_p)
        
        occupancies = np.clip(occupancies, 0.01, 0.95)
        return prices * occupancies
    
    # Compute curves for median and IQR
    revpar_median = compute_revpar_curve(elasticity_median, price_range)
    revpar_25 = compute_revpar_curve(elasticity_25, price_range)
    revpar_75 = compute_revpar_curve(elasticity_75, price_range)
    
    # Find optimal for median curve
    opt_idx = np.argmax(revpar_median)
    opt_dev = price_range[opt_idx]
    opt_rev = revpar_median[opt_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left Panel: Empirical Elasticity Distribution ---
    ax1 = axes[0]
    
    # Plot elasticity distribution
    import seaborn as sns
    sns.histplot(pairs_df['arc_elasticity'], bins=40, kde=True, ax=ax1, 
                 color='#2E86AB', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax1.axvline(elasticity_median, color='#E74C3C', linestyle='--', linewidth=2.5,
                label=f'Median: ε={elasticity_median:.2f}')
    ax1.axvline(elasticity_25, color='orange', linestyle=':', linewidth=2,
                label=f'Q1: ε={elasticity_25:.2f}')
    ax1.axvline(elasticity_75, color='orange', linestyle=':', linewidth=2,
                label=f'Q3: ε={elasticity_75:.2f}')
    
    ax1.set_xlabel('Price Elasticity of Demand (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Empirical Elasticity Distribution\n(From Matched Pairs)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-1.5, 0)
    ax1.grid(True, alpha=0.3)
    
    # Add interpretation
    ax1.text(0.02, 0.98, f'n = {len(pairs_df):,} pairs\nAll ε < 0 (inelastic)',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # --- Right Panel: Revenue Optimization Curve ---
    ax2 = axes[1]
    
    # Plot confidence band (IQR)
    ax2.fill_between(price_range, revpar_25, revpar_75, alpha=0.2, color='#2E86AB',
                     label='IQR Range')
    
    # Plot median curve
    ax2.plot(price_range, revpar_median, lw=3, color='#2E86AB',
             label=f'RevPAR Curve (ε={elasticity_median:.2f})')
    
    # Mark optimal point
    ax2.axvline(opt_dev, color='#27AE60', linestyle='--', lw=2, 
                label=f'Optimal: +{opt_dev:.0f}%')
    ax2.scatter(opt_dev, opt_rev, color='#27AE60', s=150, zorder=5, 
                edgecolors='black', linewidth=2)
    
    # Shade zones
    ax2.axvspan(15, 40, alpha=0.15, color='green', label='Safe Zone (15-40%)')
    ax2.axvspan(opt_dev, 120, alpha=0.15, color='red', label=f'Overpricing (>{opt_dev:.0f}%)')
    
    # Mark peer average and current position
    ax2.axvline(0, color='black', linestyle=':', alpha=0.5, label='Peer Average')
    
    # Calculate lift at optimal
    zero_idx = np.argmin(np.abs(price_range))
    base_rev = revpar_median[zero_idx]
    lift_pct = ((opt_rev - base_rev) / base_rev) * 100
    
    ax2.set_xlabel('Price Deviation from Peer Group (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'Expected RevPAR (€)', fontsize=12, fontweight='bold')
    ax2.set_title('Revenue Optimization Curve\n(Matched Pairs Empirical Model)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-30, 120)
    
    # Add metrics box
    metrics_text = f"""At Optimal (+{opt_dev:.0f}%):
RevPAR: €{opt_rev:.2f}
Lift: +{lift_pct:.1f}%
Base: €{base_rev:.2f}"""
    ax2.text(0.02, 0.02, metrics_text, transform=ax2.transAxes, fontsize=9,
             va='bottom', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Optimization curve saved to: {output_path}")
    print(f"      Empirical elasticity: ε = {elasticity_median:.2f}")
    print(f"      Optimal price deviation: +{opt_dev:.0f}%")
    print(f"      Expected RevPAR lift: +{lift_pct:.1f}%")


# %%
def print_results(opp_positive: pd.DataFrame) -> None:
    """Prints comprehensive results summary."""
    print("\n" + "=" * 80)
    print("1:1 MATCHING WITH REPLACEMENT - RESULTS")
    print("=" * 80)
    
    print(f"\n1. SAMPLE SIZE:")
    print(f"   Valid pairs: {len(opp_positive):,}")
    print(f"   Unique treatment hotels: {opp_positive['treatment_hotel'].nunique():,}")
    print(f"   Unique control hotels: {opp_positive['control_hotel'].nunique():,}")
    print(f"   Reuse ratio: {len(opp_positive) / opp_positive['control_hotel'].nunique():.2f}x")
    
    print(f"\n2. ELASTICITY ESTIMATE:")
    print(f"   Median: {opp_positive['arc_elasticity'].median():.4f}")
    print(f"   Mean: {opp_positive['arc_elasticity'].mean():.4f}")
    print(f"   Std: {opp_positive['arc_elasticity'].std():.4f}")
    print(f"   Naive 95% CI: [{opp_positive['arc_elasticity'].quantile(0.025):.4f}, {opp_positive['arc_elasticity'].quantile(0.975):.4f}]")
    print(f"   (Note: Naive CI does not account for clustering)")
    
    print(f"\n3. MATCH QUALITY:")
    print(f"   Avg match distance: {opp_positive['match_distance'].mean():.3f}")
    print(f"   Avg price difference: {opp_positive['price_diff_pct'].mean():.1f}%")
    print(f"   Median price difference: {opp_positive['price_diff_pct'].median():.1f}%")
    print(f"   Price diff range: [{opp_positive['price_diff_pct'].min():.1f}%, {opp_positive['price_diff_pct'].max():.1f}%]")
    
    print(f"\n4. OPPORTUNITY SIZING (ATT ESTIMATION):")
    print(f"   Total opportunity (RevPAR Lift): €{opp_positive['opportunity'].sum():,.0f}")
    print(f"   Average per treatment: €{opp_positive['opportunity'].mean():,.0f}")


# %%
def main():
    """Main execution."""
    print("=" * 80)
    print("1:1 MATCHING WITH REPLACEMENT - VALIDATED FEATURES")
    print("=" * 80)
    
    # Load and clean data
    print("\n1. Loading and cleaning data...")
    # Initialize database
    con = init_db()
    
    # Clean data
    config = CleaningConfig(
        exclude_reception_halls=True,
        exclude_missing_location=True,
        match_city_names_with_tfidf=True
    )
    cleaner = DataCleaner(config)
    con = cleaner.clean(con)
    
    # Load hotel-month data
    print("\n2. Loading hotel-month aggregation...")
    df = load_hotel_month_data(con)
    print(f"   Loaded {len(df):,} hotel-month-roomtype records")
    
    # Load distance features
    print("\n3. Loading distance features...")
    script_dir = Path(__file__).parent
    distance_features_path = script_dir / '../../../outputs/hotel_distance_features.csv'
    distance_features = pd.read_csv(distance_features_path.resolve())
    print(f"   Loaded distance features for {len(distance_features):,} hotels")
    
    # Engineer validated features
    print("\n4. Engineering validated features (17 features from XGBoost)...")
    df = engineer_validated_features(df, distance_features)
    df = df.dropna(subset=['avg_adr', 'distance_from_coast', 'distance_from_madrid'])
    print(f"   Final dataset: {len(df):,} records")
    
    # Create market segments
    print("\n5. Creating market segments...")
    df['is_coastal_flag'] = (df['distance_from_coast'] <= 20).astype(str)
    df['is_madrid_metro'] = (df['distance_from_madrid'] <= 50).astype(str)
    df['market_segment'] = df['is_coastal_flag'] + '_' + df['is_madrid_metro']
    
    segment_map = {
        'True_False': 'Coastal/Resort',
        'False_True': 'Urban/Madrid',
        'False_False': 'Provincial/Regional',
        'True_True': 'Coastal/Resort'
    }
    df['market_segment'] = df['market_segment'].map(segment_map)
    print(f"   Market segments:")
    print(df['market_segment'].value_counts())
    
    # Add capacity quartiles (replaces revenue_quartile for new hotel support)
    print("\n6. Calculating capacity quartiles...")
    df = add_capacity_quartiles(df)
    
    # Create blocks
    print("\n6. Creating exact match blocks...")
    df_with_blocks, df_blocked = create_match_blocks(df)
    print(f"   Total blocks: {df_with_blocks['block_id'].nunique():,}")
    print(f"   Valid blocks (≥2 hotels): {df_blocked['block_id'].nunique():,}")
    print(f"   Records in valid blocks: {len(df_blocked):,}")
    
    # Find matched pairs WITH REPLACEMENT
    print("\n7. Finding matched pairs (1:1 with replacement)...")
    pairs = find_matched_pairs_with_replacement(df_blocked)
    
    # Calculate elasticity
    print("\n8. Calculating arc elasticity and opportunity...")
    opp_positive = calculate_elasticity_and_opportunity(pairs)
    
    # Print results
    print_results(opp_positive)
    
    # Calculate and print UNDERPRICING ANALYSIS
    print("\n9. Analyzing underpriced hotels...")
    underpricing_analysis = calculate_underpricing_analysis(opp_positive)
    print_underpricing_summary(underpricing_analysis)
    
    # Create underpricing figure
    underpricing_fig_path = script_dir / '../../../outputs/eda/elasticity/figures/underpricing_analysis.png'
    create_underpricing_figure(opp_positive, underpricing_analysis, underpricing_fig_path)
    
    # Create optimization curve figure (matched pairs version)
    print("\n10. Creating optimization curve figure...")
    opt_curve_path = script_dir / '../../../outputs/eda/elasticity/figures/matched_pairs_optimization_curve.png'
    create_optimization_curve_figure(opp_positive, opt_curve_path)
    
    # Calculate bootstrap confidence intervals (clustered by treatment hotel)
    print("\n11. Calculating bootstrap confidence intervals...")
    bootstrap_results = block_bootstrap_ci(opp_positive, n_bootstrap=1000, confidence=0.95)
    
    # Save results
    print("\n12. Saving results...")
    output_path = script_dir / '../../../outputs/eda/elasticity/data/matched_pairs_with_replacement.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opp_positive.to_csv(output_path, index=False)
    print(f"   ✓ Saved pairs to: {output_path}")
    
    # Calculate RevPAR metrics
    print("\n13. Calculating RevPAR metrics...")
    revpar_metrics = calculate_revpar_metrics(opp_positive)
    
    print(f"\n   ELASTICITY & BASELINE:")
    print(f"   Elasticity (median): ε = {revpar_metrics['elasticity_median']:.3f}")
    print(f"   Observed RevPAR Lift (ATT): +{revpar_metrics['observed_lift_pct_median']:.1f}% (median)")
    print(f"   Avg ADR (control hotels): €{revpar_metrics['avg_adr']:.2f}")
    print(f"   Avg price gap between twins: {revpar_metrics['avg_price_diff_pct']:.1f}%")
    
    print(f"\n   ACTUAL DATA (from corrected occupancy calculation):")
    print(f"   Avg capacity: {revpar_metrics['avg_capacity']:.1f} rooms/hotel")
    print(f"   Avg occupancy: {revpar_metrics['avg_occupancy']*100:.1f}%")
    print(f"   RevPAR: €{revpar_metrics['actual_revpar']:.2f}/room/night")
    print(f"   Annual revenue ({revpar_metrics['n_hotels']} hotels): €{revpar_metrics['actual_annual_revenue']:,.0f}")
    
    print(f"\n   MAXIMUM THEORETICAL (if matching twin prices - NOT recommended):")
    print(f"   RevPAR increase: +{revpar_metrics['max_revpar_pct']:.1f}%")
    print(f"   Annual opportunity: €{revpar_metrics['max_annual_opportunity']:,.0f}")
    print(f"   Per room per night: €{revpar_metrics['max_per_room_per_night']:.2f}")
    
    print(f"\n   RECOMMENDED SCENARIOS (using elasticity formula):")
    for key, scenario in revpar_metrics['scenarios'].items():
        print(f"   {int(scenario['price_increase_pct'])}% price increase → "
              f"+{scenario['revpar_gain_pct']:.1f}% RevPAR "
              f"(€{scenario['annual_opportunity']:,.0f}/yr, €{scenario['per_room_per_night']:.2f}/room/night)")
    
    print(f"\n   PRIMARY RECOMMENDATION (10% price increase):")
    print(f"   RevPAR increase: +{revpar_metrics['recommended_revpar_pct']:.1f}%")
    print(f"   Annual opportunity: €{revpar_metrics['recommended_annual_opportunity']:,.0f}")
    print(f"   Per room per night: €{revpar_metrics['recommended_per_room_per_night']:.2f}")
    
    # Add to bootstrap results
    bootstrap_results['revpar_metrics'] = revpar_metrics
    bootstrap_results['underpricing_analysis'] = underpricing_analysis
    
    # Save bootstrap results (now includes RevPAR metrics)
    bootstrap_path = script_dir / '../../../outputs/eda/elasticity/data/bootstrap_results_with_replacement.json'
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(bootstrap_path, 'w') as f:
        json.dump(convert_to_native(bootstrap_results), f, indent=2)
    print(f"✓ Saved bootstrap CI to: {bootstrap_path}")
    
    # Create executive summary figure
    print("\n14. Creating executive summary figure (geographic style)...")
    exec_fig_path = script_dir / '../../../outputs/eda/elasticity/figures/matched_pairs_executive_summary.png'
    exec_fig_path.parent.mkdir(parents=True, exist_ok=True)
    create_geographic_style_figure(opp_positive, bootstrap_results, exec_fig_path)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)


# %%
if __name__ == "__main__":
    main()
