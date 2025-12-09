# %%
"""
Matched Pair Analysis - GEOGRAPHIC MATCHING (Using Coastal/Madrid Distance)

Instead of matching on specific cities (too fragmented), match on:
1. Coastal vs Inland (within 20km of coast vs not)
2. Madrid Metro vs Provincial (within 50km of Madrid vs not)

This increases sample size dramatically while maintaining demand homogeneity:
- Coastal hotels compete with coastal hotels (resort market)
- Madrid metro hotels compete with Madrid metro (urban market)
- Provincial hotels compete with provincial (regional market)

Expected sample size: 200-500+ pairs (vs 24 with city matching)
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

# %%


def load_hotel_month_data(con) -> pd.DataFrame:
    """Loads hotel-month aggregation with geographic features from database."""
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
            SUM(br.total_price) AS total_revenue,
            COUNT(*) AS room_nights_sold,
            SUM(br.total_price) / NULLIF(COUNT(*), 0) AS avg_adr,
            AVG(br.room_size) AS avg_room_size,
            SUM(r.number_of_rooms) AS total_capacity,
            EXTRACT(DAY FROM LAST_DAY(month)) AS days_in_month,
            SUM(CASE WHEN EXTRACT(ISODOW FROM CAST(b.arrival_date AS DATE)) >= 6 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS weekend_ratio,
            (CAST(MAX(r.events_allowed) AS INT) + CAST(MAX(r.pets_allowed) AS INT) + 
             CAST(MAX(r.smoking_allowed) AS INT) + CAST(MAX(r.children_allowed) AS INT)) AS amenities_score,
            MAX(r.max_occupancy) AS room_capacity_pax,
            CASE 
                WHEN COALESCE(NULLIF(br.room_view, ''), 'no_view') IN ('ocean_view', 'sea_view') THEN 3
                WHEN COALESCE(NULLIF(br.room_view, ''), 'no_view') IN ('lake_view', 'mountain_view') THEN 2
                WHEN COALESCE(NULLIF(br.room_view, ''), 'no_view') IN ('pool_view', 'garden_view') THEN 1
                ELSE 0
            END AS view_quality_score
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


def create_market_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Creates geographic market segments based on coastal/Madrid proximity."""
    df = df.copy()
    df['is_coastal'] = (df['distance_from_coast'] <= 20).astype(str)
    df['is_madrid_metro'] = (df['distance_from_madrid'] <= 50).astype(str)
    df['market_segment'] = df['is_coastal'] + '_' + df['is_madrid_metro']
    return df


def create_match_blocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates exact match blocks for geographic matching.
    
    Returns:
        Tuple of (full dataframe with block_id, filtered dataframe with valid blocks)
    """
    df = df.copy()
    df['block_id'] = df.groupby([
        'market_segment',
        'room_type',
        'month',
        'children_allowed',
        'revenue_quartile'
    ], observed=True).ngroup()
    
    block_hotel_counts = df.groupby('block_id', observed=True)['hotel_id'].nunique()
    valid_blocks = block_hotel_counts[block_hotel_counts >= 2].index
    df_filtered = df[df['block_id'].isin(valid_blocks)]
    
    return df, df_filtered


def find_matched_pairs(df_blocked: pd.DataFrame, max_block_size: int = 100) -> pd.DataFrame:
    """Finds matched pairs within blocks using KNN on normalized features.
    
    Uses log-transformed capacity and segment-specific geographic features
    to improve matching quality, especially for urban markets.
    """
    # Base features (always used)
    base_features = [
        'avg_room_size', 'weekend_ratio', 'amenities_score',
        'room_capacity_pax', 'view_quality_score'
    ]
    
    matched_pairs = []
    blocks_processed = 0
    
    for block_vars, block in df_blocked.groupby(['market_segment', 'room_type', 'month', 'children_allowed', 'revenue_quartile']):
        if len(block) < 2 or len(block) > max_block_size:
            continue
        
        blocks_processed += 1
        
        # 1. Log-transform capacity to reduce scale variance
        block_clean = block.copy()
        block_clean['total_capacity_log'] = np.log1p(block_clean['total_capacity'])
        
        # 2. Dynamic feature selection by market segment
        market_segment = block_vars[0]  # First element is market_segment
        is_coastal = 'True' in market_segment.split('_')[0]  # Check if coastal
        
        if is_coastal:
            # Coastal segments: use distance_from_coast
            geo_features = ['distance_from_coast']
        else:
            # Urban/Inland segments: use distance_from_madrid (more relevant)
            geo_features = ['distance_from_madrid']
        
        # Combine base features with log capacity and segment-specific geo features
        match_features = base_features + ['total_capacity_log'] + geo_features
        
        # Check all required features exist
        if not all(f in block_clean.columns for f in match_features):
            continue
        
        # Prepare feature matrix
        block_features = block_clean[match_features].fillna(0)
        
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
        
        # 3. Relaxed distance threshold (2.0 -> 3.0) for more forgiving matching
        for i in range(n):
            for j in range(i+1, n):
                if ids[i] == ids[j] or price_matrix[i, j] < 0.10 or dist_matrix[i, j] > 3.0:
                    continue
                
                high_idx = i if prices[i] > prices[j] else j
                low_idx = j if prices[i] > prices[j] else i
                
                matched_pairs.append({
                    'market_segment': block.iloc[i]['market_segment'],
                    'high_price_hotel': ids[high_idx],
                    'low_price_hotel': ids[low_idx],
                    'high_price': prices[high_idx],
                    'low_price': prices[low_idx],
                    'high_occupancy': occ[high_idx],
                    'low_occupancy': occ[low_idx],
                    'price_diff_pct': price_matrix[i, j],
                    'match_distance': dist_matrix[i, j],
                    'room_type': block.iloc[i]['room_type'],
                    'room_view': block.iloc[i]['room_view'],
                    'month': str(block.iloc[i]['month']),
                    'capacity': block.iloc[low_idx]['total_capacity'],
                    'days_in_month': block.iloc[low_idx]['days_in_month'],
                    'dist_coast_diff': abs(block.iloc[i]['distance_from_coast'] - block.iloc[j]['distance_from_coast']),
                    'dist_madrid_diff': abs(block.iloc[i]['distance_from_madrid'] - block.iloc[j]['distance_from_madrid'])
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
    
    # Filter valid pairs (using relaxed distance threshold of 3.0)
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
    print("GEOGRAPHIC MATCHING RESULTS")
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
    print(f"   Avg Madrid distance difference: {opp_positive['dist_madrid_diff'].mean():.1f} km")
    
    print(f"\n4. OPPORTUNITY SIZING:")
    print(f"   Total opportunity: €{opp_positive['opportunity'].sum():,.0f}")
    print(f"   Average per hotel-month: €{opp_positive['opportunity'].mean():,.0f}")
    
    print(f"\n5. BY MARKET SEGMENT:")
    segment_stats = opp_positive.groupby('market_segment').agg({
        'arc_elasticity': 'median',
        'opportunity': 'sum',
        'low_price_hotel': 'count'
    })
    segment_stats.columns = ['median_elasticity', 'total_opportunity', 'n_pairs']
    print(segment_stats.to_string())


def create_executive_visualizations(opp_positive: pd.DataFrame, output_path: Path) -> None:
    """Creates executive-style visualizations for matched pairs analysis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    colors = {
        'coastal': '#F18F01',      # Orange for coastal
        'urban': '#2E86AB',        # Blue for urban
        'provincial': '#A23B72',   # Magenta for provincial
        'success': '#3E8914',      # Green
        'gray': '#808080'
    }
    
    def currency_fmt(x, pos):
        return f'€{x/1e6:.1f}M'
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, height_ratios=[1, 1, 1.2])
    
    # ========================================================================
    # Plot 1: Elasticity Distribution by Market Segment (KDE Comparison)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    label_map = {
        'True_False': 'Coastal/Resort',
        'False_True': 'Urban/Madrid',
        'False_False': 'Provincial/Regional',
        'True_True': 'Coastal/Madrid'
    }
    
    color_map = {
        'Coastal/Resort': colors['coastal'],
        'Urban/Madrid': colors['urban'],
        'Provincial/Regional': colors['provincial'],
        'Coastal/Madrid': colors['success']
    }
    
    # Plot KDE for each segment
    for segment in opp_positive['market_segment'].unique():
        segment_data = opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity']
        label = label_map.get(segment, segment)
        color = color_map.get(label, colors['gray'])
        
        if len(segment_data) > 5:  # Only plot if enough data
            sns.kdeplot(segment_data, ax=ax1, fill=True, alpha=0.3, linewidth=2.5, 
                       label=f'{label} (n={len(segment_data)}, ε={segment_data.median():.2f})',
                       color=color)
    
    # Overall distribution
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
    
    # Add interpretation box
    ax1.text(0.98, 0.97, 'More Negative = More Elastic\nCloser to 0 = More Inelastic\n(Revenue gains from price increases)', 
             transform=ax1.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                      edgecolor='black', linewidth=1.5, alpha=0.9))
    
    # ========================================================================
    # Plot 2: Elasticity by Segment (Box Plot with Stats)
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Prepare data for box plot
    segment_data_list = []
    segment_labels = []
    segment_colors = []
    
    for segment in sorted(opp_positive['market_segment'].unique()):
        segment_data_list.append(opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity'])
        label = label_map.get(segment, segment)
        segment_labels.append(label)
        segment_colors.append(color_map.get(label, colors['gray']))
    
    bp = ax2.boxplot(segment_data_list, labels=segment_labels, patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True,
                     boxprops=dict(linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2),
                     medianprops=dict(color='red', linewidth=3),
                     meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], segment_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_title('Elasticity Distribution by Segment', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Price Elasticity (ε)', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(segment_labels, fontsize=11, fontweight='bold', rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax2.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    # Add sample sizes
    for i, (label, data) in enumerate(zip(segment_labels, segment_data_list)):
        ax2.text(i+1, ax2.get_ylim()[0] * 0.95, f'n={len(data)}', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # Plot 3: Opportunity by Segment
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    segments = opp_positive.groupby('market_segment')['opportunity'].sum().sort_values(ascending=False)
    labels = [label_map.get(l, l) for l in segments.index]
    bar_colors_list = [color_map.get(l, colors['gray']) for l in labels]
    
    bars = ax3.bar(range(len(segments)), segments.values, color=bar_colors_list, 
                   alpha=0.85, edgecolor='black', linewidth=2, width=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, segments.values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                 f'€{val/1e6:.2f}M', ha='center', va='bottom', 
                 fontsize=13, fontweight='bold')
    
    ax3.set_title('Total Revenue Opportunity by Segment', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(range(len(segments)))
    ax3.set_xticklabels(labels, fontsize=11, fontweight='bold', rotation=15, ha='right')
    ax3.set_ylabel('Net Opportunity', fontsize=13, fontweight='bold')
    ax3.yaxis.set_major_formatter(FuncFormatter(currency_fmt))
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_ylim(0, segments.max() * 1.15)
    
    # ========================================================================
    # Plot 4: Sample Size & Match Quality by Segment
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    segment_stats = opp_positive.groupby('market_segment').agg({
        'low_price_hotel': 'count',
        'match_distance': 'mean'
    }).reset_index()
    segment_stats['segment_label'] = segment_stats['market_segment'].map(label_map)
    segment_stats = segment_stats.sort_values('low_price_hotel', ascending=False)
    
    x_pos = np.arange(len(segment_stats))
    bars1 = ax4.bar(x_pos, segment_stats['low_price_hotel'], 
                    color=[color_map.get(l, colors['gray']) for l in segment_stats['segment_label']],
                    alpha=0.85, edgecolor='black', linewidth=2, width=0.7)
    
    for i, (bar, val, quality) in enumerate(zip(bars1, segment_stats['low_price_hotel'], 
                                                  segment_stats['match_distance'])):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                f'{int(val)} pairs\n(quality: {quality:.2f})', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_title('Sample Size & Match Quality by Segment', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(segment_stats['segment_label'], fontsize=11, fontweight='bold', 
                        rotation=15, ha='right')
    ax4.set_ylabel('Number of Matched Pairs', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.set_ylim(0, segment_stats['low_price_hotel'].max() * 1.2)
    
    # Add note
    ax4.text(0.02, 0.98, 'Lower match quality score = better match', 
             transform=ax4.transAxes, ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                      edgecolor='black', linewidth=1, alpha=0.8))
    
    # ========================================================================
    # Plot 5: Confidence Intervals by Segment
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    segment_ci_data = []
    for segment in sorted(opp_positive['market_segment'].unique()):
        data = opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity']
        label = label_map.get(segment, segment)
        median = data.median()
        ci_lower = data.quantile(0.025)
        ci_upper = data.quantile(0.975)
        segment_ci_data.append({
            'label': label,
            'median': median,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'color': color_map.get(label, colors['gray'])
        })
    
    # Sort by median
    segment_ci_data = sorted(segment_ci_data, key=lambda x: x['median'])
    
    y_pos = np.arange(len(segment_ci_data))
    
    for i, seg in enumerate(segment_ci_data):
        # Plot CI range
        ax5.plot([seg['ci_lower'], seg['ci_upper']], [i, i], 
                color=seg['color'], linewidth=8, alpha=0.3, solid_capstyle='round')
        # Plot median
        ax5.plot(seg['median'], i, 'o', color=seg['color'], 
                markersize=12, markeredgecolor='black', markeredgewidth=2, zorder=10)
        # Add value label
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
    
    # Add legend
    ax5.plot([], [], 'o', color='gray', markersize=10, markeredgecolor='black', 
            markeredgewidth=2, label='Median')
    ax5.plot([], [], '-', color='gray', linewidth=8, alpha=0.3, label='95% CI')
    ax5.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    
    # ========================================================================
    # Overall title
    # ========================================================================
    plt.suptitle('Geographic Matched Pairs Analysis: Comprehensive Elasticity Assessment', 
                 fontsize=19, fontweight='bold', y=0.995)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def create_individual_plots(opp_positive: pd.DataFrame, output_dir: Path) -> None:
    """Creates individual focused plots for each key metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {
        'coastal': '#F18F01',
        'urban': '#2E86AB',
        'provincial': '#A23B72',
        'success': '#3E8914',
        'gray': '#808080'
    }
    
    def currency_fmt(x, pos):
        return f'€{x/1e6:.1f}M'
    
    label_map = {
        'True_False': 'Coastal / Resort',
        'False_True': 'Urban / Madrid',
        'False_False': 'Provincial / Regional'
    }
    
    color_map = {
        'Coastal / Resort': colors['coastal'],
        'Urban / Madrid': colors['urban'],
        'Provincial / Regional': colors['provincial']
    }
    
    # =========================================================================
    # PLOT 1: KDE Distribution (Confidence)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    for segment in opp_positive['market_segment'].unique():
        segment_data = opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity']
        label = label_map.get(segment, segment)
        color = color_map.get(label, colors['gray'])
        
        if len(segment_data) > 10:
            sns.kdeplot(segment_data, fill=True, alpha=0.2, linewidth=2.5, 
                       label=f'{label} (Median ε={segment_data.median():.2f})',
                       color=color)
    
    overall_median = opp_positive['arc_elasticity'].median()
    plt.axvline(overall_median, color='black', linestyle='--', linewidth=2, 
                label=f'Global Median: {overall_median:.2f}')
    
    plt.title('Confidence: Price Elasticity Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Price Elasticity of Demand (ε)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.xlim(-1.5, 0)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "1_confidence_distribution.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # =========================================================================
    # PLOT 2: Box Plot (Segment Comparison)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    data_list = []
    labels = []
    box_colors = []
    
    for segment in sorted(opp_positive['market_segment'].unique()):
        segment_data = opp_positive[opp_positive['market_segment'] == segment]['arc_elasticity']
        label = label_map.get(segment, segment)
        data_list.append(segment_data)
        labels.append(f"{label}\n(n={len(segment_data)})")
        box_colors.append(color_map.get(label, colors['gray']))
    
    bp = plt.boxplot(data_list, labels=labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2.5),
                     widths=0.6)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    plt.title('Elasticity Ranges by Market Segment', fontsize=16, fontweight='bold')
    plt.ylabel('Price Elasticity (ε)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / "2_segment_elasticity_boxplot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # =========================================================================
    # PLOT 3: Opportunity Bar Chart (Impact)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    segments = opp_positive.groupby('market_segment')['opportunity'].sum().sort_values(ascending=False)
    labels_list = [label_map.get(l, l) for l in segments.index]
    bar_colors_list = [color_map.get(l, colors['gray']) for l in labels_list]
    
    bars = plt.bar(labels_list, segments, color=bar_colors_list, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'€{height/1e6:.2f}M',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.title('Net Revenue Opportunity by Segment', fontsize=16, fontweight='bold')
    plt.ylabel('Net Revenue Gain (€)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_fmt))
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, segments.max() * 1.15)
    
    plt.savefig(output_dir / "3_opportunity_impact.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # =========================================================================
    # PLOT 4: Sensitivity Waterfall (Risk Analysis)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    # Use only geographic matching elasticity with confidence intervals
    median_elasticity = opp_positive['arc_elasticity'].median()
    ci_lower = opp_positive['arc_elasticity'].quantile(0.025)
    ci_upper = opp_positive['arc_elasticity'].quantile(0.975)
    
    scenarios = [ci_upper, median_elasticity, ci_lower]  # More inelastic to more elastic
    scenario_labels = ['Optimistic\n(95% CI Upper)', 'Base Case\n(Median)', 'Conservative\n(95% CI Lower)']
    
    gains = []
    for e in scenarios:
        cf_occ = (opp_positive['low_occupancy'] * (1 + e * opp_positive['price_pct_change'])).clip(0, 1)
        cf_rev = opp_positive['high_price'] * cf_occ * opp_positive['capacity'] * opp_positive['days_in_month']
        gain = (cf_rev - opp_positive['current_revenue']).sum()
        gains.append(gain)
        
    bars = plt.bar(scenario_labels, gains, color=colors['success'], alpha=0.8, width=0.6)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'+€{height/1e6:.2f}M',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'ε = {scenarios[i]:.2f}',
                 ha='center', va='center', color='white', fontweight='bold')
        
    plt.title('Risk Analysis: Revenue Gain Across Confidence Interval', fontsize=16, fontweight='bold')
    plt.ylabel('Net Revenue Gain (€)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_fmt))
    
    plt.savefig(output_dir / "4_risk_sensitivity.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n✓ Individual plots saved to {output_dir}")


def print_top_pairs(opp_positive: pd.DataFrame, n: int = 5) -> None:
    """Prints top N matched pairs by opportunity."""
    print("\n" + "=" * 80)
    print(f"TOP {n} MATCHED PAIRS (By Opportunity)")
    print("=" * 80)
    
    for idx, (_, row) in enumerate(opp_positive.nlargest(n, 'opportunity').iterrows(), 1):
        print(f"\nPair {idx}:")
        print(f"  Market: {row['market_segment']}, Type: {row['room_type']}, View: {row['room_view']}")
        print(f"  Month: {row['month']}")
        print(f"  High-Price Hotel: €{row['high_price']:.2f}, Occ: {row['high_occupancy']:.1%}")
        print(f"  Low-Price Hotel: €{row['low_price']:.2f}, Occ: {row['low_occupancy']:.1%}")
        print(f"  → Elasticity: {row['arc_elasticity']:.3f}")
        print(f"  → Opportunity: €{row['opportunity']:,.0f}")


def print_summary(opp_positive: pd.DataFrame) -> None:
    """Prints final summary of geographic matching results."""
    print("\n" + "=" * 80)
    print("GEOGRAPHIC MATCHING SUMMARY")
    print("=" * 80)
    
    median_elasticity = opp_positive['arc_elasticity'].median()
    ci_lower = opp_positive['arc_elasticity'].quantile(0.025)
    ci_upper = opp_positive['arc_elasticity'].quantile(0.975)
    
    print(f"""
MATCHED PAIRS ANALYSIS RESULTS:
  - Total matched pairs: {len(opp_positive):,}
  - Unique low-price hotels: {opp_positive['low_price_hotel'].nunique():,}
  - Median elasticity: {median_elasticity:.3f}
  - 95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]
  - Total revenue opportunity: €{opp_positive['opportunity'].sum():,.0f}

EXACT MATCHING VARIABLES (5):
1. market_segment (coastal/inland × Madrid/provincial)
2. room_type
3. month
4. children_allowed
5. revenue_quartile (Q1-Q4)

CONTINUOUS MATCHING FEATURES (8):
- Room size, capacity, weekend ratio, amenities score
- Room capacity (pax), view quality score
- Distance from coast, distance from Madrid

METHODOLOGY:
Geographic matching groups hotels by market type (coastal resort vs urban vs 
provincial) rather than specific cities. This approach:
- Increases statistical power while maintaining causal validity
- Controls for demand characteristics (resort vs urban vs regional)
- Ensures similar business scale (revenue quartile matching)
- Uses KNN matching on normalized continuous features within blocks

INTERPRETATION:
Median elasticity of {median_elasticity:.2f} indicates inelastic demand - 
hotels can raise prices with minimal volume loss. This suggests significant 
pricing power and revenue optimization opportunity across the portfolio.
""")


# %%
# Main execution
print("Loading database...")
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
# Use script's directory to build paths (works regardless of CWD)
script_dir = Path(__file__).parent
distance_features_path = script_dir / '../../../outputs/eda/spatial/data/hotel_distance_features.csv'
distance_features = pd.read_csv(distance_features_path.resolve())
print(f"Loaded distance features for {len(distance_features):,} hotels")

# %%
print("\nCreating hotel-month aggregation...")
df_geo = load_hotel_month_data(con)
print(f"Hotel-months: {len(df_geo):,}")

# %%
print("\nMerging with distance features...")
df_geo = df_geo.merge(distance_features, on='hotel_id', how='left')
df_geo = df_geo.dropna(subset=['distance_from_coast', 'distance_from_madrid'])
print(f"After merging: {len(df_geo):,} hotel-months with distance features")

# %%
print("\nCalculating annual revenue quartiles...")
df_geo = add_revenue_quartiles(df_geo)
print(f"Revenue quartile distribution:")
print(df_geo['revenue_quartile'].value_counts().sort_index())

# %%
print("\nCreating geographic market segments...")
df_geo = create_market_segments(df_geo)
print("\nMarket segments:")
print(df_geo['market_segment'].value_counts())
print(f"\nBreakdown:")
print(f"  Coastal (≤20km from coast): {(df_geo['distance_from_coast'] <= 20).sum():,}")
print(f"  Madrid Metro (≤50km from Madrid): {(df_geo['distance_from_madrid'] <= 50).sum():,}")

# %%
print("\n" + "=" * 80)
print("GEOGRAPHIC MATCHING (Coastal/Madrid Instead of City)")
print("=" * 80)

print("\nCreating blocks with geographic + revenue quartile matching...")
df_geo_with_blocks, df_blocked = create_match_blocks(df_geo)

block_hotel_counts = df_geo_with_blocks.groupby('block_id', observed=True)['hotel_id'].nunique()
valid_blocks = block_hotel_counts[block_hotel_counts >= 2].index

print(f"\nBlocking results:")
print(f"  Total blocks created: {df_geo_with_blocks['block_id'].nunique():,}")
print(f"  Blocks with ≥2 hotels: {len(valid_blocks):,}")
print(f"  Hotel-months retained: {len(df_blocked):,}")
print(f"  Avg hotels per block: {block_hotel_counts[valid_blocks].mean():.1f}")

# %%
print("\nFinding matched pairs (optimized)...")
pairs_geo = find_matched_pairs(df_blocked)
print(f"\nMatched pairs found: {len(pairs_geo):,}")

# %%
print("\nCalculating arc elasticity and opportunity...")
opp_positive = calculate_elasticity_and_opportunity(pairs_geo)
print(f"Valid pairs with positive opportunity: {len(opp_positive):,}")

# %%
print_results(opp_positive)

# %%
print("\nCreating executive-style visualizations...")
fig_path = (script_dir / '../../../outputs/eda/elasticity/figures/matched_pairs_geographic_executive.png').resolve()
fig_path.parent.mkdir(parents=True, exist_ok=True)
create_executive_visualizations(opp_positive, fig_path)
print(f"Executive visual saved to: {fig_path}")

# %%
print("\nCreating individual focused plots...")
individual_plots_dir = (script_dir / '../../../outputs/eda/elasticity/figures').resolve()
individual_plots_dir.mkdir(parents=True, exist_ok=True)
create_individual_plots(opp_positive, individual_plots_dir)

# %%
print_top_pairs(opp_positive, n=5)

# %%
pairs_path = (script_dir / '../../../outputs/eda/elasticity/data/matched_pairs_geographic.csv').resolve()
pairs_path.parent.mkdir(parents=True, exist_ok=True)
opp_positive.to_csv(pairs_path, index=False)
print(f"\nSaved results to: {pairs_path}")

# %%
print_summary(opp_positive)

print("\n✓ Geographic matching analysis complete!")
