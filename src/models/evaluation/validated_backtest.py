"""
Validated Backtest Using Matched Pairs Methodology.

Uses the SAME methodology validated in the elasticity analysis:
1. Load matched pairs (validated at monthly level)
2. For each underperforming hotel, calculate counterfactual:
   - "If they had priced like their Twin, what would RevPAR have been?"
3. Use validated arc elasticity (ε ≈ -0.39) from matched pairs
4. Compare counterfactual to actual outcome

IMPORTANT: Market-Level vs Individual-Level Elasticity
=======================================================
The elasticity ε = -0.39 is a MARKET AVERAGE. Individual hotels vary:
- Some are MORE elastic (competitive markets, commoditized product)
- Some are LESS elastic (unique properties, captive demand)
- Some are pricing well (optimal), some poorly (over/under)

The backtest validates by showing:
1. The DISTRIBUTION of outcomes (not just averages)
2. Which hotels beat/underperform the market elasticity
3. Segmentation by hotel characteristics

Hotels that beat market elasticity → likely underpriced (raise prices)
Hotels that underperform → likely overpriced (lower prices)
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Validated elasticity from matched pairs analysis
VALIDATED_ELASTICITY = -0.39  # ε = -0.39 (median from geographic matching)
ELASTICITY_CI_LOWER = -0.25   # 95% CI lower bound
ELASTICITY_CI_UPPER = -0.55   # 95% CI upper bound


@dataclass
class BacktestResults:
    """Results from validated backtest."""
    n_pairs: int
    n_positive_opportunity: int
    
    # Elasticity validation
    estimated_elasticity: float
    elasticity_within_ci: bool
    
    # Counterfactual analysis
    total_actual_revpar: float
    total_counterfactual_revpar: float
    total_opportunity: float
    opportunity_pct: float
    
    # Accuracy metrics
    direction_accuracy: float  # Did counterfactual predict direction?
    mean_absolute_error: float
    
    # By segment
    segment_results: pd.DataFrame
    
    # Raw data
    pair_results: pd.DataFrame


def load_validated_pairs(con) -> pd.DataFrame:
    """
    Load matched pairs from the validated matching process.
    
    Uses the same matching criteria from matched_pairs_geographic.py
    """
    # Try to load pre-computed pairs first
    pairs_path = Path('outputs/data/elasticity_results/matched_pairs_validated.csv')
    if pairs_path.exists():
        print(f"Loading pre-computed validated pairs from {pairs_path}")
        return pd.read_csv(pairs_path)
    
    # Otherwise compute pairs inline using the validated methodology
    print("Computing matched pairs using validated methodology...")
    
    query = """
    WITH 
    -- Get actual hotel capacity from rooms table (sum of all room types per hotel)
    hotel_room_types AS (
        SELECT DISTINCT
            b.hotel_id,
            CAST(br.room_id AS BIGINT) as room_id
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        WHERE b.status IN ('Booked', 'confirmed')
    ),
    hotel_capacity AS (
        SELECT 
            hrt.hotel_id,
            SUM(COALESCE(r.number_of_rooms, 1)) as total_rooms
        FROM hotel_room_types hrt
        LEFT JOIN rooms r ON hrt.room_id = r.id
        GROUP BY hrt.hotel_id
    ),
    
    hotel_month AS (
        SELECT 
            b.hotel_id,
            DATE_TRUNC('month', b.arrival_date) as month,
            AVG(b.total_price / NULLIF(b.departure_date - b.arrival_date, 0)) as adr,
            COUNT(DISTINCT b.id) as n_bookings,
            hl.city,
            hl.latitude,
            hl.longitude
        FROM bookings b
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('Booked', 'confirmed')
          AND b.arrival_date >= '2024-01-01'
          AND hl.latitude IS NOT NULL
        GROUP BY b.hotel_id, DATE_TRUNC('month', b.arrival_date), hl.city, hl.latitude, hl.longitude
        HAVING COUNT(*) >= 3 AND AVG(b.total_price / NULLIF(b.departure_date - b.arrival_date, 0)) > 0
    ),
    
    -- Calculate occupancy using actual capacity from rooms table
    hotel_month_occ AS (
        SELECT 
            hm.*,
            COALESCE(hc.total_rooms, 10) as total_rooms,
            LEAST(CAST(hm.n_bookings AS FLOAT) / NULLIF(COALESCE(hc.total_rooms, 10) * 30, 0), 1.0) as occupancy,
            hm.adr * LEAST(CAST(hm.n_bookings AS FLOAT) / NULLIF(COALESCE(hc.total_rooms, 10) * 30, 0), 1.0) as revpar
        FROM hotel_month hm
        LEFT JOIN hotel_capacity hc ON hm.hotel_id = hc.hotel_id
    ),
    
    -- Create pairs within same city and month
    pairs AS (
        SELECT 
            h1.hotel_id as hotel_a,
            h2.hotel_id as hotel_b,
            h1.month,
            h1.city,
            h1.adr as price_a,
            h2.adr as price_b,
            h1.occupancy as occ_a,
            h2.occupancy as occ_b,
            h1.revpar as revpar_a,
            h2.revpar as revpar_b,
            -- Distance between hotels (km)
            111.0 * SQRT(
                POW(h1.latitude - h2.latitude, 2) + 
                POW((h1.longitude - h2.longitude) * COS(RADIANS((h1.latitude + h2.latitude) / 2)), 2)
            ) as distance_km
        FROM hotel_month_occ h1
        JOIN hotel_month_occ h2 
            ON h1.city = h2.city 
            AND h1.month = h2.month
            AND h1.hotel_id < h2.hotel_id  -- Avoid duplicates
        WHERE h1.occupancy > 0.01 AND h2.occupancy > 0.01
    )
    
    SELECT *
    FROM pairs
    WHERE distance_km < 10  -- Within 10km
    """
    
    return con.execute(query).fetchdf()


def calculate_arc_elasticity(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate arc elasticity using the same methodology as validated analysis.
    
    Arc elasticity (midpoint method):
    ε = (ΔQ / Q_avg) / (ΔP / P_avg)
    
    Handles both pre-computed pairs (from CSV) and raw pairs (from SQL).
    """
    df = pairs_df.copy()
    
    # Check if this is pre-computed (has arc_elasticity already)
    if 'arc_elasticity' in df.columns and 'high_price' in df.columns:
        # Pre-computed pairs from validated analysis
        # Rename columns to match expected format
        df['high_occ'] = df['high_occupancy']
        df['low_occ'] = df['low_occupancy']
        df['high_hotel'] = df['high_price_hotel']
        df['low_hotel'] = df['low_price_hotel']
        return df
    
    # Raw pairs need elasticity calculation
    if 'price_a' in df.columns:
        # Identify high/low price hotels in each pair
        df['high_price'] = df[['price_a', 'price_b']].max(axis=1)
        df['low_price'] = df[['price_a', 'price_b']].min(axis=1)
        df['high_occ'] = np.where(df['price_a'] > df['price_b'], df['occ_a'], df['occ_b'])
        df['low_occ'] = np.where(df['price_a'] > df['price_b'], df['occ_b'], df['occ_a'])
        df['high_hotel'] = np.where(df['price_a'] > df['price_b'], df['hotel_a'], df['hotel_b'])
        df['low_hotel'] = np.where(df['price_a'] > df['price_b'], df['hotel_b'], df['hotel_a'])
        
        # Arc elasticity calculation (same as validated methodology)
        df['price_avg'] = (df['high_price'] + df['low_price']) / 2
        df['occ_avg'] = (df['high_occ'] + df['low_occ']) / 2
        df['price_pct_change'] = (df['high_price'] - df['low_price']) / df['price_avg']
        df['occ_pct_change'] = (df['high_occ'] - df['low_occ']) / df['occ_avg']
        
        # Elasticity
        df['arc_elasticity'] = df['occ_pct_change'] / df['price_pct_change']
    
    return df


def calculate_counterfactual(pairs_df: pd.DataFrame, use_validated_elasticity: bool = True) -> pd.DataFrame:
    """
    Calculate counterfactual RevPAR.
    
    For the low-price hotel: "What if they charged the high price?"
    - Use elasticity to predict occupancy change
    - Calculate counterfactual RevPAR
    """
    df = pairs_df.copy()
    
    # Use validated elasticity or pair-specific
    if use_validated_elasticity:
        df['elasticity_used'] = VALIDATED_ELASTICITY
    else:
        # Filter to valid elasticity values (same as validated methodology)
        df = df[
            (df['arc_elasticity'] < 0) &
            (df['arc_elasticity'] > -5) &
            (df['occ_avg'] > 0.01)
        ]
        df['elasticity_used'] = df['arc_elasticity']
    
    # Counterfactual: If low-price hotel raised to high-price level
    df['counterfactual_occ'] = (
        df['low_occ'] * (1 + df['elasticity_used'] * df['price_pct_change'])
    ).clip(0, 1.0)
    
    # Current and counterfactual RevPAR
    df['current_revpar'] = df['low_price'] * df['low_occ']
    df['counterfactual_revpar'] = df['high_price'] * df['counterfactual_occ']
    df['opportunity'] = df['counterfactual_revpar'] - df['current_revpar']
    df['opportunity_pct'] = (df['opportunity'] / df['current_revpar'].clip(lower=1)) * 100
    
    return df


def classify_hotel_by_elasticity(row: pd.Series, market_elasticity: float) -> str:
    """
    Classify hotel based on how their observed elasticity compares to market.
    
    Hotels MORE elastic than market → likely overpriced (demand sensitive)
    Hotels LESS elastic than market → likely underpriced (captive demand)
    """
    hotel_elasticity = row['arc_elasticity']
    
    if hotel_elasticity < market_elasticity * 1.3:  # More elastic (e.g., -0.5 vs -0.39)
        return 'overpriced'  # Very price-sensitive, should lower
    elif hotel_elasticity > market_elasticity * 0.7:  # Less elastic (e.g., -0.3 vs -0.39)
        return 'underpriced'  # Price-insensitive, could raise
    else:
        return 'optimal'  # Close to market


def run_validated_backtest(con) -> BacktestResults:
    """
    Run backtest using validated methodology.
    
    This proves the model works by showing:
    1. Elasticity estimated from pairs matches validated value (MARKET LEVEL)
    2. Individual hotels VARY around this market average
    3. Hotels beating market elasticity = underpriced opportunities
    4. Hotels underperforming = overpriced, need to lower
    """
    print("Loading matched pairs...")
    pairs_raw = load_validated_pairs(con)
    print(f"  Found {len(pairs_raw):,} raw pairs")
    
    print("Calculating arc elasticity (validated methodology)...")
    pairs_with_elasticity = calculate_arc_elasticity(pairs_raw)
    
    # Filter valid pairs
    valid_pairs = pairs_with_elasticity[
        (pairs_with_elasticity['arc_elasticity'] < 0) &
        (pairs_with_elasticity['arc_elasticity'] > -5) &
        (pairs_with_elasticity['occ_avg'] > 0.01)
    ]
    print(f"  Valid pairs with elasticity: {len(valid_pairs):,}")
    
    # Calculate estimated elasticity - THIS IS THE MARKET AVERAGE
    estimated_elasticity = valid_pairs['arc_elasticity'].median()
    elasticity_std = valid_pairs['arc_elasticity'].std()
    elasticity_within_ci = ELASTICITY_CI_LOWER <= estimated_elasticity <= ELASTICITY_CI_UPPER
    
    print(f"\n  === MARKET-LEVEL ELASTICITY ===")
    print(f"  Estimated (median): {estimated_elasticity:.3f}")
    print(f"  Std deviation:      {elasticity_std:.3f}")
    print(f"  Validated benchmark: {VALIDATED_ELASTICITY:.3f}")
    print(f"  Within 95% CI: {elasticity_within_ci}")
    
    # Show DISTRIBUTION of individual hotel elasticities
    print(f"\n  === INDIVIDUAL HOTEL HETEROGENEITY ===")
    print(f"  10th percentile: {valid_pairs['arc_elasticity'].quantile(0.10):.3f}")
    print(f"  25th percentile: {valid_pairs['arc_elasticity'].quantile(0.25):.3f}")
    print(f"  50th percentile: {valid_pairs['arc_elasticity'].quantile(0.50):.3f}")
    print(f"  75th percentile: {valid_pairs['arc_elasticity'].quantile(0.75):.3f}")
    print(f"  90th percentile: {valid_pairs['arc_elasticity'].quantile(0.90):.3f}")
    
    # Classify hotels by their elasticity vs market
    valid_pairs['pricing_status'] = valid_pairs.apply(
        lambda x: classify_hotel_by_elasticity(x, estimated_elasticity), axis=1
    )
    
    print(f"\n  === PRICING CLASSIFICATION (based on elasticity) ===")
    status_counts = valid_pairs['pricing_status'].value_counts()
    for status, count in status_counts.items():
        pct = count / len(valid_pairs) * 100
        print(f"  {status.capitalize():12} {count:5,} ({pct:.1f}%)")
    
    print("\nCalculating counterfactuals...")
    pairs_cf = calculate_counterfactual(valid_pairs, use_validated_elasticity=True)
    
    # Positive opportunity only
    opp_positive = pairs_cf[pairs_cf['opportunity'] > 0]
    print(f"  Pairs with positive opportunity: {len(opp_positive):,}")
    
    # Calculate metrics
    total_actual = opp_positive['current_revpar'].sum()
    total_cf = opp_positive['counterfactual_revpar'].sum()
    total_opp = opp_positive['opportunity'].sum()
    opp_pct = (total_opp / total_actual) * 100 if total_actual > 0 else 0
    
    # Direction accuracy: Did CF correctly predict higher RevPAR for high-price hotel?
    pairs_cf['cf_predicts_high_better'] = pairs_cf['counterfactual_revpar'] > pairs_cf['current_revpar']
    pairs_cf['high_actually_better'] = (pairs_cf['high_price'] * pairs_cf['high_occ']) > (pairs_cf['low_price'] * pairs_cf['low_occ'])
    pairs_cf['direction_correct'] = pairs_cf['cf_predicts_high_better'] == pairs_cf['high_actually_better']
    direction_accuracy = pairs_cf['direction_correct'].mean() * 100
    
    # MAE - compare predicted occ change to actual
    pairs_cf['actual_occ_change'] = (pairs_cf['high_occ'] - pairs_cf['low_occ']) / pairs_cf['low_occ'].clip(lower=0.01)
    pairs_cf['predicted_occ_change'] = (pairs_cf['counterfactual_occ'] - pairs_cf['low_occ']) / pairs_cf['low_occ'].clip(lower=0.01)
    mae = (pairs_cf['predicted_occ_change'] - pairs_cf['actual_occ_change']).abs().mean() * 100
    
    # Show accuracy BY PRICING STATUS - this is the key insight!
    print(f"\n  === ACCURACY BY PRICING STATUS ===")
    for status in ['underpriced', 'optimal', 'overpriced']:
        status_df = pairs_cf[pairs_cf['pricing_status'] == status]
        if len(status_df) > 0:
            acc = status_df['direction_correct'].mean() * 100
            avg_opp = status_df['opportunity_pct'].mean()
            print(f"  {status.capitalize():12} accuracy: {acc:.1f}%, avg opportunity: {avg_opp:+.1f}%")
    
    # Segment analysis
    opp_positive['price_segment'] = pd.cut(
        opp_positive['low_price'],
        bins=[0, 50, 100, 150, 200, 1000],
        labels=['<€50', '€50-100', '€100-150', '€150-200', '>€200']
    )
    segment_results = opp_positive.groupby('price_segment', observed=True).agg({
        'opportunity': ['sum', 'mean', 'count'],
        'opportunity_pct': 'mean',
        'arc_elasticity': 'median'
    }).round(2)
    segment_results.columns = ['total_opp', 'avg_opp', 'n_pairs', 'avg_opp_pct', 'median_elasticity']
    
    return BacktestResults(
        n_pairs=len(valid_pairs),
        n_positive_opportunity=len(opp_positive),
        estimated_elasticity=estimated_elasticity,
        elasticity_within_ci=elasticity_within_ci,
        total_actual_revpar=total_actual,
        total_counterfactual_revpar=total_cf,
        total_opportunity=total_opp,
        opportunity_pct=opp_pct,
        direction_accuracy=direction_accuracy,
        mean_absolute_error=mae,
        segment_results=segment_results,
        pair_results=pairs_cf  # Include all pairs, not just positive
    )


def create_backtest_visualizations(results: BacktestResults, output_dir: Path) -> None:
    """Create backtest visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = results.pair_results
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Elasticity distribution
    ax = axes[0, 0]
    elasticities = df['arc_elasticity'].clip(-3, 0)
    ax.hist(elasticities, bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(results.estimated_elasticity, color='green', linestyle='--', lw=2,
               label=f'Estimated: {results.estimated_elasticity:.2f}')
    ax.axvline(VALIDATED_ELASTICITY, color='red', linestyle='--', lw=2,
               label=f'Validated: {VALIDATED_ELASTICITY:.2f}')
    ax.set_xlabel('Arc Elasticity (ε)')
    ax.set_ylabel('Count')
    ax.set_title('Elasticity Distribution from Pairs')
    ax.legend()
    
    # 2. Counterfactual vs Actual RevPAR
    ax = axes[0, 1]
    ax.scatter(df['current_revpar'], df['counterfactual_revpar'], 
              alpha=0.3, s=20, c='steelblue')
    max_val = max(df['current_revpar'].max(), df['counterfactual_revpar'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='No change')
    ax.set_xlabel('Current RevPAR (€)')
    ax.set_ylabel('Counterfactual RevPAR (€)')
    ax.set_title('Counterfactual vs Current RevPAR')
    ax.legend()
    
    # 3. Opportunity distribution
    ax = axes[0, 2]
    opp_clipped = df['opportunity_pct'].clip(-50, 100)
    ax.hist(opp_clipped, bins=40, color='coral', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='gray', linestyle='--')
    ax.axvline(opp_clipped.median(), color='green', linestyle='--',
               label=f'Median: {opp_clipped.median():+.1f}%')
    ax.set_xlabel('Opportunity (%)')
    ax.set_ylabel('Count')
    ax.set_title('RevPAR Opportunity Distribution')
    ax.legend()
    
    # 4. Validation metrics
    ax = axes[1, 0]
    metrics = ['Elasticity\n(within CI)', 'Direction\nAccuracy', 'Counterfactual\nPrecision']
    values = [
        100 if results.elasticity_within_ci else 0,
        results.direction_accuracy,
        max(0, 100 - results.mean_absolute_error * 2)
    ]
    colors = ['#2ecc71' if v >= 70 else '#f39c12' if v >= 50 else '#e74c3c' for v in values]
    ax.bar(metrics, values, color=colors, edgecolor='black')
    ax.axhline(70, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Score (%)')
    ax.set_title('Validation Metrics')
    ax.set_ylim(0, 100)
    for i, v in enumerate(values):
        ax.text(i, v + 2, f'{v:.0f}%', ha='center')
    
    # 5. By price segment
    ax = axes[1, 1]
    if not results.segment_results.empty:
        seg = results.segment_results
        ax.bar(range(len(seg)), seg['avg_opp_pct'].values, 
              color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(seg)))
        ax.set_xticklabels(seg.index, rotation=45, ha='right')
        ax.set_ylabel('Avg Opportunity (%)')
        ax.set_title('Opportunity by Price Segment')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
VALIDATED BACKTEST RESULTS
{'═' * 35}

SAMPLE SIZE
  Pairs analyzed: {results.n_pairs:,}
  With positive opp: {results.n_positive_opportunity:,}

ELASTICITY VALIDATION
  Estimated: {results.estimated_elasticity:.3f}
  Validated: {VALIDATED_ELASTICITY:.3f}
  Within CI: {'✓ YES' if results.elasticity_within_ci else '✗ NO'}

COUNTERFACTUAL ACCURACY
  Direction accuracy: {results.direction_accuracy:.1f}%
  MAE: {results.mean_absolute_error:.1f}%

OPPORTUNITY SIZING
  Total current: €{results.total_actual_revpar:,.0f}
  Total counterfactual: €{results.total_counterfactual_revpar:,.0f}
  Opportunity: €{results.total_opportunity:,.0f} ({results.opportunity_pct:+.1f}%)
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('VALIDATED BACKTEST (Using Matched Pairs Methodology)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'validated_backtest.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved to {output_dir / 'validated_backtest.png'}")


def print_backtest_summary(results: BacktestResults) -> None:
    """Print backtest summary."""
    print("\n" + "=" * 70)
    print("VALIDATED BACKTEST RESULTS")
    print("=" * 70)
    
    print("\n1. ELASTICITY VALIDATION")
    print("-" * 40)
    print(f"   Estimated from pairs: {results.estimated_elasticity:.3f}")
    print(f"   Validated benchmark:  {VALIDATED_ELASTICITY:.3f}")
    print(f"   95% CI: [{ELASTICITY_CI_UPPER:.2f}, {ELASTICITY_CI_LOWER:.2f}]")
    print(f"   Within CI: {'✓ YES' if results.elasticity_within_ci else '✗ NO'}")
    
    print("\n2. COUNTERFACTUAL ACCURACY")
    print("-" * 40)
    print(f"   Direction accuracy: {results.direction_accuracy:.1f}%")
    print(f"   Mean absolute error: {results.mean_absolute_error:.1f}%")
    
    print("\n3. OPPORTUNITY SIZING")
    print("-" * 40)
    print(f"   Pairs with opportunity: {results.n_positive_opportunity:,} / {results.n_pairs:,}")
    print(f"   Total current RevPAR: €{results.total_actual_revpar:,.0f}")
    print(f"   Counterfactual RevPAR: €{results.total_counterfactual_revpar:,.0f}")
    print(f"   Opportunity: €{results.total_opportunity:,.0f} ({results.opportunity_pct:+.1f}%)")
    
    print("\n4. BY PRICE SEGMENT")
    print("-" * 40)
    print(results.segment_results.to_string())
    
    print("\n" + "=" * 70)

