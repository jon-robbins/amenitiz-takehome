"""
Ultra-Fast SQL-Based Validation.

Performs ALL calculations in DuckDB for 100x speedup.
No Python loops - pure vectorized SQL.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SQLValidationResults:
    """Results from SQL-based validation."""
    n_hotel_weeks: int
    n_weeks: int
    
    # Category breakdown
    underpriced_pct: float
    optimal_pct: float
    overpriced_pct: float
    
    # Bounds
    within_10pct: float
    within_20pct: float
    within_30pct: float
    
    # Stats
    avg_price_change: float
    avg_revpar_lift: float
    
    # By season
    seasonal_df: pd.DataFrame
    
    # Raw data
    all_hotels: pd.DataFrame


def run_sql_validation(con, weeks: List[Tuple[date, date, str]] = None) -> SQLValidationResults:
    """
    Run ultra-fast SQL-based validation.
    
    Everything is computed in DuckDB - no Python loops!
    """
    if weeks is None:
        weeks = [
            # High season
            (date(2024, 7, 8), date(2024, 6, 24), "high"),
            (date(2024, 7, 29), date(2024, 7, 15), "high"),
            (date(2024, 8, 12), date(2024, 7, 29), "high"),
            # Low season
            (date(2024, 1, 29), date(2024, 1, 15), "low"),
            (date(2024, 2, 12), date(2024, 1, 29), "low"),
            (date(2024, 11, 11), date(2024, 10, 28), "low"),
            # Shoulder
            (date(2024, 3, 25), date(2024, 3, 11), "shoulder"),
            (date(2024, 4, 22), date(2024, 4, 8), "shoulder"),
            (date(2024, 9, 23), date(2024, 9, 9), "shoulder"),
            (date(2024, 10, 14), date(2024, 9, 30), "shoulder"),
        ]
    
    all_results = []
    
    print("Running SQL-based validation (ultra fast)...")
    
    for target_start, as_of, season in weeks:
        target_end = target_start + timedelta(days=6)
        
        # SINGLE SQL query that computes EVERYTHING for this week
        query = f"""
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
        
        -- Daily room bookings for this week
        daily_room_counts AS (
            SELECT 
                b.hotel_id,
                CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) as stay_date,
                COUNT(*) as rooms_booked
            FROM bookings b
            JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
            CROSS JOIN generate_series(0, 30) as n(n)
            WHERE b.status IN ('Booked', 'confirmed')
              AND CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) < b.departure_date
              AND CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) >= '{target_start}'
              AND CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) <= '{target_end}'
              AND b.created_at <= '{as_of}'
            GROUP BY b.hotel_id, CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE)
        ),
        
        -- Room nights sold in target week
        week_room_nights AS (
            SELECT 
                hotel_id,
                SUM(rooms_booked) as room_nights_sold
            FROM daily_room_counts
            GROUP BY hotel_id
        ),
        
        -- Hotel metrics for target period
        hotel_metrics AS (
            SELECT 
                b.hotel_id,
                AVG(CAST(b.total_price AS FLOAT) / NULLIF(b.departure_date - b.arrival_date, 0)) as adr,
                COUNT(DISTINCT b.id) as n_bookings
            FROM bookings b
            WHERE b.arrival_date >= '{target_start}'
              AND b.arrival_date <= '{target_end}'
              AND b.status IN ('Booked', 'confirmed')
              AND b.created_at <= '{as_of}'
            GROUP BY b.hotel_id
            HAVING COUNT(*) >= 1
        ),
        
        -- Calculate RevPAR using actual capacity from rooms table
        hotel_revpar AS (
            SELECT 
                hm.hotel_id,
                hm.adr as current_price,
                LEAST(CAST(COALESCE(wrn.room_nights_sold, 0) AS FLOAT) / 
                      NULLIF(COALESCE(hc.total_rooms, 10) * 7, 0), 1.0) as occupancy,
                hm.adr * LEAST(CAST(COALESCE(wrn.room_nights_sold, 0) AS FLOAT) / 
                              NULLIF(COALESCE(hc.total_rooms, 10) * 7, 0), 1.0) as revpar
            FROM hotel_metrics hm
            LEFT JOIN hotel_capacity hc ON hm.hotel_id = hc.hotel_id
            LEFT JOIN week_room_nights wrn ON hm.hotel_id = wrn.hotel_id
        ),
        
        -- Get hotel locations for peer matching
        hotel_locs AS (
            SELECT 
                hr.hotel_id,
                hr.current_price,
                hr.occupancy,
                hr.revpar,
                hl.city,
                hl.latitude,
                hl.longitude
            FROM hotel_revpar hr
            JOIN hotel_location hl ON hr.hotel_id = hl.hotel_id
            WHERE hl.latitude IS NOT NULL
        ),
        
        -- Calculate city-level peer stats (VECTORIZED peer comparison)
        city_peers AS (
            SELECT 
                city,
                AVG(current_price) as peer_price,
                AVG(occupancy) as peer_occupancy,
                AVG(revpar) as peer_revpar,
                COUNT(*) as n_peers
            FROM hotel_locs
            GROUP BY city
            HAVING COUNT(*) >= 2
        ),
        
        -- Join hotels with their peer stats
        hotel_with_peers AS (
            SELECT 
                h.hotel_id,
                h.current_price,
                h.occupancy,
                h.revpar,
                h.city,
                p.peer_price,
                p.peer_occupancy,
                p.peer_revpar,
                p.n_peers,
                -- Price gap
                (h.current_price - p.peer_price) / NULLIF(p.peer_price, 0) * 100 as price_gap_pct,
                -- RevPAR gap  
                (h.revpar - p.peer_revpar) / NULLIF(p.peer_revpar, 0) * 100 as revpar_gap_pct
            FROM hotel_locs h
            JOIN city_peers p ON h.city = p.city
        ),
        
        -- Calculate recommendations
        recommendations AS (
            SELECT 
                *,
                -- Category based on RevPAR gap
                CASE 
                    WHEN revpar_gap_pct < -15 AND price_gap_pct < 0 THEN 'underpriced'
                    WHEN revpar_gap_pct < -15 AND price_gap_pct > 10 THEN 'overpriced'
                    ELSE 'optimal'
                END as category,
                -- Recommended price change
                CASE 
                    WHEN revpar_gap_pct < -15 AND price_gap_pct < 0 THEN 
                        LEAST(30, ABS(price_gap_pct) * 0.6)  -- Move toward peer price
                    WHEN revpar_gap_pct < -15 AND price_gap_pct > 10 THEN 
                        -LEAST(10, price_gap_pct * 0.3)  -- Reduce price
                    ELSE 0
                END as change_pct,
                -- Recommended price
                CASE 
                    WHEN revpar_gap_pct < -15 AND price_gap_pct < 0 THEN 
                        current_price * (1 + LEAST(0.3, ABS(price_gap_pct/100) * 0.6))
                    WHEN revpar_gap_pct < -15 AND price_gap_pct > 10 THEN 
                        current_price * (1 - LEAST(0.1, price_gap_pct/100 * 0.3))
                    ELSE current_price
                END as recommended_price
            FROM hotel_with_peers
        ),
        
        -- Calculate expected RevPAR (using elasticity -0.4)
        final AS (
            SELECT 
                *,
                -- Expected occupancy after price change (elasticity = -0.4)
                CASE 
                    WHEN category = 'underpriced' THEN 
                        GREATEST(0.1, occupancy * (1 - 0.4 * (change_pct / 100)))
                    WHEN category = 'overpriced' THEN 
                        -- Moving toward peer occupancy
                        occupancy + (peer_occupancy - occupancy) * 0.5
                    ELSE occupancy
                END as expected_occupancy,
                -- Expected RevPAR
                CASE 
                    WHEN category = 'underpriced' THEN 
                        recommended_price * GREATEST(0.1, occupancy * (1 - 0.4 * (change_pct / 100)))
                    WHEN category = 'overpriced' THEN 
                        recommended_price * (occupancy + (peer_occupancy - occupancy) * 0.5)
                    ELSE revpar
                END as expected_revpar
            FROM recommendations
        )
        
        SELECT 
            hotel_id,
            current_price,
            occupancy as current_occupancy,
            revpar as current_revpar,
            peer_price,
            peer_occupancy,
            peer_revpar,
            price_gap_pct,
            revpar_gap_pct,
            category,
            change_pct,
            recommended_price,
            expected_occupancy,
            expected_revpar,
            (expected_revpar - revpar) / NULLIF(revpar, 0) * 100 as revpar_lift_pct
        FROM final
        WHERE current_price > 0
        """
        
        try:
            df = con.execute(query).fetchdf()
            df['week'] = target_start.strftime('%Y-W%V')
            df['season'] = season
            all_results.append(df)
            print(f"  ✓ {target_start}: {len(df)} hotels")
        except Exception as e:
            print(f"  ✗ {target_start}: {e}")
            continue
    
    if len(all_results) == 0:
        raise ValueError("No weeks could be analyzed")
    
    # Combine - single concat operation
    combined = pd.concat(all_results, ignore_index=True)
    
    # Calculate all metrics - vectorized
    n_hotel_weeks = len(combined)
    n_weeks = len(all_results)
    
    # Category percentages
    cat_counts = combined['category'].value_counts(normalize=True) * 100
    underpriced_pct = cat_counts.get('underpriced', 0)
    optimal_pct = cat_counts.get('optimal', 0)
    overpriced_pct = cat_counts.get('overpriced', 0)
    
    # Bounds
    change_abs = combined['change_pct'].abs()
    within_10 = (change_abs <= 10).mean() * 100
    within_20 = (change_abs <= 20).mean() * 100
    within_30 = (change_abs <= 30).mean() * 100
    
    # Stats
    avg_change = combined['change_pct'].mean()
    avg_lift = combined['revpar_lift_pct'].mean()
    
    # Seasonal
    seasonal_df = combined.groupby('season').agg({
        'hotel_id': 'count',
        'change_pct': 'mean',
        'revpar_lift_pct': 'mean',
        'category': lambda x: (x == 'underpriced').mean() * 100
    }).rename(columns={
        'hotel_id': 'n_hotels',
        'change_pct': 'avg_change',
        'revpar_lift_pct': 'avg_lift',
        'category': 'pct_underpriced'
    })
    
    return SQLValidationResults(
        n_hotel_weeks=n_hotel_weeks,
        n_weeks=n_weeks,
        underpriced_pct=underpriced_pct,
        optimal_pct=optimal_pct,
        overpriced_pct=overpriced_pct,
        within_10pct=within_10,
        within_20pct=within_20,
        within_30pct=within_30,
        avg_price_change=avg_change,
        avg_revpar_lift=avg_lift,
        seasonal_df=seasonal_df,
        all_hotels=combined
    )


def create_sql_validation_plots(results: SQLValidationResults, output_dir: Path) -> None:
    """Create validation plots from SQL results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = results.all_hotels
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Price change histogram
    ax = axes[0, 0]
    ax.hist(df['change_pct'].clip(-40, 40), bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.axvline(results.avg_price_change, color='green', linestyle='--', lw=2)
    ax.set_xlabel('Price Change (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Price Changes (mean: {results.avg_price_change:+.1f}%)')
    
    # 2. Category distribution
    ax = axes[0, 1]
    cats = ['underpriced', 'optimal', 'overpriced']
    pcts = [results.underpriced_pct, results.optimal_pct, results.overpriced_pct]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    ax.bar(cats, pcts, color=colors, edgecolor='black')
    ax.set_ylabel('% of Hotels')
    ax.set_title('Category Distribution')
    for i, v in enumerate(pcts):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 3. Bounds check
    ax = axes[0, 2]
    bounds = ['±10%', '±20%', '±30%']
    vals = [results.within_10pct, results.within_20pct, results.within_30pct]
    ax.bar(bounds, vals, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    ax.set_ylabel('% Within Bounds')
    ax.set_title('Recommendation Bounds')
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 4. Seasonal comparison
    ax = axes[1, 0]
    seasons = results.seasonal_df.index.tolist()
    lifts = results.seasonal_df['avg_lift'].values
    colors_s = {'high': '#e74c3c', 'low': '#3498db', 'shoulder': '#f39c12'}
    ax.bar(seasons, lifts, color=[colors_s.get(s, 'gray') for s in seasons], edgecolor='black')
    ax.set_ylabel('Avg RevPAR Lift (%)')
    ax.set_title('RevPAR Lift by Season')
    ax.axhline(0, color='gray', linestyle='--')
    
    # 5. Current vs Peer RevPAR
    ax = axes[1, 1]
    sample = df.sample(min(2000, len(df)))
    ax.scatter(sample['current_revpar'].clip(0, 500), sample['peer_revpar'].clip(0, 500),
              alpha=0.3, s=10, c='steelblue')
    ax.plot([0, 500], [0, 500], 'r--')
    ax.set_xlabel('Current RevPAR (€)')
    ax.set_ylabel('Peer RevPAR (€)')
    ax.set_title('Current vs Peer RevPAR')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
VALIDATION SUMMARY
{'═' * 30}

Hotels: {results.n_hotel_weeks:,}
Weeks: {results.n_weeks}

CATEGORIES
  Underpriced: {results.underpriced_pct:.1f}%
  Optimal: {results.optimal_pct:.1f}%
  Overpriced: {results.overpriced_pct:.1f}%

BOUNDS
  ±10%: {results.within_10pct:.1f}%
  ±20%: {results.within_20pct:.1f}%
  ±30%: {results.within_30pct:.1f}%

IMPACT
  Avg change: {results.avg_price_change:+.1f}%
  Avg lift: {results.avg_revpar_lift:+.1f}%
"""
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.suptitle('SQL-BASED VALIDATION (Ultra Fast)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'sql_validation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved to {output_dir / 'sql_validation.png'}")


def print_sql_summary(results: SQLValidationResults) -> None:
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("SQL VALIDATION SUMMARY (Ultra Fast)")
    print("=" * 60)
    
    print(f"\nData: {results.n_hotel_weeks:,} hotel-weeks across {results.n_weeks} weeks")
    
    print("\nCATEGORIES")
    print(f"  Underpriced: {results.underpriced_pct:.1f}%")
    print(f"  Optimal:     {results.optimal_pct:.1f}%")
    print(f"  Overpriced:  {results.overpriced_pct:.1f}%")
    
    print("\nBOUNDS")
    print(f"  Within ±10%: {results.within_10pct:.1f}%")
    print(f"  Within ±20%: {results.within_20pct:.1f}%")
    print(f"  Within ±30%: {results.within_30pct:.1f}%")
    
    print("\nIMPACT")
    print(f"  Avg price change: {results.avg_price_change:+.1f}%")
    print(f"  Avg RevPAR lift:  {results.avg_revpar_lift:+.1f}%")
    
    print("\nSEASONAL")
    print(results.seasonal_df.round(1).to_string())
    
    print("=" * 60)

