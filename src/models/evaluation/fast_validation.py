"""
Fast Vectorized Validation Framework.

Uses vectorized operations instead of per-hotel loops for 10-100x speedup.
Validates directly using portfolio analysis results.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.models.evaluation.portfolio_analysis import analyze_portfolio, HotelCategory


@dataclass
class FastValidationResults:
    """Fast validation results."""
    
    # Coverage
    n_hotel_weeks: int
    n_weeks: int
    
    # By category
    category_counts: Dict[str, int]
    category_pcts: Dict[str, float]
    
    # Recommendation bounds
    within_10pct: float
    within_20pct: float
    within_30pct: float
    
    # By season
    seasonal_stats: pd.DataFrame
    
    # Distribution stats
    price_change_mean: float
    price_change_std: float
    price_change_median: float
    revpar_lift_mean: float
    
    # Validation scores
    peer_validation_score: float
    signal_consistency_score: float
    
    # Raw data
    all_results: pd.DataFrame


def get_validation_weeks_fast() -> List[Tuple[date, date, str]]:
    """Get (target_start, as_of_date, season) tuples."""
    return [
        # High season (July-August 2024)
        (date(2024, 7, 8), date(2024, 6, 24), "high"),
        (date(2024, 7, 29), date(2024, 7, 15), "high"),
        (date(2024, 8, 12), date(2024, 7, 29), "high"),
        # Low season (Winter)
        (date(2024, 1, 29), date(2024, 1, 15), "low"),
        (date(2024, 2, 12), date(2024, 1, 29), "low"),
        (date(2024, 11, 11), date(2024, 10, 28), "low"),
        # Shoulder season
        (date(2024, 3, 25), date(2024, 3, 11), "shoulder"),
        (date(2024, 4, 22), date(2024, 4, 8), "shoulder"),
        (date(2024, 9, 23), date(2024, 9, 9), "shoulder"),
        (date(2024, 10, 14), date(2024, 9, 30), "shoulder"),
    ]


def run_fast_validation(
    con,
    max_hotels_per_week: int = 300
) -> FastValidationResults:
    """
    Run fast vectorized validation across multiple weeks.
    
    Uses portfolio analysis which is already vectorized.
    """
    weeks = get_validation_weeks_fast()
    all_results = []
    
    print("Running fast validation across 10 weeks...")
    
    for target_start, as_of, season in weeks:
        target_dates = [target_start + timedelta(days=i) for i in range(7)]
        
        try:
            analysis = analyze_portfolio(
                con, target_dates, as_of, 
                max_hotels=max_hotels_per_week
            )
            df = analysis.hotel_results.copy()
            df['week'] = target_start.strftime('%Y-W%V')
            df['season'] = season
            all_results.append(df)
            print(f"  ✓ {target_start}: {len(df)} hotels")
        except Exception as e:
            print(f"  ✗ {target_start}: {e}")
            continue
    
    if len(all_results) == 0:
        raise ValueError("No weeks could be analyzed")
    
    # Combine all results - VECTORIZED
    combined = pd.concat(all_results, ignore_index=True)
    
    # Calculate metrics - ALL VECTORIZED
    n_hotel_weeks = len(combined)
    n_weeks = len(all_results)
    
    # Category counts
    category_counts = combined['category'].value_counts().to_dict()
    category_pcts = (combined['category'].value_counts(normalize=True) * 100).to_dict()
    
    # Recommendation bounds - VECTORIZED
    change_abs = combined['change_pct'].abs()
    within_10 = (change_abs <= 10).mean() * 100
    within_20 = (change_abs <= 20).mean() * 100
    within_30 = (change_abs <= 30).mean() * 100
    
    # Seasonal stats - VECTORIZED GROUP BY
    seasonal_stats = combined.groupby('season').agg({
        'change_pct': ['mean', 'std', 'count'],
        'revpar_lift_pct': 'mean',
        'category': lambda x: (x == 'underpriced').mean() * 100
    }).round(2)
    seasonal_stats.columns = ['avg_change', 'std_change', 'n_hotels', 'avg_lift', 'pct_underpriced']
    
    # Distribution stats - VECTORIZED
    price_change_mean = combined['change_pct'].mean()
    price_change_std = combined['change_pct'].std()
    price_change_median = combined['change_pct'].median()
    revpar_lift_mean = combined['revpar_lift_pct'].mean()
    
    # Peer validation - VECTORIZED
    raise_mask = combined['change_pct'] > 2
    if raise_mask.sum() > 0:
        peer_val = (combined.loc[raise_mask, 'peer_revpar'] > 
                   combined.loc[raise_mask, 'current_revpar']).mean() * 100
    else:
        peer_val = 0.0
    
    # Signal consistency - VECTORIZED
    # RevPAR gap should negatively correlate with price recommendation
    revpar_gap = (combined['current_revpar'] - combined['peer_revpar']) / combined['peer_revpar'].clip(lower=1)
    valid_mask = ~(revpar_gap.isna() | combined['change_pct'].isna())
    if valid_mask.sum() > 10:
        corr, _ = stats.pearsonr(revpar_gap[valid_mask], combined.loc[valid_mask, 'change_pct'])
        signal_consistency = max(0, (1 - abs(corr + 0.3)) * 100)  # Expect ~-0.3 correlation
    else:
        signal_consistency = 0.0
    
    return FastValidationResults(
        n_hotel_weeks=n_hotel_weeks,
        n_weeks=n_weeks,
        category_counts=category_counts,
        category_pcts=category_pcts,
        within_10pct=within_10,
        within_20pct=within_20,
        within_30pct=within_30,
        seasonal_stats=seasonal_stats,
        price_change_mean=price_change_mean,
        price_change_std=price_change_std,
        price_change_median=price_change_median,
        revpar_lift_mean=revpar_lift_mean,
        peer_validation_score=peer_val,
        signal_consistency_score=signal_consistency,
        all_results=combined
    )


def create_fast_validation_plots(
    results: FastValidationResults,
    output_dir: Path
) -> None:
    """Create validation plots efficiently."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = results.all_results
    
    # Use vectorized matplotlib operations
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Price change histogram - vectorized
    ax = axes[0, 0]
    ax.hist(df['change_pct'].clip(-50, 50), bins=50, color='steelblue', 
            edgecolor='none', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.axvline(results.price_change_mean, color='green', linestyle='--', lw=2,
               label=f'Mean: {results.price_change_mean:+.1f}%')
    ax.set_xlabel('Price Change (%)')
    ax.set_ylabel('Count')
    ax.set_title('Price Change Distribution')
    ax.legend()
    
    # 2. By category - vectorized value_counts
    ax = axes[0, 1]
    cats = ['underpriced', 'optimal', 'overpriced']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    counts = [results.category_counts.get(c, 0) for c in cats]
    ax.bar(cats, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Hotels by Category')
    for i, (c, v) in enumerate(zip(cats, counts)):
        ax.text(i, v + 10, f'{v}\n({results.category_pcts.get(c, 0):.1f}%)', 
                ha='center', fontsize=10)
    
    # 3. Seasonal comparison - from pre-computed
    ax = axes[0, 2]
    seasons = results.seasonal_stats.index.tolist()
    lifts = results.seasonal_stats['avg_lift'].values
    colors_season = {'high': '#e74c3c', 'low': '#3498db', 'shoulder': '#f39c12'}
    bars = ax.bar(seasons, lifts, color=[colors_season.get(s, 'gray') for s in seasons],
                  edgecolor='black')
    ax.set_ylabel('Avg RevPAR Lift (%)')
    ax.set_title('RevPAR Lift by Season')
    ax.axhline(0, color='gray', linestyle='--')
    
    # 4. Bounds check
    ax = axes[1, 0]
    bounds = ['±10%', '±20%', '±30%']
    vals = [results.within_10pct, results.within_20pct, results.within_30pct]
    colors_bound = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.bar(bounds, vals, color=colors_bound, edgecolor='black')
    ax.set_ylabel('% of Recommendations')
    ax.set_title('Recommendations Within Bounds')
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11)
    
    # 5. Current vs Peer RevPAR scatter - sampled for speed
    ax = axes[1, 1]
    sample = df.sample(min(1000, len(df)))  # Sample for performance
    ax.scatter(sample['current_revpar'], sample['peer_revpar'], 
              alpha=0.3, s=15, c='steelblue')
    max_val = max(sample['current_revpar'].max(), sample['peer_revpar'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Equal')
    ax.set_xlabel('Current RevPAR (€)')
    ax.set_ylabel('Peer RevPAR (€)')
    ax.set_title('Current vs Peer RevPAR')
    
    # 6. Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
VALIDATION SUMMARY
{'═' * 35}

DATA COVERAGE
  Hotel-weeks: {results.n_hotel_weeks:,}
  Weeks: {results.n_weeks}

RECOMMENDATION BOUNDS
  Within ±10%: {results.within_10pct:.1f}%
  Within ±20%: {results.within_20pct:.1f}%
  Within ±30%: {results.within_30pct:.1f}%

PRICE CHANGE STATS
  Mean: {results.price_change_mean:+.1f}%
  Median: {results.price_change_median:+.1f}%
  Std: {results.price_change_std:.1f}%

EXPECTED LIFT
  Avg RevPAR lift: {results.revpar_lift_mean:+.1f}%

VALIDATION SCORES
  Peer validation: {results.peer_validation_score:.1f}%
  Signal consistency: {results.signal_consistency_score:.1f}%
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('FAST VALIDATION DASHBOARD', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fast_validation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved fast validation to {output_dir / 'fast_validation.png'}")


def print_fast_summary(results: FastValidationResults) -> None:
    """Print fast validation summary."""
    print("\n" + "=" * 70)
    print("FAST VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\nData: {results.n_hotel_weeks:,} hotel-weeks across {results.n_weeks} weeks")
    
    print("\nCATEGORY DISTRIBUTION")
    print("-" * 40)
    for cat in ['underpriced', 'optimal', 'overpriced']:
        n = results.category_counts.get(cat, 0)
        pct = results.category_pcts.get(cat, 0)
        print(f"  {cat.title():12} {n:5,} ({pct:.1f}%)")
    
    print("\nRECOMMENDATION BOUNDS")
    print("-" * 40)
    print(f"  Within ±10%: {results.within_10pct:.1f}%")
    print(f"  Within ±20%: {results.within_20pct:.1f}%")
    print(f"  Within ±30%: {results.within_30pct:.1f}%")
    
    print("\nSEASONAL ANALYSIS")
    print("-" * 40)
    print(results.seasonal_stats.to_string())
    
    print("\nVALIDATION SCORES")
    print("-" * 40)
    print(f"  Peer validation (raise recs backed by peer data): {results.peer_validation_score:.1f}%")
    print(f"  Signal consistency (RevPAR gap → recommendation): {results.signal_consistency_score:.1f}%")
    
    print("=" * 70)

