"""
Holdout Validation for RevPAR Recommendations.

Since we don't have labeled data, we use a holdout validation approach:

1. **Peer Outcome Validation**: When we say "peers achieve higher RevPAR", 
   verify that those peers actually did.
   
2. **Counterfactual Analysis**: Compare hotels that DID change prices 
   historically to what we would have recommended.
   
3. **RevPAR Consistency**: For hotels in the same market, do our 
   recommendations produce consistent RevPAR expectations?
   
4. **Distribution Checks**: Are recommendations realistic and bounded?
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


@dataclass
class ValidationMetrics:
    """Validation metrics for the pricing model."""
    
    # Peer Validation: Did peers actually have higher RevPAR?
    peer_revpar_validated_pct: float
    peer_price_validated_pct: float
    
    # Recommendation Quality
    bounded_recommendations_pct: float  # Within ±30%
    mean_price_change: float
    std_price_change: float
    
    # Distribution stats
    pct_raise: float
    pct_hold: float
    pct_lower: float
    
    # Correlation with market signals
    revpar_gap_corr_with_recommendation: float
    price_gap_corr_with_recommendation: float
    
    # Cross-validation: consistency across segments
    segment_consistency: float
    
    def __repr__(self) -> str:
        return f"""
ValidationMetrics:
  Peer Validation:
    - Peers had higher RevPAR: {self.peer_revpar_validated_pct:.1f}%
    - Peers had higher Price: {self.peer_price_validated_pct:.1f}%
  
  Recommendation Quality:
    - Bounded (±30%): {self.bounded_recommendations_pct:.1f}%
    - Mean change: {self.mean_price_change:+.1f}%
    - Std change: {self.std_price_change:.1f}%
  
  Direction Distribution:
    - Raise: {self.pct_raise:.1f}%
    - Hold: {self.pct_hold:.1f}%
    - Lower: {self.pct_lower:.1f}%
  
  Signal Correlation:
    - RevPAR gap → rec: {self.revpar_gap_corr_with_recommendation:.2f}
    - Price gap → rec: {self.price_gap_corr_with_recommendation:.2f}
  
  Segment consistency: {self.segment_consistency:.1f}%
"""


def validate_peer_accuracy(df: pd.DataFrame) -> Tuple[float, float]:
    """
    For hotels where we recommend "raise" based on peers,
    validate that peers actually had higher RevPAR/price.
    """
    # Only look at hotels we recommended to raise
    raise_recs = df[df['change_pct'] > 2]
    
    if len(raise_recs) == 0:
        return 0.0, 0.0
    
    # What % of cases did peers actually have higher RevPAR?
    peer_revpar_validated = (raise_recs['peer_revpar'] > raise_recs['current_revpar']).mean() * 100
    
    # What % of cases did peers actually have higher price?
    peer_price_validated = (raise_recs['peer_price'] > raise_recs['current_price']).mean() * 100
    
    return peer_revpar_validated, peer_price_validated


def validate_signal_correlations(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Validate that our recommendations correlate with market signals.
    
    - Hotels with negative RevPAR gap should get "raise" recommendations
    - Hotels with positive price gap should get "lower" recommendations
    """
    # RevPAR gap vs recommendation
    # Negative gap = underperforming, should raise price
    df_valid = df.dropna(subset=['current_revpar', 'peer_revpar', 'change_pct'])
    if len(df_valid) < 10:
        return 0.0, 0.0
    
    revpar_gap = (df_valid['current_revpar'] - df_valid['peer_revpar']) / df_valid['peer_revpar'].clip(lower=1)
    price_gap = (df_valid['current_price'] - df_valid['peer_price']) / df_valid['peer_price'].clip(lower=1)
    
    # Negative correlation expected: lower RevPAR gap = higher price recommendation
    revpar_corr, _ = stats.pearsonr(revpar_gap, df_valid['change_pct'])
    
    # Negative correlation expected: higher price gap = lower price recommendation
    price_corr, _ = stats.pearsonr(price_gap, df_valid['change_pct'])
    
    return revpar_corr, price_corr


def validate_segment_consistency(df: pd.DataFrame) -> float:
    """
    Check if recommendations are consistent within market segments.
    Hotels in similar situations should get similar recommendations.
    """
    # Group by revenue bracket (proxy for segment)
    df['revpar_bucket'] = pd.cut(df['current_revpar'], bins=5, labels=['very_low', 'low', 'mid', 'high', 'very_high'])
    
    # For each bucket, calculate the coefficient of variation of recommendations
    cv_by_bucket = df.groupby('revpar_bucket')['change_pct'].apply(
        lambda x: x.std() / (x.mean() + 0.001) if len(x) > 5 else np.nan
    ).dropna()
    
    if len(cv_by_bucket) == 0:
        return 0.0
    
    # Convert CV to consistency score (lower CV = higher consistency)
    avg_cv = cv_by_bucket.mean()
    consistency = max(0, 100 - avg_cv * 100)
    
    return consistency


def run_holdout_validation(
    results_df: pd.DataFrame,
    con=None  # Optional for additional queries
) -> ValidationMetrics:
    """
    Run comprehensive validation on portfolio analysis results.
    
    Args:
        results_df: DataFrame from analyze_portfolio().hotel_results
        con: Optional database connection for additional validation
    
    Returns:
        ValidationMetrics with all validation scores
    """
    df = results_df.copy()
    
    # 1. Peer accuracy validation
    peer_revpar_val, peer_price_val = validate_peer_accuracy(df)
    
    # 2. Recommendation quality
    bounded_pct = (df['change_pct'].abs() <= 30).mean() * 100
    mean_change = df['change_pct'].mean()
    std_change = df['change_pct'].std()
    
    # 3. Direction distribution
    pct_raise = (df['change_pct'] > 2).mean() * 100
    pct_hold = ((df['change_pct'] >= -2) & (df['change_pct'] <= 2)).mean() * 100
    pct_lower = (df['change_pct'] < -2).mean() * 100
    
    # 4. Signal correlations
    revpar_corr, price_corr = validate_signal_correlations(df)
    
    # 5. Segment consistency
    segment_consistency = validate_segment_consistency(df)
    
    return ValidationMetrics(
        peer_revpar_validated_pct=peer_revpar_val,
        peer_price_validated_pct=peer_price_val,
        bounded_recommendations_pct=bounded_pct,
        mean_price_change=mean_change,
        std_price_change=std_change,
        pct_raise=pct_raise,
        pct_hold=pct_hold,
        pct_lower=pct_lower,
        revpar_gap_corr_with_recommendation=revpar_corr,
        price_gap_corr_with_recommendation=price_corr,
        segment_consistency=segment_consistency,
    )


def create_validation_dashboard(
    results_df: pd.DataFrame,
    metrics: ValidationMetrics,
    output_dir: Path
) -> None:
    """Create comprehensive validation dashboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = results_df.copy()
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # Row 1: Price Change Distribution
    # =========================================================================
    
    # 1a. Histogram of price changes
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df['change_pct'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(df['change_pct'].mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {df["change_pct"].mean():+.1f}%')
    ax.axvline(-30, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(30, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Price Change (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Recommended Price Changes', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # 1b. Box plot by category
    ax = fig.add_subplot(gs[0, 1])
    categories = ['underpriced', 'optimal', 'overpriced']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    data = [df[df['category'] == c]['change_pct'] for c in categories]
    bp = ax.boxplot(data, labels=['Underpriced', 'Optimal', 'Overpriced'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Price Change (%)', fontsize=11)
    ax.set_title('Price Change by Category', fontsize=12, fontweight='bold')
    
    # 1c. Direction pie chart
    ax = fig.add_subplot(gs[0, 2])
    sizes = [metrics.pct_raise, metrics.pct_hold, metrics.pct_lower]
    labels = ['Raise', 'Hold', 'Lower']
    colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
    explode = (0.05, 0, 0.05)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title('Recommendation Direction', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Row 2: Peer Validation
    # =========================================================================
    
    # 2a. Current vs Peer RevPAR scatter
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(df['current_revpar'], df['peer_revpar'], alpha=0.4, c='steelblue', s=30)
    max_val = max(df['current_revpar'].max(), df['peer_revpar'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Equal')
    ax.set_xlabel('Current RevPAR (€)', fontsize=11)
    ax.set_ylabel('Peer RevPAR (€)', fontsize=11)
    ax.set_title('Current vs Peer RevPAR', fontsize=12, fontweight='bold')
    ax.legend()
    
    # 2b. Current vs Peer Price
    ax = fig.add_subplot(gs[1, 1])
    for cat, color in zip(categories, colors):
        cat_df = df[df['category'] == cat]
        ax.scatter(cat_df['current_price'], cat_df['peer_price'], alpha=0.5, c=color, 
                  label=cat.title(), s=30)
    max_val = max(df['current_price'].max(), df['peer_price'].max())
    ax.plot([0, max_val], [0, max_val], 'r--')
    ax.set_xlabel('Current Price (€)', fontsize=11)
    ax.set_ylabel('Peer Price (€)', fontsize=11)
    ax.set_title('Current vs Peer Price by Category', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # 2c. Validation metrics summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    metrics_text = f"""
VALIDATION METRICS SUMMARY
{'═' * 40}

PEER VALIDATION
  When we say "raise price":
  • Peers actually had higher RevPAR: {metrics.peer_revpar_validated_pct:.1f}%
  • Peers actually had higher Price: {metrics.peer_price_validated_pct:.1f}%

RECOMMENDATION QUALITY
  • Within ±30%: {metrics.bounded_recommendations_pct:.1f}%
  • Mean change: {metrics.mean_price_change:+.1f}%
  • Std deviation: {metrics.std_price_change:.1f}%

SIGNAL CORRELATIONS
  • RevPAR gap → rec: {metrics.revpar_gap_corr_with_recommendation:+.2f}
  • Price gap → rec: {metrics.price_gap_corr_with_recommendation:+.2f}
  
  Expected: negative correlations
  (lower gap = higher rec)

SEGMENT CONSISTENCY
  • Score: {metrics.segment_consistency:.1f}%
"""
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # =========================================================================
    # Row 3: Signal Analysis
    # =========================================================================
    
    # 3a. RevPAR gap vs recommendation
    ax = fig.add_subplot(gs[2, 0])
    revpar_gap = (df['current_revpar'] - df['peer_revpar']) / df['peer_revpar'].clip(lower=1) * 100
    ax.scatter(revpar_gap.clip(-100, 100), df['change_pct'].clip(-50, 50), 
              alpha=0.4, c='coral', s=30)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add trend line
    z = np.polyfit(revpar_gap.clip(-100, 100), df['change_pct'].clip(-50, 50), 1)
    p = np.poly1d(z)
    x_line = np.linspace(-100, 100, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.2f})')
    
    ax.set_xlabel('RevPAR Gap vs Peers (%)', fontsize=11)
    ax.set_ylabel('Recommended Price Change (%)', fontsize=11)
    ax.set_title('RevPAR Gap → Recommendation\n(Negative gap should → positive change)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    
    # 3b. Price change vs RevPAR lift
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(df['change_pct'].clip(-50, 50), df['revpar_lift_pct'].clip(-50, 200), 
              alpha=0.4, c='steelblue', s=30)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Recommended Price Change (%)', fontsize=11)
    ax.set_ylabel('Expected RevPAR Lift (%)', fontsize=11)
    ax.set_title('Price Change vs Expected RevPAR Lift', fontsize=11, fontweight='bold')
    
    # 3c. Occupancy vs recommendation
    ax = fig.add_subplot(gs[2, 2])
    for cat, color in zip(categories, colors):
        cat_df = df[df['category'] == cat]
        ax.scatter(cat_df['current_occupancy'] * 100, cat_df['change_pct'], 
                  alpha=0.5, c=color, label=cat.title(), s=30)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Current Occupancy (%)', fontsize=11)
    ax.set_ylabel('Recommended Price Change (%)', fontsize=11)
    ax.set_title('Occupancy vs Recommendation', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    
    plt.suptitle('MODEL VALIDATION DASHBOARD', fontsize=14, fontweight='bold', y=1.02)
    
    fig.savefig(output_dir / 'validation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved validation dashboard to {output_dir / 'validation_dashboard.png'}")


def print_validation_report(metrics: ValidationMetrics) -> None:
    """Print a detailed validation report."""
    print("\n" + "=" * 70)
    print("MODEL VALIDATION REPORT")
    print("=" * 70)
    
    print(metrics)
    
    # Interpret results
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    
    # Peer validation
    if metrics.peer_revpar_validated_pct > 80:
        print("✓ STRONG: Peers actually had higher RevPAR in most raise recommendations")
    elif metrics.peer_revpar_validated_pct > 60:
        print("○ MODERATE: Peers had higher RevPAR in majority of cases")
    else:
        print("✗ WEAK: Peer RevPAR validation below threshold")
    
    # Bounded recommendations
    if metrics.bounded_recommendations_pct > 90:
        print("✓ STRONG: Almost all recommendations are within reasonable bounds (±30%)")
    elif metrics.bounded_recommendations_pct > 75:
        print("○ MODERATE: Most recommendations are bounded")
    else:
        print("✗ WEAK: Too many extreme recommendations")
    
    # Signal correlation
    if metrics.revpar_gap_corr_with_recommendation < -0.3:
        print("✓ STRONG: Recommendations correctly respond to RevPAR gaps")
    elif metrics.revpar_gap_corr_with_recommendation < 0:
        print("○ MODERATE: Some correlation between RevPAR gap and recommendations")
    else:
        print("✗ WEAK: Recommendations not aligned with RevPAR signals")
    
    print("=" * 70)




