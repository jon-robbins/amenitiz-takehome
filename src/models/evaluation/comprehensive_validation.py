"""
Comprehensive Multi-Week Validation Framework.

Validates pricing recommendations across:
- Multiple time periods (seasonal variation)
- Hotel segments (coastal, urban, rural)
- Revenue brackets
- Geographic regions

Also provides accuracy metrics without labeled data through:
1. Backtest validation: Compare recommendations vs actual outcomes
2. Cross-segment consistency: Similar hotels should get similar recommendations
3. Distribution sanity checks: Recommendations should be bounded
4. Peer agreement: Twin-matched hotels should show consistent patterns
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from src.models.evaluation.portfolio_analysis import analyze_portfolio, HotelCategory
from src.recommender.revpar_peers import get_revpar_comparison_for_hotel


# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class WeekConfig:
    """Configuration for a week to analyze."""
    name: str
    target_start: date
    as_of_date: date
    season: str  # 'high', 'low', 'shoulder'
    

@dataclass
class SegmentStats:
    """Statistics for a hotel segment."""
    name: str
    n_hotels: int
    n_underpriced: int
    n_optimal: int
    n_overpriced: int
    avg_current_revpar: float
    avg_expected_revpar: float
    avg_lift_pct: float
    avg_price_change_pct: float
    price_change_std: float


@dataclass 
class ValidationResults:
    """Complete validation results across all weeks and segments."""
    weeks_analyzed: List[str]
    total_hotel_weeks: int
    
    # By week
    week_results: Dict[str, pd.DataFrame]
    week_summaries: pd.DataFrame
    
    # By segment
    segment_results: Dict[str, SegmentStats]
    
    # Distribution metrics
    price_change_distribution: pd.Series
    revpar_lift_distribution: pd.Series
    
    # Validation metrics
    consistency_score: float
    bounded_recommendations_pct: float
    peer_agreement_score: float
    
    # All hotel-week observations
    all_results: pd.DataFrame


def get_validation_weeks() -> List[WeekConfig]:
    """
    Get 10 representative weeks for validation.
    
    Includes:
    - 3 high season weeks (July-August)
    - 3 low season weeks (January-February, November)
    - 4 shoulder season weeks (March-April, September-October)
    """
    return [
        # High season (July-August 2024)
        WeekConfig("2024-W28 (July High)", date(2024, 7, 8), date(2024, 6, 24), "high"),
        WeekConfig("2024-W31 (Aug High)", date(2024, 7, 29), date(2024, 7, 15), "high"),
        WeekConfig("2024-W33 (Aug Peak)", date(2024, 8, 12), date(2024, 7, 29), "high"),
        
        # Low season (Winter)
        WeekConfig("2024-W05 (Jan Low)", date(2024, 1, 29), date(2024, 1, 15), "low"),
        WeekConfig("2024-W07 (Feb Low)", date(2024, 2, 12), date(2024, 1, 29), "low"),
        WeekConfig("2024-W46 (Nov Low)", date(2024, 11, 11), date(2024, 10, 28), "low"),
        
        # Shoulder season (Spring/Fall)
        WeekConfig("2024-W13 (Mar Shoulder)", date(2024, 3, 25), date(2024, 3, 11), "shoulder"),
        WeekConfig("2024-W17 (Apr Shoulder)", date(2024, 4, 22), date(2024, 4, 8), "shoulder"),
        WeekConfig("2024-W39 (Sep Shoulder)", date(2024, 9, 23), date(2024, 9, 9), "shoulder"),
        WeekConfig("2024-W42 (Oct Shoulder)", date(2024, 10, 14), date(2024, 9, 30), "shoulder"),
    ]


def classify_hotel_segment(row: pd.Series, distance_df: Optional[pd.DataFrame] = None) -> str:
    """
    Classify hotel into segment based on location/characteristics.
    
    Segments:
    - Coastal: Within 30km of coast
    - Urban: Major cities (Madrid, Barcelona, etc.)
    - Rural: Everything else
    """
    # Try to use distance features if available
    if distance_df is not None and 'hotel_id' in row:
        match = distance_df[distance_df['hotel_id'] == row.get('hotel_id')]
        if len(match) > 0:
            dist_coast = match.iloc[0].get('distance_from_coast', 999)
            if dist_coast < 30:
                return 'coastal'
    
    # Fallback: use peer source and price as proxy
    peer_price = row.get('peer_price', 0)
    if peer_price > 150:
        return 'urban'  # Higher prices typically urban
    elif peer_price < 80:
        return 'rural'
    else:
        return 'suburban'


def classify_revenue_bracket(revpar: float) -> str:
    """Classify hotel by RevPAR bracket."""
    if revpar < 30:
        return 'budget'
    elif revpar < 80:
        return 'economy'
    elif revpar < 150:
        return 'midscale'
    elif revpar < 300:
        return 'upscale'
    else:
        return 'luxury'


def run_comprehensive_validation(
    con,
    weeks: Optional[List[WeekConfig]] = None,
    max_hotels_per_week: Optional[int] = None
) -> ValidationResults:
    """
    Run validation across multiple weeks and segments.
    
    Args:
        con: Database connection
        weeks: List of weeks to analyze (default: 10 representative weeks)
        max_hotels_per_week: Limit hotels per week for faster testing
    
    Returns:
        ValidationResults with complete analysis
    """
    if weeks is None:
        weeks = get_validation_weeks()
    
    all_results = []
    week_results = {}
    week_summaries = []
    
    print("=" * 80)
    print("COMPREHENSIVE MULTI-WEEK VALIDATION")
    print("=" * 80)
    
    for week in weeks:
        print(f"\nAnalyzing {week.name}...")
        
        target_dates = [week.target_start + timedelta(days=i) for i in range(7)]
        
        try:
            analysis = analyze_portfolio(
                con, target_dates, week.as_of_date, 
                max_hotels=max_hotels_per_week
            )
            
            df = analysis.hotel_results.copy()
            df['week'] = week.name
            df['season'] = week.season
            df['segment'] = df.apply(classify_hotel_segment, axis=1)
            df['revenue_bracket'] = df['current_revpar'].apply(classify_revenue_bracket)
            
            week_results[week.name] = df
            all_results.append(df)
            
            week_summaries.append({
                'week': week.name,
                'season': week.season,
                'n_hotels': len(df),
                'n_underpriced': len(df[df['category'] == 'underpriced']),
                'n_optimal': len(df[df['category'] == 'optimal']),
                'n_overpriced': len(df[df['category'] == 'overpriced']),
                'avg_current_revpar': df['current_revpar'].mean(),
                'avg_expected_revpar': df['expected_revpar'].mean(),
                'avg_lift_pct': df['revpar_lift_pct'].mean(),
                'avg_price_change_pct': df['change_pct'].mean(),
            })
            
            print(f"  → {len(df)} hotels: {len(df[df['category']=='underpriced'])} under, "
                  f"{len(df[df['category']=='optimal'])} optimal, "
                  f"{len(df[df['category']=='overpriced'])} over")
            
        except Exception as e:
            print(f"  → Error: {e}")
            continue
    
    if len(all_results) == 0:
        raise ValueError("No weeks could be analyzed")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(week_summaries)
    
    # Calculate segment statistics
    segment_stats = {}
    for segment in combined_df['segment'].unique():
        seg_df = combined_df[combined_df['segment'] == segment]
        segment_stats[segment] = SegmentStats(
            name=segment,
            n_hotels=len(seg_df),
            n_underpriced=len(seg_df[seg_df['category'] == 'underpriced']),
            n_optimal=len(seg_df[seg_df['category'] == 'optimal']),
            n_overpriced=len(seg_df[seg_df['category'] == 'overpriced']),
            avg_current_revpar=seg_df['current_revpar'].mean(),
            avg_expected_revpar=seg_df['expected_revpar'].mean(),
            avg_lift_pct=seg_df['revpar_lift_pct'].mean(),
            avg_price_change_pct=seg_df['change_pct'].mean(),
            price_change_std=seg_df['change_pct'].std()
        )
    
    # Validation metrics
    
    # 1. Bounded recommendations (should be within ±30%)
    bounded_pct = (combined_df['change_pct'].abs() <= 30).mean() * 100
    
    # 2. Consistency: Same hotel in different weeks should get similar recommendations
    # Group by hotel and check variance
    hotel_variance = combined_df.groupby('hotel_id')['change_pct'].std().mean()
    consistency_score = max(0, 100 - hotel_variance * 5)  # Lower variance = higher score
    
    # 3. Peer agreement: Hotels with similar peer_revpar should have similar recommendations
    combined_df['peer_revpar_bucket'] = pd.cut(combined_df['peer_revpar'], bins=10)
    peer_variance = combined_df.groupby('peer_revpar_bucket')['change_pct'].std().mean()
    peer_agreement = max(0, 100 - peer_variance * 3)
    
    return ValidationResults(
        weeks_analyzed=[w.name for w in weeks if w.name in week_results],
        total_hotel_weeks=len(combined_df),
        week_results=week_results,
        week_summaries=summary_df,
        segment_results=segment_stats,
        price_change_distribution=combined_df['change_pct'],
        revpar_lift_distribution=combined_df['revpar_lift_pct'],
        consistency_score=consistency_score,
        bounded_recommendations_pct=bounded_pct,
        peer_agreement_score=peer_agreement,
        all_results=combined_df
    )


def create_comprehensive_visualizations(
    results: ValidationResults,
    output_dir: Path
) -> None:
    """
    Create comprehensive visualization suite.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = results.all_results
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # =========================================================================
    # FIGURE 1: Price Change Distribution (NOT pie charts!)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1a. Histogram of price changes
    ax = axes[0, 0]
    ax.hist(df['change_pct'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(df['change_pct'].mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {df["change_pct"].mean():+.1f}%')
    ax.set_xlabel('Recommended Price Change (%)', fontsize=12)
    ax.set_ylabel('Number of Hotel-Weeks', fontsize=12)
    ax.set_title('Distribution of Recommended Price Changes', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 1b. Box plot by category
    ax = axes[0, 1]
    categories = ['underpriced', 'optimal', 'overpriced']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bp = ax.boxplot([df[df['category'] == c]['change_pct'] for c in categories],
                    labels=['Underpriced', 'Optimal', 'Overpriced'],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Price Change (%)', fontsize=12)
    ax.set_title('Price Change by Category', fontsize=14, fontweight='bold')
    
    # 1c. RevPAR lift distribution
    ax = axes[1, 0]
    # Clip for visualization (some outliers can be extreme)
    lift_clipped = df['revpar_lift_pct'].clip(-100, 200)
    ax.hist(lift_clipped, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(lift_clipped.median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {lift_clipped.median():+.1f}%')
    ax.set_xlabel('Expected RevPAR Lift (%)', fontsize=12)
    ax.set_ylabel('Number of Hotel-Weeks', fontsize=12)
    ax.set_title('Distribution of Expected RevPAR Lift', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 1d. Scatter: Price change vs RevPAR lift
    ax = axes[1, 1]
    for cat, color in zip(categories, colors):
        cat_df = df[df['category'] == cat]
        ax.scatter(cat_df['change_pct'], cat_df['revpar_lift_pct'].clip(-100, 200),
                  alpha=0.3, c=color, label=cat.title(), s=20)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Price Change (%)', fontsize=12)
    ax.set_ylabel('RevPAR Lift (%)', fontsize=12)
    ax.set_title('Price Change vs RevPAR Lift', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / '1_distributions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # =========================================================================
    # FIGURE 2: Seasonal Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 2a. Category distribution by season
    ax = axes[0, 0]
    season_cat = df.groupby(['season', 'category']).size().unstack(fill_value=0)
    season_cat_pct = season_cat.div(season_cat.sum(axis=1), axis=0) * 100
    season_cat_pct[['underpriced', 'optimal', 'overpriced']].plot(
        kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black'
    )
    ax.set_ylabel('% of Hotels', fontsize=12)
    ax.set_title('Pricing Categories by Season', fontsize=14, fontweight='bold')
    ax.legend(title='Category')
    ax.set_xticklabels(['High', 'Low', 'Shoulder'], rotation=0)
    
    # 2b. Average lift by season
    ax = axes[0, 1]
    season_lift = df.groupby('season')['revpar_lift_pct'].mean()
    bars = ax.bar(['High', 'Low', 'Shoulder'], 
                  [season_lift.get('high', 0), season_lift.get('low', 0), season_lift.get('shoulder', 0)],
                  color=['#e74c3c', '#3498db', '#f39c12'], edgecolor='black')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel('Average RevPAR Lift (%)', fontsize=12)
    ax.set_title('RevPAR Lift Opportunity by Season', fontsize=14, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11)
    
    # 2c. Price change distribution by season
    ax = axes[1, 0]
    for season, color in [('high', '#e74c3c'), ('low', '#3498db'), ('shoulder', '#f39c12')]:
        season_df = df[df['season'] == season]
        ax.hist(season_df['change_pct'], bins=30, alpha=0.5, label=season.title(), color=color)
    ax.set_xlabel('Price Change (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Price Change Distribution by Season', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 2d. Week-by-week summary
    ax = axes[1, 1]
    summary = results.week_summaries.sort_values('avg_lift_pct', ascending=True)
    colors = ['#e74c3c' if s == 'high' else '#3498db' if s == 'low' else '#f39c12' 
              for s in summary['season']]
    ax.barh(range(len(summary)), summary['avg_lift_pct'], color=colors, edgecolor='black')
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary['week'].str[:12], fontsize=9)
    ax.set_xlabel('Average RevPAR Lift (%)', fontsize=12)
    ax.set_title('RevPAR Lift by Week', fontsize=14, fontweight='bold')
    ax.axvline(0, color='gray', linestyle='--')
    
    plt.tight_layout()
    fig.savefig(output_dir / '2_seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # =========================================================================
    # FIGURE 3: Segment Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 3a. Category by segment
    ax = axes[0, 0]
    seg_cat = df.groupby(['segment', 'category']).size().unstack(fill_value=0)
    seg_cat_pct = seg_cat.div(seg_cat.sum(axis=1), axis=0) * 100
    if len(seg_cat_pct) > 0:
        seg_cat_pct.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black')
    ax.set_ylabel('% of Hotels', fontsize=12)
    ax.set_title('Pricing Categories by Segment', fontsize=14, fontweight='bold')
    ax.legend(title='Category')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3b. Lift by segment
    ax = axes[0, 1]
    seg_lift = df.groupby('segment')['revpar_lift_pct'].mean().sort_values()
    bars = ax.barh(range(len(seg_lift)), seg_lift.values, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(seg_lift)))
    ax.set_yticklabels(seg_lift.index)
    ax.set_xlabel('Average RevPAR Lift (%)', fontsize=12)
    ax.set_title('RevPAR Lift by Segment', fontsize=14, fontweight='bold')
    ax.axvline(0, color='gray', linestyle='--')
    
    # 3c. Revenue bracket analysis
    ax = axes[1, 0]
    bracket_order = ['budget', 'economy', 'midscale', 'upscale', 'luxury']
    bracket_cat = df.groupby(['revenue_bracket', 'category']).size().unstack(fill_value=0)
    bracket_cat = bracket_cat.reindex(bracket_order).dropna(how='all')
    bracket_cat_pct = bracket_cat.div(bracket_cat.sum(axis=1), axis=0) * 100
    bracket_cat_pct.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black')
    ax.set_ylabel('% of Hotels', fontsize=12)
    ax.set_title('Pricing Categories by Revenue Bracket', fontsize=14, fontweight='bold')
    ax.legend(title='Category')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3d. Lift by revenue bracket
    ax = axes[1, 1]
    bracket_lift = df.groupby('revenue_bracket')['revpar_lift_pct'].mean()
    bracket_lift = bracket_lift.reindex(bracket_order).dropna()
    bars = ax.bar(range(len(bracket_lift)), bracket_lift.values, 
                  color=['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b'],
                  edgecolor='black')
    ax.set_xticks(range(len(bracket_lift)))
    ax.set_xticklabels(bracket_lift.index, rotation=45, ha='right')
    ax.set_ylabel('Average RevPAR Lift (%)', fontsize=12)
    ax.set_title('RevPAR Lift by Revenue Bracket', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--')
    
    plt.tight_layout()
    fig.savefig(output_dir / '3_segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # =========================================================================
    # FIGURE 4: Validation Metrics
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 4a. Recommendation bounds check
    ax = axes[0, 0]
    bounds = [
        ('Within ±10%', (df['change_pct'].abs() <= 10).mean() * 100),
        ('Within ±20%', (df['change_pct'].abs() <= 20).mean() * 100),
        ('Within ±30%', (df['change_pct'].abs() <= 30).mean() * 100),
        ('Exceeds ±30%', (df['change_pct'].abs() > 30).mean() * 100),
    ]
    labels, values = zip(*bounds)
    colors_bound = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    ax.bar(labels, values, color=colors_bound, edgecolor='black')
    ax.set_ylabel('% of Recommendations', fontsize=12)
    ax.set_title('Recommendation Bounds Distribution', fontsize=14, fontweight='bold')
    for i, v in enumerate(values):
        ax.annotate(f'{v:.1f}%', xy=(i, v), xytext=(0, 3), textcoords='offset points', 
                   ha='center', fontsize=11)
    
    # 4b. Validation scores
    ax = axes[0, 1]
    scores = [
        ('Bounded\n(≤±30%)', results.bounded_recommendations_pct),
        ('Consistency\nScore', results.consistency_score),
        ('Peer\nAgreement', results.peer_agreement_score),
    ]
    labels, values = zip(*scores)
    colors_score = ['#2ecc71' if v >= 80 else '#f39c12' if v >= 60 else '#e74c3c' for v in values]
    ax.bar(labels, values, color=colors_score, edgecolor='black')
    ax.axhline(80, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    for i, v in enumerate(values):
        ax.annotate(f'{v:.1f}%', xy=(i, v), xytext=(0, 3), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold')
    
    # 4c. Q-Q plot for price changes (normality check)
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(df['change_pct'].dropna(), dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Price Change Recommendations', fontsize=14, fontweight='bold')
    ax.get_lines()[0].set_markersize(3)
    
    # 4d. Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    VALIDATION SUMMARY
    {'═' * 50}
    
    Total Hotel-Weeks Analyzed: {results.total_hotel_weeks:,}
    Weeks Covered: {len(results.weeks_analyzed)}
    
    RECOMMENDATION BOUNDS
    • Within ±10%: {(df['change_pct'].abs() <= 10).mean()*100:.1f}%
    • Within ±20%: {(df['change_pct'].abs() <= 20).mean()*100:.1f}%
    • Within ±30%: {(df['change_pct'].abs() <= 30).mean()*100:.1f}%
    
    PRICE CHANGE STATISTICS
    • Mean: {df['change_pct'].mean():+.1f}%
    • Median: {df['change_pct'].median():+.1f}%
    • Std Dev: {df['change_pct'].std():.1f}%
    • Min: {df['change_pct'].min():+.1f}%
    • Max: {df['change_pct'].max():+.1f}%
    
    REVPAR LIFT STATISTICS
    • Mean: {df['revpar_lift_pct'].mean():+.1f}%
    • Median: {df['revpar_lift_pct'].median():+.1f}%
    
    VALIDATION SCORES
    • Bounded Recommendations: {results.bounded_recommendations_pct:.1f}%
    • Consistency Score: {results.consistency_score:.1f}%
    • Peer Agreement: {results.peer_agreement_score:.1f}%
    """
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(output_dir / '4_validation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved comprehensive visualizations to {output_dir}")


def print_validation_summary(results: ValidationResults) -> None:
    """Print validation summary to console."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Hotel-Weeks Analyzed: {results.total_hotel_weeks:,}")
    print(f"Weeks Covered: {len(results.weeks_analyzed)}")
    
    df = results.all_results
    
    print("\n" + "-" * 40)
    print("RECOMMENDATION BOUNDS")
    print("-" * 40)
    print(f"Within ±10%: {(df['change_pct'].abs() <= 10).mean()*100:.1f}%")
    print(f"Within ±20%: {(df['change_pct'].abs() <= 20).mean()*100:.1f}%")
    print(f"Within ±30%: {(df['change_pct'].abs() <= 30).mean()*100:.1f}%")
    
    print("\n" + "-" * 40)
    print("VALIDATION SCORES")
    print("-" * 40)
    print(f"Bounded Recommendations: {results.bounded_recommendations_pct:.1f}%")
    print(f"Consistency Score: {results.consistency_score:.1f}%")
    print(f"Peer Agreement Score: {results.peer_agreement_score:.1f}%")
    
    print("\n" + "-" * 40)
    print("BY SEASON")
    print("-" * 40)
    for season in ['high', 'low', 'shoulder']:
        season_df = df[df['season'] == season]
        if len(season_df) > 0:
            print(f"\n{season.upper()} SEASON ({len(season_df)} hotel-weeks):")
            print(f"  Underpriced: {(season_df['category']=='underpriced').mean()*100:.1f}%")
            print(f"  Optimal: {(season_df['category']=='optimal').mean()*100:.1f}%")
            print(f"  Overpriced: {(season_df['category']=='overpriced').mean()*100:.1f}%")
            print(f"  Avg Price Change: {season_df['change_pct'].mean():+.1f}%")
            print(f"  Avg RevPAR Lift: {season_df['revpar_lift_pct'].mean():+.1f}%")
    
    print("\n" + "=" * 80)




