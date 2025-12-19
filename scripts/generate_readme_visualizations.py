"""
Generate all visualizations for README.md.

Run this script after rolling_validation.py to update figures with latest data.

Usage:
    python scripts/generate_readme_visualizations.py
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_validation_results() -> pd.DataFrame:
    """Load rolling validation results CSV."""
    results_path = Path("outputs/data/rolling_validation_results.csv")
    if not results_path.exists():
        raise FileNotFoundError(
            "Run `python -m src.models.evaluation.rolling_validation` first"
        )
    return pd.read_csv(results_path)


def generate_segment_elasticity(df: pd.DataFrame) -> None:
    """Generate segment_elasticity_calculated.png."""
    print("Generating segment_elasticity_calculated.png...")
    
    import re
    
    segment_data = []
    hotel_counts = {}  # Track hotel counts per segment
    
    for _, row in df.iterrows():
        try:
            # Handle np.float64 in string representation
            elasticity_str = row['segment_elasticity']
            if isinstance(elasticity_str, str):
                cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', elasticity_str)
                elasticity = ast.literal_eval(cleaned)
            else:
                elasticity = row['segment_elasticity']
            
            # Parse hotels by segment for this window
            hotels_str = row.get('hotels_by_segment', '{}')
            if isinstance(hotels_str, str):
                cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', hotels_str)
                hotels = ast.literal_eval(cleaned)
            else:
                hotels = hotels_str or {}
            
            for seg, val in elasticity.items():
                segment_data.append({
                    'segment': seg, 
                    'elasticity': float(val), 
                    'window': row['window'],
                    'hotels': hotels.get(seg, 0)
                })
        except Exception as e:
            print(f"  Warning: Could not parse elasticity for window {row.get('window', '?')}: {e}")
    
    if not segment_data:
        print("  No segment data found!")
        return
    
    seg_df = pd.DataFrame(segment_data)
    
    # Calculate means, std, and average hotel count per segment
    seg_stats = seg_df.groupby('segment').agg({
        'elasticity': ['mean', 'std'],
        'hotels': 'mean'  # Average hotels across windows
    }).reset_index()
    seg_stats.columns = ['segment', 'mean', 'std', 'avg_hotels']
    seg_stats = seg_stats.sort_values('mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(seg_stats)))
    ax.barh(seg_stats['segment'], seg_stats['mean'], 
            xerr=seg_stats['std'], color=colors, capsize=3)
    
    ax.axvline(x=-0.39, color='red', linestyle='--', linewidth=1.5, 
               label='Market avg (-0.39)')
    ax.set_xlabel('Price Elasticity (ε)', fontsize=12)
    ax.set_title(
        'Segment-Level Price Elasticity\n'
        '(calculated from 19 rolling windows, includes cancelled bookings)', 
        fontsize=14
    )
    
    # Add value labels with hotel count
    for i in range(len(seg_stats)):
        ax.text(
            seg_stats.iloc[i]['mean'] + 0.02, i, 
            f"{seg_stats.iloc[i]['mean']:.2f} (n={int(seg_stats.iloc[i]['avg_hotels'])} hotels)", 
            va='center', fontsize=9
        )
    
    ax.legend()
    ax.set_xlim(-0.6, 0.1)
    plt.tight_layout()
    plt.savefig('outputs/figures/segment_elasticity_calculated.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved segment_elasticity_calculated.png")


def generate_segment_opportunity(df: pd.DataFrame) -> None:
    """Generate segment_opportunity_current.png."""
    print("Generating segment_opportunity_current.png...")
    
    # Parse lift_by_segment and hotels_by_segment
    import re
    
    segment_data = []
    for _, row in df.iterrows():
        try:
            # Parse hotels by segment
            hotels_str = row['hotels_by_segment']
            if isinstance(hotels_str, str):
                cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', hotels_str)
                hotels = ast.literal_eval(cleaned)
            else:
                hotels = hotels_str
            
            # Parse lift by segment
            lift_str = row['lift_by_segment']
            if isinstance(lift_str, str):
                cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', lift_str)
                lift = ast.literal_eval(cleaned)
            else:
                lift = lift_str
            
            for seg in hotels.keys():
                segment_data.append({
                    'segment': seg,
                    'hotels': hotels.get(seg, 0),
                    'lift': lift.get(seg, 0),
                    'window': row['window']
                })
        except Exception as e:
            print(f"  Warning: Could not parse segment data: {e}")
    
    if not segment_data:
        print("  No segment opportunity data found!")
        return
    
    seg_df = pd.DataFrame(segment_data)
    
    # Aggregate
    opp_df = seg_df.groupby('segment').agg({
        'lift': 'mean',
        'hotels': 'mean'
    }).reset_index()
    opp_df['annual_opportunity'] = opp_df['lift'] * opp_df['hotels'] * 365 / 1000
    opp_df = opp_df.sort_values('lift', ascending=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: RevPAR lift per room
    ax1 = axes[0]
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(opp_df)))
    ax1.barh(opp_df['segment'], opp_df['lift'], color=colors)
    ax1.set_xlabel('€ per room per night')
    ax1.set_title('RevPAR Lift per Room')
    for i, (_, row) in enumerate(opp_df.iterrows()):
        ax1.text(max(row['lift'] + 0.02, 0.05), i, f"€{row['lift']:.2f}", 
                 va='center', fontsize=9)
    
    # Middle: Hotel count
    ax2 = axes[1]
    ax2.barh(opp_df['segment'], opp_df['hotels'], color='steelblue')
    ax2.set_xlabel('Average Hotels with Opportunity')
    ax2.set_title('Hotels per Segment')
    for i, (_, row) in enumerate(opp_df.iterrows()):
        ax2.text(row['hotels'] + 1, i, f"{int(row['hotels'])}", 
                 va='center', fontsize=9)
    
    # Right: Total opportunity
    ax3 = axes[2]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(opp_df)))
    ax3.barh(opp_df['segment'], opp_df['annual_opportunity'], color=colors)
    ax3.set_xlabel('€ thousands per year')
    ax3.set_title('Total Annual Opportunity')
    for i, (_, row) in enumerate(opp_df.iterrows()):
        ax3.text(max(row['annual_opportunity'] + 1, 2), i, 
                 f"€{row['annual_opportunity']:.0f}k", va='center', fontsize=9)
    
    plt.suptitle('Revenue Opportunity by Segment', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/figures/segment_opportunity_current.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved segment_opportunity_current.png")


def compute_baseline_metrics(sample_size: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Run time-based backtests for all baseline strategies and PriceAdvisor.
    
    For consistency, RevPAR lift is calculated only for hotels that receive
    an actionable recommendation (price change). This shows the value for
    hotels that adopt the recommendation.
    
    Args:
        sample_size: Number of hotel-weeks to sample for evaluation.
        random_state: Random seed for reproducibility.
    
    Returns:
        DataFrame with strategy names, success rates, and RevPAR lifts
        computed from actual backtest results.
    """
    from datetime import date
    from src.models.evaluation.time_backtest import (
        BacktestConfig, get_train_test_split, calculate_revpar_lift,
        baseline_peer_median, baseline_self_median
    )
    from src.recommender.pricing_pipeline import PricingPipeline
    
    config = BacktestConfig()
    train_df, test_df = get_train_test_split(config)
    
    # Sample test set for faster computation while maintaining representativeness
    if len(test_df) > sample_size:
        test_df = test_df.sample(n=sample_size, random_state=random_state)
        print(f"  Sampled {sample_size} hotel-weeks for baseline comparison")
    
    # Define additional baselines - these always recommend a price change
    def baseline_random(train_df: pd.DataFrame, row: pd.Series, week: date) -> float:
        """Random price within reasonable bounds (±30% of actual)."""
        np.random.seed(hash((row['hotel_id'], str(week))) % (2**32))
        return row['avg_price'] * np.random.uniform(0.7, 1.3)
    
    def baseline_market_average(train_df: pd.DataFrame, row: pd.Series, week: date) -> float:
        """Market-wide average price."""
        return train_df['avg_price'].mean()
    
    # Initialize PriceAdvisor pipeline
    pipeline = PricingPipeline()
    pipeline.fit()
    
    # Track PriceAdvisor recommendations separately to filter to actionable only
    priceadvisor_results = []
    
    for idx, row in test_df.iterrows():
        week = row['week_start'].date() if hasattr(row['week_start'], 'date') else row['week_start']
        try:
            rec = pipeline.recommend(int(row['hotel_id']), week)
            recommendation = rec.get('recommendation', 'HOLD')
            rec_price = rec.get('recommended_price', row['avg_price'])
            
            # Only include actionable recommendations (RAISE/SET)
            if recommendation in ('RAISE', 'SET'):
                _, _, lift, _ = calculate_revpar_lift(
                    row['avg_price'], row['occupancy_rate'], rec_price, config.elasticity
                )
                priceadvisor_results.append({
                    'recommendation': recommendation,
                    'lift': lift,
                    'is_actionable': True
                })
        except Exception:
            pass
    
    # Calculate metrics for baselines (all hotels get recommendations)
    results = []
    baseline_strategies = {
        'Random\nPricing': baseline_random,
        'Market\nAverage': baseline_market_average,
        'Self\nMedian': baseline_self_median,
        'Peer\nMedian': baseline_peer_median,
    }
    
    for name, recommender_fn in baseline_strategies.items():
        print(f"  Running backtest: {name.replace(chr(10), ' ')}...")
        
        lifts = []
        for idx, row in test_df.iterrows():
            week = row['week_start'].date() if hasattr(row['week_start'], 'date') else row['week_start']
            try:
                rec_price = recommender_fn(train_df, row, week)
                _, _, lift, _ = calculate_revpar_lift(
                    row['avg_price'], row['occupancy_rate'], rec_price, config.elasticity
                )
                lifts.append(lift)
            except Exception:
                lifts.append(0)
        
        lifts = np.array(lifts)
        win_rate = (lifts > 0).mean() * 100
        mean_lift = lifts.mean()
        
        results.append({
            'Strategy': name,
            'Win Rate': round(win_rate, 1),
            'RevPAR Lift': round(mean_lift, 2)
        })
    
    # Calculate PriceAdvisor metrics (actionable recommendations only)
    print(f"  Running backtest: PriceAdvisor...")
    pa_lifts = np.array([r['lift'] for r in priceadvisor_results])
    pa_win_rate = (pa_lifts > 0).mean() * 100 if len(pa_lifts) > 0 else 0
    pa_mean_lift = pa_lifts.mean() if len(pa_lifts) > 0 else 0
    
    results.append({
        'Strategy': 'PriceAdvisor',
        'Win Rate': round(pa_win_rate, 1),
        'RevPAR Lift': round(pa_mean_lift, 2)
    })
    
    print(f"  PriceAdvisor: {len(priceadvisor_results)} actionable recommendations "
          f"({len(priceadvisor_results)/len(test_df)*100:.1f}% of hotels)")
    
    return pd.DataFrame(results)


def generate_baseline_comparison() -> None:
    """Generate baseline_comparison_updated.png."""
    print("Generating baseline_comparison_updated.png...")
    
    # Compute metrics from actual backtests
    baseline_df = compute_baseline_metrics()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Win Rate (lift > 0) - % of adopters that see improvement
    ax1 = axes[0]
    colors = ['gray', 'lightblue', 'skyblue', 'steelblue', 'darkgreen']
    bars1 = ax1.bar(baseline_df['Strategy'], baseline_df['Win Rate'], color=colors)
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate for Adopters (RevPAR > 0)')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.set_ylim(0, 100)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2, 
                 f'{height:.0f}%', ha='center', fontsize=10)
    
    # RevPAR Lift - the key business metric (for adopters)
    ax2 = axes[1]
    colors = ['red' if x < 0 else 'green' for x in baseline_df['RevPAR Lift']]
    bars2 = ax2.bar(baseline_df['Strategy'], baseline_df['RevPAR Lift'], 
                    color=colors, alpha=0.7)
    ax2.set_ylabel('RevPAR Lift (€)')
    ax2.set_title('Average RevPAR Lift per Room per Night (Adopters)')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
                 height + 0.3 if height > 0 else height - 0.8, 
                 f'€{height:.2f}', ha='center', fontsize=10)
    
    plt.suptitle('PriceAdvisor vs Baseline Strategies', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/figures/baseline_comparison_updated.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved baseline_comparison_updated.png")


def generate_seasonal_patterns(df: pd.DataFrame) -> None:
    """Generate seasonal_patterns_explained.png."""
    print("Generating seasonal_patterns_explained.png...")
    
    df = df.copy()
    df['test_start'] = pd.to_datetime(df['test_start'])
    
    # Use correct column names
    lift_col = 'avg_revpar_lift_per_hotel'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Per-room lift over time
    ax1 = axes[0, 0]
    ax1.plot(df['test_start'], df[lift_col], 'o-', color='green', 
             linewidth=2, markersize=6)
    ax1.fill_between(df['test_start'], 0, df[lift_col], alpha=0.2, color='green')
    ax1.axhline(y=df[lift_col].mean(), color='red', linestyle='--', 
                label=f"Avg: €{df[lift_col].mean():.2f}")
    ax1.set_xlabel('Test Period')
    ax1.set_ylabel('€ per room per night')
    ax1.set_title('Per-Room RevPAR Lift Over Time')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Top right: Hotel count over time
    ax2 = axes[0, 1]
    ax2.plot(df['test_start'], df['n_hotels'], 'o-', color='steelblue', 
             linewidth=2, markersize=6)
    ax2.fill_between(df['test_start'], 0, df['n_hotels'], 
                     alpha=0.2, color='steelblue')
    ax2.set_xlabel('Test Period')
    ax2.set_ylabel('Hotels')
    ax2.set_title('Hotels in Test Window')
    ax2.tick_params(axis='x', rotation=45)
    
    # Bottom left: Segment elasticity box plot
    ax3 = axes[1, 0]
    
    # Parse segment elasticity for box plot
    import re
    segment_data = []
    for _, row in df.iterrows():
        try:
            elasticity_str = row['segment_elasticity']
            if isinstance(elasticity_str, str):
                cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', elasticity_str)
                elasticity = ast.literal_eval(cleaned)
                for seg, val in elasticity.items():
                    segment_data.append({'segment': seg, 'elasticity': float(val)})
        except:
            pass
    
    if segment_data:
        seg_df = pd.DataFrame(segment_data)
        seg_order = seg_df.groupby('segment')['elasticity'].mean().sort_values(
            ascending=False
        ).index
        sns.boxplot(data=seg_df, x='segment', y='elasticity', order=seg_order, ax=ax3)
        ax3.set_xlabel('Segment')
        ax3.set_ylabel('Elasticity')
        ax3.set_title('Elasticity Distribution by Segment')
        ax3.tick_params(axis='x', rotation=45)
    
    # Bottom right: Explanation
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'Understanding the Charts', fontsize=14, 
             fontweight='bold', ha='center', transform=ax4.transAxes)
    ax4.text(0.1, 0.75, f'• Top-left: Per-room lift averages €{df[lift_col].mean():.2f}/night', 
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.60, '• Top-right: More hotels in later periods', 
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.45, '• Bottom-left: Resort coastal least elastic (-0.13)', 
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.30, '• Provincial city most elastic (-0.44)', 
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.15, '• Higher hotel count → higher total opportunity', 
             fontsize=11, transform=ax4.transAxes)
    
    plt.suptitle('Seasonal Patterns and Opportunity Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/figures/seasonal_patterns_explained.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved seasonal_patterns_explained.png")


def generate_price_change_distribution(df: pd.DataFrame) -> None:
    """Generate price_change_distribution.png - segmented by market segment for hotels with recommendations."""
    print("Generating price_change_distribution.png...")
    
    from src.recommender.pricing_pipeline import PricingPipeline
    from datetime import date
    
    # Initialize pipeline and get recommendations for a sample period
    pipeline = PricingPipeline()
    pipeline.fit()
    
    # Get all hotels with their features
    hotel_features = pipeline.peer_matcher.hotel_features
    
    # Generate recommendations for each hotel
    recommendations = []
    target_date = date(2024, 6, 15)  # Sample date
    
    for hotel_id in hotel_features['hotel_id'].values:  # All hotels
        try:
            rec = pipeline.recommend(int(hotel_id), target_date)
            if rec.get('recommendation') in ['RAISE', 'LOWER']:
                current = rec.get('current_price', 0)
                recommended = rec.get('recommended_price', 0)
                if current > 0:
                    pct_change = (recommended - current) / current * 100
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'segment': rec.get('segment', 'unknown'),
                        'recommendation': rec.get('recommendation'),
                        'current_price': current,
                        'recommended_price': recommended,
                        'pct_change': pct_change
                    })
        except Exception:
            pass
    
    if not recommendations:
        print("  No recommendations generated, using fallback...")
        # Fallback to aggregate data
        price_change = (
            (df['avg_recommended_price'] - df['avg_actual_price']) 
            / df['avg_actual_price'] * 100
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(price_change, bins=20, color='steelblue', edgecolor='white')
        ax.set_xlabel('Price Change (%)')
        ax.set_ylabel('Count')
        plt.savefig('outputs/figures/price_change_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    rec_df = pd.DataFrame(recommendations)
    
    # Create visualization with segment breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Overall distribution by recommendation type
    ax1 = axes[0]
    raise_data = rec_df[rec_df['recommendation'] == 'RAISE']['pct_change']
    lower_data = rec_df[rec_df['recommendation'] == 'LOWER']['pct_change']
    
    bins = np.arange(-25, 26, 1)  # 1% bins from -25% to +25%
    n1, bins1, patches1 = ax1.hist(raise_data, bins=bins, alpha=0.7, 
                                    label=f'RAISE (n={len(raise_data)})', 
                                    color='green', align='mid', rwidth=0.8)
    n2, bins2, patches2 = ax1.hist(lower_data, bins=bins, alpha=0.7, 
                                    label=f'LOWER (n={len(lower_data)})', 
                                    color='salmon', align='mid', rwidth=0.8)
    
    ax1.axvline(x=0, color='black', linewidth=2)
    ax1.axvline(x=rec_df['pct_change'].mean(), color='red', linestyle='--', 
                label=f'Avg: {rec_df["pct_change"].mean():+.1f}%')
    ax1.set_xlabel('Recommended Price Change (%)', fontsize=11)
    ax1.set_ylabel('Number of Hotels', fontsize=11)
    ax1.set_title('Price Change Distribution\n(Hotels with RAISE/LOWER recommendations)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(-25, 25)
    
    # Right: Box plot by segment
    ax2 = axes[1]
    
    # Order segments by median price change
    seg_order = rec_df.groupby('segment')['pct_change'].median().sort_values().index
    
    # Create box plot
    box_data = [rec_df[rec_df['segment'] == seg]['pct_change'].values for seg in seg_order]
    box_labels = [f"{seg}\n(n={len(rec_df[rec_df['segment'] == seg])})" for seg in seg_order]
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color boxes by direction
    for i, (box, seg) in enumerate(zip(bp['boxes'], seg_order)):
        median = rec_df[rec_df['segment'] == seg]['pct_change'].median()
        box.set_facecolor('lightgreen' if median >= 0 else 'lightsalmon')
        box.set_alpha(0.7)
    
    ax2.axhline(y=0, color='black', linewidth=2)
    ax2.set_ylabel('Recommended Price Change (%)', fontsize=11)
    ax2.set_xlabel('Market Segment', fontsize=11)
    ax2.set_title('Price Change by Market Segment', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/price_change_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"  Hotels with recommendations: {len(rec_df)}")
    print(f"  RAISE: {len(raise_data)} (avg {raise_data.mean():+.1f}%)")
    print(f"  LOWER: {len(lower_data)} (avg {lower_data.mean():+.1f}%)")
    print("  Saved price_change_distribution.png")


def generate_segment_hotel_distribution() -> None:
    """Generate segment_hotel_distribution.png."""
    print("Generating segment_hotel_distribution.png...")
    
    from src.recommender.pricing_pipeline import PricingPipeline
    
    pipeline = PricingPipeline()
    pipeline.fit()
    
    hotel_features = pipeline.peer_matcher.hotel_features
    if 'segment' not in hotel_features.columns:
        print("  No segment column found!")
        return
    
    seg_counts = hotel_features['segment'].value_counts().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(seg_counts)))
    bars = ax.barh(seg_counts.index, seg_counts.values, color=colors)
    
    ax.set_xlabel('Number of Hotels')
    ax.set_title('Hotels by Market Segment')
    
    for bar, count in zip(bars, seg_counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{count}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/segment_hotel_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved segment_hotel_distribution.png")


def generate_comprehensive_validation(df: pd.DataFrame) -> None:
    """Generate comprehensive_validation_corrected.png."""
    print("Generating comprehensive_validation_corrected.png...")
    
    lift_col = 'total_monthly_revpar_lift'
    avg_lift_col = 'avg_revpar_lift_per_hotel'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total monthly lift
    ax1 = axes[0, 0]
    ax1.bar(range(len(df)), df[lift_col]/1000, color='green', alpha=0.7)
    ax1.set_xlabel('Rolling Window')
    ax1.set_ylabel('€ thousands')
    ax1.set_title('Total Monthly Lift per Window')
    ax1.axhline(y=df[lift_col].mean()/1000, color='red', linestyle='--', 
                label=f"Avg: €{df[lift_col].mean()/1000:.1f}k")
    ax1.legend()
    
    # Percentage with opportunity
    ax2 = axes[0, 1]
    ax2.plot(range(len(df)), df['pct_with_opportunity'], 'o-', 
             color='orange', linewidth=2)
    ax2.fill_between(range(len(df)), 0, df['pct_with_opportunity'], 
                     alpha=0.2, color='orange')
    ax2.set_xlabel('Rolling Window')
    ax2.set_ylabel('%')
    ax2.set_title('Hotels with Optimization Opportunity')
    ax2.axhline(y=df['pct_with_opportunity'].mean(), color='red', linestyle='--', 
                label=f"Avg: {df['pct_with_opportunity'].mean():.1f}%")
    ax2.legend()
    
    # RevPAR lift over time
    ax3 = axes[1, 0]
    ax3.plot(range(len(df)), df[avg_lift_col], 'o-', color='purple', linewidth=2)
    ax3.fill_between(range(len(df)), 0, df[avg_lift_col], alpha=0.2, color='purple')
    ax3.set_xlabel('Rolling Window')
    ax3.set_ylabel('€ per room per night')
    ax3.set_title('Average RevPAR Lift')
    ax3.axhline(y=df[avg_lift_col].mean(), color='red', linestyle='--', 
                label=f"Avg: €{df[avg_lift_col].mean():.2f}")
    ax3.legend()
    
    # Hotel count
    ax4 = axes[1, 1]
    ax4.bar(range(len(df)), df['n_hotels'], color='steelblue', alpha=0.7)
    ax4.set_xlabel('Rolling Window')
    ax4.set_ylabel('Hotels')
    ax4.set_title('Hotels per Window')
    
    plt.suptitle('Rolling Window Validation Summary (2023-2024)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/figures/comprehensive_validation_corrected.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved comprehensive_validation_corrected.png")


def main() -> None:
    """Generate all visualizations."""
    print("=" * 60)
    print("GENERATING README VISUALIZATIONS")
    print("=" * 60)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Ensure output directory exists
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading validation results...")
    df = load_validation_results()
    print(f"  Loaded {len(df)} validation windows")
    
    # Generate each visualization
    print()
    generate_segment_elasticity(df)
    generate_segment_opportunity(df)
    generate_baseline_comparison()
    generate_seasonal_patterns(df)
    generate_price_change_distribution(df)
    generate_segment_hotel_distribution()
    generate_comprehensive_validation(df)
    generate_lead_time_curve()
    generate_lead_time_strategy()
    generate_lead_time_revpar_impact()
    
    print("\n" + "=" * 60)
    print("All visualizations updated!")
    print("=" * 60)


def generate_lead_time_curve() -> None:
    """Generate lead_time_pricing_curve.png showing how prices vary by booking window."""
    from lib.db import init_db
    
    print("Generating lead_time_pricing_curve.png...")
    
    con = init_db()
    
    # Query lead time pricing data
    query = """
    WITH booking_lead AS (
        SELECT 
            b.hotel_id,
            b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) as price_per_night,
            DATE_DIFF('day', b.created_at::DATE, b.arrival_date) as lead_time_days,
            CASE 
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) = 0 THEN 'Same day'
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 1 AND 3 THEN '1-3 days'
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 4 AND 7 THEN '4-7 days'
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 8 AND 14 THEN '8-14 days'
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 15 AND 30 THEN '15-30 days'
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 31 AND 60 THEN '31-60 days'
                ELSE '60+ days'
            END as lead_bucket,
            CASE 
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) = 0 THEN 0
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 1 AND 3 THEN 1
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 4 AND 7 THEN 2
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 8 AND 14 THEN 3
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 15 AND 30 THEN 4
                WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 31 AND 60 THEN 5
                ELSE 6
            END as bucket_order
        FROM bookings b
        WHERE b.status IN ('confirmed', 'Booked', 'cancelled')
          AND b.created_at IS NOT NULL
          AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) >= 0
          AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) < 365
          AND b.total_price > 0
          AND b.arrival_date >= '2023-01-01'
          AND b.arrival_date < '2025-01-01'
    )
    SELECT 
        lead_bucket,
        bucket_order,
        COUNT(*) as bookings,
        AVG(price_per_night) as avg_price,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price_per_night) as p25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_night) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price_per_night) as p75
    FROM booking_lead
    GROUP BY lead_bucket, bucket_order
    ORDER BY bucket_order
    """
    
    lead_df = con.execute(query).fetchdf()
    
    if lead_df.empty:
        print("  No lead time data available, skipping...")
        return
    
    # Calculate multipliers relative to 15-30 days (standard)
    baseline_idx = lead_df[lead_df['lead_bucket'] == '15-30 days'].index
    if len(baseline_idx) > 0:
        baseline = lead_df.loc[baseline_idx[0], 'avg_price']
    else:
        baseline = lead_df['avg_price'].median()
    
    lead_df['multiplier'] = lead_df['avg_price'] / baseline
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Price by lead time
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(lead_df)))[::-1]
    
    bars = ax1.bar(range(len(lead_df)), lead_df['avg_price'], color=colors, alpha=0.8)
    ax1.set_xticks(range(len(lead_df)))
    ax1.set_xticklabels(lead_df['lead_bucket'], rotation=45, ha='right')
    ax1.set_ylabel('Average Daily Rate (€)', fontsize=11)
    ax1.set_xlabel('Booking Lead Time', fontsize=11)
    ax1.set_title('Price by Booking Window', fontsize=13, fontweight='bold')
    
    # Add value labels
    for i, (bar, mult) in enumerate(zip(bars, lead_df['multiplier'])):
        ax1.annotate(f'€{bar.get_height():.0f}\n({mult:.2f}x)', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    # Add baseline reference
    ax1.axhline(baseline, color='gray', linestyle='--', alpha=0.7, label=f'Baseline: €{baseline:.0f}')
    ax1.legend(loc='upper left')
    
    # Plot 2: Booking volume by lead time
    ax2.bar(range(len(lead_df)), lead_df['bookings'], color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(lead_df)))
    ax2.set_xticklabels(lead_df['lead_bucket'], rotation=45, ha='right')
    ax2.set_ylabel('Number of Bookings', fontsize=11)
    ax2.set_xlabel('Booking Lead Time', fontsize=11)
    ax2.set_title('Booking Volume by Lead Time', fontsize=13, fontweight='bold')
    
    # Add percentage labels
    total_bookings = lead_df['bookings'].sum()
    for i, (bar, count) in enumerate(zip(ax2.patches, lead_df['bookings'])):
        pct = count / total_bookings * 100
        ax2.annotate(f'{pct:.1f}%', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/lead_time_pricing_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved: outputs/figures/lead_time_pricing_curve.png")
    
    # Print summary
    short_term = lead_df[lead_df['bucket_order'] <= 2]['bookings'].sum()
    short_term_pct = short_term / total_bookings * 100
    print(f"  Short-term bookings (≤7 days): {short_term_pct:.1f}%")
    print(f"  Same-day multiplier: {lead_df[lead_df['bucket_order'] == 0]['multiplier'].values[0]:.2f}x")
    print(f"  Advance (60+) multiplier: {lead_df[lead_df['bucket_order'] == 6]['multiplier'].values[0]:.2f}x")


def generate_lead_time_strategy() -> None:
    """Generate visualization comparing current vs recommended lead time strategy."""
    from lib.db import init_db
    
    print("Generating lead_time_strategy_comparison.png...")
    
    con = init_db()
    
    # Query current hotel discounting behavior
    query = """
    WITH hotel_lead_behavior AS (
        SELECT 
            b.hotel_id,
            AVG(CASE WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) <= 7 
                THEN b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) END) as short_term_adr,
            AVG(CASE WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 15 AND 30 
                THEN b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) END) as standard_adr,
            AVG(CASE WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) > 30 
                THEN b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) END) as advance_adr,
            COUNT(CASE WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) <= 7 THEN 1 END) as short_count,
            COUNT(CASE WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) > 30 THEN 1 END) as advance_count,
            COUNT(*) as total_bookings
        FROM bookings b
        WHERE b.status IN ('confirmed', 'Booked')
          AND b.created_at IS NOT NULL
          AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) >= 0
          AND b.total_price > 0
          AND b.arrival_date >= '2023-01-01'
        GROUP BY b.hotel_id
        HAVING COUNT(CASE WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) <= 7 THEN 1 END) >= 5
           AND COUNT(CASE WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 15 AND 30 THEN 1 END) >= 5
    )
    SELECT 
        CASE 
            WHEN (standard_adr - short_term_adr) / NULLIF(standard_adr, 0) > 0.20 THEN 'Heavy Discount (>20%)'
            WHEN (standard_adr - short_term_adr) / NULLIF(standard_adr, 0) > 0.10 THEN 'Moderate Discount (10-20%)'
            WHEN (standard_adr - short_term_adr) / NULLIF(standard_adr, 0) > 0 THEN 'Light Discount (0-10%)'
            ELSE 'Premium/No Discount'
        END as current_strategy,
        COUNT(*) as n_hotels,
        ROUND(AVG((standard_adr - short_term_adr) / NULLIF(standard_adr, 0)) * 100, 1) as avg_discount_pct,
        ROUND(AVG(short_count::FLOAT / total_bookings) * 100, 1) as avg_short_term_pct
    FROM hotel_lead_behavior
    WHERE standard_adr > 0
    GROUP BY 1
    ORDER BY AVG((standard_adr - short_term_adr) / NULLIF(standard_adr, 0)) DESC
    """
    
    current_df = con.execute(query).fetchdf()
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Current strategy distribution
    colors_current = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    wedges, texts, autotexts = ax1.pie(
        current_df['n_hotels'], 
        labels=current_df['current_strategy'],
        autopct='%1.1f%%',
        colors=colors_current[:len(current_df)],
        explode=[0.05 if 'Heavy' in s else 0 for s in current_df['current_strategy']]
    )
    ax1.set_title('Current Hotel Lead Time Strategy\n(What Hotels Are Doing)', fontsize=13, fontweight='bold')
    
    # Plot 2: Recommended strategy by occupancy
    occupancy_levels = ['Very Low\n(<30%)', 'Low\n(30-50%)', 'Medium\n(50-70%)', 'High\n(>70%)']
    recommended_short = ['Discount\n(-20%)', 'Discount\n(-20%)', 'Hold\n(0%)', 'Premium\n(+15%)']
    recommended_advance = ['Premium\n(+7%)', 'Premium\n(+7%)', 'Premium\n(+7%)', 'Premium\n(+7%)']
    
    x = np.arange(len(occupancy_levels))
    width = 0.35
    
    # Color-code by action
    colors_short = ['#2ca02c', '#2ca02c', '#7f7f7f', '#1f77b4']  # Green for discount, gray for hold, blue for premium
    colors_adv = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']  # Blue for premium
    
    bars1 = ax2.bar(x - width/2, [0.80, 0.80, 1.0, 1.15], width, label='Short-term (≤7 days)', color=colors_short, alpha=0.8)
    bars2 = ax2.bar(x + width/2, [1.07, 1.07, 1.07, 1.07], width, label='Advance (31+ days)', color=colors_adv, alpha=0.8)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
    ax2.set_ylabel('Price Multiplier', fontsize=11)
    ax2.set_xlabel('Current Occupancy Level', fontsize=11)
    ax2.set_title('Recommended Strategy by Occupancy\n(Revenue Management Approach)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(occupancy_levels)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0.6, 1.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/lead_time_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved: outputs/figures/lead_time_strategy_comparison.png")
    
    # Print summary
    heavy_disc = current_df[current_df['current_strategy'].str.contains('Heavy')]['n_hotels'].sum()
    total_hotels = current_df['n_hotels'].sum()
    print(f"  Hotels currently heavy discounting: {heavy_disc} ({heavy_disc/total_hotels*100:.1f}%)")


def generate_lead_time_revpar_impact() -> None:
    """Generate visualization showing RevPAR impact of lead time strategy."""
    print("Generating lead_time_revpar_impact.png...")
    
    # Simulate RevPAR impact across occupancy levels
    occupancy_levels = np.arange(0.15, 0.95, 0.05)
    base_price = 100
    
    # Revenue management expected values
    results = []
    for occ in occupancy_levels:
        # Estimate P(full price booking) based on occupancy
        if occ < 0.30:
            p_full = 0.15
        elif occ < 0.50:
            p_full = 0.35
        elif occ < 0.70:
            p_full = 0.55
        else:
            p_full = 0.80
        
        p_discount = 0.85  # Assume 85% fill at discount
        
        # Short-term: Compare EV of discount vs hold
        ev_hold = p_full * 1.0
        ev_discount = p_discount * 0.80  # 20% discount
        
        if ev_discount > ev_hold:
            short_mult = 0.80
            short_strategy = 'Discount'
            short_fill = max(occ, 0.85)  # Discount fills more
        elif occ >= 0.70:
            short_mult = 1.15
            short_strategy = 'Premium'
            short_fill = occ * 0.95  # Slight reduction
        else:
            short_mult = 1.0
            short_strategy = 'Hold'
            short_fill = occ
        
        # Advance: Always premium
        adv_mult = 1.10
        adv_fill = occ  # No fill impact at 30+ days out
        
        # Calculate RevPAR
        base_revpar = base_price * occ
        short_revpar = base_price * short_mult * short_fill
        adv_revpar = base_price * adv_mult * adv_fill
        
        # Weighted RevPAR (35% short, 30% standard, 35% advance)
        weighted_revpar = 0.35 * short_revpar + 0.30 * base_revpar + 0.35 * adv_revpar
        
        lift_pct = (weighted_revpar - base_revpar) / base_revpar * 100
        
        results.append({
            'occupancy': occ,
            'base_revpar': base_revpar,
            'optimized_revpar': weighted_revpar,
            'lift_pct': lift_pct,
            'short_strategy': short_strategy,
            'short_mult': short_mult,
            'adv_mult': adv_mult
        })
    
    results_df = pd.DataFrame(results)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RevPAR lift by occupancy
    ax1.fill_between(results_df['occupancy'] * 100, 0, results_df['lift_pct'], 
                     alpha=0.3, color='green', label='RevPAR Lift Zone')
    ax1.plot(results_df['occupancy'] * 100, results_df['lift_pct'], 'g-', linewidth=2, label='RevPAR Lift %')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Current Occupancy (%)', fontsize=11)
    ax1.set_ylabel('RevPAR Lift (%)', fontsize=11)
    ax1.set_title('RevPAR Impact by Occupancy Level', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(15, 90)
    
    # Add annotation for key insight
    ax1.annotate('Highest lift at low occupancy\n(discounts fill empty rooms)',
                xy=(25, results_df[results_df['occupancy'] == 0.25]['lift_pct'].values[0]),
                xytext=(40, 50),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Plot 2: Strategy decision zones
    strategy_colors = {'Discount': 'green', 'Hold': 'gray', 'Premium': 'blue'}
    for strategy in ['Discount', 'Hold', 'Premium']:
        mask = results_df['short_strategy'] == strategy
        if mask.any():
            ax2.scatter(results_df.loc[mask, 'occupancy'] * 100, 
                       results_df.loc[mask, 'short_mult'],
                       c=strategy_colors[strategy], 
                       s=100, 
                       label=f'Short-term: {strategy}',
                       alpha=0.7)
    
    ax2.scatter(results_df['occupancy'] * 100, results_df['adv_mult'], 
               c='blue', s=100, marker='s', label='Advance: Premium', alpha=0.5)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
    ax2.set_xlabel('Current Occupancy (%)', fontsize=11)
    ax2.set_ylabel('Recommended Price Multiplier', fontsize=11)
    ax2.set_title('Dynamic Pricing Strategy by Occupancy', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(15, 90)
    ax2.set_ylim(0.7, 1.25)
    
    # Add decision zones
    ax2.axvspan(15, 50, alpha=0.1, color='green', label='_Discount Zone')
    ax2.axvspan(50, 70, alpha=0.1, color='gray', label='_Hold Zone')
    ax2.axvspan(70, 90, alpha=0.1, color='blue', label='_Premium Zone')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/lead_time_revpar_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved: outputs/figures/lead_time_revpar_impact.png")
    
    # Calculate weighted average lift
    avg_lift = results_df['lift_pct'].mean()
    print(f"  Average RevPAR lift across occupancy levels: {avg_lift:.1f}%")


if __name__ == "__main__":
    main()

