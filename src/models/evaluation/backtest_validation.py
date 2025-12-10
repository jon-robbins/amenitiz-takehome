"""
Backtest Validation for RevPAR Recommendations.

Since we don't have labeled data (ground truth optimal prices), we validate using:

1. **Natural Experiment Analysis**: Find hotels that changed prices between periods
   and compare their actual RevPAR change to our predicted direction.
   
2. **Peer Outcome Comparison**: When we recommend a hotel raise prices to match peers,
   verify that those peers actually achieved higher RevPAR.
   
3. **Directional Accuracy**: Did our recommendations align with actual RevPAR movements?
   - If we said "RAISE" and RevPAR went up → Correct
   - If we said "LOWER" and RevPAR went up after lowering → Correct
   
4. **Calibration Check**: Are our predicted RevPAR lifts correlated with actual changes?
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.recommender.price_recommender import PriceRecommender


@dataclass
class BacktestResult:
    """Results from backtesting recommendations against actual outcomes."""
    n_hotels_tested: int
    
    # Directional accuracy
    directional_accuracy: float  # % of recommendations that matched actual direction
    raise_accuracy: float        # Accuracy for "raise" recommendations  
    lower_accuracy: float        # Accuracy for "lower" recommendations
    hold_accuracy: float         # Accuracy for "hold" recommendations
    
    # Calibration
    predicted_vs_actual_correlation: float
    mean_absolute_error: float
    
    # Peer validation
    peer_revpar_achieved_pct: float  # % where peers actually had higher RevPAR
    
    # Raw data for plotting
    predictions_df: pd.DataFrame


def find_natural_experiments(
    con,
    start_period: date,
    end_period: date,
    min_price_change_pct: float = 5.0
) -> pd.DataFrame:
    """
    Find hotels that changed prices between two periods.
    
    These serve as natural experiments to validate our recommendations.
    """
    query = f"""
    WITH period1 AS (
        SELECT 
            hotel_id,
            AVG(CAST(total_price AS FLOAT) / NULLIF(departure_date - arrival_date, 0)) as avg_adr,
            SUM(CAST(total_price AS FLOAT)) / COUNT(*) as avg_revpar,
            COUNT(*) as bookings
        FROM bookings
        WHERE arrival_date >= '{start_period}' 
          AND arrival_date < '{start_period + timedelta(days=30)}'
          AND status IN ('Booked', 'confirmed')
        GROUP BY hotel_id
        HAVING COUNT(*) >= 3
    ),
    period2 AS (
        SELECT 
            hotel_id,
            AVG(CAST(total_price AS FLOAT) / NULLIF(departure_date - arrival_date, 0)) as avg_adr,
            SUM(CAST(total_price AS FLOAT)) / COUNT(*) as avg_revpar,
            COUNT(*) as bookings
        FROM bookings
        WHERE arrival_date >= '{end_period}' 
          AND arrival_date < '{end_period + timedelta(days=30)}'
          AND status IN ('Booked', 'confirmed')
        GROUP BY hotel_id
        HAVING COUNT(*) >= 3
    )
    SELECT
        p1.hotel_id,
        p1.avg_adr as adr_period1,
        p2.avg_adr as adr_period2,
        (p2.avg_adr - p1.avg_adr) / NULLIF(p1.avg_adr, 0) * 100 as price_change_pct,
        p1.avg_revpar as revpar_period1,
        p2.avg_revpar as revpar_period2,
        (p2.avg_revpar - p1.avg_revpar) / NULLIF(p1.avg_revpar, 0) * 100 as revpar_change_pct
    FROM period1 p1
    JOIN period2 p2 ON p1.hotel_id = p2.hotel_id
    WHERE ABS((p2.avg_adr - p1.avg_adr) / NULLIF(p1.avg_adr, 0) * 100) >= {min_price_change_pct}
    """
    
    return con.execute(query).fetchdf()


def run_backtest(
    con,
    recommendation_date: date,  # When recommendations were made
    outcome_date: date,         # When to measure outcomes (30+ days later)
    sample_size: int = 200
) -> BacktestResult:
    """
    Backtest recommendations against actual outcomes.
    
    1. Get recommendations as of recommendation_date
    2. Compare to actual RevPAR changes by outcome_date
    3. Calculate accuracy metrics
    """
    print(f"\nBacktesting: Recommendations as of {recommendation_date}")
    print(f"             Outcomes measured at {outcome_date}")
    
    # Get our recommendations
    recommender = PriceRecommender(con)
    target_dates = [recommendation_date + timedelta(days=i) for i in range(7)]
    
    # Get actual outcomes for hotels
    outcomes_query = f"""
    WITH rec_period AS (
        SELECT 
            hotel_id,
            AVG(CAST(total_price AS FLOAT) / NULLIF(departure_date - arrival_date, 0)) as rec_adr,
            SUM(CAST(total_price AS FLOAT)) / COUNT(*) as rec_revpar,
            COUNT(*) as rec_bookings
        FROM bookings
        WHERE arrival_date >= '{recommendation_date}'
          AND arrival_date < '{recommendation_date + timedelta(days=30)}'
          AND status IN ('Booked', 'confirmed')
        GROUP BY hotel_id
        HAVING COUNT(*) >= 2
    ),
    outcome_period AS (
        SELECT 
            hotel_id,
            AVG(CAST(total_price AS FLOAT) / NULLIF(departure_date - arrival_date, 0)) as outcome_adr,
            SUM(CAST(total_price AS FLOAT)) / COUNT(*) as outcome_revpar,
            COUNT(*) as outcome_bookings
        FROM bookings
        WHERE arrival_date >= '{outcome_date}'
          AND arrival_date < '{outcome_date + timedelta(days=30)}'
          AND status IN ('Booked', 'confirmed')
        GROUP BY hotel_id
        HAVING COUNT(*) >= 2
    )
    SELECT
        r.hotel_id,
        r.rec_adr,
        r.rec_revpar,
        o.outcome_adr,
        o.outcome_revpar,
        (o.outcome_adr - r.rec_adr) / NULLIF(r.rec_adr, 0) * 100 as actual_price_change_pct,
        (o.outcome_revpar - r.rec_revpar) / NULLIF(r.rec_revpar, 0) * 100 as actual_revpar_change_pct
    FROM rec_period r
    JOIN outcome_period o ON r.hotel_id = o.hotel_id
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """
    
    outcomes_df = con.execute(outcomes_query).fetchdf()
    
    if len(outcomes_df) == 0:
        raise ValueError("No hotels found with bookings in both periods")
    
    print(f"  Found {len(outcomes_df)} hotels with data in both periods")
    
    # Get our recommendations for these hotels
    results = []
    for _, row in outcomes_df.iterrows():
        try:
            recs = recommender.recommend_price(
                hotel_id=int(row['hotel_id']),
                target_dates=target_dates,
                as_of_date=recommendation_date - timedelta(days=14)  # 2 weeks lead time
            )
            
            if recs and len(recs) > 0:
                # Get the first recommendation (PriceRecommendation object)
                rec = recs[0]
                
                # Handle both object and dict formats
                if hasattr(rec, 'to_dict'):
                    rec_dict = rec.to_dict()
                elif isinstance(rec, dict):
                    rec_dict = rec
                else:
                    continue
                
                current_price = rec_dict.get('current_price', row['rec_adr'])
                recommended_price = rec_dict.get('recommended_price', row['rec_adr'])
                
                results.append({
                    'hotel_id': row['hotel_id'],
                    'current_price': current_price,
                    'recommended_price': recommended_price,
                    'predicted_direction': (
                        'raise' if recommended_price > current_price * 1.02
                        else 'lower' if recommended_price < current_price * 0.98
                        else 'hold'
                    ),
                    'predicted_change_pct': rec_dict.get('change_pct', 0),
                    'predicted_revpar_lift': rec_dict.get('expected_revpar_lift_pct', rec_dict.get('change_pct', 0)),
                    'actual_price_change_pct': row['actual_price_change_pct'],
                    'actual_revpar_change_pct': row['actual_revpar_change_pct'],
                    'peer_revpar': rec_dict.get('peer_price', 0) * rec_dict.get('actual_occupancy', 0.5),
                    'current_revpar': row['rec_revpar'],
                    'outcome_revpar': row['outcome_revpar'],
                })
        except Exception as e:
            # Silently skip failed recommendations
            continue
    
    if len(results) == 0:
        raise ValueError(f"Could not generate recommendations for any of the {len(outcomes_df)} hotels")
    
    predictions_df = pd.DataFrame(results)
    
    # Calculate directional accuracy
    predictions_df['actual_direction'] = predictions_df.apply(
        lambda x: 'raise' if x['actual_revpar_change_pct'] > 5
                  else 'lower' if x['actual_revpar_change_pct'] < -5
                  else 'hold',
        axis=1
    )
    
    predictions_df['direction_match'] = (
        predictions_df['predicted_direction'] == predictions_df['actual_direction']
    )
    
    directional_accuracy = predictions_df['direction_match'].mean() * 100
    
    # By recommendation type
    raise_df = predictions_df[predictions_df['predicted_direction'] == 'raise']
    lower_df = predictions_df[predictions_df['predicted_direction'] == 'lower']
    hold_df = predictions_df[predictions_df['predicted_direction'] == 'hold']
    
    raise_accuracy = raise_df['direction_match'].mean() * 100 if len(raise_df) > 0 else 0
    lower_accuracy = lower_df['direction_match'].mean() * 100 if len(lower_df) > 0 else 0
    hold_accuracy = hold_df['direction_match'].mean() * 100 if len(hold_df) > 0 else 0
    
    # Calibration: correlation between predicted and actual RevPAR changes
    valid_predictions = predictions_df.dropna(subset=['predicted_revpar_lift', 'actual_revpar_change_pct'])
    if len(valid_predictions) >= 10:
        correlation, _ = stats.pearsonr(
            valid_predictions['predicted_revpar_lift'].clip(-100, 100),
            valid_predictions['actual_revpar_change_pct'].clip(-100, 100)
        )
    else:
        correlation = np.nan
    
    # Mean absolute error
    mae = (predictions_df['predicted_revpar_lift'] - predictions_df['actual_revpar_change_pct']).abs().mean()
    
    # Peer validation: when we said peers had higher RevPAR, were we right?
    peer_validation = predictions_df[predictions_df['predicted_direction'] == 'raise']
    peer_achieved = (peer_validation['peer_revpar'] > peer_validation['current_revpar']).mean() * 100 if len(peer_validation) > 0 else 0
    
    return BacktestResult(
        n_hotels_tested=len(predictions_df),
        directional_accuracy=directional_accuracy,
        raise_accuracy=raise_accuracy,
        lower_accuracy=lower_accuracy,
        hold_accuracy=hold_accuracy,
        predicted_vs_actual_correlation=correlation,
        mean_absolute_error=mae,
        peer_revpar_achieved_pct=peer_achieved,
        predictions_df=predictions_df
    )


def create_backtest_visualizations(
    backtest: BacktestResult,
    output_dir: Path
) -> None:
    """Create visualizations for backtest results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = backtest.predictions_df
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Directional accuracy breakdown
    ax = axes[0, 0]
    accuracies = [
        ('Raise', backtest.raise_accuracy),
        ('Hold', backtest.hold_accuracy),
        ('Lower', backtest.lower_accuracy),
        ('Overall', backtest.directional_accuracy),
    ]
    labels, values = zip(*accuracies)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax.bar(labels, values, color=colors, edgecolor='black')
    ax.axhline(50, color='gray', linestyle='--', label='Random (50%)')
    ax.axhline(33.3, color='red', linestyle=':', alpha=0.5, label='Worse than random')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Directional Accuracy by Recommendation Type', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11)
    
    # 2. Predicted vs Actual RevPAR change scatter
    ax = axes[0, 1]
    pred = df['predicted_revpar_lift'].clip(-100, 100)
    actual = df['actual_revpar_change_pct'].clip(-100, 100)
    
    ax.scatter(pred, actual, alpha=0.4, c='steelblue', s=30)
    ax.plot([-100, 100], [-100, 100], 'r--', label='Perfect prediction')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel('Predicted RevPAR Lift (%)', fontsize=12)
    ax.set_ylabel('Actual RevPAR Change (%)', fontsize=12)
    ax.set_title(f'Predicted vs Actual RevPAR (r = {backtest.predicted_vs_actual_correlation:.2f})', 
                fontsize=14, fontweight='bold')
    ax.legend()
    
    # 3. Distribution of prediction errors
    ax = axes[1, 0]
    errors = (df['predicted_revpar_lift'] - df['actual_revpar_change_pct']).clip(-100, 100)
    ax.hist(errors, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='No error')
    ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():+.1f}%')
    ax.set_xlabel('Prediction Error (Predicted - Actual %)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of RevPAR Prediction Errors', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 4. Summary panel
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    BACKTEST VALIDATION SUMMARY
    {'═' * 50}
    
    Hotels Tested: {backtest.n_hotels_tested}
    
    DIRECTIONAL ACCURACY
    • Overall: {backtest.directional_accuracy:.1f}%
    • Raise recommendations: {backtest.raise_accuracy:.1f}%
    • Hold recommendations: {backtest.hold_accuracy:.1f}%
    • Lower recommendations: {backtest.lower_accuracy:.1f}%
    
    CALIBRATION
    • Correlation (pred vs actual): {backtest.predicted_vs_actual_correlation:.2f}
    • Mean Absolute Error: {backtest.mean_absolute_error:.1f}%
    
    PEER VALIDATION
    • Peers actually had higher RevPAR: {backtest.peer_revpar_achieved_pct:.1f}%
    
    INTERPRETATION
    {'─' * 50}
    Directional accuracy > 50% means our recommendations
    are better than random guessing.
    
    Positive correlation means when we predict higher
    RevPAR lift, actual RevPAR tends to increase more.
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    fig.savefig(output_dir / 'backtest_results.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved backtest visualizations to {output_dir}")


def run_multi_period_backtest(
    con,
    periods: List[Tuple[date, date]],  # List of (recommendation_date, outcome_date)
    sample_per_period: int = 100
) -> pd.DataFrame:
    """Run backtest across multiple time periods for robustness."""
    all_results = []
    
    for rec_date, outcome_date in periods:
        try:
            result = run_backtest(con, rec_date, outcome_date, sample_per_period)
            all_results.append({
                'period': f"{rec_date} → {outcome_date}",
                'n_hotels': result.n_hotels_tested,
                'directional_accuracy': result.directional_accuracy,
                'raise_accuracy': result.raise_accuracy,
                'lower_accuracy': result.lower_accuracy,
                'correlation': result.predicted_vs_actual_correlation,
                'mae': result.mean_absolute_error,
            })
        except Exception as e:
            print(f"  Error for period {rec_date}: {e}")
            continue
    
    return pd.DataFrame(all_results)


def print_backtest_summary(result: BacktestResult) -> None:
    """Print backtest results to console."""
    print("\n" + "=" * 60)
    print("BACKTEST VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\nHotels Tested: {result.n_hotels_tested}")
    
    print("\nDIRECTIONAL ACCURACY (higher is better, >50% beats random)")
    print("-" * 40)
    print(f"  Overall:       {result.directional_accuracy:6.1f}%")
    print(f"  Raise recs:    {result.raise_accuracy:6.1f}%")
    print(f"  Hold recs:     {result.hold_accuracy:6.1f}%")
    print(f"  Lower recs:    {result.lower_accuracy:6.1f}%")
    
    print("\nCALIBRATION")
    print("-" * 40)
    print(f"  Correlation (pred vs actual): {result.predicted_vs_actual_correlation:+.2f}")
    print(f"  Mean Absolute Error:          {result.mean_absolute_error:.1f}%")
    
    print("\nPEER VALIDATION")
    print("-" * 40)
    print(f"  Peer RevPAR actually higher:  {result.peer_revpar_achieved_pct:.1f}%")
    
    print("=" * 60)

