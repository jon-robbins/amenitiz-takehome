"""
Evaluation and backtesting for price recommendation model.

Validates model performance using historical data where we observe
both the hotel's pricing strategy and outcome (RevPAR).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ml_pipeline.predict import PriceRecommender
from ml_pipeline.features import engineer_all_features
from ml_pipeline.config import (
    GLOBAL_ELASTICITY,
    STRATEGIES,
    calculate_expected_revpar_change,
    MODEL_DIR
)


@dataclass
class BacktestResult:
    """Results from backtesting the model."""
    n_hotels: int
    n_observations: int
    
    # Price prediction metrics
    price_mape: float  # Mean Absolute Percentage Error
    price_rmse: float  # Root Mean Squared Error
    price_r2: float    # R-squared
    
    # RevPAR metrics (for hotels that followed recommendations)
    revpar_lift_actual: float  # Observed lift for premium-priced hotels
    revpar_lift_predicted: float  # Model's predicted lift
    
    # Strategy performance
    strategy_accuracy: Dict[str, float]  # % of times strategy would have worked


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / y_true) * 100
    """
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def backtest_price_predictions(
    recommender: PriceRecommender,
    test_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Backtests price predictions against actual prices.
    
    Args:
        recommender: Trained PriceRecommender instance
        test_data: DataFrame with actual prices (avg_adr)
    
    Returns:
        Dict with MAPE, RMSE, R² metrics
    """
    predictions = []
    actuals = []
    errors = 0
    
    for idx, row in test_data.iterrows():
        try:
            # Prepare single-row DataFrame for prediction
            hotel_data = test_data.loc[[idx]].copy()
            pred_price = recommender.predict_peer_price(hotel_data)
            
            if not np.isnan(pred_price) and pred_price > 0:
                predictions.append(pred_price)
                actuals.append(row['avg_adr'])
        except Exception as e:
            errors += 1
            continue
    
    if len(predictions) == 0:
        return {
            'mape': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'),
            'n_samples': 0,
            'errors': errors
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return {
        'mape': calculate_mape(actuals, predictions),
        'rmse': calculate_rmse(actuals, predictions),
        'r2': calculate_r2(actuals, predictions),
        'n_samples': len(predictions),
        'errors': errors
    }


def backtest_revpar_strategy(
    matched_pairs: pd.DataFrame
) -> Dict[str, float]:
    """
    Backtests RevPAR strategy using matched pairs data.
    
    The matched pairs dataset contains hotels that priced higher (treatment)
    vs similar hotels that priced lower (control). We can validate that
    the premium pricing strategy actually worked.
    
    Args:
        matched_pairs: DataFrame with treatment/control pairs
    
    Returns:
        Dict with strategy performance metrics
    """
    # Calculate actual RevPAR for treatment and control
    matched_pairs = matched_pairs.copy()
    
    # RevPAR = Price × Occupancy
    matched_pairs['treatment_revpar'] = (
        matched_pairs['treatment_price'] * matched_pairs['treatment_occupancy']
    )
    matched_pairs['control_revpar'] = (
        matched_pairs['control_price'] * matched_pairs['control_occupancy']
    )
    
    # Calculate lift
    matched_pairs['revpar_lift'] = (
        (matched_pairs['treatment_revpar'] - matched_pairs['control_revpar']) / 
        matched_pairs['control_revpar']
    ) * 100
    
    # Filter valid pairs
    valid = matched_pairs['revpar_lift'].notna() & (matched_pairs['control_revpar'] > 0)
    
    # Metrics
    results = {
        'n_pairs': valid.sum(),
        'mean_revpar_lift': matched_pairs.loc[valid, 'revpar_lift'].mean(),
        'median_revpar_lift': matched_pairs.loc[valid, 'revpar_lift'].median(),
        'positive_lift_pct': (matched_pairs.loc[valid, 'revpar_lift'] > 0).mean() * 100,
        'mean_price_diff_pct': matched_pairs.loc[valid, 'price_diff_pct'].mean() * 100
    }
    
    # Strategy-specific success rates
    for strategy_name, strategy in STRATEGIES.items():
        deviation = strategy.price_deviation_pct
        
        # Find pairs where price diff is close to this strategy
        mask = (
            valid & 
            (matched_pairs['price_diff_pct'] * 100 >= deviation - 10) &
            (matched_pairs['price_diff_pct'] * 100 <= deviation + 10)
        )
        
        if mask.sum() > 0:
            success_rate = (matched_pairs.loc[mask, 'revpar_lift'] > 0).mean() * 100
            avg_lift = matched_pairs.loc[mask, 'revpar_lift'].mean()
            results[f'{strategy_name}_success_rate'] = success_rate
            results[f'{strategy_name}_avg_lift'] = avg_lift
            results[f'{strategy_name}_n_samples'] = mask.sum()
    
    return results


def evaluate_elasticity_prediction(
    matched_pairs: pd.DataFrame
) -> Dict[str, float]:
    """
    Compares model's elasticity assumption with empirical elasticity.
    
    The elasticity is key to the RevPAR calculation. If our assumed
    elasticity (-0.39) matches reality, predictions will be accurate.
    """
    # Calculate empirical elasticity for each pair
    matched_pairs = matched_pairs.copy()
    
    # Arc elasticity = (ΔQ/avg_Q) / (ΔP/avg_P)
    matched_pairs['price_pct_change'] = (
        (matched_pairs['treatment_price'] - matched_pairs['control_price']) /
        ((matched_pairs['treatment_price'] + matched_pairs['control_price']) / 2)
    )
    matched_pairs['occ_pct_change'] = (
        (matched_pairs['treatment_occupancy'] - matched_pairs['control_occupancy']) /
        ((matched_pairs['treatment_occupancy'] + matched_pairs['control_occupancy']) / 2)
    )
    
    # Handle edge cases
    valid = (
        matched_pairs['price_pct_change'].notna() & 
        (matched_pairs['price_pct_change'] != 0)
    )
    
    matched_pairs.loc[valid, 'empirical_elasticity'] = (
        matched_pairs.loc[valid, 'occ_pct_change'] / 
        matched_pairs.loc[valid, 'price_pct_change']
    )
    
    # Compare to model assumption
    empirical = matched_pairs.loc[valid, 'empirical_elasticity']
    
    return {
        'empirical_elasticity_mean': empirical.mean(),
        'empirical_elasticity_median': empirical.median(),
        'empirical_elasticity_std': empirical.std(),
        'model_elasticity': GLOBAL_ELASTICITY,
        'elasticity_error': abs(empirical.median() - GLOBAL_ELASTICITY),
        'n_pairs': valid.sum()
    }


def run_full_evaluation(
    model_dir: str = MODEL_DIR,
    matched_pairs_path: Optional[str] = None
) -> Dict:
    """
    Runs full evaluation suite.
    
    Args:
        model_dir: Directory containing trained model
        matched_pairs_path: Path to matched pairs CSV (optional)
    
    Returns:
        Dict with all evaluation metrics
    """
    from lib.db import init_db
    
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    results = {}
    
    # Load model
    print("\n1. Loading model...")
    try:
        recommender = PriceRecommender(model_dir)
    except FileNotFoundError:
        print("   ⚠️ Model not found. Train first with: python ml_pipeline/train.py")
        return {'error': 'Model not found'}
    
    # Load test data
    print("\n2. Loading test data...")
    con = init_db()
    
    # Use a holdout set (different time period or random sample)
    test_query = """
    WITH hotel_stats AS (
        SELECT 
            b.hotel_id,
            DATE_TRUNC('month', b.arrival_date) as month,
            AVG(br.total_price / (b.departure_date - b.arrival_date)) as avg_adr,
            COUNT(*) as n_bookings,
            hl.city,
            AVG(br.room_size) as avg_room_size,
            MODE() WITHIN GROUP (ORDER BY br.room_type) as room_type,
            MODE() WITHIN GROUP (ORDER BY COALESCE(NULLIF(br.room_view, ''), 'no_view')) as room_view,
            MAX(r.children_allowed) as children_allowed,
            MAX(r.max_occupancy) as room_capacity_pax,
            SUM(DISTINCT r.number_of_rooms) as total_capacity,
            -- Add missing features
            (CAST(MAX(r.events_allowed) AS INT) + 
             CAST(MAX(r.pets_allowed) AS INT) + 
             CAST(MAX(r.smoking_allowed) AS INT) + 
             CAST(MAX(r.children_allowed) AS INT)) AS amenities_score,
            SUM(CASE WHEN EXTRACT(ISODOW FROM b.arrival_date) >= 6 THEN 1 ELSE 0 END)::FLOAT / 
                NULLIF(COUNT(*), 0) AS weekend_ratio
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN rooms r ON br.room_id = r.id
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('confirmed', 'Booked')
          AND b.arrival_date >= '2024-06-01'  -- Holdout: last 6 months
        GROUP BY b.hotel_id, month, hl.city
    )
    SELECT * FROM hotel_stats
    WHERE avg_adr > 10 AND avg_adr < 1000
    """
    
    test_data = con.execute(test_query).fetchdf()
    print(f"   Loaded {len(test_data):,} test observations")
    
    # Engineer features
    test_data = engineer_all_features(test_data)
    
    # Price prediction evaluation - use batch approach matching training
    print("\n3. Evaluating price predictions...")
    try:
        from ml_pipeline.train import prepare_features
        
        # Use same feature preparation as training
        X_test, y_test, _, _ = prepare_features(test_data)
        
        # Load trained model directly
        import pickle
        from ml_pipeline.config import MODEL_DIR, MODEL_FILENAME
        with open(Path(MODEL_DIR) / MODEL_FILENAME, 'rb') as f:
            model = pickle.load(f)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Convert back from log scale
        y_test_exp = np.exp(y_test)
        y_pred_exp = np.exp(y_pred)
        
        price_metrics = {
            'mape': calculate_mape(y_test_exp.values, y_pred_exp),
            'rmse': calculate_rmse(y_test_exp.values, y_pred_exp),
            'r2': calculate_r2(y_test.values, y_pred),  # R² on log scale
            'n_samples': len(y_test)
        }
    except Exception as e:
        print(f"   Warning: {e}")
        price_metrics = {'mape': float('nan'), 'rmse': float('nan'), 'r2': float('nan'), 'n_samples': 0}
    
    results['price_prediction'] = price_metrics
    print(f"   MAPE: {price_metrics['mape']:.1f}%")
    print(f"   RMSE: €{price_metrics['rmse']:.2f}")
    print(f"   R²: {price_metrics['r2']:.4f}")
    
    # Matched pairs evaluation
    print("\n4. Evaluating RevPAR strategy...")
    if matched_pairs_path:
        pairs_path = Path(matched_pairs_path)
    else:
        pairs_path = Path('outputs/eda/elasticity/data/matched_pairs_with_replacement.csv')
    
    if pairs_path.exists():
        matched_pairs = pd.read_csv(pairs_path)
        
        revpar_metrics = backtest_revpar_strategy(matched_pairs)
        results['revpar_strategy'] = revpar_metrics
        print(f"   Matched pairs: {revpar_metrics['n_pairs']:,}")
        print(f"   Mean RevPAR lift: +{revpar_metrics['mean_revpar_lift']:.1f}%")
        print(f"   Positive lift rate: {revpar_metrics['positive_lift_pct']:.1f}%")
        
        # Elasticity evaluation
        print("\n5. Evaluating elasticity assumption...")
        elasticity_metrics = evaluate_elasticity_prediction(matched_pairs)
        results['elasticity'] = elasticity_metrics
        print(f"   Empirical elasticity: {elasticity_metrics['empirical_elasticity_median']:.3f}")
        print(f"   Model elasticity: {elasticity_metrics['model_elasticity']:.3f}")
        print(f"   Error: {elasticity_metrics['elasticity_error']:.3f}")
    else:
        print(f"   ⚠️ Matched pairs not found at {pairs_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Overall assessment
    passed = []
    failed = []
    
    if price_metrics['mape'] < 20:
        passed.append(f"✓ Price MAPE ({price_metrics['mape']:.1f}%) < 20%")
    else:
        failed.append(f"✗ Price MAPE ({price_metrics['mape']:.1f}%) >= 20%")
    
    if price_metrics['r2'] > 0.5:
        passed.append(f"✓ Price R² ({price_metrics['r2']:.3f}) > 0.5")
    else:
        failed.append(f"✗ Price R² ({price_metrics['r2']:.3f}) <= 0.5")
    
    if 'revpar_strategy' in results:
        if results['revpar_strategy']['positive_lift_pct'] > 50:
            passed.append(f"✓ RevPAR positive rate ({results['revpar_strategy']['positive_lift_pct']:.1f}%) > 50%")
        else:
            failed.append(f"✗ RevPAR positive rate ({results['revpar_strategy']['positive_lift_pct']:.1f}%) <= 50%")
    
    print("\nPassed:")
    for p in passed:
        print(f"  {p}")
    
    if failed:
        print("\nFailed:")
        for f in failed:
            print(f"  {f}")
    
    results['passed'] = len(passed)
    results['failed'] = len(failed)
    
    return results


if __name__ == "__main__":
    results = run_full_evaluation()

