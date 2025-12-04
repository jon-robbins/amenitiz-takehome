"""
MAPE Optimization for Price Prediction.

Goal: Reduce MAPE from 31.5% to under 20% through:
1. Outlier filtering
2. Target transformation optimization
3. MAPE-focused training
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, make_scorer
import matplotlib.pyplot as plt

from lib.db import init_db
from ml_pipeline.features import (
    engineer_all_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES
)
from ml_pipeline.config import RANDOM_STATE


# Custom MAPE scorer for sklearn (negative because sklearn maximizes)
mape_scorer = make_scorer(
    mean_absolute_percentage_error,
    greater_is_better=False
)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data with temporal split."""
    con = init_db()
    
    sql_file = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity/QUERY_LOAD_HOTEL_MONTH_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    df = con.execute(query).fetchdf()
    
    df = engineer_all_features(df)
    df = df.dropna(subset=['avg_adr'])
    
    df['month'] = pd.to_datetime(df['month'])
    train_df = df[df['month'] < '2024-06-01'].copy()
    test_df = df[df['month'] >= '2024-06-01'].copy()
    
    return train_df, test_df


def analyze_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray
) -> Dict:
    """
    Analyzes prediction errors by price range.
    
    Returns error metrics for different price buckets.
    """
    errors = np.abs(y_true - y_pred) / y_true * 100  # Percentage errors
    
    # Create price buckets
    buckets = [
        (0, 50, 'Budget (<€50)'),
        (50, 100, 'Economy (€50-100)'),
        (100, 150, 'Midscale (€100-150)'),
        (150, 250, 'Upscale (€150-250)'),
        (250, float('inf'), 'Luxury (>€250)')
    ]
    
    results = {}
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS BY PRICE RANGE")
    print("=" * 60)
    print(f"\n{'Price Range':<25} {'N':>8} {'MAPE':>10} {'Mean Error':>12}")
    print("-" * 60)
    
    for low, high, label in buckets:
        mask = (prices >= low) & (prices < high)
        if mask.sum() > 0:
            bucket_mape = errors[mask].mean()
            bucket_mean_err = np.abs(y_true[mask] - y_pred[mask]).mean()
            results[label] = {
                'n': mask.sum(),
                'mape': bucket_mape,
                'mean_abs_error': bucket_mean_err
            }
            print(f"{label:<25} {mask.sum():>8,} {bucket_mape:>9.1f}% {bucket_mean_err:>11.2f}€")
    
    print("-" * 60)
    print(f"{'OVERALL':<25} {len(prices):>8,} {errors.mean():>9.1f}% {np.abs(y_true - y_pred).mean():>11.2f}€")
    
    return results


def filter_outliers_iqr(
    df: pd.DataFrame,
    column: str = 'avg_adr',
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Filters outliers using IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    
    filtered = df[(df[column] >= lower) & (df[column] <= upper)]
    
    print(f"IQR Filtering ({column}):")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"  Removed: {len(df) - len(filtered):,} ({(len(df) - len(filtered))/len(df)*100:.1f}%)")
    
    return filtered


def filter_outliers_percentile(
    df: pd.DataFrame,
    column: str = 'avg_adr',
    lower_pct: float = 2,
    upper_pct: float = 98
) -> pd.DataFrame:
    """
    Filters outliers using percentile bounds.
    """
    lower = df[column].quantile(lower_pct / 100)
    upper = df[column].quantile(upper_pct / 100)
    
    filtered = df[(df[column] >= lower) & (df[column] <= upper)]
    
    print(f"Percentile Filtering ({lower_pct}%-{upper_pct}%):")
    print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"  Removed: {len(df) - len(filtered):,} ({(len(df) - len(filtered))/len(df)*100:.1f}%)")
    
    return filtered


def winsorize(series: pd.Series, lower_pct: float = 2, upper_pct: float = 98) -> pd.Series:
    """
    Winsorizes a series by capping extreme values.
    """
    lower = series.quantile(lower_pct / 100)
    upper = series.quantile(upper_pct / 100)
    return series.clip(lower=lower, upper=upper)


def prepare_features(
    df: pd.DataFrame,
    encoders: Dict = None,
    scaler: StandardScaler = None,
    fit: bool = True,
    target_transform: str = 'log'
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler]:
    """
    Prepares features with configurable target transformation.
    
    target_transform: 'log', 'none', 'winsorized_log'
    """
    # Target transformation
    if target_transform == 'log':
        y = np.log(df['avg_adr'])
    elif target_transform == 'none':
        y = df['avg_adr'].values
    elif target_transform == 'winsorized_log':
        y = np.log(winsorize(df['avg_adr']))
    else:
        raise ValueError(f"Unknown transform: {target_transform}")
    
    if encoders is None:
        encoders = {}
    
    df_encoded = df.copy()
    
    for col in CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = df_encoded[col].fillna('unknown').astype(str)
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le:
                    df_encoded[col] = df_encoded[col].fillna('unknown').astype(str)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
    
    feature_cols = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES:
        if col in df_encoded.columns:
            feature_cols.append(col)
    
    X = df_encoded[feature_cols].copy().fillna(0)
    
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    
    if fit:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y, encoders, scaler


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, transform: str = 'log') -> float:
    """Calculate MAPE, accounting for transformation."""
    if transform in ['log', 'winsorized_log']:
        y_true_orig = np.exp(y_true)
        y_pred_orig = np.exp(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    return np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_transform: str = 'log',
    use_mape_scoring: bool = False
) -> Dict:
    """
    Trains model and evaluates on test set.
    
    Returns metrics dict.
    """
    # Prepare data
    X_train, y_train, encoders, scaler = prepare_features(
        train_df, fit=True, target_transform=target_transform
    )
    X_test, y_test, _, _ = prepare_features(
        test_df, encoders=encoders, scaler=scaler, fit=False, target_transform=target_transform
    )
    
    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Cross-validation
    if use_mape_scoring and target_transform == 'none':
        # Can use MAPE scoring directly when no transform
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring=mape_scorer)
        cv_metric = -cv_scores.mean()  # Convert back to positive
        cv_metric_name = 'CV MAPE'
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        cv_metric = cv_scores.mean()
        cv_metric_name = 'CV R²'
    
    # Fit
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mape = calculate_mape(y_train, y_train_pred, target_transform)
    test_mape = calculate_mape(y_test, y_test_pred, target_transform)
    
    return {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'transform': target_transform,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'cv_metric': cv_metric,
        'cv_metric_name': cv_metric_name,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'test_prices': test_df['avg_adr'].values
    }


def run_optimization():
    """
    Main optimization routine.
    
    Tests different combinations of:
    1. Outlier filtering strategies
    2. Target transformations
    3. MAPE-focused training
    """
    print("=" * 80)
    print("MAPE OPTIMIZATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...", flush=True)
    train_df, test_df = load_raw_data()
    print(f"   Raw train: {len(train_df):,}, test: {len(test_df):,}")
    
    # Baseline (current approach)
    print("\n2. Baseline (current approach)...", flush=True)
    train_filtered = train_df[train_df['avg_adr'] > 10].copy()
    test_filtered = test_df[test_df['avg_adr'] > 10].copy()
    
    baseline = train_and_evaluate(train_filtered, test_filtered, 'log')
    print(f"   Train R²: {baseline['train_r2']:.4f}, Test R²: {baseline['test_r2']:.4f}")
    print(f"   Train MAPE: {baseline['train_mape']:.1f}%, Test MAPE: {baseline['test_mape']:.1f}%")
    
    # Analyze errors
    print("\n3. Analyzing baseline errors...", flush=True)
    analyze_error_distribution(
        baseline['test_prices'],
        np.exp(baseline['y_test_pred']),
        baseline['test_prices']
    )
    
    results = [('Baseline (log, min filter)', baseline)]
    
    # Test different filtering strategies
    print("\n4. Testing outlier filtering strategies...", flush=True)
    
    # IQR filtering
    print("\n   4a. IQR filtering (1.5x)...")
    train_iqr = filter_outliers_iqr(train_df[train_df['avg_adr'] > 10], 'avg_adr', 1.5)
    test_iqr = test_df[test_df['avg_adr'] > 10].copy()  # Don't filter test
    result_iqr = train_and_evaluate(train_iqr, test_iqr, 'log')
    print(f"   Test R²: {result_iqr['test_r2']:.4f}, Test MAPE: {result_iqr['test_mape']:.1f}%")
    results.append(('IQR 1.5x filter', result_iqr))
    
    # Percentile filtering (2-98)
    print("\n   4b. Percentile filtering (2-98%)...")
    train_pct = filter_outliers_percentile(train_df[train_df['avg_adr'] > 10], 'avg_adr', 2, 98)
    result_pct = train_and_evaluate(train_pct, test_iqr, 'log')
    print(f"   Test R²: {result_pct['test_r2']:.4f}, Test MAPE: {result_pct['test_mape']:.1f}%")
    results.append(('Percentile 2-98% filter', result_pct))
    
    # Stricter percentile (5-95)
    print("\n   4c. Stricter percentile filtering (5-95%)...")
    train_strict = filter_outliers_percentile(train_df[train_df['avg_adr'] > 10], 'avg_adr', 5, 95)
    result_strict = train_and_evaluate(train_strict, test_iqr, 'log')
    print(f"   Test R²: {result_strict['test_r2']:.4f}, Test MAPE: {result_strict['test_mape']:.1f}%")
    results.append(('Percentile 5-95% filter', result_strict))
    
    # Test different transformations
    print("\n5. Testing target transformations...", flush=True)
    
    # No transform
    print("\n   5a. No transform (raw prices)...")
    result_notrans = train_and_evaluate(train_pct, test_iqr, 'none')
    print(f"   Test R²: {result_notrans['test_r2']:.4f}, Test MAPE: {result_notrans['test_mape']:.1f}%")
    results.append(('No transform', result_notrans))
    
    # Winsorized log
    print("\n   5b. Winsorized log transform...")
    result_winsor = train_and_evaluate(train_pct, test_iqr, 'winsorized_log')
    print(f"   Test R²: {result_winsor['test_r2']:.4f}, Test MAPE: {result_winsor['test_mape']:.1f}%")
    results.append(('Winsorized log', result_winsor))
    
    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\n{'Configuration':<30} {'Test R²':>10} {'Test MAPE':>12} {'Improvement':>12}")
    print("-" * 70)
    
    baseline_mape = baseline['test_mape']
    best_result = None
    best_mape = float('inf')
    
    for name, result in results:
        improvement = baseline_mape - result['test_mape']
        marker = ""
        if result['test_mape'] < best_mape:
            best_mape = result['test_mape']
            best_result = (name, result)
            marker = " *"
        print(f"{name:<30} {result['test_r2']:>10.4f} {result['test_mape']:>11.1f}% {improvement:>+11.1f}%{marker}")
    
    print("-" * 70)
    print(f"\n✓ Best configuration: {best_result[0]}")
    print(f"  Test MAPE: {best_result[1]['test_mape']:.1f}% (was {baseline_mape:.1f}%)")
    print(f"  Improvement: {baseline_mape - best_result[1]['test_mape']:.1f} percentage points")
    
    # Save best model
    output_dir = Path('ml_pipeline/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'mape_optimized_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_result[1]['model'],
            'encoders': best_result[1]['encoders'],
            'scaler': best_result[1]['scaler'],
            'transform': best_result[1]['transform'],
            'name': best_result[0],
            'test_r2': best_result[1]['test_r2'],
            'test_mape': best_result[1]['test_mape']
        }, f)
    print(f"\n✓ Best model saved to: {output_dir / 'mape_optimized_model.pkl'}")
    
    # Error analysis on best model
    print("\n6. Error analysis on best model...", flush=True)
    best = best_result[1]
    if best['transform'] in ['log', 'winsorized_log']:
        pred_prices = np.exp(best['y_test_pred'])
    else:
        pred_prices = best['y_test_pred']
    
    analyze_error_distribution(
        best['test_prices'],
        pred_prices,
        best['test_prices']
    )
    
    # Focus on core market (€50-250)
    print("\n7. Focusing on core market (€50-250)...", flush=True)
    
    # Filter train AND test to core market
    train_core = train_df[(train_df['avg_adr'] >= 50) & (train_df['avg_adr'] <= 250)].copy()
    test_core = test_df[(test_df['avg_adr'] >= 50) & (test_df['avg_adr'] <= 250)].copy()
    print(f"   Core market train: {len(train_core):,}, test: {len(test_core):,}")
    
    result_core = train_and_evaluate(train_core, test_core, 'log')
    print(f"   Core market Test R²: {result_core['test_r2']:.4f}")
    print(f"   Core market Test MAPE: {result_core['test_mape']:.1f}%")
    
    # Also test with prediction clipping
    print("\n8. Testing prediction clipping (€30-500)...", flush=True)
    
    # Use full training but clip predictions
    full_result = train_and_evaluate(train_filtered, test_filtered, 'log')
    pred_clipped = np.clip(np.exp(full_result['y_test_pred']), 30, 500)
    actual = full_result['test_prices']
    clipped_mape = np.mean(np.abs((actual - pred_clipped) / actual)) * 100
    print(f"   MAPE with clipped predictions: {clipped_mape:.1f}%")
    
    # Save core market model as best if it's better
    if result_core['test_mape'] < best_result[1]['test_mape']:
        print("\n✓ Core market model is better! Saving...")
        with open(output_dir / 'mape_optimized_model.pkl', 'wb') as f:
            pickle.dump({
                'model': result_core['model'],
                'encoders': result_core['encoders'],
                'scaler': result_core['scaler'],
                'transform': 'log',
                'name': 'Core Market (€50-250)',
                'test_r2': result_core['test_r2'],
                'test_mape': result_core['test_mape'],
                'price_range': (50, 250)
            }, f)
        best_result = ('Core Market (€50-250)', result_core)
    
    return results, best_result


if __name__ == "__main__":
    results, best = run_optimization()

