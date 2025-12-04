"""
Relative Pricing Model.

Instead of predicting absolute price, predict deviation from peer group mean.

Concept:
- Define peer groups (city, room_type, capacity_quartile)
- Calculate peer group mean price
- Model predicts: deviation = (hotel_price - peer_mean) / peer_mean
- Final price = peer_mean × (1 + predicted_deviation)

Rationale:
- Easier to predict "is this hotel 10% above or below peers"
- Removes market-level variation
- More stable predictions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from lib.db import init_db
from ml_pipeline.features import engineer_all_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES


# Peer group definition (from matched pairs blocking)
PEER_GROUP_COLS = ['city_standardized', 'room_type', 'capacity_quartile']


def load_data_with_peer_groups() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data and calculates peer group statistics.
    """
    con = init_db()
    
    sql_file = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity/QUERY_LOAD_HOTEL_MONTH_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    df = con.execute(query).fetchdf()
    
    df = engineer_all_features(df)
    
    # Core market filter
    df = df[(df['avg_adr'] >= 50) & (df['avg_adr'] <= 250)]
    df = df.dropna(subset=['avg_adr'])
    
    # Temporal split
    df['month'] = pd.to_datetime(df['month'])
    train_df = df[df['month'] < '2024-06-01'].copy()
    test_df = df[df['month'] >= '2024-06-01'].copy()
    
    return train_df, test_df


def calculate_peer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates peer group statistics.
    
    Returns DataFrame with peer_mean, peer_median, peer_std for each group.
    """
    df = df.copy()
    
    # Fill missing peer group columns
    for col in PEER_GROUP_COLS:
        if col not in df.columns:
            df[col] = 'unknown'
        else:
            # Convert to string to avoid categorical issues
            df[col] = df[col].astype(str).fillna('unknown')
    
    # Calculate group statistics
    peer_stats = df.groupby(PEER_GROUP_COLS, observed=True).agg({
        'avg_adr': ['mean', 'median', 'std', 'count']
    }).reset_index()
    
    peer_stats.columns = PEER_GROUP_COLS + ['peer_mean', 'peer_median', 'peer_std', 'peer_count']
    
    # Fill NaN std with 0
    peer_stats['peer_std'] = peer_stats['peer_std'].fillna(0)
    
    return peer_stats


def add_peer_features(df: pd.DataFrame, peer_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Adds peer group features to DataFrame.
    """
    df = df.copy()
    
    # Fill missing values in peer group columns
    for col in PEER_GROUP_COLS:
        if col not in df.columns:
            df[col] = 'unknown'
        else:
            df[col] = df[col].astype(str).fillna('unknown')
    
    # Merge peer statistics
    df = df.merge(peer_stats, on=PEER_GROUP_COLS, how='left')
    
    # Calculate deviation from peer mean
    df['price_deviation'] = (df['avg_adr'] - df['peer_mean']) / df['peer_mean']
    
    # For hotels without peer group, use global mean
    global_mean = df['avg_adr'].mean()
    df['peer_mean'] = df['peer_mean'].fillna(global_mean)
    df['peer_median'] = df['peer_median'].fillna(global_mean)
    df['peer_std'] = df['peer_std'].fillna(df['avg_adr'].std())
    df['peer_count'] = df['peer_count'].fillna(1)
    df['price_deviation'] = df['price_deviation'].fillna(0)
    
    return df


def prepare_relative_features(
    df: pd.DataFrame,
    encoders: Dict = None,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler]:
    """
    Prepares features for deviation prediction.
    
    Target is price_deviation, not absolute price.
    """
    # Target: deviation from peer mean
    y = df['price_deviation'].values
    
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
    
    # Add peer features as inputs
    peer_features = ['peer_mean', 'peer_std', 'peer_count']
    
    feature_cols = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES + peer_features:
        if col in df_encoded.columns:
            feature_cols.append(col)
    
    X = df_encoded[feature_cols].copy().fillna(0)
    
    numeric_cols = [c for c in NUMERIC_FEATURES + peer_features if c in X.columns]
    
    if fit:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y, encoders, scaler


def evaluate_relative_pricing() -> Dict:
    """
    Evaluates relative pricing approach.
    """
    print("=" * 80)
    print("RELATIVE PRICING EVALUATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...", flush=True)
    train_df, test_df = load_data_with_peer_groups()
    print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Calculate peer statistics from training data
    print("\n2. Calculating peer group statistics...", flush=True)
    peer_stats = calculate_peer_statistics(train_df)
    print(f"   Found {len(peer_stats)} peer groups")
    
    # Add peer features
    train_df = add_peer_features(train_df, peer_stats)
    test_df = add_peer_features(test_df, peer_stats)
    
    # Show deviation distribution
    print("\n3. Deviation distribution:")
    print(f"   Train deviation mean: {train_df['price_deviation'].mean():.3f}")
    print(f"   Train deviation std: {train_df['price_deviation'].std():.3f}")
    print(f"   Train deviation range: [{train_df['price_deviation'].min():.3f}, {train_df['price_deviation'].max():.3f}]")
    
    # Prepare features
    print("\n4. Training deviation model...", flush=True)
    X_train, y_train, encoders, scaler = prepare_relative_features(train_df, fit=True)
    X_test, y_test, _, _ = prepare_relative_features(test_df, encoders=encoders, scaler=scaler, fit=False)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predict deviations
    y_pred_deviation = model.predict(X_test)
    
    # Convert back to absolute prices
    peer_means = test_df['peer_mean'].values
    predicted_prices = peer_means * (1 + y_pred_deviation)
    actual_prices = test_df['avg_adr'].values
    
    # Clip predictions to reasonable range
    predicted_prices = np.clip(predicted_prices, 50, 250)
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    
    # R² on absolute prices
    r2 = r2_score(actual_prices, predicted_prices)
    
    # R² on deviation prediction
    deviation_r2 = r2_score(y_test, y_pred_deviation)
    
    print("\n5. Results:")
    print(f"   Deviation R²: {deviation_r2:.4f}")
    print(f"   Absolute Price R²: {r2:.4f}")
    print(f"   MAPE: {mape:.1f}%")
    
    # Compare to baseline (just using peer mean)
    baseline_mape = np.mean(np.abs((actual_prices - peer_means) / actual_prices)) * 100
    print(f"\n   Baseline (peer mean only): {baseline_mape:.1f}%")
    print(f"   Improvement: {baseline_mape - mape:.1f} pp")
    
    return {
        'mape': mape,
        'r2': r2,
        'deviation_r2': deviation_r2,
        'baseline_mape': baseline_mape,
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'peer_stats': peer_stats
    }


if __name__ == "__main__":
    results = evaluate_relative_pricing()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Relative Pricing MAPE: {results['mape']:.1f}%")
    print(f"Baseline (Peer Mean): {results['baseline_mape']:.1f}%")

