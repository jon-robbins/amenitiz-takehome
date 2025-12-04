"""
Cluster-Based Models for Price Prediction.

Trains separate models for different hotel segments to capture
heterogeneity in price-feature relationships.

Hypothesis: A coastal resort's price depends on sea view,
while an urban hotel depends on city center distance.
One global model can't capture these different relationships.

Clustering Strategy:
- Primary: market_segment (Coastal/Resort, Urban/Madrid, Provincial/Regional)
- Secondary: capacity_quartile (Q1-Q4)
- Results in up to 12 cluster combinations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import re

from lib.db import init_db
from ml_pipeline.features import (
    engineer_all_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES
)
from ml_pipeline.config import RANDOM_STATE


@dataclass
class ClusterModel:
    """Container for a cluster-specific model and its metadata."""
    cluster_id: str
    model: Any
    encoders: Dict[str, LabelEncoder]
    scaler: StandardScaler
    n_samples: int
    train_r2: float
    feature_cols: List[str]


def assign_market_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns market segment based on location and city.
    
    Segments:
    - Coastal/Resort: is_coastal=1 OR city contains beach/coastal keywords
    - Urban/Madrid: city is Madrid
    - Provincial/Regional: everything else
    """
    df = df.copy()
    
    def get_segment(row):
        city = str(row.get('city', '')).lower() if pd.notna(row.get('city')) else ''
        city_std = str(row.get('city_standardized', '')).lower() if pd.notna(row.get('city_standardized')) else ''
        is_coastal = row.get('is_coastal', 0)
        
        # Coastal/Resort
        coastal_keywords = ['beach', 'costa', 'playa', 'mar', 'mediterran']
        if is_coastal == 1 or any(kw in city for kw in coastal_keywords):
            return 'Coastal/Resort'
        
        # Urban/Madrid
        if city_std == 'madrid' or 'madrid' in city:
            return 'Urban/Madrid'
        
        # Provincial/Regional (default)
        return 'Provincial/Regional'
    
    df['market_segment'] = df.apply(get_segment, axis=1)
    return df


def assign_capacity_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns capacity quartile based on total_capacity.
    """
    df = df.copy()
    
    if 'total_capacity' not in df.columns:
        df['capacity_quartile'] = 'Q2'
        return df
    
    try:
        df['capacity_quartile'] = pd.qcut(
            df['total_capacity'].rank(method='first'),
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
    except ValueError:
        # Fallback if quartiles can't be computed
        df['capacity_quartile'] = 'Q2'
    
    return df


def create_cluster_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates cluster ID from market_segment and capacity_quartile.
    
    Format: "{segment}_{quartile}" e.g., "Coastal/Resort_Q3"
    """
    df = df.copy()
    
    if 'market_segment' not in df.columns:
        df = assign_market_segment(df)
    
    if 'capacity_quartile' not in df.columns:
        df = assign_capacity_quartile(df)
    
    df['cluster_id'] = df['market_segment'] + '_' + df['capacity_quartile'].astype(str)
    
    return df


def prepare_cluster_features(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder] = None,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder], StandardScaler, List[str]]:
    """
    Prepares features for cluster model training/inference.
    """
    y = np.log(df['avg_adr'])
    
    if encoders is None:
        encoders = {}
    
    df_encoded = df.copy()
    
    # Encode categoricals
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
    
    # Select features
    feature_cols = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES:
        if col in df_encoded.columns:
            feature_cols.append(col)
    
    X = df_encoded[feature_cols].copy().fillna(0)
    
    # Scale numerics
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    
    if fit:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y, encoders, scaler, feature_cols


def train_cluster_model(
    cluster_df: pd.DataFrame,
    cluster_id: str
) -> Optional[ClusterModel]:
    """
    Trains a model for a single cluster.
    
    Uses Random Forest (best performer from model comparison).
    """
    import sys
    
    if len(cluster_df) < 30:  # Minimum samples for training
        print(f"    ⚠️ Skipping {cluster_id}: only {len(cluster_df)} samples", flush=True)
        return None
    
    print(f"    Training {cluster_id} ({len(cluster_df)} samples)...", flush=True)
    sys.stdout.flush()
    
    # Prepare features
    X, y, encoders, scaler, feature_cols = prepare_cluster_features(
        cluster_df, fit=True
    )
    
    # Train Random Forest with best params - use fewer trees for speed
    model = RandomForestRegressor(
        n_estimators=100,  # Reduced for speed
        max_depth=10,  # Limit depth
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X, y)
    y_pred = model.predict(X)
    train_r2 = r2_score(y, y_pred)
    
    print(f"    ✓ {cluster_id}: Train R² = {train_r2:.4f}", flush=True)
    sys.stdout.flush()
    
    return ClusterModel(
        cluster_id=cluster_id,
        model=model,
        encoders=encoders,
        scaler=scaler,
        n_samples=len(cluster_df),
        train_r2=train_r2,
        feature_cols=feature_cols
    )


def train_all_cluster_models(
    train_df: pd.DataFrame,
    min_cluster_size: int = 30
) -> Dict[str, ClusterModel]:
    """
    Trains models for all clusters.
    
    Also trains a global fallback model for rare clusters.
    """
    import sys
    
    print("\n" + "=" * 80, flush=True)
    print("CLUSTER-BASED MODEL TRAINING", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()
    
    # Create cluster IDs
    print("\nAssigning clusters...", flush=True)
    train_df = create_cluster_id(train_df)
    
    # Show cluster distribution
    print("\n1. Cluster Distribution:", flush=True)
    cluster_counts = train_df['cluster_id'].value_counts()
    for cluster_id, count in cluster_counts.items():
        print(f"   {cluster_id}: {count:,} samples", flush=True)
    sys.stdout.flush()
    
    # Train per-cluster models
    print("\n2. Training cluster models...", flush=True)
    sys.stdout.flush()
    cluster_models = {}
    
    total_clusters = len(cluster_counts)
    for i, cluster_id in enumerate(cluster_counts.index):
        print(f"\n   [{i+1}/{total_clusters}] Processing {cluster_id}...", flush=True)
        sys.stdout.flush()
        cluster_df = train_df[train_df['cluster_id'] == cluster_id]
        model = train_cluster_model(cluster_df, cluster_id)
        if model is not None:
            cluster_models[cluster_id] = model
    
    # Train global fallback
    print("\n3. Training global fallback model...", flush=True)
    sys.stdout.flush()
    fallback = train_cluster_model(train_df, 'GLOBAL_FALLBACK')
    if fallback:
        cluster_models['GLOBAL_FALLBACK'] = fallback
    
    print(f"\n✓ Trained {len(cluster_models)} cluster models", flush=True)
    sys.stdout.flush()
    
    return cluster_models


def evaluate_cluster_models(
    cluster_models: Dict[str, ClusterModel],
    test_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluates cluster models on test set.
    
    For each test sample:
    1. Route to appropriate cluster model
    2. If cluster model doesn't exist, use fallback
    3. Make prediction
    
    Returns per-cluster and overall metrics.
    """
    print("\n" + "=" * 80)
    print("CLUSTER MODEL EVALUATION")
    print("=" * 80)
    
    # Create cluster IDs for test data
    test_df = create_cluster_id(test_df.copy())
    
    predictions = []
    actuals = []
    cluster_assignments = []
    
    for idx, row in test_df.iterrows():
        cluster_id = row['cluster_id']
        
        # Find appropriate model
        if cluster_id in cluster_models:
            cm = cluster_models[cluster_id]
        else:
            cm = cluster_models.get('GLOBAL_FALLBACK')
            if cm is None:
                continue
        
        # Prepare single-row prediction
        row_df = test_df.loc[[idx]].copy()
        try:
            X, _, _, _, _ = prepare_cluster_features(
                row_df, 
                encoders=cm.encoders,
                scaler=cm.scaler,
                fit=False
            )
            pred = cm.model.predict(X)[0]
            predictions.append(pred)
            actuals.append(np.log(row['avg_adr']))
            cluster_assignments.append(cluster_id if cluster_id in cluster_models else 'FALLBACK')
        except Exception:
            continue
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Overall metrics
    overall_r2 = r2_score(actuals, predictions)
    overall_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # Per-cluster metrics
    results_df = pd.DataFrame({
        'cluster': cluster_assignments,
        'actual': actuals,
        'predicted': predictions
    })
    
    per_cluster_r2 = {}
    for cluster in results_df['cluster'].unique():
        cluster_data = results_df[results_df['cluster'] == cluster]
        if len(cluster_data) >= 10:
            r2 = r2_score(cluster_data['actual'], cluster_data['predicted'])
            per_cluster_r2[cluster] = r2
    
    # MAPE
    mape = np.mean(np.abs((np.exp(actuals) - np.exp(predictions)) / np.exp(actuals))) * 100
    
    print(f"\n1. Overall Metrics:")
    print(f"   Test R²: {overall_r2:.4f}")
    print(f"   Test MAPE: {mape:.1f}%")
    print(f"   Test RMSE: {overall_rmse:.4f} (log-scale)")
    
    print(f"\n2. Per-Cluster R²:")
    for cluster, r2 in sorted(per_cluster_r2.items(), key=lambda x: -x[1]):
        n = len(results_df[results_df['cluster'] == cluster])
        print(f"   {cluster}: R² = {r2:.4f} (n={n})")
    
    return {
        'overall_r2': overall_r2,
        'overall_mape': mape,
        'overall_rmse': overall_rmse,
        'per_cluster_r2': per_cluster_r2,
        'n_samples': len(predictions)
    }


def save_cluster_models(
    cluster_models: Dict[str, ClusterModel],
    output_path: str = 'ml_pipeline/models/cluster_models.pkl'
) -> None:
    """Saves cluster models to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(cluster_models, f)
    
    print(f"✓ Cluster models saved to: {output_path}")


def load_cluster_models(
    model_path: str = 'ml_pipeline/models/cluster_models.pkl'
) -> Dict[str, ClusterModel]:
    """Loads cluster models from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_with_cluster_model(
    hotel_data: pd.DataFrame,
    cluster_models: Dict[str, ClusterModel]
) -> float:
    """
    Predicts price using cluster-based models.
    
    Routes to appropriate cluster model based on hotel characteristics.
    """
    # Assign cluster
    hotel_data = create_cluster_id(hotel_data)
    cluster_id = hotel_data['cluster_id'].iloc[0]
    
    # Get model
    if cluster_id in cluster_models:
        cm = cluster_models[cluster_id]
    else:
        cm = cluster_models.get('GLOBAL_FALLBACK')
        if cm is None:
            raise ValueError(f"No model for cluster {cluster_id} and no fallback available")
    
    # Prepare features and predict
    X, _, _, _, _ = prepare_cluster_features(
        hotel_data,
        encoders=cm.encoders,
        scaler=cm.scaler,
        fit=False
    )
    
    log_price = cm.model.predict(X)[0]
    return np.exp(log_price)


def run_cluster_training() -> Tuple[Dict[str, ClusterModel], Dict]:
    """
    Main entry point for cluster model training and evaluation.
    """
    from ml_pipeline.model_comparison import load_data_with_temporal_split
    
    # Load data
    print("\n1. Loading data...")
    train_df, test_df = load_data_with_temporal_split()
    
    # Train cluster models
    cluster_models = train_all_cluster_models(train_df)
    
    # Evaluate
    metrics = evaluate_cluster_models(cluster_models, test_df)
    
    # Save
    save_cluster_models(cluster_models)
    
    return cluster_models, metrics


if __name__ == "__main__":
    cluster_models, metrics = run_cluster_training()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nCluster Models Test R²: {metrics['overall_r2']:.4f}")
    print(f"Cluster Models Test MAPE: {metrics['overall_mape']:.1f}%")

