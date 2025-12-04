"""
Training pipeline for price recommendation model.

Trains XGBoost model to predict market price (peer price) for hotels.
The recommended price is then: market_price × (1 + optimal_deviation).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from lib.db import init_db
from lib.sql_loader import load_sql_file
from ml_pipeline.features import (
    engineer_all_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    validate_features
)
from ml_pipeline.config import (
    XGBOOST_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
    MIN_R2_THRESHOLD,
    MODEL_DIR,
    MODEL_FILENAME,
    SCALER_FILENAME,
    ENCODER_FILENAME
)


def load_training_data() -> pd.DataFrame:
    """
    Loads hotel-month aggregated data for training.
    
    Returns DataFrame with hotel_id, month, avg_adr (target), and features.
    """
    con = init_db()
    
    # Load distance features
    distance_query = """
    SELECT hotel_id, distance_from_coast, distance_from_madrid
    FROM hotel_distance_features
    """
    try:
        distance_features = con.execute(distance_query).fetchdf()
    except Exception:
        # If table doesn't exist, create placeholder
        print("Warning: hotel_distance_features table not found. Using placeholder.")
        distance_features = None
    
    # Load main hotel-month data
    # The SQL file is in the elasticity folder - we need to pass a file path, not directory
    sql_dir = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity'
    sql_file = sql_dir / 'QUERY_LOAD_HOTEL_MONTH_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    
    df = con.execute(query).fetchdf()
    print(f"Loaded {len(df):,} hotel-month records")
    
    # Engineer features
    df = engineer_all_features(df, distance_features)
    
    return df


def prepare_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder], StandardScaler]:
    """
    Prepares feature matrix and target for training.
    
    Returns:
        X: Feature matrix
        y: Target (log-transformed ADR)
        encoders: Dict of LabelEncoders for categorical features
        scaler: Fitted StandardScaler for numeric features
    """
    # Target: log-transform ADR for better distribution
    y = np.log(df['avg_adr'])
    
    # Encode categorical features
    encoders = {}
    df_encoded = df.copy()
    
    for col in CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Handle unseen categories by fitting on all unique values
            df_encoded[col] = df_encoded[col].fillna('unknown')
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    # Select feature columns
    feature_cols = []
    
    # Numeric features
    for col in NUMERIC_FEATURES:
        if col in df_encoded.columns:
            feature_cols.append(col)
    
    # Categorical features (now encoded)
    for col in CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            feature_cols.append(col)
    
    # Boolean features
    for col in BOOLEAN_FEATURES:
        if col in df_encoded.columns:
            feature_cols.append(col)
    
    X = df_encoded[feature_cols].copy()
    
    # Fill missing values
    X = X.fillna(0)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features used: {feature_cols}")
    
    return X, y, encoders, scaler


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict] = None
) -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    Trains XGBoost model with cross-validation.
    
    Returns:
        model: Trained XGBRegressor
        metrics: Dict with R², RMSE, MAE
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is required for training. Install with: pip install xgboost")
    
    params = params or XGBOOST_PARAMS
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Cross-validation on training set
    model = XGBRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2')
    
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Fit final model on full training set
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }
    
    print(f"\nTest set metrics:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f} (log-scale)")
    print(f"  MAE: {metrics['mae']:.4f} (log-scale)")
    
    # Convert back to price scale for interpretability
    rmse_price = np.exp(metrics['rmse']) - 1
    mae_price = np.exp(metrics['mae']) - 1
    print(f"  RMSE: ~{rmse_price*100:.1f}% (price scale)")
    print(f"  MAE: ~{mae_price*100:.1f}% (price scale)")
    
    # Check threshold
    if metrics['r2'] < MIN_R2_THRESHOLD:
        print(f"\n⚠️ Warning: R² ({metrics['r2']:.4f}) below threshold ({MIN_R2_THRESHOLD})")
    else:
        print(f"\n✓ Model meets R² threshold ({metrics['r2']:.4f} >= {MIN_R2_THRESHOLD})")
    
    return model, metrics


def save_model(
    model: XGBRegressor,
    encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    metrics: Dict[str, float]
) -> None:
    """
    Saves trained model and preprocessing artifacts.
    """
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / MODEL_FILENAME
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = model_dir / SCALER_FILENAME
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save encoders
    encoder_path = model_dir / ENCODER_FILENAME
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"✓ Encoders saved to: {encoder_path}")
    
    # Save metrics
    metrics_path = model_dir / 'metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Metrics saved to: {metrics_path}")


def main() -> None:
    """
    Main training pipeline.
    """
    print("=" * 80)
    print("PRICE RECOMMENDATION MODEL - TRAINING PIPELINE")
    print("=" * 80)
    
    print("\n1. Loading training data...")
    df = load_training_data()
    
    # Filter valid records
    df = df[df['avg_adr'] > 10]  # Remove bad data
    df = df.dropna(subset=['avg_adr', 'occupancy_rate'])
    print(f"   Valid records after filtering: {len(df):,}")
    
    print("\n2. Preparing features...")
    X, y, encoders, scaler = prepare_features(df)
    
    print("\n3. Training model...")
    model, metrics = train_model(X, y)
    
    print("\n4. Saving model artifacts...")
    save_model(model, encoders, scaler, metrics)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel Performance:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  CV R²: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']*2:.4f})")
    
    return model, metrics


if __name__ == "__main__":
    main()

