"""
Model Selection for Baseline Pricing.

Compares multiple ML algorithms using cross-validation to find
the best model for predicting hotel baseline prices.

Uses XGBoost-validated features from src/features/engineering.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# =============================================================================
# VALIDATED FEATURE SET
# =============================================================================

# Features validated by XGBoost (R² = 0.71) in elasticity analysis
FEATURE_COLS = [
    # Geographic (validated)
    'dist_center_km',
    'is_madrid_metro', 
    'dist_coast_log',
    'is_coastal',
    
    # Product (validated)
    'log_room_size',
    'amenities_score',
    'view_quality_ordinal',
    'total_rooms',
    
    # Peer context (10km radius)
    'peer_price_mean',
    'peer_price_median',
    'peer_price_p25',
    'peer_price_p75',
    'peer_price_distance_weighted',
    'n_peers_10km',
    
    # Temporal
    'week_of_year',
    'month',
    'is_summer',
    'is_winter',
]


@dataclass
class ModelResult:
    """Result from model evaluation."""
    name: str
    mae_mean: float
    mae_std: float
    mape_mean: float
    mape_std: float
    r2_mean: float
    r2_std: float
    
    def __repr__(self) -> str:
        return (
            f"{self.name:25s} "
            f"MAE: €{self.mae_mean:6.2f} ± {self.mae_std:5.2f}  "
            f"MAPE: {self.mape_mean:5.1f}% ± {self.mape_std:4.1f}%  "
            f"R²: {self.r2_mean:.3f} ± {self.r2_std:.3f}"
        )


def get_candidate_models() -> Dict[str, object]:
    """
    Get dictionary of candidate models to evaluate.
    
    Returns:
        Dict mapping model name to sklearn-compatible estimator.
    """
    models = {
        # Linear models
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=1.0)': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        
        # Tree-based models
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        ),
    }
    
    # Optional: XGBoost
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=20,
            random_state=42,
            verbosity=0
        )
    
    # Optional: LightGBM
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=20,
            random_state=42,
            verbosity=-1
        )
    
    return models


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare feature matrix from DataFrame.
    
    Args:
        df: DataFrame with feature columns
        feature_cols: List of feature column names to use
    
    Returns:
        Tuple of (feature matrix, list of available feature names)
    """
    # Get available features
    available = [c for c in feature_cols if c in df.columns]
    
    if len(available) == 0:
        raise ValueError(f"No features found. Expected: {feature_cols}")
    
    # Extract and clean
    X = df[available].copy()
    
    # Handle non-numeric columns
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values
    X = X.fillna(0)
    
    return X.values, available


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate a model using cross-validation.
    
    Args:
        model: sklearn-compatible estimator
        X: Feature matrix
        y: Target values
        cv_folds: Number of CV folds
        random_state: Random seed
    
    Returns:
        Tuple of (mae_mean, mae_std, mape_mean, mape_std, r2_mean, r2_std)
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    mae_scores = []
    mape_scores = []
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Fit and predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_val - y_pred))
        mape = calculate_mape(y_val, y_pred)
        
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
    
    return (
        np.mean(mae_scores), np.std(mae_scores),
        np.mean(mape_scores), np.std(mape_scores),
        np.mean(r2_scores), np.std(r2_scores)
    )


def run_model_selection(
    train_df: pd.DataFrame,
    target_col: str = 'actual_price',
    feature_cols: List[str] = FEATURE_COLS,
    cv_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run cross-validated model comparison.
    
    Args:
        train_df: Training DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns to use
        cv_folds: Number of CV folds
        random_state: Random seed
    
    Returns:
        DataFrame with model comparison results, sorted by R²
    """
    print("=" * 70)
    print("MODEL SELECTION FOR BASELINE PRICING")
    print("=" * 70)
    
    # Filter valid rows
    df = train_df[train_df[target_col].notna() & (train_df[target_col] > 0)].copy()
    print(f"\nSamples: {len(df):,}")
    print(f"Target: {target_col}")
    print(f"CV Folds: {cv_folds}")
    
    # Prepare features
    X, available_features = prepare_features(df, feature_cols)
    y = df[target_col].values
    
    print(f"\nFeatures used ({len(available_features)}):")
    for i, f in enumerate(available_features):
        print(f"  {i+1:2d}. {f}")
    
    # Get models
    models = get_candidate_models()
    print(f"\nEvaluating {len(models)} models...")
    print("-" * 70)
    
    results = []
    
    for name, model in models.items():
        try:
            mae_mean, mae_std, mape_mean, mape_std, r2_mean, r2_std = evaluate_model(
                model, X, y, cv_folds, random_state
            )
            
            result = ModelResult(
                name=name,
                mae_mean=mae_mean,
                mae_std=mae_std,
                mape_mean=mape_mean,
                mape_std=mape_std,
                r2_mean=r2_mean,
                r2_std=r2_std
            )
            results.append(result)
            print(result)
            
        except Exception as e:
            print(f"{name:25s} FAILED: {e}")
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame([
        {
            'model': r.name,
            'mae_mean': r.mae_mean,
            'mae_std': r.mae_std,
            'mape_mean': r.mape_mean,
            'mape_std': r.mape_std,
            'r2_mean': r.r2_mean,
            'r2_std': r.r2_std,
        }
        for r in results
    ]).sort_values('r2_mean', ascending=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS (sorted by R²)")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Model':<25} {'MAE (€)':<12} {'MAPE (%)':<12} {'R²':<12}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(
            f"{i:<5} {row['model']:<25} "
            f"€{row['mae_mean']:>6.2f}      "
            f"{row['mape_mean']:>5.1f}%       "
            f"{row['r2_mean']:.3f}"
        )
    
    # Best model
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best['model']}")
    print(f"  MAE:  €{best['mae_mean']:.2f} ± {best['mae_std']:.2f}")
    print(f"  MAPE: {best['mape_mean']:.1f}% ± {best['mape_std']:.1f}%")
    print(f"  R²:   {best['r2_mean']:.3f} ± {best['r2_std']:.3f}")
    print("=" * 70)
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    from pathlib import Path
    
    from lib.db import init_db
    from lib.data_validator import CleaningConfig, DataCleaner
    from src.features.engineering import (
        engineer_validated_features,
        add_peer_price_features,
        PEER_RADIUS_KM,
    )
    from src.models.evaluation.comprehensive_cold_start import (
        get_hotel_ids_from_bookings,
        load_all_hotel_weeks,
    )
    
    print("Loading data...")
    
    # Get clean connection - use default config which handles most cleaning
    config = CleaningConfig(
        remove_negative_prices=True,
        remove_zero_prices=True,
        remove_low_prices=True,
        remove_null_prices=True,
    )
    cleaner = DataCleaner(config)
    con = cleaner.clean(init_db())
    
    # Split hotels from bookings table (60% train)
    all_hotel_ids = get_hotel_ids_from_bookings(con)
    n = len(all_hotel_ids)
    np.random.seed(42)
    shuffled = np.random.permutation(all_hotel_ids)
    train_hotel_ids = set(shuffled[:int(n * 0.6)])
    
    print(f"Train hotels: {len(train_hotel_ids):,} / {n:,}")
    
    # Load train data
    train_df = load_all_hotel_weeks(
        con, 
        min_price=50, 
        max_price=200,
        hotel_ids=train_hotel_ids
    )
    print(f"Train observations: {len(train_df):,}")
    
    # Engineer features
    print("Engineering features...")
    train_df = engineer_validated_features(train_df)
    train_df = add_peer_price_features(train_df, peer_df=train_df, radius_km=PEER_RADIUS_KM)
    
    # Run model selection
    results = run_model_selection(
        train_df,
        target_col='actual_price',
        cv_folds=5,
        random_state=42
    )
    
    # Save results
    output_dir = Path('outputs/model_selection')
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / 'baseline_pricing_comparison.csv', index=False)
    print(f"\n✓ Saved to {output_dir / 'baseline_pricing_comparison.csv'}")

