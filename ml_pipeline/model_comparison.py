"""
Model Comparison for Price Prediction.

Compares multiple model architectures to find the best performer:
- XGBoost (current baseline)
- Random Forest
- LightGBM
- CatBoost
- Ridge Regression
- Stacking Ensemble

Uses temporal split for realistic evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import uniform, randint

# Tree-based models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from lib.db import init_db
from ml_pipeline.features import (
    engineer_all_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES
)
from ml_pipeline.config import RANDOM_STATE, CV_FOLDS


@dataclass
class ModelResult:
    """Results from model evaluation."""
    name: str
    train_r2: float
    test_r2: float
    cv_r2_mean: float
    cv_r2_std: float
    train_rmse: float
    test_rmse: float
    train_mape: float
    test_mape: float


def load_data_with_temporal_split(
    test_start: str = '2024-06-01'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data with temporal train/test split.
    
    Args:
        test_start: Date string for test set start (YYYY-MM-DD)
    
    Returns:
        train_df, test_df
    """
    con = init_db()
    
    # Load full dataset
    sql_file = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity/QUERY_LOAD_HOTEL_MONTH_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    df = con.execute(query).fetchdf()
    
    # Engineer features
    df = engineer_all_features(df)
    
    # Filter valid records
    df = df[df['avg_adr'] > 10]
    df = df.dropna(subset=['avg_adr'])
    
    # Temporal split
    df['month'] = pd.to_datetime(df['month'])
    train_df = df[df['month'] < test_start].copy()
    test_df = df[df['month'] >= test_start].copy()
    
    print(f"Train set: {len(train_df):,} records (before {test_start})")
    print(f"Test set: {len(test_df):,} records (from {test_start})")
    
    return train_df, test_df


def prepare_features(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder] = None,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder], StandardScaler]:
    """
    Prepares feature matrix and target.
    
    Args:
        df: Input DataFrame
        encoders: Pre-fitted encoders (for test set)
        scaler: Pre-fitted scaler (for test set)
        fit: Whether to fit encoders/scaler (True for train, False for test)
    
    Returns:
        X, y, encoders, scaler
    """
    # Target: log-transform ADR
    y = np.log(df['avg_adr'])
    
    # Initialize or use provided encoders
    if encoders is None:
        encoders = {}
    
    df_encoded = df.copy()
    
    # Encode categorical features
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
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
    
    # Select feature columns
    feature_cols = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES:
        if col in df_encoded.columns:
            feature_cols.append(col)
    
    X = df_encoded[feature_cols].copy().fillna(0)
    
    # Scale numeric features
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    
    if fit:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y, encoders, scaler


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Convert from log scale
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    mask = y_true_exp > 0
    return np.mean(np.abs((y_true_exp[mask] - y_pred_exp[mask]) / y_true_exp[mask])) * 100


def get_models_and_params() -> Dict[str, Tuple[Any, Dict]]:
    """
    Returns dictionary of models and their hyperparameter search spaces.
    
    Returns:
        Dict mapping model_name -> (model_instance, param_distributions)
    """
    models_params = {}
    
    # Ridge Regression (simple baseline)
    models_params['Ridge'] = (
        Ridge(random_state=RANDOM_STATE),
        {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    )
    
    # Random Forest
    models_params['RandomForest'] = (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    )
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        models_params['XGBoost'] = (
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [0.1, 1.0, 10.0]
            }
        )
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models_params['LightGBM'] = (
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, -1],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_samples': [10, 20, 30],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [0.1, 1.0, 10.0]
            }
        )
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        models_params['CatBoost'] = (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=False),
            {
                'iterations': [200, 300, 500],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5, 10]
            }
        )
    
    return models_params


def tune_model(
    model: Any,
    param_distributions: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_iter: int = 20
) -> Tuple[Any, Dict]:
    """
    Performs hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        model: Base model instance
        param_distributions: Hyperparameter search space
        X_train: Training features
        y_train: Training target
        model_name: Name for logging
        n_iter: Number of random combinations to try
    
    Returns:
        best_model: Tuned model with best params
        best_params: Best hyperparameters found
    """
    print(f"\n  Tuning {model_name} ({n_iter} iterations)...")
    
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=3,  # Use 3-fold for speed during tuning
        scoring='r2',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    print(f"    Best CV R²: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def get_models() -> Dict[str, Any]:
    """
    Returns dictionary of models with default params (for stacking).
    """
    models = {}
    
    models['Ridge'] = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=False
        )
    
    return models


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> ModelResult:
    """
    Evaluates a single model.
    
    Returns ModelResult with all metrics.
    """
    print(f"\n  Training {model_name}...")
    
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2')
    
    # Fit on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    result = ModelResult(
        name=model_name,
        train_r2=r2_score(y_train, y_train_pred),
        test_r2=r2_score(y_test, y_test_pred),
        cv_r2_mean=cv_scores.mean(),
        cv_r2_std=cv_scores.std(),
        train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred)),
        test_rmse=np.sqrt(mean_squared_error(y_test, y_test_pred)),
        train_mape=calculate_mape(y_train.values, y_train_pred),
        test_mape=calculate_mape(y_test.values, y_test_pred)
    )
    
    print(f"    CV R²: {result.cv_r2_mean:.4f} (+/- {result.cv_r2_std*2:.4f})")
    print(f"    Test R²: {result.test_r2:.4f}")
    print(f"    Test MAPE: {result.test_mape:.1f}%")
    
    return result


def create_stacking_ensemble(base_models: Dict[str, Any]) -> StackingRegressor:
    """
    Creates a stacking ensemble from base models.
    """
    estimators = [(name, model) for name, model in base_models.items() 
                  if name != 'Ridge']  # Use Ridge as meta-learner
    
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )


def run_comparison(tune_hyperparams: bool = True, n_tuning_iter: int = 20) -> Tuple[List[ModelResult], str, Any]:
    """
    Runs full model comparison with optional hyperparameter tuning.
    
    Args:
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_tuning_iter: Number of random search iterations per model
    
    Returns:
        results: List of ModelResult for each model
        best_model_name: Name of best performing model
        best_model: The best trained model object
    """
    print("=" * 80)
    print("MODEL COMPARISON WITH HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data with temporal split...")
    train_df, test_df = load_data_with_temporal_split()
    
    # Prepare features
    print("\n2. Preparing features...")
    X_train, y_train, encoders, scaler = prepare_features(train_df, fit=True)
    X_test, y_test, _, _ = prepare_features(test_df, encoders=encoders, scaler=scaler, fit=False)
    
    print(f"   Train features: {X_train.shape}")
    print(f"   Test features: {X_test.shape}")
    
    # Get models and params
    print("\n3. Tuning and evaluating models...")
    models_params = get_models_and_params()
    
    results = []
    trained_models = {}
    best_params_all = {}
    
    for name, (model, param_dist) in models_params.items():
        if tune_hyperparams:
            # Tune hyperparameters
            tuned_model, best_params = tune_model(
                model, param_dist, X_train, y_train, name, n_iter=n_tuning_iter
            )
            best_params_all[name] = best_params
        else:
            tuned_model = model
        
        # Evaluate tuned model
        result = evaluate_model(tuned_model, X_train, y_train, X_test, y_test, name)
        results.append(result)
        trained_models[name] = tuned_model
    
    # Try stacking ensemble if we have multiple tree models
    tree_models = {k: v for k, v in trained_models.items() if k in ['RandomForest', 'XGBoost', 'LightGBM']}
    if len(tree_models) >= 2:
        print("\n  Creating Stacking Ensemble from tuned models...")
        stacking = StackingRegressor(
            estimators=[(name, model) for name, model in tree_models.items()],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1
        )
        result = evaluate_model(stacking, X_train, y_train, X_test, y_test, 'Stacking')
        results.append(result)
        trained_models['Stacking'] = stacking
    
    # Find best model
    best_result = max(results, key=lambda r: r.test_r2)
    best_model_name = best_result.name
    best_model = trained_models[best_model_name]
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (WITH TUNING)")
    print("=" * 80)
    
    print(f"\n{'Model':<15} {'CV R²':>10} {'Test R²':>10} {'Test MAPE':>12} {'Train R²':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: -x.test_r2):
        marker = " ← BEST" if r.name == best_model_name else ""
        print(f"{r.name:<15} {r.cv_r2_mean:>10.4f} {r.test_r2:>10.4f} {r.test_mape:>11.1f}% {r.train_r2:>10.4f}{marker}")
    
    print(f"\n✓ Best model: {best_model_name} (Test R² = {best_result.test_r2:.4f})")
    
    # Print best params for top model
    if best_model_name in best_params_all:
        print(f"\n   Best hyperparameters for {best_model_name}:")
        for k, v in best_params_all[best_model_name].items():
            print(f"     {k}: {v}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'model': r.name,
            'cv_r2_mean': r.cv_r2_mean,
            'cv_r2_std': r.cv_r2_std,
            'test_r2': r.test_r2,
            'test_mape': r.test_mape,
            'train_r2': r.train_r2,
            'train_mape': r.train_mape
        }
        for r in results
    ])
    
    output_dir = Path('ml_pipeline/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'model_comparison_results.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir / 'model_comparison_results.csv'}")
    
    # Save best model
    with open(output_dir / 'best_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'name': best_model_name,
            'encoders': encoders,
            'scaler': scaler,
            'test_r2': best_result.test_r2,
            'test_mape': best_result.test_mape,
            'best_params': best_params_all.get(best_model_name, {})
        }, f)
    print(f"✓ Best model saved to: {output_dir / 'best_model.pkl'}")
    
    return results, best_model_name, best_model


if __name__ == "__main__":
    results, best_name, best_model = run_comparison()

