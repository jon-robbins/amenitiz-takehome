"""
Baseline Pricing Model - Training Pipeline.

Single entry point for:
1. Data loading with train/test split (by hotel_id from bookings)
2. Model selection (8 algorithms, 5-fold CV)
3. Training best model
4. Evaluation on held-out test set
5. Diagnostic plots (residuals, predicted vs actual, etc.)

Usage:
    python src/models/train_baseline.py
    
    # Or with custom params
    python src/models/train_baseline.py --train-size 0.7 --output-dir outputs/baseline
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
from src.features.engineering import (
    engineer_validated_features,
    add_peer_price_features,
    PEER_RADIUS_KM,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Validated feature set from XGBoost analysis (R² = 0.71)
# Includes validated features + useful non-validated features
FEATURE_COLS = [
    # Geographic (validated)
    'dist_center_km', 'is_madrid_metro', 'dist_coast_log', 'is_coastal',
    # Product (validated + useful)
    'log_room_size', 'room_capacity_pax', 'amenities_score', 'view_quality_ordinal',
    'total_rooms', 'total_capacity_log',  # total_capacity_log validated, total_rooms useful
    'children_allowed',  # Validated boolean feature
    # Peer context (10km) - for cold-start (not validated but necessary)
    'peer_price_mean', 'peer_price_median', 'peer_price_p25', 'peer_price_p75',
    'peer_price_distance_weighted', 'n_peers_10km',
    # Temporal (validated + useful)
    'month_sin', 'month_cos', 'is_summer', 'is_winter', 'is_july_august',
    'week_of_year', 'month',  # Not validated but useful
    # Note: weekend_ratio and holiday_ratio require booking-level data (not available in weekly aggregates)
    # Categorical (validated)
    'city_standardized',  # Validated categorical
]


@dataclass
class TrainingResults:
    """Results from training pipeline."""
    # Data info
    n_train_hotels: int
    n_val_hotels: int
    n_test_hotels: int
    n_train_samples: int
    n_val_samples: int
    n_test_samples: int
    
    # Model selection
    model_comparison: pd.DataFrame
    best_model_name: str
    
    # Final model performance (on held-out test set)
    test_mae: float
    test_mape: float
    test_rmse: float
    test_r2: float
    
    # Predictions for diagnostics
    y_test: np.ndarray
    y_pred: np.ndarray
    test_df: pd.DataFrame
    
    # Data splits (for downstream use)
    train_df: pd.DataFrame
    val_df: pd.DataFrame


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    min_price: float = 50,
    max_price: float = 200,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set, set, set]:
    """
    Load data with hotel-level train/val/test split.
    
    CRITICAL: Split by hotel_id from bookings table, not by observation.
    This prevents data leakage in cold-start scenarios.
    
    Split:
    - Train (60%): Model selection CV + training
    - Val (20%): Hyperparameter tuning / occupancy model
    - Test (20%): Held out for final pipeline evaluation
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    assert abs(train_size + val_size + test_size - 1.0) < 0.01, "Sizes must sum to 1.0"
    
    # Get clean connection
    config = CleaningConfig(
        remove_negative_prices=True,
        remove_zero_prices=True,
        remove_low_prices=True,
    )
    cleaner = DataCleaner(config)
    con = cleaner.clean(init_db())
    
    # Get hotel_ids with locations from bookings table
    hotels_df = con.execute("""
        SELECT DISTINCT b.hotel_id, hl.latitude, hl.longitude, hl.city
        FROM bookings b
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('Booked', 'confirmed')
          AND b.hotel_id IS NOT NULL
          AND hl.latitude IS NOT NULL
    """).fetchdf()
    
    n_hotels = len(hotels_df)
    print(f"Total hotels with bookings & location: {n_hotels:,}")
    
    # Random hotel split (clean, deterministic)
    np.random.seed(random_state)
    
    hotel_ids = hotels_df['hotel_id'].values
    # We permute hotel_ids here to ensure a random, reproducible assignment of hotels
    # to train, validation, and test splits. This avoids accidental ordering biases 
    # and guarantees that each split is mutually exclusive and randomized each time with the seed.
    shuffled = np.random.permutation(hotel_ids)
    
    n_train = int(n_hotels * train_size)
    n_val = int(n_hotels * val_size)
    
    train_hotel_ids = set(shuffled[:n_train])
    val_hotel_ids = set(shuffled[n_train:n_train + n_val])
    test_hotel_ids = set(shuffled[n_train + n_val:])
    
    print(f"Train hotels: {len(train_hotel_ids):,} ({len(train_hotel_ids)/n_hotels:.0%})")
    print(f"Val hotels:   {len(val_hotel_ids):,} ({len(val_hotel_ids)/n_hotels:.0%})")
    print(f"Test hotels:  {len(test_hotel_ids):,} ({len(test_hotel_ids)/n_hotels:.0%})")
    print(f"Split strategy: Random hotel split (deterministic with seed={random_state})")
    
    # Load weekly data
    def load_hotel_weeks(hotel_ids: set) -> pd.DataFrame:
        hotel_ids_str = ",".join(str(h) for h in hotel_ids)
        query = f"""
        SELECT 
            b.hotel_id,
            DATE_TRUNC('week', b.arrival_date) as week_start,
            EXTRACT(WEEK FROM b.arrival_date) as week_of_year,
            EXTRACT(MONTH FROM b.arrival_date) as month,
            COUNT(*) as n_bookings,
            AVG(br.total_price / NULLIF(DATEDIFF('day', b.arrival_date, b.departure_date), 0)) as actual_price,
            AVG(br.room_size) as avg_room_size,
            MODE() WITHIN GROUP (ORDER BY br.room_type) as room_type,
            MAX(br.room_view) as room_view
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        WHERE b.status IN ('Booked', 'confirmed')
          AND b.arrival_date >= '2023-06-01'
          AND b.arrival_date < '2024-10-01'
          AND b.hotel_id IN ({hotel_ids_str})
        GROUP BY b.hotel_id, DATE_TRUNC('week', b.arrival_date),
                 EXTRACT(WEEK FROM b.arrival_date), EXTRACT(MONTH FROM b.arrival_date)
        HAVING COUNT(*) >= 3
          AND AVG(br.total_price / NULLIF(DATEDIFF('day', b.arrival_date, b.departure_date), 0)) BETWEEN {min_price} AND {max_price}
        """
        df = con.execute(query).fetchdf()
        
        # Add location and capacity
        location = con.execute("SELECT hotel_id, city, latitude, longitude FROM hotel_location").fetchdf()
        # Get actual hotel capacity from rooms table (sum of all room types per hotel)
        capacity = con.execute("""
            WITH hotel_room_types AS (
                -- Get unique room_ids per hotel
                SELECT DISTINCT
                    b.hotel_id,
                    CAST(br.room_id AS BIGINT) as room_id
                FROM bookings b
                JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
                WHERE b.status IN ('Booked', 'confirmed')
            ),
            hotel_capacity AS (
                -- Sum the number_of_rooms for each unique room type
                SELECT 
                    hrt.hotel_id,
                    SUM(COALESCE(r.number_of_rooms, 1)) as total_rooms
                FROM hotel_room_types hrt
                LEFT JOIN rooms r ON hrt.room_id = r.id
                GROUP BY hrt.hotel_id
            ),
            room_features AS (
                SELECT 
                    b.hotel_id,
                    MAX(r.max_occupancy) as room_capacity_pax,
                    MAX(CASE WHEN r.children_allowed THEN 1 ELSE 0 END) as children_allowed,
                    MAX(CASE WHEN r.pets_allowed THEN 1 ELSE 0 END) as pets_allowed,
                    MAX(CASE WHEN r.events_allowed THEN 1 ELSE 0 END) as events_allowed
                FROM bookings b
                JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
                LEFT JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
                GROUP BY b.hotel_id
            )
            SELECT 
                hc.hotel_id,
                COALESCE(hc.total_rooms, 10) as total_rooms,
                rf.room_capacity_pax,
                rf.children_allowed,
                rf.pets_allowed,
                rf.events_allowed
            FROM hotel_capacity hc
            LEFT JOIN room_features rf ON hc.hotel_id = rf.hotel_id
        """).fetchdf()
        
        df = df.merge(location, on='hotel_id', how='inner')
        df = df.merge(capacity, on='hotel_id', how='left')
        df['total_rooms'] = df['total_rooms'].fillna(10)
        df['room_capacity_pax'] = df['room_capacity_pax'].fillna(2)  # Default 2 pax
        
        # Load pre-calculated distance features (coast, madrid)
        distance_path = Path(__file__).parent.parent.parent / 'outputs' / 'data' / 'hotel_distance_features.csv'
        if distance_path.exists():
            distance_df = pd.read_csv(distance_path)
            df = df.merge(distance_df, on='hotel_id', how='left')
            df['distance_from_coast'] = df['distance_from_coast'].fillna(100)  # Default inland
        
        return df
    
    train_df = load_hotel_weeks(train_hotel_ids)
    val_df = load_hotel_weeks(val_hotel_ids)
    test_df = load_hotel_weeks(test_hotel_ids)
    
    print(f"\nTrain samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Test samples:  {len(test_df):,}")
    
    # Engineer features
    print("\nEngineering features...")
    train_df = engineer_validated_features(train_df)
    val_df = engineer_validated_features(val_df)
    test_df = engineer_validated_features(test_df)
    
    # Add peer features with tiered fallback radius
    # 10km -> 25km -> 50km -> 100km -> market median
    print("Adding peer features (from train hotels only, tiered radius)...")
    
    RADIUS_TIERS = [10, 25, 50, 100]
    
    def add_peer_features_tiered(df, peer_df, radius_tiers):
        """Add peer features with progressively wider radius for hotels without peers."""
        df = df.copy()
        
        # Start with smallest radius
        df = add_peer_price_features(df, peer_df=peer_df, radius_km=radius_tiers[0])
        
        # For hotels without peers, try wider radius
        for radius in radius_tiers[1:]:
            missing_mask = df['peer_price_mean'].isna()
            if missing_mask.sum() == 0:
                break
            
            # Get features at wider radius for missing hotels
            missing_df = df[missing_mask].copy()
            missing_with_peers = add_peer_price_features(
                missing_df.drop(columns=[c for c in missing_df.columns if c.startswith('peer_') or c == 'n_peers_10km'], errors='ignore'),
                peer_df=peer_df, 
                radius_km=radius
            )
            
            # Update missing rows
            peer_cols = [c for c in missing_with_peers.columns if c.startswith('peer_') or c == 'n_peers_10km']
            for col in peer_cols:
                if col in df.columns:
                    df.loc[missing_mask, col] = missing_with_peers[col].values
        
        return df
    
    train_df = add_peer_features_tiered(train_df, train_df, RADIUS_TIERS)
    val_df = add_peer_features_tiered(val_df, train_df, RADIUS_TIERS)
    test_df = add_peer_features_tiered(test_df, train_df, RADIUS_TIERS)
    
    # Final fallback: fill any remaining with train market median
    peer_cols = ['peer_price_mean', 'peer_price_median', 'peer_price_p25', 'peer_price_p75',
                 'peer_price_distance_weighted']
    train_peer_stats = {col: train_df[col].median() for col in peer_cols if col in train_df.columns}
    
    for df in [val_df, test_df]:
        for col, default_val in train_peer_stats.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
        if 'n_peers_10km' in df.columns:
            df['n_peers_10km'] = df['n_peers_10km'].fillna(0)
    
    # Report coverage
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        coverage = df['peer_price_mean'].notna().mean() * 100
        print(f"  {name}: {coverage:.0f}% peer coverage")
    
    return train_df, val_df, test_df, train_hotel_ids, val_hotel_ids, test_hotel_ids


# =============================================================================
# MODEL SELECTION
# =============================================================================

def run_model_selection(
    train_df: pd.DataFrame,
    target_col: str = 'actual_price',
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Run cross-validated model comparison with TRUE cold-start simulation.
    
    CRITICAL: For each fold:
    1. Split hotels into train/val
    2. Recalculate peer features for val hotels using ONLY train hotels
    3. Train model on train data
    4. Evaluate on val data
    
    This properly simulates cold-start where new hotel has no data in system.
    """
    from sklearn.model_selection import GroupKFold
    
    print("\n" + "=" * 60)
    print("MODEL SELECTION (True Cold-Start CV)")
    print("=" * 60)
    print("Recalculating peer features per fold (no leakage)")
    
    # Prepare base features (non-peer features)
    df = train_df[train_df[target_col].notna() & (train_df[target_col] > 0)].copy()
    
    # Non-peer feature columns
    non_peer_features = [c for c in FEATURE_COLS if c in df.columns and not c.startswith('peer_') and c != 'n_peers_10km']
    peer_features = [c for c in FEATURE_COLS if c in df.columns and (c.startswith('peer_') or c == 'n_peers_10km')]
    
    n_hotels = df['hotel_id'].nunique()
    print(f"Samples: {len(df):,}")
    print(f"Hotels: {n_hotels:,}")
    print(f"Non-peer features: {len(non_peer_features)}")
    print(f"Peer features: {len(peer_features)} (recalculated per fold)")
    print(f"CV: {cv_folds}-fold GroupKFold")
    
    # Candidate models (fewer for speed since this is expensive)
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, verbosity=-1) if HAS_LIGHTGBM else None,
        'Peer Price Only': None,
    }
    models = {k: v for k, v in models.items() if v is not None or k == 'Peer Price Only'}
    
    results = []
    gkf = GroupKFold(n_splits=cv_folds)
    groups = df['hotel_id'].values
    
    print(f"\n{'Model':<25} {'MAE (€)':<12} {'MAPE (%)':<12} {'R²':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        mae_scores, mape_scores, r2_scores = [], [], []
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, df[target_col], groups)):
            # Get train and val DataFrames
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            
            # Recalculate peer features for val using ONLY train hotels
            # This is the key step that prevents leakage
            val_fold_clean = val_fold.drop(columns=peer_features, errors='ignore')
            val_fold_with_peers = add_peer_price_features(
                val_fold_clean, 
                peer_df=train_fold,  # Only use train hotels as peers!
                radius_km=PEER_RADIUS_KM
            )
            
            # Fill missing peer features with train market median
            for col in peer_features:
                if col in val_fold_with_peers.columns and col in train_fold.columns:
                    val_fold_with_peers[col] = val_fold_with_peers[col].fillna(train_fold[col].median())
            
            # Prepare feature matrices
            all_features = non_peer_features + peer_features
            available = [c for c in all_features if c in train_fold.columns and c in val_fold_with_peers.columns]
            
            def prep_X(data):
                X = data[available].fillna(0).copy()
                for col in X.columns:
                    if not np.issubdtype(X[col].dtype, np.number):
                        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                return X.values
            
            X_train = prep_X(train_fold)
            X_val = prep_X(val_fold_with_peers)
            y_train = train_fold[target_col].values
            y_val = val_fold[target_col].values
            
            if name == 'Peer Price Only':
                # Just use peer_price_mean as prediction
                if 'peer_price_mean' in val_fold_with_peers.columns:
                    y_pred = val_fold_with_peers['peer_price_mean'].fillna(y_train.mean()).values
                else:
                    y_pred = np.full_like(y_val, y_train.mean())
            else:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_val_s)
            
            # Metrics
            mae_scores.append(np.mean(np.abs(y_val - y_pred)))
            mask = y_val > 0
            mape_scores.append(np.mean(np.abs(y_val[mask] - y_pred[mask]) / y_val[mask]) * 100)
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        
        results.append({
            'model': name,
            'mae': np.mean(mae_scores),
            'mape': np.mean(mape_scores),
            'r2': np.mean(r2_scores),
        })
        print(f"{name:<25} €{np.mean(mae_scores):>6.2f}      {np.mean(mape_scores):>5.1f}%       {np.mean(r2_scores):.3f}")
    
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    
    best = results_df.iloc[0]
    print(f"\n✓ Best model: {best['model']} (R² = {best['r2']:.3f}, MAPE = {best['mape']:.1f}%)")
    
    return results_df


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = 'Random Forest'
) -> Tuple[object, Dict]:
    """Train best model and evaluate on test set."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    
    target_col = 'actual_price'
    
    # Prepare data
    train_clean = train_df[train_df[target_col].notna() & (train_df[target_col] > 0)].copy()
    test_clean = test_df[test_df[target_col].notna() & (test_df[target_col] > 0)].copy()
    
    available = [c for c in FEATURE_COLS if c in train_clean.columns]
    
    def prep(df):
        X = df[available].fillna(0).copy()
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        return X.values
    
    X_train = prep(train_clean)
    y_train = train_clean[target_col].values
    X_test = prep(test_clean)
    y_test = test_clean[target_col].values
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    if model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    elif model_name == 'XGBoost' and HAS_XGBOOST:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    elif model_name == 'LightGBM' and HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42)
    
    print(f"Training on {len(X_train):,} samples...")
    model.fit(X_train_s, y_train)
    
    # Evaluate on test set
    print(f"Evaluating on {len(X_test):,} test samples...")
    y_pred = model.predict(X_test_s)
    
    mae = np.mean(np.abs(y_test - y_pred))
    mask = y_test > 0
    mape = np.mean(np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]) * 100
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    metrics = {'mae': mae, 'mape': mape, 'rmse': rmse, 'r2': r2}
    
    print(f"\nTEST SET PERFORMANCE:")
    print(f"  MAE:  €{mae:.2f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  RMSE: €{rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    return model, metrics, y_test, y_pred, test_clean


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def plot_diagnostics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    metrics: Dict,
    output_dir: Path
) -> None:
    """Create diagnostic plots."""
    print("\n" + "=" * 60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    residuals = y_test - y_pred
    pct_errors = (y_pred - y_test) / y_test * 100
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Baseline Pricing Model Diagnostics\nTest Set: R² = {metrics["r2"]:.3f}, MAPE = {metrics["mape"]:.1f}%', 
                 fontsize=14, fontweight='bold')
    
    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, c='steelblue')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Price (€)')
    ax.set_ylabel('Predicted Price (€)')
    ax.set_title(f'Predicted vs Actual (r = {np.corrcoef(y_test, y_pred)[0,1]:.3f})')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # 2. Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=15, c='coral')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Price (€)')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title('Residuals vs Predicted')
    
    # 3. Residual Distribution
    ax = axes[0, 2]
    ax.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.axvline(residuals.mean(), color='green', linestyle='--', lw=2, label=f'Mean: €{residuals.mean():.1f}')
    ax.set_xlabel('Residual (€)')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution (std: €{residuals.std():.1f})')
    ax.legend()
    
    # 4. Percentage Error Distribution
    ax = axes[1, 0]
    pct_clipped = np.clip(pct_errors, -50, 50)
    ax.hist(pct_clipped, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Percentage Error (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'% Error Distribution (median: {np.median(pct_errors):.1f}%)')
    
    # 5. Error by Price Range
    ax = axes[1, 1]
    bins = [50, 75, 100, 125, 150, 175, 200]
    test_df = test_df.copy()
    test_df['price_bin'] = pd.cut(y_test, bins=bins)
    test_df['abs_pct_error'] = np.abs(pct_errors)
    
    bin_errors = test_df.groupby('price_bin', observed=True)['abs_pct_error'].mean()
    ax.bar(range(len(bin_errors)), bin_errors.values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(bin_errors)))
    ax.set_xticklabels([f'€{b.left}-{b.right}' for b in bin_errors.index], rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute % Error')
    ax.set_title('Error by Price Range')
    
    # 6. Summary Stats
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
TEST SET PERFORMANCE
{'='*30}

Samples:     {len(y_test):,}
Hotels:      {test_df['hotel_id'].nunique():,}

MAE:         €{metrics['mae']:.2f}
MAPE:        {metrics['mape']:.1f}%
RMSE:        €{metrics['rmse']:.2f}
R²:          {metrics['r2']:.3f}

Residuals:
  Mean:      €{residuals.mean():.2f}
  Std:       €{residuals.std():.2f}
  Median:    €{np.median(residuals):.2f}

% within ±10%: {(np.abs(pct_errors) <= 10).mean()*100:.1f}%
% within ±20%: {(np.abs(pct_errors) <= 20).mean()*100:.1f}%
"""
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'baseline_diagnostics.png'}")
    
    # Feature importance (if available)
    # Would need to pass the model here


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_training_pipeline(
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    output_dir: str = 'outputs/baseline_model',
    min_price: float = 50,
    max_price: float = 200
) -> TrainingResults:
    """
    Run complete training pipeline.
    
    1. Load data with hotel-level train/val/test split (60/20/20)
    2. Model selection (CV on train)
    3. Train best model on train
    4. Evaluate on held-out test set
    5. Generate diagnostics
    
    Val set is returned for downstream use (e.g., occupancy model).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data with 3-way split
    train_df, val_df, test_df, train_hotels, val_hotels, test_hotels = load_data(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        min_price=min_price,
        max_price=max_price
    )
    
    # Model selection on train set
    comparison = run_model_selection(train_df)
    best_model_name = comparison.iloc[0]['model']
    
    # Train on train set, evaluate on held-out test set
    model, metrics, y_test, y_pred, test_clean = train_and_evaluate(
        train_df, test_df, best_model_name
    )
    
    # Diagnostics
    plot_diagnostics(y_test, y_pred, test_clean, metrics, output_dir)
    
    # Save comparison
    comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"✓ Saved: {output_dir / 'model_comparison.csv'}")
    
    return TrainingResults(
        n_train_hotels=len(train_hotels),
        n_val_hotels=len(val_hotels),
        n_test_hotels=len(test_hotels),
        n_train_samples=len(train_df),
        n_val_samples=len(val_df),
        n_test_samples=len(test_df),
        model_comparison=comparison,
        best_model_name=best_model_name,
        test_mae=metrics['mae'],
        test_mape=metrics['mape'],
        test_rmse=metrics['rmse'],
        test_r2=metrics['r2'],
        y_test=y_test,
        y_pred=y_pred,
        test_df=test_clean,
        train_df=train_df,
        val_df=val_df,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline pricing model')
    parser.add_argument('--train-size', type=float, default=0.6, help='Train fraction (default: 0.6)')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation fraction (default: 0.2)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test fraction (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default='outputs/baseline_model', help='Output directory')
    parser.add_argument('--min-price', type=float, default=50, help='Min price filter')
    parser.add_argument('--max-price', type=float, default=200, help='Max price filter')
    args = parser.parse_args()
    
    results = run_training_pipeline(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        output_dir=args.output_dir,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Split: {results.n_train_hotels} train / {results.n_val_hotels} val / {results.n_test_hotels} test hotels")
    print(f"Best Model: {results.best_model_name}")
    print(f"Test R²:    {results.test_r2:.3f}")
    print(f"Test MAPE:  {results.test_mape:.1f}%")
    print(f"Test MAE:   €{results.test_mae:.2f}")
    print(f"\nVal set ({results.n_val_samples:,} samples) available for downstream models.")

