"""
Daily Pricing Model.

Trains on daily booking data (not monthly aggregates) to properly learn
holiday effects and day-of-week pricing patterns.

Target: daily_price (booking_total_price / stay_nights)
Features: hotel characteristics + temporal features (holidays, weekends, seasonality)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from lib.db import init_db
from lib.holiday_features import map_hotels_to_admin1, add_date_holiday_features
from ml_pipeline.features import (
    standardize_city, 
    VIEW_QUALITY_MAP,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES
)


MODEL_DIR = Path(__file__).parent.parent / 'outputs' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CITIES500_PATH = Path(__file__).parent.parent / 'data' / 'cities500.json'


def load_daily_data() -> pd.DataFrame:
    """
    Loads daily booking data with holiday features.
    """
    con = init_db()
    
    sql_file = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity/QUERY_LOAD_DAILY_BOOKING_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    df = con.execute(query).fetchdf()
    
    print(f"Loaded {len(df):,} daily bookings")
    
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds holiday features based on arrival_date and hotel location.
    """
    df = df.copy()
    
    # Map hotels to Spanish regions
    hotels = df[['hotel_id', 'latitude', 'longitude']].drop_duplicates()
    
    if CITIES500_PATH.exists():
        hotel_admin1 = map_hotels_to_admin1(
            hotels,
            CITIES500_PATH,
            lat_col='latitude',
            lon_col='longitude',
            hotel_id_col='hotel_id'
        )
        print(f"Mapped {len(hotel_admin1)} hotels to Spanish regions")
    else:
        # Fallback: use Madrid for all
        hotel_admin1 = pd.DataFrame({
            'hotel_id': hotels['hotel_id'],
            'subdiv_code': 'MD'
        })
        print("Warning: cities500.json not found, using Madrid holidays as fallback")
    
    # Add holiday features based on arrival_date
    df = add_date_holiday_features(
        df,
        hotel_admin1,
        date_col='arrival_date',
        hotel_id_col='hotel_id'
    )
    
    # Show holiday distribution
    print(f"Holiday bookings: {df['is_holiday'].sum():,} ({df['is_holiday'].mean()*100:.1f}%)")
    print(f"Near-holiday (±1 day): {df['is_holiday_pm1'].sum():,} ({df['is_holiday_pm1'].mean()*100:.1f}%)")
    print(f"Near-holiday (±2 days): {df['is_holiday_pm2'].sum():,} ({df['is_holiday_pm2'].mean()*100:.1f}%)")
    
    return df


def engineer_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for daily pricing model.
    """
    df = df.copy()
    
    # Log transforms
    df['log_room_size'] = np.log1p(df['avg_room_size'])
    df['total_capacity_log'] = np.log1p(df['total_capacity'])
    
    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month_number'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_number'] / 12)
    
    # Seasonal flags
    df['is_summer'] = df['month_number'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month_number'].isin([12, 1, 2]).astype(int)
    
    # Standardize city
    df['city_standardized'] = df['city'].apply(standardize_city)
    
    # View quality ordinal
    df['view_quality_ordinal'] = df['room_view'].map(VIEW_QUALITY_MAP).fillna(0)
    
    return df


def prepare_features(
    df: pd.DataFrame,
    encoders: Dict = None,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, Dict, StandardScaler]:
    """
    Prepares features for model training/prediction.
    """
    # Target: log of daily price
    y = np.log(df['daily_price'])
    
    if encoders is None:
        encoders = {}
    
    df_encoded = df.copy()
    
    # Encode categoricals
    cat_features = ['room_type', 'room_view', 'city_standardized']
    for col in cat_features:
        if col in df_encoded.columns:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le:
                    df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
    
    # Feature columns
    feature_cols = [
        # Numeric
        'log_room_size', 'room_capacity_pax', 'total_capacity_log',
        'view_quality_ordinal', 'amenities_score',
        'month_sin', 'month_cos',
        # Categorical (encoded)
        'room_type', 'room_view', 'city_standardized',
        # Boolean
        'is_weekend', 'is_summer', 'is_winter', 'children_allowed',
        # Holiday features (the key new ones!)
        'is_holiday', 'is_holiday_pm1', 'is_holiday_pm2',
        # Day of week
        'day_of_week'
    ]
    
    # Only use columns that exist
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[feature_cols].copy().fillna(0)
    
    # Scale numeric features
    numeric_cols = ['log_room_size', 'room_capacity_pax', 'total_capacity_log',
                    'view_quality_ordinal', 'amenities_score', 'month_sin', 'month_cos']
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    
    if fit:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y, encoders, scaler


def train_daily_model() -> Dict:
    """
    Trains and evaluates the daily pricing model.
    """
    print("=" * 80)
    print("DAILY PRICING MODEL (with Holiday Features)")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading daily booking data...")
    df = load_daily_data()
    
    # Add holiday features
    print("\n2. Adding holiday features...")
    df = add_holiday_features(df)
    
    # Engineer features
    print("\n3. Engineering features...")
    df = engineer_daily_features(df)
    
    # Core market filter (€50-250)
    df = df[(df['daily_price'] >= 50) & (df['daily_price'] <= 250)]
    print(f"   After core market filter (€50-250): {len(df):,} bookings")
    
    # Temporal split
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    train_df = df[df['arrival_date'] < '2024-06-01'].copy()
    test_df = df[df['arrival_date'] >= '2024-06-01'].copy()
    print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Prepare features
    print("\n4. Preparing features...")
    X_train, y_train, encoders, scaler = prepare_features(train_df, fit=True)
    X_test, y_test, _, _ = prepare_features(test_df, encoders=encoders, scaler=scaler, fit=False)
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Feature list: {list(X_train.columns)}")
    
    # Train model
    print("\n5. Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n6. Evaluating...")
    y_pred = model.predict(X_test)
    
    # Metrics on log scale
    r2_log = r2_score(y_test, y_pred)
    
    # Metrics on original scale
    actual_prices = np.exp(y_test)
    predicted_prices = np.exp(y_pred)
    
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    r2_price = r2_score(actual_prices, predicted_prices)
    
    print(f"\n   Test R² (log): {r2_log:.4f}")
    print(f"   Test R² (price): {r2_price:.4f}")
    print(f"   Test MAPE: {mape:.1f}%")
    
    # Feature importance
    print("\n7. Feature Importance:")
    feature_names = X_train.columns.tolist()
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    for i in sorted_idx[:15]:
        marker = ' ⭐ HOLIDAY' if 'holiday' in feature_names[i].lower() else ''
        marker = ' 📅 TEMPORAL' if feature_names[i] in ['is_weekend', 'day_of_week', 'is_summer', 'is_winter'] else marker
        print(f"   {feature_names[i]:25s} {importances[i]:.4f}{marker}")
    
    # Compare holiday vs non-holiday prices
    print("\n8. Holiday Price Analysis:")
    print(f"   Avg price on holidays: €{test_df[test_df['is_holiday']==1]['daily_price'].mean():.2f}")
    print(f"   Avg price non-holidays: €{test_df[test_df['is_holiday']==0]['daily_price'].mean():.2f}")
    holiday_premium = (test_df[test_df['is_holiday']==1]['daily_price'].mean() / 
                       test_df[test_df['is_holiday']==0]['daily_price'].mean() - 1) * 100
    print(f"   Actual holiday premium: {holiday_premium:+.1f}%")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': feature_names,
        'test_mape': mape,
        'test_r2': r2_price
    }
    
    with open(MODEL_DIR / 'daily_pricing_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✓ Model saved to {MODEL_DIR / 'daily_pricing_model.pkl'}")
    
    return {
        'mape': mape,
        'r2': r2_price,
        'r2_log': r2_log,
        'model': model,
        'feature_importance': dict(zip(feature_names, importances))
    }


if __name__ == "__main__":
    results = train_daily_model()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Daily Pricing Model MAPE: {results['mape']:.1f}%")
    print(f"Daily Pricing Model R²: {results['r2']:.4f}")

