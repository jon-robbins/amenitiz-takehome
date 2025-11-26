# %%
"""
Feature Importance & Sufficiency Test

Objective:
Prove that observable features explain hotel pricing (R-squared > 0.70), validating 
that "unobserved quality" (decor, service, brand) plays a minor role. This 
mathematically justifies the matched pairs methodology.

Methodology:
1. Feature Engineering: Geographic, product, temporal, and categorical signals
2. Model Pipeline: Ridge, RandomForest, XGBoost, LightGBM
3. Evaluation: R-squared, RMSE, MAE with cross-validation
4. SHAP Analysis: Feature importance ranking and dependence plots
5. Sufficiency Test: Validate if observable features explain pricing
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import re

# Database and cleaning
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Tree-based models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


# %%
def get_cleaning_config() -> CleaningConfig:
    """Returns standard cleaning configuration."""
    return CleaningConfig(
        remove_negative_prices=True,
        remove_zero_prices=True,
        remove_low_prices=True,
        remove_null_prices=True,
        remove_extreme_prices=True,
        remove_null_dates=True,
        remove_null_created_at=True,
        remove_negative_stay=True,
        remove_negative_lead_time=True,
        remove_null_occupancy=True,
        remove_overcrowded_rooms=True,
        remove_null_room_id=True,
        remove_null_booking_id=True,
        remove_null_hotel_id=True,
        remove_orphan_bookings=True,
        remove_null_status=True,
        remove_cancelled_but_active=True,
        remove_bookings_before_2023=True,
        remove_bookings_after_2024=True,
        exclude_reception_halls=True,
        exclude_missing_location=True,
        fix_empty_strings=True,
        impute_children_allowed=True,
        impute_events_allowed=True,
        match_city_names_with_tfidf=True,
        set_empty_room_view_to_no_view_str=True,
        verbose=False
    )


# %%
def load_hotel_month_data(con) -> pd.DataFrame:
    """
    Loads hotel-month aggregation with all features.
    
    CORRECTED CALCULATIONS (based on schema exploration):
    1. Explode each booking to daily granularity (each night is a row)
    2. ADR = total_price / nights_stayed
    3. Hotel capacity = SUM(number_of_rooms) for all distinct room types
    4. Aggregate to HOTEL-MONTH level first to get total room-nights sold
    5. Occupancy = total_room_nights_sold / (hotel_capacity × days_in_month)
    6. Then join back room-type features for modeling
    
    Returns:
        DataFrame with hotel_id, month as grain
    """
    query = """
    -- Step 1: Get TRUE hotel capacity (sum of distinct room types per hotel)
    WITH hotel_capacity AS (
        SELECT DISTINCT
            b.hotel_id,
            r.id as room_type_id,
            r.number_of_rooms
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN rooms r ON br.room_id = r.id
        WHERE b.status IN ('confirmed', 'Booked')
    ),
    hotel_total_capacity AS (
        SELECT hotel_id, SUM(number_of_rooms) as hotel_rooms
        FROM hotel_capacity
        GROUP BY hotel_id
    ),
    
    -- Step 2: Explode bookings to daily granularity
    daily_bookings AS (
        SELECT 
            b.hotel_id,
            CAST(b.arrival_date + (n * INTERVAL '1 day') AS DATE) as stay_date,
            br.room_type,
            COALESCE(NULLIF(br.room_view, ''), 'no_view') AS room_view,
            r.children_allowed,
            hl.city,
            hl.latitude,
            hl.longitude,
            br.total_price / (b.departure_date - b.arrival_date) as nightly_rate,
            br.room_size,
            r.max_occupancy as room_capacity_pax,
            r.events_allowed,
            r.pets_allowed,
            r.smoking_allowed
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        JOIN rooms r ON br.room_id = r.id
        CROSS JOIN generate_series(0, (b.departure_date - b.arrival_date) - 1) as t(n)
        WHERE b.status IN ('confirmed', 'Booked')
          AND CAST(b.arrival_date AS DATE) BETWEEN '2023-01-01' AND '2024-12-31'
          AND hl.city IS NOT NULL
          AND (b.departure_date - b.arrival_date) > 0
    ),
    
    -- Step 3: FIRST aggregate to HOTEL-MONTH level to get correct total occupancy
    hotel_month_totals AS (
        SELECT 
            db.hotel_id,
            DATE_TRUNC('month', db.stay_date) AS month,
            MAX(db.city) as city,
            MAX(db.latitude) as latitude,
            MAX(db.longitude) as longitude,
            
            -- TOTAL revenue and room-nights for the ENTIRE hotel
            SUM(db.nightly_rate) AS total_revenue,
            COUNT(*) AS total_room_nights_sold,
            AVG(db.nightly_rate) AS avg_adr,
            
            -- Temporal
            EXTRACT(MONTH FROM MAX(db.stay_date)) AS month_number,
            EXTRACT(DAY FROM LAST_DAY(MAX(db.stay_date))) AS days_in_month,
            SUM(CASE WHEN EXTRACT(ISODOW FROM db.stay_date) >= 6 THEN 1 ELSE 0 END)::FLOAT / 
                NULLIF(COUNT(*), 0) AS weekend_ratio
        FROM daily_bookings db
        GROUP BY db.hotel_id, month
    ),
    
    -- Step 4: Get room-type features (most common per hotel-month)
    hotel_month_room_features AS (
        SELECT 
            db.hotel_id,
            DATE_TRUNC('month', db.stay_date) AS month,
            -- Use the most common room type/view for this hotel-month
            MODE() WITHIN GROUP (ORDER BY db.room_type) as room_type,
            MODE() WITHIN GROUP (ORDER BY db.room_view) as room_view,
            MAX(db.children_allowed) as children_allowed,
            AVG(db.room_size) AS avg_room_size,
            MAX(db.room_capacity_pax) AS room_capacity_pax,
            (CAST(MAX(db.events_allowed) AS INT) + 
             CAST(MAX(db.pets_allowed) AS INT) + 
             CAST(MAX(db.smoking_allowed) AS INT) + 
             CAST(MAX(db.children_allowed) AS INT)) AS amenities_score
        FROM daily_bookings db
        GROUP BY db.hotel_id, month
    )
    
    -- Step 5: Join everything and calculate CORRECT occupancy
    SELECT 
        hmt.hotel_id,
        hmt.month,
        hmrf.room_type,
        hmrf.room_view,
        hmrf.children_allowed,
        hmt.city,
        hmt.latitude,
        hmt.longitude,
        hmt.total_revenue,
        hmt.total_room_nights_sold as room_nights_sold,
        hmt.avg_adr,
        hmrf.avg_room_size,
        hmrf.room_capacity_pax,
        hmt.month_number,
        hmt.days_in_month,
        hmt.weekend_ratio,
        hmrf.amenities_score,
        -- View quality (ordinal 0-3)
        CASE 
            WHEN hmrf.room_view IN ('ocean_view', 'sea_view') THEN 3
            WHEN hmrf.room_view IN ('lake_view', 'mountain_view') THEN 2
            WHEN hmrf.room_view IN ('pool_view', 'garden_view') THEN 1
            ELSE 0
        END AS view_quality_ordinal,
        htc.hotel_rooms AS total_capacity,
        -- CORRECT occupancy: TOTAL room_nights / (hotel_capacity × days)
        (hmt.total_room_nights_sold::FLOAT / NULLIF(htc.hotel_rooms * hmt.days_in_month, 0)) AS occupancy_rate
    FROM hotel_month_totals hmt
    JOIN hotel_total_capacity htc ON hmt.hotel_id = htc.hotel_id
    JOIN hotel_month_room_features hmrf ON hmt.hotel_id = hmrf.hotel_id AND hmt.month = hmrf.month
    WHERE htc.hotel_rooms > 0 AND hmt.total_room_nights_sold > 0 AND hmt.avg_adr > 0
    """
    return con.execute(query).fetchdf()


# %%
def engineer_features(df: pd.DataFrame, distance_features: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers all features for modeling.
    
    Features:
    - Geographic: dist_center_km, is_coastal, dist_coast_log, dist_madrid_log, lat, lon
    - Product: log_room_size, view_quality_ordinal, room_capacity_pax, amenities_score, total_capacity_log
    - Temporal: month_sin, month_cos, weekend_ratio, is_summer, is_winter
    - City indicators: Binary flags for top 10 cities (is_madrid, is_barcelona, etc.)
    """
    df = df.copy()
    
    # Merge distance features
    df = df.merge(distance_features, on='hotel_id', how='left')
    
    # Top 5 cities by revenue with canonical names
    top_5_canonical = {
        'madrid': 'madrid',
        'barcelona': 'barcelona',
        'sevilla': 'sevilla',
        'malaga': 'malaga',
        'málaga': 'malaga',
        'toledo': 'toledo'
    }
    
    def clean_city_name(name):
        if pd.isna(name):
            return ''
        cleaned = re.sub(r'[^\w\s]', '', str(name).lower().strip())
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned
    
    def standardize_city(city_str):
        if pd.isna(city_str):
            return 'other'
        
        city_clean = clean_city_name(city_str)
        
        if city_clean in top_5_canonical:
            return top_5_canonical[city_clean]
        
        for canonical_key in top_5_canonical.keys():
            if canonical_key in city_clean:
                return top_5_canonical[canonical_key]
        
        return 'other'
    
    df['city_standardized'] = df['city'].apply(standardize_city)
    
    # Keep city_standardized as categorical (will be used by CatBoost)
    # No need to create binary indicators - CatBoost handles categoricals natively

    
    # Calculate city centroids (booking-weighted mean) using standardized cities
    city_centroids = df.groupby('city_standardized').apply(
        lambda x: pd.Series({
            'city_lat': np.average(x['latitude'], weights=x['room_nights_sold']),
            'city_lon': np.average(x['longitude'], weights=x['room_nights_sold'])
        }), include_groups=False
    ).reset_index()
    
    df = df.merge(city_centroids, on='city_standardized', how='left')
    
    # Geographic features
    df['dist_center_km'] = np.sqrt(
        (df['latitude'] - df['city_lat'])**2 + 
        (df['longitude'] - df['city_lon'])**2
    ) * 111  # Rough conversion to km
    
    df['is_coastal'] = (df['distance_from_coast'] < 20).astype(int)
    df['dist_coast_log'] = np.log1p(df['distance_from_coast'])
    df['dist_madrid_log'] = np.log1p(df['distance_from_madrid'])
    
    # Product features
    df['log_room_size'] = np.log1p(df['avg_room_size'])
    df['total_capacity_log'] = np.log1p(df['total_capacity'])
    
    # Temporal features
    df['is_july_august'] = df['month_number'].isin([7, 8]).astype(int)  # Peak summer
    
    # Target variable (log-transformed ADR)
    df['log_avg_adr'] = np.log(df['avg_adr'])
    
    # Fill NaN values in city centroids (for cities with no standardized match)
    df['city_lat'] = df['city_lat'].fillna(df['latitude'])
    df['city_lon'] = df['city_lon'].fillna(df['longitude'])
    df['dist_center_km'] = df['dist_center_km'].fillna(0)
    
    return df


# %%


# %%
def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
    """
    Prepares feature matrix and target variable.
    
    Returns:
        X, y, numeric_features, categorical_features, boolean_features
    """
    # Define feature groups - FULL SET WITH OCCUPANCY
    numeric_features = [
        'dist_center_km', 'dist_coast_log',
        'log_room_size', 'room_capacity_pax', 'amenities_score', 'total_capacity_log',
        'view_quality_ordinal',
        'weekend_ratio',
        'occupancy_rate'  # Now properly calculated (NULL room_ids filtered)
    ]
    
    # Categorical features (for CatBoost - no one-hot encoding needed)
    categorical_features = ['room_type', 'room_view', 'city_standardized']
    
    boolean_features = [
        'is_coastal', 'is_july_august', 'children_allowed'
    ]
    
    # Combine all features
    all_features = numeric_features + categorical_features + boolean_features
    
    # Filter to available features
    available_features = [f for f in all_features if f in df.columns]
    
    X = df[available_features].copy()
    y = df['log_avg_adr'].copy()
    
    # Update feature lists to only include available features
    numeric_features = [f for f in numeric_features if f in available_features]
    categorical_features = [f for f in categorical_features if f in available_features]
    boolean_features = [f for f in boolean_features if f in available_features]
    
    return X, y, numeric_features, categorical_features, boolean_features


# %%
def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    boolean_features: List[str]
) -> ColumnTransformer:
    """
    Creates preprocessing pipeline with StandardScaler and OneHotEncoder.
    """
    transformers = []
    
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    
    if categorical_features:
        # Use OneHotEncoder with handle_unknown='ignore' for robustness
        transformers.append(
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        )
    
    if boolean_features:
        transformers.append(('bool', 'passthrough', boolean_features))
    
    return ColumnTransformer(transformers=transformers)


# %%
def build_models() -> Dict[str, any]:
    """
    Builds dictionary of models to compare.
    """
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    # Add RandomForest
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # Add CatBoost if available (handles categoricals natively)
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(
            depth=6,
            learning_rate=0.1,
            iterations=200,
            random_state=42,
            verbose=False,
            allow_writing_files=False
        )
    
    return models


# %%
def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    models: Dict[str, any],
    cv_folds: int = 5
) -> Tuple[Dict[str, Dict[str, float]], any, any, any, any]:
    """
    Evaluates all models with train/test split and cross-validation.
    Special handling for CatBoost with categorical features.
    
    Returns:
        results_dict, best_model_name, best_pipeline, X_test, y_test
    """
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = {}
    best_r2 = -np.inf
    best_model_name = None
    best_pipeline = None
    
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        # Special handling for CatBoost
        if model_name == 'CatBoost':
            # CatBoost handles categorical features natively - don't one-hot encode
            # Prepare data with categorical features as strings
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import make_column_transformer
            
            # Get feature lists from parent scope - FULL SET WITH OCCUPANCY
            numeric_features = [
                'dist_center_km', 'dist_coast_log',
                'log_room_size', 'room_capacity_pax', 'amenities_score', 'total_capacity_log',
                'view_quality_ordinal',
                'weekend_ratio',
                'occupancy_rate'
            ]
            categorical_features = ['room_type', 'room_view', 'city_standardized']
            boolean_features = ['is_coastal', 'is_july_august', 'children_allowed']
            
            # Create preprocessor for CatBoost (no one-hot encoding)
            catboost_preprocessor = make_column_transformer(
                (StandardScaler(), numeric_features),
                ('passthrough', categorical_features),
                ('passthrough', boolean_features),
                remainder='drop'
            )
            
            # Transform data
            X_train_cb = catboost_preprocessor.fit_transform(X_train)
            X_test_cb = catboost_preprocessor.transform(X_test)
            
            # Convert to DataFrame to preserve categorical info
            all_features = numeric_features + categorical_features + boolean_features
            X_train_cb_df = pd.DataFrame(X_train_cb, columns=all_features)
            X_test_cb_df = pd.DataFrame(X_test_cb, columns=all_features)
            
            # Ensure categorical columns are strings
            for cat_col in categorical_features:
                X_train_cb_df[cat_col] = X_train_cb_df[cat_col].astype(str)
                X_test_cb_df[cat_col] = X_test_cb_df[cat_col].astype(str)
            
            # Get categorical feature indices
            cat_feature_indices = [all_features.index(col) for col in categorical_features]
            
            # Fit CatBoost
            model.fit(X_train_cb_df, y_train, cat_features=cat_feature_indices, verbose=False)
            
            # Predictions
            y_pred_train = model.predict(X_train_cb_df)
            y_pred_test = model.predict(X_test_cb_df)
            
            # Metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation (simplified for CatBoost)
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                X_tr_t = catboost_preprocessor.fit_transform(X_tr)
                X_val_t = catboost_preprocessor.transform(X_val)
                
                X_tr_df = pd.DataFrame(X_tr_t, columns=all_features)
                X_val_df = pd.DataFrame(X_val_t, columns=all_features)
                
                for cat_col in categorical_features:
                    X_tr_df[cat_col] = X_tr_df[cat_col].astype(str)
                    X_val_df[cat_col] = X_val_df[cat_col].astype(str)
                
                model_cv = CatBoostRegressor(
                    depth=6, learning_rate=0.1, iterations=200,
                    random_state=42, verbose=False, allow_writing_files=False
                )
                model_cv.fit(X_tr_df, y_tr, cat_features=cat_feature_indices, verbose=False)
                y_pred_val = model_cv.predict(X_val_df)
                cv_scores.append(r2_score(y_val, y_pred_val))
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Store as "pipeline" for consistency
            class CatBoostPipeline:
                def __init__(self, preprocessor, model, categorical_features, all_features):
                    self.preprocessor = preprocessor
                    self.model = model
                    self.categorical_features = categorical_features
                    self.all_features = all_features
                    self.named_steps = {'preprocessor': preprocessor, 'model': model}
                
                def predict(self, X):
                    X_transformed = self.preprocessor.transform(X)
                    X_df = pd.DataFrame(X_transformed, columns=self.all_features)
                    for cat_col in self.categorical_features:
                        X_df[cat_col] = X_df[cat_col].astype(str)
                    return self.model.predict(X_df)
            
            pipeline = CatBoostPipeline(catboost_preprocessor, model, categorical_features, all_features)
            
        else:
            # Standard pipeline for other models
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Fit model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            # Metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=cv_folds, scoring='r2', n_jobs=-1
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        
        results[model_name] = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        print(f"  R² (Train): {r2_train:.4f}")
        print(f"  R² (Test):  {r2_test:.4f}")
        print(f"  RMSE (Test): {rmse_test:.4f}")
        print(f"  MAE (Test):  {mae_test:.4f}")
        print(f"  CV R² (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Track best model
        if r2_test > best_r2:
            best_r2 = r2_test
            best_model_name = model_name
            best_pipeline = pipeline
    
    return results, best_model_name, best_pipeline, X_test, y_test


# %%
def print_sufficiency_test_results(results: Dict[str, Dict[str, float]], best_model_name: str) -> None:
    """
    Prints sufficiency test verdict.
    """
    print("\n" + "=" * 80)
    print("SUFFICIENCY TEST RESULTS")
    print("=" * 80)
    
    best_r2 = results[best_model_name]['r2_test']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"R-squared (Test): {best_r2:.4f}")
    
    print("\nVERDICT:")
    if best_r2 > 0.70:
        print("✓ PASS - Observable features explain pricing (R² > 0.70)")
        print("  → Matched pairs methodology is VALID")
        print("  → Unobserved quality factors play a MINOR role")
    elif best_r2 > 0.50:
        print("⚠ PARTIAL - Observable features explain most pricing (0.50 < R² < 0.70)")
        print("  → Matched pairs methodology is REASONABLE")
        print("  → Some unobserved factors may exist but are not dominant")
    elif best_r2 > 0.40:
        print("⚠ WEAK - Observable features explain some pricing (0.40 < R² < 0.50)")
        print("  → Matched pairs methodology requires CAUTION")
        print("  → Unobserved factors may be important")
    else:
        print("✗ FAIL - Observable features do not explain pricing (R² < 0.40)")
        print("  → Matched pairs methodology is QUESTIONABLE")
        print("  → Major confounders are likely missing")


# %%
def create_visualizations(
    results: Dict[str, Dict[str, float]],
    best_model_name: str,
    best_pipeline: any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """
    Creates 6 key visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    # =========================================================================
    # PLOT 1: Model Comparison Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    r2_scores = [results[m]['r2_test'] for m in model_names]
    colors = ['#2E86AB' if m != best_model_name else '#3E8914' for m in model_names]
    
    bars = ax.bar(model_names, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add threshold line
    ax.axhline(0.70, color='red', linestyle='--', linewidth=2, label='Sufficiency Threshold (0.70)')
    
    ax.set_title('Model Comparison: R-squared on Test Set', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('R-squared', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # PLOT 2: Actual vs Predicted Scatter
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pred = best_pipeline.predict(X_test)
    
    # Scatter plot with density coloring
    scatter = ax.scatter(y_test, y_pred, alpha=0.5, s=20, c='#2E86AB', edgecolors='none')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add R² annotation
    r2 = results[best_model_name]['r2_test']
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))
    
    ax.set_title(f'Actual vs Predicted: {best_model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Actual log(ADR)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted log(ADR)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # PLOT 3: Residual Distribution
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    residuals = y_test - y_pred
    
    # Histogram with KDE
    ax1.hist(residuals, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black', density=True)
    
    # Add KDE
    from scipy import stats
    kde_x = np.linspace(residuals.min(), residuals.max(), 100)
    kde = stats.gaussian_kde(residuals)
    ax1.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    
    # Add normal distribution for comparison
    mu, sigma = residuals.mean(), residuals.std()
    normal_curve = stats.norm.pdf(kde_x, mu, sigma)
    ax1.plot(kde_x, normal_curve, 'g--', linewidth=2, label='Normal Distribution')
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    ax1.set_title('Residual Distribution', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Residuals vs Normal', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualizations to {output_dir}")


# %%
def create_shap_analysis(
    best_model_name: str,
    best_pipeline: any,
    X_test: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    boolean_features: List[str],
    output_dir: Path
) -> None:
    """
    Creates SHAP analysis visualizations.
    """
    if not SHAP_AVAILABLE:
        print("\n⚠ SHAP not available. Skipping SHAP analysis.")
        return
    
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS")
    print("=" * 80)
    
    # Get the trained model from pipeline
    model = best_pipeline.named_steps['model']
    
    # Transform test data
    X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Get feature names after transformation
    feature_names = []
    
    # Numeric features
    feature_names.extend(numeric_features)
    
    # Categorical features (one-hot encoded)
    if categorical_features:
        cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    # Boolean features
    feature_names.extend(boolean_features)
    
    # Convert to DataFrame for SHAP (ensure numeric types)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    # Convert all columns to float to avoid object dtype issues
    for col in X_test_df.columns:
        X_test_df[col] = pd.to_numeric(X_test_df[col], errors='coerce')
    
    # Create SHAP explainer
    print("\nCreating SHAP explainer...")
    
    if best_model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_df)
    else:
        # Use KernelExplainer for Ridge (slower but works)
        explainer = shap.KernelExplainer(model.predict, X_test_df.sample(min(100, len(X_test_df)), random_state=42))
        shap_values = explainer(X_test_df.sample(min(500, len(X_test_df)), random_state=42))
    
    # =========================================================================
    # PLOT 4: SHAP Beeswarm (Summary Plot)
    # =========================================================================
    print("Creating SHAP beeswarm plot...")
    plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title('SHAP Feature Importance (Beeswarm)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '4_shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # PLOT 5: SHAP Dependence Plots (Top 3 Features)
    # =========================================================================
    print("Creating SHAP dependence plots...")
    
    # Get top 3 features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_3_indices = np.argsort(mean_abs_shap)[-3:][::-1]
    top_3_features = [feature_names[i] for i in top_3_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (feature_idx, feature_name) in enumerate(zip(top_3_indices, top_3_features)):
        ax = axes[idx]
        
        # Get feature values and SHAP values
        feature_values = X_test_df.iloc[:, feature_idx].values
        feature_shap_values = shap_values.values[:, feature_idx]
        
        # Scatter plot
        ax.scatter(feature_values, feature_shap_values, alpha=0.5, s=20, c='#2E86AB')
        
        # Add trend line
        z = np.polyfit(feature_values, feature_shap_values, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(feature_values.min(), feature_values.max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8)
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax.set_xlabel(feature_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('SHAP Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Dependence: {feature_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_shap_dependence_top3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # PLOT 6: Feature Importance Comparison (Tree-based vs SHAP)
    # =========================================================================
    if best_model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
        print("Creating feature importance comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Tree-based importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:][::-1]
            
            ax1.barh(range(len(indices)), importances[indices], color='#2E86AB', alpha=0.8)
            ax1.set_yticks(range(len(indices)))
            ax1.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
            ax1.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax1.set_title(f'{best_model_name} Feature Importance', fontsize=14, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
        
        # SHAP importance
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        shap_indices = np.argsort(shap_importance)[-15:][::-1]
        
        ax2.barh(range(len(shap_indices)), shap_importance[shap_indices], color='#F18F01', alpha=0.8)
        ax2.set_yticks(range(len(shap_indices)))
        ax2.set_yticklabels([feature_names[i] for i in shap_indices], fontsize=10)
        ax2.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        ax2.set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / '6_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ SHAP analysis complete. Saved to {output_dir}")


# %%
def save_results(
    results: Dict[str, Dict[str, float]],
    output_path: Path
) -> None:
    """
    Saves detailed results to CSV.
    """
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'model'
    results_df = results_df.reset_index()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved detailed results to {output_path}")


# %%
def main():
    """Main execution."""
    print("=" * 80)
    print("FEATURE IMPORTANCE & SUFFICIENCY TEST")
    print("=" * 80)
    
    # Load and clean data
    print("\n1. Loading and cleaning data...")
    config = get_cleaning_config()
    cleaner = DataCleaner(config)
    con = cleaner.clean(init_db())
    
    # Load hotel-month data
    print("\n2. Loading hotel-month aggregation...")
    df = load_hotel_month_data(con)
    print(f"   Loaded {len(df):,} hotel-month-roomtype records")
    
    # Load distance features
    print("\n3. Loading distance features...")
    script_dir = Path(__file__).parent
    distance_features_path = script_dir / '../../../outputs/eda/spatial/data/hotel_distance_features.csv'
    distance_features = pd.read_csv(distance_features_path.resolve())
    print(f"   Loaded distance features for {len(distance_features):,} hotels")
    
    # Engineer features
    print("\n4. Engineering features...")
    df = engineer_features(df, distance_features)
    df = df.dropna(subset=['log_avg_adr', 'distance_from_coast', 'distance_from_madrid'])
    print(f"   Final dataset: {len(df):,} records")
    
    # Prepare features and target
    print("\n5. Preparing features and target...")
    X, y, numeric_features, categorical_features, boolean_features = prepare_features_and_target(df)
    print(f"   Features: {len(X.columns)} total")
    print(f"     - Numeric: {len(numeric_features)}")
    print(f"     - Categorical: {len(categorical_features)}")
    print(f"     - Boolean: {len(boolean_features)}")
    
    # Create preprocessing pipeline
    print("\n6. Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features, boolean_features)
    
    # Build models
    print("\n7. Building models...")
    models = build_models()
    print(f"   Models to evaluate: {list(models.keys())}")
    
    # Evaluate models
    print("\n8. Evaluating models...")
    results, best_model_name, best_pipeline, X_test, y_test = evaluate_models(
        X, y, preprocessor, models, cv_folds=5
    )
    
    # Print sufficiency test results
    print_sufficiency_test_results(results, best_model_name)
    
    # Create visualizations
    print("\n9. Creating visualizations...")
    output_dir = script_dir / '../../../outputs/eda/elasticity/figures'
    create_visualizations(results, best_model_name, best_pipeline, X_test, y_test, output_dir)
    
    # SHAP analysis
    print("\n10. Running SHAP analysis...")
    create_shap_analysis(
        best_model_name, best_pipeline, X_test,
        numeric_features, categorical_features, boolean_features,
        output_dir
    )
    
    # Save results
    print("\n11. Saving results...")
    results_path = script_dir / '../../../outputs/eda/elasticity/data/feature_importance_results.csv'
    save_results(results, results_path)
    
    print("\n" + "=" * 80)
    print("✓ FEATURE IMPORTANCE VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - Figures: {output_dir}")
    print(f"  - Results: {results_path}")


# %%
if __name__ == "__main__":
    main()
