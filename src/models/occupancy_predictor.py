"""
Occupancy Prediction Model.

Predicts occupancy given a candidate price, enabling RevPAR optimization.

Key insight: The model learns heterogeneous elasticity - how occupancy 
responds to price varies by hotel type, market segment, and time.

Causal structure:
    Price (set by hotel) --> Occupancy (demand response) --> RevPAR = Price × Occupancy

Features:
- Hotel characteristics (location, product, capacity)
- Candidate price (what we're evaluating)
- Price relative to peers (key elasticity signal)
- Peer context (market conditions)
- Temporal (seasonality, day of week)
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

import pickle
from pathlib import Path

from src.models.peer_matching import (
    UnifiedPeerMatcher,
    PeerStats,
    DEFAULT_RADIUS_KM,
    DEFAULT_K
)
from src.features.engineering import standardize_city, get_market_segment


# Default model save path
MODEL_SAVE_PATH = Path("outputs/models/occupancy_model.pkl")


# =============================================================================
# CONSTANTS
# =============================================================================

# Path to pre-computed distance features
DISTANCE_FEATURES_PATH = "outputs/data/hotel_distance_features.csv"

# Feature columns for occupancy model
# Note: Removed raw lat/long - use derived geographic features instead
HOTEL_FEATURES = [
    # Product features
    'log_room_size',
    'amenities_score',
    'view_quality_ordinal',
    'log_total_rooms',  # Log-transformed for scale
    'room_capacity_pax',
    # Geographic features (derived, not raw coords)
    'dist_center_km',
    'dist_coast_km',
    'dist_madrid_km',
    'is_coastal',
    'is_madrid_metro',
]

TEMPORAL_FEATURES = [
    'month_sin',
    'month_cos',
    'week_of_year',
    'is_summer',
    'is_weekend_heavy',
]

PRICE_FEATURES = [
    'candidate_price',
    'price_vs_peer_median',  # candidate_price / peer_median (key!)
    'price_vs_peer_p25',
    'price_vs_peer_p75',
]

PEER_FEATURES = [
    'peer_median_occupancy',
    'peer_mean_occupancy',
    'peer_n',
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OccupancyPrediction:
    """Result of occupancy prediction."""
    predicted_occupancy: float
    predicted_revpar: float
    candidate_price: float
    confidence: float
    features_used: Dict


@dataclass
class OccupancyModelMetrics:
    """Training metrics."""
    r2_train: float
    r2_cv: float
    mae_train: float
    mae_cv: float
    n_samples: int
    model_name: str = ""


def get_candidate_models() -> Dict[str, any]:
    """
    Get all candidate models to test for occupancy prediction.
    
    Returns:
        Dict mapping model name to model instance
    """
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        ),
    }
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
    
    # MLP (neural network)
    models['MLP'] = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    return models


def select_best_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    verbose: bool = True
) -> Tuple[any, str, Dict[str, float]]:
    """
    Test multiple models and select the best one.
    
    Args:
        X: Feature matrix (scaled)
        y: Target vector
        cv_folds: Number of CV folds
        verbose: Print progress
    
    Returns:
        Tuple of (best_model, best_model_name, all_scores)
    """
    models = get_candidate_models()
    results = {}
    
    if verbose:
        print(f"Testing {len(models)} models...")
    
    best_score = -np.inf
    best_model = None
    best_name = None
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2', n_jobs=-1)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results[name] = {'mean': mean_score, 'std': std_score}
            
            if verbose:
                print(f"  {name:20}: R² = {mean_score:.4f} ± {std_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        except Exception as e:
            if verbose:
                print(f"  {name:20}: FAILED ({e})")
            results[name] = {'mean': np.nan, 'std': np.nan}
    
    if verbose:
        print(f"\nBest model: {best_name} (R² = {best_score:.4f})")
    
    return best_model, best_name, results


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features to dataframe."""
    df = df.copy()
    
    # Extract month/week from week_start
    if 'week_start' in df.columns:
        df['month'] = pd.to_datetime(df['week_start']).dt.month
        df['week_of_year'] = pd.to_datetime(df['week_start']).dt.isocalendar().week.astype(int)
    
    # Cyclical encoding
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Seasonal flags
    if 'month' in df.columns:
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    
    # Weekend ratio flag
    if 'weekend_ratio' in df.columns:
        df['is_weekend_heavy'] = (df['weekend_ratio'] > 0.4).astype(int)
    else:
        df['is_weekend_heavy'] = 0
    
    return df


def engineer_price_features(
    df: pd.DataFrame,
    peer_stats: Optional[PeerStats] = None
) -> pd.DataFrame:
    """Add price-relative features."""
    df = df.copy()
    
    # If peer_stats provided, use them
    if peer_stats is not None and peer_stats.n_peers > 0:
        df['peer_median_price'] = peer_stats.median_price
        df['peer_median_occupancy'] = peer_stats.median_occupancy
        df['peer_mean_occupancy'] = peer_stats.mean_occupancy
        df['peer_n'] = peer_stats.n_peers
        df['peer_price_p25'] = peer_stats.price_p25
        df['peer_price_p75'] = peer_stats.price_p75
    
    # Price ratios (key elasticity signals)
    if 'candidate_price' in df.columns and 'peer_median_price' in df.columns:
        df['price_vs_peer_median'] = df['candidate_price'] / df['peer_median_price'].clip(lower=1)
        df['price_vs_peer_p25'] = df['candidate_price'] / df['peer_price_p25'].clip(lower=1)
        df['price_vs_peer_p75'] = df['candidate_price'] / df['peer_price_p75'].clip(lower=1)
    
    return df


# =============================================================================
# OCCUPANCY PREDICTOR
# =============================================================================

class OccupancyPredictor:
    """
    Predicts occupancy given hotel features and candidate price.
    
    The key feature is price_vs_peer_median - this is what the model uses
    to learn elasticity. A hotel priced 20% above peers will have lower
    predicted occupancy than one priced at peer median.
    
    Usage:
        predictor = OccupancyPredictor(peer_method='geographic')
        predictor.fit(train_df)
        
        occ = predictor.predict(hotel_features, candidate_price=100, target_date)
        revpar = candidate_price * occ
    """
    
    def __init__(
        self,
        peer_method: str = 'geographic',
        geo_radius_km: float = DEFAULT_RADIUS_KM,
        knn_k: int = DEFAULT_K
    ):
        """
        Initialize occupancy predictor.
        
        Args:
            peer_method: 'geographic', 'knn', or 'segment'
            geo_radius_km: Radius for geographic peers
            knn_k: K for KNN peers
        """
        self.peer_method = peer_method
        self.peer_matcher = UnifiedPeerMatcher(
            geo_radius_km=geo_radius_km,
            knn_k=knn_k
        )
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False
        self.metrics = None
        self.train_df = None
    
    def _get_feature_cols(self) -> List[str]:
        """Get all feature columns."""
        return (
            [f for f in HOTEL_FEATURES] +
            [f for f in TEMPORAL_FEATURES] +
            [f for f in PRICE_FEATURES] +
            [f for f in PEER_FEATURES]
        )
    
    def _prepare_training_data(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with peer features.
        
        Target: rooms_booked (not occupancy_rate) to avoid capacity artifacts.
        For training, candidate_price = actual price charged.
        """
        df = df.copy()
        
        # Load distance features
        df = self._add_distance_features(df)
        
        # Add derived geographic features
        df = self._add_geographic_features(df)
        
        # Add revpar if not present
        if 'revpar' not in df.columns:
            df['revpar'] = df['avg_price'] * df['occupancy_rate']
        
        # Calculate rooms_booked as target (avoids capacity artifacts)
        if 'rooms_booked' not in df.columns:
            df['rooms_booked'] = (df['occupancy_rate'] * df['total_rooms']).round()
        
        # Engineer temporal features
        df = engineer_temporal_features(df)
        
        # For training, candidate_price = actual price
        df['candidate_price'] = df['avg_price']
        
        # Add log room size if not present
        if 'log_room_size' not in df.columns:
            if 'avg_room_size' in df.columns:
                df['log_room_size'] = np.log1p(df['avg_room_size'].fillna(25))
            else:
                df['log_room_size'] = np.log1p(25)
        
        # Add log total rooms
        if 'log_total_rooms' not in df.columns:
            df['log_total_rooms'] = np.log1p(df['total_rooms'].fillna(10))
        
        # Fill missing hotel features
        for feat in HOTEL_FEATURES:
            if feat not in df.columns:
                df[feat] = 0
            df[feat] = df[feat].fillna(0)
        
        # Compute peer features for each row
        # This is expensive - we'll batch by week
        if verbose:
            print("Computing peer features...")
        
        peer_features_list = []
        
        for week in df['week_start'].unique():
            week_df = df[df['week_start'] == week]
            week_date = pd.Timestamp(week).date()
            
            for _, row in week_df.iterrows():
                hotel_id = row['hotel_id']
                lat = row['latitude']
                lon = row['longitude']
                
                # Get peer stats using selected method
                all_stats = self.peer_matcher.get_all_peer_stats(
                    hotel_id, lat, lon, week_date, df
                )
                stats = all_stats.get(self.peer_method, all_stats.get('geographic'))
                
                peer_features_list.append({
                    'hotel_id': hotel_id,
                    'week_start': week,
                    'peer_median_price': stats.median_price if stats.n_peers > 0 else row['avg_price'],
                    'peer_median_occupancy': stats.median_occupancy if stats.n_peers > 0 else 0.65,
                    'peer_mean_occupancy': stats.mean_occupancy if stats.n_peers > 0 else 0.65,
                    'peer_n': stats.n_peers,
                    'peer_price_p25': stats.price_p25 if stats.n_peers > 0 else row['avg_price'] * 0.8,
                    'peer_price_p75': stats.price_p75 if stats.n_peers > 0 else row['avg_price'] * 1.2,
                })
        
        peer_df = pd.DataFrame(peer_features_list)
        
        # Merge peer features - handle case where merge might create duplicate columns
        if len(peer_df) > 0:
            df = df.merge(peer_df, on=['hotel_id', 'week_start'], how='left', suffixes=('', '_peer'))
            # If columns got duplicated, drop the peer suffix versions
            for col in ['peer_median_price', 'peer_median_occupancy', 'peer_mean_occupancy', 
                       'peer_n', 'peer_price_p25', 'peer_price_p75']:
                if f'{col}_peer' in df.columns:
                    df = df.drop(columns=[f'{col}_peer'])
        
        # Fill missing peer features
        # Initialize all peer columns if not present
        if 'peer_median_price' not in df.columns:
            df['peer_median_price'] = df['avg_price']
        if 'peer_median_occupancy' not in df.columns:
            df['peer_median_occupancy'] = 0.65
        if 'peer_mean_occupancy' not in df.columns:
            df['peer_mean_occupancy'] = 0.65
        if 'peer_n' not in df.columns:
            df['peer_n'] = 0
        if 'peer_price_p25' not in df.columns:
            df['peer_price_p25'] = df['avg_price'] * 0.8
        if 'peer_price_p75' not in df.columns:
            df['peer_price_p75'] = df['avg_price'] * 1.2
        
        # Fill any remaining NaN
        df['peer_median_price'] = df['peer_median_price'].fillna(df['avg_price'])
        df['peer_median_occupancy'] = df['peer_median_occupancy'].fillna(0.65)
        df['peer_mean_occupancy'] = df['peer_mean_occupancy'].fillna(0.65)
        df['peer_n'] = df['peer_n'].fillna(0)
        df['peer_price_p25'] = df['peer_price_p25'].fillna(df['avg_price'] * 0.8)
        df['peer_price_p75'] = df['peer_price_p75'].fillna(df['avg_price'] * 1.2)
        
        # Compute price ratios
        df = engineer_price_features(df)
        
        # Fill any remaining NaN in price ratios
        df['price_vs_peer_median'] = df['price_vs_peer_median'].fillna(1.0)
        df['price_vs_peer_p25'] = df['price_vs_peer_p25'].fillna(1.0)
        df['price_vs_peer_p75'] = df['price_vs_peer_p75'].fillna(1.0)
        
        # Get feature columns
        self.feature_cols = self._get_feature_cols()
        
        # Filter to available features
        available_features = [f for f in self.feature_cols if f in df.columns]
        self.feature_cols = available_features
        
        # Fill any remaining NaN
        for col in self.feature_cols:
            df[col] = df[col].fillna(0)
        
        X = df[self.feature_cols].copy()
        
        # Target: rooms_booked (not occupancy_rate)
        # This avoids capacity artifacts (small hotels have discrete occupancy)
        y = df['rooms_booked'].clip(0, df['total_rooms']).copy()
        
        return X, y, df
    
    def _add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pre-computed distance features from CSV.
        
        Features: distance_from_madrid, distance_from_coast
        """
        try:
            from pathlib import Path
            dist_path = Path(__file__).parents[2] / DISTANCE_FEATURES_PATH
            if not dist_path.exists():
                # Try relative to workspace
                dist_path = Path(DISTANCE_FEATURES_PATH)
            
            if dist_path.exists():
                dist_df = pd.read_csv(dist_path)
                # Merge on hotel_id
                df = df.merge(
                    dist_df[['hotel_id', 'distance_from_madrid', 'distance_from_coast']],
                    on='hotel_id',
                    how='left',
                    suffixes=('', '_dist')
                )
                # Rename for clarity
                df['dist_coast_km'] = df['distance_from_coast'].fillna(100)
                df['dist_madrid_km'] = df['distance_from_madrid'].fillna(300)
        except Exception as e:
            # Fallback if file not found
            df['dist_coast_km'] = 100  # Default: inland
            df['dist_madrid_km'] = 300  # Default: far from Madrid
        
        return df
    
    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived geographic features.
        
        Uses distance features to compute:
        - is_coastal: within 20km of coast
        - is_madrid_metro: within 50km of Madrid (and not coastal)
        - market_segment: coastal/madrid_metro/provincial
        - city_standardized: normalized city name
        """
        df = df.copy()
        
        # Coastal flag (within 20km)
        if 'is_coastal' not in df.columns:
            df['is_coastal'] = (df.get('dist_coast_km', 100) <= 20).astype(int)
        
        # Madrid metro flag (within 50km, not coastal)
        if 'is_madrid_metro' not in df.columns:
            df['is_madrid_metro'] = (
                (df.get('dist_madrid_km', 300) <= 50) & 
                (df.get('dist_coast_km', 100) > 20)
            ).astype(int)
        
        # Market segment
        if 'market_segment' not in df.columns:
            df['market_segment'] = df.apply(
                lambda row: get_market_segment(
                    row.get('dist_coast_km', 100),
                    row.get('dist_madrid_km', 300)
                ),
                axis=1
            )
        
        # Standardized city
        if 'city_standardized' not in df.columns and 'city' in df.columns:
            df['city_standardized'] = df['city'].apply(
                lambda x: standardize_city(str(x)) if pd.notna(x) else 'other'
            )
        
        return df
    
    def fit(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True
    ) -> 'OccupancyPredictor':
        """
        Fit the occupancy model.
        
        Args:
            train_df: Training data
            verbose: Print progress
        
        Returns:
            self
        """
        if verbose:
            print(f"Training occupancy predictor (peer_method={self.peer_method})...")
        
        # Fit peer matcher first
        self.peer_matcher.fit(train_df)
        
        # Prepare training data
        X, y, df = self._prepare_training_data(train_df, verbose=verbose)
        
        if verbose:
            print(f"Training on {len(X):,} samples with {len(self.feature_cols)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select best model from candidates
        if verbose:
            print("\nModel selection:")
        
        self.model, model_name, all_scores = select_best_model(
            X_scaled, y.values, cv_folds=5, verbose=verbose
        )
        
        # Fit final model on all data
        if verbose:
            print(f"\nFitting {model_name} on full dataset...")
        
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred_train = self.model.predict(X_scaled)
        r2_train = 1 - np.sum((y - y_pred_train)**2) / np.sum((y - y.mean())**2)
        mae_train = np.mean(np.abs(y - y_pred_train))
        
        # Get CV score for best model
        best_cv = all_scores.get(model_name, {'mean': 0, 'std': 0})
        
        self.metrics = OccupancyModelMetrics(
            r2_train=r2_train,
            r2_cv=best_cv['mean'],
            mae_train=mae_train,
            mae_cv=best_cv['std'],
            n_samples=len(X),
            model_name=model_name
        )
        
        if verbose:
            print(f"  R² (train): {r2_train:.4f}")
            print(f"  R² (CV):    {best_cv['mean']:.4f} ± {best_cv['std']:.4f}")
            print(f"  MAE (train): {mae_train:.4f}")
        
        self.train_df = train_df
        self.is_fitted = True
        return self
    
    def predict(
        self,
        hotel_id: int,
        candidate_price: float,
        target_date: date,
        history_df: Optional[pd.DataFrame] = None
    ) -> OccupancyPrediction:
        """
        Predict occupancy for a hotel at a candidate price.
        
        Args:
            hotel_id: Hotel ID
            candidate_price: Price to evaluate
            target_date: Target date
            history_df: Historical data (uses train_df if None)
        
        Returns:
            OccupancyPrediction
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if history_df is None:
            history_df = self.train_df
        
        # Get hotel features
        hotel_data = history_df[history_df['hotel_id'] == hotel_id]
        if len(hotel_data) == 0:
            # Cold start - return market average
            return OccupancyPrediction(
                predicted_occupancy=0.65,
                predicted_revpar=candidate_price * 0.65,
                candidate_price=candidate_price,
                confidence=0.1,
                features_used={}
            )
        
        hotel_row = hotel_data.iloc[-1].copy()  # Most recent
        
        # Get peer stats
        lat = hotel_row['latitude']
        lon = hotel_row['longitude']
        all_stats = self.peer_matcher.get_all_peer_stats(
            hotel_id, lat, lon, target_date, history_df
        )
        peer_stats = all_stats.get(self.peer_method, all_stats.get('geographic'))
        
        # Build feature vector
        features = {}
        
        # Hotel features
        for feat in HOTEL_FEATURES:
            features[feat] = hotel_row.get(feat, 0)
        
        # Ensure geographic features are present
        features['log_total_rooms'] = np.log1p(hotel_row.get('total_rooms', 10))
        features['dist_coast_km'] = hotel_row.get('dist_coast_km', hotel_row.get('distance_from_coast', 100))
        features['dist_madrid_km'] = hotel_row.get('dist_madrid_km', hotel_row.get('distance_from_madrid', 300))
        features['is_coastal'] = 1 if features['dist_coast_km'] <= 20 else 0
        features['is_madrid_metro'] = 1 if (features['dist_madrid_km'] <= 50 and features['dist_coast_km'] > 20) else 0
        
        # Temporal features
        features['month'] = target_date.month
        features['month_sin'] = np.sin(2 * np.pi * target_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * target_date.month / 12)
        features['week_of_year'] = target_date.isocalendar()[1]
        features['is_summer'] = 1 if target_date.month in [6, 7, 8] else 0
        features['is_weekend_heavy'] = hotel_row.get('is_weekend_heavy', 0)
        
        # Price features
        features['candidate_price'] = candidate_price
        peer_median = peer_stats.median_price if peer_stats.n_peers > 0 else candidate_price
        features['price_vs_peer_median'] = candidate_price / max(peer_median, 1)
        features['price_vs_peer_p25'] = candidate_price / max(peer_stats.price_p25, 1) if peer_stats.n_peers > 0 else 1.0
        features['price_vs_peer_p75'] = candidate_price / max(peer_stats.price_p75, 1) if peer_stats.n_peers > 0 else 1.0
        
        # Peer features
        features['peer_median_occupancy'] = peer_stats.median_occupancy if peer_stats.n_peers > 0 else 0.65
        features['peer_mean_occupancy'] = peer_stats.mean_occupancy if peer_stats.n_peers > 0 else 0.65
        features['peer_n'] = peer_stats.n_peers
        
        # Create feature vector
        X = pd.DataFrame([{col: features.get(col, 0) for col in self.feature_cols}])
        X_scaled = self.scaler.transform(X)
        
        # Predict rooms_booked, then convert to occupancy
        predicted_rooms = self.model.predict(X_scaled)[0]
        total_rooms = hotel_row.get('total_rooms', 10)
        predicted_rooms = np.clip(predicted_rooms, 0, total_rooms)
        predicted_occ = predicted_rooms / max(total_rooms, 1)
        predicted_occ = np.clip(predicted_occ, 0.01, 0.99)
        
        # Confidence based on peer count and data quality
        confidence = min(peer_stats.n_peers / 10, 1.0) if peer_stats.n_peers > 0 else 0.3
        
        return OccupancyPrediction(
            predicted_occupancy=predicted_occ,
            predicted_revpar=candidate_price * predicted_occ,
            candidate_price=candidate_price,
            confidence=confidence,
            features_used=features
        )
    
    def predict_at_prices(
        self,
        hotel_id: int,
        prices: List[float],
        target_date: date,
        history_df: Optional[pd.DataFrame] = None
    ) -> List[OccupancyPrediction]:
        """
        Predict occupancy at multiple price points.
        
        Args:
            hotel_id: Hotel ID
            prices: List of prices to evaluate
            target_date: Target date
            history_df: Historical data
        
        Returns:
            List of OccupancyPrediction
        """
        return [
            self.predict(hotel_id, price, target_date, history_df)
            for price in prices
        ]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.zeros(len(self.feature_cols))
        
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save to (default: outputs/models/occupancy_model.pkl)
        
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        save_path = path or MODEL_SAVE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'peer_method': self.peer_method,
            'metrics': self.metrics,
            'peer_matcher': self.peer_matcher,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {save_path}")
        return save_path
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'OccupancyPredictor':
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load from (default: outputs/models/occupancy_model.pkl)
        
        Returns:
            Loaded OccupancyPredictor
        """
        load_path = path or MODEL_SAVE_PATH
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        predictor = cls(peer_method=state['peer_method'])
        predictor.model = state['model']
        predictor.scaler = state['scaler']
        predictor.feature_cols = state['feature_cols']
        predictor.metrics = state['metrics']
        predictor.peer_matcher = state['peer_matcher']
        predictor.is_fitted = True
        
        print(f"Model loaded from {load_path}")
        return predictor
    
    def predict_batch(
        self,
        test_df: pd.DataFrame,
        candidate_prices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Vectorized batch prediction for multiple hotels.
        
        Args:
            test_df: DataFrame with hotel features
            candidate_prices: Optional array of prices (uses avg_price if None)
        
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        df = test_df.copy()
        
        # Use actual price if no candidate provided
        if candidate_prices is None:
            df['candidate_price'] = df['avg_price']
        else:
            df['candidate_price'] = candidate_prices
        
        # Add required features
        df = self._add_distance_features(df)
        df = self._add_geographic_features(df)
        df = engineer_temporal_features(df)
        
        # Add log features
        if 'log_room_size' not in df.columns:
            df['log_room_size'] = np.log1p(df.get('avg_room_size', 25))
        if 'log_total_rooms' not in df.columns:
            df['log_total_rooms'] = np.log1p(df.get('total_rooms', 10))
        
        # Compute peer features (vectorized by group)
        # Use pre-computed peer medians by geographic bin
        lat_bin, lon_bin = 0.1, 0.12
        df['lat_bin'] = (df['latitude'] / lat_bin).astype(int)
        df['lon_bin'] = (df['longitude'] / lon_bin).astype(int)
        
        peer_stats = df.groupby(['lat_bin', 'lon_bin']).agg({
            'avg_price': 'median',
            'occupancy_rate': 'median'
        }).rename(columns={
            'avg_price': 'peer_median_price',
            'occupancy_rate': 'peer_median_occupancy'
        })
        
        df = df.merge(peer_stats, on=['lat_bin', 'lon_bin'], how='left', suffixes=('', '_peer'))
        
        # Fill missing peer features
        market_price = df['avg_price'].median()
        df['peer_median_price'] = df['peer_median_price'].fillna(market_price)
        df['peer_median_occupancy'] = df['peer_median_occupancy'].fillna(0.65)
        df['peer_mean_occupancy'] = df['peer_median_occupancy']
        df['peer_n'] = df.groupby(['lat_bin', 'lon_bin'])['hotel_id'].transform('count')
        
        # Price ratios
        df['price_vs_peer_median'] = df['candidate_price'] / df['peer_median_price'].clip(lower=1)
        df['price_vs_peer_p25'] = df['price_vs_peer_median'] * 0.8  # Approximate
        df['price_vs_peer_p75'] = df['price_vs_peer_median'] * 1.2
        
        # Fill any missing feature columns
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        # Predict
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        predicted_rooms = self.model.predict(X_scaled)
        total_rooms = df['total_rooms'].fillna(10).values
        predicted_rooms = np.clip(predicted_rooms, 0, total_rooms)
        
        df['predicted_occupancy'] = predicted_rooms / np.maximum(total_rooms, 1)
        df['predicted_occupancy'] = np.clip(df['predicted_occupancy'], 0.01, 0.99)
        df['predicted_revpar'] = df['candidate_price'] * df['predicted_occupancy']
        
        return df


# =============================================================================
# PRICE OPTIMIZER
# =============================================================================

class PriceOptimizer:
    """
    Finds the RevPAR-maximizing price using the occupancy model.
    
    Grid search over price range, select price that maximizes
    price * predicted_occupancy.
    """
    
    def __init__(
        self,
        occupancy_model: OccupancyPredictor,
        price_min: float = 30.0,
        price_max: float = 400.0,
        n_steps: int = 30
    ):
        """
        Initialize price optimizer.
        
        Args:
            occupancy_model: Fitted OccupancyPredictor
            price_min: Minimum price to consider
            price_max: Maximum price to consider
            n_steps: Number of price points to evaluate
        """
        self.occupancy_model = occupancy_model
        self.price_min = price_min
        self.price_max = price_max
        self.n_steps = n_steps
    
    def find_optimal_price(
        self,
        hotel_id: int,
        target_date: date,
        history_df: Optional[pd.DataFrame] = None,
        price_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        Find the price that maximizes predicted RevPAR.
        
        Args:
            hotel_id: Hotel ID
            target_date: Target date
            history_df: Historical data
            price_range: Optional (min, max) to override defaults
        
        Returns:
            Tuple of (optimal_price, optimal_revpar, [(price, revpar), ...])
        """
        p_min = price_range[0] if price_range else self.price_min
        p_max = price_range[1] if price_range else self.price_max
        
        prices = np.linspace(p_min, p_max, self.n_steps)
        
        results = []
        best_price, best_revpar = None, -np.inf
        
        for price in prices:
            pred = self.occupancy_model.predict(
                hotel_id, price, target_date, history_df
            )
            revpar = pred.predicted_revpar
            results.append((price, revpar))
            
            if revpar > best_revpar:
                best_revpar = revpar
                best_price = price
        
        return best_price, best_revpar, results


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test occupancy predictor."""
    from src.models.evaluation.time_backtest import BacktestConfig, load_hotel_week_data
    
    print("=" * 60)
    print("OCCUPANCY PREDICTOR TEST")
    print("=" * 60)
    
    config = BacktestConfig()
    
    print("\nLoading data...")
    train_df = load_hotel_week_data(config, split='train')
    
    # Use a subset for faster testing
    train_subset = train_df.head(5000)
    
    print("\nTraining occupancy predictor (geographic peers)...")
    predictor = OccupancyPredictor(peer_method='geographic')
    predictor.fit(train_subset, verbose=True)
    
    print("\nFeature importance:")
    print(predictor.get_feature_importance().head(10).to_string())
    
    # Test predictions
    print("\n" + "=" * 60)
    print("PRICE-OCCUPANCY CURVE TEST")
    print("=" * 60)
    
    target_date = config.test_start
    sample_hotel = train_df.iloc[0]['hotel_id']
    
    print(f"\nHotel {sample_hotel} - Price vs Predicted Occupancy:")
    
    optimizer = PriceOptimizer(predictor)
    optimal_price, optimal_revpar, curve = optimizer.find_optimal_price(
        sample_hotel, target_date, train_df
    )
    
    print(f"\n{'Price':>8} {'Occ':>8} {'RevPAR':>10}")
    print("-" * 30)
    for price, revpar in curve[::5]:  # Every 5th point
        occ = revpar / price
        print(f"€{price:>7.0f} {occ:>7.1%} €{revpar:>9.0f}")
    
    print(f"\nOptimal: €{optimal_price:.0f} → RevPAR €{optimal_revpar:.0f}")
    
    print("\n✓ Occupancy predictor working!")


if __name__ == "__main__":
    main()

