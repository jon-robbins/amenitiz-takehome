"""
Quality Model (Path B) for cold-start hotels.

This model predicts the quality premium/discount a hotel should charge
relative to its peers. It answers: "How much more or less is this hotel
worth compared to nearby competitors?"

Target: Y = actual_price / peer_median (ratio ~0.85 to 1.15)

Features (quality focus):
- amenities_score (children, pets, events, smoking allowed)
- view_quality_ordinal (0-3 scale)
- log_room_size
- room_capacity_pax
- dist_center_km (centrality within city)
- total_rooms (hotel size)

Key insight: This model learns the quality premium based on observable
product features. The peer_median anchor captures location/season effects.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.models.anchors import get_peer_anchor, MIN_PEERS
from src.features.engineering import (
    VIEW_QUALITY_MAP,
    haversine_distance,
    standardize_city
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Prediction bounds (multiplier on peer-median)
MIN_MULTIPLIER = 0.75
MAX_MULTIPLIER = 1.25

# Feature columns for quality model
QUALITY_FEATURES = [
    'amenities_score',
    'view_quality_ordinal',
    'log_room_size',
    'room_capacity_pax',
    'dist_center_km',
    'log_total_rooms',
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QualityPrediction:
    """Result of quality model prediction."""
    multiplier: float  # 0.85 to 1.15 typically
    recommended_price: float
    peer_anchor: float
    confidence: float
    n_peers: int
    features_used: Dict[str, float]


@dataclass
class QualityModelMetrics:
    """Training metrics for quality model."""
    r2_train: float
    r2_cv: float
    mae_train: float
    mae_cv: float
    n_samples: int
    n_features: int


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer quality features for the model.
    
    Args:
        df: DataFrame with hotel product features
    
    Returns:
        DataFrame with quality features added
    """
    df = df.copy()
    
    # Amenities score (if not already computed)
    if 'amenities_score' not in df.columns:
        amenity_cols = ['children_allowed', 'pets_allowed', 'events_allowed', 'smoking_allowed']
        existing = [c for c in amenity_cols if c in df.columns]
        df['amenities_score'] = df[existing].fillna(0).sum(axis=1) if existing else 0
    
    # View quality ordinal
    if 'view_quality_ordinal' not in df.columns:
        if 'room_view' in df.columns:
            df['view_quality_ordinal'] = df['room_view'].map(
                lambda x: VIEW_QUALITY_MAP.get(str(x).lower(), 0)
            )
        else:
            df['view_quality_ordinal'] = 0
    
    # Log room size
    if 'log_room_size' not in df.columns:
        size_col = 'avg_room_size' if 'avg_room_size' in df.columns else 'room_size'
        if size_col in df.columns:
            df['log_room_size'] = np.log1p(df[size_col].fillna(25))
        else:
            df['log_room_size'] = np.log1p(25)  # Default 25 sqm
    
    # Room capacity
    if 'room_capacity_pax' not in df.columns:
        df['room_capacity_pax'] = df.get('max_occupancy', 2)
    
    # Distance to city center
    if 'dist_center_km' not in df.columns:
        # Calculate city centers
        if 'city' in df.columns:
            df['city_std'] = df['city'].apply(
                lambda x: standardize_city(str(x)) if pd.notna(x) else 'other'
            )
            city_centers = df.groupby('city_std').agg({
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            city_centers.columns = ['city_std', 'city_lat', 'city_lon']
            df = df.merge(city_centers, on='city_std', how='left')
            
            df['dist_center_km'] = haversine_distance(
                df['latitude'].values,
                df['longitude'].values,
                df['city_lat'].fillna(df['latitude']).values,
                df['city_lon'].fillna(df['longitude']).values
            )
        else:
            df['dist_center_km'] = 0
    
    # Log total rooms
    if 'log_total_rooms' not in df.columns:
        if 'total_rooms' in df.columns:
            df['log_total_rooms'] = np.log1p(df['total_rooms'].fillna(10))
        else:
            df['log_total_rooms'] = np.log1p(10)
    
    return df


# =============================================================================
# QUALITY MODEL
# =============================================================================

class QualityModel:
    """
    Path B model: Predicts quality premium for cold-start hotels.
    
    Usage:
        model = QualityModel()
        model.fit(train_df)
        
        prediction = model.predict(hotel_features, target_week, peer_df)
    """
    
    def __init__(
        self,
        min_peers: int = MIN_PEERS,
        use_lightgbm: bool = True
    ):
        """
        Initialize quality model.
        
        Args:
            min_peers: Minimum peers required for peer anchor
            use_lightgbm: Whether to use LightGBM or ElasticNet
        """
        self.min_peers = min_peers
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = QUALITY_FEATURES
        self.is_fitted = False
        self.metrics = None
    
    def _prepare_training_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Prepare training data with target variable.
        
        Target: price_ratio = actual_price / peer_median
        """
        df = df.copy()
        
        # Engineer features
        df = engineer_quality_features(df)
        
        # Filter to rows with peer prices
        valid_mask = (
            df['peer_median_price'].notna() &
            (df['peer_median_price'] > 0) &
            (df['n_peers'] >= self.min_peers)
        )
        df = df[valid_mask].copy()
        
        # Target: price ratio vs peers
        df['price_ratio'] = df['avg_price'] / df['peer_median_price']
        
        # Filter reasonable ratios
        df = df[
            (df['price_ratio'] > 0.5) &
            (df['price_ratio'] < 2.0)
        ].copy()
        
        # Fill missing features
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        X = df[self.feature_cols].copy()
        y = df['price_ratio'].copy()
        
        return X, y, df
    
    def fit(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True
    ) -> 'QualityModel':
        """
        Train the quality model.
        
        Args:
            train_df: Training data with hotel features and peer_median_price
            verbose: Print training progress
        
        Returns:
            self
        """
        if verbose:
            print("Preparing training data...")
        
        X, y, df = self._prepare_training_data(train_df)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        if verbose:
            print(f"Training on {len(X):,} samples from {df['hotel_id'].nunique():,} hotels")
            print(f"Mean price ratio: {y.mean():.3f} (std: {y.std():.3f})")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.use_lightgbm:
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=100,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1
            )
        else:
            self.model = ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        
        # Cross-validation
        if verbose:
            print("Running 5-fold cross-validation...")
        
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=5, scoring='r2'
        )
        
        # Fit final model
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred_train = self.model.predict(X_scaled)
        r2_train = 1 - np.sum((y - y_pred_train)**2) / np.sum((y - y.mean())**2)
        mae_train = np.mean(np.abs(y - y_pred_train))
        
        self.metrics = QualityModelMetrics(
            r2_train=r2_train,
            r2_cv=np.mean(cv_scores),
            mae_train=mae_train,
            mae_cv=np.std(cv_scores),
            n_samples=len(X),
            n_features=len(self.feature_cols)
        )
        
        if verbose:
            print(f"  R² (train): {r2_train:.4f}")
            print(f"  R² (CV):    {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"  MAE (train): {mae_train:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        latitude: float,
        longitude: float,
        target_week: date,
        peer_df: pd.DataFrame,
        hotel_features: Optional[Dict] = None
    ) -> QualityPrediction:
        """
        Predict price for a cold-start hotel.
        
        Args:
            latitude: Hotel latitude
            longitude: Hotel longitude
            target_week: Target week
            peer_df: Peer data for anchor calculation
            hotel_features: Optional dict with product features
        
        Returns:
            QualityPrediction with recommended price
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get peer anchor
        anchor_result = get_peer_anchor(latitude, longitude, target_week, peer_df)
        
        if anchor_result.anchor_price <= 0:
            return QualityPrediction(
                multiplier=1.0,
                recommended_price=100.0,  # Fallback
                peer_anchor=100.0,
                confidence=0.0,
                n_peers=0,
                features_used={}
            )
        
        # Prepare features
        features = hotel_features or {}
        
        feature_row = {
            'amenities_score': features.get('amenities_score', 0),
            'view_quality_ordinal': features.get('view_quality_ordinal', 0),
            'log_room_size': np.log1p(features.get('room_size', 25)),
            'room_capacity_pax': features.get('room_capacity_pax', 2),
            'dist_center_km': features.get('dist_center_km', 0),
            'log_total_rooms': np.log1p(features.get('total_rooms', 10)),
        }
        
        X = pd.DataFrame([feature_row])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Predict multiplier
        multiplier = self.model.predict(X_scaled)[0]
        multiplier = np.clip(multiplier, MIN_MULTIPLIER, MAX_MULTIPLIER)
        
        recommended_price = anchor_result.anchor_price * multiplier
        
        # Confidence based on peer count
        peer_confidence = min(anchor_result.n_observations / 20, 1.0)
        confidence = anchor_result.confidence * peer_confidence
        
        return QualityPrediction(
            multiplier=multiplier,
            recommended_price=recommended_price,
            peer_anchor=anchor_result.anchor_price,
            confidence=confidence,
            n_peers=anchor_result.n_observations,
            features_used=feature_row
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            importances = np.zeros(len(self.feature_cols))
        
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================

def train_quality_model(
    train_df: pd.DataFrame,
    verbose: bool = True
) -> QualityModel:
    """
    Convenience function to train quality model.
    
    Args:
        train_df: Training data with peer_median_price
        verbose: Print progress
    
    Returns:
        Fitted QualityModel
    """
    model = QualityModel()
    model.fit(train_df, verbose=verbose)
    return model


# =============================================================================
# TESTING / DEMO
# =============================================================================

def main():
    """Demo quality model training and prediction."""
    from src.models.evaluation.time_backtest import BacktestConfig, load_hotel_week_data
    
    print("=" * 60)
    print("QUALITY MODEL (PATH B) DEMO")
    print("=" * 60)
    
    config = BacktestConfig()
    
    print("\nLoading training data...")
    train_df = load_hotel_week_data(config, split="train")
    
    print("\nTraining quality model...")
    model = train_quality_model(train_df)
    
    print("\nFeature importance:")
    print(model.get_feature_importance().to_string())
    
    # Sample predictions for hypothetical cold-start hotels
    print("\n" + "=" * 60)
    print("SAMPLE COLD-START PREDICTIONS")
    print("=" * 60)
    
    target_week = config.test_start
    
    # Madrid hotel with good amenities
    pred = model.predict(
        latitude=40.4168,
        longitude=-3.7038,
        target_week=target_week,
        peer_df=train_df,
        hotel_features={
            'amenities_score': 3,
            'view_quality_ordinal': 2,
            'room_size': 35,
            'room_capacity_pax': 4,
            'total_rooms': 50
        }
    )
    print(f"\nMadrid luxury hotel:")
    print(f"  Peer anchor: €{pred.peer_anchor:.0f} ({pred.n_peers} peers)")
    print(f"  Quality multiplier: {pred.multiplier:.2f}x")
    print(f"  Recommended: €{pred.recommended_price:.0f}")
    
    # Budget coastal hotel
    pred = model.predict(
        latitude=36.7213,
        longitude=-4.4214,  # Malaga
        target_week=target_week,
        peer_df=train_df,
        hotel_features={
            'amenities_score': 1,
            'view_quality_ordinal': 0,
            'room_size': 20,
            'room_capacity_pax': 2,
            'total_rooms': 15
        }
    )
    print(f"\nMalaga budget hotel:")
    print(f"  Peer anchor: €{pred.peer_anchor:.0f} ({pred.n_peers} peers)")
    print(f"  Quality multiplier: {pred.multiplier:.2f}x")
    print(f"  Recommended: €{pred.recommended_price:.0f}")


if __name__ == "__main__":
    main()

