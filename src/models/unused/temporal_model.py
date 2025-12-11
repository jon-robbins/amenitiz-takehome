"""
Temporal Model (Path A) for hotels with history.

This model predicts WHEN a hotel should adjust prices relative to their
own historical median. It answers: "Given this week/month/season, should
the hotel charge more or less than usual?"

Target: Y = actual_price / self_median (ratio ~0.8 to 1.3)

Features (temporal focus):
- month_sin, month_cos (seasonality)
- week_of_year (fine-grained seasonality)
- is_holiday, holiday_proximity_days
- market_seasonality_index (peer prices this week vs annual avg)
- hotel-specific DOW pattern (if available)

Key insight: This model doesn't predict absolute prices - it predicts
the temporal adjustment factor. The self_median anchor captures the 
hotel's base price level.
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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.models.anchors import (
    get_self_anchor,
    calculate_seasonality_indices,
    MIN_WEEKS_FOR_SELF_ANCHOR
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Prediction bounds (multiplier on self-median)
MIN_MULTIPLIER = 0.7
MAX_MULTIPLIER = 1.4

# Feature columns for temporal model
TEMPORAL_FEATURES = [
    'month_sin',
    'month_cos',
    'week_of_year',
    'market_seasonality_index',
    'is_peak_season',    # July-August
    'is_shoulder_season', # Apr-Jun, Sep-Oct
    'is_low_season',     # Nov-Mar
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TemporalPrediction:
    """Result of temporal model prediction."""
    multiplier: float  # 0.8 to 1.3 typically
    recommended_price: float
    self_anchor: float
    confidence: float
    features_used: Dict[str, float]


@dataclass
class TemporalModelMetrics:
    """Training metrics for temporal model."""
    r2_train: float
    r2_cv: float
    mae_train: float
    mae_cv: float
    n_samples: int
    n_features: int


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal features for the model.
    
    Args:
        df: DataFrame with week_start column
    
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Extract components
    df['month'] = df['week_start'].dt.month
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(int)
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season flags
    df['is_peak_season'] = df['month'].isin([7, 8]).astype(int)
    df['is_shoulder_season'] = df['month'].isin([4, 5, 6, 9, 10]).astype(int)
    df['is_low_season'] = df['month'].isin([1, 2, 3, 11, 12]).astype(int)
    
    return df


def add_market_seasonality(
    df: pd.DataFrame,
    seasonality_indices: Dict[int, float]
) -> pd.DataFrame:
    """
    Add market seasonality index to DataFrame.
    
    Args:
        df: DataFrame with week_of_year
        seasonality_indices: Dict mapping week_of_year to index
    
    Returns:
        DataFrame with market_seasonality_index added
    """
    df = df.copy()
    df['market_seasonality_index'] = df['week_of_year'].map(
        lambda w: seasonality_indices.get(w, 1.0)
    )
    return df


# =============================================================================
# TEMPORAL MODEL
# =============================================================================

class TemporalModel:
    """
    Path A model: Predicts temporal price adjustment for hotels with history.
    
    Usage:
        model = TemporalModel()
        model.fit(train_df)
        
        prediction = model.predict(hotel_id, target_week, history_df)
    """
    
    def __init__(
        self,
        min_weeks_history: int = MIN_WEEKS_FOR_SELF_ANCHOR,
        use_lightgbm: bool = True
    ):
        """
        Initialize temporal model.
        
        Args:
            min_weeks_history: Minimum weeks required to use this model
            use_lightgbm: Whether to use LightGBM (faster) or sklearn GBM
        """
        self.min_weeks_history = min_weeks_history
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        
        self.model = None
        self.scaler = StandardScaler()
        self.seasonality_indices = None
        self.feature_cols = TEMPORAL_FEATURES
        self.is_fitted = False
        self.metrics = None
    
    def _prepare_training_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with target variable.
        
        Target: price_ratio = actual_price / self_median
        """
        df = df.copy()
        
        # Engineer features
        df = engineer_temporal_features(df)
        
        # Calculate seasonality indices from training data
        self.seasonality_indices = calculate_seasonality_indices(df)
        df = add_market_seasonality(df, self.seasonality_indices)
        
        # Calculate self-median for each hotel
        hotel_medians = df.groupby('hotel_id')['avg_price'].median()
        df['self_median'] = df['hotel_id'].map(hotel_medians)
        
        # Target: price ratio
        df['price_ratio'] = df['avg_price'] / df['self_median']
        
        # Filter valid rows
        valid_mask = (
            df['price_ratio'].notna() &
            (df['price_ratio'] > 0.5) &
            (df['price_ratio'] < 2.0) &
            df['self_median'].notna()
        )
        df = df[valid_mask].copy()
        
        # Filter to hotels with enough history
        hotel_counts = df.groupby('hotel_id').size()
        valid_hotels = hotel_counts[hotel_counts >= self.min_weeks_history].index
        df = df[df['hotel_id'].isin(valid_hotels)].copy()
        
        # Prepare features
        X = df[self.feature_cols].copy()
        y = df['price_ratio'].copy()
        
        return X, y, df
    
    def fit(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True
    ) -> 'TemporalModel':
        """
        Train the temporal model.
        
        Args:
            train_df: Training data with hotel_id, week_start, avg_price
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.use_lightgbm:
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_child_samples=50,
                random_state=42,
                verbosity=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=50,
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
        
        self.metrics = TemporalModelMetrics(
            r2_train=r2_train,
            r2_cv=np.mean(cv_scores),
            mae_train=mae_train,
            mae_cv=np.std(cv_scores),  # Using std as proxy
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
        hotel_id: int,
        target_week: date,
        history_df: pd.DataFrame
    ) -> TemporalPrediction:
        """
        Predict price for a hotel on a target week.
        
        Args:
            hotel_id: Hotel to predict for
            target_week: Target week
            history_df: Historical data for anchor calculation
        
        Returns:
            TemporalPrediction with recommended price
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get self anchor
        anchor_result = get_self_anchor(hotel_id, target_week, history_df)
        
        if anchor_result.anchor_type != "self":
            # Not enough history - can't use this model
            return TemporalPrediction(
                multiplier=1.0,
                recommended_price=anchor_result.anchor_price,
                self_anchor=anchor_result.anchor_price,
                confidence=0.0,
                features_used={}
            )
        
        # Engineer features for prediction
        week_df = pd.DataFrame([{
            'week_start': pd.Timestamp(target_week),
            'hotel_id': hotel_id
        }])
        week_df = engineer_temporal_features(week_df)
        week_df = add_market_seasonality(week_df, self.seasonality_indices)
        
        # Extract features
        X = week_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Predict multiplier
        multiplier = self.model.predict(X_scaled)[0]
        multiplier = np.clip(multiplier, MIN_MULTIPLIER, MAX_MULTIPLIER)
        
        recommended_price = anchor_result.anchor_price * multiplier
        
        # Confidence based on anchor confidence and prediction distance from 1.0
        prediction_confidence = 1.0 - abs(multiplier - 1.0) / 0.5
        confidence = anchor_result.confidence * prediction_confidence
        
        return TemporalPrediction(
            multiplier=multiplier,
            recommended_price=recommended_price,
            self_anchor=anchor_result.anchor_price,
            confidence=confidence,
            features_used=dict(zip(self.feature_cols, X[0]))
        )
    
    def predict_batch(
        self,
        hotel_ids: List[int],
        target_week: date,
        history_df: pd.DataFrame
    ) -> List[TemporalPrediction]:
        """
        Predict prices for multiple hotels.
        """
        return [
            self.predict(hotel_id, target_week, history_df)
            for hotel_id in hotel_ids
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


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================

def train_temporal_model(
    train_df: pd.DataFrame,
    verbose: bool = True
) -> TemporalModel:
    """
    Convenience function to train temporal model.
    
    Args:
        train_df: Training data
        verbose: Print progress
    
    Returns:
        Fitted TemporalModel
    """
    model = TemporalModel()
    model.fit(train_df, verbose=verbose)
    return model


# =============================================================================
# TESTING / DEMO
# =============================================================================

def main():
    """Demo temporal model training and prediction."""
    from src.models.evaluation.time_backtest import BacktestConfig, load_hotel_week_data
    
    print("=" * 60)
    print("TEMPORAL MODEL (PATH A) DEMO")
    print("=" * 60)
    
    config = BacktestConfig()
    
    print("\nLoading training data...")
    train_df = load_hotel_week_data(config, split="train")
    
    print("\nTraining temporal model...")
    model = train_temporal_model(train_df)
    
    print("\nFeature importance:")
    print(model.get_feature_importance().to_string())
    
    # Sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Get hotels with good history
    hotel_counts = train_df.groupby('hotel_id').size()
    good_hotels = hotel_counts[hotel_counts >= 20].index.tolist()[:5]
    
    target_week = config.test_start
    
    for hotel_id in good_hotels:
        pred = model.predict(hotel_id, target_week, train_df)
        hotel_data = train_df[train_df['hotel_id'] == hotel_id].iloc[0]
        
        print(f"\nHotel {hotel_id} ({hotel_data['city']}):")
        print(f"  Self anchor: €{pred.self_anchor:.0f}")
        print(f"  Multiplier:  {pred.multiplier:.2f}x")
        print(f"  Recommended: €{pred.recommended_price:.0f}")
        print(f"  Confidence:  {pred.confidence:.2f}")


if __name__ == "__main__":
    main()

