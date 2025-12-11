"""
Baseline Pricing Model.

Predicts what price a hotel would likely charge WITHOUT guidance.
This represents the "business as usual" pricing that hotels set based on
their features and local market conditions.

For cold-start hotels, this uses:
1. Peer prices from 10km radius (what similar hotels charge)
2. Hotel features (size, amenities, location)

The model learns the relationship between these factors and actual prices.

Model Selection (5-fold CV on 22,928 samples):
- Random Forest: R² = 0.783, MAE = €11.51, MAPE = 11.8%  <- SELECTED
- LightGBM:      R² = 0.777, MAE = €11.88, MAPE = 12.2%
- Gradient Boosting: R² = 0.777, MAE = €11.89, MAPE = 12.2%
- Linear Models: R² ~0.66, significantly worse
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class BaselinePricingModel:
    """
    Predicts baseline price from hotel features + peer context.
    
    This model answers: "What would a hotel likely charge without 
    any pricing guidance, based on their features and local market?"
    
    Input Features:
    - Hotel features: dist_center_km, amenities_score, log_room_size, etc.
    - Peer features: peer_price_mean, peer_price_median, peer_price_p25, peer_price_p75
    
    Output: Predicted baseline price (€)
    
    Usage:
        model = BaselinePricingModel()
        model.fit(train_df)  # Train on historical (price, features) pairs
        baseline_price = model.predict(hotel_features, peer_features)
    """
    
    def __init__(self):
        """Initialize the baseline pricing model."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False
        self._metrics = None
    
    def _prepare_features(
        self, 
        df: pd.DataFrame, 
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Prepare features for training/prediction.
        
        Features:
        - Hotel characteristics (validated by XGBoost)
        - Peer pricing context (10km radius)
        """
        X = df.copy()
        
        # Hotel feature columns (from validated features)
        hotel_features = [
            'dist_center_km',      # Distance to city center
            'is_madrid_metro',     # Within 50km of Madrid
            'log_room_size',       # Log of room size
            'amenities_score',     # Sum of amenity flags
            'total_rooms',         # Hotel capacity
        ]
        
        # Peer pricing features (10km radius)
        peer_features = [
            'peer_price_mean',
            'peer_price_median',
            'peer_price_p25',
            'peer_price_p75',
            'n_peers_10km',
        ]
        
        # Use available features
        all_features = hotel_features + peer_features
        available_features = [c for c in all_features if c in X.columns]
        
        # Ensure columns exist with defaults
        for col in available_features:
            if col not in X.columns:
                X[col] = 0
        
        # Handle missing values
        X_features = X[available_features].fillna(0)
        
        # Convert any non-numeric columns (safety check)
        for col in X_features.columns:
            if not np.issubdtype(X_features[col].dtype, np.number):
                X_features[col] = pd.to_numeric(X_features[col], errors='coerce').fillna(0)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_features)
        else:
            X_scaled = self.scaler.transform(X_features)
        
        self.feature_cols = available_features
        return pd.DataFrame(X_scaled, columns=available_features, index=X.index)
    
    def fit(
        self, 
        df: pd.DataFrame,
        target_col: str = 'actual_price',
        test_size: float = 0.2
    ) -> 'BaselinePricingModel':
        """
        Train the baseline pricing model.
        
        Args:
            df: Training data with features and actual_price
            target_col: Name of target column (price)
            test_size: Fraction for validation
        
        Returns:
            self
        """
        # Filter valid rows
        df_valid = df[
            (df[target_col].notna()) & 
            (df[target_col] > 0)
        ].copy()
        
        print(f"Training BaselinePricingModel on {len(df_valid):,} samples...")
        
        # Prepare features
        X = self._prepare_features(df_valid, fit=True)
        y = df_valid[target_col].values
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model (Random Forest - best performer in model selection)
        # R² = 0.783, MAE = €11.51, MAPE = 11.8% (vs 0.777 for GradientBoosting)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        mape = np.mean(np.abs(y_val - y_pred) / y_val) * 100
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
        # R²
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        self._metrics = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'r2': r2,
            'n_train': len(X_train),
            'n_val': len(X_val)
        }
        
        print(f"  MAE: €{mae:.2f}")
        print(f"  MAPE: {mape:.1f}%")
        print(f"  R²: {r2:.3f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict baseline prices.
        
        Args:
            df: Data with hotel features and peer features
        
        Returns:
            Array of predicted baseline prices (€)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(df, fit=False)
        predictions = self.model.predict(X)
        
        # Ensure reasonable price range
        return np.clip(predictions, 20, 500)
    
    def predict_single(
        self, 
        hotel_features: Dict, 
        peer_features: Dict
    ) -> float:
        """
        Predict baseline price for a single hotel.
        
        Args:
            hotel_features: Dict with hotel characteristics
            peer_features: Dict with peer pricing info
        
        Returns:
            Predicted baseline price (€)
        """
        # Combine into DataFrame
        combined = {**hotel_features, **peer_features}
        df = pd.DataFrame([combined])
        
        return self.predict(df)[0]
    
    def get_metrics(self) -> Optional[Dict]:
        """Get training metrics."""
        return self._metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
