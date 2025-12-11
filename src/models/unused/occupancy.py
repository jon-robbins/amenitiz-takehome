"""
Occupancy Prediction Model.

Predicts hotel occupancy from non-price features.
The price effect is applied separately via elasticity adjustment.

This separation avoids the endogeneity problem where price is
correlated with unobserved quality factors.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.features.engineering import MARKET_ELASTICITY


class OccupancyModel:
    """
    Predicts hotel occupancy from non-price features.
    
    Architecture:
    1. BASE OCCUPANCY: Predicts occupancy from hotel/temporal features
    2. ELASTICITY ADJUSTMENT: Applies validated price elasticity (-0.39)
    
    Usage:
        model = OccupancyModel()
        model.fit(training_data)
        base_occ = model.predict(hotel_features)
        adj_occ = model.predict_at_price(hotel_features, relative_price=1.2)
    """
    
    def __init__(self, elasticity: float = MARKET_ELASTICITY):
        """
        Initialize the model.
        
        Args:
            elasticity: Price elasticity of demand (default -0.39)
        """
        self.elasticity = elasticity
        self.model = None
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
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
        
        Uses XGBoost-validated features (non-price to avoid endogeneity):
        - Geographic: dist_coast_log, dist_center_km, is_coastal
        - Product: log_room_size, room_capacity_pax, amenities_score, view_quality_ordinal
        - Temporal: week_of_year, is_summer, is_winter
        - Capacity: total_rooms
        """
        X = df.copy()
        
        # Core numeric features (always available)
        # Note: Temporal features (week_of_year, is_summer, is_winter) only useful 
        # when data spans multiple time periods. For single-month analysis, they don't vary.
        numeric_cols = ['total_rooms']
        
        # Add temporal features only if they vary in the data
        if 'week_of_year' in X.columns and X['week_of_year'].nunique() > 1:
            numeric_cols.append('week_of_year')
        if 'is_summer' in X.columns and 'is_winter' in X.columns:
            # Only add if there's variation (data spans summer AND winter)
            if X['is_summer'].nunique() > 1 or X['is_winter'].nunique() > 1:
                numeric_cols.extend(['is_summer', 'is_winter'])
        
        # Validated XGBoost features (use if available)
        validated_features = [
            'dist_coast_log',      # Log distance to coast
            'dist_center_km',      # Distance to hotel's own city center
            'is_coastal',          # Coastal flag
            'is_madrid_metro',     # Within 50km of Madrid
            'log_room_size',       # Log of room size
            'room_capacity_pax',   # Max occupancy
            'amenities_score',     # Sum of amenity flags
            'view_quality_ordinal' # View quality score
        ]
        
        for col in validated_features:
            if col in X.columns:
                numeric_cols.append(col)
        
        # Ensure columns exist with defaults
        for col in numeric_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Handle city encoding
        if 'city_standardized' in X.columns:
            if fit:
                X['city_encoded'] = self.city_encoder.fit_transform(
                    X['city_standardized'].fillna('other')
                )
            else:
                X['city_encoded'] = X['city_standardized'].fillna('other').apply(
                    lambda x: self.city_encoder.transform([x])[0] 
                    if x in self.city_encoder.classes_ 
                    else self.city_encoder.transform(['other'])[0]
                )
            numeric_cols.append('city_encoded')
        
        # Select and scale features
        X_numeric = X[numeric_cols].fillna(0)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)
        
        self.feature_cols = numeric_cols
        return pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)
    
    def fit(
        self, 
        df: pd.DataFrame,
        target_col: str = 'occupancy_rate',
        test_size: float = 0.2
    ) -> 'OccupancyModel':
        """
        Train the occupancy model.
        
        Args:
            df: Training data with features and target
            target_col: Name of target column
            test_size: Fraction for test split
        
        Returns:
            self
        """
        # Filter valid rows
        df_valid = df[
            (df[target_col].notna()) & 
            (df[target_col] >= 0) & 
            (df[target_col] <= 1)
        ].copy()
        
        print(f"Training OccupancyModel on {len(df_valid):,} samples...")
        
        # Prepare features
        X = self._prepare_features(df_valid, fit=True)
        y = df_valid[target_col].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        self._metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print(f"  MAE: {mae:.4f} ({mae*100:.2f}%)")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict base occupancy (at reference price).
        
        Args:
            df: Data with hotel features
        
        Returns:
            Array of predicted occupancy rates (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(df, fit=False)
        predictions = self.model.predict(X)
        return np.clip(predictions, 0.01, 0.99)
    
    def predict_at_price(
        self, 
        df: pd.DataFrame, 
        relative_price: float = 1.0
    ) -> np.ndarray:
        """
        Predict occupancy at a specific price point using elasticity.
        
        Formula: occupancy = base_occupancy * (relative_price ^ elasticity)
        
        Args:
            df: Data with hotel features
            relative_price: Price relative to reference (1.0 = reference)
                           e.g., 1.1 = 10% above, 0.9 = 10% below
        
        Returns:
            Array of adjusted occupancy rates
        """
        base_occ = self.predict(df)
        price_adjustment = np.power(relative_price, self.elasticity)
        return np.clip(base_occ * price_adjustment, 0, 1)
    
    def get_metrics(self) -> Optional[Dict]:
        """Get training metrics."""
        return self._metrics
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'OccupancyModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

