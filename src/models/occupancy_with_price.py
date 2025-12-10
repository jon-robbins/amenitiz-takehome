"""
Occupancy Model with Price as Input.

Unlike the original OccupancyModel which predicts occupancy from features only
and then adjusts with elasticity, this model takes PRICE as a direct input.

This allows the model to learn the actual price→occupancy relationship (demand curve)
from the data, rather than assuming a constant elasticity.

Key Difference:
- Original: occupancy = f(features) * price^elasticity
- This:     occupancy = f(features, price)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class OccupancyModelWithPrice:
    """
    Predicts occupancy given price and hotel features.
    
    KEY DIFFERENCE from original OccupancyModel:
    - Price is an INPUT feature, not an after-the-fact adjustment
    - Model learns price→occupancy relationship directly (demand curve)
    
    Input Features:
    - price: The actual/proposed price (€)
    - log_price: Log transform for diminishing effects
    - price_vs_peer_ratio: Price relative to peer average
    - Hotel features: location, size, amenities
    - Temporal features: week_of_year, is_summer, etc.
    
    Output: Predicted occupancy (0-1)
    
    Usage:
        model = OccupancyModelWithPrice()
        model.fit(val_df)  # Train on validation set with (price, occupancy) pairs
        occ = model.predict(price=120, hotel_features=df)
    """
    
    def __init__(self):
        """Initialize the occupancy model."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False
        self._metrics = None
    
    def _prepare_features(
        self, 
        df: pd.DataFrame,
        price_col: str = 'actual_price',
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Prepare features for training/prediction.
        
        KEY: Price is included as a feature!
        
        Features:
        - Price features: price, log_price, price_vs_peer
        - Hotel features: location, size, amenities
        - Temporal features: seasonality
        """
        X = df.copy()
        
        # Ensure price column exists
        if price_col not in X.columns:
            raise ValueError(f"Price column '{price_col}' not found")
        
        # Price features (THE KEY ADDITION)
        X['log_price'] = np.log1p(X[price_col].clip(1, 1000))
        
        # Price relative to peers (if available)
        if 'peer_price_mean' in X.columns:
            X['price_vs_peer_ratio'] = X[price_col] / X['peer_price_mean'].clip(1, 1000)
            X['price_vs_peer_ratio'] = X['price_vs_peer_ratio'].clip(0.2, 5.0)
        else:
            X['price_vs_peer_ratio'] = 1.0
        
        # Feature columns
        price_features = ['log_price', 'price_vs_peer_ratio']
        
        hotel_features = [
            'dist_center_km',
            'is_madrid_metro',
            'log_room_size',
            'amenities_score',
            'total_rooms',
        ]
        
        temporal_features = [
            'week_of_year',
            'month',
        ]
        
        # Combine available features
        all_features = price_features + hotel_features + temporal_features
        available_features = [c for c in all_features if c in X.columns]
        
        # Ensure minimum features exist
        for col in available_features:
            if col not in X.columns:
                X[col] = 0
        
        # Handle missing values and ensure numeric types
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
        price_col: str = 'actual_price',
        target_col: str = 'actual_occupancy',
        test_size: float = 0.2
    ) -> 'OccupancyModelWithPrice':
        """
        Train the occupancy model with price as input.
        
        Args:
            df: Training data with price, features, and occupancy
            price_col: Name of price column
            target_col: Name of occupancy column
            test_size: Fraction for validation
        
        Returns:
            self
        """
        # Filter valid rows
        df_valid = df[
            (df[target_col].notna()) & 
            (df[target_col] >= 0) & 
            (df[target_col] <= 1) &
            (df[price_col].notna()) &
            (df[price_col] > 0)
        ].copy()
        
        print(f"Training OccupancyModelWithPrice on {len(df_valid):,} samples...")
        print(f"  Price range: €{df_valid[price_col].min():.0f} - €{df_valid[price_col].max():.0f}")
        print(f"  Occupancy range: {df_valid[target_col].min():.2f} - {df_valid[target_col].max():.2f}")
        
        # Prepare features (including price!)
        X = self._prepare_features(df_valid, price_col=price_col, fit=True)
        y = df_valid[target_col].values
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        y_pred = np.clip(y_pred, 0, 1)
        
        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
        # R²
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        self._metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_train': len(X_train),
            'n_val': len(X_val)
        }
        
        print(f"  MAE: {mae:.3f} ({mae*100:.1f}%)")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        df: pd.DataFrame,
        price_col: str = 'actual_price'
    ) -> np.ndarray:
        """
        Predict occupancy given prices and features.
        
        Args:
            df: Data with price and hotel features
            price_col: Name of price column
        
        Returns:
            Array of predicted occupancy rates (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(df, price_col=price_col, fit=False)
        predictions = self.model.predict(X)
        
        return np.clip(predictions, 0.01, 0.99)
    
    def predict_at_price(
        self, 
        hotel_features: pd.DataFrame,
        price: float,
        peer_price_mean: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict occupancy at a specific price point.
        
        Args:
            hotel_features: Hotel feature DataFrame
            price: Price to predict occupancy at (€)
            peer_price_mean: Peer average price (optional)
        
        Returns:
            Array of predicted occupancy at this price
        """
        df = hotel_features.copy()
        df['actual_price'] = price
        
        if peer_price_mean is not None:
            df['peer_price_mean'] = peer_price_mean
        
        return self.predict(df, price_col='actual_price')
    
    def predict_demand_curve(
        self,
        hotel_features: pd.DataFrame,
        price_range: tuple = (50, 300),
        n_points: int = 50,
        peer_price_mean: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate demand curve (price vs occupancy) for a hotel.
        
        Args:
            hotel_features: Single hotel feature row
            price_range: (min_price, max_price)
            n_points: Number of price points
            peer_price_mean: Peer average price
        
        Returns:
            DataFrame with price, predicted_occupancy, predicted_revpar
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        
        occupancies = []
        for p in prices:
            occ = self.predict_at_price(hotel_features, p, peer_price_mean)[0]
            occupancies.append(occ)
        
        result = pd.DataFrame({
            'price': prices,
            'predicted_occupancy': occupancies,
            'predicted_revpar': prices * np.array(occupancies)
        })
        
        return result
    
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

