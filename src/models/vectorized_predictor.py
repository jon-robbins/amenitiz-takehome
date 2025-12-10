"""
Vectorized Occupancy Predictor.

Truly vectorized implementation using:
1. Pre-computed peer statistics (no row-by-row loops)
2. All validated features from feature_importance_validation.py
3. Three peer methods: Geographic, KNN, Segment
4. Batch prediction for all hotels at once
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from src.features.engineering import get_market_segment, standardize_city


# =============================================================================
# VALIDATED FEATURES (from feature_importance_validation.py)
# =============================================================================

# Features for KNN similarity matching
KNN_FEATURES = [
    'log_room_size',
    'room_capacity_pax', 
    'amenities_score',
    'log_total_rooms',
    'view_quality_ordinal',
    'dist_center_km',
    'dist_coast_log',
    'is_coastal',
]

# All features for occupancy prediction
NUMERIC_FEATURES = [
    'dist_center_km',
    'dist_coast_log',
    'log_room_size',
    'room_capacity_pax',
    'amenities_score',
    'log_total_rooms',
    'view_quality_ordinal',
    'weekend_ratio',
]

TEMPORAL_FEATURES = [
    'month_sin',
    'month_cos',
    'week_of_year',
    'is_summer',
]

PRICE_FEATURES = [
    'candidate_price',
    'price_vs_peer_median',
    'log_candidate_price',
]

PEER_FEATURES = [
    'peer_median_price',
    'peer_median_occupancy',
    'peer_count',
]

BOOLEAN_FEATURES = [
    'is_coastal',
    'is_madrid_metro',
]

MODEL_SAVE_PATH = Path("outputs/models/vectorized_occupancy_model.pkl")


# =============================================================================
# VECTORIZED PEER COMPUTATION
# =============================================================================

class VectorizedPeerComputer:
    """
    Pre-computes peer statistics using vectorized operations.
    
    Three methods:
    1. Geographic: Spatial tree for fast radius queries
    2. KNN: Feature-based nearest neighbors
    3. Segment: Group-by on market_segment + room_type
    """
    
    def __init__(self, k_neighbors: int = 10, geo_radius_km: float = 10.0):
        self.k_neighbors = k_neighbors
        self.geo_radius_km = geo_radius_km
        
        # Will be fitted
        self.geo_tree = None
        self.knn_model = None
        self.knn_scaler = StandardScaler()
        self.hotel_data = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'VectorizedPeerComputer':
        """
        Fit peer computation models on hotel data.
        
        Args:
            df: DataFrame with hotel features (one row per hotel-week)
        """
        # Get unique hotels with their features
        self.hotel_data = df.groupby('hotel_id').agg({
            'latitude': 'first',
            'longitude': 'first',
            'avg_price': 'median',
            'occupancy_rate': 'median',
            'total_rooms': 'first',
            **{col: 'first' for col in KNN_FEATURES if col in df.columns}
        }).reset_index()
        
        # Ensure required columns exist
        self._add_derived_features(self.hotel_data)
        
        # 1. Build geographic spatial tree
        coords = np.column_stack([
            self.hotel_data['latitude'].values * 111,  # Convert to km
            self.hotel_data['longitude'].values * 85   # Approx at Spain latitude
        ])
        self.geo_tree = cKDTree(coords)
        
        # 2. Build KNN model on feature space
        knn_features = [f for f in KNN_FEATURES if f in self.hotel_data.columns]
        X_knn = self.hotel_data[knn_features].fillna(0).values
        X_knn_scaled = self.knn_scaler.fit_transform(X_knn)
        
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, len(self.hotel_data)),
            metric='euclidean'
        )
        self.knn_model.fit(X_knn_scaled)
        self.knn_features = knn_features
        
        self.is_fitted = True
        return self
    
    def _add_derived_features(self, df: pd.DataFrame) -> None:
        """Add derived features if missing."""
        if 'log_room_size' not in df.columns:
            room_size = df['avg_room_size'] if 'avg_room_size' in df.columns else 25
            df['log_room_size'] = np.log1p(room_size)
        if 'log_total_rooms' not in df.columns:
            total_rooms = df['total_rooms'] if 'total_rooms' in df.columns else 10
            df['log_total_rooms'] = np.log1p(total_rooms)
        if 'dist_coast_log' not in df.columns:
            dist_coast = df['distance_from_coast'] if 'distance_from_coast' in df.columns else 100
            df['dist_coast_log'] = np.log1p(dist_coast)
        if 'dist_center_km' not in df.columns:
            df['dist_center_km'] = 0
        if 'is_coastal' not in df.columns:
            if 'distance_from_coast' in df.columns:
                df['is_coastal'] = (df['distance_from_coast'] <= 20).astype(int)
            else:
                df['is_coastal'] = 0
        if 'view_quality_ordinal' not in df.columns:
            df['view_quality_ordinal'] = 0
        if 'amenities_score' not in df.columns:
            df['amenities_score'] = 0
        if 'room_capacity_pax' not in df.columns:
            df['room_capacity_pax'] = 2
    
    def get_geographic_peers_batch(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get geographic peer stats for multiple hotels at once.
        
        Returns:
            Tuple of (peer_median_price, peer_median_occupancy, peer_count)
        """
        if not self.is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        n = len(latitudes)
        peer_prices = np.zeros(n)
        peer_occs = np.zeros(n)
        peer_counts = np.zeros(n)
        
        # Convert to km coordinates
        coords = np.column_stack([latitudes * 111, longitudes * 85])
        
        # Query all at once
        indices_list = self.geo_tree.query_ball_point(coords, r=self.geo_radius_km)
        
        prices = self.hotel_data['avg_price'].values
        occs = self.hotel_data['occupancy_rate'].values
        
        for i, indices in enumerate(indices_list):
            if len(indices) > 1:
                # Exclude self (closest point)
                peer_prices[i] = np.median(prices[indices])
                peer_occs[i] = np.median(occs[indices])
                peer_counts[i] = len(indices)
            else:
                peer_prices[i] = np.nan
                peer_occs[i] = np.nan
                peer_counts[i] = 0
        
        return peer_prices, peer_occs, peer_counts
    
    def get_knn_peers_batch(
        self,
        feature_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get KNN peer stats for multiple hotels at once.
        
        Args:
            feature_matrix: (n_hotels, n_features) array of KNN features
        
        Returns:
            Tuple of (peer_median_price, peer_median_occupancy, peer_count)
        """
        if not self.is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        # Scale features
        X_scaled = self.knn_scaler.transform(feature_matrix)
        
        # Query all at once
        distances, indices = self.knn_model.kneighbors(X_scaled)
        
        prices = self.hotel_data['avg_price'].values
        occs = self.hotel_data['occupancy_rate'].values
        
        # Exclude first neighbor (self) and compute medians
        # indices shape: (n_hotels, k+1)
        peer_indices = indices[:, 1:]  # Exclude self
        
        peer_prices = np.median(prices[peer_indices], axis=1)
        peer_occs = np.median(occs[peer_indices], axis=1)
        peer_counts = np.full(len(feature_matrix), self.k_neighbors)
        
        return peer_prices, peer_occs, peer_counts
    
    def get_segment_peers_batch(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get segment peer stats using groupby (vectorized).
        
        Segment = market_segment + room_type
        """
        # Compute segment stats
        if 'market_segment' not in df.columns:
            df = df.copy()
            df['market_segment'] = 'provincial'
        
        segment_cols = ['market_segment']
        if 'room_type' in df.columns:
            segment_cols.append('room_type')
        
        segment_stats = df.groupby(segment_cols).agg({
            'avg_price': 'median',
            'occupancy_rate': 'median',
            'hotel_id': 'count'
        }).rename(columns={
            'avg_price': 'seg_price',
            'occupancy_rate': 'seg_occ',
            'hotel_id': 'seg_count'
        })
        
        # Merge back
        df_merged = df.merge(segment_stats, on=segment_cols, how='left')
        
        return (
            df_merged['seg_price'].values,
            df_merged['seg_occ'].values,
            df_merged['seg_count'].values
        )


# =============================================================================
# VECTORIZED OCCUPANCY PREDICTOR
# =============================================================================

@dataclass
class PredictionResult:
    """Result of batch prediction."""
    predicted_occupancy: np.ndarray
    predicted_revpar: np.ndarray
    optimal_prices: np.ndarray
    optimal_revpar: np.ndarray
    peer_method: str


class VectorizedOccupancyPredictor:
    """
    Fully vectorized occupancy predictor.
    
    No row-by-row loops - everything is batch operations.
    """
    
    def __init__(
        self,
        peer_method: str = 'geographic',  # 'geographic', 'knn', 'segment'
        k_neighbors: int = 10,
        geo_radius_km: float = 10.0
    ):
        self.peer_method = peer_method
        self.peer_computer = VectorizedPeerComputer(
            k_neighbors=k_neighbors,
            geo_radius_km=geo_radius_km
        )
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False
    
    def _get_all_feature_cols(self) -> List[str]:
        """Get all feature columns for the model."""
        return (
            NUMERIC_FEATURES +
            TEMPORAL_FEATURES +
            PRICE_FEATURES +
            PEER_FEATURES +
            BOOLEAN_FEATURES
        )
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features for the model."""
        df = df.copy()
        
        # Temporal features
        if 'week_start' in df.columns:
            df['month'] = pd.to_datetime(df['week_start']).dt.month
            df['week_of_year'] = pd.to_datetime(df['week_start']).dt.isocalendar().week.astype(int)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Derived features
        if 'log_room_size' not in df.columns:
            room_size = df['avg_room_size'] if 'avg_room_size' in df.columns else 25
            df['log_room_size'] = np.log1p(room_size)
        if 'log_total_rooms' not in df.columns:
            total_rooms = df['total_rooms'] if 'total_rooms' in df.columns else 10
            df['log_total_rooms'] = np.log1p(total_rooms)
        if 'dist_coast_log' not in df.columns:
            dist_coast = df['distance_from_coast'] if 'distance_from_coast' in df.columns else 100
            df['dist_coast_log'] = np.log1p(dist_coast)
        if 'log_candidate_price' not in df.columns and 'candidate_price' in df.columns:
            df['log_candidate_price'] = np.log1p(df['candidate_price'])
        
        # Boolean features
        if 'is_coastal' not in df.columns:
            if 'distance_from_coast' in df.columns:
                df['is_coastal'] = (df['distance_from_coast'] <= 20).astype(int)
            else:
                df['is_coastal'] = 0
        if 'is_madrid_metro' not in df.columns:
            if 'distance_from_madrid' in df.columns and 'distance_from_coast' in df.columns:
                df['is_madrid_metro'] = (
                    (df['distance_from_madrid'] <= 50) &
                    (df['distance_from_coast'] > 20)
                ).astype(int)
            else:
                df['is_madrid_metro'] = 0
        
        # Fill missing
        for col in self._get_all_feature_cols():
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        return df
    
    def _add_peer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add peer features using selected method."""
        df = df.copy()
        
        if self.peer_method == 'geographic':
            prices, occs, counts = self.peer_computer.get_geographic_peers_batch(
                df['latitude'].values,
                df['longitude'].values
            )
        elif self.peer_method == 'knn':
            # Prepare KNN features
            knn_cols = [c for c in KNN_FEATURES if c in df.columns]
            X_knn = df[knn_cols].fillna(0).values
            prices, occs, counts = self.peer_computer.get_knn_peers_batch(X_knn)
        else:  # segment
            prices, occs, counts = self.peer_computer.get_segment_peers_batch(df)
        
        df['peer_median_price'] = prices
        df['peer_median_occupancy'] = occs
        df['peer_count'] = counts
        
        # Fill NaN with market medians
        market_price = df['avg_price'].median()
        df['peer_median_price'] = df['peer_median_price'].fillna(market_price)
        df['peer_median_occupancy'] = df['peer_median_occupancy'].fillna(0.65)
        df['peer_count'] = df['peer_count'].fillna(0)
        
        # Price ratios
        df['price_vs_peer_median'] = df['candidate_price'] / df['peer_median_price'].clip(lower=1)
        
        return df
    
    def fit(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True
    ) -> 'VectorizedOccupancyPredictor':
        """
        Fit the model on training data.
        
        Target: rooms_booked (not occupancy_rate)
        """
        if verbose:
            print(f"Training vectorized predictor (peer_method={self.peer_method})...")
        
        df = train_df.copy()
        
        # Compute rooms_booked as target
        df['rooms_booked'] = (df['occupancy_rate'] * df['total_rooms']).round()
        
        # Set candidate_price = actual price for training
        df['candidate_price'] = df['avg_price']
        
        # Fit peer computer
        if verbose:
            print("  Fitting peer computer...")
        self.peer_computer.fit(df)
        
        # Engineer features
        if verbose:
            print("  Engineering features...")
        df = self._engineer_features(df)
        df = self._add_peer_features(df)
        
        # Get feature columns
        self.feature_cols = self._get_all_feature_cols()
        available = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols = available
        
        if verbose:
            print(f"  Using {len(self.feature_cols)} features")
        
        # Prepare training data
        X = df[self.feature_cols].values
        y = df['rooms_booked'].values
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if verbose:
            print("  Training model...")
        
        if HAS_CATBOOST:
            self.model = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=False,
                train_dir=None,  # Disable saving training info
                allow_writing_files=False
            )
        elif HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
        
        self.model.fit(X_scaled, y)
        
        # Compute training metrics
        y_pred = self.model.predict(X_scaled)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        
        if verbose:
            print(f"  R² (train): {r2:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict_occupancy_batch(
        self,
        df: pd.DataFrame,
        candidate_prices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict occupancy for all rows at once.
        
        Args:
            df: DataFrame with hotel features
            candidate_prices: Prices to evaluate (uses avg_price if None)
        
        Returns:
            Array of predicted occupancy rates
        """
        if not self.is_fitted:
            raise ValueError("Not fitted.")
        
        df = df.copy()
        
        if candidate_prices is not None:
            df['candidate_price'] = candidate_prices
        else:
            df['candidate_price'] = df['avg_price']
        
        df = self._engineer_features(df)
        df = self._add_peer_features(df)
        
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        predicted_rooms = self.model.predict(X_scaled)
        total_rooms = df['total_rooms'].fillna(10).values
        
        predicted_rooms = np.clip(predicted_rooms, 0, total_rooms)
        predicted_occ = predicted_rooms / np.maximum(total_rooms, 1)
        
        return np.clip(predicted_occ, 0.01, 0.99)
    
    def find_optimal_prices_batch(
        self,
        df: pd.DataFrame,
        price_min: float = 30,
        price_max: float = 400,
        n_prices: int = 20,
        elasticity: float = -0.39
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find RevPAR-maximizing price for all hotels at once.
        
        Uses elasticity to simulate occupancy at different prices:
        1. Predict baseline occupancy at actual price
        2. Apply elasticity: occ(P) = baseline_occ * (1 + e * (P - actual) / actual)
        3. Find P that maximizes P * occ(P)
        
        Args:
            df: DataFrame with hotel features
            price_min: Minimum price to consider  
            price_max: Maximum price to consider
            n_prices: Number of price points to evaluate
            elasticity: Price elasticity of demand
        
        Returns:
            Tuple of (optimal_prices, optimal_revpars)
        """
        if not self.is_fitted:
            raise ValueError("Not fitted.")
        
        n_hotels = len(df)
        
        # Step 1: Predict baseline occupancy at actual price
        baseline_occ = self.predict_occupancy_batch(df)  # At actual price
        actual_prices = df['avg_price'].values
        
        # Step 2: Grid search using elasticity
        prices = np.linspace(price_min, price_max, n_prices)
        
        # For each hotel, compute RevPAR at each price point
        # prices shape: (n_prices,)
        # actual_prices shape: (n_hotels,)
        # baseline_occ shape: (n_hotels,)
        
        # Broadcast to (n_hotels, n_prices)
        price_grid = np.broadcast_to(prices, (n_hotels, n_prices))
        actual_prices_2d = actual_prices[:, np.newaxis]
        baseline_occ_2d = baseline_occ[:, np.newaxis]
        
        # Apply elasticity: occ = baseline * (1 + e * pct_change)
        pct_change = (price_grid - actual_prices_2d) / actual_prices_2d
        simulated_occ = baseline_occ_2d * (1 + elasticity * pct_change)
        simulated_occ = np.clip(simulated_occ, 0.01, 0.99)
        
        # Compute RevPAR
        simulated_revpar = price_grid * simulated_occ
        
        # Find argmax for each hotel
        best_idx = np.argmax(simulated_revpar, axis=1)
        optimal_prices = prices[best_idx]
        optimal_revpars = simulated_revpar[np.arange(n_hotels), best_idx]
        
        return optimal_prices, optimal_revpars
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        save_path = path or MODEL_SAVE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'peer_method': self.peer_method,
            'peer_computer': self.peer_computer,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {save_path}")
        return save_path
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'VectorizedOccupancyPredictor':
        """Load model from disk."""
        load_path = path or MODEL_SAVE_PATH
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        predictor = cls(peer_method=state['peer_method'])
        predictor.model = state['model']
        predictor.scaler = state['scaler']
        predictor.feature_cols = state['feature_cols']
        predictor.peer_computer = state['peer_computer']
        predictor.is_fitted = True
        
        return predictor


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test vectorized predictor."""
    import time
    from src.models.evaluation.time_backtest import BacktestConfig, load_hotel_week_data
    
    print("=" * 60)
    print("VECTORIZED OCCUPANCY PREDICTOR TEST")
    print("=" * 60)
    
    config = BacktestConfig()
    full_df = load_hotel_week_data(config, split='all')
    
    # Load distance features
    dist_df = pd.read_csv('outputs/data/hotel_distance_features.csv')
    full_df = full_df.merge(dist_df, on='hotel_id', how='left')
    
    print(f"\nLoaded {len(full_df):,} records, {full_df['hotel_id'].nunique()} hotels")
    
    # Train on subset
    train_df = full_df.sample(n=5000, random_state=42)
    
    # Test all three peer methods
    for method in ['geographic', 'knn', 'segment']:
        print(f"\n{'='*60}")
        print(f"PEER METHOD: {method.upper()}")
        print("=" * 60)
        
        predictor = VectorizedOccupancyPredictor(peer_method=method)
        
        start = time.time()
        predictor.fit(train_df, verbose=True)
        fit_time = time.time() - start
        print(f"  Fit time: {fit_time:.1f}s")
        
        # Test batch prediction
        test_df = full_df.sample(n=1000, random_state=123)
        
        start = time.time()
        optimal_prices, optimal_revpars = predictor.find_optimal_prices_batch(test_df)
        pred_time = time.time() - start
        print(f"  Prediction time (1000 hotels): {pred_time:.2f}s")
        
        # Compare to actual
        actual_revpar = test_df['avg_price'].values * test_df['occupancy_rate'].values
        lift = optimal_revpars - actual_revpar
        win_rate = (lift > 0).mean()
        
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Mean lift: €{lift.mean():.2f}")
    
    # Save best model
    print("\n" + "=" * 60)
    print("Saving model...")
    predictor.save()


if __name__ == "__main__":
    main()

