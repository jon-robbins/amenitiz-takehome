"""
KNN Similarity-Based Pricing.

For each hotel, find K most similar hotels and use their weighted average price.
No model training needed - pure similarity matching.

Similarity features:
- Geographic: distance between hotels (lat/lon)
- Product: room_type, room_view, capacity
- Quality: amenities_score

Advantages:
- Explainable: "Similar hotels A, B, C charge €X"
- No training required
- Naturally handles market variation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from lib.db import init_db
from ml_pipeline.features import engineer_all_features


@dataclass
class KNNPrediction:
    """Result from KNN prediction."""
    predicted_price: float
    neighbor_prices: List[float]
    neighbor_hotel_ids: List[int]
    neighbor_distances: List[float]
    confidence: str  # 'high', 'medium', 'low'


# Features for similarity computation
# Note: Holiday features tested but hurt KNN performance (21.0% -> 21.8%)
# KNN matches on static hotel similarity, not temporal patterns
SIMILARITY_FEATURES = [
    # Geographic (most important)
    'latitude',
    'longitude',
    # Product characteristics
    'log_room_size',
    'room_capacity_pax',
    'total_capacity_log',
    'view_quality_ordinal',
    'amenities_score',
    # Encoded categoricals
    'room_type_encoded',
    'room_view_encoded',
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def load_hotel_prices() -> pd.DataFrame:
    """
    Loads hotel data with prices for KNN matching.
    """
    con = init_db()
    
    sql_file = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity/QUERY_LOAD_HOTEL_MONTH_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    df = con.execute(query).fetchdf()
    
    # Path to cities500.json for holiday features
    cities500_path = Path(__file__).parent.parent / 'data' / 'cities500.json'
    
    df = engineer_all_features(df, cities500_path=cities500_path)
    
    # Core market filter
    df = df[(df['avg_adr'] >= 50) & (df['avg_adr'] <= 250)]
    df = df.dropna(subset=['avg_adr', 'latitude', 'longitude'])
    
    return df


def prepare_knn_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Prepares feature matrix for KNN.
    
    Encodes categoricals and scales numerics.
    """
    df = df.copy()
    
    # Encode room_type
    room_type_map = {
        'single': 1, 'double': 2, 'twin': 3, 'triple': 4,
        'suite': 5, 'apartment': 6, 'studio': 7, 'villa': 8
    }
    df['room_type_encoded'] = df['room_type'].map(room_type_map).fillna(2)
    
    # Encode room_view
    view_map = {
        'no_view': 0, 'city_view': 1, 'garden_view': 2, 'pool_view': 3,
        'mountain_view': 4, 'lake_view': 5, 'sea_view': 6, 'ocean_view': 7
    }
    df['room_view_encoded'] = df['room_view'].map(view_map).fillna(0)
    
    # Select features
    feature_cols = [c for c in SIMILARITY_FEATURES if c in df.columns]
    X = df[feature_cols].copy().fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols,
        index=df.index
    )
    
    return X_scaled, scaler


class KNNPricer:
    """
    KNN-based price predictor.
    
    Finds K most similar hotels and returns weighted average of their prices.
    """
    
    def __init__(self, k: int = 10, max_distance_km: float = 50.0):
        """
        Args:
            k: Number of neighbors to use
            max_distance_km: Maximum geographic distance for neighbors
        """
        self.k = k
        self.max_distance_km = max_distance_km
        self.knn = None
        self.data = None
        self.features = None
        self.scaler = None
    
    def fit(self, df: pd.DataFrame) -> 'KNNPricer':
        """
        Fits the KNN model on hotel data.
        
        Args:
            df: DataFrame with hotel features and avg_adr
        """
        print(f"Fitting KNN on {len(df):,} hotel-months...", flush=True)
        
        self.data = df.copy()
        self.features, self.scaler = prepare_knn_features(df)
        
        # Fit KNN
        self.knn = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean', n_jobs=-1)
        self.knn.fit(self.features)
        
        print(f"✓ KNN fitted with k={self.k}", flush=True)
        return self
    
    def predict_single(
        self,
        hotel_features: pd.DataFrame,
        exclude_hotel_id: Optional[int] = None
    ) -> KNNPrediction:
        """
        Predicts price for a single hotel.
        
        Args:
            hotel_features: Single-row DataFrame with hotel features
            exclude_hotel_id: Hotel ID to exclude from neighbors (for leave-one-out)
        
        Returns:
            KNNPrediction with price and neighbor info
        """
        # Prepare features
        hotel_features = hotel_features.copy()
        
        # Encode categoricals
        room_type_map = {
            'single': 1, 'double': 2, 'twin': 3, 'triple': 4,
            'suite': 5, 'apartment': 6, 'studio': 7, 'villa': 8
        }
        view_map = {
            'no_view': 0, 'city_view': 1, 'garden_view': 2, 'pool_view': 3,
            'mountain_view': 4, 'lake_view': 5, 'sea_view': 6, 'ocean_view': 7
        }
        hotel_features['room_type_encoded'] = hotel_features['room_type'].map(room_type_map).fillna(2)
        hotel_features['room_view_encoded'] = hotel_features['room_view'].map(view_map).fillna(0)
        
        # Select and scale features
        feature_cols = [c for c in SIMILARITY_FEATURES if c in hotel_features.columns]
        X_query = hotel_features[feature_cols].fillna(0)
        X_query_scaled = self.scaler.transform(X_query)
        
        # Find neighbors
        distances, indices = self.knn.kneighbors(X_query_scaled)
        distances = distances[0]
        indices = indices[0]
        
        # Get neighbor info
        neighbor_data = self.data.iloc[indices]
        
        # Filter by geographic distance if we have lat/lon
        if 'latitude' in hotel_features.columns:
            query_lat = hotel_features['latitude'].iloc[0]
            query_lon = hotel_features['longitude'].iloc[0]
            
            geo_distances = []
            for idx in indices:
                n_lat = self.data.iloc[idx]['latitude']
                n_lon = self.data.iloc[idx]['longitude']
                geo_dist = haversine_distance(query_lat, query_lon, n_lat, n_lon)
                geo_distances.append(geo_dist)
            
            # Filter by max distance
            valid_mask = np.array(geo_distances) <= self.max_distance_km
        else:
            valid_mask = np.ones(len(indices), dtype=bool)
        
        # Exclude self
        if exclude_hotel_id is not None:
            valid_mask &= (neighbor_data['hotel_id'].values != exclude_hotel_id)
        
        # Get valid neighbors
        valid_indices = indices[valid_mask][:self.k]
        valid_distances = distances[valid_mask][:self.k]
        
        if len(valid_indices) == 0:
            # No valid neighbors - return median
            return KNNPrediction(
                predicted_price=self.data['avg_adr'].median(),
                neighbor_prices=[],
                neighbor_hotel_ids=[],
                neighbor_distances=[],
                confidence='low'
            )
        
        # Get neighbor prices
        neighbor_prices = self.data.iloc[valid_indices]['avg_adr'].values
        neighbor_hotel_ids = self.data.iloc[valid_indices]['hotel_id'].values
        
        # Weighted average (inverse distance weighting)
        if valid_distances.sum() > 0:
            weights = 1 / (valid_distances + 1e-6)
            weights = weights / weights.sum()
            predicted_price = np.average(neighbor_prices, weights=weights)
        else:
            predicted_price = np.mean(neighbor_prices)
        
        # Determine confidence
        if len(valid_indices) >= self.k * 0.8:
            confidence = 'high'
        elif len(valid_indices) >= self.k * 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return KNNPrediction(
            predicted_price=float(predicted_price),
            neighbor_prices=neighbor_prices.tolist(),
            neighbor_hotel_ids=neighbor_hotel_ids.tolist(),
            neighbor_distances=valid_distances.tolist(),
            confidence=confidence
        )
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts prices for multiple hotels.
        
        Uses leave-one-out for hotels in training set.
        """
        predictions = []
        
        for idx, row in df.iterrows():
            hotel_features = df.loc[[idx]]
            hotel_id = row.get('hotel_id')
            
            pred = self.predict_single(hotel_features, exclude_hotel_id=hotel_id)
            predictions.append(pred.predicted_price)
        
        return np.array(predictions)


def evaluate_knn_pricing(k_values: List[int] = [5, 10, 15, 20]) -> Dict:
    """
    Evaluates KNN pricing with different K values.
    """
    print("=" * 80)
    print("KNN SIMILARITY PRICING EVALUATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...", flush=True)
    df = load_hotel_prices()
    
    # Split
    df['month'] = pd.to_datetime(df['month'])
    train_df = df[df['month'] < '2024-06-01'].copy()
    test_df = df[df['month'] >= '2024-06-01'].copy()
    print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    results = {}
    
    print("\n2. Testing different K values...", flush=True)
    
    for k in k_values:
        print(f"\n   K = {k}:", flush=True)
        
        # Fit
        pricer = KNNPricer(k=k, max_distance_km=50)
        pricer.fit(train_df)
        
        # Predict on test set (sample for speed)
        test_sample = test_df.sample(min(1000, len(test_df)), random_state=42)
        
        predictions = []
        actuals = []
        
        for idx, row in test_sample.iterrows():
            pred = pricer.predict_single(test_sample.loc[[idx]])
            predictions.append(pred.predicted_price)
            actuals.append(row['avg_adr'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # MAPE
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"      MAPE: {mape:.1f}%")
        print(f"      R²: {r2:.4f}")
        
        results[k] = {'mape': mape, 'r2': r2}
    
    # Best K
    best_k = min(results.keys(), key=lambda k: results[k]['mape'])
    print(f"\n✓ Best K = {best_k} (MAPE = {results[best_k]['mape']:.1f}%)")
    
    return results, best_k


if __name__ == "__main__":
    results, best_k = evaluate_knn_pricing()

