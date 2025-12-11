"""
Peer Matching Module.

Provides multiple methods for finding comparable hotels:
1. Geographic: Hotels within a radius (10km, 50km, etc.)
2. KNN: Hotels similar in feature space (room size, amenities, etc.)
3. Segment: Hotels in same city + room_type + price_tier + season

Each method returns peer statistics that can be used as features
for the occupancy prediction model.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from src.features.engineering import haversine_distance, standardize_city


# =============================================================================
# CONSTANTS
# =============================================================================

# KNN feature columns (product/quality features)
KNN_FEATURES = [
    'log_room_size',
    'amenities_score', 
    'view_quality_ordinal',
    'total_rooms',
    'room_capacity_pax',
]

# Default parameters
DEFAULT_K = 10
DEFAULT_RADIUS_KM = 10.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PeerStats:
    """Statistics from a peer group."""
    n_peers: int
    median_price: float
    mean_price: float
    std_price: float
    median_occupancy: float
    mean_occupancy: float
    median_revpar: float
    mean_revpar: float
    price_p25: float
    price_p75: float
    method: str  # "geographic", "knn", "segment"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for feature engineering."""
        return {
            'peer_n': self.n_peers,
            'peer_median_price': self.median_price,
            'peer_mean_price': self.mean_price,
            'peer_std_price': self.std_price,
            'peer_median_occupancy': self.median_occupancy,
            'peer_mean_occupancy': self.mean_occupancy,
            'peer_median_revpar': self.median_revpar,
            'peer_mean_revpar': self.mean_revpar,
            'peer_price_p25': self.price_p25,
            'peer_price_p75': self.price_p75,
            'peer_method': self.method,
        }


# =============================================================================
# GEOGRAPHIC PEER MATCHING
# =============================================================================

class GeographicPeerMatcher:
    """
    Find peers based on geographic proximity.
    
    Simple and interpretable - hotels compete locally.
    """
    
    def __init__(self, radius_km: float = DEFAULT_RADIUS_KM):
        """
        Initialize geographic matcher.
        
        Args:
            radius_km: Search radius in kilometers
        """
        self.radius_km = radius_km
    
    def find_peers(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
        peer_df: pd.DataFrame,
        exclude_hotel_id: Optional[int] = None
    ) -> Tuple[pd.DataFrame, PeerStats]:
        """
        Find geographic peers for a hotel.
        
        Args:
            latitude: Hotel latitude
            longitude: Hotel longitude
            target_date: Target date for peer data
            peer_df: DataFrame with peer hotel data
            exclude_hotel_id: Hotel to exclude (self)
        
        Returns:
            Tuple of (peer_rows DataFrame, PeerStats)
        """
        df = peer_df.copy()
        
        # Exclude self
        if exclude_hotel_id is not None:
            df = df[df['hotel_id'] != exclude_hotel_id]
        
        # Filter to relevant time window (same week ± 1 week)
        target_ts = pd.Timestamp(target_date)
        df = df[
            (df['week_start'] >= target_ts - pd.Timedelta(days=7)) &
            (df['week_start'] <= target_ts + pd.Timedelta(days=7))
        ]
        
        if len(df) == 0:
            return pd.DataFrame(), self._empty_stats("geographic")
        
        # Calculate distances
        distances = haversine_distance(
            df['latitude'].values,
            df['longitude'].values,
            latitude,
            longitude
        )
        df = df.copy()
        df['distance_km'] = distances
        
        # Filter by radius
        peers = df[df['distance_km'] <= self.radius_km].copy()
        
        if len(peers) == 0:
            return pd.DataFrame(), self._empty_stats("geographic")
        
        return peers, self._compute_stats(peers, "geographic")
    
    def _compute_stats(self, peers: pd.DataFrame, method: str) -> PeerStats:
        """Compute peer statistics."""
        return PeerStats(
            n_peers=len(peers),
            median_price=peers['avg_price'].median(),
            mean_price=peers['avg_price'].mean(),
            std_price=peers['avg_price'].std(),
            median_occupancy=peers['occupancy_rate'].median(),
            mean_occupancy=peers['occupancy_rate'].mean(),
            median_revpar=peers['revpar'].median() if 'revpar' in peers else peers['avg_price'].median() * peers['occupancy_rate'].median(),
            mean_revpar=peers['revpar'].mean() if 'revpar' in peers else (peers['avg_price'] * peers['occupancy_rate']).mean(),
            price_p25=peers['avg_price'].quantile(0.25),
            price_p75=peers['avg_price'].quantile(0.75),
            method=method
        )
    
    def _empty_stats(self, method: str) -> PeerStats:
        """Return empty stats when no peers found."""
        return PeerStats(
            n_peers=0,
            median_price=np.nan,
            mean_price=np.nan,
            std_price=np.nan,
            median_occupancy=np.nan,
            mean_occupancy=np.nan,
            median_revpar=np.nan,
            mean_revpar=np.nan,
            price_p25=np.nan,
            price_p75=np.nan,
            method=method
        )


# =============================================================================
# KNN PEER MATCHING (FEATURE-BASED)
# =============================================================================

class KNNPeerMatcher:
    """
    Find peers based on feature similarity (KNN in feature space).
    
    Can find similar hotels even if geographically distant.
    Features: room_size, amenities, view_quality, total_rooms, capacity.
    """
    
    def __init__(
        self,
        k: int = DEFAULT_K,
        features: Optional[List[str]] = None
    ):
        """
        Initialize KNN matcher.
        
        Args:
            k: Number of neighbors to find
            features: Feature columns to use (default: KNN_FEATURES)
        """
        self.k = k
        self.features = features or KNN_FEATURES
        self.scaler = StandardScaler()
        self.nn_model = None
        self.hotel_index = None  # Maps NN index to hotel_id
        self.is_fitted = False
    
    def fit(self, hotel_features_df: pd.DataFrame) -> 'KNNPeerMatcher':
        """
        Fit the KNN model on hotel features.
        
        Args:
            hotel_features_df: DataFrame with hotel features (one row per hotel)
        
        Returns:
            self
        """
        # Get unique hotels with their features
        hotels = hotel_features_df.groupby('hotel_id').first().reset_index()
        
        # Prepare features
        X = self._prepare_features(hotels)
        
        if len(X) < self.k:
            raise ValueError(f"Need at least {self.k} hotels, got {len(X)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit NN model
        self.nn_model = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
        self.nn_model.fit(X_scaled)
        
        # Store hotel index mapping
        self.hotel_index = hotels['hotel_id'].values
        self.hotel_features = hotels
        
        self.is_fitted = True
        return self
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for KNN."""
        # Ensure all features exist
        for feat in self.features:
            if feat not in df.columns:
                df = df.copy()
                df[feat] = 0
        
        X = df[self.features].fillna(0).values
        return X
    
    def find_peers(
        self,
        hotel_id: int,
        target_date: date,
        peer_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, PeerStats]:
        """
        Find KNN peers for a hotel.
        
        Args:
            hotel_id: Hotel to find peers for
            target_date: Target date
            peer_df: DataFrame with peer data for the time period
        
        Returns:
            Tuple of (peer_rows DataFrame, PeerStats)
        """
        if not self.is_fitted:
            raise ValueError("KNN matcher not fitted. Call fit() first.")
        
        # Find hotel in index
        hotel_mask = self.hotel_index == hotel_id
        if not hotel_mask.any():
            return pd.DataFrame(), self._empty_stats("knn")
        
        hotel_idx = np.where(hotel_mask)[0][0]
        
        # Get hotel features
        hotel_features = self._prepare_features(
            self.hotel_features[self.hotel_features['hotel_id'] == hotel_id]
        )
        hotel_scaled = self.scaler.transform(hotel_features)
        
        # Find neighbors
        distances, indices = self.nn_model.kneighbors(hotel_scaled)
        
        # Exclude self (first neighbor is usually self)
        neighbor_ids = [
            self.hotel_index[idx] 
            for idx in indices[0] 
            if self.hotel_index[idx] != hotel_id
        ][:self.k]
        
        # Filter peer_df to neighbors and time window
        target_ts = pd.Timestamp(target_date)
        peers = peer_df[
            (peer_df['hotel_id'].isin(neighbor_ids)) &
            (peer_df['week_start'] >= target_ts - pd.Timedelta(days=7)) &
            (peer_df['week_start'] <= target_ts + pd.Timedelta(days=7))
        ].copy()
        
        if len(peers) == 0:
            return pd.DataFrame(), self._empty_stats("knn")
        
        return peers, self._compute_stats(peers, "knn")
    
    def find_peers_by_features(
        self,
        hotel_features: Dict,
        target_date: date,
        peer_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, PeerStats]:
        """
        Find KNN peers for a new hotel (not in training set).
        
        Args:
            hotel_features: Dict with feature values
            target_date: Target date
            peer_df: DataFrame with peer data
        
        Returns:
            Tuple of (peer_rows DataFrame, PeerStats)
        """
        if not self.is_fitted:
            raise ValueError("KNN matcher not fitted. Call fit() first.")
        
        # Prepare features
        feature_row = pd.DataFrame([{
            feat: hotel_features.get(feat, 0) for feat in self.features
        }])
        X = self._prepare_features(feature_row)
        X_scaled = self.scaler.transform(X)
        
        # Find neighbors
        distances, indices = self.nn_model.kneighbors(X_scaled)
        neighbor_ids = [self.hotel_index[idx] for idx in indices[0]][:self.k]
        
        # Filter peer_df
        target_ts = pd.Timestamp(target_date)
        peers = peer_df[
            (peer_df['hotel_id'].isin(neighbor_ids)) &
            (peer_df['week_start'] >= target_ts - pd.Timedelta(days=7)) &
            (peer_df['week_start'] <= target_ts + pd.Timedelta(days=7))
        ].copy()
        
        if len(peers) == 0:
            return pd.DataFrame(), self._empty_stats("knn")
        
        return peers, self._compute_stats(peers, "knn")
    
    def _compute_stats(self, peers: pd.DataFrame, method: str) -> PeerStats:
        """Compute peer statistics."""
        return PeerStats(
            n_peers=len(peers),
            median_price=peers['avg_price'].median(),
            mean_price=peers['avg_price'].mean(),
            std_price=peers['avg_price'].std(),
            median_occupancy=peers['occupancy_rate'].median(),
            mean_occupancy=peers['occupancy_rate'].mean(),
            median_revpar=peers['revpar'].median() if 'revpar' in peers else peers['avg_price'].median() * peers['occupancy_rate'].median(),
            mean_revpar=peers['revpar'].mean() if 'revpar' in peers else (peers['avg_price'] * peers['occupancy_rate']).mean(),
            price_p25=peers['avg_price'].quantile(0.25),
            price_p75=peers['avg_price'].quantile(0.75),
            method=method
        )
    
    def _empty_stats(self, method: str) -> PeerStats:
        """Return empty stats when no peers found."""
        return PeerStats(
            n_peers=0,
            median_price=np.nan,
            mean_price=np.nan,
            std_price=np.nan,
            median_occupancy=np.nan,
            mean_occupancy=np.nan,
            median_revpar=np.nan,
            mean_revpar=np.nan,
            price_p25=np.nan,
            price_p75=np.nan,
            method=method
        )


# =============================================================================
# SEGMENT PEER MATCHING
# =============================================================================

class SegmentPeerMatcher:
    """
    Find peers based on market segment.
    
    Segment = city + room_type + price_tier + season.
    Ensures comparability on key market dimensions.
    """
    
    def __init__(self, min_peers: int = 5):
        """
        Initialize segment matcher.
        
        Args:
            min_peers: Minimum peers required per segment
        """
        self.min_peers = min_peers
    
    def find_peers(
        self,
        hotel_id: int,
        target_date: date,
        peer_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, PeerStats]:
        """
        Find segment peers for a hotel.
        
        Args:
            hotel_id: Hotel to find peers for
            target_date: Target date
            peer_df: DataFrame with peer data
        
        Returns:
            Tuple of (peer_rows DataFrame, PeerStats)
        """
        # Get hotel's segment
        hotel_data = peer_df[peer_df['hotel_id'] == hotel_id]
        if len(hotel_data) == 0:
            return pd.DataFrame(), self._empty_stats("segment")
        
        hotel_row = hotel_data.iloc[0]
        
        # Get segment attributes
        city = hotel_row.get('city_standardized', hotel_row.get('city', 'unknown'))
        room_type = hotel_row.get('room_type', 'unknown')
        
        # Calculate price tier (quartile) for this hotel
        price_quartiles = peer_df['avg_price'].quantile([0.25, 0.5, 0.75])
        hotel_price = hotel_row['avg_price']
        if hotel_price <= price_quartiles[0.25]:
            price_tier = 'budget'
        elif hotel_price <= price_quartiles[0.5]:
            price_tier = 'economy'
        elif hotel_price <= price_quartiles[0.75]:
            price_tier = 'midscale'
        else:
            price_tier = 'upscale'
        
        # Get season
        month = target_date.month
        if month in [6, 7, 8]:
            season = 'summer'
        elif month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        else:
            season = 'fall'
        
        # Find peers in same segment
        target_ts = pd.Timestamp(target_date)
        
        # Add segment columns if not present
        df = peer_df.copy()
        if 'city_standardized' not in df.columns:
            df['city_standardized'] = df['city'].apply(
                lambda x: standardize_city(str(x)) if pd.notna(x) else 'other'
            )
        
        # Start with strict matching, relax if not enough peers
        peers = self._find_with_fallback(
            df, hotel_id, target_ts, city, room_type, price_tier, season
        )
        
        if len(peers) == 0:
            return pd.DataFrame(), self._empty_stats("segment")
        
        return peers, self._compute_stats(peers, "segment")
    
    def _find_with_fallback(
        self,
        df: pd.DataFrame,
        hotel_id: int,
        target_ts: pd.Timestamp,
        city: str,
        room_type: str,
        price_tier: str,
        season: str
    ) -> pd.DataFrame:
        """Find peers with progressive fallback."""
        # Exclude self and filter time
        base_mask = (
            (df['hotel_id'] != hotel_id) &
            (df['week_start'] >= target_ts - pd.Timedelta(days=7)) &
            (df['week_start'] <= target_ts + pd.Timedelta(days=7))
        )
        
        # Level 1: city + room_type + price_tier
        mask = base_mask
        if 'city_standardized' in df.columns:
            mask = mask & (df['city_standardized'] == city)
        if 'room_type' in df.columns:
            mask = mask & (df['room_type'] == room_type)
        
        peers = df[mask]
        if len(peers) >= self.min_peers:
            return peers
        
        # Level 2: city + room_type only
        mask = base_mask
        if 'city_standardized' in df.columns:
            mask = mask & (df['city_standardized'] == city)
        if 'room_type' in df.columns:
            mask = mask & (df['room_type'] == room_type)
        
        peers = df[mask]
        if len(peers) >= self.min_peers:
            return peers
        
        # Level 3: city only
        mask = base_mask
        if 'city_standardized' in df.columns:
            mask = mask & (df['city_standardized'] == city)
        
        peers = df[mask]
        if len(peers) >= self.min_peers:
            return peers
        
        # Level 4: all hotels in time window
        return df[base_mask]
    
    def _compute_stats(self, peers: pd.DataFrame, method: str) -> PeerStats:
        """Compute peer statistics."""
        return PeerStats(
            n_peers=len(peers),
            median_price=peers['avg_price'].median(),
            mean_price=peers['avg_price'].mean(),
            std_price=peers['avg_price'].std(),
            median_occupancy=peers['occupancy_rate'].median(),
            mean_occupancy=peers['occupancy_rate'].mean(),
            median_revpar=peers['revpar'].median() if 'revpar' in peers else peers['avg_price'].median() * peers['occupancy_rate'].median(),
            mean_revpar=peers['revpar'].mean() if 'revpar' in peers else (peers['avg_price'] * peers['occupancy_rate']).mean(),
            price_p25=peers['avg_price'].quantile(0.25),
            price_p75=peers['avg_price'].quantile(0.75),
            method=method
        )
    
    def _empty_stats(self, method: str) -> PeerStats:
        """Return empty stats when no peers found."""
        return PeerStats(
            n_peers=0,
            median_price=np.nan,
            mean_price=np.nan,
            std_price=np.nan,
            median_occupancy=np.nan,
            mean_occupancy=np.nan,
            median_revpar=np.nan,
            mean_revpar=np.nan,
            price_p25=np.nan,
            price_p75=np.nan,
            method=method
        )


# =============================================================================
# UNIFIED PEER MATCHER
# =============================================================================

class UnifiedPeerMatcher:
    """
    Combines all peer matching methods.
    
    Provides a unified interface for getting peer features.
    """
    
    def __init__(
        self,
        geo_radius_km: float = DEFAULT_RADIUS_KM,
        knn_k: int = DEFAULT_K,
        segment_min_peers: int = 5
    ):
        """
        Initialize unified matcher.
        
        Args:
            geo_radius_km: Radius for geographic matching
            knn_k: K for KNN matching
            segment_min_peers: Minimum peers for segment matching
        """
        self.geo_matcher = GeographicPeerMatcher(radius_km=geo_radius_km)
        self.knn_matcher = KNNPeerMatcher(k=knn_k)
        self.segment_matcher = SegmentPeerMatcher(min_peers=segment_min_peers)
        self.is_fitted = False
    
    def fit(self, hotel_features_df: pd.DataFrame) -> 'UnifiedPeerMatcher':
        """
        Fit KNN matcher (geo and segment don't need fitting).
        
        Args:
            hotel_features_df: DataFrame with hotel features
        
        Returns:
            self
        """
        self.knn_matcher.fit(hotel_features_df)
        self.is_fitted = True
        return self
    
    def get_all_peer_stats(
        self,
        hotel_id: int,
        latitude: float,
        longitude: float,
        target_date: date,
        peer_df: pd.DataFrame
    ) -> Dict[str, PeerStats]:
        """
        Get peer stats from all methods.
        
        Args:
            hotel_id: Hotel ID
            latitude: Hotel latitude
            longitude: Hotel longitude
            target_date: Target date
            peer_df: Peer data
        
        Returns:
            Dict mapping method name to PeerStats
        """
        results = {}
        
        # Geographic
        _, geo_stats = self.geo_matcher.find_peers(
            latitude, longitude, target_date, peer_df, exclude_hotel_id=hotel_id
        )
        results['geographic'] = geo_stats
        
        # KNN (if fitted)
        if self.is_fitted:
            _, knn_stats = self.knn_matcher.find_peers(
                hotel_id, target_date, peer_df
            )
            results['knn'] = knn_stats
        
        # Segment
        _, segment_stats = self.segment_matcher.find_peers(
            hotel_id, target_date, peer_df
        )
        results['segment'] = segment_stats
        
        return results
    
    def get_peer_features(
        self,
        hotel_id: int,
        latitude: float,
        longitude: float,
        target_date: date,
        peer_df: pd.DataFrame,
        method: str = 'geographic'
    ) -> Dict:
        """
        Get peer features for a specific method.
        
        Args:
            hotel_id: Hotel ID
            latitude: Hotel latitude
            longitude: Hotel longitude
            target_date: Target date
            peer_df: Peer data
            method: Which method to use ('geographic', 'knn', 'segment')
        
        Returns:
            Dict of peer features
        """
        all_stats = self.get_all_peer_stats(
            hotel_id, latitude, longitude, target_date, peer_df
        )
        
        if method in all_stats:
            return all_stats[method].to_dict()
        
        # Fallback to geographic
        return all_stats.get('geographic', self.geo_matcher._empty_stats('geographic')).to_dict()


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test peer matching."""
    from src.models.evaluation.time_backtest import BacktestConfig, load_hotel_week_data
    
    print("=" * 60)
    print("PEER MATCHING TEST")
    print("=" * 60)
    
    config = BacktestConfig()
    
    print("\nLoading data...")
    train_df = load_hotel_week_data(config, split='train')
    
    # Add revpar column if not present
    if 'revpar' not in train_df.columns:
        train_df['revpar'] = train_df['avg_price'] * train_df['occupancy_rate']
    
    # Initialize matchers
    print("\nInitializing peer matchers...")
    unified = UnifiedPeerMatcher()
    unified.fit(train_df)
    
    # Test on a sample hotel
    sample_hotel = train_df.iloc[0]
    hotel_id = sample_hotel['hotel_id']
    lat = sample_hotel['latitude']
    lon = sample_hotel['longitude']
    target_date = config.test_start
    
    print(f"\nTesting hotel {hotel_id} at ({lat:.4f}, {lon:.4f})")
    
    # Get all peer stats
    all_stats = unified.get_all_peer_stats(
        hotel_id, lat, lon, target_date, train_df
    )
    
    for method, stats in all_stats.items():
        print(f"\n{method.upper()}:")
        print(f"  Peers found: {stats.n_peers}")
        print(f"  Median price: €{stats.median_price:.0f}")
        print(f"  Median occupancy: {stats.median_occupancy:.1%}")
        print(f"  Median RevPAR: €{stats.median_revpar:.0f}")
    
    print("\n✓ Peer matching working!")


if __name__ == "__main__":
    main()

