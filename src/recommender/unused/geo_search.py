"""
Geographic search for finding nearby peer hotels.

Uses KDTree spatial indexing for efficient nearest-neighbor queries.
Provides similarity-weighted peer selection based on VALIDATED features
from feature_importance_validation.py:

Feature weights (based on SHAP importance from XGBoost R²=0.75):
- room_type: 0.25 (categorical match)
- room_size: 0.20 (log-transformed)
- view_quality: 0.10 (ordinal 0-3)
- capacity: 0.15 (total hotel capacity)
- amenities: 0.10 (score 0-4)
- is_coastal: 0.10 (boolean)
- distance: 0.10 (geographic proximity)
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.temporal_loader import (
    HotelProfile,
    PeerMetrics,
    get_peer_revpar_metrics,
    get_weighted_peer_average,
    load_hotel_locations,
)


# Earth radius in kilometers for haversine calculations
EARTH_RADIUS_KM = 6371.0

# =============================================================================
# VALIDATED SIMILARITY WEIGHTS
# Based on SHAP feature importance from feature_importance_validation.py
# These weights reflect how much each feature contributes to price prediction
# =============================================================================
SIMILARITY_WEIGHTS = {
    'room_type': 0.25,       # Categorical match (room, apartment, villa, etc.)
    'room_size': 0.20,       # Log-transformed room size in sqm
    'view_quality': 0.10,    # Ordinal 0-3 (no view, garden, mountain, ocean)
    'capacity': 0.15,        # Total hotel capacity (log-transformed)
    'amenities': 0.10,       # Score 0-4 (children, pets, events, smoking)
    'is_coastal': 0.10,      # Boolean - within 20km of coast
    'distance': 0.10,        # Geographic proximity
}

# Room type similarity matrix (same category = 1.0, similar = partial, different = 0.0)
ROOM_TYPE_SIMILARITY = {
    # Exact matches
    ('room', 'room'): 1.0,
    ('apartment', 'apartment'): 1.0,
    ('villa', 'villa'): 1.0,
    ('studio', 'studio'): 1.0,
    ('suite', 'suite'): 1.0,
    ('house', 'house'): 1.0,
    # Similar types
    ('room', 'suite'): 0.7,
    ('suite', 'room'): 0.7,
    ('apartment', 'studio'): 0.8,
    ('studio', 'apartment'): 0.8,
    ('villa', 'house'): 0.8,
    ('house', 'villa'): 0.8,
    ('villa', 'apartment'): 0.5,
    ('apartment', 'villa'): 0.5,
}

# View quality ordinal mapping (from feature_importance_validation.py)
VIEW_QUALITY_MAP = {
    'ocean_view': 3, 'sea_view': 3,
    'lake_view': 2, 'mountain_view': 2,
    'pool_view': 1, 'garden_view': 1,
    'city_view': 0, 'standard': 0, 'no_view': 0, 'unknown': 0, '': 0
}


@dataclass
class NearbyHotel:
    """
    A hotel found in geographic proximity search.
    
    Includes all validated features from feature_importance_validation.py
    for accurate similarity scoring.
    
    Attributes:
        hotel_id: Unique hotel identifier
        lat: Latitude
        lon: Longitude
        distance_km: Distance from query point
        room_type: Primary room type (room, apartment, villa, etc.)
        room_size: Average room size in sqm
        total_rooms: Number of rooms available (hotel capacity)
        view_quality: Ordinal 0-3 (no view, garden, mountain, ocean)
        amenities_score: Score 0-4 (children, pets, events, smoking allowed)
        is_coastal: Whether within 20km of coast
        amenities: List of individual amenity names
        similarity_score: Weighted similarity to query profile (0-1)
        similarity_breakdown: Dict of individual feature similarity scores
    """
    hotel_id: int
    lat: float
    lon: float
    distance_km: float
    room_type: str
    room_size: float
    total_rooms: int
    view_quality: int = 0
    amenities_score: int = 0
    is_coastal: bool = False
    amenities: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    similarity_breakdown: dict = field(default_factory=dict)


class HotelSpatialIndex:
    """
    Spatial index for fast geographic hotel search.
    
    Uses a KDTree built on hotel coordinates for O(log n) queries.
    Provides similarity-weighted peer selection.
    
    Usage:
        index = HotelSpatialIndex()
        index.build(hotel_locations_df)
        nearby = index.find_nearby(lat=40.42, lon=-3.70, radius_km=10)
    """
    
    def __init__(self):
        """Initialize empty spatial index."""
        self._tree: Optional[cKDTree] = None
        self._hotel_data: Optional[pd.DataFrame] = None
        self._coords: Optional[np.ndarray] = None
    
    @property
    def is_built(self) -> bool:
        """Check if index has been built."""
        return self._tree is not None
    
    @property
    def n_hotels(self) -> int:
        """Number of hotels in the index."""
        return len(self._hotel_data) if self._hotel_data is not None else 0
    
    def build(self, hotel_df: pd.DataFrame) -> 'HotelSpatialIndex':
        """
        Build the spatial index from hotel data.
        
        Args:
            hotel_df: DataFrame with columns: hotel_id, latitude, longitude,
                     room_type, avg_room_size, total_rooms
        
        Returns:
            self for method chaining
        """
        # Validate required columns
        required_cols = ['hotel_id', 'latitude', 'longitude']
        missing = [c for c in required_cols if c not in hotel_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Filter out invalid coordinates
        valid_mask = (
            hotel_df['latitude'].notna() & 
            hotel_df['longitude'].notna() &
            (hotel_df['latitude'].between(-90, 90)) &
            (hotel_df['longitude'].between(-180, 180))
        )
        self._hotel_data = hotel_df[valid_mask].copy().reset_index(drop=True)
        
        if len(self._hotel_data) == 0:
            raise ValueError("No valid hotel coordinates found")
        
        # Build KDTree using lat/lon (approximate Euclidean distance OK for nearby)
        self._coords = self._hotel_data[['latitude', 'longitude']].values
        self._tree = cKDTree(self._coords)
        
        return self
    
    def _haversine_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: np.ndarray, 
        lon2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate haversine distance between a point and array of points.
        
        Args:
            lat1, lon1: Query point coordinates (degrees)
            lat2, lon2: Target point coordinates (degrees)
        
        Returns:
            Array of distances in kilometers
        """
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return EARTH_RADIUS_KM * c
    
    def _calculate_validated_similarity(
        self,
        query_profile: dict,
        candidates: pd.DataFrame,
        distances: np.ndarray,
        max_distance: float
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Calculate similarity scores using VALIDATED features from XGBoost model.
        
        Uses weighted combination based on SHAP feature importance:
        - room_type: 0.25 (categorical match)
        - room_size: 0.20 (log-transformed size similarity)
        - view_quality: 0.10 (ordinal 0-3)
        - capacity: 0.15 (log-transformed hotel capacity)
        - amenities: 0.10 (score 0-4)
        - is_coastal: 0.10 (boolean match)
        - distance: 0.10 (geographic proximity)
        
        Args:
            query_profile: Dict with query hotel features
            candidates: DataFrame of candidate hotels
            distances: Array of distances in km
            max_distance: Maximum distance for normalization
        
        Returns:
            Tuple of (similarity scores array, list of breakdown dicts)
        """
        n = len(candidates)
        breakdowns = []
        
        # Extract query features with defaults
        q_room_type = (query_profile.get('room_type') or 'room').lower()
        q_room_size = query_profile.get('room_size', 30.0)
        q_view_quality = query_profile.get('view_quality', 0)
        q_capacity = query_profile.get('total_rooms', 1)
        q_amenities = query_profile.get('amenities_score', 0)
        q_coastal = query_profile.get('is_coastal', False)
        
        # 1. Room Type Similarity (0.25)
        room_type_scores = np.zeros(n)
        for i, target_type in enumerate(candidates['room_type']):
            target_lower = (target_type or 'unknown').lower()
            if q_room_type == target_lower:
                room_type_scores[i] = 1.0
            else:
                room_type_scores[i] = ROOM_TYPE_SIMILARITY.get(
                    (q_room_type, target_lower), 0.0
                )
        
        # 2. Room Size Similarity (0.20) - using log-scale as validated
        target_sizes = candidates['avg_room_size'].fillna(30.0).values
        # Log-scale comparison (validated feature is log_room_size)
        log_q_size = np.log1p(q_room_size)
        log_t_sizes = np.log1p(target_sizes)
        max_log_diff = np.log1p(50)  # ~50 sqm difference in log space
        size_diff = np.abs(log_t_sizes - log_q_size)
        room_size_scores = np.maximum(0, 1 - size_diff / max_log_diff)
        
        # 3. View Quality Similarity (0.10)
        if 'view_quality' in candidates.columns:
            target_views = candidates['view_quality'].fillna(0).values
        else:
            # Compute from room_view if available
            target_views = np.zeros(n)
            if 'room_view' in candidates.columns:
                for i, view in enumerate(candidates['room_view']):
                    target_views[i] = VIEW_QUALITY_MAP.get(
                        (view or '').lower(), 0
                    )
        max_view_diff = 3  # Max difference is 0-3
        view_diff = np.abs(target_views - q_view_quality)
        view_scores = 1 - view_diff / max_view_diff
        
        # 4. Capacity Similarity (0.15) - using log-scale as validated
        target_capacity = candidates['total_rooms'].fillna(1).clip(lower=1).values
        log_q_cap = np.log1p(q_capacity)
        log_t_cap = np.log1p(target_capacity)
        max_log_cap_diff = np.log1p(100)  # ~100 room difference in log space
        cap_diff = np.abs(log_t_cap - log_q_cap)
        capacity_scores = np.maximum(0, 1 - cap_diff / max_log_cap_diff)
        
        # 5. Amenities Similarity (0.10)
        if 'amenities_score' in candidates.columns:
            target_amenities = candidates['amenities_score'].fillna(0).values
        else:
            # Compute from boolean columns
            target_amenities = np.zeros(n)
            for col in ['children_allowed', 'pets_allowed', 'events_allowed', 'smoking_allowed']:
                if col in candidates.columns:
                    target_amenities += candidates[col].fillna(False).astype(int).values
        max_amenity_diff = 4  # Max difference is 0-4
        amenity_diff = np.abs(target_amenities - q_amenities)
        amenity_scores = 1 - amenity_diff / max_amenity_diff
        
        # 6. Coastal Similarity (0.10) - boolean match
        if 'is_coastal' in candidates.columns:
            target_coastal = candidates['is_coastal'].fillna(False).values
        else:
            target_coastal = np.zeros(n, dtype=bool)
        coastal_scores = (target_coastal == q_coastal).astype(float)
        
        # 7. Distance Proximity (0.10)
        distance_scores = np.maximum(0, 1 - distances / max_distance)
        
        # Weighted combination using VALIDATED weights
        similarity = (
            SIMILARITY_WEIGHTS['room_type'] * room_type_scores +
            SIMILARITY_WEIGHTS['room_size'] * room_size_scores +
            SIMILARITY_WEIGHTS['view_quality'] * view_scores +
            SIMILARITY_WEIGHTS['capacity'] * capacity_scores +
            SIMILARITY_WEIGHTS['amenities'] * amenity_scores +
            SIMILARITY_WEIGHTS['is_coastal'] * coastal_scores +
            SIMILARITY_WEIGHTS['distance'] * distance_scores
        )
        
        # Build breakdown for each candidate
        for i in range(n):
            breakdowns.append({
                'room_type': room_type_scores[i],
                'room_size': room_size_scores[i],
                'view_quality': view_scores[i],
                'capacity': capacity_scores[i],
                'amenities': amenity_scores[i],
                'is_coastal': coastal_scores[i],
                'distance': distance_scores[i],
            })
        
        return similarity, breakdowns
    
    def find_nearby(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        room_type: Optional[str] = None,
        room_size: Optional[float] = None,
        view_quality: int = 0,
        total_rooms: int = 1,
        amenities_score: int = 0,
        is_coastal: bool = False,
        min_similarity: float = 0.0,
        max_results: int = 100
    ) -> List[NearbyHotel]:
        """
        Find hotels within a radius of a point using VALIDATED similarity features.
        
        Args:
            lat: Query latitude
            lon: Query longitude
            radius_km: Search radius in kilometers
            room_type: Query hotel room type (room, apartment, villa, etc.)
            room_size: Query hotel avg room size in sqm
            view_quality: Query hotel view quality (0-3)
            total_rooms: Query hotel total capacity
            amenities_score: Query hotel amenities score (0-4)
            is_coastal: Whether query hotel is coastal
            min_similarity: Minimum similarity score to include
            max_results: Maximum number of results to return
        
        Returns:
            List of NearbyHotel objects sorted by similarity (desc)
        """
        if not self.is_built:
            raise RuntimeError("Spatial index not built. Call build() first.")
        
        # Convert radius to approximate lat/lon degrees for KDTree query
        lat_degree = radius_km / 111.0
        lon_degree = radius_km / (111.0 * np.cos(np.radians(lat)))
        max_degree = max(lat_degree, lon_degree) * 1.5  # Add buffer
        
        # Query KDTree for candidates
        query_point = np.array([[lat, lon]])
        indices = self._tree.query_ball_point(query_point[0], max_degree)
        
        if len(indices) == 0:
            return []
        
        # Get candidate hotels
        candidates = self._hotel_data.iloc[indices].copy()
        
        # Calculate actual haversine distances
        distances = self._haversine_distance(
            lat, lon,
            candidates['latitude'].values,
            candidates['longitude'].values
        )
        candidates['distance_km'] = distances
        
        # Filter to actual radius
        candidates = candidates[candidates['distance_km'] <= radius_km].copy()
        
        if len(candidates) == 0:
            return []
        
        # Set defaults for missing columns
        if 'room_type' not in candidates.columns:
            candidates['room_type'] = 'unknown'
        if 'avg_room_size' not in candidates.columns:
            candidates['avg_room_size'] = 30.0
        if 'total_rooms' not in candidates.columns:
            candidates['total_rooms'] = 1
        
        # Build query profile with all validated features
        query_profile = {
            'room_type': room_type or 'room',
            'room_size': room_size or candidates['avg_room_size'].median(),
            'view_quality': view_quality,
            'total_rooms': total_rooms,
            'amenities_score': amenities_score,
            'is_coastal': is_coastal,
        }
        
        # Calculate similarity using VALIDATED features
        similarity_scores, breakdowns = self._calculate_validated_similarity(
            query_profile,
            candidates,
            candidates['distance_km'].values,
            radius_km
        )
        candidates['similarity_score'] = similarity_scores
        
        # Filter by minimum similarity
        mask = candidates['similarity_score'] >= min_similarity
        candidates = candidates[mask].copy()
        breakdowns = [b for b, m in zip(breakdowns, mask) if m]
        
        # Sort by similarity (descending) and limit results
        sort_idx = candidates['similarity_score'].values.argsort()[::-1][:max_results]
        candidates = candidates.iloc[sort_idx]
        breakdowns = [breakdowns[i] for i in sort_idx]
        
        # Build result list with all validated features
        results = []
        for (_, row), breakdown in zip(candidates.iterrows(), breakdowns):
            # Extract amenities list from boolean columns
            amenities = []
            amenities_count = 0
            for col in ['children_allowed', 'pets_allowed', 'events_allowed', 'smoking_allowed']:
                if row.get(col, False):
                    amenities.append(col)
                    amenities_count += 1
            
            # Get view quality
            if 'view_quality' in row:
                vq = int(row['view_quality'])
            elif 'room_view' in row:
                vq = VIEW_QUALITY_MAP.get((row['room_view'] or '').lower(), 0)
            else:
                vq = 0
            
            # Get is_coastal
            if 'is_coastal' in row:
                coastal = bool(row['is_coastal'])
            else:
                coastal = False
            
            results.append(NearbyHotel(
                hotel_id=int(row['hotel_id']),
                lat=row['latitude'],
                lon=row['longitude'],
                distance_km=row['distance_km'],
                room_type=row.get('room_type', 'unknown'),
                room_size=row.get('avg_room_size', 30.0),
                total_rooms=int(row.get('total_rooms', 1)),
                view_quality=vq,
                amenities_score=amenities_count,
                is_coastal=coastal,
                amenities=amenities,
                similarity_score=row['similarity_score'],
                similarity_breakdown=breakdown
            ))
        
        return results
    
    def get_coverage_stats(self, radius_km: float = 10.0) -> dict:
        """
        Calculate coverage statistics for the index.
        
        Returns dict with:
        - total_hotels: Total hotels in index
        - hotels_with_peers: Hotels that have at least 1 peer within radius
        - coverage_pct: Percentage of hotels with peers
        
        Args:
            radius_km: Radius to check for peers
        
        Returns:
            Dictionary with coverage statistics
        """
        if not self.is_built:
            raise RuntimeError("Spatial index not built. Call build() first.")
        
        hotels_with_peers = 0
        
        for i, row in self._hotel_data.iterrows():
            nearby = self.find_nearby(
                row['latitude'], 
                row['longitude'],
                radius_km=radius_km,
                max_results=2  # Just need to know if there's at least 1 other
            )
            # Exclude self
            non_self = [h for h in nearby if h.hotel_id != row['hotel_id']]
            if len(non_self) > 0:
                hotels_with_peers += 1
        
        return {
            'total_hotels': self.n_hotels,
            'hotels_with_peers': hotels_with_peers,
            'coverage_pct': hotels_with_peers / self.n_hotels * 100 if self.n_hotels > 0 else 0
        }


def build_hotel_index(con) -> HotelSpatialIndex:
    """
    Build a spatial index from database connection.
    
    Convenience function that loads hotel locations and builds the index.
    
    Args:
        con: DuckDB connection with hotel data
    
    Returns:
        Built HotelSpatialIndex ready for queries
    """
    locations_df = load_hotel_locations(con)
    index = HotelSpatialIndex()
    index.build(locations_df)
    return index


def find_geographic_peers(
    con,
    lat: float,
    lon: float,
    target_dates: List[date],
    as_of_date: date,
    radius_km: float = 10.0,
    room_type: Optional[str] = None,
    room_size: Optional[float] = None,
    min_peers: int = 3
) -> Tuple[List[PeerMetrics], Optional[PeerMetrics]]:
    """
    Find geographic peers and calculate their RevPAR metrics.
    
    This is the main entry point for geographic peer comparison.
    It combines spatial search with RevPAR calculation.
    
    Args:
        con: DuckDB connection
        lat: Query latitude
        lon: Query longitude
        target_dates: Dates to get metrics for
        as_of_date: Only include bookings before this date
        radius_km: Search radius
        room_type: Query hotel room type for similarity
        room_size: Query hotel room size for similarity
        min_peers: Minimum peers required (expands radius if needed)
    
    Returns:
        Tuple of (list of individual PeerMetrics, weighted average PeerMetrics)
    """
    # Get RevPAR metrics for nearby hotels
    daily_df, peer_list = get_peer_revpar_metrics(
        con, target_dates, as_of_date, lat, lon, radius_km, room_type
    )
    
    # If not enough peers, try expanding radius
    current_radius = radius_km
    max_radius = 50.0  # Don't go beyond 50km
    
    while len(peer_list) < min_peers and current_radius < max_radius:
        current_radius *= 1.5
        daily_df, peer_list = get_peer_revpar_metrics(
            con, target_dates, as_of_date, lat, lon, current_radius, room_type
        )
    
    # Calculate weighted average
    avg_metrics = get_weighted_peer_average(peer_list, weight_by_similarity=True)
    
    return peer_list, avg_metrics

