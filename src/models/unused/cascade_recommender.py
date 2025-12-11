"""
Cascade Price Recommender - Main Orchestrator.

This module combines the temporal and quality models into a unified
recommendation system that routes hotels to the appropriate path:

- Path A (Temporal): Hotels with 8+ weeks history → predict temporal adjustment
- Path B (Quality): Cold-start hotels → predict quality premium vs peers

Final prices are then passed through RevPAR optimization to maximize revenue.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.temporal_model import TemporalModel, TemporalPrediction
from src.models.quality_model import QualityModel, QualityPrediction
from src.models.anchors import (
    get_self_anchor,
    get_peer_anchor,
    MIN_WEEKS_FOR_SELF_ANCHOR,
    MIN_PEERS
)
from src.models.evaluation.time_backtest import (
    get_adjusted_elasticity,
    DEFAULT_ELASTICITY
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Safety bounds for final price recommendations
MIN_PRICE = 30.0
MAX_PRICE = 500.0

# Maximum price change from anchor
MAX_INCREASE_PCT = 0.35
MAX_DECREASE_PCT = 0.25


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriceRecommendation:
    """Complete price recommendation result."""
    hotel_id: Optional[int]
    target_date: date
    
    # Recommended price
    recommended_price: float
    
    # Path information
    path: str  # "path_a", "path_b", "fallback"
    anchor_price: float
    anchor_type: str
    multiplier: float
    
    # Optimization
    base_price: float  # Before RevPAR optimization
    optimization_adjustment: float  # RevPAR optimization delta
    
    # Confidence and context
    confidence: float
    n_peers: int
    
    # Diagnostics
    details: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'hotel_id': self.hotel_id,
            'target_date': self.target_date.isoformat(),
            'recommended_price': self.recommended_price,
            'path': self.path,
            'anchor_price': self.anchor_price,
            'anchor_type': self.anchor_type,
            'multiplier': self.multiplier,
            'base_price': self.base_price,
            'optimization_adjustment': self.optimization_adjustment,
            'confidence': self.confidence,
            'n_peers': self.n_peers,
            'details': self.details
        }


# =============================================================================
# REVPAR OPTIMIZER
# =============================================================================

def optimize_for_revpar(
    base_price: float,
    expected_occupancy: float,
    peer_occupancy: float,
    elasticity: float = DEFAULT_ELASTICITY,
    segment: Optional[str] = None,
    max_increase: float = 0.20,
    max_decrease: float = 0.15
) -> Tuple[float, float]:
    """
    Optimize price for maximum RevPAR using elasticity.
    
    Logic:
    - If expected_occupancy > 85%: Raise price (capture surplus demand)
    - If expected_occupancy < 50%: Lower price (stimulate demand)
    - Otherwise: Hold at base price
    
    Uses occupancy-adjusted elasticity for realistic simulation.
    
    Args:
        base_price: Starting price from model
        expected_occupancy: Forecasted occupancy (0-1)
        peer_occupancy: Peer average occupancy for reference
        elasticity: Base elasticity
        segment: Optional segment for elasticity adjustment
        max_increase: Maximum price increase (fraction)
        max_decrease: Maximum price decrease (fraction)
    
    Returns:
        Tuple of (optimized_price, adjustment_pct)
    """
    # Get occupancy-adjusted elasticity
    adj_elasticity = get_adjusted_elasticity(elasticity, expected_occupancy, segment)
    
    # Determine optimization direction based on occupancy
    if expected_occupancy >= 0.85:
        # High demand - try to raise price
        # Grid search for optimal increase
        best_revpar = base_price * expected_occupancy
        best_adjustment = 0.0
        
        for pct in np.arange(0.05, max_increase + 0.01, 0.05):
            test_price = base_price * (1 + pct)
            # Simulate new occupancy
            occ_change = adj_elasticity * pct
            test_occ = expected_occupancy * (1 + occ_change)
            test_occ = np.clip(test_occ, 0.01, 0.99)
            
            test_revpar = test_price * test_occ
            if test_revpar > best_revpar:
                best_revpar = test_revpar
                best_adjustment = pct
        
        return base_price * (1 + best_adjustment), best_adjustment
    
    elif expected_occupancy < 0.50:
        # Low demand - consider price reduction
        # Only reduce if significantly below peer occupancy
        if expected_occupancy < peer_occupancy * 0.8:
            # Simulate price reduction
            best_revpar = base_price * expected_occupancy
            best_adjustment = 0.0
            
            for pct in np.arange(-0.05, -max_decrease - 0.01, -0.05):
                test_price = base_price * (1 + pct)
                occ_change = adj_elasticity * pct  # Negative elasticity * negative change = positive
                test_occ = expected_occupancy * (1 + occ_change)
                test_occ = np.clip(test_occ, 0.01, 0.99)
                
                test_revpar = test_price * test_occ
                if test_revpar > best_revpar:
                    best_revpar = test_revpar
                    best_adjustment = pct
            
            return base_price * (1 + best_adjustment), best_adjustment
    
    # Default: no adjustment
    return base_price, 0.0


# =============================================================================
# CASCADE RECOMMENDER
# =============================================================================

class CascadeRecommender:
    """
    Main recommendation engine combining Path A and Path B models.
    
    Usage:
        recommender = CascadeRecommender()
        recommender.fit(train_df)
        
        rec = recommender.recommend(hotel_id, target_date, history_df)
    """
    
    def __init__(
        self,
        min_weeks_history: int = MIN_WEEKS_FOR_SELF_ANCHOR,
        min_peers: int = MIN_PEERS,
        enable_revpar_optimization: bool = True
    ):
        """
        Initialize cascade recommender.
        
        Args:
            min_weeks_history: Weeks required for Path A
            min_peers: Peers required for Path B
            enable_revpar_optimization: Apply RevPAR optimization
        """
        self.min_weeks_history = min_weeks_history
        self.min_peers = min_peers
        self.enable_revpar_optimization = enable_revpar_optimization
        
        self.temporal_model = TemporalModel(min_weeks_history=min_weeks_history)
        self.quality_model = QualityModel(min_peers=min_peers)
        
        self.is_fitted = False
        self.train_df = None
    
    def fit(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True
    ) -> 'CascadeRecommender':
        """
        Fit both Path A and Path B models.
        
        Args:
            train_df: Training data
            verbose: Print progress
        
        Returns:
            self
        """
        if verbose:
            print("=" * 60)
            print("FITTING CASCADE RECOMMENDER")
            print("=" * 60)
        
        # Fit temporal model (Path A)
        if verbose:
            print("\n--- Path A: Temporal Model ---")
        self.temporal_model.fit(train_df, verbose=verbose)
        
        # Fit quality model (Path B)
        if verbose:
            print("\n--- Path B: Quality Model ---")
        self.quality_model.fit(train_df, verbose=verbose)
        
        self.train_df = train_df
        self.is_fitted = True
        
        if verbose:
            print("\n✓ Cascade recommender fitted successfully!")
        
        return self
    
    def _get_hotel_features(
        self,
        hotel_id: int,
        history_df: pd.DataFrame
    ) -> Dict:
        """Extract hotel features from history."""
        hotel_data = history_df[history_df['hotel_id'] == hotel_id]
        
        if len(hotel_data) == 0:
            return {}
        
        row = hotel_data.iloc[0]
        
        return {
            'amenities_score': row.get('amenities_score', 0),
            'view_quality_ordinal': row.get('view_quality_ordinal', 0),
            'room_size': row.get('avg_room_size', 25),
            'room_capacity_pax': row.get('room_capacity_pax', 2),
            'dist_center_km': row.get('dist_center_km', 0),
            'total_rooms': row.get('total_rooms', 10),
        }
    
    def recommend(
        self,
        hotel_id: int,
        target_date: date,
        history_df: Optional[pd.DataFrame] = None,
        hotel_features: Optional[Dict] = None,
        expected_occupancy: Optional[float] = None
    ) -> PriceRecommendation:
        """
        Get price recommendation for a hotel.
        
        Args:
            hotel_id: Hotel to recommend for
            target_date: Target date
            history_df: Historical data (uses train_df if None)
            hotel_features: Optional pre-computed features
            expected_occupancy: Optional occupancy forecast for optimization
        
        Returns:
            PriceRecommendation
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        if history_df is None:
            history_df = self.train_df
        
        # Check history availability
        weeks_history = len(history_df[
            (history_df['hotel_id'] == hotel_id) &
            (history_df['week_start'] < pd.Timestamp(target_date))
        ])
        
        # Get hotel location
        hotel_data = history_df[history_df['hotel_id'] == hotel_id]
        if len(hotel_data) > 0:
            lat = hotel_data['latitude'].iloc[0]
            lon = hotel_data['longitude'].iloc[0]
            city = hotel_data.get('city', pd.Series(['unknown'])).iloc[0]
        else:
            # No data at all - pure cold start
            return self._fallback_recommendation(hotel_id, target_date, hotel_features)
        
        # Route to appropriate path
        if weeks_history >= self.min_weeks_history:
            # Path A: Temporal model
            return self._path_a_recommendation(
                hotel_id, target_date, history_df, lat, lon, expected_occupancy
            )
        else:
            # Path B: Quality model
            if hotel_features is None:
                hotel_features = self._get_hotel_features(hotel_id, history_df)
            
            return self._path_b_recommendation(
                hotel_id, target_date, history_df, lat, lon,
                hotel_features, expected_occupancy
            )
    
    def _get_peer_occupancy(
        self,
        lat: float,
        lon: float,
        target_date: date,
        history_df: pd.DataFrame,
        radius_km: float = 10.0
    ) -> float:
        """
        Get average peer occupancy from nearby hotels.
        
        Falls back to market average if no peers found.
        """
        # Get same week from history
        week_start = pd.Timestamp(target_date)
        
        # Look for data within ±1 week
        week_data = history_df[
            (history_df['week_start'] >= week_start - pd.Timedelta(days=7)) &
            (history_df['week_start'] <= week_start + pd.Timedelta(days=7))
        ].copy()
        
        if len(week_data) == 0:
            # Fall back to overall average
            return history_df['occupancy_rate'].median()
        
        # Calculate distances
        lat_km, lon_km = 111.0, 111.0 * np.cos(np.radians(lat))
        week_data['distance_km'] = np.sqrt(
            ((week_data['latitude'] - lat) * lat_km) ** 2 +
            ((week_data['longitude'] - lon) * lon_km) ** 2
        )
        
        # Get peers within radius
        peers = week_data[week_data['distance_km'] <= radius_km]
        
        if len(peers) >= 3:
            return peers['occupancy_rate'].median()
        
        # Fall back to market average
        return history_df['occupancy_rate'].median()
    
    def _path_a_recommendation(
        self,
        hotel_id: int,
        target_date: date,
        history_df: pd.DataFrame,
        lat: float,
        lon: float,
        expected_occupancy: Optional[float]
    ) -> PriceRecommendation:
        """Generate recommendation using Path A (temporal model)."""
        # Get temporal prediction
        temporal_pred = self.temporal_model.predict(hotel_id, target_date, history_df)
        
        base_price = temporal_pred.recommended_price
        anchor_price = temporal_pred.self_anchor
        multiplier = temporal_pred.multiplier
        
        # RevPAR optimization
        if self.enable_revpar_optimization and expected_occupancy is not None:
            # Get actual peer occupancy from data
            peer_occ = self._get_peer_occupancy(lat, lon, target_date, history_df)
            
            optimized_price, opt_adj = optimize_for_revpar(
                base_price, expected_occupancy, peer_occ
            )
        else:
            optimized_price = base_price
            opt_adj = 0.0
            peer_occ = None
        
        # Safety bounds
        final_price = np.clip(optimized_price, MIN_PRICE, MAX_PRICE)
        
        return PriceRecommendation(
            hotel_id=hotel_id,
            target_date=target_date,
            recommended_price=final_price,
            path="path_a",
            anchor_price=anchor_price,
            anchor_type="self_median",
            multiplier=multiplier,
            base_price=base_price,
            optimization_adjustment=opt_adj,
            confidence=temporal_pred.confidence,
            n_peers=0,
            details={
                'temporal_features': temporal_pred.features_used,
                'expected_occupancy': expected_occupancy,
                'peer_occupancy': peer_occ
            }
        )
    
    def _path_b_recommendation(
        self,
        hotel_id: int,
        target_date: date,
        history_df: pd.DataFrame,
        lat: float,
        lon: float,
        hotel_features: Dict,
        expected_occupancy: Optional[float]
    ) -> PriceRecommendation:
        """Generate recommendation using Path B (quality model)."""
        # Get quality prediction
        quality_pred = self.quality_model.predict(
            lat, lon, target_date, history_df, hotel_features
        )
        
        base_price = quality_pred.recommended_price
        anchor_price = quality_pred.peer_anchor
        multiplier = quality_pred.multiplier
        
        # RevPAR optimization
        if self.enable_revpar_optimization and expected_occupancy is not None:
            # Get actual peer occupancy from data
            peer_occ = self._get_peer_occupancy(lat, lon, target_date, history_df)
            
            optimized_price, opt_adj = optimize_for_revpar(
                base_price, expected_occupancy, peer_occ
            )
        else:
            optimized_price = base_price
            opt_adj = 0.0
            peer_occ = None
        
        # Safety bounds
        final_price = np.clip(optimized_price, MIN_PRICE, MAX_PRICE)
        
        return PriceRecommendation(
            hotel_id=hotel_id,
            target_date=target_date,
            recommended_price=final_price,
            path="path_b",
            anchor_price=anchor_price,
            anchor_type="peer_median",
            multiplier=multiplier,
            base_price=base_price,
            optimization_adjustment=opt_adj,
            confidence=quality_pred.confidence,
            n_peers=quality_pred.n_peers,
            details={
                'quality_features': quality_pred.features_used,
                'expected_occupancy': expected_occupancy,
                'peer_occupancy': peer_occ
            }
        )
    
    def _fallback_recommendation(
        self,
        hotel_id: int,
        target_date: date,
        hotel_features: Optional[Dict]
    ) -> PriceRecommendation:
        """Fallback when no data available."""
        default_price = 100.0  # Conservative default
        
        return PriceRecommendation(
            hotel_id=hotel_id,
            target_date=target_date,
            recommended_price=default_price,
            path="fallback",
            anchor_price=default_price,
            anchor_type="default",
            multiplier=1.0,
            base_price=default_price,
            optimization_adjustment=0.0,
            confidence=0.05,
            n_peers=0,
            details={'fallback': 'no_data'}
        )
    
    def recommend_batch(
        self,
        hotel_ids: List[int],
        target_date: date,
        history_df: Optional[pd.DataFrame] = None
    ) -> List[PriceRecommendation]:
        """Get recommendations for multiple hotels."""
        return [
            self.recommend(hotel_id, target_date, history_df)
            for hotel_id in hotel_ids
        ]


# =============================================================================
# BACKTEST INTEGRATION
# =============================================================================

def cascade_recommender_fn(
    recommender: CascadeRecommender,
    train_df: pd.DataFrame,
    row: pd.Series,
    week: date
) -> float:
    """
    Wrapper function for use with run_backtest.
    
    Args:
        recommender: Fitted CascadeRecommender
        train_df: Training data
        row: Test row with hotel info
        week: Target week
    
    Returns:
        Recommended price
    """
    rec = recommender.recommend(
        hotel_id=row['hotel_id'],
        target_date=week,
        history_df=train_df,
        expected_occupancy=row.get('occupancy_rate', 0.65)
    )
    return rec.recommended_price


# =============================================================================
# DEMO
# =============================================================================

def main():
    """Demo cascade recommender."""
    from src.models.evaluation.time_backtest import (
        BacktestConfig, 
        load_hotel_week_data,
        run_backtest
    )
    from functools import partial
    
    print("=" * 60)
    print("CASCADE RECOMMENDER DEMO")
    print("=" * 60)
    
    config = BacktestConfig()
    
    print("\nLoading data...")
    train_df = load_hotel_week_data(config, split="train")
    test_df = load_hotel_week_data(config, split="test")
    
    print("\nFitting recommender...")
    recommender = CascadeRecommender(enable_revpar_optimization=False)
    recommender.fit(train_df, verbose=True)
    
    # Sample recommendations
    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 60)
    
    target_date = config.test_start
    
    # Get some hotels
    hotel_counts = train_df.groupby('hotel_id').size()
    
    # Path A hotel (lots of history)
    path_a_hotel = hotel_counts[hotel_counts >= 30].index[0]
    rec_a = recommender.recommend(path_a_hotel, target_date, train_df)
    print(f"\nPath A Hotel {path_a_hotel}:")
    print(f"  Anchor: €{rec_a.anchor_price:.0f} ({rec_a.anchor_type})")
    print(f"  Multiplier: {rec_a.multiplier:.2f}x")
    print(f"  Recommended: €{rec_a.recommended_price:.0f}")
    print(f"  Confidence: {rec_a.confidence:.2f}")
    
    # Path B hotel (little history)
    path_b_hotel = hotel_counts[hotel_counts < 5].index[0]
    rec_b = recommender.recommend(path_b_hotel, target_date, train_df)
    print(f"\nPath B Hotel {path_b_hotel}:")
    print(f"  Anchor: €{rec_b.anchor_price:.0f} ({rec_b.anchor_type})")
    print(f"  Multiplier: {rec_b.multiplier:.2f}x")
    print(f"  Recommended: €{rec_b.recommended_price:.0f}")
    print(f"  Confidence: {rec_b.confidence:.2f}")
    
    # Quick backtest
    print("\n" + "=" * 60)
    print("QUICK BACKTEST (500 samples)")
    print("=" * 60)
    
    test_subset = test_df.head(500)
    
    recommender_fn = partial(cascade_recommender_fn, recommender)
    results, metrics = run_backtest(
        recommender_fn, train_df, test_subset, config, verbose=False
    )
    
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Mean RevPAR Lift: €{metrics.mean_revpar_lift:.2f}")
    print(f"Price MAPE: {metrics.price_mape:.1%}")
    print(f"Path A: {metrics.n_path_a} ({metrics.win_rate_path_a:.1%} win)")
    print(f"Path B: {metrics.n_path_b} ({metrics.win_rate_path_b:.1%} win)")


if __name__ == "__main__":
    main()

