"""
Anchor computation for cascade pricing strategy.

This module provides functions to compute price anchors:
1. Self Anchor (Path A): Hotel's own trailing median price
2. Peer Anchor (Path B): Median price of nearby peers

The anchor serves as the baseline price, which is then adjusted
by the temporal or quality model.

Key concepts:
- Self anchor requires minimum 8 weeks of history
- Peer anchor uses tiered fallback: 10km -> 50km -> city median
- Minimum 5 peers required for peer anchor
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loader import get_clean_connection


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum history for self-anchor
MIN_WEEKS_FOR_SELF_ANCHOR = 8

# Peer search tiers (in km)
PEER_RADIUS_TIERS = [10, 50, 100]
MIN_PEERS = 5

# Lookback period for self-anchor
LOOKBACK_MONTHS = 12


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AnchorResult:
    """Result of anchor computation."""
    anchor_price: float
    anchor_type: str  # "self", "peer_10km", "peer_50km", "city", "fallback"
    confidence: float  # 0-1, based on data quality
    n_observations: int
    details: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'anchor_price': self.anchor_price,
            'anchor_type': self.anchor_type,
            'confidence': self.confidence,
            'n_observations': self.n_observations,
            'details': self.details
        }


@dataclass
class HotelAnchorData:
    """Pre-computed anchor data for a hotel."""
    hotel_id: int
    self_anchor: Optional[float]
    self_anchor_weeks: int
    peer_anchor_10km: Optional[float]
    peer_anchor_50km: Optional[float]
    city_anchor: Optional[float]
    n_peers_10km: int
    n_peers_50km: int
    city: str
    latitude: float
    longitude: float


# =============================================================================
# SELF ANCHOR (Path A)
# =============================================================================

def get_self_anchor(
    hotel_id: int,
    target_date: date,
    history_df: pd.DataFrame,
    lookback_months: int = LOOKBACK_MONTHS,
    min_weeks: int = MIN_WEEKS_FOR_SELF_ANCHOR
) -> AnchorResult:
    """
    Compute hotel's own trailing median price.
    
    Args:
        hotel_id: Hotel to compute anchor for
        target_date: Reference date (anchor computed from data before this)
        history_df: Historical price data with hotel_id, week_start, avg_price
        lookback_months: How far back to look for history
        min_weeks: Minimum weeks required
    
    Returns:
        AnchorResult with self anchor or None if insufficient data
    """
    lookback_start = target_date - timedelta(days=lookback_months * 30)
    
    hotel_history = history_df[
        (history_df['hotel_id'] == hotel_id) &
        (history_df['week_start'] < pd.Timestamp(target_date)) &
        (history_df['week_start'] >= pd.Timestamp(lookback_start))
    ].copy()
    
    n_weeks = len(hotel_history)
    
    if n_weeks < min_weeks:
        return AnchorResult(
            anchor_price=0.0,
            anchor_type="insufficient_data",
            confidence=0.0,
            n_observations=n_weeks,
            details={'required': min_weeks, 'found': n_weeks}
        )
    
    # Compute median and percentiles
    prices = hotel_history['avg_price']
    median_price = prices.median()
    p25 = prices.quantile(0.25)
    p75 = prices.quantile(0.75)
    std = prices.std()
    
    # Confidence based on data quantity and stability
    quantity_score = min(n_weeks / 52, 1.0)  # Max at 1 year
    stability_score = 1.0 - min(std / median_price, 0.5) if median_price > 0 else 0.5
    confidence = 0.6 * quantity_score + 0.4 * stability_score
    
    return AnchorResult(
        anchor_price=median_price,
        anchor_type="self",
        confidence=confidence,
        n_observations=n_weeks,
        details={
            'p25': p25,
            'p50': median_price,
            'p75': p75,
            'std': std,
            'lookback_start': lookback_start.isoformat(),
            'lookback_end': target_date.isoformat()
        }
    )


def compute_self_anchors_batch(
    hotel_ids: List[int],
    target_date: date,
    history_df: pd.DataFrame
) -> Dict[int, AnchorResult]:
    """
    Compute self anchors for multiple hotels efficiently.
    
    Args:
        hotel_ids: List of hotel IDs
        target_date: Reference date
        history_df: Historical data
    
    Returns:
        Dict mapping hotel_id to AnchorResult
    """
    results = {}
    for hotel_id in hotel_ids:
        results[hotel_id] = get_self_anchor(hotel_id, target_date, history_df)
    return results


# =============================================================================
# PEER ANCHOR (Path B)
# =============================================================================

def get_peer_anchor(
    latitude: float,
    longitude: float,
    target_week: date,
    history_df: pd.DataFrame,
    room_type: Optional[str] = None,
    radius_tiers: List[float] = None,
    min_peers: int = MIN_PEERS
) -> AnchorResult:
    """
    Compute peer median price with tiered radius fallback.
    
    Args:
        latitude: Hotel latitude
        longitude: Hotel longitude
        target_week: Week to find peer prices for
        history_df: Historical data with hotel prices
        room_type: Optional room type filter
        radius_tiers: Search radii to try (default: [10, 50, 100] km)
        min_peers: Minimum peers required
    
    Returns:
        AnchorResult with peer anchor
    """
    if radius_tiers is None:
        radius_tiers = PEER_RADIUS_TIERS
    
    # Get data for target week
    week_start = pd.Timestamp(target_week)
    week_data = history_df[history_df['week_start'] == week_start].copy()
    
    if len(week_data) == 0:
        # Try nearby weeks (±1 week)
        one_week = timedelta(days=7)
        week_data = history_df[
            (history_df['week_start'] >= week_start - one_week) &
            (history_df['week_start'] <= week_start + one_week)
        ].copy()
    
    if len(week_data) == 0:
        return AnchorResult(
            anchor_price=0.0,
            anchor_type="no_data",
            confidence=0.0,
            n_observations=0,
            details={'error': 'No peer data for target week'}
        )
    
    # Calculate distances
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians(latitude))
    
    week_data['distance_km'] = np.sqrt(
        ((week_data['latitude'] - latitude) * lat_km) ** 2 +
        ((week_data['longitude'] - longitude) * lon_km) ** 2
    )
    
    # Filter by room type if specified
    if room_type is not None and 'room_type' in week_data.columns:
        room_type_match = week_data[week_data['room_type'] == room_type]
        if len(room_type_match) >= min_peers:
            week_data = room_type_match
    
    # Try each radius tier
    for radius in radius_tiers:
        peers = week_data[
            (week_data['distance_km'] <= radius) &
            (week_data['distance_km'] > 0)  # Exclude self
        ]
        
        if len(peers) >= min_peers:
            median_price = peers['avg_price'].median()
            
            # Confidence based on peer count and proximity
            count_score = min(len(peers) / 20, 1.0)  # Max at 20 peers
            proximity_score = 1.0 - (peers['distance_km'].mean() / radius)
            confidence = 0.5 * count_score + 0.5 * proximity_score
            
            return AnchorResult(
                anchor_price=median_price,
                anchor_type=f"peer_{radius}km",
                confidence=confidence,
                n_observations=len(peers),
                details={
                    'radius_km': radius,
                    'n_peers': len(peers),
                    'median_price': median_price,
                    'mean_distance_km': peers['distance_km'].mean(),
                    'peer_prices': peers['avg_price'].describe().to_dict()
                }
            )
    
    # Fallback to city median if available
    if 'city' in week_data.columns and len(week_data) > 0:
        # Get city from closest hotel
        closest_idx = week_data['distance_km'].idxmin()
        target_city = week_data.loc[closest_idx, 'city']
        
        city_peers = week_data[week_data['city'] == target_city]
        if len(city_peers) >= 3:
            median_price = city_peers['avg_price'].median()
            return AnchorResult(
                anchor_price=median_price,
                anchor_type="city",
                confidence=0.3,
                n_observations=len(city_peers),
                details={
                    'city': target_city,
                    'n_hotels': len(city_peers)
                }
            )
    
    # Ultimate fallback: market median
    if len(week_data) > 0:
        median_price = week_data['avg_price'].median()
        return AnchorResult(
            anchor_price=median_price,
            anchor_type="market",
            confidence=0.1,
            n_observations=len(week_data),
            details={'fallback': 'market_median'}
        )
    
    return AnchorResult(
        anchor_price=0.0,
        anchor_type="no_data",
        confidence=0.0,
        n_observations=0,
        details={'error': 'No peers found at any radius'}
    )


def get_tiered_peer_anchor(
    hotel_id: int,
    latitude: float,
    longitude: float,
    target_week: date,
    history_df: pd.DataFrame
) -> Tuple[Optional[float], str, int]:
    """
    Simplified tiered peer anchor lookup.
    
    Returns:
        Tuple of (anchor_price, anchor_type, n_peers)
    """
    result = get_peer_anchor(latitude, longitude, target_week, history_df)
    return result.anchor_price, result.anchor_type, result.n_observations


# =============================================================================
# MARKET SEASONALITY INDEX
# =============================================================================

def calculate_market_seasonality_index(
    target_week: date,
    history_df: pd.DataFrame,
    lookback_months: int = 12
) -> float:
    """
    Calculate market seasonality index for a week.
    
    Index = (peer prices this week) / (annual average peer prices)
    
    A value of 1.2 means this week prices are typically 20% above average.
    
    Args:
        target_week: Week to calculate for
        history_df: Historical price data
        lookback_months: Period for annual average
    
    Returns:
        Seasonality index (typically 0.7 - 1.5)
    """
    week_of_year = pd.Timestamp(target_week).isocalendar()[1]
    
    # Get prices for this week of year across all years
    history_df = history_df.copy()
    history_df['week_of_year'] = pd.to_datetime(history_df['week_start']).apply(
        lambda x: x.isocalendar()[1]
    )
    
    same_week_prices = history_df[history_df['week_of_year'] == week_of_year]['avg_price']
    annual_avg = history_df['avg_price'].mean()
    
    if len(same_week_prices) == 0 or annual_avg == 0:
        return 1.0
    
    return same_week_prices.mean() / annual_avg


def calculate_seasonality_indices(
    history_df: pd.DataFrame
) -> Dict[int, float]:
    """
    Pre-compute seasonality indices for all weeks of year.
    
    Returns:
        Dict mapping week_of_year (1-53) to seasonality index
    """
    history_df = history_df.copy()
    history_df['week_of_year'] = pd.to_datetime(history_df['week_start']).apply(
        lambda x: x.isocalendar()[1]
    )
    
    annual_avg = history_df['avg_price'].mean()
    
    if annual_avg == 0:
        return {w: 1.0 for w in range(1, 54)}
    
    weekly_avgs = history_df.groupby('week_of_year')['avg_price'].mean()
    
    indices = {}
    for week in range(1, 54):
        if week in weekly_avgs.index:
            indices[week] = weekly_avgs[week] / annual_avg
        else:
            indices[week] = 1.0
    
    return indices


# =============================================================================
# COMBINED ANCHOR SELECTION
# =============================================================================

def select_best_anchor(
    hotel_id: int,
    latitude: float,
    longitude: float,
    target_date: date,
    history_df: pd.DataFrame,
    prefer_self: bool = True
) -> AnchorResult:
    """
    Select the best anchor for a hotel using cascade logic.
    
    Priority (if prefer_self=True):
    1. Self anchor (if >= 8 weeks history)
    2. Peer anchor (10km, then 50km, then city)
    3. Market fallback
    
    Args:
        hotel_id: Hotel to get anchor for
        latitude: Hotel latitude
        longitude: Hotel longitude
        target_date: Target date for recommendation
        history_df: Historical data
        prefer_self: Whether to prefer self-anchor over peers
    
    Returns:
        AnchorResult with selected anchor
    """
    # Try self anchor first
    if prefer_self:
        self_result = get_self_anchor(hotel_id, target_date, history_df)
        if self_result.anchor_type == "self" and self_result.confidence >= 0.5:
            return self_result
    
    # Try peer anchor
    peer_result = get_peer_anchor(latitude, longitude, target_date, history_df)
    
    if peer_result.anchor_price > 0:
        return peer_result
    
    # If we have any self data, use it even with low confidence
    if not prefer_self:
        self_result = get_self_anchor(hotel_id, target_date, history_df)
        if self_result.anchor_price > 0:
            return self_result
    
    # Ultimate fallback
    return AnchorResult(
        anchor_price=100.0,  # Conservative default
        anchor_type="default",
        confidence=0.05,
        n_observations=0,
        details={'fallback': 'default_100'}
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def compute_all_anchors(
    hotels_df: pd.DataFrame,
    history_df: pd.DataFrame,
    target_date: date
) -> pd.DataFrame:
    """
    Compute anchors for all hotels in a dataset.
    
    Args:
        hotels_df: DataFrame with hotel_id, latitude, longitude
        history_df: Historical price data
        target_date: Reference date
    
    Returns:
        DataFrame with anchor results for each hotel
    """
    results = []
    
    for _, hotel in hotels_df.iterrows():
        anchor = select_best_anchor(
            hotel_id=hotel['hotel_id'],
            latitude=hotel['latitude'],
            longitude=hotel['longitude'],
            target_date=target_date,
            history_df=history_df
        )
        
        results.append({
            'hotel_id': hotel['hotel_id'],
            'anchor_price': anchor.anchor_price,
            'anchor_type': anchor.anchor_type,
            'confidence': anchor.confidence,
            'n_observations': anchor.n_observations
        })
    
    return pd.DataFrame(results)


# =============================================================================
# TESTING / DEMO
# =============================================================================

def main():
    """Demo anchor computation."""
    from src.models.evaluation.time_backtest import load_hotel_week_data, BacktestConfig
    
    print("=" * 60)
    print("ANCHOR COMPUTATION DEMO")
    print("=" * 60)
    
    config = BacktestConfig()
    
    # Load training data
    print("\nLoading training data...")
    history_df = load_hotel_week_data(config, split="train")
    
    # Get unique hotels
    hotels = history_df.groupby('hotel_id').agg({
        'latitude': 'first',
        'longitude': 'first',
        'city': 'first'
    }).reset_index()
    
    print(f"\nComputing anchors for {len(hotels):,} hotels...")
    
    # Sample some hotels
    sample_hotels = hotels.sample(min(10, len(hotels)), random_state=42)
    target_date = config.test_start
    
    print(f"\nSample anchors (target: {target_date}):")
    print("-" * 80)
    
    for _, hotel in sample_hotels.iterrows():
        anchor = select_best_anchor(
            hotel['hotel_id'],
            hotel['latitude'],
            hotel['longitude'],
            target_date,
            history_df
        )
        print(f"Hotel {hotel['hotel_id']:5d} ({hotel['city'][:15]:<15}): "
              f"€{anchor.anchor_price:6.0f} ({anchor.anchor_type:<10}) "
              f"conf={anchor.confidence:.2f} n={anchor.n_observations}")
    
    # Seasonality indices
    print("\n" + "=" * 60)
    print("SEASONALITY INDICES")
    print("=" * 60)
    
    indices = calculate_seasonality_indices(history_df)
    
    print("\nWeekly seasonality (selected weeks):")
    for week in [1, 10, 20, 30, 40, 52]:
        print(f"  Week {week:2d}: {indices[week]:.2f}x")


if __name__ == "__main__":
    main()

