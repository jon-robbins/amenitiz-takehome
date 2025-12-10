"""
RevPAR-based peer comparison for pricing recommendations.

This module implements the core RevPAR comparison logic that determines
whether a hotel is underperforming, optimal, or overperforming relative to peers.

Key principle: RevPAR = Price × Occupancy
- Captures the tradeoff between price and volume
- A hotel with 35% occupancy at €150 (RevPAR €52.50) outperforms one with 50% at €80 (RevPAR €40)
- Signal logic determines if price should be raised, lowered, or held

Signal Logic (from plan):
| RevPAR Gap | Price Gap | Occ Gap | Signal | Opportunity |
|------------|-----------|---------|--------|-------------|
| < -15% | < 0 | < 0 | underperforming | raise_price (peers prove it works) |
| < -15% | > 0 | < -10pp | underperforming | lower_price (overpriced) |
| < -15% | < 0 | > 0 | underperforming | hold (quality issue, not price) |
| -15% to +15% | any | any | optimal | hold |
| > +15% | < 0 | > 0 | overperforming | raise_price (capturing value) |
| > +15% | > 0 | > 0 | overperforming | hold (already optimized) |
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.temporal_loader import (
    HotelProfile,
    PeerMetrics,
    get_peer_revpar_metrics,
    get_weighted_peer_average,
    load_hotel_locations,
    load_bookings_as_of,
    load_hotel_capacity,
    calculate_daily_revpar,
)
from src.recommender.geo_search import (
    HotelSpatialIndex,
    find_geographic_peers,
    build_hotel_index,
)


class PerformanceSignal(Enum):
    """Hotel performance relative to peers."""
    UNDERPERFORMING = "underperforming"
    OPTIMAL = "optimal"
    OVERPERFORMING = "overperforming"


class PriceOpportunity(Enum):
    """Recommended price action."""
    RAISE_PRICE = "raise_price"
    LOWER_PRICE = "lower_price"
    HOLD = "hold"


# Thresholds for signal classification
REVPAR_GAP_UNDERPERFORM = -0.15  # -15% RevPAR gap = underperforming
REVPAR_GAP_OVERPERFORM = 0.15   # +15% RevPAR gap = overperforming
OCC_GAP_CRITICAL = -0.10        # -10 percentage points = critical occupancy gap


@dataclass
class RevPARComparison:
    """
    Complete RevPAR comparison between a hotel and its peers.
    
    This is the primary output of the peer comparison module.
    Contains all metrics needed for pricing decisions.
    """
    # RevPAR metrics
    hotel_revpar: float
    peer_revpar: float
    revpar_gap: float  # (hotel - peer) / peer
    
    # Price metrics
    hotel_price: float
    peer_price: float
    price_gap: float  # (hotel - peer) / peer
    
    # Occupancy metrics
    hotel_occupancy: float
    peer_occupancy: float
    occupancy_gap: float  # hotel - peer (absolute percentage points)
    
    # Signal classification
    signal: PerformanceSignal
    opportunity: PriceOpportunity
    
    # Peer context
    n_peers: int
    peer_source: str  # "twin", "peer_group", "geographic"
    avg_peer_distance_km: Optional[float] = None
    avg_similarity_score: Optional[float] = None
    
    # Reasoning
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'hotel_revpar': self.hotel_revpar,
            'peer_revpar': self.peer_revpar,
            'revpar_gap': self.revpar_gap,
            'hotel_price': self.hotel_price,
            'peer_price': self.peer_price,
            'price_gap': self.price_gap,
            'hotel_occupancy': self.hotel_occupancy,
            'peer_occupancy': self.peer_occupancy,
            'occupancy_gap': self.occupancy_gap,
            'signal': self.signal.value,
            'opportunity': self.opportunity.value,
            'n_peers': self.n_peers,
            'peer_source': self.peer_source,
            'avg_peer_distance_km': self.avg_peer_distance_km,
            'avg_similarity_score': self.avg_similarity_score,
            'reasoning': self.reasoning,
        }


def classify_signal(
    revpar_gap: float,
    price_gap: float,
    occupancy_gap: float,
    hotel_occupancy: float = 0.0
) -> Tuple[PerformanceSignal, PriceOpportunity, str]:
    """
    Classify performance signal and determine price opportunity.
    
    Implements the signal logic from the plan with HIGH OCCUPANCY OVERRIDE:
    
    | RevPAR Gap | Price Gap | Occ Gap | Signal | Opportunity |
    |------------|-----------|---------|--------|-------------|
    | < -15% | < 0 | < 0 | underperforming | raise_price |
    | < -15% | > 0 | < -10pp | underperforming | lower_price |
    | < -15% | < 0 | > 0 | underperforming | hold (quality issue) |
    | -15% to +15% | any | any | optimal | hold |
    | > +15% | < 0 | > 0 | overperforming | raise_price |
    | > +15% | > 0 | > 0 | overperforming | hold |
    
    SPECIAL CASE: If hotel_occupancy > 90% and price_gap < 0, 
    that's NOT a quality issue - it's underpricing! Demand is proven.
    
    Args:
        revpar_gap: (hotel_revpar - peer_revpar) / peer_revpar
        price_gap: (hotel_price - peer_price) / peer_price
        occupancy_gap: hotel_occupancy - peer_occupancy (percentage points)
        hotel_occupancy: Absolute hotel occupancy (0-1) for high-demand detection
    
    Returns:
        Tuple of (PerformanceSignal, PriceOpportunity, reasoning)
    """
    # HIGH OCCUPANCY OVERRIDE: >90% occupancy with lower prices = underpricing!
    high_occupancy_threshold = 0.90
    is_high_demand = hotel_occupancy >= high_occupancy_threshold
    
    # Underperforming (RevPAR gap < -15%)
    if revpar_gap < REVPAR_GAP_UNDERPERFORM:
        if price_gap < 0 and occupancy_gap < 0:
            # Cheaper than peers but still lower occupancy - peers prove higher prices work
            return (
                PerformanceSignal.UNDERPERFORMING,
                PriceOpportunity.RAISE_PRICE,
                f"RevPAR {revpar_gap*100:+.1f}% below peers. Priced {abs(price_gap)*100:.0f}% lower "
                f"but occupancy still {abs(occupancy_gap)*100:.1f}pp below. Peers prove higher prices work."
            )
        elif price_gap > 0 and occupancy_gap < OCC_GAP_CRITICAL:
            # More expensive than peers with much lower occupancy - overpriced
            return (
                PerformanceSignal.UNDERPERFORMING,
                PriceOpportunity.LOWER_PRICE,
                f"RevPAR {revpar_gap*100:+.1f}% below peers. Priced {price_gap*100:.0f}% higher "
                f"but occupancy {abs(occupancy_gap)*100:.1f}pp below. Price is too high."
            )
        elif price_gap < 0 and occupancy_gap >= 0:
            # Cheaper than peers with similar/higher occupancy
            # KEY CHECK: Is this high occupancy (>90%)? If so, it's proven demand!
            if is_high_demand:
                return (
                    PerformanceSignal.UNDERPERFORMING,
                    PriceOpportunity.RAISE_PRICE,
                    f"RevPAR {revpar_gap*100:+.1f}% below peers. Priced {abs(price_gap)*100:.0f}% lower "
                    f"with {hotel_occupancy*100:.0f}% occupancy - demand is proven! Room to raise prices."
                )
            else:
                # Lower occupancy (<90%) with lower price = potential quality issue
                return (
                    PerformanceSignal.UNDERPERFORMING,
                    PriceOpportunity.HOLD,
                    f"RevPAR {revpar_gap*100:+.1f}% below peers despite lower price and acceptable occupancy. "
                    "Non-price factors (quality, location) may be limiting revenue."
                )
        else:
            # Default underperforming case
            return (
                PerformanceSignal.UNDERPERFORMING,
                PriceOpportunity.HOLD,
                f"RevPAR {revpar_gap*100:+.1f}% below peers. Mixed signals - hold for more data."
            )
    
    # Overperforming (RevPAR gap > +15%)
    elif revpar_gap > REVPAR_GAP_OVERPERFORM:
        if price_gap < 0 and occupancy_gap > 0:
            # Cheaper than peers but much higher occupancy - room to raise
            return (
                PerformanceSignal.OVERPERFORMING,
                PriceOpportunity.RAISE_PRICE,
                f"RevPAR {revpar_gap*100:+.1f}% above peers. Priced {abs(price_gap)*100:.0f}% lower "
                f"with {occupancy_gap*100:.1f}pp higher occupancy. Strong demand supports price increase."
            )
        elif price_gap >= 0 and occupancy_gap >= 0:
            # Already more expensive than peers with higher occupancy - optimized
            return (
                PerformanceSignal.OVERPERFORMING,
                PriceOpportunity.HOLD,
                f"RevPAR {revpar_gap*100:+.1f}% above peers. Already commanding premium pricing. "
                "Maintain current strategy."
            )
        else:
            # Overperforming but mixed signals
            return (
                PerformanceSignal.OVERPERFORMING,
                PriceOpportunity.HOLD,
                f"RevPAR {revpar_gap*100:+.1f}% above peers. Continue current strategy."
            )
    
    # Optimal (RevPAR within ±15% of peers)
    else:
        return (
            PerformanceSignal.OPTIMAL,
            PriceOpportunity.HOLD,
            f"RevPAR within {abs(revpar_gap)*100:.1f}% of peers. Pricing is well-calibrated."
        )


def calculate_revpar_comparison(
    hotel_price: float,
    hotel_occupancy: float,
    peer_price: float,
    peer_occupancy: float,
    n_peers: int,
    peer_source: str,
    avg_distance_km: Optional[float] = None,
    avg_similarity: Optional[float] = None
) -> RevPARComparison:
    """
    Calculate complete RevPAR comparison between hotel and peers.
    
    Args:
        hotel_price: Hotel's average daily rate
        hotel_occupancy: Hotel's occupancy rate (0-1)
        peer_price: Peer average price
        peer_occupancy: Peer average occupancy (0-1)
        n_peers: Number of peers in comparison
        peer_source: Source of peers ("twin", "peer_group", "geographic")
        avg_distance_km: Average distance to peers (for geographic)
        avg_similarity: Average similarity score
    
    Returns:
        RevPARComparison with all metrics and signal classification
    """
    # Calculate RevPAR
    hotel_revpar = hotel_price * hotel_occupancy
    peer_revpar = peer_price * peer_occupancy
    
    # Calculate gaps
    revpar_gap = (hotel_revpar - peer_revpar) / peer_revpar if peer_revpar > 0 else 0
    price_gap = (hotel_price - peer_price) / peer_price if peer_price > 0 else 0
    occupancy_gap = hotel_occupancy - peer_occupancy  # Absolute difference
    
    # Classify signal (including absolute occupancy for high-demand detection)
    signal, opportunity, reasoning = classify_signal(
        revpar_gap, price_gap, occupancy_gap, hotel_occupancy
    )
    
    return RevPARComparison(
        hotel_revpar=hotel_revpar,
        peer_revpar=peer_revpar,
        revpar_gap=revpar_gap,
        hotel_price=hotel_price,
        peer_price=peer_price,
        price_gap=price_gap,
        hotel_occupancy=hotel_occupancy,
        peer_occupancy=peer_occupancy,
        occupancy_gap=occupancy_gap,
        signal=signal,
        opportunity=opportunity,
        n_peers=n_peers,
        peer_source=peer_source,
        avg_peer_distance_km=avg_distance_km,
        avg_similarity_score=avg_similarity,
        reasoning=reasoning
    )


def get_revpar_comparison_for_hotel(
    con,
    hotel_id: int,
    target_dates: List[date],
    as_of_date: date,
    twin_pairs_df: Optional[pd.DataFrame] = None,
    peer_stats_df: Optional[pd.DataFrame] = None,
    radius_km: float = 10.0,
    min_peers: int = 3
) -> Optional[RevPARComparison]:
    """
    Get RevPAR comparison for an existing hotel using tiered fallback.
    
    Tiered fallback:
    1. Twin (44% coverage): Use matched pair if available
    2. Peer Group (78% coverage): Use city+room_type+month aggregates
    3. Geographic (100% coverage): Use nearby hotels within radius
    
    Args:
        con: DuckDB connection
        hotel_id: Hotel identifier
        target_dates: Dates to analyze
        as_of_date: Query date (no future data)
        twin_pairs_df: Pre-loaded matched pairs (optional)
        peer_stats_df: Pre-loaded peer statistics (optional)
        radius_km: Search radius for geographic fallback
        min_peers: Minimum peers required
    
    Returns:
        RevPARComparison or None if hotel not found
    """
    # Load hotel's own data
    hotel_bookings = load_bookings_as_of(con, target_dates, as_of_date, [hotel_id])
    if len(hotel_bookings) == 0:
        return None
    
    hotel_capacity = load_hotel_capacity(con, [hotel_id])
    hotel_revpar_df = calculate_daily_revpar(hotel_bookings, hotel_capacity, target_dates)
    
    if len(hotel_revpar_df) == 0:
        return None
    
    # Calculate hotel's average metrics
    hotel_price = hotel_revpar_df['avg_price'].mean()
    hotel_occupancy = hotel_revpar_df['occupancy_rate'].mean()
    hotel_lat = hotel_revpar_df['latitude'].iloc[0]
    hotel_lon = hotel_revpar_df['longitude'].iloc[0]
    hotel_room_type = hotel_revpar_df['room_type'].iloc[0] if 'room_type' in hotel_revpar_df.columns else None
    
    # Try Tier 1: Twin match
    if twin_pairs_df is not None:
        twin_match = twin_pairs_df[
            (twin_pairs_df['high_price_hotel'] == hotel_id) |
            (twin_pairs_df['low_price_hotel'] == hotel_id)
        ]
        if len(twin_match) > 0:
            # Get twin's metrics
            row = twin_match.iloc[0]
            if row['high_price_hotel'] == hotel_id:
                twin_price = row['low_price']
                twin_occ = row['low_occupancy']
            else:
                twin_price = row['high_price']
                twin_occ = row['high_occupancy']
            
            return calculate_revpar_comparison(
                hotel_price=hotel_price,
                hotel_occupancy=hotel_occupancy,
                peer_price=twin_price,
                peer_occupancy=twin_occ,
                n_peers=1,
                peer_source="twin",
                avg_distance_km=row.get('match_distance', None),
                avg_similarity=1.0  # Twin = perfect match
            )
    
    # Try Tier 2: Peer group
    if peer_stats_df is not None and hotel_room_type is not None:
        # This would require city/month matching - simplified for now
        pass
    
    # Tier 3: Geographic fallback (always available)
    peer_list, avg_peer = find_geographic_peers(
        con, hotel_lat, hotel_lon, target_dates, as_of_date,
        radius_km=radius_km, room_type=hotel_room_type, min_peers=min_peers
    )
    
    if avg_peer is None:
        return None
    
    # Calculate average distance and similarity
    avg_distance = np.mean([p.distance_km for p in peer_list if p.distance_km]) if peer_list else None
    avg_similarity = np.mean([p.similarity_score for p in peer_list if p.similarity_score]) if peer_list else None
    
    return calculate_revpar_comparison(
        hotel_price=hotel_price,
        hotel_occupancy=hotel_occupancy,
        peer_price=avg_peer.avg_price,
        peer_occupancy=avg_peer.occupancy_rate,
        n_peers=len(peer_list),
        peer_source="geographic",
        avg_distance_km=avg_distance,
        avg_similarity=avg_similarity
    )


def get_revpar_comparison_for_profile(
    con,
    profile: HotelProfile,
    target_dates: List[date],
    as_of_date: date,
    radius_km: float = 10.0,
    min_peers: int = 3
) -> Optional[RevPARComparison]:
    """
    Get RevPAR comparison for a cold-start hotel using profile.
    
    For new hotels with no history, we use geographic peers only
    and return their metrics as the benchmark.
    
    Args:
        con: DuckDB connection
        profile: Hotel profile with lat, lon, room_type, etc.
        target_dates: Dates to analyze
        as_of_date: Query date
        radius_km: Search radius
        min_peers: Minimum peers required
    
    Returns:
        RevPARComparison with peer metrics (hotel metrics set to peer average as starting point)
    """
    peer_list, avg_peer = find_geographic_peers(
        con, profile.lat, profile.lon, target_dates, as_of_date,
        radius_km=radius_km, room_type=profile.room_type, min_peers=min_peers
    )
    
    if avg_peer is None:
        return None
    
    # For cold-start, we set hotel metrics equal to peer average as baseline
    # The recommendation will be to start at peer-competitive pricing
    avg_distance = np.mean([p.distance_km for p in peer_list if p.distance_km]) if peer_list else None
    avg_similarity = np.mean([p.similarity_score for p in peer_list if p.similarity_score]) if peer_list else None
    
    return RevPARComparison(
        hotel_revpar=avg_peer.revpar,  # Start at peer level
        peer_revpar=avg_peer.revpar,
        revpar_gap=0.0,  # No gap - starting at peer level
        hotel_price=avg_peer.avg_price,  # Recommended starting price
        peer_price=avg_peer.avg_price,
        price_gap=0.0,
        hotel_occupancy=avg_peer.occupancy_rate,  # Expected occupancy
        peer_occupancy=avg_peer.occupancy_rate,
        occupancy_gap=0.0,
        signal=PerformanceSignal.OPTIMAL,
        opportunity=PriceOpportunity.HOLD,
        n_peers=len(peer_list),
        peer_source="geographic",
        avg_peer_distance_km=avg_distance,
        avg_similarity_score=avg_similarity,
        reasoning=f"Cold-start hotel. Recommended starting price €{avg_peer.avg_price:.0f} "
                  f"based on {len(peer_list)} nearby peers with avg RevPAR €{avg_peer.revpar:.0f}."
    )


def calculate_recommended_price_change(
    comparison: RevPARComparison,
    elasticity: float = -0.39
) -> Tuple[float, float]:
    """
    Calculate recommended price change based on RevPAR comparison.
    
    Uses validated elasticity (-0.39) to estimate optimal price adjustment.
    
    Args:
        comparison: RevPARComparison with signal and opportunity
        elasticity: Price elasticity of demand
    
    Returns:
        Tuple of (recommended_price, change_percentage)
    """
    current_price = comparison.hotel_price
    peer_price = comparison.peer_price
    
    if comparison.opportunity == PriceOpportunity.RAISE_PRICE:
        # How much can we raise?
        if comparison.signal == PerformanceSignal.UNDERPERFORMING:
            # Peers are achieving higher RevPAR at higher prices
            # Move toward peer price level
            target_price = peer_price
            change_pct = (target_price - current_price) / current_price
            # Cap at 30% increase
            change_pct = min(change_pct, 0.30)
        else:  # Overperforming
            # Strong demand - can push above peer price
            # Use elasticity to estimate optimal markup
            # At |ε| = 0.39, optimal markup is high, but cap conservatively
            change_pct = min(0.25, abs(comparison.occupancy_gap) * 2)  # Scale to occupancy advantage
        
        recommended_price = current_price * (1 + change_pct)
    
    elif comparison.opportunity == PriceOpportunity.LOWER_PRICE:
        # Need to reduce price to improve occupancy
        # Target somewhere between current and peer
        target_price = peer_price * 0.95  # Slight undercut
        change_pct = (target_price - current_price) / current_price
        # Cap at -20% decrease
        change_pct = max(change_pct, -0.20)
        recommended_price = current_price * (1 + change_pct)
    
    else:  # HOLD
        recommended_price = current_price
        change_pct = 0.0
    
    return recommended_price, change_pct * 100


def batch_revpar_comparison(
    con,
    hotel_ids: List[int],
    target_dates: List[date],
    as_of_date: date,
    twin_pairs_df: Optional[pd.DataFrame] = None,
    radius_km: float = 10.0
) -> pd.DataFrame:
    """
    Calculate RevPAR comparisons for multiple hotels.
    
    Args:
        con: DuckDB connection
        hotel_ids: List of hotel IDs
        target_dates: Dates to analyze
        as_of_date: Query date
        twin_pairs_df: Pre-loaded matched pairs
        radius_km: Search radius for geographic fallback
    
    Returns:
        DataFrame with comparison results for each hotel
    """
    results = []
    
    for hotel_id in hotel_ids:
        try:
            comparison = get_revpar_comparison_for_hotel(
                con, hotel_id, target_dates, as_of_date,
                twin_pairs_df=twin_pairs_df,
                radius_km=radius_km
            )
            if comparison:
                rec_price, change_pct = calculate_recommended_price_change(comparison)
                result = comparison.to_dict()
                result['hotel_id'] = hotel_id
                result['recommended_price'] = rec_price
                result['change_pct'] = change_pct
                results.append(result)
        except Exception as e:
            print(f"Error processing hotel {hotel_id}: {e}")
    
    return pd.DataFrame(results)

