"""
Triangulated Scoring for Hotel Pricing Recommendations.

Combines three signals to determine pricing actions:
1. RevPAR Peer Comparison (External Signal) - How does hotel compare to similar peers?
2. XGBoost Fair Value Model (Internal Signal) - What should this hotel charge based on features?
3. Occupancy Model (Demand Signal) - What occupancy is expected at different prices?

The final recommendation uses the occupancy model with elasticity to find the 
**RevPAR-optimal price** - the price that maximizes Price × Occupancy.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from src.recommender.revpar_peers import (
    RevPARComparison,
    PerformanceSignal,
    PriceOpportunity,
    get_revpar_comparison_for_hotel,
    get_revpar_comparison_for_profile,
)
from src.data.temporal_loader import HotelProfile
from src.models.occupancy import OccupancyModel
from src.features.engineering import MARKET_ELASTICITY


class RecommendationScenario(Enum):
    """Classification of recommendation scenarios from plan."""
    A = "A"  # Underperforming + Underpriced → RAISE +20-30%
    B = "B"  # Underperforming + Fair value → RAISE +10-15%
    C = "C"  # Overperforming + Overpriced → LOWER 5-10%
    D = "D"  # Underperforming but quality issue → HOLD


class Confidence(Enum):
    """Confidence level of recommendation."""
    HIGH = "high"       # Twin match or strong peer group
    MEDIUM = "medium"   # Geographic peers with good similarity
    LOW = "low"         # Few peers or low similarity


# Value residual thresholds (from plan)
VALUE_RESIDUAL_UNDERPRICED = 20.0   # €20+ residual = structurally underpriced
VALUE_RESIDUAL_OVERPRICED = -20.0   # €-20 residual = structurally overpriced

# Occupancy safety valve threshold (relative to peers)
OCC_GAP_QUALITY_ISSUE = -0.10  # -10pp = quality issue, not price issue


@dataclass
class FairValueResult:
    """Result of XGBoost fair value prediction."""
    predicted_price: float      # What the model thinks hotel should charge
    actual_price: float         # What hotel is actually charging
    value_residual: float       # predicted - actual (positive = underpriced)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_underpriced(self) -> bool:
        """Hotel is structurally underpriced (hardware worth more)."""
        return self.value_residual > VALUE_RESIDUAL_UNDERPRICED
    
    @property
    def is_overpriced(self) -> bool:
        """Hotel is structurally overpriced."""
        return self.value_residual < VALUE_RESIDUAL_OVERPRICED
    
    @property
    def is_fair_value(self) -> bool:
        """Hotel is priced appropriately for its features."""
        return VALUE_RESIDUAL_OVERPRICED <= self.value_residual <= VALUE_RESIDUAL_UNDERPRICED


@dataclass
class TriangulatedRecommendation:
    """
    Complete triangulated pricing recommendation.
    
    This is the final output combining all three signals.
    """
    # Hotel identification
    hotel_id: Optional[int]
    target_date: date
    
    # Current state
    current_price: float
    current_occupancy: float
    current_revpar: float
    
    # Recommendation
    recommended_price: float
    change_pct: float
    scenario: RecommendationScenario
    confidence: Confidence
    
    # Component signals
    revpar_comparison: RevPARComparison
    fair_value: Optional[FairValueResult]
    
    # Safety valve
    occupancy_safety_triggered: bool
    
    # Explanation
    reasoning: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'hotel_id': self.hotel_id,
            'target_date': self.target_date.isoformat() if self.target_date else None,
            'current_price': self.current_price,
            'current_occupancy': self.current_occupancy,
            'current_revpar': self.current_revpar,
            'recommended_price': self.recommended_price,
            'change_pct': self.change_pct,
            'scenario': self.scenario.value,
            'confidence': self.confidence.value,
            'peer_revpar': self.revpar_comparison.peer_revpar,
            'revpar_gap': self.revpar_comparison.revpar_gap,
            'revpar_signal': self.revpar_comparison.signal.value,
            'fair_value_predicted': self.fair_value.predicted_price if self.fair_value else None,
            'value_residual': self.fair_value.value_residual if self.fair_value else None,
            'n_peers': self.revpar_comparison.n_peers,
            'peer_source': self.revpar_comparison.peer_source,
            'occupancy_safety_triggered': self.occupancy_safety_triggered,
            'reasoning': self.reasoning,
        }


def classify_scenario(
    revpar_comparison: RevPARComparison,
    fair_value: Optional[FairValueResult]
) -> Tuple[RecommendationScenario, str]:
    """
    Classify recommendation into one of 4 scenarios.
    
    | Scenario | RevPAR Signal | Model Signal | Recommendation |
    |----------|---------------|--------------|----------------|
    | **A** | Underperforming, peers higher price | Underpriced (residual > €20) | RAISE +20-30% |
    | **B** | Underperforming, peers similar price | Fair value | RAISE +10-15% |
    | **C** | Overperforming | Overpriced (residual < -€20) | LOWER 5-10% |
    | **D** | Underperforming, occ gap > -10pp | Any | HOLD (quality issue) |
    
    Args:
        revpar_comparison: RevPAR comparison result
        fair_value: Fair value prediction result (optional)
    
    Returns:
        Tuple of (RecommendationScenario, reasoning)
    """
    revpar_signal = revpar_comparison.signal
    revpar_opp = revpar_comparison.opportunity
    occ_gap = revpar_comparison.occupancy_gap
    
    # Check for quality issue first (Scenario D)
    if revpar_signal == PerformanceSignal.UNDERPERFORMING and occ_gap > OCC_GAP_QUALITY_ISSUE:
        # Lower RevPAR despite similar/higher occupancy → quality issue
        return (
            RecommendationScenario.D,
            "Underperforming RevPAR with acceptable occupancy suggests non-price factors. HOLD."
        )
    
    # Determine if underpriced/overpriced based on model
    is_underpriced = fair_value.is_underpriced if fair_value else False
    is_overpriced = fair_value.is_overpriced if fair_value else False
    is_fair = fair_value.is_fair_value if fair_value else True
    
    # Scenario A: Underperforming + Structurally Underpriced
    if revpar_signal == PerformanceSignal.UNDERPERFORMING and is_underpriced:
        return (
            RecommendationScenario.A,
            f"RevPAR {revpar_comparison.revpar_gap*100:+.1f}% below peers. "
            f"Model shows €{fair_value.value_residual:.0f} underpriced. RAISE +20-30%."
        )
    
    # Scenario B: Underperforming + Fair Value
    if revpar_signal == PerformanceSignal.UNDERPERFORMING and is_fair:
        return (
            RecommendationScenario.B,
            f"RevPAR {revpar_comparison.revpar_gap*100:+.1f}% below peers. "
            "Priced at fair value. RAISE +10-15% cautiously."
        )
    
    # Scenario C: Overperforming + Overpriced (or underperforming + overpriced)
    if is_overpriced:
        return (
            RecommendationScenario.C,
            f"Model shows €{abs(fair_value.value_residual):.0f} overpriced. LOWER 5-10%."
        )
    
    # Default to B or D based on RevPAR signal
    if revpar_signal == PerformanceSignal.UNDERPERFORMING:
        if revpar_opp == PriceOpportunity.RAISE_PRICE:
            return (
                RecommendationScenario.B,
                f"RevPAR below peers. Conservative RAISE +10-15%."
            )
        else:
            return (
                RecommendationScenario.D,
                "Mixed signals. HOLD for more data."
            )
    
    # Optimal or Overperforming without overpricing → hold
    return (
        RecommendationScenario.D,
        f"RevPAR is {revpar_comparison.signal.value}. Maintain current strategy."
    )


def calculate_triangulated_price(
    current_price: float,
    scenario: RecommendationScenario,
    revpar_comparison: RevPARComparison,
    fair_value: Optional[FairValueResult] = None
) -> Tuple[float, float]:
    """
    Calculate recommended price based on scenario classification.
    
    Args:
        current_price: Current hotel price
        scenario: Classified scenario
        revpar_comparison: RevPAR comparison result
        fair_value: Fair value prediction (optional)
    
    Returns:
        Tuple of (recommended_price, change_pct)
    """
    peer_price = revpar_comparison.peer_price
    
    if scenario == RecommendationScenario.A:
        # RAISE +20-30%
        if fair_value and fair_value.value_residual > 0:
            # Move toward fair value, capped at 30%
            target = min(fair_value.predicted_price, current_price * 1.30)
            target = max(target, current_price * 1.20)  # At least 20%
        else:
            # Move toward peer price
            target = min(peer_price * 0.95, current_price * 1.30)
            target = max(target, current_price * 1.20)
        
        change_pct = (target - current_price) / current_price * 100
        return target, change_pct
    
    elif scenario == RecommendationScenario.B:
        # RAISE +10-15%
        target = current_price * 1.125  # 12.5% increase
        change_pct = 12.5
        return target, change_pct
    
    elif scenario == RecommendationScenario.C:
        # LOWER 5-10%
        if fair_value:
            # Move toward fair value
            target = max(fair_value.predicted_price, current_price * 0.90)
            target = min(target, current_price * 0.95)  # At most -5%
        else:
            target = current_price * 0.925  # 7.5% decrease
        
        change_pct = (target - current_price) / current_price * 100
        return target, change_pct
    
    else:  # Scenario D - HOLD
        return current_price, 0.0


def determine_confidence(
    revpar_comparison: RevPARComparison,
    fair_value: Optional[FairValueResult] = None
) -> Confidence:
    """
    Determine confidence level based on peer quality and signal agreement.
    
    Args:
        revpar_comparison: RevPAR comparison result
        fair_value: Fair value prediction (optional)
    
    Returns:
        Confidence level
    """
    # High confidence: Twin match
    if revpar_comparison.peer_source == "twin":
        return Confidence.HIGH
    
    # Medium confidence: Good peer group or geographic with high similarity
    if revpar_comparison.peer_source == "peer_group":
        return Confidence.MEDIUM
    
    if revpar_comparison.peer_source == "geographic":
        n_peers = revpar_comparison.n_peers
        avg_similarity = revpar_comparison.avg_similarity_score or 0
        
        if n_peers >= 5 and avg_similarity >= 0.7:
            return Confidence.MEDIUM
        elif n_peers >= 3 and avg_similarity >= 0.5:
            return Confidence.MEDIUM
        else:
            return Confidence.LOW
    
    return Confidence.LOW


def get_triangulated_recommendation(
    revpar_comparison: RevPARComparison,
    fair_value: Optional[FairValueResult] = None,
    hotel_id: Optional[int] = None,
    target_date: Optional[date] = None
) -> TriangulatedRecommendation:
    """
    Generate complete triangulated recommendation.
    
    Combines RevPAR comparison with fair value model and classifies
    into one of 4 scenarios with appropriate price recommendations.
    
    Args:
        revpar_comparison: RevPAR comparison result
        fair_value: Fair value prediction (optional, for cold-start this may be None)
        hotel_id: Hotel identifier (None for cold-start)
        target_date: Target date for recommendation
    
    Returns:
        TriangulatedRecommendation with all details
    """
    current_price = revpar_comparison.hotel_price
    current_occupancy = revpar_comparison.hotel_occupancy
    current_revpar = revpar_comparison.hotel_revpar
    
    # Check occupancy safety valve
    # If occupancy is critically low relative to peers, force HOLD
    occ_gap = revpar_comparison.occupancy_gap
    occupancy_safety_triggered = (
        occ_gap <= OCC_GAP_QUALITY_ISSUE and  # <= for safety (at threshold = triggered)
        revpar_comparison.opportunity == PriceOpportunity.RAISE_PRICE
    )
    
    # Classify scenario
    scenario, reasoning = classify_scenario(revpar_comparison, fair_value)
    
    # Override if safety valve triggered
    if occupancy_safety_triggered and scenario in [RecommendationScenario.A, RecommendationScenario.B]:
        scenario = RecommendationScenario.D
        reasoning = (
            f"Occupancy {occ_gap*100:.1f}pp below peers triggers safety valve. "
            "Fix occupancy before raising prices. HOLD."
        )
    
    # Calculate recommended price
    recommended_price, change_pct = calculate_triangulated_price(
        current_price, scenario, revpar_comparison, fair_value
    )
    
    # Determine confidence
    confidence = determine_confidence(revpar_comparison, fair_value)
    
    return TriangulatedRecommendation(
        hotel_id=hotel_id,
        target_date=target_date,
        current_price=current_price,
        current_occupancy=current_occupancy,
        current_revpar=current_revpar,
        recommended_price=recommended_price,
        change_pct=change_pct,
        scenario=scenario,
        confidence=confidence,
        revpar_comparison=revpar_comparison,
        fair_value=fair_value,
        occupancy_safety_triggered=occupancy_safety_triggered,
        reasoning=reasoning
    )


class FairValueModel:
    """
    XGBoost-based fair value predictor.
    
    Predicts what a hotel "should" charge based on its features.
    Trained on historical data from feature_importance_validation.py.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize fair value model.
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.model = None
        self.preprocessor = None
        self.feature_names: List[str] = []
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def load(self, model_path: Path) -> None:
        """Load trained model from disk."""
        saved = joblib.load(model_path)
        self.model = saved.get('model')
        self.preprocessor = saved.get('preprocessor')
        self.feature_names = saved.get('feature_names', [])
    
    def save(self, model_path: Path) -> None:
        """Save trained model to disk."""
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }, model_path)
    
    def predict(
        self,
        hotel_features: Dict[str, Any],
        actual_price: float
    ) -> FairValueResult:
        """
        Predict fair value for a hotel.
        
        Args:
            hotel_features: Dict with feature values
            actual_price: Current price for residual calculation
        
        Returns:
            FairValueResult with prediction and residual
        """
        if self.model is None:
            # No model loaded - return neutral result
            return FairValueResult(
                predicted_price=actual_price,
                actual_price=actual_price,
                value_residual=0.0
            )
        
        # Prepare features
        X = pd.DataFrame([hotel_features])
        
        # Transform if preprocessor available
        if self.preprocessor:
            X_transformed = self.preprocessor.transform(X)
        else:
            X_transformed = X.values
        
        # Predict (model predicts log(ADR))
        log_predicted = self.model.predict(X_transformed)[0]
        predicted_price = np.exp(log_predicted)
        
        # Calculate residual
        value_residual = predicted_price - actual_price
        
        return FairValueResult(
            predicted_price=predicted_price,
            actual_price=actual_price,
            value_residual=value_residual
        )
    
    def predict_for_profile(
        self,
        profile: HotelProfile,
        actual_price: float
    ) -> FairValueResult:
        """
        Predict fair value for a hotel profile (cold-start).
        
        Args:
            profile: HotelProfile with features
            actual_price: Current/proposed price
        
        Returns:
            FairValueResult
        """
        # Convert profile to feature dict
        features = {
            'log_room_size': np.log1p(profile.room_size),
            'room_capacity_pax': profile.max_occupancy,
            'total_capacity_log': np.log1p(profile.num_rooms),
            'amenities_score': len(profile.amenities),
            'room_type': profile.room_type,
            'dist_center_km': 0.0,  # Default - would need city centroid
            'dist_coast_log': np.log1p(100),  # Default 100km from coast
            'view_quality_ordinal': 0,  # Default no view
            'weekend_ratio': 0.3,  # Default weekday bias
            'occupancy_rate': 0.5,  # Default 50%
            'is_coastal': 0,
            'is_july_august': 0,
            'children_allowed': 1 if 'children_allowed' in profile.amenities else 0,
        }
        
        return self.predict(features, actual_price)


def get_scenario_distribution(recommendations: List[TriangulatedRecommendation]) -> Dict[str, int]:
    """
    Calculate distribution of scenarios across recommendations.
    
    Args:
        recommendations: List of recommendations
    
    Returns:
        Dict with scenario counts
    """
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for rec in recommendations:
        counts[rec.scenario.value] += 1
    return counts


# =============================================================================
# REVPAR OPTIMIZATION using Occupancy Model
# =============================================================================

def find_revpar_optimal_price(
    current_price: float,
    base_occupancy: float,
    elasticity: float = MARKET_ELASTICITY,
    min_price_pct: float = 0.70,
    max_price_pct: float = 1.50,
    n_points: int = 50
) -> Tuple[float, float, float]:
    """
    Find the price that maximizes RevPAR using elasticity-adjusted occupancy.
    
    RevPAR = Price × Occupancy
    Occupancy = base_occupancy × (price_ratio ^ elasticity)
    
    The optimal price depends on elasticity:
    - With ε = -0.39 (inelastic), higher prices generally win
    - But there's a ceiling where occupancy drops too much
    
    Args:
        current_price: Current hotel price
        base_occupancy: Occupancy at current price
        elasticity: Price elasticity of demand (-0.39 validated)
        min_price_pct: Minimum price as % of current (0.70 = 30% cut)
        max_price_pct: Maximum price as % of current (1.50 = 50% increase)
        n_points: Number of price points to evaluate
    
    Returns:
        Tuple of (optimal_price, optimal_revpar, optimal_occupancy)
    """
    # Generate price grid
    price_ratios = np.linspace(min_price_pct, max_price_pct, n_points)
    prices = current_price * price_ratios
    
    # Calculate occupancy at each price point
    # Occupancy = base × (price_ratio ^ elasticity)
    occupancies = base_occupancy * np.power(price_ratios, elasticity)
    occupancies = np.clip(occupancies, 0.01, 0.99)  # Bound to valid range
    
    # Calculate RevPAR at each point
    revpars = prices * occupancies
    
    # Find optimal
    optimal_idx = np.argmax(revpars)
    
    return (
        prices[optimal_idx],
        revpars[optimal_idx],
        occupancies[optimal_idx]
    )


def calculate_revpar_curve(
    current_price: float,
    base_occupancy: float,
    elasticity: float = MARKET_ELASTICITY,
    min_price_pct: float = 0.70,
    max_price_pct: float = 1.50,
    n_points: int = 50
) -> pd.DataFrame:
    """
    Calculate full RevPAR curve across price range.
    
    Useful for visualization and understanding the tradeoff.
    
    Args:
        current_price: Current hotel price
        base_occupancy: Occupancy at current price
        elasticity: Price elasticity of demand
        min_price_pct: Minimum price as % of current
        max_price_pct: Maximum price as % of current
        n_points: Number of price points
    
    Returns:
        DataFrame with columns: price, price_pct, occupancy, revpar
    """
    price_ratios = np.linspace(min_price_pct, max_price_pct, n_points)
    prices = current_price * price_ratios
    occupancies = base_occupancy * np.power(price_ratios, elasticity)
    occupancies = np.clip(occupancies, 0.01, 0.99)
    revpars = prices * occupancies
    
    return pd.DataFrame({
        'price': prices,
        'price_pct': (price_ratios - 1) * 100,  # % change from current
        'occupancy': occupancies,
        'revpar': revpars
    })


@dataclass
class RevPAROptimizedRecommendation:
    """
    Recommendation that explicitly optimizes RevPAR.
    
    Combines peer context with occupancy model to find the price
    that maximizes expected revenue per available room.
    """
    hotel_id: Optional[int]
    target_date: Optional[date]
    
    # Current state
    current_price: float
    current_occupancy: float
    current_revpar: float
    
    # Peer context
    peer_revpar: float
    peer_price: float
    peer_occupancy: float
    revpar_gap: float
    
    # Optimized recommendation
    optimal_price: float
    optimal_occupancy: float
    optimal_revpar: float
    change_pct: float
    
    # Expected improvement
    revpar_lift: float  # (optimal - current) / current
    revpar_vs_peer: float  # (optimal - peer) / peer
    
    # Confidence and reasoning
    confidence: Confidence
    reasoning: str
    
    # Additional context
    elasticity_used: float
    n_peers: int
    peer_source: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'hotel_id': self.hotel_id,
            'target_date': self.target_date.isoformat() if self.target_date else None,
            'current_price': round(self.current_price, 2),
            'current_occupancy': round(self.current_occupancy, 3),
            'current_revpar': round(self.current_revpar, 2),
            'peer_revpar': round(self.peer_revpar, 2),
            'peer_price': round(self.peer_price, 2),
            'peer_occupancy': round(self.peer_occupancy, 3),
            'revpar_gap_pct': round(self.revpar_gap * 100, 1),
            'optimal_price': round(self.optimal_price, 2),
            'optimal_occupancy': round(self.optimal_occupancy, 3),
            'optimal_revpar': round(self.optimal_revpar, 2),
            'change_pct': round(self.change_pct, 1),
            'revpar_lift_pct': round(self.revpar_lift * 100, 1),
            'revpar_vs_peer_pct': round(self.revpar_vs_peer * 100, 1),
            'confidence': self.confidence.value,
            'reasoning': self.reasoning,
            'elasticity_used': self.elasticity_used,
            'n_peers': self.n_peers,
            'peer_source': self.peer_source,
        }


def get_revpar_optimized_recommendation(
    revpar_comparison: RevPARComparison,
    occupancy_model: Optional[OccupancyModel] = None,
    hotel_features: Optional[pd.DataFrame] = None,
    hotel_id: Optional[int] = None,
    target_date: Optional[date] = None,
    elasticity: float = MARKET_ELASTICITY,
    max_increase_pct: float = 0.30,
    max_decrease_pct: float = 0.20
) -> RevPAROptimizedRecommendation:
    """
    Generate RevPAR-optimized pricing recommendation.
    
    Uses PEER-CONSTRAINED optimization:
    - Can't just maximize price because inelastic demand always favors higher prices
    - Must consider peer context: if peers charge €100, recommending €150 is risky
    - Occupancy gap matters: if already below peers, raising prices is dangerous
    
    The recommendation targets:
    1. If underperforming: Move TOWARD peer price (not above it)
    2. If overperforming: Modest increase, capped at peer price + 10%
    3. If optimal: Hold current pricing
    
    Args:
        revpar_comparison: Peer comparison result
        occupancy_model: Fitted occupancy model (optional)
        hotel_features: Hotel feature DataFrame for model prediction
        hotel_id: Hotel identifier
        target_date: Target date
        elasticity: Price elasticity to use
        max_increase_pct: Maximum price increase (0.30 = 30%)
        max_decrease_pct: Maximum price decrease (0.20 = 20%)
    
    Returns:
        RevPAROptimizedRecommendation with optimal price
    """
    current_price = revpar_comparison.hotel_price
    current_occupancy = revpar_comparison.hotel_occupancy
    current_revpar = revpar_comparison.hotel_revpar
    
    peer_price = revpar_comparison.peer_price
    peer_occupancy = revpar_comparison.peer_occupancy
    peer_revpar = revpar_comparison.peer_revpar
    
    price_gap = revpar_comparison.price_gap
    occ_gap = revpar_comparison.occupancy_gap
    revpar_gap = revpar_comparison.revpar_gap
    signal = revpar_comparison.signal
    opportunity = revpar_comparison.opportunity
    
    # Use occupancy model if available, otherwise use current occupancy as base
    if occupancy_model is not None and hotel_features is not None:
        base_occupancy = occupancy_model.predict(hotel_features)[0]
    else:
        base_occupancy = current_occupancy
    
    # =========================================================================
    # PEER-CONSTRAINED OPTIMIZATION
    # The key insight: pure elasticity optimization always recommends max price
    # because ε=-0.39 is inelastic. We must use PEER CONTEXT to constrain.
    # =========================================================================
    
    if signal == PerformanceSignal.UNDERPERFORMING:
        if opportunity == PriceOpportunity.RAISE_PRICE:
            # Peers prove higher prices work - move TOWARD peer price
            # But don't go ABOVE peer price (they've validated that price point)
            if price_gap < 0:  # Currently below peer price
                # Target: Close 50-70% of the gap to peer price
                gap_to_close = 0.6 if occ_gap > -0.05 else 0.4  # Less aggressive if low occ
                target_price = current_price + gap_to_close * (peer_price - current_price)
                target_price = min(target_price, peer_price * 1.05)  # Max 5% above peers
            else:
                # Already above peers but underperforming - don't raise further
                target_price = current_price
        
        elif opportunity == PriceOpportunity.LOWER_PRICE:
            # Overpriced relative to peers - reduce toward peer level
            target_price = current_price * 0.9  # 10% reduction
            target_price = max(target_price, peer_price * 0.85)  # Don't go too low
        
        else:  # HOLD
            target_price = current_price
    
    elif signal == PerformanceSignal.OVERPERFORMING:
        if opportunity == PriceOpportunity.RAISE_PRICE:
            # Strong demand - can push above peers, but modestly
            if price_gap < 0:  # Cheaper than peers with higher occupancy
                # Can raise toward peer level + premium
                target_price = peer_price * 1.10  # Target 10% above peers
            else:
                # Already premium, small increase
                target_price = current_price * 1.08
        else:
            # Already optimized
            target_price = current_price
    
    else:  # OPTIMAL
        # Well-calibrated - hold or tiny adjustment
        target_price = current_price
    
    # Apply caps
    max_price = current_price * (1 + max_increase_pct)
    min_price = current_price * (1 - max_decrease_pct)
    optimal_price = np.clip(target_price, min_price, max_price)
    
    # =========================================================================
    # EXPECTED OCCUPANCY: Different logic for over vs underperforming hotels
    # 
    # UNDERPERFORMING (hotel_occ < peer_occ): 
    #   Moving toward peer price → expect occupancy to improve toward peer level
    #   (The market has validated that price point works)
    #
    # OVERPERFORMING (hotel_occ >= peer_occ, especially 100% sold out):
    #   Hotel has STRONGER demand than peers
    #   Raising prices → use ELASTICITY to predict modest occupancy drop
    #   RevPAR should INCREASE (inelastic demand)
    # =========================================================================
    
    change_pct = (optimal_price - current_price) / current_price if current_price > 0 else 0
    
    if abs(change_pct) < 0.02:
        # No change - keep current
        optimal_occupancy = current_occupancy
        optimal_revpar = current_revpar
    else:
        is_overperforming = current_occupancy >= peer_occupancy
        is_high_occupancy = current_occupancy >= 0.85  # >85% = strong demand
        
        if is_overperforming or is_high_occupancy:
            # =========================================================
            # OVERPERFORMING: Use elasticity - demand is proven strong
            # A 100% hotel raising prices 10% should drop to ~96% (not 80%)
            # =========================================================
            price_ratio = optimal_price / current_price
            # Apply elasticity: new_occ = current_occ * (price_ratio ^ elasticity)
            optimal_occupancy = current_occupancy * np.power(price_ratio, elasticity)
            # But don't drop below peer occupancy - we're still better than them
            optimal_occupancy = max(optimal_occupancy, peer_occupancy * 0.95)
        else:
            # =========================================================
            # UNDERPERFORMING: Interpolate toward peer performance
            # If we price like peers, expect occupancy closer to peers
            # =========================================================
            if peer_price != current_price:
                price_position = (optimal_price - current_price) / (peer_price - current_price)
                price_position = np.clip(price_position, 0, 1.2)
            else:
                price_position = 0
            
            # Conservative: assume we achieve 80% of the improvement
            occ_delta = peer_occupancy - current_occupancy
            optimal_occupancy = current_occupancy + (price_position * occ_delta * 0.8)
        
        optimal_occupancy = np.clip(optimal_occupancy, 0.05, 0.99)
        optimal_revpar = optimal_price * optimal_occupancy
    
    # Calculate improvements
    revpar_lift = (optimal_revpar - current_revpar) / current_revpar if current_revpar > 0 else 0
    revpar_vs_peer = (optimal_revpar - peer_revpar) / peer_revpar if peer_revpar > 0 else 0
    
    # Generate reasoning
    if change_pct > 0.02:
        action = f"RAISE +{change_pct*100:.0f}%"
        if price_gap < 0:
            reason = f"Currently {abs(price_gap)*100:.0f}% below peers. Moving toward peer level."
        else:
            reason = f"Strong demand supports {change_pct*100:.0f}% increase."
    elif change_pct < -0.02:
        action = f"LOWER {change_pct*100:.0f}%"
        reason = f"Currently {price_gap*100:.0f}% above peers with lower occupancy. Reducing to improve volume."
    else:
        action = "HOLD"
        if abs(revpar_gap) < 0.10:
            reason = "RevPAR within 10% of peers. Pricing is well-calibrated."
        else:
            reason = "Mixed signals. Holding for more data."
    
    # Determine confidence
    confidence = determine_confidence(revpar_comparison)
    
    return RevPAROptimizedRecommendation(
        hotel_id=hotel_id,
        target_date=target_date,
        current_price=current_price,
        current_occupancy=current_occupancy,
        current_revpar=current_revpar,
        peer_revpar=peer_revpar,
        peer_price=peer_price,
        peer_occupancy=peer_occupancy,
        revpar_gap=revpar_comparison.revpar_gap,
        optimal_price=optimal_price,
        optimal_occupancy=optimal_occupancy,
        optimal_revpar=optimal_revpar,
        change_pct=change_pct * 100,
        revpar_lift=revpar_lift,
        revpar_vs_peer=revpar_vs_peer,
        confidence=confidence,
        reasoning=f"{action}: {reason}",
        elasticity_used=elasticity,
        n_peers=revpar_comparison.n_peers,
        peer_source=revpar_comparison.peer_source
    )

