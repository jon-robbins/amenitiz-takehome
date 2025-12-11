"""
Pricing diagnosis logic.

Determines if a hotel is overpriced, underpriced, or fairly priced
based on price position relative to peers and occupancy performance.
"""

from dataclasses import dataclass
from typing import Tuple


# Thresholds for diagnosis
PREMIUM_THRESHOLD = 0.10      # 10% above/below peers
RESIDUAL_THRESHOLD = 0.05    # 5 percentage points occupancy difference

# Price adjustment limits
MAX_INCREASE_PCT = 30.0
MAX_DECREASE_PCT = 30.0


@dataclass
class PriceDiagnosis:
    """Result of pricing diagnosis."""
    direction: str      # "increase", "decrease", "maintain"
    reasoning: str      # Human-readable explanation
    is_overpriced: bool
    is_underpriced: bool


def diagnose_pricing(
    price_premium: float,
    occ_residual: float,
    premium_threshold: float = PREMIUM_THRESHOLD,
    residual_threshold: float = RESIDUAL_THRESHOLD
) -> PriceDiagnosis:
    """
    Diagnose pricing based on price position and occupancy performance.
    
    Core Logic:
    - OVERPRICED: High price + Low occupancy → Decrease
    - UNDERPRICED: Low price + High occupancy → Increase
    - FAIR: Aligned with market → Maintain
    
    Args:
        price_premium: (hotel_price - peer_price) / peer_price
        occ_residual: actual_occupancy - expected_occupancy
        premium_threshold: Threshold for "expensive" vs "cheap"
        residual_threshold: Threshold for "underperforming" vs "overperforming"
    
    Returns:
        PriceDiagnosis with direction and reasoning
    """
    is_expensive = price_premium > premium_threshold
    is_cheap = price_premium < -premium_threshold
    underperforming = occ_residual < -residual_threshold
    overperforming = occ_residual > residual_threshold
    
    # OVERPRICED: High price AND low occupancy
    if is_expensive and underperforming:
        return PriceDiagnosis(
            direction="decrease",
            reasoning="Priced above peers but occupancy below expected - reduce price to capture demand",
            is_overpriced=True,
            is_underpriced=False
        )
    
    # UNDERPRICED: Low price AND high occupancy
    if is_cheap and overperforming:
        return PriceDiagnosis(
            direction="increase",
            reasoning="Priced below peers with strong demand - increase price to capture value",
            is_overpriced=False,
            is_underpriced=True
        )
    
    # Premium pricing justified by demand
    if is_expensive and overperforming:
        return PriceDiagnosis(
            direction="maintain",
            reasoning="Premium pricing justified by strong demand",
            is_overpriced=False,
            is_underpriced=False
        )
    
    # Low price but still underperforming - other issues
    if is_cheap and underperforming:
        return PriceDiagnosis(
            direction="maintain",
            reasoning="Low price not driving demand - other factors at play (quality, location)",
            is_overpriced=False,
            is_underpriced=False
        )
    
    # Expensive but acceptable occupancy
    if is_expensive and not underperforming:
        return PriceDiagnosis(
            direction="maintain",
            reasoning="Premium pricing with acceptable occupancy",
            is_overpriced=False,
            is_underpriced=False
        )
    
    # Cheap with room to grow
    if is_cheap and not overperforming:
        return PriceDiagnosis(
            direction="increase",
            reasoning="Room to increase price toward peer level",
            is_overpriced=False,
            is_underpriced=True
        )
    
    # Default: aligned with market
    return PriceDiagnosis(
        direction="maintain",
        reasoning="Pricing aligned with market",
        is_overpriced=False,
        is_underpriced=False
    )


def calculate_recommended_price(
    direction: str,
    current_price: float,
    peer_price: float,
    elasticity: float = -0.39
) -> float:
    """
    Calculate recommended price based on diagnosis.
    
    Args:
        direction: "increase", "decrease", or "maintain"
        current_price: Hotel's current price
        peer_price: Average peer price
        elasticity: Price elasticity of demand (default -0.39)
    
    Returns:
        Recommended price in euros
    """
    if direction == "decrease":
        # Move halfway toward peer price (conservative)
        target = peer_price
        new_price = current_price - 0.5 * (current_price - target)
        # Cap at max decrease
        min_price = current_price * (1 - MAX_DECREASE_PCT / 100)
        return max(new_price, min_price)
    
    elif direction == "increase":
        # Calculate optimal markup based on elasticity
        # At |ε| = 0.39, optimal is high but we cap conservatively
        optimal_markup = min(MAX_INCREASE_PCT / 100, 0.30)
        target = peer_price * (1 + optimal_markup)
        # Move halfway toward target
        new_price = current_price + 0.5 * (target - current_price)
        # Cap at max increase
        max_price = current_price * (1 + MAX_INCREASE_PCT / 100)
        return min(new_price, max_price)
    
    else:  # maintain
        return current_price

