"""Tests for pricing diagnosis logic."""

import pytest
from src.recommender.diagnosis import diagnose_pricing, calculate_recommended_price


class TestDiagnosePricing:
    """Test cases for diagnose_pricing function."""
    
    def test_overpriced_hotel(self):
        """High price + low occupancy → decrease."""
        result = diagnose_pricing(
            price_premium=0.25,     # 25% above peers
            occ_residual=-0.10     # 10pp below expected
        )
        assert result.direction == "decrease"
        assert result.is_overpriced
        assert not result.is_underpriced
    
    def test_underpriced_hotel(self):
        """Low price + high occupancy → increase."""
        result = diagnose_pricing(
            price_premium=-0.20,    # 20% below peers
            occ_residual=0.15       # 15pp above expected
        )
        assert result.direction == "increase"
        assert result.is_underpriced
        assert not result.is_overpriced
    
    def test_fair_pricing(self):
        """Aligned with market → maintain."""
        result = diagnose_pricing(
            price_premium=0.05,     # Only 5% above (within threshold)
            occ_residual=0.02       # Only 2pp difference (within threshold)
        )
        assert result.direction == "maintain"
        assert not result.is_overpriced
        assert not result.is_underpriced
    
    def test_premium_justified(self):
        """High price but high demand → maintain."""
        result = diagnose_pricing(
            price_premium=0.30,     # 30% above peers
            occ_residual=0.10       # But 10pp above expected
        )
        assert result.direction == "maintain"
        assert not result.is_overpriced
    
    def test_cheap_but_weak_demand(self):
        """Low price but low demand → maintain (other issues)."""
        result = diagnose_pricing(
            price_premium=-0.25,    # 25% below peers
            occ_residual=-0.15      # But still 15pp below expected
        )
        assert result.direction == "maintain"
        assert not result.is_underpriced


class TestCalculateRecommendedPrice:
    """Test cases for calculate_recommended_price function."""
    
    def test_decrease_moves_toward_peer(self):
        """Decrease should move halfway toward peer price."""
        result = calculate_recommended_price(
            direction="decrease",
            current_price=150.0,
            peer_price=100.0
        )
        # Should be 150 - 0.5 * (150 - 100) = 125
        assert result == 125.0
    
    def test_increase_caps_at_max(self):
        """Increase should not exceed max increase."""
        result = calculate_recommended_price(
            direction="increase",
            current_price=100.0,
            peer_price=100.0
        )
        # Should be capped at 130 (30% max increase)
        assert result <= 130.0
    
    def test_maintain_keeps_current(self):
        """Maintain should return current price."""
        result = calculate_recommended_price(
            direction="maintain",
            current_price=100.0,
            peer_price=120.0
        )
        assert result == 100.0
    
    def test_decrease_respects_min(self):
        """Decrease should not go below max decrease limit."""
        result = calculate_recommended_price(
            direction="decrease",
            current_price=100.0,
            peer_price=50.0  # Way below current
        )
        # Should not go below 70 (30% max decrease)
        assert result >= 70.0

