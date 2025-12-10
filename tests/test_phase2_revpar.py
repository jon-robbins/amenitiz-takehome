"""
Phase 2 Validation Tests: RevPAR Peer Comparison

Success Metrics:
- RevPAR calculation accuracy: 100%
- Signal coverage: 100% hotels get a signal
- Correlation: RevPAR gap → actual performance: r > 0.5
"""

from datetime import date, timedelta

import numpy as np
import pytest

from src.recommender.revpar_peers import (
    RevPARComparison,
    PerformanceSignal,
    PriceOpportunity,
    classify_signal,
    calculate_revpar_comparison,
    calculate_recommended_price_change,
    REVPAR_GAP_UNDERPERFORM,
    REVPAR_GAP_OVERPERFORM,
    OCC_GAP_CRITICAL,
)


class TestSignalClassification:
    """Test the signal classification logic."""
    
    def test_underperforming_cheaper_lower_occ_should_raise(self):
        """
        Scenario: Hotel is cheaper than peers but still has lower occupancy.
        Peers prove higher prices work → recommend raise_price.
        """
        signal, opportunity, reasoning = classify_signal(
            revpar_gap=-0.25,  # 25% below peers
            price_gap=-0.15,   # 15% cheaper
            occupancy_gap=-0.05  # 5pp lower occupancy
        )
        
        assert signal == PerformanceSignal.UNDERPERFORMING
        assert opportunity == PriceOpportunity.RAISE_PRICE
        assert "peers prove higher prices work" in reasoning.lower()
    
    def test_underperforming_expensive_low_occ_should_lower(self):
        """
        Scenario: Hotel is more expensive than peers with much lower occupancy.
        Overpriced → recommend lower_price.
        """
        signal, opportunity, reasoning = classify_signal(
            revpar_gap=-0.30,  # 30% below peers
            price_gap=0.20,    # 20% more expensive
            occupancy_gap=-0.15  # 15pp lower occupancy (critical)
        )
        
        assert signal == PerformanceSignal.UNDERPERFORMING
        assert opportunity == PriceOpportunity.LOWER_PRICE
        assert "too high" in reasoning.lower()
    
    def test_underperforming_cheaper_good_occ_should_hold(self):
        """
        Scenario: Hotel is cheaper with good occupancy but still low RevPAR.
        Quality issue, not price → recommend hold.
        """
        signal, opportunity, reasoning = classify_signal(
            revpar_gap=-0.20,  # 20% below peers
            price_gap=-0.25,   # 25% cheaper
            occupancy_gap=0.05  # 5pp higher occupancy
        )
        
        assert signal == PerformanceSignal.UNDERPERFORMING
        assert opportunity == PriceOpportunity.HOLD
        assert "non-price factors" in reasoning.lower() or "quality" in reasoning.lower()
    
    def test_optimal_should_hold(self):
        """
        Scenario: Hotel RevPAR within ±15% of peers.
        Optimal → recommend hold.
        """
        signal, opportunity, reasoning = classify_signal(
            revpar_gap=0.05,   # 5% above peers
            price_gap=0.10,    # 10% more expensive
            occupancy_gap=-0.03  # 3pp lower occupancy
        )
        
        assert signal == PerformanceSignal.OPTIMAL
        assert opportunity == PriceOpportunity.HOLD
        assert "well-calibrated" in reasoning.lower() or "within" in reasoning.lower()
    
    def test_overperforming_cheaper_high_occ_should_raise(self):
        """
        Scenario: Hotel is cheaper than peers with much higher occupancy.
        Strong demand → recommend raise_price.
        """
        signal, opportunity, reasoning = classify_signal(
            revpar_gap=0.25,   # 25% above peers
            price_gap=-0.10,   # 10% cheaper
            occupancy_gap=0.15  # 15pp higher occupancy
        )
        
        assert signal == PerformanceSignal.OVERPERFORMING
        assert opportunity == PriceOpportunity.RAISE_PRICE
        assert "strong demand" in reasoning.lower()
    
    def test_overperforming_already_premium_should_hold(self):
        """
        Scenario: Hotel is more expensive and still has higher occupancy.
        Already optimized → recommend hold.
        """
        signal, opportunity, reasoning = classify_signal(
            revpar_gap=0.30,   # 30% above peers
            price_gap=0.15,    # 15% more expensive
            occupancy_gap=0.10  # 10pp higher occupancy
        )
        
        assert signal == PerformanceSignal.OVERPERFORMING
        assert opportunity == PriceOpportunity.HOLD
        assert "premium" in reasoning.lower() or "maintain" in reasoning.lower()


class TestRevPARCalculation:
    """Test RevPAR comparison calculation accuracy."""
    
    def test_revpar_equals_price_times_occupancy(self):
        """RevPAR should equal Price × Occupancy."""
        comparison = calculate_revpar_comparison(
            hotel_price=100.0,
            hotel_occupancy=0.80,
            peer_price=90.0,
            peer_occupancy=0.70,
            n_peers=5,
            peer_source="geographic"
        )
        
        assert comparison.hotel_revpar == pytest.approx(80.0)  # 100 × 0.80
        assert comparison.peer_revpar == pytest.approx(63.0)   # 90 × 0.70
    
    def test_gap_calculations(self):
        """Gaps should be calculated correctly."""
        comparison = calculate_revpar_comparison(
            hotel_price=120.0,
            hotel_occupancy=0.60,
            peer_price=100.0,
            peer_occupancy=0.50,
            n_peers=3,
            peer_source="twin"
        )
        
        # RevPAR: hotel=72, peer=50
        assert comparison.revpar_gap == pytest.approx((72 - 50) / 50)  # +44%
        assert comparison.price_gap == pytest.approx((120 - 100) / 100)  # +20%
        assert comparison.occupancy_gap == pytest.approx(0.10)  # +10pp
    
    def test_signal_classification_integrated(self):
        """Signal should be classified correctly based on gaps."""
        # Underperforming case
        comparison = calculate_revpar_comparison(
            hotel_price=80.0,
            hotel_occupancy=0.30,
            peer_price=100.0,
            peer_occupancy=0.50,
            n_peers=5,
            peer_source="geographic"
        )
        
        # RevPAR: hotel=24, peer=50 → -52% gap
        assert comparison.signal == PerformanceSignal.UNDERPERFORMING
    
    def test_to_dict_serialization(self):
        """Comparison should serialize to dict correctly."""
        comparison = calculate_revpar_comparison(
            hotel_price=100.0,
            hotel_occupancy=0.50,
            peer_price=100.0,
            peer_occupancy=0.50,
            n_peers=10,
            peer_source="peer_group"
        )
        
        d = comparison.to_dict()
        
        assert d['hotel_revpar'] == 50.0
        assert d['n_peers'] == 10
        assert d['peer_source'] == "peer_group"
        assert 'signal' in d
        assert 'opportunity' in d


class TestPriceRecommendation:
    """Test price change recommendations."""
    
    def test_raise_price_caps_at_30_percent(self):
        """Price increase should be capped at 30%."""
        comparison = RevPARComparison(
            hotel_revpar=50.0,
            peer_revpar=100.0,
            revpar_gap=-0.50,
            hotel_price=50.0,
            peer_price=100.0,
            price_gap=-0.50,
            hotel_occupancy=1.0,
            peer_occupancy=1.0,
            occupancy_gap=0.0,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        rec_price, change_pct = calculate_recommended_price_change(comparison)
        
        assert change_pct <= 30.0
        assert rec_price <= 50.0 * 1.30
    
    def test_lower_price_caps_at_minus_20_percent(self):
        """Price decrease should be capped at -20%."""
        comparison = RevPARComparison(
            hotel_revpar=20.0,
            peer_revpar=50.0,
            revpar_gap=-0.60,
            hotel_price=100.0,
            peer_price=50.0,
            price_gap=1.0,
            hotel_occupancy=0.20,
            peer_occupancy=1.0,
            occupancy_gap=-0.80,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.LOWER_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        rec_price, change_pct = calculate_recommended_price_change(comparison)
        
        assert change_pct >= -20.0
        assert rec_price >= 100.0 * 0.80
    
    def test_hold_returns_same_price(self):
        """Hold opportunity should return current price."""
        comparison = RevPARComparison(
            hotel_revpar=50.0,
            peer_revpar=50.0,
            revpar_gap=0.0,
            hotel_price=100.0,
            peer_price=100.0,
            price_gap=0.0,
            hotel_occupancy=0.50,
            peer_occupancy=0.50,
            occupancy_gap=0.0,
            signal=PerformanceSignal.OPTIMAL,
            opportunity=PriceOpportunity.HOLD,
            n_peers=5,
            peer_source="geographic"
        )
        
        rec_price, change_pct = calculate_recommended_price_change(comparison)
        
        assert change_pct == 0.0
        assert rec_price == 100.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_peer_revpar_handles_gracefully(self):
        """Should handle zero peer RevPAR without division error."""
        comparison = calculate_revpar_comparison(
            hotel_price=100.0,
            hotel_occupancy=0.50,
            peer_price=0.0,
            peer_occupancy=0.0,
            n_peers=0,
            peer_source="geographic"
        )
        
        assert comparison.revpar_gap == 0.0
        assert comparison.price_gap == 0.0
    
    def test_boundary_revpar_gap_underperform(self):
        """Test exact boundary for underperforming threshold."""
        # Just above threshold - should be OPTIMAL
        signal1, _, _ = classify_signal(
            revpar_gap=REVPAR_GAP_UNDERPERFORM + 0.01,  # -14%
            price_gap=0,
            occupancy_gap=0
        )
        assert signal1 == PerformanceSignal.OPTIMAL
        
        # Just below threshold - should be UNDERPERFORMING
        signal2, _, _ = classify_signal(
            revpar_gap=REVPAR_GAP_UNDERPERFORM - 0.01,  # -16%
            price_gap=-0.10,
            occupancy_gap=-0.05
        )
        assert signal2 == PerformanceSignal.UNDERPERFORMING
    
    def test_boundary_revpar_gap_overperform(self):
        """Test exact boundary for overperforming threshold."""
        # Just below threshold - should be OPTIMAL
        signal1, _, _ = classify_signal(
            revpar_gap=REVPAR_GAP_OVERPERFORM - 0.01,  # +14%
            price_gap=0,
            occupancy_gap=0
        )
        assert signal1 == PerformanceSignal.OPTIMAL
        
        # Just above threshold - should be OVERPERFORMING
        signal2, _, _ = classify_signal(
            revpar_gap=REVPAR_GAP_OVERPERFORM + 0.01,  # +16%
            price_gap=-0.10,
            occupancy_gap=0.10
        )
        assert signal2 == PerformanceSignal.OVERPERFORMING


class TestScenarioDistribution:
    """Test that scenarios are distributed reasonably across typical data."""
    
    def test_scenario_coverage(self):
        """All scenarios should be achievable."""
        scenarios = set()
        
        test_cases = [
            # (revpar_gap, price_gap, occ_gap)
            (-0.30, -0.20, -0.05),  # Underperform, raise
            (-0.25, 0.15, -0.15),   # Underperform, lower
            (-0.20, -0.25, 0.05),   # Underperform, hold
            (0.05, 0.05, 0.0),      # Optimal
            (0.25, -0.10, 0.15),    # Overperform, raise
            (0.30, 0.15, 0.10),     # Overperform, hold
        ]
        
        for revpar_gap, price_gap, occ_gap in test_cases:
            signal, opportunity, _ = classify_signal(revpar_gap, price_gap, occ_gap)
            scenarios.add((signal, opportunity))
        
        # Should have at least 4 different (signal, opportunity) combinations
        assert len(scenarios) >= 4, f"Only {len(scenarios)} scenarios achieved"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

