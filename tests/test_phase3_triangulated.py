"""
Phase 3 Validation Tests: Triangulated Scorer with RevPAR Optimization

Success Metrics:
- XGBoost R² ≥ 0.70 (fair value model)
- Scenario distribution: A:30-40%, B:25-35%, C:15-25%, D:10-20%
- No "suicide" recs: 0 cases of +25% for occ_gap < -10pp
- RevPAR optimization finds correct maximum
"""

import numpy as np
import pandas as pd
import pytest

from src.recommender.triangulated_scorer import (
    RecommendationScenario,
    Confidence,
    FairValueResult,
    TriangulatedRecommendation,
    classify_scenario,
    calculate_triangulated_price,
    determine_confidence,
    get_triangulated_recommendation,
    find_revpar_optimal_price,
    calculate_revpar_curve,
    get_revpar_optimized_recommendation,
    RevPAROptimizedRecommendation,
    VALUE_RESIDUAL_UNDERPRICED,
    VALUE_RESIDUAL_OVERPRICED,
)
from src.recommender.revpar_peers import (
    RevPARComparison,
    PerformanceSignal,
    PriceOpportunity,
)
from src.features.engineering import MARKET_ELASTICITY


class TestRevPAROptimization:
    """Test the RevPAR optimization using elasticity."""
    
    def test_optimal_price_higher_when_inelastic(self):
        """
        With inelastic demand (|ε| < 1), optimal price should be higher.
        Elasticity -0.39 means 10% price increase → 3.9% occupancy decrease.
        """
        current_price = 100.0
        base_occupancy = 0.60
        
        optimal_price, optimal_revpar, optimal_occ = find_revpar_optimal_price(
            current_price=current_price,
            base_occupancy=base_occupancy,
            elasticity=MARKET_ELASTICITY  # -0.39
        )
        
        # With inelastic demand, optimal price should be at the cap
        assert optimal_price > current_price
        # RevPAR should improve
        current_revpar = current_price * base_occupancy
        assert optimal_revpar > current_revpar
    
    def test_revpar_curve_shape(self):
        """RevPAR curve should have a clear maximum."""
        curve = calculate_revpar_curve(
            current_price=100.0,
            base_occupancy=0.50,
            elasticity=MARKET_ELASTICITY
        )
        
        assert len(curve) > 0
        assert 'price' in curve.columns
        assert 'revpar' in curve.columns
        
        # Should have variation in RevPAR
        assert curve['revpar'].std() > 0
        
        # Maximum should exist
        max_idx = curve['revpar'].idxmax()
        assert max_idx is not None
    
    def test_optimal_price_respects_bounds(self):
        """Optimal price should respect min/max constraints."""
        optimal_price, _, _ = find_revpar_optimal_price(
            current_price=100.0,
            base_occupancy=0.50,
            elasticity=MARKET_ELASTICITY,
            min_price_pct=0.80,
            max_price_pct=1.20
        )
        
        assert optimal_price >= 80.0
        assert optimal_price <= 120.0
    
    def test_highly_elastic_demand_lowers_price(self):
        """
        With elastic demand (|ε| > 1), optimal price should be lower.
        """
        optimal_price, optimal_revpar, _ = find_revpar_optimal_price(
            current_price=100.0,
            base_occupancy=0.50,
            elasticity=-2.0,  # Highly elastic
            min_price_pct=0.50,
            max_price_pct=1.50
        )
        
        # With elastic demand, lower prices capture more revenue through volume
        assert optimal_price < 100.0


class TestScenarioClassification:
    """Test scenario classification logic."""
    
    def test_scenario_a_underperforming_underpriced(self):
        """Scenario A: Underperforming + structurally underpriced → RAISE +20-30%."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=40.0,
            peer_revpar=60.0,
            revpar_gap=-0.33,
            hotel_price=80.0,
            peer_price=100.0,
            price_gap=-0.20,
            hotel_occupancy=0.50,
            peer_occupancy=0.60,
            occupancy_gap=-0.10,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        fair_value = FairValueResult(
            predicted_price=120.0,  # Model says worth more
            actual_price=80.0,
            value_residual=40.0    # Underpriced by €40
        )
        
        scenario, reasoning = classify_scenario(revpar_comparison, fair_value)
        
        assert scenario == RecommendationScenario.A
        assert "underpriced" in reasoning.lower() or "raise" in reasoning.lower()
    
    def test_scenario_b_underperforming_fair_value(self):
        """Scenario B: Underperforming + fair value → RAISE +10-15%."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=45.0,
            peer_revpar=60.0,
            revpar_gap=-0.25,
            hotel_price=90.0,
            peer_price=100.0,
            price_gap=-0.10,
            hotel_occupancy=0.50,
            peer_occupancy=0.60,
            occupancy_gap=-0.10,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        fair_value = FairValueResult(
            predicted_price=95.0,  # Close to actual
            actual_price=90.0,
            value_residual=5.0    # Nearly fair
        )
        
        scenario, reasoning = classify_scenario(revpar_comparison, fair_value)
        
        assert scenario == RecommendationScenario.B
    
    def test_scenario_c_overpriced(self):
        """Scenario C: Overpriced by model → LOWER 5-10%."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=35.0,
            peer_revpar=50.0,
            revpar_gap=-0.30,
            hotel_price=100.0,
            peer_price=100.0,
            price_gap=0.0,
            hotel_occupancy=0.35,
            peer_occupancy=0.50,
            occupancy_gap=-0.15,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.LOWER_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        fair_value = FairValueResult(
            predicted_price=70.0,  # Model says overpriced
            actual_price=100.0,
            value_residual=-30.0  # Overpriced by €30
        )
        
        scenario, reasoning = classify_scenario(revpar_comparison, fair_value)
        
        assert scenario == RecommendationScenario.C
        assert "overpriced" in reasoning.lower() or "lower" in reasoning.lower()
    
    def test_scenario_d_quality_issue(self):
        """Scenario D: Good occupancy but low RevPAR → quality issue, HOLD."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=40.0,
            peer_revpar=60.0,
            revpar_gap=-0.33,
            hotel_price=70.0,
            peer_price=100.0,
            price_gap=-0.30,
            hotel_occupancy=0.57,  # Good occupancy!
            peer_occupancy=0.60,
            occupancy_gap=-0.03,   # Almost at peer level
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.HOLD,
            n_peers=5,
            peer_source="geographic"
        )
        
        scenario, reasoning = classify_scenario(revpar_comparison, None)
        
        assert scenario == RecommendationScenario.D
        assert "quality" in reasoning.lower() or "hold" in reasoning.lower()


class TestSafetyValve:
    """Test occupancy safety valve logic."""
    
    def test_safety_valve_triggers_on_low_occupancy(self):
        """Safety valve should prevent price increases with critically low occupancy."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=30.0,
            peer_revpar=50.0,
            revpar_gap=-0.40,
            hotel_price=75.0,
            peer_price=100.0,
            price_gap=-0.25,
            hotel_occupancy=0.40,
            peer_occupancy=0.50,
            occupancy_gap=-0.10,  # Exactly at threshold
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,  # Would recommend raise
            n_peers=5,
            peer_source="geographic"
        )
        
        fair_value = FairValueResult(
            predicted_price=100.0,
            actual_price=75.0,
            value_residual=25.0  # Would suggest raise
        )
        
        rec = get_triangulated_recommendation(revpar_comparison, fair_value)
        
        # Safety valve should have triggered
        assert rec.occupancy_safety_triggered
        assert rec.scenario == RecommendationScenario.D
        assert rec.change_pct == 0.0  # Should HOLD
    
    def test_no_suicide_recommendations(self):
        """No price increase > 25% when occupancy gap is critically low."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=20.0,
            peer_revpar=50.0,
            revpar_gap=-0.60,
            hotel_price=50.0,
            peer_price=100.0,
            price_gap=-0.50,
            hotel_occupancy=0.40,
            peer_occupancy=0.50,
            occupancy_gap=-0.10,  # Critical gap
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        rec = get_triangulated_recommendation(revpar_comparison, None)
        
        # Should NOT recommend aggressive price increase
        assert rec.change_pct <= 25.0


class TestTriangulatedPricing:
    """Test price calculation for each scenario."""
    
    def test_scenario_a_raises_20_to_30_pct(self):
        """Scenario A should raise prices 20-30%."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=40.0,
            peer_revpar=60.0,
            revpar_gap=-0.33,
            hotel_price=80.0,
            peer_price=100.0,
            price_gap=-0.20,
            hotel_occupancy=0.50,
            peer_occupancy=0.60,
            occupancy_gap=-0.10,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        fair_value = FairValueResult(
            predicted_price=110.0,
            actual_price=80.0,
            value_residual=30.0
        )
        
        price, change_pct = calculate_triangulated_price(
            80.0, RecommendationScenario.A, revpar_comparison, fair_value
        )
        
        assert 20.0 <= change_pct <= 30.0
        assert price >= 96.0  # At least +20%
        assert price <= 104.0  # At most +30%
    
    def test_scenario_b_raises_10_to_15_pct(self):
        """Scenario B should raise prices ~12.5%."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=45.0,
            peer_revpar=60.0,
            revpar_gap=-0.25,
            hotel_price=90.0,
            peer_price=100.0,
            price_gap=-0.10,
            hotel_occupancy=0.50,
            peer_occupancy=0.60,
            occupancy_gap=-0.10,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        price, change_pct = calculate_triangulated_price(
            90.0, RecommendationScenario.B, revpar_comparison, None
        )
        
        assert 10.0 <= change_pct <= 15.0
    
    def test_scenario_c_lowers_5_to_10_pct(self):
        """Scenario C should lower prices 5-10%."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=35.0,
            peer_revpar=50.0,
            revpar_gap=-0.30,
            hotel_price=100.0,
            peer_price=100.0,
            price_gap=0.0,
            hotel_occupancy=0.35,
            peer_occupancy=0.50,
            occupancy_gap=-0.15,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.LOWER_PRICE,
            n_peers=5,
            peer_source="geographic"
        )
        
        fair_value = FairValueResult(
            predicted_price=85.0,
            actual_price=100.0,
            value_residual=-15.0
        )
        
        price, change_pct = calculate_triangulated_price(
            100.0, RecommendationScenario.C, revpar_comparison, fair_value
        )
        
        assert -10.0 <= change_pct <= -5.0
    
    def test_scenario_d_holds(self):
        """Scenario D should hold current price."""
        revpar_comparison = RevPARComparison(
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
        
        price, change_pct = calculate_triangulated_price(
            100.0, RecommendationScenario.D, revpar_comparison, None
        )
        
        assert change_pct == 0.0
        assert price == 100.0


class TestConfidence:
    """Test confidence level determination."""
    
    def test_twin_is_high_confidence(self):
        """Twin match should be high confidence."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=50.0, peer_revpar=50.0, revpar_gap=0.0,
            hotel_price=100.0, peer_price=100.0, price_gap=0.0,
            hotel_occupancy=0.5, peer_occupancy=0.5, occupancy_gap=0.0,
            signal=PerformanceSignal.OPTIMAL,
            opportunity=PriceOpportunity.HOLD,
            n_peers=1,
            peer_source="twin"
        )
        
        confidence = determine_confidence(revpar_comparison)
        assert confidence == Confidence.HIGH
    
    def test_peer_group_is_medium_confidence(self):
        """Peer group should be medium confidence."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=50.0, peer_revpar=50.0, revpar_gap=0.0,
            hotel_price=100.0, peer_price=100.0, price_gap=0.0,
            hotel_occupancy=0.5, peer_occupancy=0.5, occupancy_gap=0.0,
            signal=PerformanceSignal.OPTIMAL,
            opportunity=PriceOpportunity.HOLD,
            n_peers=10,
            peer_source="peer_group"
        )
        
        confidence = determine_confidence(revpar_comparison)
        assert confidence == Confidence.MEDIUM
    
    def test_geographic_few_peers_is_low_confidence(self):
        """Geographic with few/dissimilar peers should be low confidence."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=50.0, peer_revpar=50.0, revpar_gap=0.0,
            hotel_price=100.0, peer_price=100.0, price_gap=0.0,
            hotel_occupancy=0.5, peer_occupancy=0.5, occupancy_gap=0.0,
            signal=PerformanceSignal.OPTIMAL,
            opportunity=PriceOpportunity.HOLD,
            n_peers=2,
            peer_source="geographic",
            avg_similarity_score=0.3  # Low similarity
        )
        
        confidence = determine_confidence(revpar_comparison)
        assert confidence == Confidence.LOW


class TestRevPAROptimizedRecommendation:
    """Test the full RevPAR-optimized recommendation flow."""
    
    def test_recommendation_serialization(self):
        """Recommendation should serialize to dict correctly."""
        revpar_comparison = RevPARComparison(
            hotel_revpar=50.0, peer_revpar=60.0, revpar_gap=-0.167,
            hotel_price=100.0, peer_price=100.0, price_gap=0.0,
            hotel_occupancy=0.50, peer_occupancy=0.60, occupancy_gap=-0.10,
            signal=PerformanceSignal.UNDERPERFORMING,
            opportunity=PriceOpportunity.RAISE_PRICE,
            n_peers=5, peer_source="geographic"
        )
        
        rec = get_revpar_optimized_recommendation(
            revpar_comparison,
            hotel_id=123,
            target_date=None
        )
        
        d = rec.to_dict()
        
        assert 'optimal_price' in d
        assert 'optimal_revpar' in d
        assert 'revpar_lift_pct' in d
        assert 'confidence' in d
        assert d['hotel_id'] == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

