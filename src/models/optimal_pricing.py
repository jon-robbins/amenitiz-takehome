"""
Optimal Pricing Model.

Uses the OccupancyModelWithPrice to find the RevPAR-maximizing price.

Given an occupancy model that predicts occupancy = f(price, features),
this model searches for the price that maximizes:

    RevPAR = Price × Occupancy(Price)

The search is constrained by:
1. Price bounds relative to peer prices
2. Maximum increase/decrease limits
3. Occupancy floor (don't drive occupancy too low)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.models.occupancy_with_price import OccupancyModelWithPrice


@dataclass
class OptimalPriceResult:
    """Result of optimal price search."""
    # Optimal recommendation
    optimal_price: float
    optimal_occupancy: float
    optimal_revpar: float
    
    # Baseline comparison
    baseline_price: float
    baseline_occupancy: float
    baseline_revpar: float
    
    # Improvement
    price_change_pct: float
    revpar_lift_pct: float
    
    # Peer context
    peer_price: float
    price_vs_peer_pct: float
    
    # Metadata
    search_method: str  # "grid_search", "analytical", "constrained"
    
    def __repr__(self) -> str:
        direction = "↑" if self.price_change_pct > 0 else "↓" if self.price_change_pct < 0 else "→"
        return (
            f"OptimalPriceResult:\n"
            f"  Baseline: €{self.baseline_price:.0f} → {self.baseline_occupancy:.0%} → RevPAR €{self.baseline_revpar:.0f}\n"
            f"  Optimal:  €{self.optimal_price:.0f} → {self.optimal_occupancy:.0%} → RevPAR €{self.optimal_revpar:.0f}\n"
            f"  Change:   {direction} {abs(self.price_change_pct):.1f}% price, {self.revpar_lift_pct:+.1f}% RevPAR\n"
            f"  vs Peers: {self.price_vs_peer_pct:+.1f}%"
        )


class OptimalPricingModel:
    """
    Finds RevPAR-maximizing price using the occupancy model.
    
    Method:
    1. For each candidate price in a range around peer/baseline
    2. Use occupancy model to predict occupancy at that price
    3. Calculate RevPAR = price × occupancy
    4. Find the price that maximizes RevPAR
    
    Constraints:
    - Price bounded by peer_price × [0.7, 1.3] (default)
    - Maximum 30% increase, 20% decrease from baseline
    - Minimum 20% occupancy floor
    
    Usage:
        optimal_model = OptimalPricingModel(occupancy_model)
        result = optimal_model.find_optimal_price(
            hotel_features,
            baseline_price=100,
            peer_price=95
        )
    """
    
    def __init__(
        self,
        occupancy_model: OccupancyModelWithPrice,
        max_increase_pct: float = 0.30,
        max_decrease_pct: float = 0.20,
        min_occupancy: float = 0.20,
        peer_price_range: Tuple[float, float] = (0.7, 1.3)
    ):
        """
        Initialize optimal pricing model.
        
        Args:
            occupancy_model: Fitted OccupancyModelWithPrice
            max_increase_pct: Maximum price increase (default 30%)
            max_decrease_pct: Maximum price decrease (default 20%)
            min_occupancy: Minimum acceptable occupancy (default 20%)
            peer_price_range: Search range as fraction of peer price
        """
        self.occupancy_model = occupancy_model
        self.max_increase_pct = max_increase_pct
        self.max_decrease_pct = max_decrease_pct
        self.min_occupancy = min_occupancy
        self.peer_price_range = peer_price_range
    
    def find_optimal_price(
        self,
        hotel_features: pd.DataFrame,
        baseline_price: float,
        peer_price: float,
        n_search_points: int = 50
    ) -> OptimalPriceResult:
        """
        Find the RevPAR-maximizing price for a hotel.
        
        Args:
            hotel_features: Hotel feature DataFrame (single row)
            baseline_price: Current/baseline price (€)
            peer_price: Average peer price (€)
            n_search_points: Number of prices to evaluate
        
        Returns:
            OptimalPriceResult with optimal price and expected outcomes
        """
        if not self.occupancy_model.is_fitted:
            raise ValueError("Occupancy model not fitted")
        
        # Define search bounds
        min_price = max(
            peer_price * self.peer_price_range[0],
            baseline_price * (1 - self.max_decrease_pct),
            30  # Absolute floor
        )
        max_price = min(
            peer_price * self.peer_price_range[1],
            baseline_price * (1 + self.max_increase_pct),
            500  # Absolute ceiling
        )
        
        # Generate candidate prices
        prices = np.linspace(min_price, max_price, n_search_points)
        
        # Evaluate each price
        best_revpar = -np.inf
        best_price = baseline_price
        best_occupancy = 0.5
        
        for price in prices:
            # Predict occupancy at this price
            occ = self.occupancy_model.predict_at_price(
                hotel_features, 
                price, 
                peer_price_mean=peer_price
            )[0]
            
            # Skip if occupancy too low
            if occ < self.min_occupancy:
                continue
            
            # Calculate RevPAR
            revpar = price * occ
            
            if revpar > best_revpar:
                best_revpar = revpar
                best_price = price
                best_occupancy = occ
        
        # Calculate baseline RevPAR
        baseline_occ = self.occupancy_model.predict_at_price(
            hotel_features,
            baseline_price,
            peer_price_mean=peer_price
        )[0]
        baseline_revpar = baseline_price * baseline_occ
        
        # Calculate improvements
        price_change_pct = (best_price - baseline_price) / baseline_price * 100
        revpar_lift_pct = (best_revpar - baseline_revpar) / max(baseline_revpar, 1) * 100
        price_vs_peer_pct = (best_price - peer_price) / peer_price * 100
        
        return OptimalPriceResult(
            optimal_price=best_price,
            optimal_occupancy=best_occupancy,
            optimal_revpar=best_revpar,
            baseline_price=baseline_price,
            baseline_occupancy=baseline_occ,
            baseline_revpar=baseline_revpar,
            price_change_pct=price_change_pct,
            revpar_lift_pct=revpar_lift_pct,
            peer_price=peer_price,
            price_vs_peer_pct=price_vs_peer_pct,
            search_method="grid_search"
        )
    
    def find_optimal_prices_batch(
        self,
        df: pd.DataFrame,
        baseline_price_col: str = 'baseline_price',
        peer_price_col: str = 'peer_price_mean'
    ) -> pd.DataFrame:
        """
        Find optimal prices for multiple hotels.
        
        Args:
            df: DataFrame with hotel features and prices
            baseline_price_col: Column with baseline prices
            peer_price_col: Column with peer prices
        
        Returns:
            DataFrame with optimal prices and expected outcomes
        """
        results = []
        
        for idx, row in df.iterrows():
            hotel_features = row.to_frame().T
            baseline_price = row[baseline_price_col]
            peer_price = row.get(peer_price_col, baseline_price)
            
            try:
                result = self.find_optimal_price(
                    hotel_features,
                    baseline_price,
                    peer_price
                )
                
                results.append({
                    'hotel_id': row.get('hotel_id', idx),
                    'optimal_price': result.optimal_price,
                    'optimal_occupancy': result.optimal_occupancy,
                    'optimal_revpar': result.optimal_revpar,
                    'baseline_price': result.baseline_price,
                    'baseline_occupancy': result.baseline_occupancy,
                    'baseline_revpar': result.baseline_revpar,
                    'price_change_pct': result.price_change_pct,
                    'revpar_lift_pct': result.revpar_lift_pct,
                })
            except Exception as e:
                results.append({
                    'hotel_id': row.get('hotel_id', idx),
                    'optimal_price': baseline_price,
                    'optimal_occupancy': np.nan,
                    'optimal_revpar': np.nan,
                    'baseline_price': baseline_price,
                    'baseline_occupancy': np.nan,
                    'baseline_revpar': np.nan,
                    'price_change_pct': 0.0,
                    'revpar_lift_pct': 0.0,
                })
        
        return pd.DataFrame(results)
    
    def get_demand_curve(
        self,
        hotel_features: pd.DataFrame,
        peer_price: float,
        price_range: Tuple[float, float] = (50, 300),
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Generate demand curve for visualization.
        
        Returns DataFrame with price, occupancy, revpar columns.
        """
        return self.occupancy_model.predict_demand_curve(
            hotel_features,
            price_range=price_range,
            n_points=n_points,
            peer_price_mean=peer_price
        )

