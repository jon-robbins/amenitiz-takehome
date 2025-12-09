"""
Unified Price Recommender.

Main API for generating price recommendations.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import pickle

import numpy as np
import pandas as pd

from src.data.loader import load_hotel_month_data, get_clean_connection, load_distance_features
from src.features.engineering import engineer_features, standardize_city, get_market_segment, MARKET_ELASTICITY
from src.features.peers import compute_peer_stats, add_peer_features
from src.models.occupancy import OccupancyModel
from src.recommender.diagnosis import diagnose_pricing, calculate_recommended_price, PriceDiagnosis


@dataclass
class PriceRecommendation:
    """Result of price recommendation for a hotel."""
    hotel_id: int
    date: str
    
    # Recommendation
    recommended_price: float
    direction: str  # "increase", "decrease", "maintain"
    change_pct: float
    
    # Current state
    current_price: float
    peer_price: float
    price_premium_pct: float
    
    # Occupancy
    actual_occupancy: float
    expected_occupancy: float
    occ_residual: float
    
    # Context
    market_segment: str
    n_peers: int
    confidence: str
    reasoning: str
    
    def __repr__(self) -> str:
        return (
            f"PriceRecommendation(\n"
            f"  hotel={self.hotel_id}, date='{self.date}'\n"
            f"  {self.direction.upper()}: €{self.current_price:.0f} → €{self.recommended_price:.0f} ({self.change_pct:+.1f}%)\n"
            f"  reasoning='{self.reasoning}'\n"
            f")"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'hotel_id': self.hotel_id,
            'date': self.date,
            'recommended_price': self.recommended_price,
            'direction': self.direction,
            'change_pct': self.change_pct,
            'current_price': self.current_price,
            'peer_price': self.peer_price,
            'price_premium_pct': self.price_premium_pct,
            'actual_occupancy': self.actual_occupancy,
            'expected_occupancy': self.expected_occupancy,
            'occ_residual': self.occ_residual,
            'market_segment': self.market_segment,
            'n_peers': self.n_peers,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }


class PriceRecommender:
    """
    Unified price recommendation system.
    
    Uses:
    1. Peer comparison for price positioning
    2. Occupancy model for demand prediction
    3. Diagnosis logic for direction (increase/decrease/maintain)
    4. Validated market elasticity for adjustments
    
    Usage:
        recommender = PriceRecommender()
        recommender.fit()
        rec = recommender.recommend_price(hotel_id=123, date='2025-01-15')
    """
    
    def __init__(self, elasticity: float = MARKET_ELASTICITY):
        self.elasticity = elasticity
        self.occupancy_model = None
        self.hotel_data = None
        self.peer_stats = None
        self.distance_features = None
        self._city_encoder = None
        self._is_fitted = False
    
    def fit(self, quick: bool = False) -> 'PriceRecommender':
        """
        Fit the recommender.
        
        Args:
            quick: Use sample data for faster fitting
        
        Returns:
            self
        """
        print("=" * 70)
        print("PRICE RECOMMENDER - FITTING")
        print("=" * 70)
        
        # 1. Load data
        print("\n1. Loading data...")
        con = get_clean_connection()
        self.hotel_data = load_hotel_month_data(con)
        
        # 2. Engineer features
        print("\n2. Engineering features...")
        self.hotel_data['city_standardized'] = self.hotel_data['city'].apply(standardize_city)
        self.hotel_data = engineer_features(self.hotel_data)
        
        # 3. Compute peer stats
        print("\n3. Computing peer statistics...")
        self.peer_stats = compute_peer_stats(self.hotel_data)
        print(f"   {len(self.peer_stats):,} peer groups")
        
        # Add peer features to data
        self.hotel_data = add_peer_features(self.hotel_data, self.peer_stats)
        
        # 4. Train occupancy model
        print("\n4. Training occupancy model...")
        train_data = self.hotel_data
        if quick:
            train_data = train_data.sample(n=min(5000, len(train_data)), random_state=42)
        
        self.occupancy_model = OccupancyModel(elasticity=self.elasticity)
        self.occupancy_model.fit(train_data)
        
        # Predict expected occupancy for all data
        self.hotel_data['expected_occupancy'] = self.occupancy_model.predict(self.hotel_data)
        self.hotel_data['occ_residual'] = self.hotel_data['occupancy_rate'] - self.hotel_data['expected_occupancy']
        
        # 5. Load distance features
        print("\n5. Loading distance features...")
        self.distance_features = load_distance_features()
        
        self._is_fitted = True
        print("\n✓ Recommender fitted")
        
        return self
    
    def _get_hotel_context(
        self, 
        hotel_id: int, 
        target_month: int
    ) -> Optional[Dict]:
        """Get hotel's current state and peer comparison."""
        # Find hotel data for this month
        mask = (
            (self.hotel_data['hotel_id'] == hotel_id) & 
            (self.hotel_data['month_number'] == target_month)
        )
        matches = self.hotel_data[mask]
        
        if len(matches) == 0:
            # Try any month for this hotel
            mask = self.hotel_data['hotel_id'] == hotel_id
            matches = self.hotel_data[mask]
            if len(matches) == 0:
                return None
        
        row = matches.iloc[-1]  # Most recent
        
        return {
            'current_price': row['avg_price'],
            'peer_price': row['peer_price'],
            'price_premium': row['price_premium'],
            'actual_occ': row['occupancy_rate'],
            'expected_occ': row['expected_occupancy'],
            'occ_residual': row['occ_residual'],
            'n_peers': row.get('n_peers', 1),
            'city': row.get('city_standardized', 'other'),
            'room_type': row.get('room_type', 'unknown')
        }
    
    def _get_market_segment(self, hotel_id: int) -> str:
        """Get market segment for hotel."""
        if self.distance_features is None:
            return 'unknown'
        
        match = self.distance_features[self.distance_features['hotel_id'] == hotel_id]
        if len(match) == 0:
            return 'unknown'
        
        row = match.iloc[0]
        return get_market_segment(
            row.get('distance_from_coast', 999),
            row.get('distance_from_madrid', 999)
        )
    
    def recommend_price(
        self, 
        hotel_id: int, 
        date: str
    ) -> PriceRecommendation:
        """
        Get price recommendation for a hotel on a date.
        
        Args:
            hotel_id: Hotel identifier
            date: Target date (YYYY-MM-DD)
        
        Returns:
            PriceRecommendation with details
        """
        if not self._is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        target_date = pd.to_datetime(date)
        target_month = target_date.month
        
        # Get context
        context = self._get_hotel_context(hotel_id, target_month)
        
        if context is None:
            return PriceRecommendation(
                hotel_id=hotel_id,
                date=date,
                recommended_price=0,
                direction="maintain",
                change_pct=0,
                current_price=0,
                peer_price=0,
                price_premium_pct=0,
                actual_occupancy=0,
                expected_occupancy=0,
                occ_residual=0,
                market_segment="unknown",
                n_peers=0,
                confidence="low",
                reasoning="Insufficient data for this hotel"
            )
        
        # Diagnose
        diagnosis = diagnose_pricing(
            context['price_premium'],
            context['occ_residual']
        )
        
        # Calculate recommended price
        recommended = calculate_recommended_price(
            diagnosis.direction,
            context['current_price'],
            context['peer_price'],
            self.elasticity
        )
        
        change_pct = (recommended / context['current_price'] - 1) * 100 if context['current_price'] > 0 else 0
        
        # Confidence
        n_peers = context['n_peers']
        confidence = "high" if n_peers >= 10 else ("medium" if n_peers >= 3 else "low")
        
        return PriceRecommendation(
            hotel_id=hotel_id,
            date=date,
            recommended_price=round(recommended, 2),
            direction=diagnosis.direction,
            change_pct=round(change_pct, 1),
            current_price=round(context['current_price'], 2),
            peer_price=round(context['peer_price'], 2),
            price_premium_pct=round(context['price_premium'] * 100, 1),
            actual_occupancy=round(context['actual_occ'], 3),
            expected_occupancy=round(context['expected_occ'], 3),
            occ_residual=round(context['occ_residual'], 3),
            market_segment=self._get_market_segment(hotel_id),
            n_peers=n_peers,
            confidence=confidence,
            reasoning=diagnosis.reasoning
        )
    
    def recommend_batch(
        self, 
        hotel_ids: List[int], 
        date: str
    ) -> pd.DataFrame:
        """Get recommendations for multiple hotels."""
        results = []
        for hotel_id in hotel_ids:
            try:
                rec = self.recommend_price(hotel_id, date)
                results.append(rec.to_dict())
            except Exception as e:
                print(f"Error for hotel {hotel_id}: {e}")
        
        return pd.DataFrame(results)
    
    def get_recommendation_distribution(self, n_samples: int = 200) -> Dict:
        """
        Get distribution of recommendations across hotels.
        
        Returns dict with percentages for increase/decrease/maintain.
        """
        if not self._is_fitted:
            raise ValueError("Not fitted")
        
        hotel_ids = self.hotel_data['hotel_id'].unique()
        sample_ids = np.random.choice(hotel_ids, size=min(n_samples, len(hotel_ids)), replace=False)
        
        results = self.recommend_batch(list(sample_ids), '2024-06-15')
        
        if len(results) == 0:
            return {'error': 'No results'}
        
        dist = results['direction'].value_counts(normalize=True) * 100
        
        return {
            'n_samples': len(results),
            'pct_increase': dist.get('increase', 0),
            'pct_decrease': dist.get('decrease', 0),
            'pct_maintain': dist.get('maintain', 0),
            'avg_increase_pct': results[results['direction'] == 'increase']['change_pct'].mean(),
            'avg_decrease_pct': results[results['direction'] == 'decrease']['change_pct'].mean()
        }
    
    def save(self, path: Path) -> None:
        """Save recommender to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved recommender to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'PriceRecommender':
        """Load recommender from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

