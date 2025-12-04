"""
Inference pipeline for price recommendation.

Given (hotel_id, date), outputs recommended price to maximize RevPAR.

Uses two-stage model:
1. Stage 1: KNN peer price statistics from similar hotels
2. Stage 2: Temporal adjustment for date-specific pricing (holidays, weekends)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, date
from typing import Dict, Optional, List
from dataclasses import dataclass

from ml_pipeline.config import (
    GLOBAL_ELASTICITY,
    SEGMENT_ELASTICITY,
    get_strategy,
    calculate_expected_revpar_change,
    SAFE_ZONE,
    RISK_THRESHOLD
)
from ml_pipeline.two_stage_model import TwoStageModel, Stage1PeerPrice, Stage2TemporalModel

# Model directory (where two_stage_model.py saves)
MODEL_DIR = Path(__file__).parent.parent / 'outputs' / 'models'
from lib.holiday_features import is_holiday, is_near_holiday, map_hotels_to_admin1

CITIES500_PATH = Path(__file__).parent.parent / 'data' / 'cities500.json'


@dataclass
class PriceRecommendation:
    """Result of price recommendation."""
    hotel_id: int
    date: str
    strategy: str
    peer_price: float
    recommended_price: float
    price_deviation_pct: float
    expected_revpar_lift_pct: float
    confidence: str
    elasticity_used: float
    market_segment: Optional[str] = None
    is_holiday: bool = False
    peer_price_stats: Optional[Dict] = None


class PriceRecommender:
    """
    Price recommendation using two-stage model.
    
    Stage 1: KNN finds similar hotels → peer price statistics
    Stage 2: RF adjusts for date → final predicted price
    
    Applies elasticity-based markup for RevPAR optimization.
    """
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Loads two-stage model."""
        model_path = self.model_dir / 'two_stage_model.pkl'
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded two-stage model (MAPE ≈ 25.7%)")
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python ml_pipeline/two_stage_model.py` first."
            )
    
    def _get_segment(self, hotel_data: pd.DataFrame) -> str:
        """Determines market segment for elasticity."""
        if 'is_coastal' in hotel_data.columns and hotel_data['is_coastal'].iloc[0] == 1:
            return 'Coastal/Resort'
        if 'city_standardized' in hotel_data.columns:
            city = hotel_data['city_standardized'].iloc[0]
            if city == 'madrid':
                return 'Urban/Madrid'
        return 'Provincial/Regional'
    
    def _get_holiday_features(self, target_date: date, subdiv: str = 'MD') -> Dict:
        """Gets holiday features for a specific date."""
        return {
            'is_holiday': int(is_holiday(target_date, subdiv)),
            'is_holiday_pm1': int(is_near_holiday(target_date, subdiv, buffer_days=1)),
            'is_holiday_pm2': int(is_near_holiday(target_date, subdiv, buffer_days=2))
        }
    
    def predict_price(
        self,
        hotel_data: pd.DataFrame,
        target_date: date,
        exclude_hotel_id: Optional[int] = None
    ) -> float:
        """
        Predicts price for hotel on specific date.
        
        Args:
            hotel_data: DataFrame with hotel features
            target_date: Date for prediction
            exclude_hotel_id: Hotel ID to exclude from Stage 1 neighbors
        
        Returns:
            Predicted daily price
        """
        df = hotel_data.copy()
        
        # Add temporal features
        df['day_of_week'] = target_date.weekday()
        df['month_number'] = target_date.month
        df['month_sin'] = np.sin(2 * np.pi * target_date.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * target_date.month / 12)
        df['is_weekend'] = int(target_date.weekday() >= 5)
        df['is_summer'] = int(target_date.month in [6, 7, 8])
        df['is_winter'] = int(target_date.month in [12, 1, 2])
        
        # Add holiday features
        subdiv = 'MD'  # Default to Madrid; could be improved with hotel location
        holiday_feats = self._get_holiday_features(target_date, subdiv)
        for k, v in holiday_feats.items():
            df[k] = v
        
        # Get prediction from two-stage model
        predictions = self.model.predict(df, exclude_hotel_id=exclude_hotel_id)
        
        return float(predictions[0])
    
    def recommend_price(
        self,
        hotel_id: int,
        date_str: str,
        hotel_data: pd.DataFrame,
        strategy: str = 'safe'
    ) -> PriceRecommendation:
        """
        Generates price recommendation for a hotel-date.
        
        Args:
            hotel_id: Hotel identifier
            date_str: Target date (YYYY-MM-DD)
            hotel_data: DataFrame with hotel features
            strategy: One of 'conservative', 'safe', 'optimal'
        
        Returns:
            PriceRecommendation with peer_price, recommended_price, etc.
        """
        target_date = pd.to_datetime(date_str).date()
        
        # Get predicted price (this is the "fair market price")
        peer_price = self.predict_price(hotel_data, target_date, exclude_hotel_id=hotel_id)
        
        # Get strategy parameters
        strat = get_strategy(strategy)
        
        # Get segment-specific elasticity
        segment = self._get_segment(hotel_data)
        elasticity = SEGMENT_ELASTICITY.get(segment, GLOBAL_ELASTICITY)
        
        # Calculate recommended price with markup
        deviation_pct = strat.price_deviation_pct
        recommended_price = peer_price * (1 + deviation_pct / 100)
        
        # Calculate expected RevPAR lift
        revpar_lift = calculate_expected_revpar_change(deviation_pct, elasticity)
        
        # Check if it's a holiday
        holiday_flag = is_holiday(target_date, 'MD')
        
        # Determine confidence
        confidence = 'high' if peer_price > 0 else 'low'
        
        return PriceRecommendation(
            hotel_id=hotel_id,
            date=date_str,
            strategy=strategy,
            peer_price=round(peer_price, 2),
            recommended_price=round(recommended_price, 2),
            price_deviation_pct=round(deviation_pct, 1),
            expected_revpar_lift_pct=round(revpar_lift, 1),
            confidence=confidence,
            elasticity_used=elasticity,
            market_segment=segment,
            is_holiday=holiday_flag
        )


def recommend_price(
    hotel_id: int,
    date: str,
    strategy: str = 'safe',
    model_dir: Path = MODEL_DIR
) -> Dict:
    """
    High-level API for price recommendation.
    
    Args:
        hotel_id: The hotel to price
        date: Target date (YYYY-MM-DD)
        strategy: 'conservative' (+15%), 'safe' (+30%), 'optimal' (+45%)
        model_dir: Directory containing trained model
    
    Returns:
        Dict with peer_price, recommended_price, expected_revpar_lift, etc.
    
    Example:
        >>> result = recommend_price(hotel_id=12345, date='2024-12-25', strategy='safe')
        >>> print(f"Charge €{result['recommended_price']}/night")
    """
    from lib.db import init_db
    from ml_pipeline.features import standardize_city, VIEW_QUALITY_MAP
    
    con = init_db()
    
    # Load hotel data
    query = f"""
    SELECT 
        b.hotel_id,
        hl.city,
        hl.latitude,
        hl.longitude,
        AVG(br.room_size) as avg_room_size,
        MODE() WITHIN GROUP (ORDER BY br.room_type) as room_type,
        MODE() WITHIN GROUP (ORDER BY COALESCE(NULLIF(br.room_view, ''), 'no_view')) as room_view,
        MAX(r.children_allowed) as children_allowed,
        MAX(r.max_occupancy) as room_capacity_pax,
        SUM(DISTINCT r.number_of_rooms) as total_capacity,
        (CAST(MAX(r.events_allowed) AS INT) + 
         CAST(MAX(r.pets_allowed) AS INT) + 
         CAST(MAX(r.smoking_allowed) AS INT) + 
         CAST(MAX(r.children_allowed) AS INT)) AS amenities_score
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN rooms r ON br.room_id = r.id
    JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
    WHERE b.hotel_id = {hotel_id}
    GROUP BY b.hotel_id, hl.city, hl.latitude, hl.longitude
    """
    
    hotel_data = con.execute(query).fetchdf()
    
    if hotel_data.empty:
        raise ValueError(f"Hotel {hotel_id} not found in database")
    
    # Engineer features
    hotel_data['log_room_size'] = np.log1p(hotel_data['avg_room_size'])
    hotel_data['total_capacity_log'] = np.log1p(hotel_data['total_capacity'])
    hotel_data['city_standardized'] = hotel_data['city'].apply(standardize_city)
    hotel_data['view_quality_ordinal'] = hotel_data['room_view'].map(VIEW_QUALITY_MAP).fillna(0)
    
    # Get recommendation
    recommender = PriceRecommender(model_dir)
    rec = recommender.recommend_price(hotel_id, date, hotel_data, strategy)
    
    return {
        'hotel_id': rec.hotel_id,
        'date': rec.date,
        'peer_price': rec.peer_price,
        'recommended_price': rec.recommended_price,
        'price_deviation_pct': rec.price_deviation_pct,
        'expected_revpar_lift_pct': rec.expected_revpar_lift_pct,
        'confidence': rec.confidence,
        'strategy': rec.strategy,
        'market_segment': rec.market_segment,
        'is_holiday': rec.is_holiday
    }


if __name__ == "__main__":
    print("=" * 80)
    print("PRICE RECOMMENDATION - TWO-STAGE MODEL")
    print("=" * 80)
    
    try:
        from lib.db import init_db
        con = init_db()
        
        # Get a sample hotel
        sample = con.execute("SELECT DISTINCT hotel_id FROM bookings LIMIT 1").fetchdf()
        
        if not sample.empty:
            hotel_id = sample['hotel_id'].iloc[0]
            
            # Test on regular day vs holiday
            test_dates = [
                ('2024-12-15', 'Regular Sunday'),
                ('2024-12-25', 'Christmas Day'),
                ('2024-08-15', 'Assumption Day')
            ]
            
            for date_str, desc in test_dates:
                print(f"\n{desc} ({date_str}):")
                result = recommend_price(hotel_id, date_str, strategy='safe')
                print(f"  Peer Price: €{result['peer_price']:.2f}")
                print(f"  Recommended: €{result['recommended_price']:.2f} (+{result['price_deviation_pct']:.0f}%)")
                print(f"  Expected RevPAR Lift: +{result['expected_revpar_lift_pct']:.1f}%")
                print(f"  Is Holiday: {result['is_holiday']}")
                
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
    except Exception as e:
        print(f"\nError: {e}")
