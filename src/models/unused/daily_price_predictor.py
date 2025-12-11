"""
Daily Price Predictor.

Unified interface for predicting hotel prices for specific dates.
Combines:
1. BaselinePricingModel (Random Forest, R² = 0.78) → Weekly average
2. DOWAdjustmentModel → Day-of-week multipliers

Usage:
    predictor = DailyPricePredictor()
    predictor.fit(train_df, con)
    
    # Predict price for a specific date
    price = predictor.predict_date(
        hotel_features={'latitude': 40.4, 'longitude': -3.7, ...},
        date='2024-07-12'  # Friday
    )
    
    # Predict full week
    prices = predictor.predict_week(
        hotel_features={...},
        week_start='2024-07-08'
    )
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from src.models.baseline_pricing import BaselinePricingModel
from src.models.dow_adjustments import DOWAdjustmentModel, DOW_NAMES


@dataclass
class DailyPricePrediction:
    """Prediction for a single day."""
    date: str
    day_of_week: str
    predicted_price: float
    baseline_price: float
    dow_multiplier: float
    
    def __repr__(self) -> str:
        pct = (self.dow_multiplier - 1) * 100
        return f"{self.date} ({self.day_of_week[:3]}): €{self.predicted_price:.0f} (base €{self.baseline_price:.0f} × {self.dow_multiplier:.2f} = {pct:+.0f}%)"


@dataclass
class WeeklyPricePrediction:
    """Prediction for a full week."""
    week_start: str
    baseline_price: float
    daily_prices: Dict[str, float]
    daily_multipliers: Dict[str, float]
    
    def __repr__(self) -> str:
        lines = [f"Week of {self.week_start} (baseline: €{self.baseline_price:.0f})"]
        lines.append("-" * 45)
        for day in DOW_NAMES:
            price = self.daily_prices[day]
            mult = self.daily_multipliers[day]
            pct = (mult - 1) * 100
            lines.append(f"  {day:10s}: €{price:6.0f}  ({pct:+.0f}%)")
        lines.append("-" * 45)
        avg = sum(self.daily_prices.values()) / 7
        lines.append(f"  {'Weekly avg':10s}: €{avg:6.0f}")
        return "\n".join(lines)


class DailyPricePredictor:
    """
    Predicts hotel prices for specific dates.
    
    For cold-start hotels, uses:
    1. Hotel features + peer prices → baseline weekly rate
    2. Segment-based DOW patterns → daily adjustments
    
    Model Performance:
    - Baseline: R² = 0.783, MAPE = 11.8%
    - DOW patterns: Empirical from 665K bookings
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.baseline_model = BaselinePricingModel()
        self.dow_model = DOWAdjustmentModel()
        self.is_fitted = False
    
    def fit(
        self,
        train_df: pd.DataFrame,
        con=None
    ) -> 'DailyPricePredictor':
        """
        Fit both models.
        
        Args:
            train_df: Training DataFrame with features and actual_price
            con: Database connection for DOW model (optional if train_df has booking-level data)
        
        Returns:
            self
        """
        print("Fitting DailyPricePredictor...")
        
        # Fit baseline model on weekly data
        print("\n1. Fitting baseline pricing model...")
        self.baseline_model.fit(train_df, target_col='actual_price')
        
        # Fit DOW model
        print("\n2. Fitting DOW adjustment model...")
        if con is not None:
            # Use booking-level data for more accurate DOW patterns
            from src.models.dow_adjustments import fit_dow_model_from_bookings
            self.dow_model = fit_dow_model_from_bookings(con)
        else:
            # Use aggregated data (less accurate but works)
            self.dow_model.fit(train_df)
        
        self.is_fitted = True
        print("\n✓ DailyPricePredictor ready")
        return self
    
    def predict_date(
        self,
        hotel_features: Union[pd.DataFrame, Dict],
        date: Union[str, datetime],
        hotel_id: Optional[int] = None
    ) -> DailyPricePrediction:
        """
        Predict price for a specific date.
        
        Args:
            hotel_features: Hotel feature dict or single-row DataFrame
            date: Target date (str 'YYYY-MM-DD' or datetime)
            hotel_id: Optional hotel ID for hotel-specific DOW patterns
        
        Returns:
            DailyPricePrediction with price and components
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Parse date
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        day_name = DOW_NAMES[day_of_week]
        
        # Convert features to DataFrame if needed
        if isinstance(hotel_features, dict):
            features_df = pd.DataFrame([hotel_features])
        else:
            features_df = hotel_features.copy()
        
        # Get baseline price
        baseline_price = self.baseline_model.predict(features_df)[0]
        
        # Get DOW multiplier
        feature_dict = hotel_features if isinstance(hotel_features, dict) else hotel_features.iloc[0].to_dict()
        dow_multiplier = self.dow_model.get_multiplier(
            day_of_week=day_of_week,
            hotel_id=hotel_id,
            hotel_features=feature_dict
        )
        
        # Calculate daily price
        daily_price = baseline_price * dow_multiplier
        
        return DailyPricePrediction(
            date=date.strftime('%Y-%m-%d'),
            day_of_week=day_name,
            predicted_price=daily_price,
            baseline_price=baseline_price,
            dow_multiplier=dow_multiplier
        )
    
    def predict_week(
        self,
        hotel_features: Union[pd.DataFrame, Dict],
        week_start: Union[str, datetime],
        hotel_id: Optional[int] = None
    ) -> WeeklyPricePrediction:
        """
        Predict prices for a full week (Monday-Sunday).
        
        Args:
            hotel_features: Hotel feature dict or single-row DataFrame
            week_start: Start of week (Monday)
            hotel_id: Optional hotel ID for hotel-specific DOW patterns
        
        Returns:
            WeeklyPricePrediction with all 7 days
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Parse date
        if isinstance(week_start, str):
            week_start = datetime.strptime(week_start, '%Y-%m-%d')
        
        # Convert features to DataFrame if needed
        if isinstance(hotel_features, dict):
            features_df = pd.DataFrame([hotel_features])
        else:
            features_df = hotel_features.copy()
        
        # Get baseline price
        baseline_price = self.baseline_model.predict(features_df)[0]
        
        # Get feature dict for DOW model
        feature_dict = hotel_features if isinstance(hotel_features, dict) else hotel_features.iloc[0].to_dict()
        
        # Calculate daily prices
        daily_prices = {}
        daily_multipliers = {}
        
        for i, day_name in enumerate(DOW_NAMES):
            mult = self.dow_model.get_multiplier(
                day_of_week=i,
                hotel_id=hotel_id,
                hotel_features=feature_dict
            )
            daily_multipliers[day_name] = mult
            daily_prices[day_name] = baseline_price * mult
        
        return WeeklyPricePrediction(
            week_start=week_start.strftime('%Y-%m-%d'),
            baseline_price=baseline_price,
            daily_prices=daily_prices,
            daily_multipliers=daily_multipliers
        )
    
    def predict_date_range(
        self,
        hotel_features: Union[pd.DataFrame, Dict],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        hotel_id: Optional[int] = None
    ) -> List[DailyPricePrediction]:
        """
        Predict prices for a date range.
        
        Args:
            hotel_features: Hotel features
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            hotel_id: Optional hotel ID
        
        Returns:
            List of DailyPricePrediction for each day
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        predictions = []
        current = start_date
        while current <= end_date:
            pred = self.predict_date(hotel_features, current, hotel_id)
            predictions.append(pred)
            current += timedelta(days=1)
        
        return predictions


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    from pathlib import Path
    from lib.db import init_db
    from lib.data_validator import CleaningConfig, DataCleaner
    from src.features.engineering import engineer_validated_features, add_peer_price_features, PEER_RADIUS_KM
    from src.models.evaluation.comprehensive_cold_start import load_all_hotel_weeks, get_hotel_ids_from_bookings
    
    print("=" * 60)
    print("DAILY PRICE PREDICTOR - Demo")
    print("=" * 60)
    
    # Load data
    config = CleaningConfig(
        remove_negative_prices=True,
        remove_zero_prices=True,
        remove_low_prices=True,
    )
    cleaner = DataCleaner(config)
    con = cleaner.clean(init_db())
    
    # Get train hotels (60%)
    all_hotel_ids = get_hotel_ids_from_bookings(con)
    np.random.seed(42)
    train_hotel_ids = set(np.random.permutation(all_hotel_ids)[:int(len(all_hotel_ids) * 0.6)])
    
    # Load and prepare training data
    train_df = load_all_hotel_weeks(con, min_price=50, max_price=200, hotel_ids=train_hotel_ids)
    train_df = engineer_validated_features(train_df)
    train_df = add_peer_price_features(train_df, radius_km=PEER_RADIUS_KM)
    
    # Fit predictor
    predictor = DailyPricePredictor()
    predictor.fit(train_df, con)
    
    # Example: Cold-start Madrid hotel
    print("\n" + "=" * 60)
    print("EXAMPLE: New Madrid hotel asking for next week's prices")
    print("=" * 60)
    
    madrid_hotel = {
        'latitude': 40.42,
        'longitude': -3.70,
        'is_coastal': 0,
        'is_madrid_metro': 1,
        'dist_center_km': 2.0,
        'log_room_size': np.log1p(25),
        'amenities_score': 3,
        'total_rooms': 50,
        'peer_price_mean': 110,
        'peer_price_median': 105,
        'peer_price_p25': 85,
        'peer_price_p75': 130,
        'n_peers_10km': 45,
    }
    
    week = predictor.predict_week(madrid_hotel, '2024-07-15')
    print(week)
    
    # Example: Specific date
    print("\n" + "=" * 60)
    print("EXAMPLE: Price for Friday July 19, 2024")
    print("=" * 60)
    
    friday = predictor.predict_date(madrid_hotel, '2024-07-19')
    print(friday)
    
    # Example: Date range
    print("\n" + "=" * 60)
    print("EXAMPLE: 10-day forecast")
    print("=" * 60)
    
    forecast = predictor.predict_date_range(madrid_hotel, '2024-07-15', '2024-07-24')
    for pred in forecast:
        print(pred)

