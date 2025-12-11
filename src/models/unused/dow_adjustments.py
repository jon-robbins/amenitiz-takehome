"""
Day-of-Week Price Adjustments.

Hotels have different pricing patterns based on their type:
- Business hotels (city center): Higher Mon-Thu
- Resort/leisure hotels (coastal): Higher Fri-Sat
- Mixed: Varies

This module learns segment-level DOW patterns from training data
and applies them for cold-start hotels based on their features.

Hierarchy:
1. Hotel-specific (if n_bookings >= threshold)
2. Segment-level (coastal, madrid_metro, provincial)
3. Market-level fallback
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# Day of week constants
DOW_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DOW_MAP = {name: i for i, name in enumerate(DOW_NAMES)}

# Minimum bookings to trust hotel-specific patterns
MIN_BOOKINGS_HOTEL_SPECIFIC = 50

# Segments
SEGMENTS = ['coastal', 'madrid_metro', 'provincial']


@dataclass
class DOWPattern:
    """Day-of-week pricing pattern."""
    multipliers: Dict[int, float]  # {0: 0.92, 1: 0.90, ..., 5: 1.15, 6: 0.98}
    n_bookings: int
    segment: str
    
    def get_multiplier(self, day: int) -> float:
        """Get multiplier for day of week (0=Monday, 6=Sunday)."""
        return self.multipliers.get(day, 1.0)
    
    def __repr__(self) -> str:
        lines = [f"DOWPattern ({self.segment}, n={self.n_bookings:,})"]
        for i, name in enumerate(DOW_NAMES):
            mult = self.multipliers.get(i, 1.0)
            pct = (mult - 1) * 100
            bar = "+" * int(max(0, pct / 2)) + "-" * int(max(0, -pct / 2))
            lines.append(f"  {name[:3]}: {mult:.2f} ({pct:+.0f}%) {bar}")
        return "\n".join(lines)


class DOWAdjustmentModel:
    """
    Hierarchical day-of-week price adjustment model.
    
    Learns DOW patterns at three levels:
    1. Market-level (all hotels)
    2. Segment-level (coastal, madrid_metro, provincial)
    3. Hotel-level (if enough data)
    
    For cold-start hotels, uses segment-level based on features.
    
    Usage:
        model = DOWAdjustmentModel()
        model.fit(train_df)
        
        # Get multiplier for a cold-start coastal hotel on Saturday
        mult = model.get_multiplier(
            hotel_features={'is_coastal': 1, 'is_madrid_metro': 0},
            day_of_week=5  # Saturday
        )
        
        # Apply to baseline price
        saturday_price = baseline_price * mult
    """
    
    def __init__(self, min_bookings_hotel: int = MIN_BOOKINGS_HOTEL_SPECIFIC):
        """
        Initialize DOW adjustment model.
        
        Args:
            min_bookings_hotel: Minimum bookings to use hotel-specific pattern
        """
        self.min_bookings_hotel = min_bookings_hotel
        self.market_pattern: Optional[DOWPattern] = None
        self.segment_patterns: Dict[str, DOWPattern] = {}
        self.hotel_patterns: Dict[int, DOWPattern] = {}
        self.is_fitted = False
    
    def _calculate_dow_pattern(
        self,
        df: pd.DataFrame,
        segment: str = 'market'
    ) -> DOWPattern:
        """
        Calculate DOW multipliers from booking data.
        
        Multiplier = segment_dow_avg / segment_overall_avg
        
        This normalizes so that average multiplier ≈ 1.0
        """
        if len(df) == 0:
            return DOWPattern(
                multipliers={i: 1.0 for i in range(7)},
                n_bookings=0,
                segment=segment
            )
        
        # Calculate average ADR by day of week
        dow_avg = df.groupby('arrival_dow')['adr'].mean()
        overall_avg = df['adr'].mean()
        
        # Calculate multipliers (normalize to overall average)
        multipliers = {}
        for dow in range(7):
            if dow in dow_avg.index:
                multipliers[dow] = dow_avg[dow] / overall_avg
            else:
                multipliers[dow] = 1.0
        
        return DOWPattern(
            multipliers=multipliers,
            n_bookings=len(df),
            segment=segment
        )
    
    def _get_segment(self, row: pd.Series) -> str:
        """Determine segment from hotel features."""
        if row.get('is_coastal', 0) == 1:
            return 'coastal'
        elif row.get('is_madrid_metro', 0) == 1:
            return 'madrid_metro'
        else:
            return 'provincial'
    
    def fit(self, df: pd.DataFrame) -> 'DOWAdjustmentModel':
        """
        Fit DOW patterns from training data.
        
        Args:
            df: Training DataFrame with columns:
                - arrival_date or arrival_dow
                - adr or actual_price (per-night price)
                - is_coastal, is_madrid_metro (segment features)
                - hotel_id (optional, for hotel-specific patterns)
        
        Returns:
            self
        """
        print("Fitting DOW adjustment model...")
        
        df = df.copy()
        
        # Ensure we have day of week
        if 'arrival_dow' not in df.columns:
            if 'arrival_date' in df.columns:
                df['arrival_dow'] = pd.to_datetime(df['arrival_date']).dt.dayofweek
            elif 'week_start' in df.columns:
                # For weekly data, we need to expand or estimate
                # Use the week_start as a proxy (not ideal but workable)
                df['arrival_dow'] = pd.to_datetime(df['week_start']).dt.dayofweek
            else:
                raise ValueError("Need arrival_date, arrival_dow, or week_start column")
        
        # Ensure we have ADR column
        if 'adr' not in df.columns:
            if 'actual_price' in df.columns:
                df['adr'] = df['actual_price']
            elif 'avg_adr' in df.columns:
                df['adr'] = df['avg_adr']
            else:
                raise ValueError("Need adr, actual_price, or avg_adr column")
        
        # Add segment if not present
        if 'segment' not in df.columns:
            df['segment'] = df.apply(self._get_segment, axis=1)
        
        # 1. Market-level pattern
        self.market_pattern = self._calculate_dow_pattern(df, 'market')
        print(f"  Market pattern: {len(df):,} bookings")
        
        # 2. Segment-level patterns
        for segment in SEGMENTS:
            segment_df = df[df['segment'] == segment]
            if len(segment_df) > 0:
                self.segment_patterns[segment] = self._calculate_dow_pattern(
                    segment_df, segment
                )
                print(f"  {segment}: {len(segment_df):,} bookings")
        
        # 3. Hotel-level patterns (for hotels with enough data)
        if 'hotel_id' in df.columns:
            hotel_counts = df.groupby('hotel_id').size()
            eligible_hotels = hotel_counts[hotel_counts >= self.min_bookings_hotel].index
            
            for hotel_id in eligible_hotels:
                hotel_df = df[df['hotel_id'] == hotel_id]
                segment = hotel_df['segment'].mode().iloc[0] if len(hotel_df) > 0 else 'provincial'
                self.hotel_patterns[hotel_id] = self._calculate_dow_pattern(
                    hotel_df, f'hotel_{hotel_id}'
                )
            
            print(f"  Hotel-specific patterns: {len(self.hotel_patterns):,} hotels")
        
        self.is_fitted = True
        return self
    
    def get_multiplier(
        self,
        day_of_week: int,
        hotel_id: Optional[int] = None,
        hotel_features: Optional[Dict] = None,
        shrinkage: float = 0.7
    ) -> float:
        """
        Get DOW multiplier for pricing.
        
        Hierarchy:
        1. If hotel_id has specific pattern → blend with segment
        2. Else use segment based on features
        3. Fallback to market
        
        Args:
            day_of_week: 0=Monday, 6=Sunday
            hotel_id: Optional hotel ID (for hotel-specific patterns)
            hotel_features: Dict with is_coastal, is_madrid_metro
            shrinkage: Weight for segment pattern when blending (0-1)
        
        Returns:
            Price multiplier for that day (e.g., 1.15 = +15%)
        """
        if not self.is_fitted:
            return 1.0
        
        # Determine segment
        if hotel_features:
            if hotel_features.get('is_coastal', 0) == 1:
                segment = 'coastal'
            elif hotel_features.get('is_madrid_metro', 0) == 1:
                segment = 'madrid_metro'
            else:
                segment = 'provincial'
        else:
            segment = 'provincial'
        
        # Get segment pattern
        segment_pattern = self.segment_patterns.get(segment, self.market_pattern)
        segment_mult = segment_pattern.get_multiplier(day_of_week) if segment_pattern else 1.0
        
        # Check for hotel-specific pattern
        if hotel_id and hotel_id in self.hotel_patterns:
            hotel_pattern = self.hotel_patterns[hotel_id]
            hotel_mult = hotel_pattern.get_multiplier(day_of_week)
            
            # Blend hotel and segment (shrinkage toward segment)
            return shrinkage * segment_mult + (1 - shrinkage) * hotel_mult
        
        return segment_mult
    
    def get_weekly_multipliers(
        self,
        hotel_id: Optional[int] = None,
        hotel_features: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Get all 7 DOW multipliers for a hotel.
        
        Returns:
            Dict like {'Monday': 0.92, 'Tuesday': 0.90, ..., 'Saturday': 1.15}
        """
        return {
            DOW_NAMES[i]: self.get_multiplier(i, hotel_id, hotel_features)
            for i in range(7)
        }
    
    def apply_to_baseline(
        self,
        baseline_price: float,
        hotel_id: Optional[int] = None,
        hotel_features: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Apply DOW adjustments to get daily prices.
        
        Args:
            baseline_price: Weekly average baseline price (€)
            hotel_id: Optional hotel ID
            hotel_features: Hotel feature dict
        
        Returns:
            Dict with prices for each day: {'Monday': 92.0, ..., 'Saturday': 115.0}
        """
        multipliers = self.get_weekly_multipliers(hotel_id, hotel_features)
        return {
            day: baseline_price * mult
            for day, mult in multipliers.items()
        }
    
    def print_patterns(self) -> None:
        """Print all learned patterns."""
        if not self.is_fitted:
            print("Model not fitted.")
            return
        
        print("\n" + "=" * 60)
        print("DAY-OF-WEEK PRICING PATTERNS")
        print("=" * 60)
        
        print("\nMARKET-LEVEL:")
        print(self.market_pattern)
        
        print("\nSEGMENT-LEVEL:")
        for segment, pattern in self.segment_patterns.items():
            print(f"\n{pattern}")
        
        if self.hotel_patterns:
            print(f"\nHOTEL-SPECIFIC: {len(self.hotel_patterns)} hotels with patterns")


def fit_dow_model_from_bookings(con) -> DOWAdjustmentModel:
    """
    Fit DOW model directly from booking data.
    
    This extracts booking-level data with arrival dates to learn
    true DOW patterns (not aggregated weekly data).
    
    Args:
        con: Database connection
    
    Returns:
        Fitted DOWAdjustmentModel
    """
    # Query booking-level data with DOW
    query = """
    WITH booking_adr AS (
        SELECT 
            b.hotel_id,
            b.arrival_date,
            EXTRACT(DOW FROM b.arrival_date) as arrival_dow,
            br.total_price / NULLIF(
                DATEDIFF('day', b.arrival_date, b.departure_date), 0
            ) as adr,
            hl.latitude,
            hl.longitude
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('Booked', 'confirmed')
          AND b.arrival_date >= '2023-06-01'
          AND b.arrival_date < '2024-10-01'
          AND br.total_price > 0
          AND DATEDIFF('day', b.arrival_date, b.departure_date) > 0
    )
    SELECT 
        hotel_id,
        arrival_date,
        arrival_dow,
        adr,
        latitude,
        longitude
    FROM booking_adr
    WHERE adr >= 20 AND adr <= 500
    """
    
    df = con.execute(query).fetchdf()
    
    if len(df) == 0:
        raise ValueError("No booking data found")
    
    print(f"Loaded {len(df):,} bookings for DOW analysis")
    
    # Add segment features
    # Madrid coordinates
    MADRID_LAT, MADRID_LON = 40.4168, -3.7038
    
    # Calculate distance to Madrid
    from src.features.engineering import haversine_distance
    
    df['dist_madrid'] = haversine_distance(
        df['latitude'].values,
        df['longitude'].values,
        MADRID_LAT,
        MADRID_LON
    )
    
    # Segment assignment (simplified - coastal needs coast distance)
    df['is_madrid_metro'] = (df['dist_madrid'] <= 50).astype(int)
    df['is_coastal'] = 0  # Would need coast distance data
    
    # Convert DOW (DuckDB uses 0=Sunday, we want 0=Monday)
    df['arrival_dow'] = (df['arrival_dow'] - 1) % 7
    
    # Fit model
    model = DOWAdjustmentModel()
    model.fit(df)
    
    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    from lib.db import init_db
    from lib.data_validator import CleaningConfig, DataCleaner
    
    # Get clean connection
    config = CleaningConfig(
        remove_negative_prices=True,
        remove_zero_prices=True,
        remove_low_prices=True,
    )
    cleaner = DataCleaner(config)
    con = cleaner.clean(init_db())
    
    # Fit DOW model
    model = fit_dow_model_from_bookings(con)
    
    # Print patterns
    model.print_patterns()
    
    # Example: Apply to cold-start coastal hotel
    print("\n" + "=" * 60)
    print("EXAMPLE: Cold-start coastal hotel, baseline €100/night")
    print("=" * 60)
    
    daily_prices = model.apply_to_baseline(
        baseline_price=100,
        hotel_features={'is_coastal': 1, 'is_madrid_metro': 0}
    )
    
    for day, price in daily_prices.items():
        mult = model.get_multiplier(
            DOW_MAP[day],
            hotel_features={'is_coastal': 1}
        )
        pct_change = (mult - 1) * 100
        print(f"  {day:10s}: €{price:6.2f} ({pct_change:+.0f}%)")
    
    # Example: Madrid business hotel
    print("\n" + "=" * 60)
    print("EXAMPLE: Cold-start Madrid hotel, baseline €120/night")
    print("=" * 60)
    
    daily_prices = model.apply_to_baseline(
        baseline_price=120,
        hotel_features={'is_coastal': 0, 'is_madrid_metro': 1}
    )
    
    for day, price in daily_prices.items():
        mult = model.get_multiplier(
            DOW_MAP[day],
            hotel_features={'is_madrid_metro': 1}
        )
        pct_change = (mult - 1) * 100
        print(f"  {day:10s}: €{price:6.2f} ({pct_change:+.0f}%)")

