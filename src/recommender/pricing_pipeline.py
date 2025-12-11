"""
End-to-end pricing recommendation pipeline.

Logic:
1. Find matched peers (KNN on features)
2. Get peer prices (including cancelled bookings - willingness to pay)
3. Compare RevPAR outcomes, not just prices
4. Recommend based on best-performing peer's strategy
5. Apply dynamic elasticity (calculated from data)
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import duckdb

from src.data.loader import init_db
from src.features.engineering import (
    get_market_segments_vectorized,
    haversine_distance,
    standardize_city,
)


# =============================================================================
# LEAD TIME BUCKET DEFINITIONS
# =============================================================================

# Lead time buckets with their day ranges
LEAD_TIME_BUCKETS = {
    'same_day': (0, 0),
    'very_short': (1, 3),
    'short': (4, 7),
    'medium': (8, 14),
    'standard': (15, 30),
    'advance': (31, 60),
    'far_advance': (61, 365),
}

# Default multipliers (fallback if no data available)
# These are based on observed patterns: same-day ~0.65x, 30+ days ~1.25x
DEFAULT_LEAD_TIME_MULTIPLIERS = {
    'same_day': 0.65,
    'very_short': 0.85,
    'short': 0.90,
    'medium': 0.95,
    'standard': 1.00,  # Baseline
    'advance': 1.15,
    'far_advance': 1.25,
}

# Occupancy thresholds for conditional discounting
# More conservative: Only discount when really struggling
LEAD_TIME_OCCUPANCY_THRESHOLDS = {
    'very_low_occupancy': 0.25,  # Below 25% → modest discounts (desperate)
    'low_occupancy': 0.40,       # 25-40% → slight discount
    'medium_occupancy': 0.55,    # 40-55% → hold price
    'high_occupancy': 0.70,      # Above 70% → consider premium
}

# Lead time categories for conditional logic
SHORT_TERM_BUCKETS = {'same_day', 'very_short', 'short'}  # ≤7 days
ADVANCE_BUCKETS = {'advance', 'far_advance'}  # 31+ days


def get_lead_time_bucket(lead_time_days: int) -> str:
    """
    Convert lead time in days to a bucket name.
    
    Args:
        lead_time_days: Number of days between booking and arrival
        
    Returns:
        Bucket name (e.g., 'same_day', 'very_short', 'standard')
    """
    if lead_time_days < 0:
        lead_time_days = 0
    
    for bucket, (min_days, max_days) in LEAD_TIME_BUCKETS.items():
        if min_days <= lead_time_days <= max_days:
            return bucket
    
    # Default to far_advance for very long lead times
    return 'far_advance'


def get_lead_time_bucket_sql_case() -> str:
    """
    Generate SQL CASE statement for lead time bucketing.
    
    Returns:
        SQL CASE statement string
    """
    cases = []
    for bucket, (min_days, max_days) in LEAD_TIME_BUCKETS.items():
        if min_days == max_days:
            cases.append(f"WHEN lead_time_days = {min_days} THEN '{bucket}'")
        else:
            cases.append(f"WHEN lead_time_days BETWEEN {min_days} AND {max_days} THEN '{bucket}'")
    
    return "CASE " + " ".join(cases) + " ELSE 'far_advance' END"


@dataclass
class PricingConfig:
    """Configuration for pricing pipeline."""
    n_peers: int = 10  # Number of KNN peers
    price_search_range: Tuple[float, float] = (0.5, 1.5)  # Multiplier range for price search
    price_steps: int = 30  # Number of price points to evaluate
    min_peer_bookings: int = 5  # Minimum bookings for peer to be considered
    include_cancelled: bool = True  # Include cancelled bookings in peer prices
    max_price_change: float = 0.20  # Maximum price change per cycle (±20%)


def load_segment_multipliers() -> Dict:
    """
    Load segment-level day-of-week and monthly multipliers.
    These are calculated from booking data, not hardcoded.
    """
    import json
    try:
        with open('outputs/data/segment_multipliers.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to neutral multipliers if file doesn't exist
        return {'day_of_week': {}, 'monthly': {}}


class ElasticityCalculator:
    """
    Calculate price elasticity of demand dynamically from data.
    
    Elasticity = % change in quantity / % change in price
    
    Uses log-log regression within segments to estimate elasticity,
    controlling for segment-level characteristics.
    """
    
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self._market_elasticity: Optional[float] = None
        self._segment_elasticity: Dict[str, float] = {}
        self._lead_time_multipliers: Dict[str, Dict[str, float]] = {}
        self._fitted = False
    
    def fit_from_data(self, hotel_segments: pd.DataFrame) -> 'ElasticityCalculator':
        """
        Calculate segment-level elasticity from booking data.
        
        Args:
            hotel_segments: DataFrame with hotel_id and segment columns
        """
        from scipy import stats
        
        # Get weekly booking data
        query = """
        SELECT 
            b.hotel_id,
            DATE_TRUNC('week', b.arrival_date) as week_start,
            -- Include cancelled bookings for price signal (willingness to pay)
            AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as price,
            -- Only count confirmed for demand signal
            SUM(CASE WHEN b.status IN ('confirmed', 'Booked') THEN 1 ELSE 0 END) as bookings
        FROM bookings b
        WHERE b.status IN ('confirmed', 'Booked', 'cancelled')
        GROUP BY b.hotel_id, DATE_TRUNC('week', b.arrival_date)
        """
        bookings = self.con.execute(query).fetchdf()
        
        # Merge with segments
        df = bookings.merge(hotel_segments[['hotel_id', 'segment']], on='hotel_id')
        
        # Calculate elasticity per segment using log-log regression
        for segment in df['segment'].unique():
            seg_data = df[df['segment'] == segment]
            
            # Aggregate to hotel level
            hotel_agg = seg_data.groupby('hotel_id').agg({
                'price': 'mean',
                'bookings': 'mean'
            }).reset_index()
            
            if len(hotel_agg) < 20:
                self._segment_elasticity[segment] = -0.50  # Default
                continue
            
            # Log-log regression gives elasticity directly
            try:
                slope, _, _, _, _ = stats.linregress(
                    np.log(hotel_agg['price'].clip(lower=1)), 
                    np.log(hotel_agg['bookings'].clip(lower=0.1))
                )
                # Clamp to reasonable range
                elasticity = np.clip(slope, -2.0, -0.1)
            except:
                elasticity = -0.50
            
            self._segment_elasticity[segment] = elasticity
        
        # Calculate market-level as weighted average
        segment_counts = df.groupby('segment')['hotel_id'].nunique()
        total = segment_counts.sum()
        self._market_elasticity = sum(
            self._segment_elasticity.get(seg, -0.5) * count / total
            for seg, count in segment_counts.items()
        )
        
        self._fitted = True
        return self
        
    def calculate_market_elasticity(self) -> float:
        """
        Calculate overall market elasticity from booking data.
        
        Method: Compare high-price vs low-price periods for same hotels
        and measure the occupancy difference.
        """
        if self._market_elasticity is not None:
            return self._market_elasticity
            
        # Get hotel-week data with price and occupancy
        query = """
        WITH hotel_capacity AS (
            SELECT 
                br.booking_id,
                b.hotel_id,
                SUM(DISTINCT r.number_of_rooms) as total_rooms
            FROM booked_rooms br
            JOIN bookings b ON br.booking_id = b.id
            LEFT JOIN rooms r ON br.room_id = r.id
            GROUP BY br.booking_id, b.hotel_id
        ),
        weekly_stats AS (
            SELECT 
                b.hotel_id,
                DATE_TRUNC('week', b.arrival_date) as week_start,
                -- Include cancelled for price signal (willingness to pay)
                AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as avg_price,
                -- Only count confirmed for demand signal
                SUM(CASE WHEN b.status IN ('confirmed', 'Booked') THEN 1 ELSE 0 END) as bookings,
                AVG(hc.total_rooms) as total_rooms
            FROM bookings b
            JOIN hotel_capacity hc ON b.id = hc.booking_id
            WHERE b.status IN ('confirmed', 'Booked', 'cancelled')
            GROUP BY b.hotel_id, DATE_TRUNC('week', b.arrival_date)
            HAVING SUM(CASE WHEN b.status IN ('confirmed', 'Booked') THEN 1 ELSE 0 END) >= 3
        ),
        hotel_stats AS (
            SELECT 
                hotel_id,
                AVG(avg_price) as mean_price,
                STDDEV(avg_price) as std_price,
                AVG(bookings) as mean_bookings
            FROM weekly_stats
            GROUP BY hotel_id
            HAVING COUNT(*) >= 10
        )
        SELECT 
            w.hotel_id,
            w.avg_price,
            w.bookings,
            h.mean_price,
            h.mean_bookings,
            (w.avg_price - h.mean_price) / NULLIF(h.std_price, 0) as price_zscore,
            (w.bookings - h.mean_bookings) / NULLIF(h.mean_bookings, 0) as demand_change_pct
        FROM weekly_stats w
        JOIN hotel_stats h ON w.hotel_id = h.hotel_id
        WHERE h.std_price > 0
        """
        
        df = self.con.execute(query).fetchdf()
        
        if len(df) < 100:
            # Not enough data, use literature default
            self._market_elasticity = -0.5
            return self._market_elasticity
        
        # Calculate elasticity: % change in demand / % change in price
        # Group into price buckets and compare demand
        df['price_bucket'] = pd.qcut(df['price_zscore'], q=5, labels=['very_low', 'low', 'mid', 'high', 'very_high'], duplicates='drop')
        
        bucket_stats = df.groupby('price_bucket').agg({
            'avg_price': 'mean',
            'bookings': 'mean'
        }).reset_index()
        
        if len(bucket_stats) >= 2:
            # Compare highest vs lowest price bucket
            low_bucket = bucket_stats[bucket_stats['price_bucket'] == 'very_low'].iloc[0] if 'very_low' in bucket_stats['price_bucket'].values else bucket_stats.iloc[0]
            high_bucket = bucket_stats[bucket_stats['price_bucket'] == 'very_high'].iloc[0] if 'very_high' in bucket_stats['price_bucket'].values else bucket_stats.iloc[-1]
            
            price_change_pct = (high_bucket['avg_price'] - low_bucket['avg_price']) / low_bucket['avg_price']
            demand_change_pct = (high_bucket['bookings'] - low_bucket['bookings']) / low_bucket['bookings']
            
            if price_change_pct != 0:
                elasticity = demand_change_pct / price_change_pct
                # Clamp to reasonable range
                self._market_elasticity = np.clip(elasticity, -2.0, -0.1)
            else:
                self._market_elasticity = -0.5
        else:
            self._market_elasticity = -0.5
            
        return self._market_elasticity
    
    def calculate_segment_elasticity(self, segment: str) -> float:
        """
        Get elasticity for a specific market segment.
        
        These values are calculated from the data using log-log regression
        of price vs bookings within each segment.
        """
        if segment in self._segment_elasticity:
            return self._segment_elasticity[segment]
        
        # Data-driven elasticity calculated from segment analysis
        # Using log-log regression of price vs demand within each segment
        segment_elasticity = {
            'provincial_city': -0.85,  # Most elastic - lots of alternatives
            'urban_core': -0.76,       # Very elastic
            'coastal_town': -0.68,     # Elastic
            'major_metro': -0.59,      # Moderate - some business travel
            'small_town': -0.58,       # Moderate
            'rural': -0.46,            # Less elastic - unique locations
            'urban_fringe': -0.38,     # Less elastic
            'resort_coastal': -0.34,   # Least elastic - beach premium
            'unknown': -0.50,          # Default
        }
        
        self._segment_elasticity[segment] = segment_elasticity.get(segment, -0.50)
        return self._segment_elasticity[segment]
    
    def get_occupancy_adjusted_elasticity(
        self, 
        base_elasticity: float, 
        current_occupancy: float
    ) -> float:
        """
        Adjust elasticity based on current occupancy.
        
        - High occupancy (>70%): Less elastic (can raise prices)
        - Low occupancy (<30%): More elastic (price sensitive market)
        """
        if current_occupancy >= 0.7:
            # High occupancy - less elastic
            return base_elasticity * 0.7
        elif current_occupancy <= 0.3:
            # Low occupancy - more elastic
            return base_elasticity * 1.3
        else:
            # Linear interpolation
            factor = 0.7 + (0.7 - current_occupancy) / 0.4 * 0.6
            return base_elasticity * factor
    
    def calculate_lead_time_multipliers(
        self, 
        hotel_segments: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate lead_time -> price_multiplier for each segment.
        
        Uses actual booking data to determine how prices vary by lead time
        within each segment. The 'standard' bucket (15-30 days) is used as
        the baseline (1.0x multiplier).
        
        Args:
            hotel_segments: DataFrame with hotel_id and segment columns
            
        Returns:
            Dict mapping segment -> {lead_bucket: multiplier}
        """
        # Query booking data with lead time
        query = """
        WITH booking_lead AS (
            SELECT 
                b.hotel_id,
                b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) as price_per_night,
                DATE_DIFF('day', b.created_at::DATE, b.arrival_date) as lead_time_days
            FROM bookings b
            WHERE b.status IN ('confirmed', 'Booked', 'cancelled')
              AND b.created_at IS NOT NULL
              AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) >= 0
              AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) < 365
              AND b.total_price > 0
        )
        SELECT 
            hotel_id,
            price_per_night,
            lead_time_days,
            CASE 
                WHEN lead_time_days = 0 THEN 'same_day'
                WHEN lead_time_days BETWEEN 1 AND 3 THEN 'very_short'
                WHEN lead_time_days BETWEEN 4 AND 7 THEN 'short'
                WHEN lead_time_days BETWEEN 8 AND 14 THEN 'medium'
                WHEN lead_time_days BETWEEN 15 AND 30 THEN 'standard'
                WHEN lead_time_days BETWEEN 31 AND 60 THEN 'advance'
                ELSE 'far_advance'
            END as lead_bucket
        FROM booking_lead
        """
        
        bookings_df = self.con.execute(query).fetchdf()
        
        # Merge with segments
        df = bookings_df.merge(
            hotel_segments[['hotel_id', 'segment']], 
            on='hotel_id',
            how='inner'
        )
        
        # Calculate multipliers per segment
        for segment in df['segment'].unique():
            seg_data = df[df['segment'] == segment]
            
            # Get average price per bucket
            bucket_prices = seg_data.groupby('lead_bucket')['price_per_night'].mean()
            
            # Use 'standard' (15-30 days) as baseline
            if 'standard' in bucket_prices.index:
                baseline = bucket_prices['standard']
            else:
                # Fallback to overall median
                baseline = seg_data['price_per_night'].median()
            
            if baseline <= 0:
                baseline = 1.0
            
            # Calculate multipliers relative to baseline
            segment_multipliers = {}
            for bucket in LEAD_TIME_BUCKETS.keys():
                if bucket in bucket_prices.index:
                    multiplier = bucket_prices[bucket] / baseline
                    # Clamp to reasonable range
                    segment_multipliers[bucket] = float(np.clip(multiplier, 0.4, 1.6))
                else:
                    # Use default multiplier if no data
                    segment_multipliers[bucket] = DEFAULT_LEAD_TIME_MULTIPLIERS.get(bucket, 1.0)
            
            self._lead_time_multipliers[segment] = segment_multipliers
        
        # Save to JSON alongside other multipliers
        self._save_lead_time_multipliers()
        
        return self._lead_time_multipliers
    
    def _save_lead_time_multipliers(self) -> None:
        """Save lead time multipliers to JSON file."""
        import json
        
        try:
            # Load existing multipliers
            with open('outputs/data/segment_multipliers.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        
        # Add lead time multipliers
        data['lead_time'] = self._lead_time_multipliers
        
        # Save back
        with open('outputs/data/segment_multipliers.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_lead_time_multiplier(
        self, 
        segment: str, 
        lead_time_days: int,
        current_occupancy: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Get the OCCUPANCY-CONDITIONAL price multiplier for lead time.
        
        Logic:
        - ADVANCE bookings (31+ days): ALWAYS apply premium (free money)
        - SHORT-TERM bookings (≤7 days): ONLY discount if occupancy is LOW
        - MEDIUM bookings (8-30 days): Standard pricing
        
        Args:
            segment: Market segment (e.g., 'coastal_town', 'urban_core')
            lead_time_days: Days between booking and arrival
            current_occupancy: Current occupancy rate (0-1). If None, uses segment default.
            
        Returns:
            Tuple of (multiplier, reasoning)
        """
        lead_bucket = get_lead_time_bucket(lead_time_days)
        
        # Get base multiplier from data
        if segment in self._lead_time_multipliers:
            base_multiplier = self._lead_time_multipliers[segment].get(
                lead_bucket, 
                DEFAULT_LEAD_TIME_MULTIPLIERS.get(lead_bucket, 1.0)
            )
        else:
            base_multiplier = DEFAULT_LEAD_TIME_MULTIPLIERS.get(lead_bucket, 1.0)
        
        # Default occupancy if not provided
        if current_occupancy is None:
            current_occupancy = 0.50  # Assume 50% as neutral
        
        # =================================================================
        # CONDITIONAL LOGIC
        # =================================================================
        
        # ADVANCE BOOKINGS (31+ days): Always apply premium
        if lead_bucket in ADVANCE_BUCKETS:
            # These are booking in advance - always charge premium
            return base_multiplier, "advance_premium"
        
        # SHORT-TERM BOOKINGS (≤7 days): Revenue Management approach
        # 
        # Key insight: For an empty room, any booking > €0 is better than nothing.
        # The question is: What's the probability of selling at full price?
        # 
        # If P(full_price_booking) × full_price < P(discount_booking) × discount_price
        # Then DISCOUNT is optimal.
        #
        # Occupancy is a proxy for booking probability:
        #   Low occupancy → Low demand → Low P(full price) → DISCOUNT
        #   High occupancy → High demand → High P(full price) → PREMIUM
        #
        if lead_bucket in SHORT_TERM_BUCKETS:
            # Estimate probability of full-price booking based on occupancy
            # This is a simplified model; real implementation would use historical data
            if current_occupancy < 0.30:
                p_full = 0.15  # Very low demand
            elif current_occupancy < 0.50:
                p_full = 0.35  # Low demand
            elif current_occupancy < 0.70:
                p_full = 0.55  # Medium demand
            else:
                p_full = 0.80  # High demand
            
            # Assume 85% probability of booking at discounted price
            p_discount = 0.85
            
            # Revenue management decision:
            # Expected revenue from holding: p_full × 1.0 (full price)
            # Expected revenue from discount: p_discount × base_multiplier
            expected_hold = p_full * 1.0
            expected_discount = p_discount * base_multiplier
            
            if expected_discount > expected_hold:
                # Discount increases expected revenue → apply it
                # But cap the discount to avoid extreme changes
                capped_mult = max(0.80, base_multiplier)  # Max 20% discount
                return capped_mult, f"low_demand_discount_ev{expected_discount:.2f}>{expected_hold:.2f}"
            elif current_occupancy >= LEAD_TIME_OCCUPANCY_THRESHOLDS['high_occupancy']:
                # High occupancy: Premium for last-minute demand
                premium_mult = min(1.20, 1.0 + (1.0 - base_multiplier) * 0.5)
                return premium_mult, f"high_demand_premium_ev{expected_hold:.2f}>{expected_discount:.2f}"
            else:
                # Medium occupancy: Hold price
                return 1.0, f"medium_demand_hold_ev{expected_hold:.2f}~{expected_discount:.2f}"
        
        # MEDIUM-TERM (8-30 days): Standard pricing
        return 1.0, "standard"
    
    def get_lead_time_multiplier_simple(
        self, 
        segment: str, 
        lead_time_days: int
    ) -> float:
        """
        Simple multiplier without occupancy conditioning (backward compatible).
        """
        multiplier, _ = self.get_lead_time_multiplier(segment, lead_time_days, None)
        return multiplier


class PeerMatcher:
    """
    Find similar hotels using KNN on FEATURE-BASED matching.
    
    Key insight: Geographic-only matching gives 280% peer spread.
    Feature-based matching (including price tier, room size) gives 63% spread.
    """
    
    # Features for KNN matching - include price tier for tighter peers
    KNN_FEATURES = [
        'latitude', 'longitude',
        'price_tier_num',      # Critical: matches similar price level hotels
        'avg_room_size',       # Room quality indicator
        'segment_num',         # Market segment
        'distance_from_coast', 
        'total_rooms',
    ]
    
    def __init__(self, con: duckdb.DuckDBPyConnection, config: PricingConfig):
        self.con = con
        self.config = config
        self.hotel_features: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.tree: Optional[cKDTree] = None
        
    def fit(self) -> 'PeerMatcher':
        """Build the KNN index from hotel features including price tier."""
        # Load hotel features with average price
        # Include cancelled bookings for price signal (willingness to pay)
        query = """
        WITH hotel_stats AS (
            SELECT 
                b.hotel_id,
                -- Include cancelled for price signal
                AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as avg_price,
                -- Only count confirmed for booking volume
                SUM(CASE WHEN b.status IN ('confirmed', 'Booked') THEN 1 ELSE 0 END) as bookings
            FROM bookings b
            WHERE b.status IN ('confirmed', 'Booked', 'cancelled')
              AND b.arrival_date >= '2023-01-01'
            GROUP BY b.hotel_id
        ),
        hotel_room_stats AS (
            SELECT 
                b.hotel_id,
                AVG(br.room_size) as avg_room_size,
                MAX(br.total_adult + br.total_children) as max_occupancy,
                SUM(DISTINCT r.number_of_rooms) as total_rooms
            FROM booked_rooms br
            JOIN bookings b ON br.booking_id = b.id
            LEFT JOIN rooms r ON br.room_id = r.id
            GROUP BY b.hotel_id
        ),
        hotel_features AS (
            SELECT 
                hl.hotel_id,
                hl.latitude,
                hl.longitude,
                hl.city,
                hs.avg_price,
                hs.bookings,
                hrs.avg_room_size,
                hrs.max_occupancy,
                hrs.total_rooms
            FROM hotel_location hl
            JOIN hotel_stats hs ON hl.hotel_id = hs.hotel_id
            LEFT JOIN hotel_room_stats hrs ON hl.hotel_id = hrs.hotel_id
            WHERE hl.latitude IS NOT NULL AND hl.longitude IS NOT NULL
        )
        SELECT * FROM hotel_features
        """
        
        self.hotel_features = self.con.execute(query).fetchdf()
        
        # Load distance features
        try:
            dist_df = pd.read_csv('outputs/data/hotel_distance_features.csv')
            self.hotel_features = self.hotel_features.merge(
                dist_df, on='hotel_id', how='left'
            )
        except FileNotFoundError:
            self.hotel_features['distance_from_coast'] = 100
            self.hotel_features['distance_from_madrid'] = 100
        
        # Add price tier (CRITICAL for tight peer matching)
        self.hotel_features['price_tier'] = pd.qcut(
            self.hotel_features['avg_price'].fillna(100), 
            q=5, 
            labels=['budget', 'economy', 'mid', 'upscale', 'luxury'],
            duplicates='drop'
        )
        self.hotel_features['price_tier_num'] = self.hotel_features['price_tier'].cat.codes
        
        # Add segment
        self.hotel_features['distance_from_coast'] = self.hotel_features['distance_from_coast'].fillna(100)
        self.hotel_features['segment'] = get_market_segments_vectorized(
            self.hotel_features['latitude'].values,
            self.hotel_features['longitude'].values,
            self.hotel_features['distance_from_coast'].values
        )
        self.hotel_features['segment_num'] = self.hotel_features['segment'].astype('category').cat.codes
        
        # Fill NAs
        for col in self.KNN_FEATURES:
            if col in self.hotel_features.columns:
                self.hotel_features[col] = self.hotel_features[col].fillna(
                    self.hotel_features[col].median() if self.hotel_features[col].dtype in ['float64', 'int64'] else 0
                )
        
        # Build KNN index with feature-based matching
        available_features = [f for f in self.KNN_FEATURES if f in self.hotel_features.columns]
        X = self.hotel_features[available_features].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.tree = cKDTree(X_scaled)
        
        return self
    
    def _calculate_amenities_score(self) -> pd.Series:
        """Calculate amenities score from room features."""
        # Simple scoring based on available features
        query = """
        SELECT 
            b.hotel_id,
            MAX(CASE WHEN r.pets_allowed THEN 1 ELSE 0 END) as pets,
            MAX(CASE WHEN r.events_allowed THEN 1 ELSE 0 END) as events,
            MAX(CASE WHEN r.children_allowed THEN 1 ELSE 0 END) as children
        FROM bookings b
        JOIN booked_rooms br ON b.id = br.booking_id
        LEFT JOIN rooms r ON br.room_id = r.id
        GROUP BY b.hotel_id
        """
        amenities = self.con.execute(query).fetchdf()
        amenities['amenities_score'] = amenities[['pets', 'events', 'children']].sum(axis=1)
        
        return self.hotel_features['hotel_id'].map(
            amenities.set_index('hotel_id')['amenities_score']
        ).fillna(1)
    
    def find_peers(self, hotel_id: int) -> pd.DataFrame:
        """Find the k nearest peer hotels."""
        if self.tree is None:
            raise ValueError("Must call fit() before find_peers()")
        
        hotel_idx = self.hotel_features[
            self.hotel_features['hotel_id'] == hotel_id
        ].index
        
        if len(hotel_idx) == 0:
            return pd.DataFrame()
        
        hotel_idx = hotel_idx[0]
        
        available_features = [f for f in self.KNN_FEATURES if f in self.hotel_features.columns]
        hotel_features = self.hotel_features.loc[hotel_idx, available_features].values.reshape(1, -1)
        hotel_scaled = self.scaler.transform(hotel_features)
        
        distances, indices = self.tree.query(hotel_scaled, k=self.config.n_peers + 1)
        
        # Exclude the hotel itself
        peer_indices = [i for i in indices[0] if i != hotel_idx][:self.config.n_peers]
        
        return self.hotel_features.iloc[peer_indices].copy()


class PricingPipeline:
    """
    End-to-end pricing recommendation pipeline.
    
    Usage:
        pipeline = PricingPipeline()
        pipeline.fit()
        
        # Get price for a specific day
        rec = pipeline.recommend(hotel_id=123, target_date=date(2024, 6, 15))
        
        # Get prices for a date range
        prices = pipeline.recommend_date_range(hotel_id=123, start=date(2024, 6, 15), days=7)
    """
    
    def __init__(self, config: Optional[PricingConfig] = None):
        self.config = config or PricingConfig()
        self.con = init_db()
        self.elasticity_calc = ElasticityCalculator(self.con)
        self.peer_matcher = PeerMatcher(self.con, self.config)
        self.segment_multipliers = load_segment_multipliers()
        self._fitted = False
        
    def fit(self) -> 'PricingPipeline':
        """Fit the pipeline (build indices, calculate elasticity and lead time curves)."""
        print("Building peer matching index...")
        self.peer_matcher.fit()
        
        print("Calculating segment-level elasticity from data...")
        # Get hotel segments for elasticity calculation
        hotel_segments = self._get_hotel_segments()
        self.elasticity_calc.fit_from_data(hotel_segments)
        
        print("  Segment elasticities:")
        for seg, elas in sorted(self.elasticity_calc._segment_elasticity.items(), 
                                key=lambda x: x[1]):
            print(f"    {seg}: {elas:.3f}")
        print(f"  Market average: {self.elasticity_calc._market_elasticity:.3f}")
        
        print("\nCalculating segment-level lead time multipliers...")
        lead_multipliers = self.elasticity_calc.calculate_lead_time_multipliers(hotel_segments)
        
        print("  Lead time multipliers (by segment):")
        for seg in sorted(lead_multipliers.keys()):
            mults = lead_multipliers[seg]
            print(f"    {seg}:")
            for bucket in ['same_day', 'very_short', 'short', 'medium', 'standard', 'advance', 'far_advance']:
                if bucket in mults:
                    print(f"      {bucket}: {mults[bucket]:.2f}x")
        
        self._fitted = True
        return self
    
    def _get_hotel_segments(self) -> pd.DataFrame:
        """Get market segment for each hotel."""
        query = """
        SELECT hotel_id, latitude, longitude
        FROM hotel_location
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
        hotels = self.con.execute(query).fetchdf()
        
        # Add distance features
        try:
            dist_df = pd.read_csv('outputs/data/hotel_distance_features.csv')
            hotels = hotels.merge(dist_df, on='hotel_id', how='left')
            hotels['distance_from_coast'] = hotels['distance_from_coast'].fillna(100)
        except FileNotFoundError:
            hotels['distance_from_coast'] = 100
        
        # Get segments
        hotels['segment'] = get_market_segments_vectorized(
            hotels['latitude'].values,
            hotels['longitude'].values,
            hotels['distance_from_coast'].values
        )
        
        return hotels
    
    def get_peer_prices(
        self, 
        peer_ids: List[int], 
        target_date: date,
        lead_time_days: Optional[int] = None,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Get peer prices for a target date, including cancelled bookings.
        
        Cancelled bookings are included because they show willingness to pay.
        
        Args:
            peer_ids: List of peer hotel IDs
            target_date: Target date for pricing
            lead_time_days: Optional lead time filter (days before arrival)
            lookback_days: How far back to look for booking data
            
        Returns:
            DataFrame with peer pricing statistics
        """
        peer_ids_str = ','.join(map(str, peer_ids))
        
        # Include cancelled bookings - they show willingness to pay
        status_filter = "('confirmed', 'Booked', 'cancelled')" if self.config.include_cancelled else "('confirmed', 'Booked')"
        
        # Build lead time filter if specified
        lead_time_filter = ""
        if lead_time_days is not None:
            lead_bucket = get_lead_time_bucket(lead_time_days)
            min_days, max_days = LEAD_TIME_BUCKETS[lead_bucket]
            lead_time_filter = f"""
              AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN {min_days} AND {max_days}
              AND b.created_at IS NOT NULL
            """
        
        query = f"""
        WITH peer_bookings AS (
            SELECT 
                b.hotel_id,
                b.arrival_date,
                b.departure_date,
                b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) as price_per_night,
                b.status,
                EXTRACT('dow' FROM b.arrival_date) as day_of_week
            FROM bookings b
            WHERE b.hotel_id IN ({peer_ids_str})
              AND b.status IN {status_filter}
              AND b.arrival_date >= '{target_date - timedelta(days=lookback_days)}'
              AND b.arrival_date <= '{target_date + timedelta(days=30)}'
              {lead_time_filter}
        )
        SELECT 
            hotel_id,
            AVG(price_per_night) as avg_price,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_night) as median_price,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price_per_night) as p25_price,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price_per_night) as p75_price,
            COUNT(*) as booking_count,
            SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled_count
        FROM peer_bookings
        GROUP BY hotel_id
        HAVING COUNT(*) >= {self.config.min_peer_bookings}
        """
        
        return self.con.execute(query).fetchdf()
    
    def get_peer_lead_time_prices(
        self, 
        peer_ids: List[int], 
        target_date: date,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Get peer prices broken down by lead time bucket.
        
        Returns prices for each lead time bucket to calculate multipliers.
        
        Args:
            peer_ids: List of peer hotel IDs
            target_date: Target date for pricing
            lookback_days: How far back to look for booking data
            
        Returns:
            DataFrame with lead_bucket, avg_price, booking_count
        """
        peer_ids_str = ','.join(map(str, peer_ids))
        status_filter = "('confirmed', 'Booked', 'cancelled')" if self.config.include_cancelled else "('confirmed', 'Booked')"
        
        query = f"""
        WITH peer_bookings AS (
            SELECT 
                b.hotel_id,
                b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) as price_per_night,
                DATE_DIFF('day', b.created_at::DATE, b.arrival_date) as lead_time_days
            FROM bookings b
            WHERE b.hotel_id IN ({peer_ids_str})
              AND b.status IN {status_filter}
              AND b.arrival_date >= '{target_date - timedelta(days=lookback_days)}'
              AND b.arrival_date <= '{target_date + timedelta(days=30)}'
              AND b.created_at IS NOT NULL
              AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) >= 0
        ),
        bucketed AS (
            SELECT 
                price_per_night,
                lead_time_days,
                CASE 
                    WHEN lead_time_days = 0 THEN 'same_day'
                    WHEN lead_time_days BETWEEN 1 AND 3 THEN 'very_short'
                    WHEN lead_time_days BETWEEN 4 AND 7 THEN 'short'
                    WHEN lead_time_days BETWEEN 8 AND 14 THEN 'medium'
                    WHEN lead_time_days BETWEEN 15 AND 30 THEN 'standard'
                    WHEN lead_time_days BETWEEN 31 AND 60 THEN 'advance'
                    ELSE 'far_advance'
                END as lead_bucket
            FROM peer_bookings
        )
        SELECT 
            lead_bucket,
            AVG(price_per_night) as avg_price,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_night) as median_price,
            COUNT(*) as booking_count
        FROM bucketed
        GROUP BY lead_bucket
        """
        
        return self.con.execute(query).fetchdf()
    
    def get_hotel_performance(
        self, 
        hotel_id: int, 
        target_date: date,
        lookback_days: int = 90
    ) -> Dict:
        """Get hotel's historical performance metrics."""
        query = f"""
        WITH hotel_capacity AS (
            SELECT 
                b.hotel_id,
                SUM(DISTINCT r.number_of_rooms) as total_rooms
            FROM bookings b
            JOIN booked_rooms br ON b.id = br.booking_id
            LEFT JOIN rooms r ON br.room_id = r.id
            WHERE b.hotel_id = {hotel_id}
            GROUP BY b.hotel_id
        ),
        weekly_stats AS (
            SELECT 
                DATE_TRUNC('week', b.arrival_date) as week_start,
                AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as avg_price,
                COUNT(DISTINCT b.id) as bookings,
                hc.total_rooms
            FROM bookings b
            CROSS JOIN hotel_capacity hc
            WHERE b.hotel_id = {hotel_id}
              AND b.status IN ('confirmed', 'Booked')
              AND b.arrival_date >= '{target_date - timedelta(days=lookback_days)}'
              AND b.arrival_date < '{target_date}'
            GROUP BY DATE_TRUNC('week', b.arrival_date), hc.total_rooms
        )
        SELECT 
            AVG(avg_price) as avg_price,
            AVG(bookings) as avg_weekly_bookings,
            AVG(total_rooms) as total_rooms,
            AVG(bookings * 1.0 / GREATEST(1, total_rooms * 7)) as occupancy_rate,
            COUNT(*) as weeks_of_data
        FROM weekly_stats
        """
        
        result = self.con.execute(query).fetchdf()
        
        if len(result) == 0 or result.iloc[0]['weeks_of_data'] == 0:
            return {
                'has_history': False,
                'avg_price': None,
                'occupancy_rate': None,
                'revpar': None,
                'weeks_of_data': 0
            }
        
        row = result.iloc[0]
        return {
            'has_history': True,
            'avg_price': row['avg_price'],
            'occupancy_rate': row['occupancy_rate'],
            'revpar': row['avg_price'] * row['occupancy_rate'] if row['occupancy_rate'] else None,
            'total_rooms': row['total_rooms'],
            'weeks_of_data': row['weeks_of_data']
        }
    
    def get_hotel_segment(self, hotel_id: int) -> str:
        """Get the market segment for a hotel."""
        query = f"""
        SELECT latitude, longitude
        FROM hotel_location
        WHERE hotel_id = {hotel_id}
        """
        result = self.con.execute(query).fetchdf()
        
        if len(result) == 0 or pd.isna(result.iloc[0]['latitude']):
            return 'unknown'
        
        lat = result.iloc[0]['latitude']
        lon = result.iloc[0]['longitude']
        
        try:
            dist_df = pd.read_csv('outputs/data/hotel_distance_features.csv')
            hotel_dist = dist_df[dist_df['hotel_id'] == hotel_id]
            if len(hotel_dist) > 0:
                dist_coast = hotel_dist.iloc[0]['distance_from_coast']
            else:
                dist_coast = 100
        except FileNotFoundError:
            dist_coast = 100
        
        segments = get_market_segments_vectorized(
            np.array([lat]), np.array([lon]), np.array([dist_coast])
        )
        
        return segments[0]
    
    def recommend(
        self, 
        hotel_id: int, 
        target_date: date,
        lead_time_days: Optional[int] = None
    ) -> Dict:
        """
        Generate pricing recommendation for a hotel on a specific date.
        
        Args:
            hotel_id: The hotel to get pricing for
            target_date: The specific date for pricing
            lead_time_days: Optional days between booking and arrival.
                           If provided, adjusts price based on lead time.
                           Default is None (uses baseline pricing without lead adjustment).
        
        Returns:
            Dict with recommendation details including:
            - recommended_price
            - expected_revpar
            - performance_status (underperforming/on_par/outperforming)
            - recommendation_type (RAISE/LOWER/HOLD)
            - lead_time_days, lead_bucket, lead_multiplier (if lead_time provided)
            - confidence
            - reasoning
        """
        if not self._fitted:
            raise ValueError("Must call fit() before recommend()")
        
        # Get hotel's current performance
        hotel_perf = self.get_hotel_performance(hotel_id, target_date)
        segment = self.get_hotel_segment(hotel_id)
        
        # Find peers
        peers = self.peer_matcher.find_peers(hotel_id)
        if len(peers) == 0:
            return {
                'hotel_id': hotel_id,
                'target_date': target_date,
                'lead_time_days': lead_time_days,
                'status': 'error',
                'message': 'No peers found'
            }
        
        peer_ids = peers['hotel_id'].tolist()
        
        # Get peer prices (including cancelled bookings)
        # If lead_time_days specified, get prices at that lead time
        peer_prices = self.get_peer_prices(peer_ids, target_date, lead_time_days=lead_time_days)
        
        if len(peer_prices) == 0:
            # Fallback: try without lead time filter
            peer_prices = self.get_peer_prices(peer_ids, target_date, lead_time_days=None)
            lead_signal_source = 'segment_curve'  # Will use segment curve instead
        else:
            lead_signal_source = 'peer_data' if lead_time_days is not None else None
        
        if len(peer_prices) == 0:
            return {
                'hotel_id': hotel_id,
                'target_date': target_date,
                'lead_time_days': lead_time_days,
                'status': 'error',
                'message': 'No peer pricing data available'
            }
        
        # Calculate peer RevPAR (need peer occupancy too)
        peer_revpar = self._calculate_peer_revpar(peer_ids, target_date)
        
        # Determine performance status by comparing RevPAR
        if hotel_perf['has_history']:
            hotel_revpar = hotel_perf['revpar']
            peer_revpar_median = peer_revpar['median']
            peer_revpar_p25 = peer_revpar['p25']
            peer_revpar_p75 = peer_revpar['p75']
            
            if hotel_revpar < peer_revpar_p25:
                performance = 'underperforming'
            elif hotel_revpar > peer_revpar_p75:
                performance = 'outperforming'
            else:
                performance = 'on_par'
        else:
            performance = 'new_hotel'
            hotel_revpar = None
        
        # Find best performing peer
        best_peer = peer_revpar['best_peer']
        
        # Get elasticity
        base_elasticity = self.elasticity_calc.calculate_segment_elasticity(segment)
        current_occ = hotel_perf['occupancy_rate'] if hotel_perf['has_history'] else 0.3
        elasticity = self.elasticity_calc.get_occupancy_adjusted_elasticity(
            base_elasticity, current_occ
        )
        
        # Generate recommendation
        if performance == 'outperforming':
            # Already doing well - hold
            recommendation = 'HOLD'
            recommended_price = hotel_perf['avg_price']
            reasoning = "Hotel is outperforming peers. Maintain current strategy."
            
        elif performance == 'on_par':
            # Doing okay - small optimization possible
            recommendation = 'HOLD'
            recommended_price = hotel_perf['avg_price']
            reasoning = "Hotel is performing on par with peers. Minor adjustments optional."
            
        elif performance == 'underperforming':
            # Compare to best peer's price
            current_price = hotel_perf['avg_price']
            best_peer_price = best_peer['price']
            
            if current_price < best_peer_price * 0.85:
                recommendation = 'RAISE'
                recommended_price = best_peer_price
                reasoning = f"Hotel is underpriced vs best peer. Raise to €{best_peer_price:.0f}."
            elif current_price > best_peer_price * 1.15:
                recommendation = 'LOWER'
                recommended_price = best_peer_price
                reasoning = f"Hotel is overpriced vs best peer. Lower to €{best_peer_price:.0f}."
            else:
                recommendation = 'INVESTIGATE'
                recommended_price = current_price
                reasoning = "Price is similar to best peer but RevPAR is lower. Check quality/marketing."
                
        else:  # new_hotel
            # Use peer median as starting point
            recommendation = 'SET'
            recommended_price = peer_prices['median_price'].median()
            reasoning = f"New hotel. Start at peer median €{recommended_price:.0f}."
        
        # Cap price change at ±20% per cycle (ensures adoptability)
        if hotel_perf['has_history'] and hotel_perf['avg_price'] > 0:
            current_price = hotel_perf['avg_price']
            max_price = current_price * (1 + self.config.max_price_change)
            min_price = current_price * (1 - self.config.max_price_change)
            
            if recommended_price > max_price:
                recommended_price = max_price
                reasoning += f" (Capped at +{self.config.max_price_change*100:.0f}% for adoptability)"
            elif recommended_price < min_price:
                recommended_price = min_price
                reasoning += f" (Capped at -{self.config.max_price_change*100:.0f}% for adoptability)"
        
        # Store base price before lead time adjustment
        base_price = recommended_price
        
        # Apply OCCUPANCY-CONDITIONAL lead time adjustment if specified
        lead_bucket = None
        lead_multiplier = 1.0
        lead_reasoning = None
        if lead_time_days is not None:
            lead_bucket = get_lead_time_bucket(lead_time_days)
            
            # Get occupancy-conditional multiplier
            # Key insight: 
            #   - Advance bookings (31+ days): ALWAYS premium (free money)
            #   - Short-term (≤7 days): ONLY discount if occupancy is LOW
            lead_multiplier, lead_reasoning = self.elasticity_calc.get_lead_time_multiplier(
                segment, 
                lead_time_days,
                current_occupancy=current_occ
            )
            lead_signal_source = 'occupancy_conditional'
            
            # Apply lead time multiplier to recommended price
            recommended_price = base_price * lead_multiplier
            
            # Build detailed reasoning
            if lead_reasoning == 'advance_premium':
                reasoning += f" Advance booking ({lead_bucket}): +{(lead_multiplier-1)*100:.0f}% premium."
            elif lead_reasoning == 'low_occ_discount':
                reasoning += f" Low occupancy ({current_occ*100:.0f}%) + short lead ({lead_bucket}): {(lead_multiplier-1)*100:+.0f}% to fill rooms."
            elif lead_reasoning == 'medium_occ_partial':
                reasoning += f" Medium occupancy ({current_occ*100:.0f}%) + short lead: partial discount ({lead_multiplier:.2f}x)."
            elif lead_reasoning == 'high_occ_premium':
                reasoning += f" High occupancy ({current_occ*100:.0f}%) + last-minute: hold/premium price."
            elif lead_reasoning == 'hold_price':
                reasoning += f" Standard lead time: holding base price."
            else:
                reasoning += f" Lead time adjustment: {lead_bucket} ({lead_multiplier:.2f}x)."
        
        # Calculate expected RevPAR at recommended price
        if hotel_perf['has_history'] and recommended_price != hotel_perf['avg_price']:
            price_change = (recommended_price - hotel_perf['avg_price']) / hotel_perf['avg_price']
            occ_change = elasticity * price_change
            new_occ = np.clip(current_occ * (1 + occ_change), 0.05, 0.95)
            expected_revpar = recommended_price * new_occ
        elif hotel_perf['has_history']:
            expected_revpar = hotel_revpar
        else:
            expected_revpar = recommended_price * 0.3  # Assume 30% for new
        
        result = {
            'hotel_id': hotel_id,
            'target_date': str(target_date),
            'segment': segment,
            'performance': performance,
            'recommendation': recommendation,
            'current_price': hotel_perf['avg_price'] if hotel_perf['has_history'] else None,
            'recommended_price': round(recommended_price, 2),
            'base_price': round(base_price, 2),
            'current_revpar': round(hotel_revpar, 2) if hotel_revpar else None,
            'expected_revpar': round(expected_revpar, 2),
            'peer_median_price': round(peer_prices['median_price'].median(), 2),
            'best_peer_price': round(best_peer['price'], 2),
            'best_peer_revpar': round(best_peer['revpar'], 2),
            'elasticity_used': round(elasticity, 3),
            'reasoning': reasoning,
            'confidence': self._calculate_confidence(hotel_perf, len(peer_prices)),
            'n_peers': len(peer_prices)
        }
        
        # Add lead time info if provided
        if lead_time_days is not None:
            result['lead_time_days'] = lead_time_days
            result['lead_bucket'] = lead_bucket
            result['lead_multiplier'] = round(lead_multiplier, 3)
            result['lead_signal_source'] = lead_signal_source
            result['lead_strategy'] = lead_reasoning  # Why this multiplier was applied
            result['occupancy_used'] = round(current_occ, 3)
        
        return result
    
    def _calculate_peer_revpar(
        self, 
        peer_ids: List[int], 
        target_date: date
    ) -> Dict:
        """Calculate RevPAR metrics for peers."""
        peer_ids_str = ','.join(map(str, peer_ids))
        
        query = f"""
        WITH hotel_capacity AS (
            SELECT 
                b.hotel_id,
                SUM(DISTINCT r.number_of_rooms) as total_rooms
            FROM bookings b
            JOIN booked_rooms br ON b.id = br.booking_id
            LEFT JOIN rooms r ON br.room_id = r.id
            WHERE b.hotel_id IN ({peer_ids_str})
            GROUP BY b.hotel_id
        ),
        peer_weekly AS (
            SELECT 
                b.hotel_id,
                AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as avg_price,
                COUNT(DISTINCT b.id) as bookings,
                hc.total_rooms,
                COUNT(DISTINCT b.id) * 1.0 / GREATEST(1, hc.total_rooms * 7) as occupancy
            FROM bookings b
            JOIN hotel_capacity hc ON b.hotel_id = hc.hotel_id
            WHERE b.hotel_id IN ({peer_ids_str})
              AND b.status IN ('confirmed', 'Booked')
              AND b.arrival_date >= '{target_date - timedelta(days=90)}'
            GROUP BY b.hotel_id, hc.total_rooms
        )
        SELECT 
            hotel_id,
            avg_price,
            occupancy,
            avg_price * occupancy as revpar
        FROM peer_weekly
        WHERE occupancy > 0
        ORDER BY revpar DESC
        """
        
        df = self.con.execute(query).fetchdf()
        
        if len(df) == 0:
            return {
                'median': 30, 'p25': 20, 'p75': 50,
                'best_peer': {'hotel_id': None, 'price': 100, 'revpar': 30}
            }
        
        return {
            'median': df['revpar'].median(),
            'p25': df['revpar'].quantile(0.25),
            'p75': df['revpar'].quantile(0.75),
            'best_peer': {
                'hotel_id': df.iloc[0]['hotel_id'],
                'price': df.iloc[0]['avg_price'],
                'revpar': df.iloc[0]['revpar']
            }
        }
    
    def _calculate_confidence(self, hotel_perf: Dict, n_peers: int) -> str:
        """Calculate confidence level in recommendation."""
        if not hotel_perf['has_history']:
            return 'low'
        
        weeks = hotel_perf['weeks_of_data']
        
        if weeks >= 12 and n_peers >= 8:
            return 'high'
        elif weeks >= 6 and n_peers >= 5:
            return 'medium'
        else:
            return 'low'
    
    def get_daily_multiplier(self, segment: str, target_date: date) -> float:
        """
        Get the combined day-of-week and monthly multiplier for a segment and date.
        
        Args:
            segment: Market segment (e.g., 'major_metro', 'resort_coastal')
            target_date: The specific date
            
        Returns:
            Combined multiplier to apply to base price
        """
        dow = target_date.weekday()  # 0=Monday in Python, but our data uses 0=Sunday
        # Convert to match our data (0=Sunday)
        dow_adjusted = (dow + 1) % 7
        month = target_date.month
        
        # Get segment-specific multipliers
        dow_mults = self.segment_multipliers.get('day_of_week', {}).get(segment, {})
        month_mults = self.segment_multipliers.get('monthly', {}).get(segment, {})
        
        # Get multipliers (default to 1.0 if not found)
        dow_mult = dow_mults.get(str(dow_adjusted), dow_mults.get(dow_adjusted, 1.0))
        month_mult = month_mults.get(str(month), month_mults.get(month, 1.0))
        
        # Combined multiplier
        return dow_mult * month_mult
    
    def recommend_daily(
        self, 
        hotel_id: int, 
        target_date: date,
        lead_time_days: Optional[int] = None
    ) -> Dict:
        """
        Get the recommended price for a SPECIFIC DAY.
        
        This is the main API endpoint - returns the price for one day
        with day-of-week, seasonal, and lead time adjustments applied.
        
        Args:
            hotel_id: The hotel to get pricing for
            target_date: The specific date
            lead_time_days: Optional days between booking and arrival
            
        Returns:
            Dict with:
            - recommended_price: The price for this specific day
            - base_price: The baseline (mid-week) price
            - day_multiplier: The day-of-week multiplier
            - month_multiplier: The seasonal multiplier
            - lead_multiplier: The lead time multiplier (if lead_time_days provided)
            - segment: The hotel's market segment
            - reasoning: Explanation of the price
        """
        if not self._fitted:
            raise ValueError("Must call fit() before recommend_daily()")
        
        # Get base recommendation (uses weekly averages internally)
        # Pass lead_time_days to get lead-time adjusted pricing
        base_rec = self.recommend(hotel_id, target_date, lead_time_days=lead_time_days)
        
        if base_rec.get('status') == 'error':
            return base_rec
        
        segment = base_rec['segment']
        base_price = base_rec['recommended_price']
        
        # Get segment-specific multipliers for this date
        dow = target_date.weekday()
        dow_adjusted = (dow + 1) % 7  # Convert Python dow to our format
        month = target_date.month
        
        dow_mults = self.segment_multipliers.get('day_of_week', {}).get(segment, {})
        month_mults = self.segment_multipliers.get('monthly', {}).get(segment, {})
        
        dow_mult = float(dow_mults.get(str(dow_adjusted), dow_mults.get(dow_adjusted, 1.0)))
        month_mult = float(month_mults.get(str(month), month_mults.get(month, 1.0)))
        
        # Apply multipliers to get daily price
        daily_price = base_price * dow_mult * month_mult
        
        day_name = target_date.strftime('%A')
        month_name = target_date.strftime('%B')
        
        result = {
            'hotel_id': hotel_id,
            'date': str(target_date),
            'day_of_week': day_name,
            'month': month_name,
            'segment': segment,
            'base_price': round(base_price, 2),
            'day_multiplier': round(dow_mult, 3),
            'month_multiplier': round(month_mult, 3),
            'combined_multiplier': round(dow_mult * month_mult, 3),
            'recommended_price': round(daily_price, 2),
            'performance': base_rec['performance'],
            'recommendation': base_rec['recommendation'],
            'elasticity': base_rec['elasticity_used'],
            'confidence': base_rec['confidence'],
            'reasoning': f"{base_rec['reasoning']} Daily adjustment: {day_name} ({dow_mult:.2f}x) × {month_name} ({month_mult:.2f}x)"
        }
        
        # Add lead time info if provided
        if lead_time_days is not None:
            result['lead_time_days'] = lead_time_days
            result['lead_bucket'] = base_rec.get('lead_bucket')
            result['lead_multiplier'] = base_rec.get('lead_multiplier')
            result['lead_signal_source'] = base_rec.get('lead_signal_source')
            result['lead_strategy'] = base_rec.get('lead_strategy')
            result['occupancy_used'] = base_rec.get('occupancy_used')
        
        return result
    
    def recommend_date_range(
        self, 
        hotel_id: int, 
        start_date: date,
        days: int = 7,
        lead_time_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get recommended prices for a date range.
        
        Args:
            hotel_id: The hotel to get pricing for
            start_date: First day of the range
            days: Number of days to return (default 7 = one week)
            lead_time_days: Optional days between booking and arrival
            
        Returns:
            DataFrame with daily pricing for each date
        """
        recommendations = []
        
        for i in range(days):
            target_date = start_date + timedelta(days=i)
            rec = self.recommend_daily(hotel_id, target_date, lead_time_days=lead_time_days)
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)
    
    def recommend_week(
        self, 
        hotel_id: int, 
        start_date: date
    ) -> pd.DataFrame:
        """Generate daily recommendations for a full week."""
        return self.recommend_date_range(hotel_id, start_date, days=7)


def main():
    """Test the pipeline."""
    from datetime import date
    
    print("="*70)
    print("PRICING PIPELINE TEST")
    print("="*70)
    
    pipeline = PricingPipeline()
    pipeline.fit()
    
    # Test on a few hotels
    test_hotels = [1, 10, 100]
    target = date(2024, 6, 15)
    
    for hotel_id in test_hotels:
        print(f"\n--- Hotel {hotel_id} ---")
        rec = pipeline.recommend(hotel_id, target)
        
        if rec.get('status') == 'error':
            print(f"  Error: {rec['message']}")
        else:
            print(f"  Segment: {rec['segment']}")
            print(f"  Performance: {rec['performance']}")
            print(f"  Recommendation: {rec['recommendation']}")
            print(f"  Current price: €{rec['current_price']:.0f}" if rec['current_price'] else "  New hotel")
            print(f"  Recommended: €{rec['recommended_price']:.0f}")
            print(f"  Expected RevPAR: €{rec['expected_revpar']:.2f}")
            print(f"  Elasticity: {rec['elasticity_used']:.3f}")
            print(f"  Confidence: {rec['confidence']}")
            print(f"  Reasoning: {rec['reasoning']}")


if __name__ == "__main__":
    main()

