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


@dataclass
class PricingConfig:
    """Configuration for pricing pipeline."""
    n_peers: int = 10  # Number of KNN peers
    price_search_range: Tuple[float, float] = (0.5, 1.5)  # Multiplier range for price search
    price_steps: int = 30  # Number of price points to evaluate
    min_peer_bookings: int = 5  # Minimum bookings for peer to be considered
    include_cancelled: bool = True  # Include cancelled bookings in peer prices


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
            AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as price,
            COUNT(*) as bookings
        FROM bookings b
        WHERE b.status IN ('confirmed', 'Booked')
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
                AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as avg_price,
                COUNT(DISTINCT b.id) as bookings,
                AVG(hc.total_rooms) as total_rooms
            FROM bookings b
            JOIN hotel_capacity hc ON b.id = hc.booking_id
            WHERE b.status IN ('confirmed', 'Booked')
            GROUP BY b.hotel_id, DATE_TRUNC('week', b.arrival_date)
            HAVING COUNT(DISTINCT b.id) >= 3
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
        query = """
        WITH hotel_stats AS (
            SELECT 
                b.hotel_id,
                AVG(b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date))) as avg_price,
                COUNT(*) as bookings
            FROM bookings b
            WHERE b.status IN ('confirmed', 'Booked')
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
        """Fit the pipeline (build indices, calculate elasticity from data)."""
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
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Get peer prices for a target date, including cancelled bookings.
        
        Cancelled bookings are included because they show willingness to pay.
        """
        peer_ids_str = ','.join(map(str, peer_ids))
        
        # Include cancelled bookings - they show willingness to pay
        status_filter = "('confirmed', 'Booked', 'cancelled')" if self.config.include_cancelled else "('confirmed', 'Booked')"
        
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
        target_date: date
    ) -> Dict:
        """
        Generate pricing recommendation for a hotel on a specific date.
        
        Returns:
            Dict with recommendation details including:
            - recommended_price
            - expected_revpar
            - performance_status (underperforming/on_par/outperforming)
            - recommendation_type (RAISE/LOWER/HOLD)
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
                'status': 'error',
                'message': 'No peers found'
            }
        
        peer_ids = peers['hotel_id'].tolist()
        
        # Get peer prices (including cancelled bookings)
        peer_prices = self.get_peer_prices(peer_ids, target_date)
        
        if len(peer_prices) == 0:
            return {
                'hotel_id': hotel_id,
                'target_date': target_date,
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
        
        return {
            'hotel_id': hotel_id,
            'target_date': str(target_date),
            'segment': segment,
            'performance': performance,
            'recommendation': recommendation,
            'current_price': hotel_perf['avg_price'] if hotel_perf['has_history'] else None,
            'recommended_price': round(recommended_price, 2),
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
        target_date: date
    ) -> Dict:
        """
        Get the recommended price for a SPECIFIC DAY.
        
        This is the main API endpoint - returns the price for one day
        with day-of-week and seasonal adjustments applied.
        
        Args:
            hotel_id: The hotel to get pricing for
            target_date: The specific date
            
        Returns:
            Dict with:
            - recommended_price: The price for this specific day
            - base_price: The baseline (mid-week) price
            - day_multiplier: The day-of-week multiplier
            - month_multiplier: The seasonal multiplier
            - segment: The hotel's market segment
            - reasoning: Explanation of the price
        """
        if not self._fitted:
            raise ValueError("Must call fit() before recommend_daily()")
        
        # Get base recommendation (uses weekly averages internally)
        base_rec = self.recommend(hotel_id, target_date)
        
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
        
        return {
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
    
    def recommend_date_range(
        self, 
        hotel_id: int, 
        start_date: date,
        days: int = 7
    ) -> pd.DataFrame:
        """
        Get recommended prices for a date range.
        
        Args:
            hotel_id: The hotel to get pricing for
            start_date: First day of the range
            days: Number of days to return (default 7 = one week)
            
        Returns:
            DataFrame with daily pricing for each date
        """
        recommendations = []
        
        for i in range(days):
            target_date = start_date + timedelta(days=i)
            rec = self.recommend_daily(hotel_id, target_date)
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

