"""
Rolling Window Validation for Pricing Pipeline

Setup:
- 4 months training window
- 1 month prediction window
- Roll forward month by month

Metrics per window:
1. Segment-level elasticity
2. Pricing prediction accuracy
3. % of hotels with optimization opportunity
4. Expected RevPAR increase
5. Total incremental revenue for Spain
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import duckdb
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import init_db
from src.features.engineering import get_market_segments_vectorized


@dataclass
class ValidationConfig:
    """Configuration for rolling validation."""
    train_months: int = 4
    test_months: int = 1
    min_bookings_train: int = 10
    min_bookings_test: int = 3
    n_peers: int = 10
    # Only use clean data from 2023-2024
    min_date: str = '2023-01-01'
    max_date: str = '2024-12-31'


class RollingValidator:
    """
    Validates the pricing pipeline using rolling windows.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.con = init_db()
        self.results: List[Dict] = []
        
    def get_date_range(self) -> Tuple[date, date]:
        """Get the full date range of available data."""
        query = """
        SELECT 
            MIN(arrival_date) as min_date,
            MAX(arrival_date) as max_date
        FROM bookings
        WHERE status IN ('confirmed', 'Booked')
        """
        result = self.con.execute(query).fetchdf()
        return (
            result.iloc[0]['min_date'].date(),
            result.iloc[0]['max_date'].date()
        )
    
    def get_hotel_segments(self) -> pd.DataFrame:
        """Get segment for each hotel."""
        query = """
        SELECT hotel_id, latitude, longitude
        FROM hotel_location
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
        hotels = self.con.execute(query).fetchdf()
        
        try:
            dist_df = pd.read_csv('outputs/data/hotel_distance_features.csv')
            hotels = hotels.merge(dist_df, on='hotel_id', how='left')
            hotels['distance_from_coast'] = hotels['distance_from_coast'].fillna(100)
        except FileNotFoundError:
            hotels['distance_from_coast'] = 100
        
        hotels['segment'] = get_market_segments_vectorized(
            hotels['latitude'].values,
            hotels['longitude'].values,
            hotels['distance_from_coast'].values
        )
        
        return hotels[['hotel_id', 'segment', 'latitude', 'longitude']]
    
    def get_booking_data(
        self, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """Get booking data for a date range."""
        query = f"""
        WITH hotel_capacity AS (
            SELECT 
                b.hotel_id,
                SUM(DISTINCT r.number_of_rooms) as total_rooms
            FROM bookings b
            JOIN booked_rooms br ON b.id = br.booking_id
            LEFT JOIN rooms r ON br.room_id = r.id
            GROUP BY b.hotel_id
        ),
        daily_bookings AS (
            SELECT 
                b.hotel_id,
                b.arrival_date as booking_date,
                EXTRACT(dow FROM b.arrival_date) as day_of_week,
                EXTRACT(month FROM b.arrival_date) as month,
                b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) as price_per_night,
                1 as booking_count
            FROM bookings b
            WHERE b.status IN ('confirmed', 'Booked')
              AND b.arrival_date >= '{start_date}'
              AND b.arrival_date < '{end_date}'
        )
        SELECT 
            d.hotel_id,
            d.booking_date,
            d.day_of_week,
            d.month,
            AVG(d.price_per_night) as avg_price,
            SUM(d.booking_count) as bookings,
            hc.total_rooms
        FROM daily_bookings d
        LEFT JOIN hotel_capacity hc ON d.hotel_id = hc.hotel_id
        GROUP BY d.hotel_id, d.booking_date, d.day_of_week, d.month, hc.total_rooms
        """
        return self.con.execute(query).fetchdf()
    
    def calculate_segment_elasticity(
        self, 
        train_data: pd.DataFrame,
        segments: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate elasticity per segment from training data."""
        # Merge with segments
        df = train_data.merge(segments, on='hotel_id')
        
        # Aggregate to hotel level
        hotel_agg = df.groupby(['hotel_id', 'segment']).agg({
            'avg_price': 'mean',
            'bookings': 'mean'
        }).reset_index()
        
        segment_elasticity = {}
        
        for segment in hotel_agg['segment'].unique():
            seg_data = hotel_agg[hotel_agg['segment'] == segment]
            
            if len(seg_data) < 20:
                segment_elasticity[segment] = -0.5  # Default
                continue
            
            try:
                slope, _, _, _, _ = stats.linregress(
                    np.log(seg_data['avg_price'].clip(lower=1)),
                    np.log(seg_data['bookings'].clip(lower=0.1))
                )
                segment_elasticity[segment] = np.clip(slope, -2.0, -0.1)
            except:
                segment_elasticity[segment] = -0.5
        
        return segment_elasticity
    
    def calculate_dow_multipliers(
        self,
        train_data: pd.DataFrame,
        segments: pd.DataFrame
    ) -> Dict[str, Dict[int, float]]:
        """Calculate day-of-week multipliers per segment."""
        df = train_data.merge(segments, on='hotel_id')
        
        multipliers = {}
        
        for segment in df['segment'].unique():
            seg_data = df[df['segment'] == segment]
            dow_prices = seg_data.groupby('day_of_week')['avg_price'].mean()
            
            # Baseline = Wednesday (day 3)
            baseline = dow_prices.get(3, dow_prices.mean())
            if baseline > 0:
                multipliers[segment] = (dow_prices / baseline).to_dict()
            else:
                multipliers[segment] = {i: 1.0 for i in range(7)}
        
        return multipliers
    
    def find_peers_and_prices(
        self,
        train_data: pd.DataFrame,
        segments: pd.DataFrame
    ) -> pd.DataFrame:
        """Build peer index using FEATURE-BASED matching (not geographic-only)."""
        # Aggregate training data to hotel level
        hotel_train = train_data.groupby('hotel_id').agg({
            'avg_price': 'mean',
            'bookings': 'sum',
            'total_rooms': 'first'
        }).reset_index()
        
        # Merge with location/segment
        hotel_train = hotel_train.merge(segments, on='hotel_id')
        
        # Add price tier for tighter peer matching
        hotel_train['price_tier'] = pd.qcut(
            hotel_train['avg_price'].fillna(100),
            q=5,
            labels=[0, 1, 2, 3, 4],
            duplicates='drop'
        ).astype(float)
        
        # Add segment as numeric
        hotel_train['segment_num'] = hotel_train['segment'].astype('category').cat.codes
        
        # Fill missing values
        hotel_train['avg_room_size'] = train_data.groupby('hotel_id')['avg_price'].transform('mean').reindex(hotel_train['hotel_id']).fillna(25)
        
        valid = hotel_train.dropna(subset=['latitude', 'longitude'])
        if len(valid) < 10:
            return pd.DataFrame()
        
        # FEATURE-BASED matching (not just lat/lon!)
        feature_cols = ['latitude', 'longitude', 'price_tier', 'segment_num']
        scaler = StandardScaler()
        X = scaler.fit_transform(valid[feature_cols].fillna(0).values)
        tree = cKDTree(X)
        
        # Find peers and compute peer metrics
        results = []
        for idx, row in valid.iterrows():
            # Use feature-based query
            hotel_features = [row['latitude'], row['longitude'], row['price_tier'], row['segment_num']]
            hotel_scaled = scaler.transform([hotel_features])
            distances, indices = tree.query(hotel_scaled, k=self.config.n_peers + 1)
            
            peer_indices = [i for i in indices[0] if i != idx][:self.config.n_peers]
            peers = valid.iloc[peer_indices]
            
            # Calculate peer RevPAR
            peers_revpar = peers['avg_price'] * (peers['bookings'] / peers['total_rooms'].clip(lower=1) / 30)
            hotel_revpar = row['avg_price'] * (row['bookings'] / max(1, row['total_rooms']) / 30)
            
            # Best peer
            best_peer_idx = peers_revpar.idxmax() if len(peers_revpar) > 0 else None
            best_peer = peers.loc[best_peer_idx] if best_peer_idx is not None else None
            
            results.append({
                'hotel_id': row['hotel_id'],
                'segment': row['segment'],
                'actual_price': row['avg_price'],
                'actual_revpar': hotel_revpar,
                'peer_median_price': peers['avg_price'].median(),
                'peer_median_revpar': peers_revpar.median(),
                'peer_p25_revpar': peers_revpar.quantile(0.25),
                'peer_p75_revpar': peers_revpar.quantile(0.75),
                'best_peer_price': best_peer['avg_price'] if best_peer is not None else row['avg_price'],
                'best_peer_revpar': peers_revpar.max() if len(peers_revpar) > 0 else hotel_revpar,
            })
        
        return pd.DataFrame(results)
    
    def evaluate_window(
        self,
        train_start: date,
        train_end: date,
        test_start: date,
        test_end: date,
        segments: pd.DataFrame
    ) -> Dict:
        """Evaluate one rolling window."""
        # Get data
        train_data = self.get_booking_data(train_start, train_end)
        test_data = self.get_booking_data(test_start, test_end)
        
        if len(train_data) == 0 or len(test_data) == 0:
            return None
        
        # Calculate segment elasticity
        segment_elasticity = self.calculate_segment_elasticity(train_data, segments)
        
        # Calculate day-of-week multipliers
        dow_multipliers = self.calculate_dow_multipliers(train_data, segments)
        
        # Find peers and get peer prices
        peer_data = self.find_peers_and_prices(train_data, segments)
        
        if len(peer_data) == 0:
            return None
        
        # Classify performance and generate recommendations
        peer_data['performance'] = peer_data.apply(
            lambda r: 'underperforming' if r['actual_revpar'] < r['peer_p25_revpar']
            else ('outperforming' if r['actual_revpar'] > r['peer_p75_revpar'] else 'on_par'),
            axis=1
        )
        
        # Generate recommendations
        def get_recommendation(row):
            if row['performance'] == 'outperforming':
                return 'HOLD', row['actual_price']
            elif row['performance'] == 'on_par':
                return 'HOLD', row['actual_price']
            else:
                # Underperforming - compare to best peer
                if row['actual_price'] < row['best_peer_price'] * 0.85:
                    return 'RAISE', row['best_peer_price']
                elif row['actual_price'] > row['best_peer_price'] * 1.15:
                    return 'LOWER', row['best_peer_price']
                else:
                    return 'INVESTIGATE', row['actual_price']
        
        peer_data[['recommendation', 'recommended_price']] = peer_data.apply(
            lambda r: pd.Series(get_recommendation(r)), axis=1
        )
        
        # Calculate expected RevPAR at recommended price
        def calc_expected_revpar(row):
            if row['recommendation'] == 'HOLD':
                return row['actual_revpar']
            
            elasticity = segment_elasticity.get(row['segment'], -0.5)
            if row['actual_price'] == 0:
                return row['actual_revpar']
            price_change = (row['recommended_price'] - row['actual_price']) / row['actual_price']
            
            # Current occupancy proxy
            current_occ = row['actual_revpar'] / row['actual_price'] if row['actual_price'] > 0 else 0.3
            
            # New occupancy with elasticity
            new_occ = current_occ * (1 + elasticity * price_change)
            new_occ = np.clip(new_occ, 0.05, 0.95)
            
            return row['recommended_price'] * new_occ
        
        peer_data['expected_revpar'] = peer_data.apply(calc_expected_revpar, axis=1)
        peer_data['revpar_lift'] = peer_data['expected_revpar'] - peer_data['actual_revpar']
        
        # Calculate metrics (no MAE - we're recommending optimal price, not predicting what they'll charge)
        n_hotels = len(peer_data)
        n_with_opportunity = len(peer_data[peer_data['recommendation'] != 'HOLD'])
        pct_with_opportunity = n_with_opportunity / n_hotels * 100 if n_hotels > 0 else 0
        
        # RevPAR lift for hotels with optimization opportunity
        hotels_with_opp = peer_data[peer_data['recommendation'] != 'HOLD']
        avg_revpar_lift = hotels_with_opp['revpar_lift'].mean() if len(hotels_with_opp) > 0 else 0
        total_monthly_lift = peer_data['revpar_lift'].sum() * 30  # Monthly
        
        # Average recommended price vs actual price
        avg_recommended = peer_data['recommended_price'].mean()
        avg_actual = peer_data['actual_price'].mean()
        avg_price_diff = avg_recommended - avg_actual
        
        return {
            'train_start': str(train_start),
            'train_end': str(train_end),
            'test_start': str(test_start),
            'test_end': str(test_end),
            'n_hotels': n_hotels,
            'segment_elasticity': segment_elasticity,
            'performance_distribution': peer_data['performance'].value_counts().to_dict(),
            'recommendation_distribution': peer_data['recommendation'].value_counts().to_dict(),
            'pct_with_opportunity': pct_with_opportunity,
            'avg_revpar_lift_per_hotel': avg_revpar_lift,
            'total_monthly_revpar_lift': total_monthly_lift,
            'avg_recommended_price': avg_recommended,
            'avg_actual_price': avg_actual,
            'avg_price_diff': avg_price_diff,
            'hotels_by_segment': peer_data.groupby('segment').size().to_dict(),
            'lift_by_segment': peer_data.groupby('segment')['revpar_lift'].mean().to_dict(),
        }
    
    def run_validation(self) -> pd.DataFrame:
        """Run full rolling window validation on clean 2023-2024 data."""
        segments = self.get_hotel_segments()
        
        # Use only clean 2023-2024 data
        min_date = date.fromisoformat(self.config.min_date)
        max_date = date.fromisoformat(self.config.max_date)
        
        print(f"Using clean data: {min_date} to {max_date}")
        print(f"Hotels with location: {len(segments)}")
        print()
        
        current_start = min_date
        window_num = 0
        
        while True:
            train_start = current_start
            train_end = train_start + relativedelta(months=self.config.train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.config.test_months)
            
            if test_end > max_date:
                break
            
            window_num += 1
            print(f"Window {window_num}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
            
            result = self.evaluate_window(
                train_start, train_end, test_start, test_end, segments
            )
            
            if result:
                result['window'] = window_num
                self.results.append(result)
                
                print(f"  Hotels: {result['n_hotels']}, With opportunity: {result['pct_with_opportunity']:.1f}%")
                print(f"  Avg RevPAR lift: €{result['avg_revpar_lift_per_hotel']:.2f}")
                print(f"  Total monthly lift: €{result['total_monthly_revpar_lift']:,.0f}")
            
            # Roll forward
            current_start = current_start + relativedelta(months=1)
        
        return pd.DataFrame(self.results)
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics across all windows."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Aggregate elasticity across windows
        all_elasticity = {}
        for result in self.results:
            for seg, elas in result['segment_elasticity'].items():
                if seg not in all_elasticity:
                    all_elasticity[seg] = []
                all_elasticity[seg].append(elas)
        
        avg_elasticity = {seg: np.mean(vals) for seg, vals in all_elasticity.items()}
        
        # Calculate total Spain opportunity
        avg_monthly_lift = df['total_monthly_revpar_lift'].mean()
        annual_lift = avg_monthly_lift * 12
        
        return {
            'n_windows': len(df),
            'avg_hotels_per_window': df['n_hotels'].mean(),
            'avg_pct_with_opportunity': df['pct_with_opportunity'].mean(),
            'avg_revpar_lift_per_hotel': df['avg_revpar_lift_per_hotel'].mean(),
            'avg_monthly_lift_spain': avg_monthly_lift,
            'annual_lift_spain': annual_lift,
            'avg_recommended_price': df['avg_recommended_price'].mean(),
            'avg_actual_price': df['avg_actual_price'].mean(),
            'avg_price_diff': df['avg_price_diff'].mean(),
            'segment_elasticity': avg_elasticity,
        }
    
    def calculate_lead_time_metrics(self) -> Dict:
        """
        Calculate lead time pricing patterns across the data.
        
        This analyzes how prices vary by lead time (booking window)
        and calculates segment-level lead time multipliers.
        
        Returns:
            Dict with lead time metrics and multipliers
        """
        print("\n=== LEAD TIME ANALYSIS ===")
        
        # Get lead time distribution and pricing patterns
        query = """
        WITH booking_lead AS (
            SELECT 
                b.hotel_id,
                b.total_price / GREATEST(1, DATE_DIFF('day', b.arrival_date, b.departure_date)) as price_per_night,
                DATE_DIFF('day', b.created_at::DATE, b.arrival_date) as lead_time_days,
                CASE 
                    WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) = 0 THEN 'same_day'
                    WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 1 AND 3 THEN 'very_short'
                    WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 4 AND 7 THEN 'short'
                    WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 8 AND 14 THEN 'medium'
                    WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 15 AND 30 THEN 'standard'
                    WHEN DATE_DIFF('day', b.created_at::DATE, b.arrival_date) BETWEEN 31 AND 60 THEN 'advance'
                    ELSE 'far_advance'
                END as lead_bucket
            FROM bookings b
            WHERE b.status IN ('confirmed', 'Booked', 'cancelled')
              AND b.created_at IS NOT NULL
              AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) >= 0
              AND DATE_DIFF('day', b.created_at::DATE, b.arrival_date) < 365
              AND b.total_price > 0
              AND b.arrival_date >= '2023-01-01'
              AND b.arrival_date < '2025-01-01'
        )
        SELECT 
            lead_bucket,
            COUNT(*) as bookings,
            AVG(price_per_night) as avg_price,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_night) as median_price
        FROM booking_lead
        GROUP BY lead_bucket
        ORDER BY MIN(lead_time_days)
        """
        
        lead_df = self.con.execute(query).fetchdf()
        
        # Calculate multipliers relative to 'standard' (15-30 days)
        if 'standard' in lead_df['lead_bucket'].values:
            baseline = lead_df[lead_df['lead_bucket'] == 'standard']['avg_price'].values[0]
        else:
            baseline = lead_df['avg_price'].median()
        
        lead_multipliers = {}
        for _, row in lead_df.iterrows():
            lead_multipliers[row['lead_bucket']] = round(row['avg_price'] / baseline, 3)
        
        print("Lead time pricing patterns:")
        print(lead_df.to_string(index=False))
        
        print("\nLead time multipliers (vs standard booking window):")
        for bucket in ['same_day', 'very_short', 'short', 'medium', 'standard', 'advance', 'far_advance']:
            if bucket in lead_multipliers:
                print(f"  {bucket}: {lead_multipliers[bucket]:.2f}x")
        
        # Calculate potential RevPAR impact from lead time optimization
        total_bookings = lead_df['bookings'].sum()
        short_term_bookings = lead_df[lead_df['lead_bucket'].isin(['same_day', 'very_short', 'short'])]['bookings'].sum()
        pct_short_term = short_term_bookings / total_bookings * 100
        
        # If we raised short-term prices by 10%, what's the impact?
        short_term_revenue = (lead_df[lead_df['lead_bucket'].isin(['same_day', 'very_short', 'short'])]['avg_price'] * 
                             lead_df[lead_df['lead_bucket'].isin(['same_day', 'very_short', 'short'])]['bookings']).sum()
        potential_uplift = short_term_revenue * 0.1  # 10% increase
        
        print(f"\nShort-term bookings (≤7 days): {pct_short_term:.1f}%")
        print(f"Potential uplift from 10% price increase on short-term: €{potential_uplift:,.0f}")
        
        return {
            'lead_multipliers': lead_multipliers,
            'lead_df': lead_df,
            'pct_short_term': pct_short_term,
            'potential_uplift': potential_uplift
        }


def main():
    """Run rolling validation."""
    print("="*70)
    print("ROLLING WINDOW VALIDATION")
    print("4 months training, 1 month prediction (2023-2024 data only)")
    print("="*70)
    print()
    
    validator = RollingValidator()
    results_df = validator.run_validation()
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    summary = validator.generate_summary()
    
    print(f"\nWindows evaluated: {summary['n_windows']}")
    print(f"Avg hotels per window: {summary['avg_hotels_per_window']:.0f}")
    print(f"Avg % with optimization opportunity: {summary['avg_pct_with_opportunity']:.1f}%")
    print(f"Avg RevPAR lift per hotel: €{summary['avg_revpar_lift_per_hotel']:.2f}")
    print(f"Avg monthly lift (all Spain): €{summary['avg_monthly_lift_spain']:,.0f}")
    print(f"Estimated annual lift (all Spain): €{summary['annual_lift_spain']:,.0f}")
    
    print(f"\n=== PRICING ANALYSIS ===")
    print(f"Avg recommended price: €{summary['avg_recommended_price']:.0f}")
    print(f"Avg actual price: €{summary['avg_actual_price']:.0f}")
    print(f"Avg difference: €{summary['avg_price_diff']:+.0f}")
    
    print("\n=== SEGMENT ELASTICITY (calculated per window) ===")
    for seg, elas in sorted(summary['segment_elasticity'].items(), key=lambda x: x[1]):
        print(f"  {seg}: {elas:.3f}")
    
    # Lead time analysis
    lead_metrics = validator.calculate_lead_time_metrics()
    
    # Save results
    results_df.to_csv('outputs/data/rolling_validation_results.csv', index=False)
    print(f"\nSaved to outputs/data/rolling_validation_results.csv")
    
    # Save lead time metrics
    import json
    lead_multipliers_path = 'outputs/data/lead_time_multipliers.json'
    with open(lead_multipliers_path, 'w') as f:
        json.dump(lead_metrics['lead_multipliers'], f, indent=2)
    print(f"Saved lead time multipliers to {lead_multipliers_path}")
    
    return results_df, summary


if __name__ == "__main__":
    results, summary = main()

