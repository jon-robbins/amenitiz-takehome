"""
Competitor and Historical Pricing Features (Optimized).

For each booking, computes:
1. What similar hotels are charging for the same date (current market)
2. What this hotel charged last year for the same date (historical baseline)
3. What similar hotels charged last year (historical market)
4. City-level YoY as fallback for market trends
5. Peer last month average price (market rate for similar hotels)
6. Hotel's own last month average price (self-comparison)
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, Optional
from datetime import timedelta


# Configuration
DATE_WINDOW_DAYS = 1  # ±1 day for current competitor prices
HISTORICAL_WINDOW_DAYS = 7  # ±7 days for historical lookup


class CompetitorPriceCalculator:
    """
    Calculates competitor and historical pricing features.
    Optimized for vectorized operations where possible.
    """
    
    def __init__(self, similar_hotels: Optional[Dict[int, Set[int]]] = None):
        self.similar_hotels: Dict[int, Set[int]] = similar_hotels or {}
        self.hotel_city: Dict[int, str] = {}  # Pre-computed hotel -> city mapping
    
    def build_similar_hotels_from_knn(
        self,
        stage1_model,
        df: pd.DataFrame
    ) -> None:
        """Builds similar hotels mapping using KNN (skip if already provided)."""
        if self.similar_hotels:
            print("   Using pre-built similar hotels mapping...", flush=True)
            # Just build city mapping
            if 'city_standardized' in df.columns:
                self.hotel_city = df.groupby('hotel_id')['city_standardized'].first().to_dict()
            return
        
        print("   Building similar hotels mapping from KNN...", flush=True)
        
        unique_hotels = df.drop_duplicates('hotel_id').copy()
        n_hotels = len(unique_hotels)
        
        unique_hotels_encoded = stage1_model._encode_features(unique_hotels.copy(), fit=False)
        X_query = unique_hotels_encoded[stage1_model.feature_cols].fillna(0).values
        X_query_scaled = stage1_model.scaler.transform(X_query)
        
        distances, indices = stage1_model.knn.kneighbors(X_query_scaled)
        
        training_hotel_ids = stage1_model.data['hotel_id'].values
        query_hotel_ids = unique_hotels['hotel_id'].values
        
        for i in range(n_hotels):
            hotel_id = query_hotel_ids[i]
            neighbor_idx = indices[i]
            neighbor_hotel_ids = training_hotel_ids[neighbor_idx]
            self.similar_hotels[hotel_id] = set(neighbor_hotel_ids) - {hotel_id}
        
        # Pre-compute hotel -> city mapping
        if 'city_standardized' in df.columns:
            self.hotel_city = df.groupby('hotel_id')['city_standardized'].first().to_dict()
        
        print(f"   Built similarity mapping for {n_hotels:,} hotels", flush=True)
    
    def compute_batch(self, df: pd.DataFrame, all_bookings: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all pricing features using vectorized operations.
        """
        n_bookings = len(df)
        print(f"   Computing pricing features for {n_bookings:,} bookings...", flush=True)
        
        df = df.copy()
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        all_bookings = all_bookings.copy()
        all_bookings['arrival_date'] = pd.to_datetime(all_bookings['arrival_date'])
        all_bookings['created_at'] = pd.to_datetime(all_bookings['created_at'])
        
        # === STEP 0: Build last-month price lookup tables ===
        print("   Building last-month price lookup tables...", flush=True)
        all_bookings['year'] = all_bookings['arrival_date'].dt.year
        all_bookings['month'] = all_bookings['arrival_date'].dt.month
        all_bookings['year_month'] = all_bookings['year'] * 100 + all_bookings['month']
        
        # Hotel-level monthly averages: {(hotel_id, year_month): avg_price}
        hotel_monthly_avg = all_bookings.groupby(
            ['hotel_id', 'year_month']
        )['daily_price'].mean().to_dict()
        
        # === STEP 1: Compute city-level YoY (vectorized) ===
        print("   Computing city-level YoY...", flush=True)
        city_year_avg = all_bookings.groupby(['city_standardized', 'year'])['daily_price'].mean().unstack(fill_value=0)
        
        city_yoy = {}
        if 2023 in city_year_avg.columns and 2024 in city_year_avg.columns:
            for city in city_year_avg.index:
                if city_year_avg.loc[city, 2023] > 0:
                    city_yoy[city] = city_year_avg.loc[city, 2024] / city_year_avg.loc[city, 2023]
            else:
                    city_yoy[city] = 1.0
        
        for city, yoy in sorted(city_yoy.items(), key=lambda x: -x[1]):
            print(f"      {city}: {(yoy-1)*100:+.1f}% YoY", flush=True)
        
        # === STEP 2: Pre-compute historical price lookups (vectorized) ===
        print("   Building historical price lookup tables...", flush=True)
        
        # Create date keys for historical lookup (last year ± window)
        all_bookings['date_key'] = all_bookings['arrival_date'].dt.strftime('%Y-%m-%d')
        
        # Historical same-hotel prices: group by (hotel_id, month-day) to match across years
        all_bookings['month_day'] = all_bookings['arrival_date'].dt.strftime('%m-%d')
        
        hist_same_hotel = all_bookings[all_bookings['year'] == 2023].groupby(
            ['hotel_id', 'month_day']
        )['daily_price'].mean().to_dict()
        
        # Historical similar-hotels prices by month-day
        hist_2023 = all_bookings[all_bookings['year'] == 2023][['month_day', 'hotel_id', 'daily_price']]
        hist_similar_by_date = hist_2023.groupby('month_day').apply(
            lambda g: g.groupby('hotel_id')['daily_price'].mean().to_dict(),
            include_groups=False
        ).to_dict()
        
        # === STEP 3: Current competitor prices (vectorized where possible) ===
        print("   Computing current competitor prices...", flush=True)
        
        # Group all_bookings by arrival_date for fast lookup
        price_cols = all_bookings[['date_key', 'hotel_id', 'created_at', 'daily_price']]
        price_by_date_hotel = price_cols.groupby(['date_key', 'hotel_id']).apply(
            lambda g: g.sort_values('created_at')[['created_at', 'daily_price']].values.tolist(),
            include_groups=False
        ).to_dict()
        
        # Pre-allocate result arrays
        competitor_avg = np.zeros(n_bookings)
        competitor_min = np.zeros(n_bookings)
        competitor_max = np.zeros(n_bookings)
        competitor_count = np.zeros(n_bookings, dtype=int)
        hist_same = np.zeros(n_bookings)
        hist_similar = np.zeros(n_bookings)
        hist_similar_cnt = np.zeros(n_bookings, dtype=int)
        yoy_ratio = np.ones(n_bookings)
        yoy_source = np.zeros(n_bookings, dtype=int)
        # Peer last month features
        peer_last_month_avg = np.zeros(n_bookings)
        peer_last_month_cnt = np.zeros(n_bookings, dtype=int)
        hotel_last_month_avg = np.zeros(n_bookings)
        
        # Process in chunks for progress updates
        chunk_size = 10000
        
        for start in range(0, n_bookings, chunk_size):
            end = min(start + chunk_size, n_bookings)
            
            for i in range(start, end):
                row = df.iloc[i]
                hotel_id = row['hotel_id']
                arrival_date = row['arrival_date']
                created_at = row['created_at']
                city = row.get('city_standardized', 'other')
                
                month_day = arrival_date.strftime('%m-%d')
                similar = self.similar_hotels.get(hotel_id, set())
                
                # --- Peer last month average price ---
                # For a March 15 booking, get avg price of similar hotels in February
                arrival_year = arrival_date.year
                arrival_month = arrival_date.month
                # Previous month: handle January -> December of prior year
                if arrival_month == 1:
                    last_month_ym = (arrival_year - 1) * 100 + 12
                else:
                    last_month_ym = arrival_year * 100 + (arrival_month - 1)
                
                # Get peer prices from last month
                peer_lm_prices = []
                for comp_id in similar:
                    if (comp_id, last_month_ym) in hotel_monthly_avg:
                        peer_lm_prices.append(hotel_monthly_avg[(comp_id, last_month_ym)])
                
                if peer_lm_prices:
                    peer_last_month_avg[i] = np.mean(peer_lm_prices)
                    peer_last_month_cnt[i] = len(peer_lm_prices)
                
                # This hotel's own last month average
                if (hotel_id, last_month_ym) in hotel_monthly_avg:
                    hotel_last_month_avg[i] = hotel_monthly_avg[(hotel_id, last_month_ym)]
                
                # --- Current competitor prices ---
                comp_prices = []
                for day_offset in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1):
                    check_date = (arrival_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                    for comp_id in similar:
                        prices_list = price_by_date_hotel.get((check_date, comp_id), [])
                        for booking_time, price in prices_list:
                            if pd.Timestamp(booking_time) < created_at:
                                comp_prices.append(price)
                                break  # Take latest before created_at
                
                if comp_prices:
                    competitor_avg[i] = np.mean(comp_prices)
                    competitor_min[i] = np.min(comp_prices)
                    competitor_max[i] = np.max(comp_prices)
                    competitor_count[i] = len(comp_prices)
                
                # --- Historical same-hotel price (only for 2024 bookings) ---
                booking_year = arrival_date.year
                if booking_year >= 2024:  # Only look up history for 2024+ bookings
                    hist_same[i] = hist_same_hotel.get((hotel_id, month_day), 0.0)
                    
                    # --- Historical similar-hotels price ---
                    similar_hist_prices = []
                    date_hist = hist_similar_by_date.get(month_day, {})
                    for comp_id in similar:
                        if comp_id in date_hist:
                            similar_hist_prices.append(date_hist[comp_id])
                    
                    if similar_hist_prices:
                        hist_similar[i] = np.mean(similar_hist_prices)
                        hist_similar_cnt[i] = len(similar_hist_prices)
                
                # --- YoY ratio with city fallback ---
                if competitor_avg[i] > 0 and hist_similar[i] > 0:
                    yoy_ratio[i] = competitor_avg[i] / hist_similar[i]
                    yoy_source[i] = 1  # Similar hotels
                elif city in city_yoy:
                    yoy_ratio[i] = city_yoy[city]
                    yoy_source[i] = 2  # City fallback
            
            if end % 50000 == 0 or end == n_bookings:
                print(f"   Processed {end:,}/{n_bookings:,} bookings...", flush=True)
        
        # === STEP 4: Compute derived features ===
        implied_price = np.where(hist_same > 0, hist_same * yoy_ratio, 0.0)
        
        # Add all columns to DataFrame
        df['competitor_avg_price'] = competitor_avg
        df['competitor_min_price'] = competitor_min
        df['competitor_max_price'] = competitor_max
        df['competitor_price_count'] = competitor_count
        df['hist_same_hotel_price'] = hist_same
        df['hist_similar_hotels_price'] = hist_similar
        df['hist_similar_count'] = hist_similar_cnt
        df['competitor_yoy_ratio'] = yoy_ratio
        df['competitor_yoy_log_change'] = np.log(yoy_ratio)
        df['implied_price'] = implied_price
        df['implied_price_log'] = np.log1p(implied_price)
        # Peer last month features
        df['peer_last_month_avg'] = peer_last_month_avg
        df['peer_last_month_count'] = peer_last_month_cnt
        df['hotel_last_month_avg'] = hotel_last_month_avg
        
        # Summary stats
        print(f"   Done.", flush=True)
        print(f"   Competitor prices: {(competitor_count > 0).mean()*100:.1f}% coverage", flush=True)
        print(f"   Historical same-hotel: {(hist_same > 0).mean()*100:.1f}% coverage", flush=True)
        print(f"   Historical similar: {(hist_similar > 0).mean()*100:.1f}% coverage", flush=True)
        print(f"   YoY source: {(yoy_source==1).mean()*100:.1f}% similar, {(yoy_source==2).mean()*100:.1f}% city", flush=True)
        print(f"   Implied price: {(implied_price > 0).mean()*100:.1f}% coverage", flush=True)
        print(f"   Peer last month: {(peer_last_month_avg > 0).mean()*100:.1f}% coverage (avg {peer_last_month_cnt[peer_last_month_cnt > 0].mean():.0f} peers)", flush=True)
        print(f"   Hotel last month: {(hotel_last_month_avg > 0).mean()*100:.1f}% coverage", flush=True)
        
        return df


def add_competitor_features(
    df: pd.DataFrame,
    stage1_model,
    all_bookings: pd.DataFrame,
    similar_hotels: Optional[Dict[int, Set[int]]] = None
) -> pd.DataFrame:
    """
    Adds competitor and historical pricing features to a DataFrame.
    
    Features added:
    - competitor_avg_price, competitor_min_price, competitor_max_price, competitor_price_count
    - hist_same_hotel_price, hist_similar_hotels_price, hist_similar_count
    - competitor_yoy_ratio, competitor_yoy_log_change
    - implied_price, implied_price_log
    - peer_last_month_avg, peer_last_month_count (market rate from similar hotels)
    - hotel_last_month_avg (hotel's own recent pricing)
    
    Args:
        df: DataFrame to add features to
        stage1_model: Stage 1 model with KNN for similarity
        all_bookings: All bookings for historical lookups
        similar_hotels: Optional pre-built similar hotels mapping (skips KNN rebuild)
    """
    print("\nAdding competitor and historical pricing features...", flush=True)
    
    calculator = CompetitorPriceCalculator(similar_hotels=similar_hotels)
    calculator.build_similar_hotels_from_knn(stage1_model, all_bookings)
    
    df = calculator.compute_batch(df, all_bookings)
    
    return df
