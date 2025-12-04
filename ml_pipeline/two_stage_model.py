"""
Two-Stage Pricing Model.

Stage 1: KNN Peer Price Statistics
- Finds similar hotels by validated features
- Returns price statistics vector (mean, median, std, count, min, max)
- Excludes target hotel to work for new hotels

Stage 2: Full Feature Model (Daily)
- Uses all validated features + Stage 1 peer stats + temporal features
- Predicts actual daily price
- Learns holiday/weekend adjustments from data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

from lib.db import init_db
from lib.holiday_features import map_hotels_to_admin1, add_date_holiday_features
from ml_pipeline.features import (
    standardize_city,
    VIEW_QUALITY_MAP,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES
)
from ml_pipeline.competitor_features import add_competitor_features


MODEL_DIR = Path(__file__).parent.parent / 'outputs' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CITIES500_PATH = Path(__file__).parent.parent / 'data' / 'cities500.json'
DISTANCE_FEATURES_PATH = Path(__file__).parent.parent / 'outputs' / 'hotel_distance_features.csv'

# Features for blocking (exact match) - from elasticity estimation
BLOCK_FEATURES = [
    'market_segment',  # coastal/inland × Madrid/provincial
    'room_type',
    'children_allowed',
]

# Features for KNN within blocks (continuous only) - validated in elasticity
KNN_CONTINUOUS_FEATURES = [
    'log_room_size', 'room_capacity_pax', 'view_quality_ordinal', 
    'amenities_score', 'total_capacity_log', 'weekend_ratio',
]

# Segment-specific geographic features (used dynamically)
GEO_FEATURES_COASTAL = ['dist_coast_log']
GEO_FEATURES_INLAND = ['dist_center_km']

# Peer price statistics returned by Stage 1
PEER_STAT_COLS = [
    'peer_price_mean', 'peer_price_median', 'peer_price_std',
    'peer_price_count', 'peer_price_min', 'peer_price_max'
]


@dataclass
class PeerPriceStats:
    """Statistics from similar hotels."""
    mean: float
    median: float
    std: float
    count: int
    min: float
    max: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'peer_price_mean': self.mean,
            'peer_price_median': self.median,
            'peer_price_std': self.std,
            'peer_price_count': self.count,
            'peer_price_min': self.min,
            'peer_price_max': self.max
        }


class Stage1PeerPrice:
    """
    Stage 1: Blocking + KNN peer price statistics.
    
    Uses elasticity methodology:
    1. Block on exact features (market_segment, room_type, children_allowed)
    2. KNN within blocks on continuous features only
    3. Segment-specific geographic features (coastal vs inland)
    """
    
    def __init__(self, k: int = 20):
        self.k = k
        self.data = None
        self.block_knn = {}  # KNN model per block
        self.block_scaler = {}  # Scaler per block
        self.block_data = {}  # Data per block
        self.feature_cols = []
    
    def _get_block_key(self, row: pd.Series) -> str:
        """Creates block key from blocking features."""
        return f"{row.get('market_segment', 'Provincial')}_{row.get('room_type', 'double')}_{row.get('children_allowed', 1)}"
    
    def _get_knn_features(self, market_segment: str) -> List[str]:
        """Gets KNN features based on market segment (segment-specific geo features)."""
        base_features = list(KNN_CONTINUOUS_FEATURES)
        
        # Add segment-specific geographic feature
        if 'Coastal' in market_segment:
            return base_features + GEO_FEATURES_COASTAL
        else:
            return base_features + GEO_FEATURES_INLAND
    
    def fit(self, df: pd.DataFrame) -> 'Stage1PeerPrice':
        """
        Fits KNN models per block.
        
        Args:
            df: DataFrame with hotel features and daily_price
        """
        print("Stage 1: Fitting blocked KNN peer price model...", flush=True)
        
        self.data = df.copy()
        
        # Create block keys
        self.data['_block_key'] = self.data.apply(self._get_block_key, axis=1)
        
        # Fit KNN per block
        block_counts = self.data['_block_key'].value_counts()
        n_blocks = len(block_counts)
        n_valid_blocks = 0
        
        for block_key, block_df in self.data.groupby('_block_key'):
            if len(block_df) < 5:  # Skip tiny blocks
                continue
            
            # Get segment-specific features
            market_segment = block_key.split('_')[0]
            feature_cols = self._get_knn_features(market_segment)
            feature_cols = [c for c in feature_cols if c in block_df.columns]
            
            if len(feature_cols) < 3:  # Need enough features
                continue
            
            # Prepare features
            X = block_df[feature_cols].fillna(0).values
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit KNN (k+1 to account for self-exclusion)
            k_neighbors = min(self.k + 1, len(block_df))
            knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', n_jobs=-1)
            knn.fit(X_scaled)
            
            # Store
            self.block_knn[block_key] = knn
            self.block_scaler[block_key] = scaler
            self.block_data[block_key] = block_df.copy()
            self.feature_cols = feature_cols  # Store last (all should be similar)
            
            n_valid_blocks += 1
        
        print(f"   Fitted {n_valid_blocks:,} blocks (of {n_blocks:,} total)", flush=True)
        print(f"   Total records: {len(self.data):,}", flush=True)
        print(f"   Features per block: {len(self.feature_cols)}", flush=True)
        
        return self
    
    def _get_peer_prices_for_hotel(
        self, 
        hotel_row: pd.Series,
        exclude_hotel_id: Optional[int] = None
    ) -> np.ndarray:
        """Gets peer prices for a single hotel using blocked KNN."""
        block_key = self._get_block_key(hotel_row)
        
        # Check if block exists
        if block_key not in self.block_knn:
            # Fallback to global prices
            return self.data['daily_price'].values
        
        knn = self.block_knn[block_key]
        scaler = self.block_scaler[block_key]
        block_data = self.block_data[block_key]
        
        # Get segment-specific features
        market_segment = block_key.split('_')[0]
        feature_cols = self._get_knn_features(market_segment)
        feature_cols = [c for c in feature_cols if c in hotel_row.index]
        
        if len(feature_cols) < 3:
            return self.data['daily_price'].values
        
        # Prepare query (convert to float to avoid FutureWarning)
        X_query = hotel_row[feature_cols].astype(float).fillna(0).values.reshape(1, -1)
        X_query_scaled = scaler.transform(X_query)
        
        # Find neighbors
        distances, indices = knn.kneighbors(X_query_scaled)
        indices = indices[0]
        
        # Get neighbor data
        neighbor_data = block_data.iloc[indices]
        
        # Exclude target hotel if specified
        if exclude_hotel_id is not None:
            mask = neighbor_data['hotel_id'] != exclude_hotel_id
            neighbor_data = neighbor_data[mask]
        
        return neighbor_data['daily_price'].head(self.k).values
    
    def predict_single(
        self, 
        hotel_features: pd.DataFrame, 
        exclude_hotel_id: Optional[int] = None
    ) -> PeerPriceStats:
        """Gets peer price statistics for a single hotel."""
        row = hotel_features.iloc[0]
        prices = self._get_peer_prices_for_hotel(row, exclude_hotel_id)
        
        if len(prices) == 0:
            prices = self.data['daily_price'].values
        
        return PeerPriceStats(
            mean=float(np.mean(prices)),
            median=float(np.median(prices)),
            std=float(np.std(prices)) if len(prices) > 1 else 0.0,
            count=len(prices),
            min=float(np.min(prices)),
            max=float(np.max(prices))
        )
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gets peer price statistics using blocked KNN.
        
        Optimization: Computes once per unique (hotel_id, block_key) combination.
        """
        n_bookings = len(df)
        n_hotels = df['hotel_id'].nunique()
        print(f"   Computing blocked peer stats: {n_hotels:,} hotels, {n_bookings:,} bookings...", flush=True)
        
        # Add block keys to input data
        df = df.copy()
        df['_block_key'] = df.apply(self._get_block_key, axis=1)
        
        # Get unique hotels with their features
        unique_hotels = df.drop_duplicates('hotel_id').copy()
        
        # Compute peer stats for each unique hotel
        peer_stats = []
        n_in_block = 0
        n_fallback = 0
        
        for _, row in unique_hotels.iterrows():
            hotel_id = row['hotel_id']
            block_key = row['_block_key']
            
            if block_key in self.block_knn:
                prices = self._get_peer_prices_for_hotel(row, exclude_hotel_id=hotel_id)
                n_in_block += 1
            else:
                # Fallback to global
                prices = self.data['daily_price'].values
                n_fallback += 1
            
            if len(prices) == 0:
                prices = self.data['daily_price'].values
            
            peer_stats.append({
                'hotel_id': hotel_id,
                'peer_price_mean': np.mean(prices),
                'peer_price_median': np.median(prices),
                'peer_price_std': np.std(prices) if len(prices) > 1 else 0,
                'peer_price_count': len(prices),
                'peer_price_min': np.min(prices),
                'peer_price_max': np.max(prices)
            })
        
        # Join to all bookings
        hotel_peer_stats = pd.DataFrame(peer_stats)
        df = df.merge(hotel_peer_stats, on='hotel_id', how='left')
        df = df.drop(columns=['_block_key'])
        
        print(f"   Done. ({n_in_block:,} in-block, {n_fallback:,} fallback)", flush=True)
        return df
    
    def get_similar_hotels(self) -> Dict[int, set]:
        """Returns similar hotels mapping for competitor features."""
        similar_hotels = {}
        
        for block_key, block_data in self.block_data.items():
            knn = self.block_knn[block_key]
            scaler = self.block_scaler[block_key]
            
            market_segment = block_key.split('_')[0]
            feature_cols = self._get_knn_features(market_segment)
            feature_cols = [c for c in feature_cols if c in block_data.columns]
            
            if len(feature_cols) < 3:
                continue
            
            X = block_data[feature_cols].fillna(0).values
            X_scaled = scaler.transform(X)
            
            distances, indices = knn.kneighbors(X_scaled)
            hotel_ids = block_data['hotel_id'].values
            
            for i, hotel_id in enumerate(hotel_ids):
                neighbor_idx = indices[i]
                neighbor_ids = hotel_ids[neighbor_idx]
                similar_hotels[hotel_id] = set(neighbor_ids) - {hotel_id}
        
        return similar_hotels


class Stage2TemporalModel:
    """
    Stage 2: Full feature model with temporal adjustments.
    
    Uses all validated features + Stage 1 peer stats + temporal features.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_cols = []
    
    def _prepare_features(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Prepares features for Stage 2 model."""
        df = df.copy()
        
        # Target (only if available - not during inference)
        if 'daily_price' in df.columns:
            y = np.log(df['daily_price'].values)
        else:
            y = None
        
        # Encode categoricals
        cat_cols = ['room_type', 'room_view', 'city_standardized', 'market_segment']
        for col in cat_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str).fillna('unknown'))
                    self.encoders[col] = le
                else:
                    le = self.encoders.get(col)
                    if le:
                        df[col] = df[col].astype(str).fillna('unknown').apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
        
        # Feature columns: all validated + peer stats + temporal + lead time + competitor
        feature_cols = [
            # Hotel features
            'log_room_size', 'room_capacity_pax', 'total_capacity_log',
            'view_quality_ordinal', 'amenities_score',
            'room_type', 'room_view', 'city_standardized', 'market_segment',
            # Geographic features (now included in Stage 2)
            'dist_coast_log', 'dist_center_km', 'is_coastal',
            # Behavioral features
            'weekend_ratio',
            # Booking-specific features
            'stay_nights', 'total_guests', 'rooms_booked',
            # Peer price stats (from Stage 1)
            'peer_price_mean', 'peer_price_median', 'peer_price_std',
            'peer_price_count', 'peer_price_min', 'peer_price_max',
            # Temporal features
            'day_of_week', 'month_sin', 'month_cos',
            'is_weekend', 'is_summer', 'is_winter',
            'is_holiday', 'is_holiday_pm1', 'is_holiday_pm2',
            # Lead time (key dynamic pricing signal!)
            'lead_time_days', 'lead_time_log', 'is_last_minute',
            # Competitor pricing features (latest prices from similar hotels)
            'competitor_avg_price', 'competitor_min_price', 'competitor_max_price',
            'competitor_price_count',
            # Historical pricing features (same date last year)
            'hist_same_hotel_price', 'hist_similar_hotels_price', 'hist_similar_count',
            # Year-over-year market change + implied price
            'competitor_yoy_ratio', 'competitor_yoy_log_change',
            'implied_price', 'implied_price_log',
            # Peer last month features (market rate for similar hotels)
            'peer_last_month_avg', 'peer_last_month_count', 'hotel_last_month_avg',
            # Indicator features (for low-coverage data)
            'has_peer_last_month', 'has_competitor_price', 'has_hist_price',
            # Relative pricing features (hotel vs peer comparison)
            'price_ratio_vs_peer', 'price_ratio_vs_competitor'
        ]
        
        # Only use columns that exist
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        if fit:
            self.feature_cols = feature_cols
        
        X = df[self.feature_cols].fillna(0)
        
        # Scale numeric features
        numeric_cols = [
            'log_room_size', 'room_capacity_pax', 'total_capacity_log',
            'view_quality_ordinal', 'amenities_score',
            # Geographic
            'dist_coast_log', 'dist_center_km',
            # Behavioral
            'weekend_ratio',
            # Booking-specific
            'stay_nights', 'total_guests', 'rooms_booked',
            # Peer stats
            'peer_price_mean', 'peer_price_median', 'peer_price_std',
            'peer_price_count', 'peer_price_min', 'peer_price_max',
            'month_sin', 'month_cos',
            'lead_time_days', 'lead_time_log',
            'competitor_avg_price', 'competitor_min_price', 'competitor_max_price',
            'competitor_price_count',
            'hist_same_hotel_price', 'hist_similar_hotels_price', 'hist_similar_count',
            'competitor_yoy_ratio', 'competitor_yoy_log_change',
            'implied_price', 'implied_price_log',
            'peer_last_month_avg', 'peer_last_month_count', 'hotel_last_month_avg',
            # Relative pricing
            'price_ratio_vs_peer', 'price_ratio_vs_competitor'
        ]
        numeric_cols = [c for c in numeric_cols if c in X.columns]
        
        if fit:
            self.scaler = StandardScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            self.numeric_cols = numeric_cols
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X, y
    
    def fit(self, df: pd.DataFrame) -> 'Stage2TemporalModel':
        """
        Trains Stage 2 model.
        
        Args:
            df: DataFrame with all features including peer stats
        """
        print("Stage 2: Training temporal adjustment model...", flush=True)
        
        X, y = self._prepare_features(df, fit=True)
        
        print(f"   Features: {len(self.feature_cols)}", flush=True)
        print(f"   Training samples: {len(X):,}", flush=True)
        
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts daily prices."""
        X, _ = self._prepare_features(df, fit=False)
        log_prices = self.model.predict(X)
        return np.exp(log_prices)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Returns feature importance dict."""
        return dict(zip(self.feature_cols, self.model.feature_importances_))


class TwoStageModel:
    """
    Combined two-stage pricing model.
    
    Stage 1: KNN peer price statistics
    Stage 2: Full feature model with temporal adjustments + competitor occupancy
    """
    
    def __init__(self, k: int = 50):
        self.stage1 = Stage1PeerPrice(k=k)
        self.stage2 = Stage2TemporalModel()
        self._all_bookings = None  # Store for competitor feature computation
        self._similar_hotels = None  # Cache similar hotels mapping
    
    def _build_similar_hotels(self, df: pd.DataFrame) -> Dict[int, set]:
        """Builds similar hotels mapping from Stage 1 blocked KNN."""
        print("   Building similar hotels mapping from blocked KNN...", flush=True)
        similar_hotels = self.stage1.get_similar_hotels()
        print(f"   Built similarity mapping for {len(similar_hotels):,} hotels", flush=True)
        return similar_hotels
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all pricing features in a single pass:
        - Peer price stats (from Stage 1 KNN)
        - Competitor/historical pricing features
        - Indicator features for data availability
        - Relative pricing features
        
        Uses cached similar_hotels mapping to avoid redundant KNN queries.
        """
        # Add peer stats
        df = self.stage1.predict_batch(df)
        
        # Add competitor features (reusing cached similarity mapping)
        df = add_competitor_features(
            df,
            self.stage1,
            self._all_bookings,
            similar_hotels=self._similar_hotels
        )
        
        # === INDICATOR FEATURES ===
        # These help the model know when peer data is available vs. fallback
        df['has_peer_last_month'] = (df['peer_last_month_avg'] > 0).astype(int)
        df['has_competitor_price'] = (df['competitor_avg_price'] > 0).astype(int)
        df['has_hist_price'] = (df['hist_same_hotel_price'] > 0).astype(int)
        
        # === RELATIVE PRICING FEATURES ===
        # How does this hotel's pricing compare to peers? (avoid division by zero)
        df['price_ratio_vs_peer'] = np.where(
            df['peer_last_month_avg'] > 0,
            df['hotel_last_month_avg'] / df['peer_last_month_avg'],
            1.0  # Default to 1.0 (at market rate) when no peer data
        )
        df['price_ratio_vs_competitor'] = np.where(
            df['competitor_avg_price'] > 0,
            df['hotel_last_month_avg'] / df['competitor_avg_price'],
            1.0
        )
        
        # Clip extreme ratios (0.5x to 2x is reasonable range)
        df['price_ratio_vs_peer'] = df['price_ratio_vs_peer'].clip(0.5, 2.0)
        df['price_ratio_vs_competitor'] = df['price_ratio_vs_competitor'].clip(0.5, 2.0)
        
        return df
    
    def fit(
        self, 
        train_df: pd.DataFrame,
        all_bookings: Optional[pd.DataFrame] = None
    ) -> 'TwoStageModel':
        """
        Trains both stages.
        
        Stage 1 is fit on all data for KNN lookup.
        Competitor features are computed using Stage 1's KNN similarity.
        Stage 2 is trained on data with Stage 1 predictions + competitor features.
        
        Args:
            train_df: Training data for the model
            all_bookings: All bookings data for competitor occupancy calculation.
                         If None, uses train_df. Should include all dates for
                         accurate competitor occupancy.
        """
        # Store all bookings for competitor feature computation
        self._all_bookings = all_bookings if all_bookings is not None else train_df.copy()
        
        # Stage 1: Fit KNN
        self.stage1.fit(train_df)
        
        # Build similar hotels mapping ONCE
        self._similar_hotels = self._build_similar_hotels(self._all_bookings)
        
        # Add all features in one pass
        print("\n3b. Adding all pricing features...", flush=True)
        train_with_features = self.add_all_features(train_df)
        
        # Stage 2: Train on data with peer stats + competitor features
        self.stage2.fit(train_with_features)
        
        return self
    
    def predict(
        self, 
        hotel_features: pd.DataFrame,
        exclude_hotel_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Predicts daily prices for hotels.
        
        Args:
            hotel_features: DataFrame with hotel features including date features
            exclude_hotel_id: Hotel ID to exclude from Stage 1 neighbors
        """
        # Add all features (peer stats + competitor features)
        if self._all_bookings is not None and 'created_at' in hotel_features.columns:
            hotel_features = self.add_all_features(hotel_features)
        else:
            # Fallback for single predictions without booking context
            if len(hotel_features) == 1:
                peer_stats = self.stage1.predict_single(hotel_features, exclude_hotel_id)
                for col, val in peer_stats.to_dict().items():
                    hotel_features[col] = val
            else:
                hotel_features = self.stage1.predict_batch(hotel_features)
            
            # Set default competitor features
            for col, default in [
                ('competitor_avg_price', 0.0), ('competitor_min_price', 0.0),
                ('competitor_max_price', 0.0), ('competitor_price_count', 0),
                ('hist_same_hotel_price', 0.0), ('hist_similar_hotels_price', 0.0),
                ('hist_similar_count', 0), ('competitor_yoy_ratio', 1.0),
                ('competitor_yoy_log_change', 0.0), ('implied_price', 0.0),
                ('implied_price_log', 0.0), ('peer_last_month_avg', 0.0),
                ('peer_last_month_count', 0), ('hotel_last_month_avg', 0.0),
                # Indicator features
                ('has_peer_last_month', 0), ('has_competitor_price', 0), ('has_hist_price', 0),
                # Relative pricing features
                ('price_ratio_vs_peer', 1.0), ('price_ratio_vs_competitor', 1.0)
            ]:
                hotel_features[col] = default
        
        # Stage 2 prediction
        return self.stage2.predict(hotel_features)
    
    def save(self, path: Path = MODEL_DIR) -> None:
        """Saves the model."""
        with open(path / 'two_stage_model.pkl', 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Saved to {path / 'two_stage_model.pkl'}")
    
    @classmethod
    def load(cls, path: Path = MODEL_DIR) -> 'TwoStageModel':
        """Loads the model."""
        with open(path / 'two_stage_model.pkl', 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Training Pipeline
# =============================================================================

def load_daily_data() -> pd.DataFrame:
    """Loads daily booking data."""
    con = init_db()
    
    sql_file = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity/QUERY_LOAD_DAILY_BOOKING_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    df = con.execute(query).fetchdf()
    
    print(f"Loaded {len(df):,} daily bookings")
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds holiday features based on arrival_date and hotel location."""
    df = df.copy()
    
    hotels = df[['hotel_id', 'latitude', 'longitude']].drop_duplicates()
    
    if CITIES500_PATH.exists():
        hotel_admin1 = map_hotels_to_admin1(
            hotels, CITIES500_PATH,
            lat_col='latitude', lon_col='longitude', hotel_id_col='hotel_id'
        )
        print(f"Mapped {len(hotel_admin1)} hotels to Spanish regions")
    else:
        hotel_admin1 = pd.DataFrame({
            'hotel_id': hotels['hotel_id'],
            'subdiv_code': 'MD'
        })
    
    df = add_date_holiday_features(df, hotel_admin1, date_col='arrival_date', hotel_id_col='hotel_id')
    
    print(f"Holiday bookings: {df['is_holiday'].sum():,} ({df['is_holiday'].mean()*100:.1f}%)")
    
    return df


def load_distance_features() -> pd.DataFrame:
    """Loads pre-computed hotel distance features."""
    if DISTANCE_FEATURES_PATH.exists():
        df = pd.read_csv(DISTANCE_FEATURES_PATH)
        print(f"   Loaded distance features for {len(df):,} hotels")
        return df
    else:
        print("   Warning: Distance features file not found")
        return pd.DataFrame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers all features for the model."""
    df = df.copy()
    
    # === OUTLIER CLIPPING ===
    # Clip extreme values before transformations to reduce outlier impact
    if 'avg_room_size' in df.columns:
        df['avg_room_size'] = df['avg_room_size'].clip(5, df['avg_room_size'].quantile(0.99))
    if 'total_capacity' in df.columns:
        df['total_capacity'] = df['total_capacity'].clip(1, df['total_capacity'].quantile(0.99))
    if 'total_guests' in df.columns:
        df['total_guests'] = df['total_guests'].clip(1, 20)  # Max 20 guests per booking
    if 'rooms_booked' in df.columns:
        df['rooms_booked'] = df['rooms_booked'].clip(1, 10)  # Max 10 rooms per booking
    if 'stay_nights' in df.columns:
        df['stay_nights'] = df['stay_nights'].clip(1, 30)  # Max 30 nights
    
    # Load and merge distance features
    distance_df = load_distance_features()
    if len(distance_df) > 0:
        df = df.merge(distance_df, on='hotel_id', how='left')
        # Create derived features
        df['dist_coast_log'] = np.log1p(df['distance_from_coast'].fillna(100))
        df['is_coastal'] = (df['distance_from_coast'] < 20).astype(int)
        # dist_center_km uses Madrid distance as proxy for city center distance
        df['dist_center_km'] = df['distance_from_madrid'].fillna(300)
    
    # Log transforms
    df['log_room_size'] = np.log1p(df['avg_room_size'])
    df['total_capacity_log'] = np.log1p(df['total_capacity'])
    
    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month_number'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_number'] / 12)
    
    # Seasonal flags
    df['is_summer'] = df['month_number'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month_number'].isin([12, 1, 2]).astype(int)
    
    # City standardization
    df['city_standardized'] = df['city'].apply(standardize_city)
    
    # View quality
    df['view_quality_ordinal'] = df['room_view'].map(VIEW_QUALITY_MAP).fillna(0)
    
    # Lead time features (key for dynamic pricing!)
    if 'lead_time_days' in df.columns:
        df['lead_time_days'] = df['lead_time_days'].clip(lower=0)  # No negative lead times
        df['lead_time_log'] = np.log1p(df['lead_time_days'])
        df['is_last_minute'] = (df['lead_time_days'] <= 7).astype(int)
    
    # Market segment (from elasticity methodology)
    # Coastal: within 20km of coast, Madrid metro: within 50km of Madrid
    if 'distance_from_coast' in df.columns and 'distance_from_madrid' in df.columns:
        is_coastal = (df['distance_from_coast'] <= 20).astype(str)
        is_madrid_metro = (df['distance_from_madrid'] <= 50).astype(str)
        df['market_segment'] = is_coastal + '_' + is_madrid_metro
        
        # Map to readable labels
        segment_map = {
            'True_False': 'Coastal',
            'False_True': 'Madrid',
            'False_False': 'Provincial',
            'True_True': 'Coastal'  # Coastal near Madrid still coastal
        }
        df['market_segment'] = df['market_segment'].map(segment_map).fillna('Provincial')
    else:
        df['market_segment'] = 'Provincial'  # Default fallback
    
    # Weekend ratio per hotel (proportion of bookings on weekends)
    # This is a hotel-level behavioral feature
    if 'is_weekend' in df.columns:
        hotel_weekend_ratio = df.groupby('hotel_id')['is_weekend'].mean().reset_index()
        hotel_weekend_ratio.columns = ['hotel_id', 'weekend_ratio']
        df = df.merge(hotel_weekend_ratio, on='hotel_id', how='left')
        df['weekend_ratio'] = df['weekend_ratio'].fillna(0.3)  # Default ~30% weekends
    else:
        df['weekend_ratio'] = 0.3
    
    return df


def train_two_stage_model(sample_size: Optional[int] = None) -> Dict:
    """
    Trains and evaluates the two-stage model.
    
    Args:
        sample_size: If provided, sample training data for faster iteration
    """
    print("=" * 80)
    print("TWO-STAGE PRICING MODEL")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading daily booking data...")
    df = load_daily_data()
    
    # Add holiday features
    print("\n2. Adding holiday features...")
    df = add_holiday_features(df)
    
    # Engineer features
    print("\n3. Engineering features...")
    df = engineer_features(df)
    
    # Core market filter
    df = df[(df['daily_price'] >= 50) & (df['daily_price'] <= 250)]
    print(f"   After core market filter: {len(df):,} bookings")
    
    # Keep full data for competitor occupancy calculation
    all_bookings = df.copy()
    
    # Temporal split
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    train_df = df[df['arrival_date'] < '2024-06-01'].copy().reset_index(drop=True)
    test_df = df[df['arrival_date'] >= '2024-06-01'].copy().reset_index(drop=True)
    
    # Optional sampling for faster iteration
    if sample_size is not None:
        train_df = train_df.sample(min(sample_size, len(train_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(min(sample_size // 4, len(test_df)), random_state=42).reset_index(drop=True)
    
    print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Train model (pass all_bookings for competitor occupancy)
    print("\n4. Training two-stage model...")
    model = TwoStageModel(k=50)
    model.fit(train_df, all_bookings=all_bookings)
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    
    # Add all features using cached similarity mapping
    test_with_features = model.add_all_features(test_df)
    
    # Stage 2 predictions
    predictions = model.stage2.predict(test_with_features)
    actuals = test_df['daily_price'].values
    
    # Metrics
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    mae = np.mean(np.abs(actuals - predictions))
    r2 = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2)
    
    print(f"\n   Test MAPE: {mape:.1f}%")
    print(f"   Test MAE: €{mae:.2f}")
    print(f"   Test R²: {r2:.4f}")
    
    # Feature importance
    print("\n6. Feature Importance (Stage 2):")
    importance = model.stage2.get_feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    
    for feat, imp in sorted_imp[:15]:
        marker = ''
        if 'peer_last_month' in feat or 'hotel_last_month' in feat:
            marker = ' ← PEER LAST MONTH'
        elif 'peer' in feat:
            marker = ' ← PEER STATS'
        elif 'holiday' in feat or feat in ['is_weekend', 'is_summer', 'is_winter', 'day_of_week']:
            marker = ' ← TEMPORAL'
        print(f"   {feat:25s} {imp:.4f}{marker}")
    
    # Holiday analysis
    print("\n7. Holiday Effect Analysis:")
    test_with_features['predicted'] = predictions
    
    hol_actual = test_with_features[test_with_features['is_holiday'] == 1]['daily_price'].mean()
    non_hol_actual = test_with_features[test_with_features['is_holiday'] == 0]['daily_price'].mean()
    hol_pred = test_with_features[test_with_features['is_holiday'] == 1]['predicted'].mean()
    non_hol_pred = test_with_features[test_with_features['is_holiday'] == 0]['predicted'].mean()
    
    print(f"   Actual holiday premium: {(hol_actual/non_hol_actual - 1)*100:+.1f}%")
    print(f"   Predicted holiday premium: {(hol_pred/non_hol_pred - 1)*100:+.1f}%")
    
    # Save model
    model.save()
    
    return {
        'mape': mape,
        'mae': mae,
        'r2': r2,
        'model': model,
        'feature_importance': importance
    }


if __name__ == "__main__":
    import sys
    
    # Use sample for quick testing, full data for final run
    sample_size = 50000 if '--quick' in sys.argv else None
    
    results = train_two_stage_model(sample_size=sample_size)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Two-Stage Model MAPE: {results['mape']:.1f}%")
    print(f"Two-Stage Model MAE: €{results['mae']:.2f}")
    print(f"Two-Stage Model R²: {results['r2']:.4f}")
    
    if sample_size:
        print(f"\n(Used {sample_size:,} sample - run without --quick for full data)")

