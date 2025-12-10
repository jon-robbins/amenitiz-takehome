"""
Time-based backtesting evaluation framework for price recommendations.

This module provides:
1. Time-based train/test split (realistic deployment simulation)
2. RevPAR simulation engine using validated elasticity
3. Comprehensive evaluation metrics

Split strategy:
- Train: Jun 2023 - Jun 2024 (13 months)
- Test: Jul 2024 - Sep 2024 (3 months)

Key metrics:
1. Net RevPAR Lift: Simulated RevPAR - Actual RevPAR
2. Win Rate: % of recommendations with positive lift
3. Price MAPE: vs actual charged
4. Peer Band Accuracy: % within ±20% of peer median
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from src.data.loader import get_clean_connection, get_project_root
from src.features.engineering import MARKET_ELASTICITY


# =============================================================================
# CONSTANTS
# =============================================================================

# Time split boundaries
TRAIN_START = date(2023, 6, 1)
TRAIN_END = date(2024, 6, 30)
TEST_START = date(2024, 7, 1)
TEST_END = date(2024, 9, 30)

# Default elasticity (validated from matched pairs analysis)
DEFAULT_ELASTICITY = MARKET_ELASTICITY  # -0.39

# Minimum data requirements
MIN_WEEKS_FOR_SELF_ANCHOR = 8
MIN_PEERS_FOR_PEER_ANCHOR = 5


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    train_start: date = TRAIN_START
    train_end: date = TRAIN_END
    test_start: date = TEST_START
    test_end: date = TEST_END
    elasticity: float = DEFAULT_ELASTICITY
    min_weeks_history: int = MIN_WEEKS_FOR_SELF_ANCHOR
    min_peers: int = MIN_PEERS_FOR_PEER_ANCHOR
    peer_radius_km: float = 10.0


@dataclass
class HotelWeekRecord:
    """A single hotel-week observation for backtesting."""
    hotel_id: int
    week_start: date
    actual_price: float
    actual_occupancy: float
    actual_revpar: float
    peer_median_price: Optional[float]
    n_peers: int
    city: str
    latitude: float
    longitude: float
    room_type: str
    total_rooms: int


@dataclass
class RecommendationResult:
    """Result of a price recommendation with simulated outcome."""
    hotel_id: int
    week_start: date
    actual_price: float
    actual_occupancy: float
    actual_revpar: float
    recommended_price: float
    simulated_occupancy: float
    simulated_revpar: float
    revpar_lift: float
    revpar_lift_pct: float
    price_change_pct: float
    peer_median_price: Optional[float]
    within_peer_band: bool
    recommendation_path: str  # "path_a", "path_b", "fallback"


@dataclass
class BacktestMetrics:
    """Aggregated metrics from backtesting."""
    # Core metrics
    n_recommendations: int
    win_rate: float  # % with positive RevPAR lift
    mean_revpar_lift: float  # Average RevPAR lift (€)
    mean_revpar_lift_pct: float  # Average RevPAR lift (%)
    total_revpar_lift: float  # Sum of all RevPAR lifts
    
    # Price accuracy
    price_mae: float  # Mean Absolute Error vs actual
    price_mape: float  # Mean Absolute Percentage Error
    
    # Peer alignment
    peer_band_accuracy: float  # % within ±20% of peer median
    mean_price_vs_peer: float  # Avg recommended price / peer median
    
    # Path breakdown
    n_path_a: int
    n_path_b: int
    n_fallback: int
    win_rate_path_a: float
    win_rate_path_b: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'n_recommendations': self.n_recommendations,
            'win_rate': self.win_rate,
            'mean_revpar_lift': self.mean_revpar_lift,
            'mean_revpar_lift_pct': self.mean_revpar_lift_pct,
            'total_revpar_lift': self.total_revpar_lift,
            'price_mae': self.price_mae,
            'price_mape': self.price_mape,
            'peer_band_accuracy': self.peer_band_accuracy,
            'mean_price_vs_peer': self.mean_price_vs_peer,
            'n_path_a': self.n_path_a,
            'n_path_b': self.n_path_b,
            'n_fallback': self.n_fallback,
            'win_rate_path_a': self.win_rate_path_a,
            'win_rate_path_b': self.win_rate_path_b,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return f"""
Backtest Results ({self.n_recommendations:,} recommendations)
{'='*50}
Core Metrics:
  Win Rate:           {self.win_rate:.1%}
  Mean RevPAR Lift:   €{self.mean_revpar_lift:.2f} ({self.mean_revpar_lift_pct:+.1%})
  Total RevPAR Lift:  €{self.total_revpar_lift:,.0f}

Price Accuracy:
  MAE:                €{self.price_mae:.2f}
  MAPE:               {self.price_mape:.1%}

Peer Alignment:
  Within ±20% Band:   {self.peer_band_accuracy:.1%}
  Mean vs Peer:       {self.mean_price_vs_peer:.1%}

Path Breakdown:
  Path A (History):   {self.n_path_a:,} ({self.n_path_a/self.n_recommendations:.1%}) - Win: {self.win_rate_path_a:.1%}
  Path B (Cold):      {self.n_path_b:,} ({self.n_path_b/self.n_recommendations:.1%}) - Win: {self.win_rate_path_b:.1%}
  Fallback:           {self.n_fallback:,} ({self.n_fallback/self.n_recommendations:.1%})
"""


# =============================================================================
# REVPAR SIMULATION ENGINE
# =============================================================================

def get_adjusted_elasticity(
    base_elasticity: float,
    occupancy: float,
    segment: Optional[str] = None
) -> float:
    """
    Adjust elasticity based on occupancy level and segment.
    
    Key insight: High-occupancy hotels have proven demand and are LESS
    price-elastic. Low-occupancy hotels are MORE price-elastic.
    
    Adjustment logic:
    - At 90%+ occupancy: elasticity reduced by 50% (can raise prices)
    - At 50-70% occupancy: use base elasticity
    - At <30% occupancy: elasticity increased by 30% (price-sensitive)
    
    Segment adjustments:
    - Coastal (leisure): slightly more elastic (-0.45 base)
    - Urban/business: less elastic (-0.30 base)
    - Provincial: base elasticity (-0.39)
    
    Args:
        base_elasticity: Market-level elasticity (e.g., -0.39)
        occupancy: Current occupancy rate (0-1)
        segment: Optional segment ("coastal", "urban", "provincial")
    
    Returns:
        Adjusted elasticity value
    """
    # Segment adjustment
    segment_multipliers = {
        'coastal': 1.15,    # More elastic (leisure travelers)
        'urban': 0.77,      # Less elastic (business travelers)
        'madrid_metro': 0.77,
        'provincial': 1.0,  # Base
    }
    segment_mult = segment_multipliers.get(segment, 1.0)
    
    # Occupancy adjustment: high occupancy = less elastic
    if occupancy >= 0.90:
        # Very high demand - can raise prices with minimal volume loss
        occ_mult = 0.5
    elif occupancy >= 0.80:
        occ_mult = 0.7
    elif occupancy >= 0.60:
        occ_mult = 1.0  # Base
    elif occupancy >= 0.40:
        occ_mult = 1.15
    else:
        # Low occupancy - very price sensitive
        occ_mult = 1.30
    
    adjusted = base_elasticity * segment_mult * occ_mult
    
    # Bound to reasonable range
    return np.clip(adjusted, -0.8, -0.15)


def simulate_revpar(
    actual_price: float,
    actual_occupancy: float,
    recommended_price: float,
    elasticity: float = DEFAULT_ELASTICITY,
    segment: Optional[str] = None,
    adjust_for_occupancy: bool = True
) -> Tuple[float, float]:
    """
    Simulate RevPAR at recommended price using elasticity.
    
    Formula:
        pct_change = (rec_price - actual_price) / actual_price
        new_occupancy = actual_occupancy * (1 + elasticity * pct_change)
        simulated_revpar = rec_price * new_occupancy
    
    Key improvement: Elasticity is adjusted based on current occupancy.
    - 90%+ occupancy: Less elastic (proven demand, can raise prices)
    - <40% occupancy: More elastic (price-sensitive)
    
    Args:
        actual_price: Price actually charged
        actual_occupancy: Occupancy achieved (0-1)
        recommended_price: Recommended price to simulate
        elasticity: Base price elasticity of demand (default -0.39)
        segment: Optional segment for elasticity adjustment
        adjust_for_occupancy: Whether to adjust elasticity for occupancy level
    
    Returns:
        Tuple of (simulated_occupancy, simulated_revpar)
    """
    if actual_price <= 0:
        return actual_occupancy, recommended_price * actual_occupancy
    
    # Adjust elasticity based on occupancy and segment
    if adjust_for_occupancy:
        effective_elasticity = get_adjusted_elasticity(
            elasticity, actual_occupancy, segment
        )
    else:
        effective_elasticity = elasticity
    
    pct_change = (recommended_price - actual_price) / actual_price
    
    # Apply elasticity: occupancy changes inversely to price
    new_occupancy = actual_occupancy * (1 + effective_elasticity * pct_change)
    new_occupancy = np.clip(new_occupancy, 0.01, 0.99)  # Realistic bounds
    
    simulated_revpar = recommended_price * new_occupancy
    
    return new_occupancy, simulated_revpar


def calculate_revpar_lift(
    actual_price: float,
    actual_occupancy: float,
    recommended_price: float,
    elasticity: float = DEFAULT_ELASTICITY,
    segment: Optional[str] = None,
    adjust_for_occupancy: bool = True
) -> Tuple[float, float, float, float]:
    """
    Calculate RevPAR lift from a price recommendation.
    
    Args:
        actual_price: Price actually charged
        actual_occupancy: Occupancy achieved (0-1)
        recommended_price: Recommended price
        elasticity: Base price elasticity of demand
        segment: Optional segment for elasticity adjustment
        adjust_for_occupancy: Whether to adjust elasticity for occupancy level
    
    Returns:
        Tuple of (simulated_occ, simulated_revpar, revpar_lift, revpar_lift_pct)
    """
    actual_revpar = actual_price * actual_occupancy
    sim_occ, sim_revpar = simulate_revpar(
        actual_price, actual_occupancy, recommended_price, elasticity,
        segment=segment, adjust_for_occupancy=adjust_for_occupancy
    )
    
    revpar_lift = sim_revpar - actual_revpar
    revpar_lift_pct = revpar_lift / actual_revpar if actual_revpar > 0 else 0
    
    return sim_occ, sim_revpar, revpar_lift, revpar_lift_pct


# =============================================================================
# DATA LOADING FOR BACKTEST
# =============================================================================

def load_hotel_week_data(
    config: BacktestConfig,
    split: str = "all"
) -> pd.DataFrame:
    """
    Load hotel-week aggregated data for backtesting.
    
    Args:
        config: BacktestConfig with date boundaries
        split: "train", "test", or "all"
    
    Returns:
        DataFrame with hotel-week records including:
        - hotel_id, week_start, week_of_year, month
        - avg_price (actual), occupancy_rate, revpar
        - peer_median_price, n_peers
        - city, latitude, longitude, room_type, total_rooms
    """
    con = get_clean_connection()
    
    # Determine date filter
    if split == "train":
        start_date = config.train_start
        end_date = config.train_end
    elif split == "test":
        start_date = config.test_start
        end_date = config.test_end
    else:
        start_date = config.train_start
        end_date = config.test_end
    
    query = f"""
    WITH daily_stays AS (
        -- Explode bookings to daily granularity
        SELECT 
            b.hotel_id,
            CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) as stay_date,
            br.total_price / NULLIF(DATEDIFF('day', b.arrival_date, b.departure_date), 0) as nightly_rate,
            br.room_type,
            br.room_size,
            hl.city,
            hl.latitude,
            hl.longitude
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        CROSS JOIN generate_series(0, 30) as n(n)
        WHERE b.status IN ('Booked', 'confirmed')
          AND b.arrival_date >= '{start_date.isoformat()}'
          AND b.departure_date <= '{(end_date + timedelta(days=7)).isoformat()}'
          AND CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) < b.departure_date
          AND CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) >= '{start_date.isoformat()}'
          AND CAST(b.arrival_date + (n.n * INTERVAL '1 day') AS DATE) <= '{end_date.isoformat()}'
          AND hl.latitude IS NOT NULL
          AND br.total_price > 0
    ),
    -- Get actual hotel capacity from rooms table (sum of all room types)
    hotel_room_types AS (
        SELECT DISTINCT
            b.hotel_id,
            CAST(br.room_id AS BIGINT) as room_id
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        WHERE b.status IN ('Booked', 'confirmed')
    ),
    hotel_capacity AS (
        SELECT 
            hrt.hotel_id,
            SUM(COALESCE(r.number_of_rooms, 1)) as total_rooms
        FROM hotel_room_types hrt
        LEFT JOIN rooms r ON hrt.room_id = r.id
        GROUP BY hrt.hotel_id
    ),
    hotel_week_stats AS (
        SELECT 
            ds.hotel_id,
            DATE_TRUNC('week', ds.stay_date) as week_start,
            AVG(ds.nightly_rate) as avg_price,
            COUNT(*) as rooms_sold,
            MODE() WITHIN GROUP (ORDER BY ds.room_type) as room_type,
            AVG(ds.room_size) as avg_room_size,
            MAX(ds.city) as city,
            MAX(ds.latitude) as latitude,
            MAX(ds.longitude) as longitude
        FROM daily_stays ds
        WHERE ds.nightly_rate BETWEEN 20 AND 500
        GROUP BY ds.hotel_id, DATE_TRUNC('week', ds.stay_date)
        HAVING COUNT(*) >= 3  -- Minimum 3 room-nights per week
    )
    SELECT 
        hws.*,
        EXTRACT(WEEK FROM hws.week_start) as week_of_year,
        EXTRACT(MONTH FROM hws.week_start) as month,
        COALESCE(hc.total_rooms, 10) as total_rooms,
        -- Occupancy = rooms_sold / (total_rooms * 7 days), capped at 100%
        LEAST(hws.rooms_sold::FLOAT / NULLIF(COALESCE(hc.total_rooms, 10) * 7, 0), 1.0) as occupancy_rate
    FROM hotel_week_stats hws
    LEFT JOIN hotel_capacity hc ON hws.hotel_id = hc.hotel_id
    WHERE hws.avg_price BETWEEN 30 AND 400
    ORDER BY hws.hotel_id, hws.week_start
    """
    
    df = con.execute(query).fetchdf()
    
    # Calculate RevPAR
    df['revpar'] = df['avg_price'] * df['occupancy_rate']
    
    # Add peer median prices
    df = _add_peer_prices(df, config.peer_radius_km)
    
    print(f"Loaded {len(df):,} hotel-week records for {df['hotel_id'].nunique():,} hotels")
    print(f"  Date range: {df['week_start'].min()} to {df['week_start'].max()}")
    
    return df


def _add_peer_prices(df: pd.DataFrame, radius_km: float = 10.0) -> pd.DataFrame:
    """
    Add peer median price for each hotel-week observation.
    
    For each row, finds hotels within radius_km and calculates
    their median price for the same week.
    """
    df = df.copy()
    
    # Pre-compute for efficiency: group by week
    peer_prices = []
    
    for week in df['week_start'].unique():
        week_df = df[df['week_start'] == week].copy()
        
        for idx, row in week_df.iterrows():
            # Calculate distances to all other hotels this week
            lat_km = 111.0
            lon_km = 111.0 * np.cos(np.radians(row['latitude']))
            
            distances = np.sqrt(
                ((week_df['latitude'] - row['latitude']) * lat_km) ** 2 +
                ((week_df['longitude'] - row['longitude']) * lon_km) ** 2
            )
            
            # Find peers within radius (excluding self)
            peers_mask = (distances <= radius_km) & (distances > 0)
            peers = week_df[peers_mask]
            
            if len(peers) >= 1:
                peer_median = peers['avg_price'].median()
                n_peers = len(peers)
            else:
                peer_median = np.nan
                n_peers = 0
            
            peer_prices.append({
                'idx': idx,
                'peer_median_price': peer_median,
                'n_peers': n_peers
            })
    
    peer_df = pd.DataFrame(peer_prices).set_index('idx')
    df['peer_median_price'] = peer_df['peer_median_price']
    df['n_peers'] = peer_df['n_peers'].fillna(0).astype(int)
    
    return df


def get_hotel_history_weeks(df: pd.DataFrame, hotel_id: int, as_of_week: date) -> int:
    """
    Count weeks of history for a hotel prior to a given week.
    
    Args:
        df: Full dataset
        hotel_id: Hotel to check
        as_of_week: Reference week (exclusive)
    
    Returns:
        Number of weeks with data before as_of_week
    """
    hotel_df = df[
        (df['hotel_id'] == hotel_id) & 
        (df['week_start'] < pd.Timestamp(as_of_week))
    ]
    return len(hotel_df)


def get_train_test_split(
    config: BacktestConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split data into train and test sets.
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = load_hotel_week_data(config, split="train")
    test_df = load_hotel_week_data(config, split="test")
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df):,} records, {train_df['hotel_id'].nunique():,} hotels")
    print(f"  Test:  {len(test_df):,} records, {test_df['hotel_id'].nunique():,} hotels")
    
    return train_df, test_df


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

def evaluate_recommendations(
    results: List[RecommendationResult]
) -> BacktestMetrics:
    """
    Calculate comprehensive metrics from recommendation results.
    
    Args:
        results: List of RecommendationResult from backtesting
    
    Returns:
        BacktestMetrics with all evaluation metrics
    """
    if len(results) == 0:
        return BacktestMetrics(
            n_recommendations=0,
            win_rate=0.0,
            mean_revpar_lift=0.0,
            mean_revpar_lift_pct=0.0,
            total_revpar_lift=0.0,
            price_mae=0.0,
            price_mape=0.0,
            peer_band_accuracy=0.0,
            mean_price_vs_peer=0.0,
            n_path_a=0,
            n_path_b=0,
            n_fallback=0,
            win_rate_path_a=0.0,
            win_rate_path_b=0.0,
        )
    
    n = len(results)
    
    # Core metrics
    wins = sum(1 for r in results if r.revpar_lift > 0)
    win_rate = wins / n
    
    mean_lift = np.mean([r.revpar_lift for r in results])
    mean_lift_pct = np.mean([r.revpar_lift_pct for r in results])
    total_lift = sum(r.revpar_lift for r in results)
    
    # Price accuracy
    price_errors = [abs(r.recommended_price - r.actual_price) for r in results]
    price_mae = np.mean(price_errors)
    
    price_pct_errors = [
        abs(r.recommended_price - r.actual_price) / r.actual_price 
        for r in results if r.actual_price > 0
    ]
    price_mape = np.mean(price_pct_errors) if price_pct_errors else 0.0
    
    # Peer alignment
    results_with_peers = [r for r in results if r.peer_median_price is not None]
    peer_band_accuracy = (
        np.mean([r.within_peer_band for r in results_with_peers])
        if results_with_peers else 0.0
    )
    
    price_vs_peer = [
        r.recommended_price / r.peer_median_price 
        for r in results_with_peers if r.peer_median_price > 0
    ]
    mean_price_vs_peer = np.mean(price_vs_peer) if price_vs_peer else 1.0
    
    # Path breakdown
    path_a = [r for r in results if r.recommendation_path == "path_a"]
    path_b = [r for r in results if r.recommendation_path == "path_b"]
    fallback = [r for r in results if r.recommendation_path == "fallback"]
    
    win_rate_a = np.mean([r.revpar_lift > 0 for r in path_a]) if path_a else 0.0
    win_rate_b = np.mean([r.revpar_lift > 0 for r in path_b]) if path_b else 0.0
    
    return BacktestMetrics(
        n_recommendations=n,
        win_rate=win_rate,
        mean_revpar_lift=mean_lift,
        mean_revpar_lift_pct=mean_lift_pct,
        total_revpar_lift=total_lift,
        price_mae=price_mae,
        price_mape=price_mape,
        peer_band_accuracy=peer_band_accuracy,
        mean_price_vs_peer=mean_price_vs_peer,
        n_path_a=len(path_a),
        n_path_b=len(path_b),
        n_fallback=len(fallback),
        win_rate_path_a=win_rate_a,
        win_rate_path_b=win_rate_b,
    )


def run_backtest(
    recommender_fn: Callable[[pd.DataFrame, pd.DataFrame, date], float],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BacktestConfig,
    verbose: bool = True
) -> Tuple[List[RecommendationResult], BacktestMetrics]:
    """
    Run full backtest with a recommender function.
    
    Args:
        recommender_fn: Function that takes (train_df, hotel_row, target_week) 
                       and returns recommended_price
        train_df: Training data
        test_df: Test data to evaluate on
        config: BacktestConfig
        verbose: Print progress
    
    Returns:
        Tuple of (list of RecommendationResult, BacktestMetrics)
    """
    results = []
    
    for idx, row in test_df.iterrows():
        hotel_id = row['hotel_id']
        week_start = row['week_start'].date() if hasattr(row['week_start'], 'date') else row['week_start']
        
        # Determine path based on history
        weeks_history = get_hotel_history_weeks(train_df, hotel_id, week_start)
        
        if weeks_history >= config.min_weeks_history:
            path = "path_a"
        elif row['n_peers'] >= config.min_peers:
            path = "path_b"
        else:
            path = "fallback"
        
        # Get recommendation
        try:
            recommended_price = recommender_fn(train_df, row, week_start)
        except Exception as e:
            if verbose:
                print(f"Error for hotel {hotel_id}: {e}")
            continue
        
        # Simulate outcome
        sim_occ, sim_revpar, lift, lift_pct = calculate_revpar_lift(
            row['avg_price'],
            row['occupancy_rate'],
            recommended_price,
            config.elasticity
        )
        
        # Check peer band
        within_band = False
        if pd.notna(row['peer_median_price']) and row['peer_median_price'] > 0:
            ratio = recommended_price / row['peer_median_price']
            within_band = 0.8 <= ratio <= 1.2
        
        result = RecommendationResult(
            hotel_id=hotel_id,
            week_start=week_start,
            actual_price=row['avg_price'],
            actual_occupancy=row['occupancy_rate'],
            actual_revpar=row['revpar'],
            recommended_price=recommended_price,
            simulated_occupancy=sim_occ,
            simulated_revpar=sim_revpar,
            revpar_lift=lift,
            revpar_lift_pct=lift_pct,
            price_change_pct=(recommended_price - row['avg_price']) / row['avg_price'],
            peer_median_price=row['peer_median_price'] if pd.notna(row['peer_median_price']) else None,
            within_peer_band=within_band,
            recommendation_path=path
        )
        results.append(result)
    
    metrics = evaluate_recommendations(results)
    
    if verbose:
        print(metrics)
    
    return results, metrics


# =============================================================================
# BASELINE RECOMMENDERS (for comparison)
# =============================================================================

def baseline_actual_price(train_df: pd.DataFrame, row: pd.Series, week: date) -> float:
    """Baseline: Just recommend the actual price (sanity check - should give 0 lift)."""
    return row['avg_price']


def baseline_peer_median(train_df: pd.DataFrame, row: pd.Series, week: date) -> float:
    """Baseline: Recommend peer median price."""
    if pd.notna(row['peer_median_price']) and row['peer_median_price'] > 0:
        return row['peer_median_price']
    return row['avg_price']


def baseline_self_median(train_df: pd.DataFrame, row: pd.Series, week: date) -> float:
    """Baseline: Recommend hotel's own historical median."""
    hotel_history = train_df[train_df['hotel_id'] == row['hotel_id']]
    if len(hotel_history) >= 4:
        return hotel_history['avg_price'].median()
    return row['avg_price']


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run baseline evaluation to validate framework."""
    print("=" * 60)
    print("TIME-BASED BACKTEST EVALUATION FRAMEWORK")
    print("=" * 60)
    
    config = BacktestConfig()
    
    print(f"\nConfiguration:")
    print(f"  Train: {config.train_start} to {config.train_end}")
    print(f"  Test:  {config.test_start} to {config.test_end}")
    print(f"  Elasticity: {config.elasticity}")
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = get_train_test_split(config)
    
    # Run baseline evaluations
    print("\n" + "=" * 60)
    print("BASELINE: Actual Price (Sanity Check)")
    print("=" * 60)
    _, metrics_actual = run_backtest(
        baseline_actual_price, train_df, test_df, config
    )
    
    print("\n" + "=" * 60)
    print("BASELINE: Peer Median")
    print("=" * 60)
    _, metrics_peer = run_backtest(
        baseline_peer_median, train_df, test_df, config
    )
    
    print("\n" + "=" * 60)
    print("BASELINE: Self Median")
    print("=" * 60)
    _, metrics_self = run_backtest(
        baseline_self_median, train_df, test_df, config
    )
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"{'Baseline':<20} {'Win Rate':>10} {'Mean Lift':>12} {'MAPE':>10}")
    print("-" * 55)
    print(f"{'Actual Price':<20} {metrics_actual.win_rate:>10.1%} {metrics_actual.mean_revpar_lift:>10.2f}€ {metrics_actual.price_mape:>10.1%}")
    print(f"{'Peer Median':<20} {metrics_peer.win_rate:>10.1%} {metrics_peer.mean_revpar_lift:>10.2f}€ {metrics_peer.price_mape:>10.1%}")
    print(f"{'Self Median':<20} {metrics_self.win_rate:>10.1%} {metrics_self.mean_revpar_lift:>10.2f}€ {metrics_self.price_mape:>10.1%}")


if __name__ == "__main__":
    main()

