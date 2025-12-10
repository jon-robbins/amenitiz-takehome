"""
Rolling Backtest Framework with Cold-Start Evaluation.

Key design:
1. Hold out 20% of hotel_ids entirely (never seen during training)
2. Use rolling time windows for temporal validation
3. Compare ML models against naive baselines
4. Report metrics separately for:
   - Warm hotels (seen in training)
   - Cold hotels (never seen in training)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.evaluation.time_backtest import (
    BacktestConfig,
    load_hotel_week_data,
    simulate_revpar,
    get_adjusted_elasticity,
    DEFAULT_ELASTICITY,
)
from src.features.engineering import get_market_segment


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RollingBacktestConfig:
    """Configuration for rolling backtest."""
    # Hotel holdout
    hotel_holdout_pct: float = 0.20  # 20% of hotels held out for cold-start
    
    # Time windows
    train_months: int = 6  # Months of training data
    test_months: int = 1   # Months of test data
    
    # Date range (based on data availability)
    data_start: date = field(default_factory=lambda: date(2023, 1, 1))
    data_end: date = field(default_factory=lambda: date(2024, 9, 30))
    
    # Random seed for reproducibility
    random_state: int = 42
    
    def get_time_windows(self) -> List[Tuple[date, date, date, date]]:
        """
        Generate all rolling time windows.
        
        Returns list of (train_start, train_end, test_start, test_end)
        """
        windows = []
        
        # First possible test start = data_start + train_months
        test_start = self.data_start + relativedelta(months=self.train_months)
        
        while test_start + relativedelta(months=self.test_months) <= self.data_end:
            train_start = test_start - relativedelta(months=self.train_months)
            train_end = test_start - timedelta(days=1)
            test_end = test_start + relativedelta(months=self.test_months) - timedelta(days=1)
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # Roll forward by 1 month
            test_start = test_start + relativedelta(months=1)
        
        return windows


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class WindowMetrics:
    """Metrics for a single time window."""
    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    
    # Overall metrics
    n_predictions: int
    win_rate: float
    mean_revpar_lift: float
    median_revpar_lift: float
    price_mape: float
    
    # Cold-start metrics
    n_cold: int
    cold_win_rate: float
    cold_mean_lift: float
    
    # Warm hotel metrics
    n_warm: int
    warm_win_rate: float
    warm_mean_lift: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all windows."""
    n_windows: int
    
    # Overall
    mean_win_rate: float
    std_win_rate: float
    mean_revpar_lift: float
    std_revpar_lift: float
    
    # Cold-start
    mean_cold_win_rate: float
    std_cold_win_rate: float
    mean_cold_lift: float
    
    # Warm
    mean_warm_win_rate: float
    std_warm_win_rate: float
    mean_warm_lift: float
    
    # Per-window details
    window_metrics: List[WindowMetrics] = field(default_factory=list)


# =============================================================================
# NAIVE BASELINES (Vectorized for Speed)
# =============================================================================

def baseline_random_vectorized(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Random price within market P25-P75 (vectorized)."""
    p25 = train_df['avg_price'].quantile(0.25)
    p75 = train_df['avg_price'].quantile(0.75)
    return np.random.uniform(p25, p75, size=len(test_df))

baseline_random_vectorized.__vectorized__ = True


def baseline_market_average_vectorized(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Market-wide average price (vectorized)."""
    return np.full(len(test_df), train_df['avg_price'].mean())

baseline_market_average_vectorized.__vectorized__ = True


def baseline_self_median_vectorized(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Hotel's own historical median price (vectorized)."""
    # Compute median per hotel from training data
    hotel_medians = train_df.groupby('hotel_id')['avg_price'].median()
    market_median = train_df['avg_price'].median()
    
    # Map to test hotels, fallback to market median for cold-start
    return test_df['hotel_id'].map(hotel_medians).fillna(market_median).values

baseline_self_median_vectorized.__vectorized__ = True


def baseline_peer_median_vectorized(train_df: pd.DataFrame, test_df: pd.DataFrame, radius_km: float = 10.0) -> np.ndarray:
    """Geographic peer median price (vectorized using spatial binning)."""
    # Create spatial grid for fast lookup
    lat_bin_size = radius_km / 111.0  # degrees
    lon_bin_size = radius_km / 85.0   # approximate degrees at mid-latitude
    
    train_df = train_df.copy()
    train_df['lat_bin'] = (train_df['latitude'] / lat_bin_size).astype(int)
    train_df['lon_bin'] = (train_df['longitude'] / lon_bin_size).astype(int)
    
    # Compute median per spatial bin
    bin_medians = train_df.groupby(['lat_bin', 'lon_bin'])['avg_price'].median()
    market_median = train_df['avg_price'].median()
    
    # Map test hotels to bins
    test_df = test_df.copy()
    test_df['lat_bin'] = (test_df['latitude'] / lat_bin_size).astype(int)
    test_df['lon_bin'] = (test_df['longitude'] / lon_bin_size).astype(int)
    
    # Lookup with fallback
    result = []
    for _, row in test_df.iterrows():
        key = (row['lat_bin'], row['lon_bin'])
        if key in bin_medians.index:
            result.append(bin_medians[key])
        else:
            # Check neighboring bins
            found = False
            for dlat in [-1, 0, 1]:
                for dlon in [-1, 0, 1]:
                    neighbor_key = (row['lat_bin'] + dlat, row['lon_bin'] + dlon)
                    if neighbor_key in bin_medians.index:
                        result.append(bin_medians[neighbor_key])
                        found = True
                        break
                if found:
                    break
            if not found:
                result.append(market_median)
    
    return np.array(result)

baseline_peer_median_vectorized.__vectorized__ = True


# Legacy non-vectorized versions (for compatibility)
def baseline_random(train_df, row, target_date):
    p25 = train_df['avg_price'].quantile(0.25)
    p75 = train_df['avg_price'].quantile(0.75)
    return np.random.uniform(p25, p75)

def baseline_market_average(train_df, row, target_date):
    return train_df['avg_price'].mean()

def baseline_self_median(train_df, row, target_date):
    hotel_data = train_df[train_df['hotel_id'] == row['hotel_id']]
    if len(hotel_data) > 0:
        return hotel_data['avg_price'].median()
    return train_df['avg_price'].median()

def baseline_peer_median(train_df, row, target_date):
    return train_df['avg_price'].median()  # Simplified fallback


# =============================================================================
# HOTEL SPLIT
# =============================================================================

def split_hotels(
    df: pd.DataFrame,
    holdout_pct: float = 0.20,
    random_state: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split hotel_ids into train and holdout sets.
    
    Args:
        df: DataFrame with hotel_id column
        holdout_pct: Fraction to hold out (cold-start test)
        random_state: Random seed
    
    Returns:
        Tuple of (train_hotel_ids, holdout_hotel_ids)
    """
    all_hotels = df['hotel_id'].unique()
    
    train_hotels, holdout_hotels = train_test_split(
        all_hotels,
        test_size=holdout_pct,
        random_state=random_state
    )
    
    return list(train_hotels), list(holdout_hotels)


# =============================================================================
# SINGLE WINDOW EVALUATION
# =============================================================================

def evaluate_window_vectorized(
    recommender_fn: Callable,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    warm_hotels: List[int],
    cold_hotels: List[int],
    config: RollingBacktestConfig,
    window_id: int,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    elasticity: float = -0.39,
    verbose: bool = False
) -> WindowMetrics:
    """
    Evaluate a single time window (vectorized for speed).
    """
    test_df = test_df.copy()
    
    # Mark cold hotels
    cold_set = set(cold_hotels)
    test_df['is_cold'] = test_df['hotel_id'].isin(cold_set)
    
    # Get recommendations - vectorized if possible
    if hasattr(recommender_fn, '__vectorized__'):
        # Use vectorized version
        test_df['rec_price'] = recommender_fn(train_df, test_df)
    else:
        # Batch processing with apply
        target_dates = pd.to_datetime(test_df['week_start']).dt.date
        test_df['rec_price'] = test_df.apply(
            lambda row: recommender_fn(train_df, row, target_dates[row.name]),
            axis=1
        )
    
    # Vectorized RevPAR simulation with segment-adjusted elasticity
    actual_price = test_df['avg_price'].values
    actual_occ = test_df['occupancy_rate'].values
    rec_price = test_df['rec_price'].values
    
    # Get market segment for each hotel (for elasticity adjustment)
    if 'market_segment' not in test_df.columns:
        if 'dist_coast_km' in test_df.columns and 'dist_madrid_km' in test_df.columns:
            test_df['market_segment'] = test_df.apply(
                lambda r: get_market_segment(r.get('dist_coast_km', 100), r.get('dist_madrid_km', 300)),
                axis=1
            )
        else:
            test_df['market_segment'] = 'provincial'  # Default
    
    # Vectorized elasticity - adjusted per segment and occupancy
    # Segment multipliers: coastal=1.15, madrid_metro=0.77, provincial=1.0
    segment_mult = test_df['market_segment'].map({
        'coastal': 1.15,
        'madrid_metro': 0.77,
        'provincial': 1.0,
        'unknown': 1.0
    }).fillna(1.0).values
    
    # Occupancy adjustment: high occ = less elastic, low occ = more elastic
    occ_factor = np.where(
        actual_occ >= 0.6,
        1.0 - 0.5 * ((actual_occ - 0.6) / 0.4),  # Scale from 1.0 at 60% to 0.5 at 100%
        1.0 + 0.3 * ((0.6 - actual_occ) / 0.4)   # Scale from 1.0 at 60% to 1.3 at 20%
    )
    occ_factor = np.clip(occ_factor, 0.5, 1.3)
    
    # Combined elasticity
    adjusted_elasticity = DEFAULT_ELASTICITY * segment_mult * occ_factor
    
    # Calculate new occupancy
    pct_change = (rec_price - actual_price) / np.maximum(actual_price, 1)
    new_occ = actual_occ * (1 + adjusted_elasticity * pct_change)
    new_occ = np.clip(new_occ, 0.01, 0.99)
    
    test_df['actual_revpar'] = actual_price * actual_occ
    test_df['rec_revpar'] = rec_price * new_occ
    test_df['revpar_lift'] = test_df['rec_revpar'] - test_df['actual_revpar']
    test_df['is_win'] = test_df['revpar_lift'] > 0
    test_df['price_error'] = np.abs(rec_price - actual_price) / np.maximum(actual_price, 1)
    
    # Filter out any failed predictions
    results_df = test_df[test_df['rec_price'].notna()].copy()
    
    if len(results_df) == 0:
        return WindowMetrics(
            window_id=window_id,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            n_predictions=0, win_rate=0, mean_revpar_lift=0,
            median_revpar_lift=0, price_mape=0,
            n_cold=0, cold_win_rate=0, cold_mean_lift=0,
            n_warm=0, warm_win_rate=0, warm_mean_lift=0
        )
    
    # Overall metrics
    n_predictions = len(results_df)
    win_rate = results_df['is_win'].mean()
    mean_lift = results_df['revpar_lift'].mean()
    median_lift = results_df['revpar_lift'].median()
    price_mape = results_df['price_error'].mean()
    
    # Cold-start metrics
    cold_df = results_df[results_df['is_cold']]
    n_cold = len(cold_df)
    cold_win_rate = cold_df['is_win'].mean() if n_cold > 0 else 0
    cold_mean_lift = cold_df['revpar_lift'].mean() if n_cold > 0 else 0
    
    # Warm metrics
    warm_df = results_df[~results_df['is_cold']]
    n_warm = len(warm_df)
    warm_win_rate = warm_df['is_win'].mean() if n_warm > 0 else 0
    warm_mean_lift = warm_df['revpar_lift'].mean() if n_warm > 0 else 0
    
    return WindowMetrics(
        window_id=window_id,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        n_predictions=n_predictions,
        win_rate=win_rate,
        mean_revpar_lift=mean_lift,
        median_revpar_lift=median_lift,
        price_mape=price_mape,
        n_cold=n_cold, cold_win_rate=cold_win_rate, cold_mean_lift=cold_mean_lift,
        n_warm=n_warm, warm_win_rate=warm_win_rate, warm_mean_lift=warm_mean_lift
    )


# =============================================================================
# FULL ROLLING BACKTEST
# =============================================================================

def run_rolling_backtest(
    recommender_fn: Callable,
    full_df: pd.DataFrame,
    config: RollingBacktestConfig,
    verbose: bool = True
) -> AggregateMetrics:
    """
    Run full rolling backtest with hotel holdout.
    
    Args:
        recommender_fn: Function(train_df, row, date) -> recommended_price
        full_df: Complete dataset
        config: Backtest configuration
        verbose: Print progress
    
    Returns:
        AggregateMetrics with per-window and aggregate results
    """
    # Split hotels once (same split across all windows)
    warm_hotels, cold_hotels = split_hotels(
        full_df, 
        holdout_pct=config.hotel_holdout_pct,
        random_state=config.random_state
    )
    
    if verbose:
        print(f"Hotel split: {len(warm_hotels)} warm, {len(cold_hotels)} cold ({config.hotel_holdout_pct:.0%} holdout)")
    
    # Get time windows
    windows = config.get_time_windows()
    
    if verbose:
        print(f"Time windows: {len(windows)}")
    
    # Evaluate each window
    window_metrics = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        if verbose:
            print(f"\nWindow {i+1}/{len(windows)}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
        
        # Filter data for this window
        train_df = full_df[
            (full_df['week_start'] >= pd.Timestamp(train_start)) &
            (full_df['week_start'] <= pd.Timestamp(train_end)) &
            (full_df['hotel_id'].isin(warm_hotels))  # Only warm hotels in training!
        ].copy()
        
        test_df = full_df[
            (full_df['week_start'] >= pd.Timestamp(test_start)) &
            (full_df['week_start'] <= pd.Timestamp(test_end))
            # Test includes BOTH warm and cold hotels
        ].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            if verbose:
                print(f"  Skipping - insufficient data")
            continue
        
        if verbose:
            n_cold_test = len(test_df[test_df['hotel_id'].isin(cold_hotels)])
            print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows ({n_cold_test} cold)")
        
        # Evaluate (use vectorized version for speed)
        metrics = evaluate_window_vectorized(
            recommender_fn, train_df, test_df,
            warm_hotels, cold_hotels, config,
            window_id=i+1,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            verbose=verbose
        )
        
        window_metrics.append(metrics)
        
        if verbose:
            print(f"  Win rate: {metrics.win_rate:.1%} (warm: {metrics.warm_win_rate:.1%}, cold: {metrics.cold_win_rate:.1%})")
    
    # Aggregate metrics
    if len(window_metrics) == 0:
        return AggregateMetrics(
            n_windows=0,
            mean_win_rate=0, std_win_rate=0,
            mean_revpar_lift=0, std_revpar_lift=0,
            mean_cold_win_rate=0, std_cold_win_rate=0, mean_cold_lift=0,
            mean_warm_win_rate=0, std_warm_win_rate=0, mean_warm_lift=0,
            window_metrics=[]
        )
    
    win_rates = [m.win_rate for m in window_metrics]
    lifts = [m.mean_revpar_lift for m in window_metrics]
    cold_win_rates = [m.cold_win_rate for m in window_metrics if m.n_cold > 0]
    cold_lifts = [m.cold_mean_lift for m in window_metrics if m.n_cold > 0]
    warm_win_rates = [m.warm_win_rate for m in window_metrics if m.n_warm > 0]
    warm_lifts = [m.warm_mean_lift for m in window_metrics if m.n_warm > 0]
    
    return AggregateMetrics(
        n_windows=len(window_metrics),
        mean_win_rate=np.mean(win_rates),
        std_win_rate=np.std(win_rates),
        mean_revpar_lift=np.mean(lifts),
        std_revpar_lift=np.std(lifts),
        mean_cold_win_rate=np.mean(cold_win_rates) if cold_win_rates else 0,
        std_cold_win_rate=np.std(cold_win_rates) if cold_win_rates else 0,
        mean_cold_lift=np.mean(cold_lifts) if cold_lifts else 0,
        mean_warm_win_rate=np.mean(warm_win_rates) if warm_win_rates else 0,
        std_warm_win_rate=np.std(warm_win_rates) if warm_win_rates else 0,
        mean_warm_lift=np.mean(warm_lifts) if warm_lifts else 0,
        window_metrics=window_metrics
    )


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(
    models: Dict[str, Callable],
    full_df: pd.DataFrame,
    config: RollingBacktestConfig,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models on rolling backtest.
    
    Args:
        models: Dict mapping model name to recommender function
        full_df: Complete dataset
        config: Backtest configuration
        verbose: Print progress
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, recommender_fn in models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print('='*60)
        
        metrics = run_rolling_backtest(recommender_fn, full_df, config, verbose=verbose)
        
        results.append({
            'model': name,
            'n_windows': metrics.n_windows,
            'win_rate': metrics.mean_win_rate,
            'win_rate_std': metrics.std_win_rate,
            'mean_lift': metrics.mean_revpar_lift,
            'lift_std': metrics.std_revpar_lift,
            'cold_win_rate': metrics.mean_cold_win_rate,
            'cold_lift': metrics.mean_cold_lift,
            'warm_win_rate': metrics.mean_warm_win_rate,
            'warm_lift': metrics.mean_warm_lift,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test rolling backtest framework."""
    print("=" * 60)
    print("ROLLING BACKTEST TEST")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    base_config = BacktestConfig()
    full_df = load_hotel_week_data(base_config, split='all')
    
    print(f"Loaded {len(full_df):,} records")
    print(f"Hotels: {full_df['hotel_id'].nunique()}")
    print(f"Date range: {full_df['week_start'].min()} to {full_df['week_start'].max()}")
    
    # Configure backtest
    config = RollingBacktestConfig(
        hotel_holdout_pct=0.20,
        train_months=6,
        test_months=1,
        data_start=date(2023, 5, 1),  # Based on actual data
        data_end=date(2024, 9, 30),
    )
    
    windows = config.get_time_windows()
    print(f"\nTime windows: {len(windows)}")
    for i, (ts, te, vs, ve) in enumerate(windows[:3]):
        print(f"  {i+1}: Train {ts} to {te}, Test {vs} to {ve}")
    if len(windows) > 3:
        print(f"  ... and {len(windows)-3} more")
    
    # Test with naive baselines
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON (Quick Test)")
    print("=" * 60)
    
    # Use subset for speed
    subset_df = full_df.sample(frac=0.3, random_state=42)
    
    baselines = {
        'Random': baseline_random,
        'Market Avg': baseline_market_average,
        'Self Median': baseline_self_median,
        'Peer Median': baseline_peer_median,
    }
    
    results = compare_models(baselines, subset_df, config, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

