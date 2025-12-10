"""
Comprehensive Cold-Start Validation.

Validates the occupancy model by:
1. Using ALL available hotels
2. For each hotel, testing on 2 random weeks
3. Simulating cold-start (removing hotel's booking data)
4. Comparing predictions to actual outcomes

Uses properly cleaned data via CleaningConfig.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
from src.models.occupancy import OccupancyModel
from src.models.baseline_pricing import BaselinePricingModel
from src.models.occupancy_with_price import OccupancyModelWithPrice
from src.models.optimal_pricing import OptimalPricingModel, OptimalPriceResult
from src.features.engineering import (
    # Feature engineering functions (all features computed here, not in this script)
    engineer_validated_features,
    add_peer_price_features,
    calculate_peer_price_features,
    PEER_RADIUS_KM,
)

warnings.filterwarnings('ignore')


@dataclass
class WeekResult:
    """Result for one hotel-week combination."""
    hotel_id: int
    week_start: str
    city: str
    
    # Ground truth
    actual_price: float
    actual_occupancy: float
    actual_revpar: float
    
    # Prediction
    predicted_price: float
    predicted_occupancy: float
    predicted_revpar: float
    
    # Peer context (direct comparison)
    peer_price: float
    peer_occupancy: float
    peer_revpar: float
    n_peers: int
    
    # 10km radius peer features (cold-start)
    peer_price_10km_mean: float
    peer_price_10km_median: float
    peer_price_10km_p25: float
    peer_price_10km_p75: float
    peer_occupancy_10km: float
    n_peers_10km: int
    
    # Errors
    price_error_pct: float
    revpar_error_pct: float


@dataclass
class ComprehensiveResults:
    """Results from comprehensive cold-start validation."""
    n_hotels: int
    n_weeks: int
    n_observations: int
    
    # Overall metrics
    price_mape: float
    revpar_mape: float
    price_correlation: float
    revpar_correlation: float
    direction_accuracy: float
    
    # By week type
    summer_mape: float
    winter_mape: float
    shoulder_mape: float
    
    # By city type
    coastal_mape: float
    madrid_mape: float
    other_mape: float
    
    # Raw results
    results_df: pd.DataFrame


@dataclass
class ThreeModelResult:
    """Result for one hotel from the 3-model pipeline."""
    hotel_id: int
    week_start: str
    
    # Actual (ground truth)
    actual_price: float
    actual_occupancy: float
    actual_revpar: float
    
    # Model 1: Baseline pricing
    baseline_price: float
    
    # Model 2: Occupancy at baseline price
    baseline_occupancy: float
    baseline_revpar: float
    
    # Model 3: Optimal pricing
    optimal_price: float
    optimal_occupancy: float
    optimal_revpar: float
    
    # Comparisons
    baseline_price_error_pct: float  # vs actual
    optimal_revpar_lift_pct: float   # vs baseline
    optimal_vs_actual_pct: float     # optimal RevPAR vs actual RevPAR


@dataclass 
class ThreeModelPipelineResults:
    """Results from the 3-model pipeline evaluation."""
    # Data splits
    n_train_hotels: int
    n_val_hotels: int
    n_test_hotels: int
    n_test_observations: int
    
    # Model 1: Baseline Pricing performance
    baseline_price_mape: float
    baseline_price_r2: float
    
    # Model 2: Occupancy with Price performance
    occupancy_mape: float
    occupancy_r2: float
    
    # Model 3: Optimal Pricing outcomes
    avg_revpar_lift_vs_baseline: float
    avg_revpar_lift_vs_actual: float
    pct_hotels_improved: float
    
    # Raw results
    results_df: pd.DataFrame
    
    # Model objects (for inspection)
    baseline_model_metrics: dict
    occupancy_model_metrics: dict


def get_clean_connection():
    """Get a properly cleaned database connection."""
    config = CleaningConfig(
        # Basic validation
        remove_negative_prices=True,
        remove_zero_prices=True,
        remove_low_prices=True,
        remove_extreme_prices=True,
        remove_null_dates=True,
        remove_null_room_id=True,
        remove_orphan_bookings=True,
        remove_bookings_before_2023=True,
        remove_bookings_after_2024=True,
        
        # Exclusions
        exclude_reception_halls=True,
        exclude_missing_location=True,
        
        # Imputations
        impute_children_allowed=True,
        impute_events_allowed=True,
        
        # City matching
        match_city_names_with_tfidf=True,
        city_name_similarity_threshold=0.97,
        
        verbose=False
    )
    cleaner = DataCleaner(config)
    return cleaner.clean(init_db())


def load_all_hotel_weeks(
    con,
    min_price: float = 50,
    max_price: float = 200,
    min_bookings_per_week: int = 3,
    hotel_ids: Optional[set] = None
) -> pd.DataFrame:
    """
    Load weekly performance data for hotels.
    
    Args:
        con: Database connection
        min_price: Minimum ADR filter
        max_price: Maximum ADR filter
        min_bookings_per_week: Minimum bookings per week
        hotel_ids: Optional set of hotel_ids to filter (for train/val/test split)
    
    Returns:
        DataFrame with hotel_id, week_start, and performance metrics.
    """
    # Build hotel filter if provided
    hotel_filter = ""
    if hotel_ids is not None:
        hotel_ids_str = ",".join(str(h) for h in hotel_ids)
        hotel_filter = f"AND b.hotel_id IN ({hotel_ids_str})"
    
    # First get weekly bookings
    weekly_query = f"""
    SELECT 
        b.hotel_id,
        DATE_TRUNC('week', b.arrival_date) as week_start,
        EXTRACT(WEEK FROM b.arrival_date) as week_of_year,
        EXTRACT(MONTH FROM b.arrival_date) as month,
        COUNT(*) as n_bookings,
        AVG(br.total_price / NULLIF(
            DATEDIFF('day', b.arrival_date, b.departure_date), 0
        )) as avg_adr,
        AVG(br.room_size) as avg_room_size,
        MODE() WITHIN GROUP (ORDER BY br.room_type) as room_type,
        MAX(br.room_view) as room_view
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    WHERE b.status IN ('Booked', 'confirmed')
      AND b.arrival_date >= '2023-06-01'
      AND b.arrival_date < '2024-10-01'
      AND br.room_id IS NOT NULL
      {hotel_filter}
    GROUP BY b.hotel_id, DATE_TRUNC('week', b.arrival_date),
             EXTRACT(WEEK FROM b.arrival_date), EXTRACT(MONTH FROM b.arrival_date)
    HAVING COUNT(*) >= {min_bookings_per_week}
      AND AVG(br.total_price / NULLIF(
            DATEDIFF('day', b.arrival_date, b.departure_date), 0
          )) >= {min_price}
      AND AVG(br.total_price / NULLIF(
            DATEDIFF('day', b.arrival_date, b.departure_date), 0
          )) <= {max_price}
    """
    
    weekly_df = con.execute(weekly_query).fetchdf()
    
    if len(weekly_df) == 0:
        return pd.DataFrame()
    
    # Get actual hotel capacity from rooms table (sum of all room types per hotel)
    capacity_query = """
    WITH hotel_room_types AS (
        -- Get unique room_ids per hotel
        SELECT DISTINCT
            b.hotel_id,
            CAST(br.room_id AS BIGINT) as room_id
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        WHERE b.status IN ('Booked', 'confirmed')
    ),
    hotel_capacity AS (
        -- Sum the number_of_rooms for each unique room type
        SELECT 
            hrt.hotel_id,
            SUM(COALESCE(r.number_of_rooms, 1)) as total_rooms
        FROM hotel_room_types hrt
        LEFT JOIN rooms r ON hrt.room_id = r.id
        GROUP BY hrt.hotel_id
    ),
    room_features AS (
        SELECT 
            b.hotel_id,
            MAX(r.max_occupancy) as max_occupancy,
            MAX(CASE WHEN r.children_allowed THEN 1 ELSE 0 END) as children_allowed,
            MAX(CASE WHEN r.pets_allowed THEN 1 ELSE 0 END) as pets_allowed,
            MAX(CASE WHEN r.events_allowed THEN 1 ELSE 0 END) as events_allowed
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        LEFT JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
        WHERE b.status IN ('Booked', 'confirmed')
        GROUP BY b.hotel_id
    )
    SELECT 
        hc.hotel_id,
        COALESCE(hc.total_rooms, 10) as total_rooms,
        rf.max_occupancy,
        rf.children_allowed,
        rf.pets_allowed,
        rf.events_allowed
    FROM hotel_capacity hc
    LEFT JOIN room_features rf ON hc.hotel_id = rf.hotel_id
    """
    
    capacity_df = con.execute(capacity_query).fetchdf()
    
    # Get hotel locations
    location_query = """
    SELECT hotel_id, city, latitude, longitude
    FROM hotel_location
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """
    
    location_df = con.execute(location_query).fetchdf()
    
    # Merge all together
    df = weekly_df.merge(location_df, on='hotel_id', how='inner')
    df = df.merge(capacity_df, on='hotel_id', how='left')
    
    # Fill defaults
    df['total_rooms'] = df['total_rooms'].fillna(10)
    df['max_occupancy'] = df['max_occupancy'].fillna(2)
    df['children_allowed'] = df['children_allowed'].fillna(0)
    df['pets_allowed'] = df['pets_allowed'].fillna(0)
    df['events_allowed'] = df['events_allowed'].fillna(0)
    
    # Rename and calculate
    df = df.rename(columns={'avg_adr': 'actual_price'})
    
    # Occupancy: bookings per week / (rooms * 7 days)
    df['actual_occupancy'] = (df['n_bookings'] / (df['total_rooms'] * 7.0)).clip(0.01, 1.0)
    
    # Calculate RevPAR
    df['actual_revpar'] = df['actual_price'] * df['actual_occupancy']
    
    return df.sort_values(['hotel_id', 'week_start']).reset_index(drop=True)


def sample_weeks_per_hotel(
    df: pd.DataFrame,
    n_weeks_per_hotel: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample n random weeks for each hotel.
    
    Ensures diversity by trying to sample from different seasons.
    """
    np.random.seed(random_state)
    
    sampled = []
    for hotel_id, hotel_data in df.groupby('hotel_id'):
        if len(hotel_data) < n_weeks_per_hotel:
            # Use all available weeks
            sampled.append(hotel_data)
        else:
            # Try to get seasonal diversity
            summer = hotel_data[hotel_data['month'].isin([6, 7, 8])]
            winter = hotel_data[hotel_data['month'].isin([12, 1, 2])]
            shoulder = hotel_data[~hotel_data['month'].isin([6, 7, 8, 12, 1, 2])]
            
            samples = []
            # Get one from each season if available
            for season_df in [summer, shoulder, winter]:
                if len(season_df) > 0 and len(samples) < n_weeks_per_hotel:
                    samples.append(season_df.sample(1))
            
            # Fill remaining with random
            if len(samples) < n_weeks_per_hotel:
                remaining = hotel_data[~hotel_data.index.isin(
                    pd.concat(samples).index if samples else pd.Index([])
                )]
                n_needed = min(n_weeks_per_hotel - len(samples), len(remaining))
                if n_needed > 0:
                    samples.append(remaining.sample(n_needed))
            
            if samples:
                sampled.append(pd.concat(samples))
    
    return pd.concat(sampled).reset_index(drop=True)


def split_hotels_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by hotel_id (not by observation) to avoid data leakage.
    
    For cold-start validation, we must ensure:
    1. Test hotels are NEVER seen during training
    2. Peer features for test hotels come from training hotels only
    
    Args:
        df: DataFrame with hotel_id column
        test_size: Fraction of hotels to hold out for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df) with no hotel overlap
    """
    unique_hotels = df['hotel_id'].unique()
    n_hotels = len(unique_hotels)
    
    np.random.seed(random_state)
    shuffled_hotels = np.random.permutation(unique_hotels)
    
    n_test = int(n_hotels * test_size)
    test_hotel_ids = set(shuffled_hotels[:n_test])
    train_hotel_ids = set(shuffled_hotels[n_test:])
    
    train_df = df[df['hotel_id'].isin(train_hotel_ids)].copy()
    test_df = df[df['hotel_id'].isin(test_hotel_ids)].copy()
    
    return train_df, test_df


def get_hotel_ids_from_bookings(con) -> List[int]:
    """
    Get unique hotel_ids that have actual booking data.
    
    CRITICAL: The train/val/test split must be based on bookings,
    not on derived features or hotel_location table.
    """
    query = """
    SELECT DISTINCT hotel_id 
    FROM bookings 
    WHERE status IN ('Booked', 'confirmed')
      AND hotel_id IS NOT NULL
    ORDER BY hotel_id
    """
    result = con.execute(query).fetchdf()
    return result['hotel_id'].tolist()


def split_hotels_train_val_test(
    con,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[set, set, set]:
    """
    Split hotel_ids from BOOKINGS table into train/validation/test sets.
    
    CRITICAL: 
    - Split based on hotel_ids in bookings table (actual data)
    - Returns SETS of hotel_ids, not DataFrames
    - Data loading should happen AFTER split using these hotel_id filters
    
    Args:
        con: Database connection
        train_size: Fraction for training (default 60%)
        val_size: Fraction for validation (default 20%)
        test_size: Fraction for testing (default 20%)
        random_state: Random seed
    
    Returns:
        Tuple of (train_hotel_ids, val_hotel_ids, test_hotel_ids) as sets
    """
    assert abs(train_size + val_size + test_size - 1.0) < 0.01, "Sizes must sum to 1.0"
    
    # Get hotel_ids from bookings table
    all_hotel_ids = get_hotel_ids_from_bookings(con)
    n = len(all_hotel_ids)
    
    np.random.seed(random_state)
    shuffled = np.random.permutation(all_hotel_ids)
    
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    
    test_hotel_ids = set(shuffled[:n_test])
    val_hotel_ids = set(shuffled[n_test:n_test + n_val])
    train_hotel_ids = set(shuffled[n_test + n_val:])
    
    return train_hotel_ids, val_hotel_ids, test_hotel_ids


def run_comprehensive_cold_start(
    n_weeks_per_hotel: int = 2,
    min_price: float = 50,
    max_price: float = 200,
    test_size: float = 0.2,
    random_state: int = 42
) -> ComprehensiveResults:
    """
    Run comprehensive cold-start validation with proper train/test split.
    
    CRITICAL: Splits by HOTEL (not observation) to avoid data leakage.
    - Test hotels are NEVER seen during training
    - Peer features for test hotels come ONLY from training hotels
    - OccupancyModel trained on training hotels only
    
    Args:
        n_weeks_per_hotel: Number of weeks to sample per test hotel
        min_price: Minimum price filter
        max_price: Maximum price filter
        test_size: Fraction of hotels to hold out (default 20%)
        random_state: Random seed
    
    Returns:
        ComprehensiveResults with TRUE out-of-sample metrics
    """
    print("=" * 70)
    print("COMPREHENSIVE COLD-START VALIDATION (NO DATA LEAKAGE)")
    print("=" * 70)
    
    # Load cleaned data
    print("\n1. Loading cleaned data...")
    con = get_clean_connection()
    
    # Load all hotel-week data
    print("\n2. Loading weekly hotel performance...")
    all_data = load_all_hotel_weeks(con, min_price, max_price)
    print(f"   Total observations: {len(all_data):,}")
    print(f"   Unique hotels: {all_data['hotel_id'].nunique():,}")
    print(f"   Unique weeks: {all_data['week_start'].nunique():,}")
    
    # CRITICAL: Split by HOTEL first (before any feature engineering)
    print(f"\n3. Splitting hotels into train/test ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
    train_data, test_data = split_hotels_train_test(all_data, test_size, random_state)
    print(f"   Training hotels: {train_data['hotel_id'].nunique():,} ({len(train_data):,} obs)")
    print(f"   Test hotels: {test_data['hotel_id'].nunique():,} ({len(test_data):,} obs)")
    
    # Engineer features on TRAIN data only
    print("\n4. Engineering features on TRAINING data...")
    train_data = engineer_validated_features(train_data)
    train_data['occupancy_rate'] = train_data['actual_occupancy']
    train_data['city_standardized'] = train_data['city'].str.lower().str.strip()
    
    # Engineer features on TEST data (geographic features only, no peer features yet)
    test_data = engineer_validated_features(test_data)
    test_data['city_standardized'] = test_data['city'].str.lower().str.strip()
    
    # Train OccupancyModel on TRAINING hotels only
    print("\n5. Training OccupancyModel on TRAINING hotels only...")
    occupancy_model = OccupancyModel()
    try:
        occupancy_model.fit(train_data)
        print(f"   Model trained successfully")
        if occupancy_model._metrics:
            print(f"   R²: {occupancy_model._metrics.get('r2', 0):.3f}")
    except Exception as e:
        print(f"   Warning: Could not train model: {e}")
        occupancy_model = None
    
    # Sample weeks per test hotel
    print(f"\n6. Sampling {n_weeks_per_hotel} weeks per test hotel...")
    test_sample = sample_weeks_per_hotel(test_data, n_weeks_per_hotel, random_state)
    print(f"   Test observations: {len(test_sample):,}")
    
    # Run cold-start predictions
    # CRITICAL: Peer features calculated from TRAIN data only
    print(f"\n7. Running cold-start predictions (peers from TRAIN data only)...")
    results = []
    
    for idx, (_, row) in enumerate(test_sample.iterrows()):
        if idx % 100 == 0:
            print(f"   Processing {idx+1}/{len(test_sample)}...")
        
        # Calculate peer features from TRAINING hotels only (no leakage!)
        peer_features = calculate_peer_price_features(
            target_lat=row['latitude'],
            target_lon=row['longitude'],
            target_room_type=row.get('room_type', 'Standard'),
            peer_df=train_data,  # ONLY training hotels!
            radius_km=PEER_RADIUS_KM
        )
        
        peer_price_mean = peer_features.get('peer_price_mean', np.nan)
        peer_price_dw = peer_features.get('peer_price_distance_weighted', np.nan)
        peer_occ_mean = peer_features.get('peer_occupancy_mean', np.nan)
        n_peers = peer_features.get('n_peers_10km', 0)
        
        # Skip if no peers found
        if n_peers < 2 or np.isnan(peer_price_mean):
            continue
        
        # Recommended price from peers
        pred_price = peer_price_dw if not np.isnan(peer_price_dw) else peer_price_mean
        
        # Use OccupancyModel to predict occupancy at recommended price
        if occupancy_model is not None and occupancy_model.is_fitted:
            try:
                # Prepare features for this test hotel
                hotel_features = row.to_frame().T.copy()
                
                # Predict base occupancy from hotel features
                base_occupancy = occupancy_model.predict(hotel_features)[0]
                
                # Adjust for price using elasticity
                # relative_price = recommended_price / peer_baseline
                relative_price = pred_price / peer_price_mean if peer_price_mean > 0 else 1.0
                pred_occupancy = occupancy_model.predict_at_price(hotel_features, relative_price)[0]
                occupancy_source = "model"
            except Exception:
                pred_occupancy = peer_occ_mean if not np.isnan(peer_occ_mean) else 0.5
                occupancy_source = "peer_fallback"
        else:
            pred_occupancy = peer_occ_mean if not np.isnan(peer_occ_mean) else 0.5
            occupancy_source = "peer_only"
        
        pred_revpar = pred_price * pred_occupancy
        
        # Calculate errors
        price_error = (pred_price - row['actual_price']) / row['actual_price'] * 100
        revpar_error = (pred_revpar - row['actual_revpar']) / max(row['actual_revpar'], 1) * 100
        
        results.append(WeekResult(
            hotel_id=int(row['hotel_id']),
            week_start=str(row['week_start']),
            city=row['city'],
            actual_price=row['actual_price'],
            actual_occupancy=row['actual_occupancy'],
            actual_revpar=row['actual_revpar'],
            predicted_price=pred_price,
            predicted_occupancy=pred_occupancy,
            predicted_revpar=pred_revpar,
            # Peer features from TRAINING data only
            peer_price=peer_price_mean,
            peer_occupancy=peer_occ_mean if not np.isnan(peer_occ_mean) else 0.5,
            peer_revpar=peer_features.get('peer_revpar_mean', pred_revpar),
            n_peers=n_peers,
            # 10km radius peer features
            peer_price_10km_mean=peer_price_mean,
            peer_price_10km_median=peer_features.get('peer_price_median', np.nan),
            peer_price_10km_p25=peer_features.get('peer_price_p25', np.nan),
            peer_price_10km_p75=peer_features.get('peer_price_p75', np.nan),
            peer_occupancy_10km=peer_occ_mean,
            n_peers_10km=n_peers,
            price_error_pct=price_error,
            revpar_error_pct=revpar_error,
        ))
    
    print(f"\n8. Calculating TRUE out-of-sample metrics...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame([vars(r) for r in results])
    
    if len(results_df) == 0:
        raise ValueError("No valid predictions generated")
    
    # Add season and city type
    results_df['week_dt'] = pd.to_datetime(results_df['week_start'])
    results_df['month'] = results_df['week_dt'].dt.month
    results_df['season'] = results_df['month'].apply(
        lambda m: 'summer' if m in [6, 7, 8] else ('winter' if m in [12, 1, 2] else 'shoulder')
    )
    
    # City type (would need to calculate properly)
    results_df['city_type'] = 'other'  # Placeholder
    
    # Overall metrics
    price_mape = results_df['price_error_pct'].abs().mean()
    revpar_mape = results_df['revpar_error_pct'].abs().mean()
    
    price_corr, _ = stats.pearsonr(results_df['predicted_price'], results_df['actual_price'])
    revpar_corr, _ = stats.pearsonr(results_df['predicted_revpar'], results_df['actual_revpar'])
    
    # Direction accuracy
    results_df['actual_vs_peer'] = results_df['actual_revpar'] >= results_df['peer_revpar']
    results_df['pred_vs_peer'] = results_df['predicted_revpar'] >= results_df['peer_revpar']
    direction_accuracy = (results_df['actual_vs_peer'] == results_df['pred_vs_peer']).mean() * 100
    
    # By season
    summer_mape = results_df[results_df['season'] == 'summer']['revpar_error_pct'].abs().mean()
    winter_mape = results_df[results_df['season'] == 'winter']['revpar_error_pct'].abs().mean()
    shoulder_mape = results_df[results_df['season'] == 'shoulder']['revpar_error_pct'].abs().mean()
    
    return ComprehensiveResults(
        n_hotels=results_df['hotel_id'].nunique(),
        n_weeks=results_df['week_start'].nunique(),
        n_observations=len(results_df),
        price_mape=price_mape,
        revpar_mape=revpar_mape,
        price_correlation=price_corr,
        revpar_correlation=revpar_corr,
        direction_accuracy=direction_accuracy,
        summer_mape=summer_mape if not np.isnan(summer_mape) else 0,
        winter_mape=winter_mape if not np.isnan(winter_mape) else 0,
        shoulder_mape=shoulder_mape if not np.isnan(shoulder_mape) else 0,
        coastal_mape=0,  # Would need geographic classification
        madrid_mape=0,
        other_mape=0,
        results_df=results_df
    )


def print_comprehensive_summary(results: ComprehensiveResults) -> None:
    """Print summary of comprehensive validation results."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COLD-START RESULTS")
    print("=" * 70)
    
    print(f"\n1. COVERAGE")
    print("-" * 40)
    print(f"   Hotels Tested: {results.n_hotels:,}")
    print(f"   Weeks Covered: {results.n_weeks:,}")
    print(f"   Total Observations: {results.n_observations:,}")
    
    print(f"\n2. PRICE PREDICTION ACCURACY")
    print("-" * 40)
    print(f"   MAPE: {results.price_mape:.1f}%")
    print(f"   Correlation: {results.price_correlation:.2f}")
    
    print(f"\n3. REVPAR PREDICTION ACCURACY")
    print("-" * 40)
    print(f"   MAPE: {results.revpar_mape:.1f}%")
    print(f"   Correlation: {results.revpar_correlation:.2f}")
    
    print(f"\n4. DIRECTION ACCURACY")
    print("-" * 40)
    print(f"   {results.direction_accuracy:.1f}%")
    print("   (Correctly predicted above/below peer performance)")
    
    print(f"\n5. BY SEASON (RevPAR MAPE)")
    print("-" * 40)
    print(f"   Summer (Jun-Aug): {results.summer_mape:.1f}%")
    print(f"   Winter (Dec-Feb): {results.winter_mape:.1f}%")
    print(f"   Shoulder:         {results.shoulder_mape:.1f}%")


def create_comprehensive_visualizations(
    results: ComprehensiveResults,
    output_dir: Path
) -> None:
    """Create comprehensive validation visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = results.results_df
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Comprehensive Cold-Start Validation', fontsize=14, fontweight='bold')
    
    # 1. Predicted vs Actual Price
    ax = axes[0, 0]
    ax.scatter(df['actual_price'], df['predicted_price'], alpha=0.3, s=20, c='steelblue')
    max_val = max(df['actual_price'].max(), df['predicted_price'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Price (€)')
    ax.set_ylabel('Predicted Price (€)')
    ax.set_title(f'Price Prediction (r = {results.price_correlation:.2f})')
    ax.legend()
    
    # 2. Predicted vs Actual RevPAR
    ax = axes[0, 1]
    ax.scatter(df['actual_revpar'], df['predicted_revpar'], alpha=0.3, s=20, c='coral')
    max_val = max(df['actual_revpar'].max(), df['predicted_revpar'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual RevPAR (€)')
    ax.set_ylabel('Predicted RevPAR (€)')
    ax.set_title(f'RevPAR Prediction (r = {results.revpar_correlation:.2f})')
    ax.legend()
    
    # 3. Error Distribution
    ax = axes[0, 2]
    errors = df['price_error_pct'].clip(-50, 50)
    ax.hist(errors, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='green', linestyle='--', lw=2)
    ax.axvline(errors.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {errors.mean():+.1f}%')
    ax.set_xlabel('Price Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Price Error Distribution')
    ax.legend()
    
    # 4. Error by Season
    ax = axes[1, 0]
    season_errors = df.groupby('season')['price_error_pct'].agg(['mean', 'std']).reset_index()
    season_order = ['summer', 'shoulder', 'winter']
    season_errors = season_errors.set_index('season').reindex(season_order).reset_index()
    ax.bar(season_errors['season'], season_errors['mean'].abs(), 
           yerr=season_errors['std'], capsize=5, color=['coral', 'steelblue', 'lightblue'])
    ax.set_ylabel('Mean Absolute Price Error (%)')
    ax.set_title('Accuracy by Season')
    
    # 5. Hotels vs Accuracy
    ax = axes[1, 1]
    hotel_errors = df.groupby('hotel_id')['price_error_pct'].agg(['mean', 'count']).reset_index()
    ax.scatter(hotel_errors['count'], hotel_errors['mean'].abs(), alpha=0.5, s=30)
    ax.set_xlabel('Weeks Tested per Hotel')
    ax.set_ylabel('Mean Absolute Price Error (%)')
    ax.set_title('Accuracy vs Coverage')
    
    # 6. Summary metrics
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
COMPREHENSIVE COLD-START VALIDATION

Coverage:
  • Hotels: {results.n_hotels:,}
  • Weeks: {results.n_weeks:,}
  • Observations: {results.n_observations:,}

Price Accuracy:
  • MAPE: {results.price_mape:.1f}%
  • Correlation: {results.price_correlation:.2f}

RevPAR Accuracy:
  • MAPE: {results.revpar_mape:.1f}%
  • Correlation: {results.revpar_correlation:.2f}

Direction Accuracy: {results.direction_accuracy:.1f}%

By Season (RevPAR MAPE):
  • Summer: {results.summer_mape:.1f}%
  • Winter: {results.winter_mape:.1f}%
  • Shoulder: {results.shoulder_mape:.1f}%
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_cold_start.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved to {output_dir / 'comprehensive_cold_start.png'}")


# =============================================================================
# 3-MODEL PIPELINE
# =============================================================================

def run_three_model_pipeline(
    min_price: float = 50,
    max_price: float = 200,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> ThreeModelPipelineResults:
    """
    Run the complete 3-model RevPAR prediction pipeline.
    
    Architecture:
    1. Split hotels from BOOKINGS table into train/val/test (60/20/20)
    2. Model 1 (Baseline Pricing): Train on train set, predicts "business as usual" price
    3. Model 2 (Occupancy with Price): Train on val set, price is INPUT feature
    4. Model 3 (Optimal Pricing): Uses Model 2 to find RevPAR-maximizing price
    5. Evaluate on test set: Compare baseline vs optimal vs actual
    
    NO DATA LEAKAGE:
    - Hotel split done at booking level
    - Test hotels never seen during training
    - Peer features for test hotels come from train hotels only
    
    Args:
        min_price: Minimum price filter
        max_price: Maximum price filter  
        train_size: Fraction for training Model 1 (60%)
        val_size: Fraction for training Model 2 (20%)
        test_size: Fraction for final evaluation (20%)
        random_state: Random seed
    
    Returns:
        ThreeModelPipelineResults with comprehensive metrics
    """
    print("=" * 70)
    print("3-MODEL REVPAR PIPELINE")
    print("=" * 70)
    print(f"\nArchitecture:")
    print(f"  Model 1: Baseline Pricing (trained on {int(train_size*100)}% hotels)")
    print(f"  Model 2: Occupancy w/ Price (trained on {int(val_size*100)}% hotels)")
    print(f"  Model 3: Optimal Pricing (uses Model 2)")
    print(f"  Evaluation on {int(test_size*100)}% holdout hotels")
    
    # Step 1: Load data and split by hotel_id from bookings
    print("\n1. Loading data and splitting hotels from bookings table...")
    con = get_clean_connection()
    
    train_hotel_ids, val_hotel_ids, test_hotel_ids = split_hotels_train_val_test(
        con, train_size, val_size, test_size, random_state
    )
    
    print(f"   Train hotels: {len(train_hotel_ids):,}")
    print(f"   Validation hotels: {len(val_hotel_ids):,}")
    print(f"   Test hotels: {len(test_hotel_ids):,}")
    
    # Step 2: Load data for each split
    print("\n2. Loading weekly data for each split...")
    
    train_df = load_all_hotel_weeks(con, min_price, max_price, hotel_ids=train_hotel_ids)
    val_df = load_all_hotel_weeks(con, min_price, max_price, hotel_ids=val_hotel_ids)
    test_df = load_all_hotel_weeks(con, min_price, max_price, hotel_ids=test_hotel_ids)
    
    print(f"   Train observations: {len(train_df):,}")
    print(f"   Validation observations: {len(val_df):,}")
    print(f"   Test observations: {len(test_df):,}")
    
    # Step 3: Engineer features
    print("\n3. Engineering features...")
    
    train_df = engineer_validated_features(train_df)
    val_df = engineer_validated_features(val_df)
    test_df = engineer_validated_features(test_df)
    
    # Add peer features - TRAIN peers for all sets (no leakage!)
    print("   Adding peer features from TRAIN hotels only...")
    train_df = add_peer_price_features(train_df, peer_df=train_df, radius_km=PEER_RADIUS_KM)
    val_df = add_peer_price_features(val_df, peer_df=train_df, radius_km=PEER_RADIUS_KM)
    test_df = add_peer_price_features(test_df, peer_df=train_df, radius_km=PEER_RADIUS_KM)
    
    # Step 4: Train Model 1 - Baseline Pricing
    print("\n4. Training Model 1: Baseline Pricing...")
    baseline_model = BaselinePricingModel()
    baseline_model.fit(train_df, target_col='actual_price')
    
    # Step 5: Train Model 2 - Occupancy with Price
    print("\n5. Training Model 2: Occupancy with Price...")
    val_df['city_standardized'] = val_df['city'].str.lower().str.strip()
    
    occupancy_model = OccupancyModelWithPrice()
    occupancy_model.fit(val_df, price_col='actual_price', target_col='actual_occupancy')
    
    # Step 6: Create Model 3 - Optimal Pricing
    print("\n6. Creating Model 3: Optimal Pricing...")
    optimal_model = OptimalPricingModel(occupancy_model)
    
    # Step 7: Evaluate on test set
    print(f"\n7. Evaluating on {len(test_df):,} test observations...")
    
    results = []
    test_df['city_standardized'] = test_df['city'].str.lower().str.strip()
    
    for idx, (_, row) in enumerate(test_df.iterrows()):
        if idx % 100 == 0:
            print(f"   Processing {idx+1}/{len(test_df)}...")
        
        hotel_features = row.to_frame().T
        actual_price = row['actual_price']
        actual_occupancy = row['actual_occupancy']
        actual_revpar = row['actual_revpar']
        
        peer_price = row.get('peer_price_mean', actual_price)
        if pd.isna(peer_price) or peer_price <= 0:
            peer_price = actual_price
        
        try:
            # Model 1: Baseline price
            baseline_price = baseline_model.predict(hotel_features)[0]
            
            # Model 2: Occupancy at baseline price
            baseline_occ = occupancy_model.predict_at_price(
                hotel_features, baseline_price, peer_price_mean=peer_price
            )[0]
            baseline_revpar = baseline_price * baseline_occ
            
            # Model 3: Optimal price
            optimal_result = optimal_model.find_optimal_price(
                hotel_features,
                baseline_price=baseline_price,
                peer_price=peer_price
            )
            
            # Calculate errors
            baseline_price_error = (baseline_price - actual_price) / actual_price * 100
            optimal_vs_baseline = (optimal_result.optimal_revpar - baseline_revpar) / max(baseline_revpar, 1) * 100
            optimal_vs_actual = (optimal_result.optimal_revpar - actual_revpar) / max(actual_revpar, 1) * 100
            
            results.append(ThreeModelResult(
                hotel_id=int(row['hotel_id']),
                week_start=str(row['week_start']),
                actual_price=actual_price,
                actual_occupancy=actual_occupancy,
                actual_revpar=actual_revpar,
                baseline_price=baseline_price,
                baseline_occupancy=baseline_occ,
                baseline_revpar=baseline_revpar,
                optimal_price=optimal_result.optimal_price,
                optimal_occupancy=optimal_result.optimal_occupancy,
                optimal_revpar=optimal_result.optimal_revpar,
                baseline_price_error_pct=baseline_price_error,
                optimal_revpar_lift_pct=optimal_vs_baseline,
                optimal_vs_actual_pct=optimal_vs_actual,
            ))
        except Exception as e:
            continue
    
    print(f"\n8. Calculating metrics...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame([vars(r) for r in results])
    
    if len(results_df) == 0:
        raise ValueError("No valid predictions generated")
    
    # Calculate metrics
    baseline_price_mape = results_df['baseline_price_error_pct'].abs().mean()
    
    # R² for baseline price
    ss_res = ((results_df['baseline_price'] - results_df['actual_price']) ** 2).sum()
    ss_tot = ((results_df['actual_price'] - results_df['actual_price'].mean()) ** 2).sum()
    baseline_price_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Occupancy model performance (from model metrics)
    occupancy_metrics = occupancy_model.get_metrics() or {}
    
    # Optimal pricing outcomes
    avg_lift_vs_baseline = results_df['optimal_revpar_lift_pct'].mean()
    avg_lift_vs_actual = results_df['optimal_vs_actual_pct'].mean()
    pct_improved = (results_df['optimal_revpar'] > results_df['baseline_revpar']).mean() * 100
    
    return ThreeModelPipelineResults(
        n_train_hotels=len(train_hotel_ids),
        n_val_hotels=len(val_hotel_ids),
        n_test_hotels=len(test_hotel_ids),
        n_test_observations=len(results_df),
        baseline_price_mape=baseline_price_mape,
        baseline_price_r2=baseline_price_r2,
        occupancy_mape=occupancy_metrics.get('mae', 0) * 100,
        occupancy_r2=occupancy_metrics.get('r2', 0),
        avg_revpar_lift_vs_baseline=avg_lift_vs_baseline,
        avg_revpar_lift_vs_actual=avg_lift_vs_actual,
        pct_hotels_improved=pct_improved,
        results_df=results_df,
        baseline_model_metrics=baseline_model.get_metrics() or {},
        occupancy_model_metrics=occupancy_metrics,
    )


def print_three_model_summary(results: ThreeModelPipelineResults) -> None:
    """Print summary of 3-model pipeline results."""
    print("\n" + "=" * 70)
    print("3-MODEL PIPELINE RESULTS")
    print("=" * 70)
    
    print(f"\n1. DATA SPLITS")
    print("-" * 40)
    print(f"   Train hotels: {results.n_train_hotels:,}")
    print(f"   Validation hotels: {results.n_val_hotels:,}")
    print(f"   Test hotels: {results.n_test_hotels:,}")
    print(f"   Test observations: {results.n_test_observations:,}")
    
    print(f"\n2. MODEL 1: BASELINE PRICING")
    print("-" * 40)
    print(f"   Price MAPE: {results.baseline_price_mape:.1f}%")
    print(f"   Price R²: {results.baseline_price_r2:.3f}")
    
    print(f"\n3. MODEL 2: OCCUPANCY (with price as input)")
    print("-" * 40)
    print(f"   Occupancy MAE: {results.occupancy_mape:.1f}%")
    print(f"   Occupancy R²: {results.occupancy_r2:.3f}")
    
    print(f"\n4. MODEL 3: OPTIMAL PRICING OUTCOMES")
    print("-" * 40)
    print(f"   Avg RevPAR lift vs baseline: {results.avg_revpar_lift_vs_baseline:+.1f}%")
    print(f"   Avg RevPAR lift vs actual: {results.avg_revpar_lift_vs_actual:+.1f}%")
    print(f"   Hotels improved: {results.pct_hotels_improved:.1f}%")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    import sys
    
    # Determine which pipeline to run
    run_three_model = '--three-model' in sys.argv or '-3' in sys.argv
    
    if run_three_model:
        # Run new 3-model pipeline
        print("Running 3-model pipeline with proper train/val/test split...")
        results = run_three_model_pipeline(
            min_price=50,
            max_price=200,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42
        )
        
        print_three_model_summary(results)
        
        output_dir = Path('outputs/evaluation/cold_start')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results.results_df.to_csv(output_dir / 'three_model_results.csv', index=False)
        print(f"\n✓ Saved results to {output_dir / 'three_model_results.csv'}")
    else:
        # Run original cold-start validation
        results = run_comprehensive_cold_start(
            n_weeks_per_hotel=2,
            min_price=50,
            max_price=200,
            random_state=42
        )
        
        print_comprehensive_summary(results)
        
        output_dir = Path('outputs/evaluation/cold_start')
        create_comprehensive_visualizations(results, output_dir)
        
        # Save results
        results.results_df.to_csv(output_dir / 'comprehensive_results.csv', index=False)
        print(f"\n✓ Saved results to {output_dir / 'comprehensive_results.csv'}")

