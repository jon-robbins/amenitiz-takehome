"""Feature engineering for price recommendation."""
from .engineering import (
    # Constants
    VIEW_QUALITY_MAP,
    MARKET_ELASTICITY,
    COASTAL_THRESHOLD_KM,
    MADRID_THRESHOLD_KM,
    MADRID_LAT,
    MADRID_LON,
    PEER_RADIUS_KM,
    
    # Geographic utilities
    haversine_distance,
    calculate_city_centers,
    add_geographic_features,
    
    # Product features
    add_amenities_score,
    
    # Cold-start peer features (10km radius)
    calculate_peer_price_features,
    add_peer_price_features,
    
    # City standardization
    clean_city_name,
    standardize_city,
    get_market_segment,
    
    # Temporal features
    add_temporal_features,
    add_view_quality,
    
    # Main feature engineering
    engineer_features,
    engineer_validated_features,
)
from .peers import compute_peer_stats, get_peer_price

