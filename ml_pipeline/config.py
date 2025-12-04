"""
Configuration for price recommendation model.

Contains elasticity estimates, segment multipliers, and strategy thresholds
derived from matched pairs and GPS analysis.
"""

from dataclasses import dataclass
from typing import Dict


# =============================================================================
# ELASTICITY ESTIMATES (from matched pairs analysis)
# =============================================================================

# Global elasticity estimate (median from 809 matched pairs)
GLOBAL_ELASTICITY = -0.39

# Segment-specific elasticity (from segmented analysis)
SEGMENT_ELASTICITY = {
    'Coastal/Resort': -0.38,
    'Provincial/Regional': -0.39,
    'Urban/Madrid': -0.41
}

# Elasticity confidence interval (95% CI from bootstrap)
ELASTICITY_CI = {
    'lower': -0.41,
    'upper': -0.37
}


# =============================================================================
# PRICING STRATEGIES
# =============================================================================

@dataclass
class PricingStrategy:
    """Defines a pricing strategy with deviation and expected metrics."""
    name: str
    price_deviation_pct: float  # % above peer price
    expected_revpar_lift_pct: float  # Expected RevPAR improvement
    risk_level: str  # 'low', 'medium', 'high'
    description: str


STRATEGIES = {
    'conservative': PricingStrategy(
        name='Conservative',
        price_deviation_pct=15.0,
        expected_revpar_lift_pct=8.3,
        risk_level='low',
        description='Modest price increase with minimal occupancy risk'
    ),
    'safe': PricingStrategy(
        name='Safe Zone',
        price_deviation_pct=30.0,
        expected_revpar_lift_pct=14.5,
        risk_level='medium',
        description='Balanced approach within validated safe zone (15-40%)'
    ),
    'optimal': PricingStrategy(
        name='Optimal',
        price_deviation_pct=45.0,
        expected_revpar_lift_pct=14.8,
        risk_level='high',
        description='Maximum RevPAR based on matched pairs (near peak of curve)'
    )
}

# Safe zone boundaries (from GPS and matched pairs analysis)
SAFE_ZONE = {
    'lower': 15.0,  # Minimum recommended price deviation %
    'upper': 40.0   # Maximum recommended price deviation %
}

# Risk zone (beyond this, elasticity becomes more negative)
RISK_THRESHOLD = 50.0  # % above peer price


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# XGBoost hyperparameters (validated in feature_importance_validation.py)
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

# Cross-validation configuration
CV_FOLDS = 5
RANDOM_STATE = 42

# Model performance thresholds
MIN_R2_THRESHOLD = 0.65  # Minimum acceptable R² for price prediction
MAX_MAPE_THRESHOLD = 15.0  # Maximum acceptable MAPE (%)


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Date range for training data
TRAINING_DATE_RANGE = {
    'start': '2023-01-01',
    'end': '2024-12-31'
}

# Minimum observations per hotel for reliable estimation
MIN_HOTEL_OBSERVATIONS = 3

# Price outlier thresholds (IQR multiplier)
OUTLIER_IQR_MULTIPLIER = 1.5


# =============================================================================
# FILE PATHS
# =============================================================================

MODEL_DIR = 'ml_pipeline/models'
MODEL_FILENAME = 'price_model.pkl'
SCALER_FILENAME = 'feature_scaler.pkl'
ENCODER_FILENAME = 'category_encoder.pkl'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_strategy(name: str) -> PricingStrategy:
    """
    Returns pricing strategy by name.
    
    Args:
        name: One of 'conservative', 'safe', 'optimal'
    
    Returns:
        PricingStrategy dataclass
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Choose from: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def calculate_expected_revpar_change(
    price_deviation_pct: float,
    elasticity: float = GLOBAL_ELASTICITY
) -> float:
    """
    Calculates expected RevPAR change given price deviation.
    
    Uses the formula: RevPAR_change = price_change + elasticity * price_change
    Simplifies to: RevPAR_change = price_change * (1 + elasticity)
    
    Args:
        price_deviation_pct: % change in price (e.g., 30 for +30%)
        elasticity: Price elasticity of demand (default: global estimate)
    
    Returns:
        Expected % change in RevPAR
    """
    price_change_ratio = price_deviation_pct / 100
    occupancy_change_ratio = elasticity * price_change_ratio
    
    # RevPAR = Price × Occupancy
    # New RevPAR / Old RevPAR = (1 + price_change) × (1 + occ_change)
    revpar_ratio = (1 + price_change_ratio) * (1 + occupancy_change_ratio)
    revpar_change_pct = (revpar_ratio - 1) * 100
    
    return revpar_change_pct


def get_segment_elasticity(market_segment: str) -> float:
    """
    Returns elasticity for a market segment.
    
    Falls back to global elasticity if segment not found.
    """
    return SEGMENT_ELASTICITY.get(market_segment, GLOBAL_ELASTICITY)

