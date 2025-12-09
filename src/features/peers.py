"""
Peer matching for price comparison.

Finds comparable hotels based on:
- City (geographic proximity)
- Room type (product similarity)
- Month (temporal alignment)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def compute_peer_stats(
    df: pd.DataFrame,
    group_cols: list = ['city_standardized', 'room_type', 'month_number']
) -> pd.DataFrame:
    """
    Compute peer group statistics for price and occupancy.
    
    Args:
        df: Hotel data with avg_price and occupancy_rate
        group_cols: Columns to group by for peer comparison
    
    Returns:
        DataFrame with peer statistics per group
    """
    # Filter to valid rows
    valid = df[df['avg_price'].notna() & (df['avg_price'] > 0)]
    
    # Use only available columns
    available_cols = [c for c in group_cols if c in valid.columns]
    if not available_cols:
        raise ValueError(f"No valid group columns found. Available: {valid.columns.tolist()}")
    
    # Compute statistics
    peer_stats = valid.groupby(available_cols, observed=True).agg({
        'avg_price': ['mean', 'median', 'std', 'count'],
        'occupancy_rate': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    peer_stats.columns = (
        available_cols + 
        ['peer_price_mean', 'peer_price_median', 'peer_price_std', 'n_peers',
         'peer_occ_mean', 'peer_occ_std']
    )
    
    # Fill NaN std with 0
    peer_stats['peer_price_std'] = peer_stats['peer_price_std'].fillna(0)
    peer_stats['peer_occ_std'] = peer_stats['peer_occ_std'].fillna(0)
    
    return peer_stats


def get_peer_price(
    hotel_row: pd.Series,
    peer_stats: pd.DataFrame,
    group_cols: list = ['city_standardized', 'room_type', 'month_number']
) -> Tuple[float, int]:
    """
    Get peer average price for a hotel.
    
    Falls back to less specific groupings if exact match not found.
    
    Args:
        hotel_row: Series with hotel attributes
        peer_stats: DataFrame from compute_peer_stats
        group_cols: Columns used for grouping
    
    Returns:
        Tuple of (peer_price, n_peers)
    """
    # Try exact match
    available_cols = [c for c in group_cols if c in hotel_row.index and c in peer_stats.columns]
    
    if not available_cols:
        # No matching possible
        return peer_stats['peer_price_mean'].mean(), 1
    
    # Build filter
    mask = pd.Series([True] * len(peer_stats))
    for col in available_cols:
        mask = mask & (peer_stats[col] == hotel_row[col])
    
    matches = peer_stats[mask]
    
    if len(matches) > 0:
        return matches.iloc[0]['peer_price_mean'], int(matches.iloc[0]['n_peers'])
    
    # Fall back to city + month
    if 'city_standardized' in available_cols and 'month_number' in available_cols:
        mask = (
            (peer_stats['city_standardized'] == hotel_row['city_standardized']) &
            (peer_stats['month_number'] == hotel_row['month_number'])
        )
        matches = peer_stats[mask]
        if len(matches) > 0:
            return matches['peer_price_mean'].mean(), int(matches['n_peers'].sum())
    
    # Fall back to just month
    if 'month_number' in available_cols:
        mask = peer_stats['month_number'] == hotel_row['month_number']
        matches = peer_stats[mask]
        if len(matches) > 0:
            return matches['peer_price_mean'].mean(), int(matches['n_peers'].sum())
    
    # Global fallback
    return peer_stats['peer_price_mean'].mean(), 1


def add_peer_features(
    df: pd.DataFrame,
    peer_stats: Optional[pd.DataFrame] = None,
    group_cols: list = ['city_standardized', 'room_type', 'month_number']
) -> pd.DataFrame:
    """
    Add peer price and price premium to DataFrame.
    
    Args:
        df: Hotel data
        peer_stats: Pre-computed peer stats (if None, computes from df)
        group_cols: Columns for grouping
    
    Returns:
        DataFrame with peer_price and price_premium columns
    """
    df = df.copy()
    
    if peer_stats is None:
        peer_stats = compute_peer_stats(df, group_cols)
    
    # Merge peer stats
    available_cols = [c for c in group_cols if c in df.columns and c in peer_stats.columns]
    
    if available_cols:
        df = df.merge(
            peer_stats[available_cols + ['peer_price_mean', 'n_peers']],
            on=available_cols,
            how='left'
        )
        df = df.rename(columns={'peer_price_mean': 'peer_price'})
    else:
        df['peer_price'] = peer_stats['peer_price_mean'].mean()
        df['n_peers'] = 1
    
    # Fill missing peer prices with global mean
    global_mean = df['avg_price'].mean()
    df['peer_price'] = df['peer_price'].fillna(global_mean)
    df['n_peers'] = df['n_peers'].fillna(1).astype(int)
    
    # Calculate price premium
    df['price_premium'] = (df['avg_price'] - df['peer_price']) / df['peer_price']
    df['price_premium'] = df['price_premium'].fillna(0)
    
    return df

