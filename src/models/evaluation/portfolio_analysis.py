"""
Portfolio Analysis for Pricing Recommendations.

Analyzes the distribution of hotels across pricing categories
and calculates the impact of recommended price changes.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.recommender.revpar_peers import (
    RevPARComparison,
    PerformanceSignal,
    PriceOpportunity,
    get_revpar_comparison_for_hotel,
)
from src.recommender.triangulated_scorer import (
    get_revpar_optimized_recommendation,
    RevPAROptimizedRecommendation,
)


class HotelCategory(Enum):
    """Hotel pricing category based on peer comparison."""
    UNDERPRICED = "underpriced"      # Price too low, leaving money on table
    OPTIMAL = "optimal"               # Price well-calibrated to market
    OVERPRICED = "overpriced"         # Price too high, losing bookings


@dataclass
class CategoryStats:
    """Statistics for a pricing category."""
    count: int
    pct_of_total: float
    
    # Current state
    avg_price: float
    avg_occupancy: float
    avg_revpar: float
    
    # Peer comparison
    avg_peer_price: float
    avg_peer_occupancy: float
    avg_peer_revpar: float
    avg_price_gap_pct: float
    avg_revpar_gap_pct: float
    
    # Recommended state
    avg_recommended_price: float
    avg_expected_occupancy: float
    avg_expected_revpar: float
    avg_change_pct: float
    avg_revpar_lift_pct: float
    
    # Sample hotels for examples
    sample_hotel_ids: List[int] = field(default_factory=list)


@dataclass
class PortfolioAnalysis:
    """Complete portfolio analysis results."""
    analysis_date: date
    n_hotels_analyzed: int
    n_hotels_total: int
    coverage_pct: float
    
    # Category breakdown
    categories: Dict[HotelCategory, CategoryStats]
    
    # Overall portfolio impact
    total_current_revpar: float
    total_expected_revpar: float
    total_revpar_lift_pct: float
    
    # Individual hotel results
    hotel_results: pd.DataFrame
    
    def get_category_summary(self) -> pd.DataFrame:
        """Get summary table of categories."""
        rows = []
        for cat, stats in self.categories.items():
            rows.append({
                'Category': cat.value.title(),
                'Hotels': stats.count,
                '% of Portfolio': f"{stats.pct_of_total:.1f}%",
                'Avg Price': f"€{stats.avg_price:.0f}",
                'Avg Occ': f"{stats.avg_occupancy:.0%}",
                'Avg RevPAR': f"€{stats.avg_revpar:.0f}",
                'vs Peer Price': f"{stats.avg_price_gap_pct:+.0f}%",
                'vs Peer RevPAR': f"{stats.avg_revpar_gap_pct:+.0f}%",
                'Recommended': f"€{stats.avg_recommended_price:.0f}",
                'Expected Occ': f"{stats.avg_expected_occupancy:.0%}",
                'Expected RevPAR': f"€{stats.avg_expected_revpar:.0f}",
                'RevPAR Lift': f"{stats.avg_revpar_lift_pct:+.1f}%",
            })
        return pd.DataFrame(rows)
    
    def print_summary(self) -> None:
        """Print executive summary to console."""
        print("=" * 80)
        print("PORTFOLIO PRICING ANALYSIS")
        print("=" * 80)
        print(f"Analysis Date: {self.analysis_date}")
        print(f"Hotels Analyzed: {self.n_hotels_analyzed:,} of {self.n_hotels_total:,} ({self.coverage_pct:.1f}%)")
        print()
        
        print("CATEGORY BREAKDOWN")
        print("-" * 80)
        summary = self.get_category_summary()
        print(summary.to_string(index=False))
        print()
        
        print("PORTFOLIO IMPACT")
        print("-" * 80)
        print(f"Current Portfolio RevPAR:  €{self.total_current_revpar:,.0f}")
        print(f"Expected Portfolio RevPAR: €{self.total_expected_revpar:,.0f}")
        print(f"Total RevPAR Lift:         {self.total_revpar_lift_pct:+.1f}%")
        print("=" * 80)


def categorize_hotel(comparison: RevPARComparison) -> HotelCategory:
    """
    Categorize a hotel based on its pricing position.
    
    Categories:
    - UNDERPRICED: Price significantly below peers (>10% below)
    - OVERPRICED: Price significantly above peers with lower occupancy
    - OPTIMAL: Price within reasonable range of peers
    
    Args:
        comparison: RevPAR comparison result
    
    Returns:
        HotelCategory classification
    """
    price_gap = comparison.price_gap
    occ_gap = comparison.occupancy_gap
    revpar_gap = comparison.revpar_gap
    
    # Underpriced: Charging less than peers
    if price_gap < -0.10:  # More than 10% below peer price
        return HotelCategory.UNDERPRICED
    
    # Overpriced: Charging more than peers with lower occupancy
    if price_gap > 0.10 and occ_gap < -0.05:  # >10% above peers, >5pp lower occ
        return HotelCategory.OVERPRICED
    
    # Also overpriced if RevPAR is way below despite higher prices
    if price_gap > 0 and revpar_gap < -0.20:
        return HotelCategory.OVERPRICED
    
    # Otherwise optimal
    return HotelCategory.OPTIMAL


def analyze_portfolio(
    con,
    target_dates: List[date],
    as_of_date: date,
    max_hotels: Optional[int] = None,
    twin_pairs_df: Optional[pd.DataFrame] = None
) -> PortfolioAnalysis:
    """
    Analyze pricing across the hotel portfolio.
    
    Args:
        con: Database connection
        target_dates: Dates to analyze
        as_of_date: Query date
        max_hotels: Maximum hotels to analyze (None = all)
        twin_pairs_df: Pre-loaded matched pairs
    
    Returns:
        PortfolioAnalysis with complete results
    """
    # Get all hotels with bookings in period
    hotel_query = f"""
        SELECT DISTINCT b.hotel_id 
        FROM bookings b 
        WHERE b.arrival_date >= '{target_dates[0].isoformat()}'
          AND b.arrival_date <= '{target_dates[-1].isoformat()}'
          AND b.status IN ('Booked', 'confirmed')
    """
    if max_hotels:
        hotel_query += f" LIMIT {max_hotels}"
    
    all_hotels = con.execute(hotel_query).fetchdf()
    n_hotels_total = len(all_hotels)
    
    # Analyze each hotel
    results = []
    for hotel_id in all_hotels['hotel_id'].tolist():
        hotel_id = int(hotel_id)
        try:
            comparison = get_revpar_comparison_for_hotel(
                con, hotel_id, target_dates, as_of_date,
                twin_pairs_df=twin_pairs_df
            )
            if comparison is None:
                continue
            
            rec = get_revpar_optimized_recommendation(comparison, hotel_id=hotel_id)
            category = categorize_hotel(comparison)
            
            results.append({
                'hotel_id': hotel_id,
                'category': category.value,
                'current_price': rec.current_price,
                'current_occupancy': rec.current_occupancy,
                'current_revpar': rec.current_revpar,
                'peer_price': rec.peer_price,
                'peer_occupancy': rec.peer_occupancy,
                'peer_revpar': rec.peer_revpar,
                'price_gap_pct': comparison.price_gap * 100,
                'revpar_gap_pct': comparison.revpar_gap * 100,
                'recommended_price': rec.optimal_price,
                'expected_occupancy': rec.optimal_occupancy,
                'expected_revpar': rec.optimal_revpar,
                'change_pct': rec.change_pct,
                'revpar_lift_pct': rec.revpar_lift * 100,
                'signal': comparison.signal.value,
                'opportunity': comparison.opportunity.value,
                'reasoning': rec.reasoning,
                'n_peers': comparison.n_peers,
                'peer_source': comparison.peer_source,
            })
        except Exception as e:
            print(f"Error analyzing hotel {hotel_id}: {e}")
    
    if len(results) == 0:
        raise ValueError("No hotels could be analyzed")
    
    df = pd.DataFrame(results)
    n_analyzed = len(df)
    
    # Calculate category statistics
    category_stats = {}
    for cat in HotelCategory:
        cat_df = df[df['category'] == cat.value]
        if len(cat_df) == 0:
            category_stats[cat] = CategoryStats(
                count=0, pct_of_total=0,
                avg_price=0, avg_occupancy=0, avg_revpar=0,
                avg_peer_price=0, avg_peer_occupancy=0, avg_peer_revpar=0,
                avg_price_gap_pct=0, avg_revpar_gap_pct=0,
                avg_recommended_price=0, avg_expected_occupancy=0,
                avg_expected_revpar=0, avg_change_pct=0, avg_revpar_lift_pct=0,
                sample_hotel_ids=[]
            )
            continue
        
        # Get sample hotels for examples
        sample_ids = cat_df.nlargest(3, 'current_revpar')['hotel_id'].tolist()
        
        category_stats[cat] = CategoryStats(
            count=len(cat_df),
            pct_of_total=len(cat_df) / n_analyzed * 100,
            avg_price=cat_df['current_price'].mean(),
            avg_occupancy=cat_df['current_occupancy'].mean(),
            avg_revpar=cat_df['current_revpar'].mean(),
            avg_peer_price=cat_df['peer_price'].mean(),
            avg_peer_occupancy=cat_df['peer_occupancy'].mean(),
            avg_peer_revpar=cat_df['peer_revpar'].mean(),
            avg_price_gap_pct=cat_df['price_gap_pct'].mean(),
            avg_revpar_gap_pct=cat_df['revpar_gap_pct'].mean(),
            avg_recommended_price=cat_df['recommended_price'].mean(),
            avg_expected_occupancy=cat_df['expected_occupancy'].mean(),
            avg_expected_revpar=cat_df['expected_revpar'].mean(),
            avg_change_pct=cat_df['change_pct'].mean(),
            avg_revpar_lift_pct=cat_df['revpar_lift_pct'].mean(),
            sample_hotel_ids=sample_ids
        )
    
    # Calculate portfolio totals
    total_current = df['current_revpar'].sum()
    total_expected = df['expected_revpar'].sum()
    total_lift = (total_expected - total_current) / total_current * 100 if total_current > 0 else 0
    
    return PortfolioAnalysis(
        analysis_date=as_of_date,
        n_hotels_analyzed=n_analyzed,
        n_hotels_total=n_hotels_total,
        coverage_pct=n_analyzed / n_hotels_total * 100 if n_hotels_total > 0 else 0,
        categories=category_stats,
        total_current_revpar=total_current,
        total_expected_revpar=total_expected,
        total_revpar_lift_pct=total_lift,
        hotel_results=df
    )


def get_category_examples(
    analysis: PortfolioAnalysis,
    category: HotelCategory,
    n_examples: int = 3
) -> pd.DataFrame:
    """
    Get example hotels from a category for stakeholder presentation.
    
    Args:
        analysis: Portfolio analysis results
        category: Category to get examples from
        n_examples: Number of examples to return
    
    Returns:
        DataFrame with example hotels
    """
    df = analysis.hotel_results
    cat_df = df[df['category'] == category.value].copy()
    
    if len(cat_df) == 0:
        return pd.DataFrame()
    
    # Get diverse examples: high RevPAR, low RevPAR, and middle
    if len(cat_df) >= n_examples:
        high = cat_df.nlargest(1, 'current_revpar')
        low = cat_df.nsmallest(1, 'current_revpar')
        mid = cat_df.iloc[[len(cat_df) // 2]]
        examples = pd.concat([high, mid, low]).head(n_examples)
    else:
        examples = cat_df
    
    # Select relevant columns for presentation (include category for explainer)
    return examples[[
        'hotel_id', 'category', 'current_price', 'current_occupancy', 'current_revpar',
        'peer_price', 'peer_occupancy', 'peer_revpar', 'price_gap_pct', 'revpar_gap_pct',
        'recommended_price', 'expected_occupancy', 'expected_revpar',
        'change_pct', 'revpar_lift_pct', 'reasoning'
    ]]

