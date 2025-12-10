"""
Explainability module for pricing recommendations.

Generates human-readable explanations for stakeholders.
"""

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .portfolio_analysis import PortfolioAnalysis, HotelCategory, get_category_examples


def explain_recommendation(row: pd.Series) -> str:
    """
    Generate human-readable explanation for a single hotel recommendation.
    
    Args:
        row: Row from hotel_results DataFrame
    
    Returns:
        Formatted explanation string
    """
    # Handle both dict-like access and index-based access
    def get_val(key, default=0):
        try:
            return row[key]
        except (KeyError, TypeError):
            return default
    
    hotel_id = get_val('hotel_id', 'Unknown')
    category = get_val('category', 'unknown')
    
    current_price = get_val('current_price', 0)
    current_occ = get_val('current_occupancy', 0)
    current_revpar = get_val('current_revpar', 0)
    
    peer_price = get_val('peer_price', 0)
    peer_occ = get_val('peer_occupancy', 0)
    peer_revpar = get_val('peer_revpar', 0)
    
    rec_price = get_val('recommended_price', 0)
    exp_occ = get_val('expected_occupancy', 0)
    exp_revpar = get_val('expected_revpar', 0)
    
    change_pct = get_val('change_pct', 0)
    lift_pct = get_val('revpar_lift_pct', 0)
    
    price_gap_pct = get_val('price_gap_pct', 0)
    revpar_gap_pct = get_val('revpar_gap_pct', 0)
    reasoning = get_val('reasoning', '')
    
    # Build explanation
    lines = [
        f"═══════════════════════════════════════════════════════════════",
        f"HOTEL {hotel_id} - {category.upper()}",
        f"═══════════════════════════════════════════════════════════════",
        f"",
        f"CURRENT STATE:",
        f"  • Price: €{current_price:.0f}/night",
        f"  • Occupancy: {current_occ:.0%}",
        f"  • RevPAR: €{current_revpar:.0f}",
        f"",
        f"PEER BENCHMARK:",
        f"  • Peer Avg Price: €{peer_price:.0f}/night",
        f"  • Peer Avg Occupancy: {peer_occ:.0%}",
        f"  • Peer Avg RevPAR: €{peer_revpar:.0f}",
        f"  • Your Price vs Peers: {row['price_gap_pct']:+.0f}%",
        f"  • Your RevPAR vs Peers: {row['revpar_gap_pct']:+.0f}%",
        f"",
    ]
    
    # Category-specific insights
    if category == 'underpriced':
        lines.extend([
            f"DIAGNOSIS: UNDERPRICED",
            f"  You are charging {abs(price_gap_pct):.0f}% less than comparable hotels.",
        ])
        if current_occ >= 0.85:
            lines.append(f"  With {current_occ:.0%} occupancy, demand clearly supports higher prices.")
        else:
            lines.append(f"  Moving toward peer pricing should improve revenue.")
    
    elif category == 'overpriced':
        lines.extend([
            f"DIAGNOSIS: OVERPRICED",
            f"  You are charging {price_gap_pct:.0f}% more than comparable hotels,",
            f"  but achieving {abs(revpar_gap_pct):.0f}% lower RevPAR.",
            f"  Consider reducing prices to capture more bookings.",
        ])
    
    else:  # optimal
        lines.extend([
            f"DIAGNOSIS: OPTIMALLY PRICED",
            f"  Your pricing is well-aligned with the market.",
            f"  Minor adjustments may still improve performance.",
        ])
    
    lines.extend([
        f"",
        f"RECOMMENDATION:",
        f"  • New Price: €{rec_price:.0f}/night ({change_pct:+.1f}%)",
        f"  • Expected Occupancy: {exp_occ:.0%}",
        f"  • Expected RevPAR: €{exp_revpar:.0f}",
        f"  • RevPAR Lift: {lift_pct:+.1f}%",
        f"",
        f"REASONING: {reasoning}",
        f"",
    ])
    
    return "\n".join(lines)


def generate_stakeholder_report(
    analysis: PortfolioAnalysis,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate complete stakeholder report with examples.
    
    Args:
        analysis: Portfolio analysis results
        output_path: Optional path to save report
    
    Returns:
        Report text
    """
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║           PRICING OPTIMIZATION REPORT - STAKEHOLDER SUMMARY                  ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
        f"Report Date: {analysis.analysis_date}",
        f"Hotels Analyzed: {analysis.n_hotels_analyzed:,} of {analysis.n_hotels_total:,} ({analysis.coverage_pct:.1f}%)",
        "",
        "─" * 80,
        "EXECUTIVE SUMMARY",
        "─" * 80,
        "",
        "Our pricing analysis identified three categories of hotels:",
        "",
    ]
    
    # Category summaries
    for cat in [HotelCategory.UNDERPRICED, HotelCategory.OPTIMAL, HotelCategory.OVERPRICED]:
        stats = analysis.categories[cat]
        if stats.count == 0:
            continue
        
        icon = "💰" if cat == HotelCategory.UNDERPRICED else ("✅" if cat == HotelCategory.OPTIMAL else "⚠️")
        
        lines.extend([
            f"{icon} {cat.value.upper()}: {stats.count} hotels ({stats.pct_of_total:.1f}%)",
            f"   Current: €{stats.avg_price:.0f} @ {stats.avg_occupancy:.0%} = €{stats.avg_revpar:.0f} RevPAR",
            f"   vs Peers: {stats.avg_price_gap_pct:+.0f}% price, {stats.avg_revpar_gap_pct:+.0f}% RevPAR",
            f"   Recommended: €{stats.avg_recommended_price:.0f} → €{stats.avg_expected_revpar:.0f} RevPAR ({stats.avg_revpar_lift_pct:+.1f}% lift)",
            "",
        ])
    
    # Portfolio impact
    lines.extend([
        "─" * 80,
        "PORTFOLIO IMPACT",
        "─" * 80,
        "",
        f"Current Total RevPAR:  €{analysis.total_current_revpar:,.0f}",
        f"Expected Total RevPAR: €{analysis.total_expected_revpar:,.0f}",
        f"",
        f"═══ TOTAL REVPAR LIFT: {analysis.total_revpar_lift_pct:+.1f}% ═══",
        "",
    ])
    
    # Examples for each category
    lines.extend([
        "─" * 80,
        "DETAILED EXAMPLES BY CATEGORY",
        "─" * 80,
        "",
    ])
    
    for cat in [HotelCategory.UNDERPRICED, HotelCategory.OPTIMAL, HotelCategory.OVERPRICED]:
        examples = get_category_examples(analysis, cat, n_examples=2)
        if len(examples) == 0:
            continue
        
        lines.append(f"\n{'='*40}")
        lines.append(f"CATEGORY: {cat.value.upper()}")
        lines.append(f"{'='*40}\n")
        
        for _, row in examples.iterrows():
            lines.append(explain_recommendation(row))
    
    # Recommendations
    lines.extend([
        "─" * 80,
        "ACTIONABLE RECOMMENDATIONS",
        "─" * 80,
        "",
    ])
    
    underpriced = analysis.categories[HotelCategory.UNDERPRICED]
    overpriced = analysis.categories[HotelCategory.OVERPRICED]
    
    if underpriced.count > 0:
        lines.extend([
            f"1. UNDERPRICED HOTELS ({underpriced.count} properties)",
            f"   • These hotels can raise prices by an average of {underpriced.avg_change_pct:+.0f}%",
            f"   • Expected RevPAR improvement: {underpriced.avg_revpar_lift_pct:+.1f}%",
            f"   • Priority: HIGH - immediate revenue opportunity",
            "",
        ])
    
    if overpriced.count > 0:
        lines.extend([
            f"2. OVERPRICED HOTELS ({overpriced.count} properties)",
            f"   • These hotels should consider price reductions",
            f"   • Lower prices will improve occupancy and overall RevPAR",
            f"   • Priority: MEDIUM - monitor booking velocity",
            "",
        ])
    
    lines.extend([
        "─" * 80,
        "END OF REPORT",
        "─" * 80,
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"✓ Saved stakeholder report to {output_path}")
    
    return report


def export_recommendations_csv(
    analysis: PortfolioAnalysis,
    output_path: Path
) -> None:
    """
    Export hotel recommendations to CSV for distribution.
    
    Args:
        analysis: Portfolio analysis results
        output_path: Path to save CSV
    """
    df = analysis.hotel_results.copy()
    
    # Format for readability
    df['current_price'] = df['current_price'].round(2)
    df['recommended_price'] = df['recommended_price'].round(2)
    df['current_occupancy'] = (df['current_occupancy'] * 100).round(1)
    df['expected_occupancy'] = (df['expected_occupancy'] * 100).round(1)
    df['current_revpar'] = df['current_revpar'].round(2)
    df['expected_revpar'] = df['expected_revpar'].round(2)
    df['change_pct'] = df['change_pct'].round(1)
    df['revpar_lift_pct'] = df['revpar_lift_pct'].round(1)
    
    # Rename for clarity
    df = df.rename(columns={
        'current_price': 'Current Price (€)',
        'recommended_price': 'Recommended Price (€)',
        'current_occupancy': 'Current Occupancy (%)',
        'expected_occupancy': 'Expected Occupancy (%)',
        'current_revpar': 'Current RevPAR (€)',
        'expected_revpar': 'Expected RevPAR (€)',
        'change_pct': 'Price Change (%)',
        'revpar_lift_pct': 'RevPAR Lift (%)',
    })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(df)} recommendations to {output_path}")

