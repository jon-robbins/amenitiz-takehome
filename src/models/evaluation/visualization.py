"""
Visualization for pricing recommendations.

Creates charts for stakeholder presentations.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .portfolio_analysis import PortfolioAnalysis, HotelCategory


def plot_pricing_distribution(
    analysis: PortfolioAnalysis,
    output_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot distribution of current vs recommended prices by category.
    
    Args:
        analysis: Portfolio analysis results
        output_path: Optional path to save figure
        ax: Optional axes to plot on
    
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    df = analysis.hotel_results
    categories = ['underpriced', 'optimal', 'overpriced']
    colors = {'underpriced': '#2ecc71', 'optimal': '#3498db', 'overpriced': '#e74c3c'}
    
    x = np.arange(len(categories))
    width = 0.35
    
    current_prices = [df[df['category'] == c]['current_price'].mean() for c in categories]
    recommended_prices = [df[df['category'] == c]['recommended_price'].mean() for c in categories]
    
    bars1 = ax.bar(x - width/2, current_prices, width, label='Current Price', 
                   color=[colors[c] for c in categories], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, recommended_prices, width, label='Recommended Price',
                   color=[colors[c] for c in categories], alpha=1.0, edgecolor='black', hatch='//')
    
    ax.set_ylabel('Average Price (€)', fontsize=12, fontweight='bold')
    ax.set_title('Current vs Recommended Prices by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in categories], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'€{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'€{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_revpar_impact(
    analysis: PortfolioAnalysis,
    output_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot RevPAR impact of recommendations by category.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    df = analysis.hotel_results
    categories = ['underpriced', 'optimal', 'overpriced']
    colors = {'underpriced': '#2ecc71', 'optimal': '#3498db', 'overpriced': '#e74c3c'}
    
    x = np.arange(len(categories))
    width = 0.35
    
    current_revpar = [df[df['category'] == c]['current_revpar'].mean() for c in categories]
    expected_revpar = [df[df['category'] == c]['expected_revpar'].mean() for c in categories]
    
    bars1 = ax.bar(x - width/2, current_revpar, width, label='Current RevPAR',
                   color=[colors[c] for c in categories], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, expected_revpar, width, label='Expected RevPAR',
                   color=[colors[c] for c in categories], alpha=1.0, edgecolor='black', hatch='//')
    
    ax.set_ylabel('Average RevPAR (€)', fontsize=12, fontweight='bold')
    ax.set_title('Current vs Expected RevPAR by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in categories], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add lift percentages
    for i, (cur, exp) in enumerate(zip(current_revpar, expected_revpar)):
        if cur > 0:
            lift = (exp - cur) / cur * 100
            ax.annotate(f'{lift:+.1f}%', xy=(i, max(cur, exp) + 2),
                       ha='center', fontsize=10, fontweight='bold',
                       color='green' if lift > 0 else 'red')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_category_breakdown(
    analysis: PortfolioAnalysis,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create pie chart of category distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart of hotel counts
    categories = [HotelCategory.UNDERPRICED, HotelCategory.OPTIMAL, HotelCategory.OVERPRICED]
    labels = ['Underpriced', 'Optimal', 'Overpriced']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    counts = [analysis.categories[c].count for c in categories]
    
    ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0, 0.05))
    ax1.set_title('Hotels by Pricing Category', fontsize=14, fontweight='bold')
    
    # Bar chart of RevPAR lift by category
    lifts = [analysis.categories[c].avg_revpar_lift_pct for c in categories]
    bars = ax2.bar(labels, lifts, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Average RevPAR Lift (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Expected RevPAR Lift by Category', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, lift in zip(bars, lifts):
        ax2.annotate(f'{lift:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, lift),
                    xytext=(0, 3 if lift >= 0 else -12), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_executive_summary(
    analysis: PortfolioAnalysis,
    output_dir: Path
) -> None:
    """
    Create complete executive summary with all visualizations.
    
    Args:
        analysis: Portfolio analysis results
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('Pricing Optimization: Executive Summary', fontsize=18, fontweight='bold', y=0.98)
    
    # Panel 1: Category breakdown
    ax1 = fig.add_subplot(2, 2, 1)
    categories = ['Underpriced', 'Optimal', 'Overpriced']
    counts = [analysis.categories[HotelCategory.UNDERPRICED].count,
              analysis.categories[HotelCategory.OPTIMAL].count,
              analysis.categories[HotelCategory.OVERPRICED].count]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    ax1.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Portfolio Distribution\n({analysis.n_hotels_analyzed} hotels)', fontsize=12, fontweight='bold')
    
    # Panel 2: Price comparison
    ax2 = fig.add_subplot(2, 2, 2)
    plot_pricing_distribution(analysis, ax=ax2)
    
    # Panel 3: RevPAR impact
    ax3 = fig.add_subplot(2, 2, 3)
    plot_revpar_impact(analysis, ax=ax3)
    
    # Panel 4: Summary metrics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    KEY METRICS
    {'─' * 40}
    
    Portfolio Coverage: {analysis.coverage_pct:.1f}%
    Hotels Analyzed: {analysis.n_hotels_analyzed:,}
    
    Current Portfolio RevPAR: €{analysis.total_current_revpar:,.0f}
    Expected Portfolio RevPAR: €{analysis.total_expected_revpar:,.0f}
    
    TOTAL REVPAR LIFT: {analysis.total_revpar_lift_pct:+.1f}%
    
    {'─' * 40}
    CATEGORY BREAKDOWN
    
    Underpriced: {analysis.categories[HotelCategory.UNDERPRICED].count} hotels
      → Avg lift: {analysis.categories[HotelCategory.UNDERPRICED].avg_revpar_lift_pct:+.1f}%
    
    Optimal: {analysis.categories[HotelCategory.OPTIMAL].count} hotels
      → Avg lift: {analysis.categories[HotelCategory.OPTIMAL].avg_revpar_lift_pct:+.1f}%
    
    Overpriced: {analysis.categories[HotelCategory.OVERPRICED].count} hotels
      → Avg lift: {analysis.categories[HotelCategory.OVERPRICED].avg_revpar_lift_pct:+.1f}%
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / 'executive_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save individual plots
    plot_category_breakdown(analysis, output_dir / 'category_breakdown.png')
    
    print(f"✓ Saved executive summary to {output_dir}")

