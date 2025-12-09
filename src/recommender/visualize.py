"""
Visualization functions for price recommendations.

Creates diagnostic plots to validate recommendation quality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.recommender.price_recommender import PriceRecommender


def plot_recommendation_distribution(
    recommender: PriceRecommender,
    n_samples: int = 500,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot distribution of recommendation directions.
    
    Shows pie chart of increase/decrease/maintain.
    
    Args:
        recommender: Fitted PriceRecommender
        n_samples: Number of hotels to sample
        output_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Get recommendations
    hotel_ids = recommender.hotel_data['hotel_id'].unique()
    sample_ids = np.random.choice(hotel_ids, size=min(n_samples, len(hotel_ids)), replace=False)
    results = recommender.recommend_batch(list(sample_ids), '2024-06-15')
    
    # Count directions
    counts = results['direction'].value_counts()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    colors = {'increase': '#2ecc71', 'decrease': '#e74c3c', 'maintain': '#95a5a6'}
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=counts.index.str.upper(),
        autopct='%1.1f%%',
        colors=[colors.get(d, '#95a5a6') for d in counts.index],
        startangle=90,
        explode=[0.02] * len(counts)
    )
    ax1.set_title(f'Recommendation Distribution\n(n={len(results)} hotels)', fontsize=14)
    
    # Bar chart of price changes
    ax2 = axes[1]
    for direction, color in colors.items():
        subset = results[results['direction'] == direction]
        if len(subset) > 0:
            ax2.hist(
                subset['change_pct'],
                bins=20,
                alpha=0.7,
                color=color,
                label=f"{direction.upper()} (n={len(subset)})"
            )
    
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Price Change (%)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Recommended Price Changes', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    return fig


def plot_diagnosis_quadrants(
    recommender: PriceRecommender,
    n_samples: int = 500,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot diagnosis quadrants: price premium vs occupancy residual.
    
    Shows how hotels are classified based on price position and occupancy.
    
    Args:
        recommender: Fitted PriceRecommender
        n_samples: Number of hotels to sample
        output_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Get recommendations
    hotel_ids = recommender.hotel_data['hotel_id'].unique()
    sample_ids = np.random.choice(hotel_ids, size=min(n_samples, len(hotel_ids)), replace=False)
    results = recommender.recommend_batch(list(sample_ids), '2024-06-15')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {'increase': '#2ecc71', 'decrease': '#e74c3c', 'maintain': '#95a5a6'}
    
    for direction in ['increase', 'decrease', 'maintain']:
        subset = results[results['direction'] == direction]
        ax.scatter(
            subset['price_premium_pct'],
            subset['occ_residual'] * 100,
            c=colors[direction],
            alpha=0.6,
            s=50,
            label=f"{direction.upper()} (n={len(subset)})"
        )
    
    # Add quadrant lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Threshold lines
    ax.axhline(5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(-5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(10, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(-10, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Quadrant labels
    ax.text(30, 20, 'Premium but\nstrong demand\n→ MAINTAIN', ha='center', fontsize=10, color='#666')
    ax.text(-30, 20, 'Cheap with\nstrong demand\n→ INCREASE', ha='center', fontsize=10, color='#2ecc71')
    ax.text(30, -20, 'Expensive but\nweak demand\n→ DECREASE', ha='center', fontsize=10, color='#e74c3c')
    ax.text(-30, -20, 'Cheap but\nweak demand\n→ OTHER ISSUES', ha='center', fontsize=10, color='#666')
    
    ax.set_xlabel('Price Premium vs Peers (%)', fontsize=12)
    ax.set_ylabel('Occupancy Residual (actual - expected) (%)', fontsize=12)
    ax.set_title('Pricing Diagnosis: Price Position vs Occupancy Performance', fontsize=14)
    ax.legend(loc='upper right')
    
    # Set reasonable axis limits
    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 40)
    
    ax.grid(True, alpha=0.3)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    return fig


def plot_sample_hotels(
    recommender: PriceRecommender,
    n_hotels: int = 10,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot detailed view of sample hotel recommendations.
    
    Args:
        recommender: Fitted PriceRecommender
        n_hotels: Number of hotels to show
        output_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Get diverse sample (one from each direction)
    hotel_ids = recommender.hotel_data['hotel_id'].unique()
    results = recommender.recommend_batch(list(hotel_ids[:100]), '2024-06-15')
    
    sample = []
    for direction in ['increase', 'decrease', 'maintain']:
        subset = results[results['direction'] == direction]
        if len(subset) > 0:
            sample.append(subset.sample(n=min(n_hotels // 3 + 1, len(subset))))
    
    sample = pd.concat(sample).head(n_hotels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, n_hotels * 0.6 + 2))
    
    y_positions = range(len(sample))
    colors = {'increase': '#2ecc71', 'decrease': '#e74c3c', 'maintain': '#95a5a6'}
    
    for i, (_, row) in enumerate(sample.iterrows()):
        # Current price bar
        ax.barh(i, row['current_price'], height=0.3, color='#3498db', alpha=0.7, label='Current' if i == 0 else '')
        
        # Recommended price bar
        ax.barh(i - 0.3, row['recommended_price'], height=0.3, color=colors[row['direction']], alpha=0.7,
                label='Recommended' if i == 0 else '')
        
        # Peer price line
        ax.axvline(row['peer_price'], ymin=(len(sample) - i - 0.5) / len(sample),
                   ymax=(len(sample) - i + 0.5) / len(sample), color='black', linestyle='--', linewidth=1)
        
        # Labels
        ax.text(max(row['current_price'], row['recommended_price']) + 5, i - 0.15,
                f"{row['direction'].upper()} {row['change_pct']:+.1f}%",
                va='center', fontsize=9, color=colors[row['direction']])
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Hotel {int(row['hotel_id'])}" for _, row in sample.iterrows()])
    ax.set_xlabel('Price (€)', fontsize=12)
    ax.set_title('Sample Hotel Recommendations\n(dashed line = peer average)', fontsize=14)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    return fig


def create_validation_report(
    recommender: PriceRecommender,
    output_dir: Path,
    n_samples: int = 500
) -> None:
    """
    Create full validation report with all visualizations.
    
    Args:
        recommender: Fitted PriceRecommender
        output_dir: Directory to save figures
        n_samples: Number of hotels to sample
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating validation report...")
    
    # 1. Distribution plot
    print("  1. Recommendation distribution...")
    plot_recommendation_distribution(
        recommender, n_samples,
        output_dir / 'recommendation_distribution.png'
    )
    plt.close()
    
    # 2. Diagnosis quadrants
    print("  2. Diagnosis quadrants...")
    plot_diagnosis_quadrants(
        recommender, n_samples,
        output_dir / 'diagnosis_quadrants.png'
    )
    plt.close()
    
    # 3. Sample hotels
    print("  3. Sample hotels...")
    plot_sample_hotels(
        recommender, 12,
        output_dir / 'sample_hotels.png'
    )
    plt.close()
    
    print(f"\n✓ Report saved to {output_dir}/")

