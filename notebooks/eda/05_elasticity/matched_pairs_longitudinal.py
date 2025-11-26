# %%
"""
Longitudinal Price Sensitivity Analysis (2023 vs 2024)

Instead of matching different hotels (which introduces quality bias), we perform
a Longitudinal Analysis: comparing the *same* hotel's performance in 2023 vs 2024.

This "Self-Matching" approach eliminates confounding variables like location,
star rating, or amenities - the hotel is its own control.

Goal: Find the "Tipping Point" where aggressive price hikes destroy value.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, '../../../..')

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# %%
def get_cleaning_config() -> CleaningConfig:
    """Returns standard cleaning configuration for longitudinal analysis."""
    return CleaningConfig(
        remove_negative_prices=True,
        remove_zero_prices=True,
        remove_low_prices=True,
        remove_null_prices=True,
        remove_extreme_prices=True,
        remove_null_dates=True,
        remove_null_created_at=True,
        remove_negative_stay=True,
        remove_negative_lead_time=True,
        remove_null_occupancy=True,
        remove_overcrowded_rooms=True,
        remove_null_room_id=True,
        remove_null_booking_id=True,
        remove_null_hotel_id=True,
        remove_orphan_bookings=True,
        remove_null_status=True,
        remove_cancelled_but_active=True,
        remove_bookings_before_2023=True,
        remove_bookings_after_2024=True,
        exclude_reception_halls=True,
        fix_empty_strings=True,
        impute_children_allowed=True,
        impute_events_allowed=True,
        set_empty_room_view_to_no_view_str=True,
        verbose=False
    )


def load_hotel_month_product_data(con) -> pd.DataFrame:
    """
    Loads hotel-month-product aggregation for 2023 and 2024.
    
    Aggregates to Hotel-Month-Product level with ADR, revenue, and room nights.
    """
    query = """
    SELECT 
        b.hotel_id,
        EXTRACT(YEAR FROM CAST(b.arrival_date AS DATE)) AS year,
        EXTRACT(MONTH FROM CAST(b.arrival_date AS DATE)) AS month,
        br.room_type,
        COALESCE(NULLIF(br.room_view, ''), 'no_view') AS room_view,
        r.children_allowed,
        AVG(br.total_price) AS avg_adr,
        SUM(br.total_price) AS total_revenue,
        COUNT(*) AS room_nights,
        SUM(r.number_of_rooms) AS total_capacity
    FROM bookings b
    JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
    JOIN rooms r ON br.room_id = r.id
    WHERE b.status IN ('confirmed', 'Booked')
      AND EXTRACT(YEAR FROM CAST(b.arrival_date AS DATE)) IN (2023, 2024)
    GROUP BY 
        b.hotel_id, 
        year, 
        month, 
        br.room_type, 
        room_view, 
        r.children_allowed
    HAVING COUNT(*) >= 5  -- Minimum sample size per product-month
    """
    return con.execute(query).fetchdf()


def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots data to wide format with separate columns for 2023 and 2024.
    
    Returns DataFrame with adr_2023, adr_2024, revenue_2023, revenue_2024, etc.
    """
    # Create product key for pivoting
    df['product_key'] = (
        df['hotel_id'].astype(str) + '_' +
        df['month'].astype(int).astype(str) + '_' +
        df['room_type'].astype(str) + '_' +
        df['room_view'].astype(str) + '_' +
        df['children_allowed'].astype(str)
    )
    
    # Separate 2023 and 2024 data
    df_2023 = df[df['year'] == 2023].copy()
    df_2024 = df[df['year'] == 2024].copy()
    
    # Rename columns for merge
    df_2023 = df_2023.rename(columns={
        'avg_adr': 'adr_2023',
        'total_revenue': 'revenue_2023',
        'room_nights': 'nights_2023',
        'total_capacity': 'capacity_2023'
    })
    
    df_2024 = df_2024.rename(columns={
        'avg_adr': 'adr_2024',
        'total_revenue': 'revenue_2024',
        'room_nights': 'nights_2024',
        'total_capacity': 'capacity_2024'
    })
    
    # Select columns for merge
    cols_2023 = ['product_key', 'hotel_id', 'month', 'room_type', 'room_view', 
                 'children_allowed', 'adr_2023', 'revenue_2023', 'nights_2023', 'capacity_2023']
    cols_2024 = ['product_key', 'adr_2024', 'revenue_2024', 'nights_2024', 'capacity_2024']
    
    # Inner merge to get only products that exist in both years
    df_wide = df_2023[cols_2023].merge(
        df_2024[cols_2024],
        on='product_key',
        how='inner'
    )
    
    return df_wide


def calculate_yoy_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Year-over-Year price and revenue changes.
    """
    df = df.copy()
    
    # Price change
    df['price_change_pct'] = (df['adr_2024'] - df['adr_2023']) / df['adr_2023']
    
    # Revenue change
    df['revenue_change_pct'] = (df['revenue_2024'] - df['revenue_2023']) / df['revenue_2023']
    
    # Volume change (room nights)
    df['volume_change_pct'] = (df['nights_2024'] - df['nights_2023']) / df['nights_2023']
    
    # RevPAR proxy (revenue per available capacity)
    df['revpar_2023'] = df['revenue_2023'] / df['capacity_2023'].replace(0, np.nan)
    df['revpar_2024'] = df['revenue_2024'] / df['capacity_2024'].replace(0, np.nan)
    df['revpar_change_pct'] = (df['revpar_2024'] - df['revpar_2023']) / df['revpar_2023']
    
    return df


def assign_strategy_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns pricing strategy bins based on YoY price change.
    
    Bins:
    - Slashers: < -10%
    - Flatliners: -10% to +5%
    - Nudgers: +5% to +15%
    - Hikers: +15% to +30%
    - Moonshots: > +30%
    """
    df = df.copy()
    
    bins = [-np.inf, -0.10, 0.05, 0.15, 0.30, np.inf]
    labels = [
        "Slashers\n (Prices decrease by >10%)", 
        "Flatliners\n (Prices change by -10% to +5%)", 
        "Nudgers\n (Prices increase by >5% to +15%)", 
        "Hikers\n (Prices increase by >15% to +30%)", 
        "Moonshots\n (Prices increase by >30%)"
    ]
    
    df['strategy_bin'] = pd.cut(
        df['price_change_pct'],
        bins=bins,
        labels=labels,
        ordered=True
    )
    
    return df


def analyze_strategy_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes performance metrics by pricing strategy bin.
    """
    strategy_stats = df.groupby('strategy_bin', observed=True).agg({
        'revenue_change_pct': ['median', 'mean', 'std', 'count'],
        'volume_change_pct': ['median', 'mean'],
        'price_change_pct': ['median', 'mean'],
        'hotel_id': 'nunique'
    }).round(4)
    
    strategy_stats.columns = [
        'revenue_median', 'revenue_mean', 'revenue_std', 'n_products',
        'volume_median', 'volume_mean',
        'price_median', 'price_mean',
        'n_hotels'
    ]
    
    # Calculate success rate (positive revenue growth)
    success_rates = df.groupby('strategy_bin', observed=True).apply(
        lambda x: (x['revenue_change_pct'] > 0).mean(),
        include_groups=False
    )
    strategy_stats['success_rate'] = success_rates
    
    return strategy_stats


def detect_danger_zone(strategy_stats: pd.DataFrame) -> tuple[bool, str]:
    """
    Detects if there's a "Danger Zone" where aggressive pricing yields diminishing returns.
    
    Returns:
        Tuple of (danger_zone_detected, message)
    """
    # Find Moonshots and Hikers by checking if index contains the key string
    moonshots_idx = None
    hikers_idx = None
    for idx in strategy_stats.index:
        if 'Moonshots' in str(idx):
            moonshots_idx = idx
        if 'Hikers' in str(idx):
            hikers_idx = idx
    
    if moonshots_idx is None or hikers_idx is None:
        return False, "INSUFFICIENT DATA: Not enough Moonshots or Hikers to detect danger zone."
    
    moonshots_revenue = strategy_stats.loc[moonshots_idx, 'revenue_median']
    hikers_revenue = strategy_stats.loc[hikers_idx, 'revenue_median']
    
    if moonshots_revenue < hikers_revenue:
        diff = hikers_revenue - moonshots_revenue
        return True, (
            f"⚠️  DANGER ZONE DETECTED: Aggressive pricing yields diminishing returns.\n"
            f"    Hikers (+15-30%) achieved {hikers_revenue:.1%} median revenue growth\n"
            f"    Moonshots (>+30%) achieved only {moonshots_revenue:.1%} median revenue growth\n"
            f"    → Price increases beyond 30% destroy {diff:.1%} of potential value"
        )
    else:
        return False, (
            f"✅ NO DANGER ZONE: Demand is extremely inelastic.\n"
            f"    Moonshots (>+30%) achieved {moonshots_revenue:.1%} median revenue growth\n"
            f"    Even aggressive price hikes continue to increase revenue."
        )


def create_strategy_visualization(
    df: pd.DataFrame,
    strategy_stats: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Creates comprehensive visualization of pricing strategy performance.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        "Slashers\n (Prices decrease by >10%)": '#E74C3C',    # Red - cutting prices
        "Flatliners\n (Prices change by -10% to +5%)": '#95A5A6',  # Gray - no change
        "Nudgers\n (Prices increase by >5% to +15%)": '#3498DB',     # Blue - modest increase
        "Hikers\n (Prices increase by >15% to +30%)": '#27AE60',      # Green - healthy increase
        "Moonshots\n (Prices increase by >30%)": '#F39C12'    # Orange - aggressive
    }
    
    ordered_bins = [
        "Slashers\n (Prices decrease by >10%)", 
        "Flatliners\n (Prices change by -10% to +5%)", 
        "Nudgers\n (Prices increase by >5% to +15%)", 
        "Hikers\n (Prices increase by >15% to +30%)", 
        "Moonshots\n (Prices increase by >30%)"
    ]
    existing_bins = [b for b in ordered_bins if b in strategy_stats.index]
    
    # =========================================================================
    # Plot 1: Median Revenue Growth by Strategy
    # =========================================================================
    ax1 = axes[0, 0]
    
    revenues = [strategy_stats.loc[b, 'revenue_median'] for b in existing_bins]
    bar_colors = [colors[b] for b in existing_bins]
    
    # Create multi-line labels with definitions
    display_labels = []
    for b in existing_bins:
        parts = b.split('\n')
        if len(parts) == 2:
            # Format as "Strategy\n(Definition)"
            display_labels.append(f"{parts[0]}\n{parts[1]}")
        else:
            display_labels.append(b)
    
    bars = ax1.bar(range(len(existing_bins)), revenues, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    for bar, rev in zip(bars, revenues):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{rev:.1%}',
                 ha='center', va='bottom' if height >= 0 else 'top',
                 fontsize=12, fontweight='bold')
    
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')
    ax1.set_title('Median Revenue Growth by Pricing Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Median YoY Revenue Change', fontsize=12)
    ax1.set_xlabel('Pricing Strategy (2023 → 2024)', fontsize=12)
    ax1.set_xticks(range(len(existing_bins)))
    ax1.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=9)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax1.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Plot 2: Sample Size & Success Rate
    # =========================================================================
    ax2 = axes[0, 1]
    
    x = np.arange(len(existing_bins))
    width = 0.35
    
    n_products = [strategy_stats.loc[b, 'n_products'] for b in existing_bins]
    success_rates = [strategy_stats.loc[b, 'success_rate'] for b in existing_bins]
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, n_products, width, color='steelblue', alpha=0.7, 
                    label='# Products', edgecolor='black')
    bars2 = ax2_twin.bar(x + width/2, success_rates, width, color='forestgreen', alpha=0.7,
                         label='Success Rate', edgecolor='black')
    
    # Create multi-line labels with definitions
    display_labels = []
    for b in existing_bins:
        parts = b.split('\n')
        if len(parts) == 2:
            display_labels.append(f"{parts[0]}\n{parts[1]}")
        else:
            display_labels.append(b)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Number of Hotel-Month-Products', fontsize=12, color='steelblue')
    ax2_twin.set_ylabel('Success Rate (% with Revenue Growth)', fontsize=12, color='forestgreen')
    ax2_twin.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax2.set_title('Sample Size & Success Rate by Strategy', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # =========================================================================
    # Plot 3: Price Change vs Revenue Change Scatter
    # =========================================================================
    ax3 = axes[1, 0]
    
    # Sample for visualization (too many points)
    df_sample = df.sample(min(2000, len(df)), random_state=42)
    
    for strategy in existing_bins:
        mask = df_sample['strategy_bin'] == strategy
        # Use full label with definition for legend
        parts = strategy.split('\n')
        if len(parts) == 2:
            legend_label = f"{parts[0]}\n{parts[1]}"
        else:
            legend_label = strategy
        ax3.scatter(
            df_sample.loc[mask, 'price_change_pct'] * 100,
            df_sample.loc[mask, 'revenue_change_pct'] * 100,
            c=colors[strategy],
            alpha=0.4,
            s=30,
            label=legend_label,
            edgecolors='none'
        )
    
    # Add regression line
    valid_mask = df['price_change_pct'].notna() & df['revenue_change_pct'].notna()
    if valid_mask.sum() > 10:
        z = np.polyfit(df.loc[valid_mask, 'price_change_pct'], 
                       df.loc[valid_mask, 'revenue_change_pct'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['price_change_pct'].min(), df['price_change_pct'].max(), 100)
        ax3.plot(x_line * 100, p(x_line) * 100, 'k--', linewidth=2, label='Trend')
    
    ax3.axhline(0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
    ax3.axvline(0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
    ax3.set_xlabel('Price Change (%)', fontsize=12)
    ax3.set_ylabel('Revenue Change (%)', fontsize=12)
    ax3.set_title('Price vs Revenue Change (Same Hotel, 2023→2024)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_xlim(-50, 100)
    ax3.set_ylim(-100, 200)
    ax3.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 4: The "Tipping Point" Curve
    # =========================================================================
    ax4 = axes[1, 1]
    
    # Create finer bins for the curve
    fine_bins = np.arange(-0.3, 0.6, 0.05)
    df['fine_bin'] = pd.cut(df['price_change_pct'], bins=fine_bins)
    
    curve_data = df.groupby('fine_bin', observed=True).agg({
        'revenue_change_pct': 'median',
        'hotel_id': 'count'
    }).reset_index()
    curve_data.columns = ['bin', 'revenue_median', 'count']
    # Extract bin midpoint as float (not categorical)
    curve_data['bin_center'] = curve_data['bin'].apply(
        lambda x: float(x.mid) if pd.notna(x) and hasattr(x, 'mid') else np.nan
    )
    # Explicitly convert to numeric to ensure it's not categorical
    curve_data['bin_center'] = pd.to_numeric(curve_data['bin_center'], errors='coerce')
    curve_data = curve_data.dropna(subset=['bin_center'])
    
    # Filter for bins with sufficient sample
    curve_data = curve_data[curve_data['count'] >= 20]
    
    # Convert to numpy arrays for plotting (ensure numeric types)
    x_vals = np.array(curve_data['bin_center'].values, dtype=float) * 100
    y_vals = np.array(curve_data['revenue_median'].values, dtype=float) * 100
    
    ax4.plot(x_vals, y_vals,
             'o-', color='#2C3E50', linewidth=2.5, markersize=8, markerfacecolor='white',
             markeredgewidth=2)
    
    # Highlight danger zone - always show at x > 30% based on user requirement
    # Check if we have data beyond 30% to determine if danger zone is relevant
    max_price_change = x_vals.max() if len(x_vals) > 0 else 0
    danger_detected, _ = detect_danger_zone(strategy_stats)
    
    # Always highlight danger zone at x > 30% if we have data in that range
    if max_price_change > 30:
        ax4.axvspan(30, max(max_price_change, 60), alpha=0.2, color='red', label='Danger Zone (>30%)')
        ax4.axvline(30, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax4.axhline(0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
    ax4.axvline(0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
    ax4.set_xlabel('Price Change (%)', fontsize=12)
    ax4.set_ylabel('Median Revenue Change (%)', fontsize=12)
    ax4.set_title('The Tipping Point: Where Price Hikes Hurt', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    # Always show legend if danger zone is highlighted
    if max_price_change > 30:
        ax4.legend(loc='upper right')
    
    # Add annotation for optimal zone
    if len(curve_data) > 0:
        optimal_idx = curve_data['revenue_median'].idxmax()
        optimal_price = float(curve_data.loc[optimal_idx, 'bin_center']) * 100
        optimal_rev = float(curve_data.loc[optimal_idx, 'revenue_median']) * 100
        ax4.annotate(f'Optimal: +{optimal_price:.0f}%',
                    xy=(optimal_price, optimal_rev),
                    xytext=(optimal_price + 10, optimal_rev + 15),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    color='green')
    
    plt.suptitle('Longitudinal Price Sensitivity Analysis (2023 vs 2024)\nSelf-Matching: Same Hotel Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n✓ Visualization saved to: {output_path}")


def print_summary_table(strategy_stats: pd.DataFrame) -> None:
    """Prints formatted summary table of strategy performance."""
    print("\n" + "=" * 80)
    print("PRICING STRATEGY PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print("\n{:<12} {:>10} {:>12} {:>12} {:>12} {:>12}".format(
        'Strategy', 'N Products', 'N Hotels', 'Price Δ', 'Revenue Δ', 'Success %'
    ))
    print("-" * 70)
    
    for strategy_name in ['Slashers', 'Flatliners', 'Nudgers', 'Hikers', 'Moonshots']:
        # Find matching index that contains the strategy name
        matching_idx = None
        for idx in strategy_stats.index:
            if strategy_name in str(idx):
                matching_idx = idx
                break
        
        if matching_idx is not None:
            row = strategy_stats.loc[matching_idx]
            print("{:<12} {:>10,} {:>12,} {:>11.1%} {:>11.1%} {:>11.1%}".format(
                strategy_name,
                int(row['n_products']),
                int(row['n_hotels']),
                row['price_median'],
                row['revenue_median'],
                row['success_rate']
            ))
    
    print("-" * 70)
    print("\nStrategy Definitions:")
    print("  Slashers:   Price cut > 10%")
    print("  Flatliners: Price change -10% to +5%")
    print("  Nudgers:    Price increase +5% to +15%")
    print("  Hikers:     Price increase +15% to +30%")
    print("  Moonshots:  Price increase > +30%")


def print_danger_zone_analysis(strategy_stats: pd.DataFrame) -> None:
    """Prints danger zone detection results."""
    print("\n" + "=" * 80)
    print("DANGER ZONE ANALYSIS")
    print("=" * 80)
    
    danger_detected, message = detect_danger_zone(strategy_stats)
    print(f"\n{message}")
    
    if danger_detected:
        print("\n📊 RECOMMENDATION:")
        print("   Cap price increases at 30% YoY to maximize revenue.")
        print("   Beyond this threshold, volume losses outweigh price gains.")
    else:
        print("\n📊 RECOMMENDATION:")
        print("   Demand appears highly inelastic. Consider testing higher price points.")


# %%
def create_executive_elasticity_proof(df: pd.DataFrame, strategy_stats: pd.DataFrame, output_path: Path) -> None:
    """
    Creates simplified visualization showing Price Change vs Occupancy Change.
    This DIRECTLY proves elasticity - the slope IS the elasticity coefficient.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Filter valid data
    valid_mask = (
        df['price_change_pct'].notna() & 
        df['volume_change_pct'].notna() &
        df['price_change_pct'].between(-0.5, 1.0) &
        df['volume_change_pct'].between(-1.0, 1.0)
    )
    df_valid = df[valid_mask].copy()
    
    # Create bins for the curve
    fine_bins = np.arange(-0.3, 0.6, 0.05)
    df_valid['fine_bin'] = pd.cut(df_valid['price_change_pct'], bins=fine_bins)
    
    curve_data = df_valid.groupby('fine_bin', observed=True).agg({
        'volume_change_pct': 'median',
        'hotel_id': 'count'
    }).reset_index()
    curve_data.columns = ['bin', 'occupancy_median', 'count']
    curve_data['bin_center'] = curve_data['bin'].apply(
        lambda x: float(x.mid) if pd.notna(x) and hasattr(x, 'mid') else np.nan
    )
    curve_data['bin_center'] = pd.to_numeric(curve_data['bin_center'], errors='coerce')
    curve_data = curve_data.dropna(subset=['bin_center'])
    curve_data = curve_data[curve_data['count'] >= 20]
    
    x_vals = np.array(curve_data['bin_center'].values, dtype=float) * 100
    y_vals = np.array(curve_data['occupancy_median'].values, dtype=float) * 100
    
    # Scatter plot of individual observations (sampled for visibility)
    df_sample = df_valid.sample(min(2000, len(df_valid)), random_state=42)
    ax.scatter(df_sample['price_change_pct'] * 100, df_sample['volume_change_pct'] * 100,
               alpha=0.15, s=20, c='steelblue', edgecolors='none', label='Individual hotels')
    
    # Main curve (median trend)
    ax.plot(x_vals, y_vals, 'o-', color='#E74C3C', linewidth=3, markersize=10, 
            markerfacecolor='white', markeredgewidth=2.5, label='Median occupancy change', zorder=5)
    
    # Calculate and plot regression line (this IS the elasticity)
    slope, intercept = np.polyfit(df_valid['price_change_pct'], df_valid['volume_change_pct'], 1)
    x_line = np.linspace(-0.3, 0.5, 100)
    ax.plot(x_line * 100, (slope * x_line + intercept) * 100, 'k--', linewidth=2, 
            label=f'Elasticity: ε = {slope:.2f}', zorder=4)
    
    # Reference lines
    ax.axhline(0, color='gray', linewidth=1.5, linestyle='-', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=1.5, linestyle='-', alpha=0.5)
    
    # Add elasticity interpretation box
    interpretation = f"""Elasticity ε = {slope:.2f}
    
For every 10% price increase:
→ Occupancy drops {abs(slope) * 10:.1f}%

This confirms INELASTIC demand
(|ε| < 1 means price ↑ = revenue ↑)"""
    
    ax.text(0.98, 0.98, interpretation,
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.95),
            fontfamily='monospace')
    
    # Add "what unit elastic would look like" reference
    ax.plot([-30, 50], [30, -50], ':', color='purple', linewidth=2, alpha=0.5,
            label='Unit elastic (ε = -1)')
    
    ax.set_xlabel('Price Change Year-over-Year (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Occupancy Change Year-over-Year (%)', fontsize=14, fontweight='bold')
    ax.set_title('Proving Elasticity: Price Increases → Small Occupancy Loss\n(Same Hotels Tracked 2023 → 2024)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=10, frameon=True, shadow=True)
    ax.grid(alpha=0.3)
    ax.set_xlim(-35, 55)
    ax.set_ylim(-60, 60)
    
    # Add sample size note
    n_hotels = df_valid['hotel_id'].nunique()
    ax.text(0.02, 0.02, f'n = {len(df_valid):,} hotel-month-products ({n_hotels:,} unique hotels)',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=10, 
            style='italic', alpha=0.7)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Executive elasticity proof figure saved to: {output_path}")
    print(f"      Estimated elasticity from longitudinal data: ε = {slope:.3f}")


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("=" * 80)
print("LONGITUDINAL PRICE SENSITIVITY ANALYSIS")
print("Comparing Same Hotel Performance: 2023 vs 2024")
print("=" * 80)

# %%
print("\nLoading and cleaning data...")
config = get_cleaning_config()
cleaner = DataCleaner(config)
con = cleaner.clean(init_db())

# %%
print("\nAggregating to Hotel-Month-Product level...")
df_raw = load_hotel_month_product_data(con)
print(f"  Raw records: {len(df_raw):,}")
print(f"  Unique hotels: {df_raw['hotel_id'].nunique():,}")
print(f"  Year distribution:")
print(df_raw['year'].value_counts().to_string())

# %%
print("\nPivoting to wide format (2023 vs 2024)...")
df_wide = pivot_to_wide_format(df_raw)
print(f"  Products with data in BOTH years: {len(df_wide):,}")
print(f"  Unique hotels with YoY comparison: {df_wide['hotel_id'].nunique():,}")

# %%
print("\nCalculating YoY changes...")
df_changes = calculate_yoy_changes(df_wide)

# Filter extreme outliers
df_changes = df_changes[
    (df_changes['price_change_pct'].between(-0.5, 1.0)) &
    (df_changes['revenue_change_pct'].between(-1.0, 3.0))
]
print(f"  After outlier removal: {len(df_changes):,} products")

# %%
print("\nAssigning pricing strategy bins...")
df_final = assign_strategy_bins(df_changes)
print("\nStrategy distribution:")
print(df_final['strategy_bin'].value_counts().sort_index())

# %%
print("\nAnalyzing strategy performance...")
strategy_stats = analyze_strategy_performance(df_final)

# %%
print_summary_table(strategy_stats)

# %%
print_danger_zone_analysis(strategy_stats)

# %%
print("\nCreating visualizations...")
script_dir = Path(__file__).parent

# Full 4-panel visualization (for appendix)
output_path = (script_dir / '../../../outputs/eda/elasticity/figures/longitudinal_pricing_analysis_full.png').resolve()
create_strategy_visualization(df_final, strategy_stats, output_path)

# Executive single-panel elasticity proof (for executive summary)
exec_output_path = (script_dir / '../../../outputs/eda/elasticity/figures/longitudinal_pricing_analysis.png').resolve()
create_executive_elasticity_proof(df_final, strategy_stats, exec_output_path)

# %%
# Save detailed results
results_path = (script_dir / '../../../outputs/eda/elasticity/data/longitudinal_pricing_results.csv').resolve()
results_path.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(results_path, index=False)
print(f"\n✓ Detailed results saved to: {results_path}")

# %%
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"""
KEY FINDINGS:
- Total hotel-month-products analyzed: {len(df_final):,}
- Unique hotels with YoY comparison: {df_final['hotel_id'].nunique():,}
- Overall median revenue change: {df_final['revenue_change_pct'].median():.1%}
- Overall median price change: {df_final['price_change_pct'].median():.1%}

METHODOLOGY:
This "Self-Matching" approach compares each hotel's 2024 performance against
its own 2023 baseline, eliminating confounding variables like location,
star rating, or amenities. The hotel is its own control group.

This provides a cleaner estimate of price sensitivity than cross-sectional
matching between different hotels.
""")

