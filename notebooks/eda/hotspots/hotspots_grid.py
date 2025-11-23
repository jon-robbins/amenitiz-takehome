"""
Baseline hotspot detection via fixed spatial grid aggregation.
Produces summary tables and a scatter plot suitable for choropleth seeding.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebooks.eda.hotspots.spatial_utils import (  # noqa: E402
    ensure_output_dir,
    load_clean_booking_locations,
)


def assign_grid_bins(df: pd.DataFrame, resolution_deg: float = 0.1) -> pd.DataFrame:
    """
    Bin latitude/longitude into a fixed grid and aggregate metrics.
    """
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")

    df = df.copy()
    df["lat_bin"] = (np.floor(df["latitude"] / resolution_deg) * resolution_deg).round(6)
    df["lon_bin"] = (np.floor(df["longitude"] / resolution_deg) * resolution_deg).round(6)

    grouped = (
        df.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(
            booking_count=("booking_id", "count"),
            total_revenue=("total_price", "sum"),
            arrival_min=("arrival_date", "min"),
            arrival_max=("arrival_date", "max"),
        )
        .sort_values("booking_count", ascending=False)
    )

    grouped["centroid_lat"] = grouped["lat_bin"] + (resolution_deg / 2)
    grouped["centroid_lon"] = grouped["lon_bin"] + (resolution_deg / 2)
    grouped["avg_revenue_per_booking"] = grouped["total_revenue"] / grouped["booking_count"]
    return grouped


def plot_grid_hotspots(grouped: pd.DataFrame, output_path: Path) -> None:
    """
    Create a scatter plot sized by booking count to visualize hotspots.
    Filter to Spain bounds and add basemap context.
    """
    if grouped.empty:
        print("No data available for plotting grid hotspots.")
        return
    
    # Filter to Spain geographic bounds
    spain_mask = (
        (grouped["centroid_lat"] >= 35) & (grouped["centroid_lat"] <= 44) &
        (grouped["centroid_lon"] >= -10) & (grouped["centroid_lon"] <= 5)
    )
    grouped_spain = grouped[spain_mask].copy()
    
    if grouped_spain.empty:
        print("Warning: No grid cells within Spain bounds.")
        return
    
    outliers_count = len(grouped) - len(grouped_spain)
    if outliers_count > 0:
        print(f"Filtered {outliers_count} grid cells outside Spain bounds.")

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Size by booking count, color by revenue
    scatter = ax.scatter(
        grouped_spain["centroid_lon"],
        grouped_spain["centroid_lat"],
        s=np.clip(grouped_spain["booking_count"] * 5, 20, 800),
        c=grouped_spain["avg_revenue_per_booking"],
        cmap="plasma",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.3,
    )

    ax.set_title("Grid-Based Booking Hotspots (Spain)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_xlim(-10, 5)
    ax.set_ylim(35, 44)
    cbar = plt.colorbar(scatter, ax=ax, label="Avg Revenue per Booking (€)")
    cbar.set_label("Avg Revenue per Booking (€)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Add legend for size
    for size_val in [100, 1000, 5000]:
        if size_val <= grouped_spain["booking_count"].max():
            ax.scatter([], [], s=np.clip(size_val * 5, 20, 800), c='gray', 
                      alpha=0.6, edgecolor='k', linewidth=0.3,
                      label=f'{size_val:,} bookings')
    ax.legend(scatterpoints=1, frameon=True, labelspacing=2, title='Grid Cell Size',
             loc='upper left', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    output_dir = ensure_output_dir("outputs/hotspots")
    figure_dir = ensure_output_dir("outputs/figures")

    df = load_clean_booking_locations()
    print(f"Loaded {len(df):,} cleaned bookings with location data.")

    resolution_deg = 0.1
    grouped = assign_grid_bins(df, resolution_deg=resolution_deg)

    grouped.to_csv(output_dir / "grid_hotspots.csv", index=False)
    grouped.head(50).to_csv(output_dir / "grid_hotspots_top50.csv", index=False)
    print(f"Saved aggregated grid hotspots to {output_dir}.")

    plot_path = figure_dir / "grid_hotspots.png"
    plot_grid_hotspots(grouped, plot_path)
    print(f"Saved grid hotspot plot to {plot_path}.")


if __name__ == "__main__":
    main()


