"""
Hotspot detection using DBSCAN clustering on geographic coordinates.
Outputs cluster summaries and scatter plot for choropleth seeding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import using absolute path - add the hotspots directory to path
HOTSPOTS_DIR = Path(__file__).resolve().parent
if str(HOTSPOTS_DIR) not in sys.path:
    sys.path.insert(0, str(HOTSPOTS_DIR))

import spatial_utils  # noqa: E402
ensure_output_dir = spatial_utils.ensure_output_dir
load_clean_booking_locations = spatial_utils.load_clean_booking_locations


def run_dbscan(df: pd.DataFrame, eps_km: float = 5.0, min_samples: int = 25) -> pd.DataFrame:
    """
    Execute DBSCAN on lat/lon using haversine distance.
    eps_km: neighborhood radius in kilometers.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    radians = np.radians(df[["latitude", "longitude"]].to_numpy())
    eps = eps_km / 6371.0  # Earth radius in km to radians

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    labels = dbscan.fit_predict(radians)

    df = df.copy()
    df["cluster_id"] = labels
    return df


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize booking metrics per detected cluster.
    """
    clusters = (
        df[df["cluster_id"] >= 0]
        .groupby("cluster_id", as_index=False)
        .agg(
            booking_count=("booking_id", "count"),
            total_revenue=("total_price", "sum"),
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
        )
        .sort_values("booking_count", ascending=False)
    )
    clusters["avg_revenue_per_booking"] = clusters["total_revenue"] / clusters["booking_count"]
    return clusters


def plot_dbscan(df: pd.DataFrame, clusters: pd.DataFrame, output_path: Path) -> None:
    """
    Scatter plot of DBSCAN clusters with noise highlighted.
    Filter to Spain bounds to avoid outliers.
    """
    # Filter to Spain geographic bounds (with small buffer)
    spain_mask = (
        (df["latitude"] >= 35) & (df["latitude"] <= 44) &
        (df["longitude"] >= -10) & (df["longitude"] <= 5)
    )
    df_spain = df[spain_mask].copy()
    
    if df_spain.empty:
        print("Warning: No data within Spain bounds for plotting.")
        return
    
    outliers_count = len(df) - len(df_spain)
    if outliers_count > 0:
        print(f"Filtered {outliers_count} outlier coordinates outside Spain bounds.")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    noise = df_spain[df_spain["cluster_id"] == -1]
    core = df_spain[df_spain["cluster_id"] >= 0]

    if not core.empty:
        scatter = ax.scatter(
            core["longitude"],
            core["latitude"],
            c=core["cluster_id"],
            cmap="tab20",
            s=20,
            alpha=0.7,
            linewidth=0,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster ID")

    if not noise.empty:
        ax.scatter(
            noise["longitude"],
            noise["latitude"],
            c="lightgrey",
            s=10,
            alpha=0.3,
            label="Noise",
        )
        ax.legend(loc="upper right")

    ax.set_title("DBSCAN Booking Hotspots (Spain)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_xlim(-10, 5)
    ax.set_ylim(35, 44)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    output_dir = ensure_output_dir("outputs/hotspots")
    figure_dir = ensure_output_dir("outputs/figures")

    df = load_clean_booking_locations()
    print(f"Loaded {len(df):,} cleaned bookings with location data.")

    eps_km = 5.0
    min_samples = 25
    clustered = run_dbscan(df, eps_km=eps_km, min_samples=min_samples)
    clusters = summarize_clusters(clustered)

    clustered.to_csv(output_dir / "dbscan_labeled_bookings.csv", index=False)
    clusters.to_csv(output_dir / "dbscan_cluster_summary.csv", index=False)
    print(f"Saved DBSCAN outputs to {output_dir}.")

    plot_path = figure_dir / "dbscan_hotspots.png"
    plot_dbscan(clustered, clusters, plot_path)
    print(f"Saved DBSCAN plot to {plot_path}.")


if __name__ == "__main__":
    main()


