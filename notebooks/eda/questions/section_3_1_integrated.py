"""
Section 3.1: Geographic Hotspot Analysis (Integrated)

Combines city-level analysis with three hotspot detection methodologies:
1. Grid-based aggregation
2. DBSCAN clustering
3. KDE with administrative overlay

All methods use cleaned data (reception halls and missing locations excluded).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lib.db import init_db  # noqa: E402
from lib.data_validator import validate_and_clean  # noqa: E402
from notebooks.eda.hotspots.spatial_utils import (  # noqa: E402
    ensure_output_dir,
    load_clean_booking_locations,
)
from notebooks.eda.hotspots.hotspots_grid import assign_grid_bins  # noqa: E402
from notebooks.eda.hotspots.hotspots_dbscan import run_dbscan, summarize_clusters  # noqa: E402
from notebooks.data_prep.city_consolidation import (  # noqa: E402
    create_city_mapping,
    apply_city_mapping,
    print_mapping_examples,
)


def filter_spain_bounds(
    df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"
) -> pd.DataFrame:
    """Filter dataframe to Spain geographic bounds."""
    spain_mask = (
        (df[lat_col] >= 35)
        & (df[lat_col] <= 44)
        & (df[lon_col] >= -10)
        & (df[lon_col] <= 5)
    )
    filtered = df[spain_mask].copy()
    outliers = len(df) - len(filtered)
    if outliers > 0:
        print(f"  Filtered {outliers:,} outlier coordinates outside Spain bounds.")
    return filtered


def load_city_analysis(con, use_canonical: bool = True) -> tuple[pd.DataFrame, dict[str, str] | None]:
    """
    Load city-level supply and demand metrics.
    
    Args:
        con: Database connection
        use_canonical: If True, apply TF-IDF city name matching to consolidate cities
    
    Returns:
        (city_df, city_mapping) where city_mapping is None if use_canonical=False
    """
    # Load raw city data
    df = con.execute(
        """
        SELECT 
            hl.city,
            b.id as booking_id,
            hl.hotel_id,
            br.room_id,
            br.total_price,
            (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) as nights
        FROM bookings b
        JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('confirmed', 'Booked')
          AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
          AND hl.city IS NOT NULL
    """
    ).fetchdf()
    
    city_mapping = None
    if use_canonical:
        # Create city mapping using TF-IDF
        print("\nApplying TF-IDF city name matching...")
        city_mapping = create_city_mapping(
            df,
            city_col="city",
            min_bookings_canonical=1000,
            similarity_threshold=0.6,
            verbose=True,
        )
        df = apply_city_mapping(df, city_mapping, city_col="city", new_col="city_canonical")
        city_col = "city_canonical"
    else:
        city_col = "city"
    
    # Aggregate by city
    city_df = df.groupby(city_col, as_index=False).agg(
        num_hotels=("hotel_id", "nunique"),
        num_room_configs=("room_id", "nunique"),
        num_bookings=("booking_id", "nunique"),
        num_booked_rooms=("booking_id", "count"),
        total_revenue=("total_price", "sum"),
        avg_daily_price=("total_price", lambda x: (x / df.loc[x.index, "nights"]).mean()),
        median_daily_price=("total_price", lambda x: (x / df.loc[x.index, "nights"]).median()),
    ).sort_values("num_bookings", ascending=False)
    
    city_df.rename(columns={city_col: "city"}, inplace=True)
    
    return city_df, city_mapping


def create_integrated_visualizations(
    city_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    dbscan_df: pd.DataFrame,
    dbscan_clusters: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create comprehensive visualization comparing all three methods."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Hotspot visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Grid hotspots
    grid_spain = filter_spain_bounds(
        grid_df, lat_col="centroid_lat", lon_col="centroid_lon"
    )
    if not grid_spain.empty:
        scatter1 = ax1.scatter(
            grid_spain["centroid_lon"],
            grid_spain["centroid_lat"],
            s=np.clip(grid_spain["booking_count"] * 5, 20, 800),
            c=grid_spain["avg_revenue_per_booking"],
            cmap="plasma",
            alpha=0.7,
            edgecolor="k",
            linewidth=0.3,
        )
        plt.colorbar(scatter1, ax=ax1, label="Avg Revenue (€)")
    ax1.set_title("Grid-Based Hotspots", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim(-10, 5)
    ax1.set_ylim(35, 44)
    ax1.grid(True, alpha=0.3)

    # DBSCAN hotspots
    dbscan_spain = filter_spain_bounds(dbscan_df)
    if not dbscan_spain.empty:
        core = dbscan_spain[dbscan_spain["cluster_id"] >= 0]
        noise = dbscan_spain[dbscan_spain["cluster_id"] == -1]
        if not core.empty:
            scatter2 = ax2.scatter(
                core["longitude"],
                core["latitude"],
                c=core["cluster_id"],
                cmap="tab20",
                s=20,
                alpha=0.7,
                linewidth=0,
            )
            plt.colorbar(scatter2, ax=ax2, label="Cluster ID")
        if not noise.empty:
            ax2.scatter(
                noise["longitude"],
                noise["latitude"],
                c="lightgrey",
                s=10,
                alpha=0.3,
                label="Noise",
            )
    ax2.set_title("DBSCAN Clusters", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xlim(-10, 5)
    ax2.set_ylim(35, 44)
    ax2.grid(True, alpha=0.3)

    # Top cities by bookings
    top_15 = city_df.head(15)
    bars = ax3.barh(
        range(len(top_15)),
        top_15["num_bookings"],
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax3.set_yticks(range(len(top_15)))
    ax3.set_yticklabels(top_15["city"], fontsize=9)
    ax3.set_xlabel("Number of Bookings")
    ax3.set_title("Top 15 Cities (Named)", fontsize=12, fontweight="bold")
    ax3.invert_yaxis()
    ax3.grid(axis="x", alpha=0.3)

    # Row 2: Comparative metrics
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Grid: Top hotspots by booking count
    top_grid = grid_spain.head(15)
    ax4.barh(
        range(len(top_grid)),
        top_grid["booking_count"],
        color="coral",
        edgecolor="black",
        alpha=0.7,
    )
    ax4.set_yticks(range(len(top_grid)))
    ax4.set_yticklabels(
        [f"({lat:.2f}, {lon:.2f})" for lat, lon in zip(top_grid["centroid_lat"], top_grid["centroid_lon"])],
        fontsize=8,
    )
    ax4.set_xlabel("Bookings per Grid Cell")
    ax4.set_title("Top 15 Grid Cells", fontsize=12, fontweight="bold")
    ax4.invert_yaxis()
    ax4.grid(axis="x", alpha=0.3)

    # DBSCAN: Cluster size distribution
    if not dbscan_clusters.empty:
        top_clusters = dbscan_clusters.head(15)
        ax5.barh(
            range(len(top_clusters)),
            top_clusters["booking_count"],
            color="mediumseagreen",
            edgecolor="black",
            alpha=0.7,
        )
        ax5.set_yticks(range(len(top_clusters)))
        ax5.set_yticklabels([f"Cluster {int(c)}" for c in top_clusters["cluster_id"]], fontsize=9)
        ax5.set_xlabel("Bookings per Cluster")
        ax5.set_title("Top 15 DBSCAN Clusters", fontsize=12, fontweight="bold")
        ax5.invert_yaxis()
        ax5.grid(axis="x", alpha=0.3)

    # City: Price vs bookings scatter
    ax6.scatter(
        city_df["num_bookings"],
        city_df["median_daily_price"],
        c=city_df["num_hotels"],
        s=100,
        alpha=0.6,
        cmap="viridis",
        edgecolors="black",
    )
    ax6.set_xlabel("Number of Bookings")
    ax6.set_ylabel("Median Daily Price (€)")
    ax6.set_title("City Price vs Volume", fontsize=12, fontweight="bold")
    ax6.set_xscale("log")
    ax6.grid(True, alpha=0.3)

    # Row 3: Distribution comparisons
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    # Grid: Booking count distribution
    ax7.hist(
        grid_spain["booking_count"],
        bins=50,
        color="coral",
        edgecolor="black",
        alpha=0.7,
    )
    ax7.set_xlabel("Bookings per Grid Cell")
    ax7.set_ylabel("Frequency")
    ax7.set_title("Grid Cell Size Distribution", fontsize=12, fontweight="bold")
    ax7.set_yscale("log")
    ax7.grid(True, alpha=0.3)

    # DBSCAN: Cluster size distribution
    if not dbscan_clusters.empty:
        ax8.hist(
            dbscan_clusters["booking_count"],
            bins=50,
            color="mediumseagreen",
            edgecolor="black",
            alpha=0.7,
        )
        ax8.set_xlabel("Bookings per Cluster")
        ax8.set_ylabel("Frequency")
        ax8.set_title("DBSCAN Cluster Size Distribution", fontsize=12, fontweight="bold")
        ax8.set_yscale("log")
        ax8.grid(True, alpha=0.3)

    # City: Bookings distribution
    ax9.hist(
        city_df["num_bookings"],
        bins=50,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax9.set_xlabel("Bookings per City")
    ax9.set_ylabel("Frequency")
    ax9.set_title("City Size Distribution", fontsize=12, fontweight="bold")
    ax9.set_yscale("log")
    ax9.grid(True, alpha=0.3)

    fig.suptitle(
        "Section 3.1: Geographic Hotspot Analysis - Three Methodologies",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary_analysis(
    city_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    dbscan_df: pd.DataFrame,
    dbscan_clusters: pd.DataFrame,
    geo_df: pd.DataFrame,
    city_mapping: dict[str, str] | None = None,
) -> None:
    """Print comprehensive summary comparing all methodologies."""
    print("\n" + "=" * 80)
    print("SECTION 3.1: GEOGRAPHIC HOTSPOT ANALYSIS - SUMMARY")
    print("=" * 80)
    
    if city_mapping:
        print("\n--- CITY NAME CONSOLIDATION (TF-IDF) ---")
        changes = {k: v for k, v in city_mapping.items() if k != v}
        print(f"Original unique cities: {len(city_mapping)}")
        print(f"Consolidated to: {len(set(city_mapping.values()))}")
        print(f"Reduction: {len(city_mapping) - len(set(city_mapping.values()))} cities ({(len(city_mapping) - len(set(city_mapping.values())))/len(city_mapping)*100:.1f}%)")
        print(f"Name changes applied: {len(changes)}")
        print("\nExample consolidations:")
        for i, (orig, canon) in enumerate(list(changes.items())[:5]):
            print(f"  • {orig} → {canon}")

    # Filter to Spain bounds
    grid_spain = filter_spain_bounds(
        grid_df, lat_col="centroid_lat", lon_col="centroid_lon"
    )
    dbscan_spain = filter_spain_bounds(dbscan_df)
    geo_spain = filter_spain_bounds(geo_df)

    print("\n--- DATASET OVERVIEW ---")
    print(f"Total bookings analyzed: {len(geo_spain):,}")
    print(f"Total cities with named locations: {len(city_df):,}")
    print(f"Total revenue: €{city_df['total_revenue'].sum() / 1_000_000:.1f}M")

    print("\n--- METHOD 1: GRID-BASED AGGREGATION ---")
    print(f"Grid resolution: 0.1° (~11km)")
    print(f"Total grid cells with bookings: {len(grid_spain):,}")
    print(f"Bookings per cell (median): {grid_spain['booking_count'].median():.0f}")
    print(f"Bookings per cell (mean): {grid_spain['booking_count'].mean():.1f}")
    print(f"Largest grid cell: {grid_spain['booking_count'].max():,} bookings")
    print(f"Top 10 cells capture: {grid_spain.head(10)['booking_count'].sum() / grid_spain['booking_count'].sum() * 100:.1f}% of bookings")

    print("\n--- METHOD 2: DBSCAN CLUSTERING ---")
    print(f"Parameters: eps=5km, min_samples=25")
    print(f"Clusters detected: {len(dbscan_clusters):,}")
    noise_count = (dbscan_spain["cluster_id"] == -1).sum()
    print(f"Noise points (outliers): {noise_count:,} ({noise_count / len(dbscan_spain) * 100:.1f}%)")
    if not dbscan_clusters.empty:
        print(f"Bookings per cluster (median): {dbscan_clusters['booking_count'].median():.0f}")
        print(f"Bookings per cluster (mean): {dbscan_clusters['booking_count'].mean():.1f}")
        print(f"Largest cluster: {dbscan_clusters['booking_count'].max():,} bookings")
        print(f"Top 10 clusters capture: {dbscan_clusters.head(10)['booking_count'].sum() / dbscan_clusters['booking_count'].sum() * 100:.1f}% of clustered bookings")

    print("\n--- METHOD 3: CITY-LEVEL (NAMED LOCATIONS) ---")
    print(f"Named cities: {len(city_df):,}")
    print(f"Bookings per city (median): {city_df['num_bookings'].median():.0f}")
    print(f"Bookings per city (mean): {city_df['num_bookings'].mean():.1f}")
    print(f"Largest city: {city_df.iloc[0]['city']} with {city_df.iloc[0]['num_bookings']:,} bookings")
    print(f"Top 10 cities capture: {city_df.head(10)['num_bookings'].sum() / city_df['num_bookings'].sum() * 100:.1f}% of bookings")

    print("\n--- MARKET CONCENTRATION (Herfindahl Index) ---")
    grid_spain["market_share"] = grid_spain["booking_count"] / grid_spain["booking_count"].sum()
    hhi_grid = (grid_spain["market_share"] ** 2).sum()
    print(f"Grid-based HHI: {hhi_grid:.4f}")

    if not dbscan_clusters.empty:
        dbscan_clusters["market_share"] = (
            dbscan_clusters["booking_count"] / dbscan_clusters["booking_count"].sum()
        )
        hhi_dbscan = (dbscan_clusters["market_share"] ** 2).sum()
        print(f"DBSCAN HHI: {hhi_dbscan:.4f}")

    city_df["market_share"] = city_df["num_bookings"] / city_df["num_bookings"].sum()
    hhi_city = (city_df["market_share"] ** 2).sum()
    print(f"City-based HHI: {hhi_city:.4f}")
    print("\nInterpretation: Lower HHI = more distributed demand")

    print("\n--- TOP 5 HOTSPOTS BY METHOD ---")
    print("\nGrid (lat, lon):")
    for i, row in grid_spain.head(5).iterrows():
        print(
            f"  {i+1}. ({row['centroid_lat']:.2f}, {row['centroid_lon']:.2f}): "
            f"{row['booking_count']:,} bookings, €{row['avg_revenue_per_booking']:.0f} avg"
        )

    if not dbscan_clusters.empty:
        print("\nDBSCAN Clusters:")
        for i, row in dbscan_clusters.head(5).iterrows():
            print(
                f"  {i+1}. Cluster {int(row['cluster_id'])} @ ({row['centroid_lat']:.2f}, {row['centroid_lon']:.2f}): "
                f"{row['booking_count']:,} bookings, €{row['avg_revenue_per_booking']:.0f} avg"
            )

    print("\nCities:")
    for i, row in city_df.head(5).iterrows():
        print(
            f"  {i+1}. {row['city']}: {row['num_bookings']:,} bookings, "
            f"{row['num_hotels']} hotels, €{row['median_daily_price']:.0f} median price"
        )

    print("\n--- PRICING INSIGHTS ---")
    print(f"\nMedian daily price across all cities: €{city_df['median_daily_price'].median():.2f}")
    print(f"Price range: €{city_df['median_daily_price'].min():.0f} - €{city_df['median_daily_price'].max():.0f}")
    print(f"Price std dev: €{city_df['median_daily_price'].std():.2f}")

    high_price_cities = city_df[city_df["median_daily_price"] > 150]
    print(f"\nPremium cities (>€150/night): {len(high_price_cities)}")
    if not high_price_cities.empty:
        print(f"  Top premium: {high_price_cities.nlargest(3, 'median_daily_price')['city'].tolist()}")

    print("\n--- RECOMMENDATIONS ---")
    print("\n1. GRID METHOD:")
    print("   ✓ Best for: Uniform spatial analysis, heat mapping")
    print("   ✓ Captures: Fine-grained geographic patterns")
    print("   ✗ Limitation: Arbitrary boundaries, doesn't respect natural clusters")

    print("\n2. DBSCAN METHOD:")
    print("   ✓ Best for: Discovering natural geographic clusters, metro areas")
    print("   ✓ Captures: Density-based hotspots without predefined boundaries")
    print("   ✗ Limitation: Sensitive to parameters, treats sparse areas as noise")

    print("\n3. CITY METHOD (Named Locations):")
    print("   ✓ Best for: Business reporting, market segmentation")
    print("   ✓ Captures: Meaningful administrative units, easy interpretation")
    print("   ✗ Limitation: Depends on data quality, misses unnamed/miscategorized locations")

    print("\n--- RECOMMENDED APPROACH FOR PRICING MODEL ---")
    print("Hybrid strategy:")
    print("  1. Use CITY names as primary geographic feature (interpretable)")
    print("  2. Fall back to DBSCAN cluster ID for bookings with missing/poor city data")
    print("  3. Use GRID coordinates as continuous lat/lon features for fine-tuning")
    print("  4. Create 'metro area' tiers based on DBSCAN clusters + city mapping")

    print("\n" + "=" * 80)


def main() -> None:
    """Run integrated geographic hotspot analysis."""
    output_dir = ensure_output_dir("outputs/hotspots")
    figure_dir = ensure_output_dir("outputs/figures")

    print("=" * 80)
    print("SECTION 3.1: GEOGRAPHIC HOTSPOT ANALYSIS (INTEGRATED)")
    print("=" * 80)

    # Load cleaned data
    print("\nLoading cleaned booking data...")
    con = validate_and_clean(
        init_db(),
        verbose=False,
        rooms_to_exclude=["reception_hall"],
        exclude_missing_location_bookings=True,
    )
    geo_df = load_clean_booking_locations()
    print(f"Loaded {len(geo_df):,} bookings with location data.")

    # City analysis with TF-IDF matching
    print("\nRunning city-level analysis with TF-IDF name matching...")
    city_df, city_mapping = load_city_analysis(con, use_canonical=True)
    print(f"Found {len(city_df):,} consolidated cities with bookings.")

    # Grid analysis
    print("\nRunning grid-based hotspot detection...")
    grid_df = assign_grid_bins(geo_df, resolution_deg=0.1)
    print(f"Created {len(grid_df):,} grid cells.")

    # DBSCAN analysis
    print("\nRunning DBSCAN clustering...")
    dbscan_df = run_dbscan(geo_df, eps_km=5.0, min_samples=25)
    dbscan_clusters = summarize_clusters(dbscan_df)
    print(f"Detected {len(dbscan_clusters):,} clusters.")

    # Create integrated visualization
    print("\nCreating integrated visualizations...")
    viz_path = figure_dir / "section_3_1_integrated.png"
    create_integrated_visualizations(
        city_df, grid_df, dbscan_df, dbscan_clusters, viz_path
    )
    print(f"Saved integrated visualization to {viz_path}.")

    # Print summary analysis
    print_summary_analysis(city_df, grid_df, dbscan_df, dbscan_clusters, geo_df, city_mapping)
    
    # Save city mapping if generated
    if city_mapping:
        mapping_path = output_dir / "city_name_mapping.csv"
        mapping_df = pd.DataFrame(
            list(city_mapping.items()), columns=["original_city", "canonical_city"]
        )
        mapping_df.to_csv(mapping_path, index=False)
        print(f"\n✓ Saved city name mapping to {mapping_path}")

    print(f"\n✓ Analysis complete. Outputs saved to {output_dir} and {figure_dir}.")


if __name__ == "__main__":
    main()

