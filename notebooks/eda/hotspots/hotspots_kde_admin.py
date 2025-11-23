"""
Hierarchical hotspot scoring using kernel density estimation and
administrative overlays (municipal polygons when available, otherwise a fallback grid).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

GEOSPAIN_PATH = PROJECT_ROOT / "notebooks" / "eda" / "geo_data" / "GeoSpain"
if GEOSPAIN_PATH.exists():
    sys.path.insert(0, str(GEOSPAIN_PATH))
    pkg_path = GEOSPAIN_PATH / "geospain"
    if pkg_path.exists():
        sys.path.insert(0, str(pkg_path))

try:
    from geospain.gadm import GADM  # type: ignore  # noqa: E402
except ImportError:
    GADM = None  # type: ignore

from notebooks.eda.hotspots.spatial_utils import (  # noqa: E402
    ensure_output_dir,
    load_clean_booking_locations,
)

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
except ImportError as exc:  # pragma: no cover - dependency check
    raise RuntimeError(
        "geopandas and shapely are required for the KDE administrative overlay method."
    ) from exc

# Ensure GDAL can regenerate missing SHX indexes if a shapefile is incomplete
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")


def load_admin_polygons(
    bbox: dict[str, float],
    fallback_resolution_deg: float = 1.0,
    municipalities_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """
    Load municipal polygons if available, else construct a fallback grid.
    """
    if municipalities_path and municipalities_path.exists():
        gdf = gpd.read_file(municipalities_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf

    # Try geospain's GADM municipal boundaries (level 2) as backup
    if GADM is not None:
        try:
            print("Loading municipal polygons from geospain GADM level 2.")
            gadm = GADM(country="ESP", resolution=2, admin_level="municipality")
            gdf = gadm.get_geodataframe()
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.rename(columns={"gadm_id": "admin_id"})
            return gdf
        except Exception as exc:  # pragma: no cover
            print(f"geospain load failed with {exc}; falling back to grid.")

    print("Municipalities file not found. Using fallback grid polygons.")
    lats = np.arange(
        math.floor(bbox["min_lat"]),
        math.ceil(bbox["max_lat"]),
        fallback_resolution_deg,
    )
    lons = np.arange(
        math.floor(bbox["min_lon"]),
        math.ceil(bbox["max_lon"]),
        fallback_resolution_deg,
    )
    polygons = []
    ids = []
    idx = 0
    for lat in lats:
        for lon in lons:
            polygon = Polygon(
                [
                    (lon, lat),
                    (lon + fallback_resolution_deg, lat),
                    (lon + fallback_resolution_deg, lat + fallback_resolution_deg),
                    (lon, lat + fallback_resolution_deg),
                ]
            )
            polygons.append(polygon)
            ids.append(f"grid_{idx}")
            idx += 1
    gdf = gpd.GeoDataFrame({"admin_id": ids}, geometry=polygons, crs="EPSG:4326")
    return gdf


def build_kde(df: pd.DataFrame, bandwidth_km: float = 20.0) -> Tuple[KernelDensity, float]:
    """
    Fit a Kernel Density Estimator on latitude/longitude converted to km.
    """
    lat_mean = df["latitude"].mean()
    lat_km = df["latitude"] * 111.0
    lon_km = df["longitude"] * 111.0 * math.cos(math.radians(lat_mean))
    coords = np.vstack([lat_km, lon_km]).T

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth_km)
    kde.fit(coords)
    return kde, lat_mean


def evaluate_polygons(
    df: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
    kde: KernelDensity,
    lat_mean: float,
) -> gpd.GeoDataFrame:
    """
    Evaluate KDE at polygon centroids and count bookings per polygon.
    """
    points = gpd.GeoDataFrame(
        df[["booking_id", "total_price"]],
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    polygons = polygons.to_crs("EPSG:4326").copy()

    # Spatial join to count bookings per polygon
    joined = gpd.sjoin(points, polygons, how="left", predicate="within")
    counts = (
        joined.groupby("index_right")
        .agg(
            booking_count=("booking_id", "count"),
            total_revenue=("total_price", "sum"),
        )
        .rename_axis("index_right")
    )
    polygons["booking_count"] = counts["booking_count"]
    polygons["total_revenue"] = counts["total_revenue"]
    polygons["booking_count"] = polygons["booking_count"].fillna(0)
    polygons["total_revenue"] = polygons["total_revenue"].fillna(0.0)

    centroids = polygons.geometry.centroid
    centroid_lat_km = centroids.y * 111.0
    centroid_lon_km = centroids.x * 111.0 * math.cos(math.radians(lat_mean))
    coords = np.vstack([centroid_lat_km, centroid_lon_km]).T
    log_density = kde.score_samples(coords)
    polygons["density_score"] = np.exp(log_density)
    polygons["avg_revenue_per_booking"] = np.where(
        polygons["booking_count"] > 0,
        polygons["total_revenue"] / polygons["booking_count"],
        0.0,
    )
    polygons["hotspot_rank"] = polygons["density_score"].rank(ascending=False, method="dense")
    return polygons


def plot_choropleth(polygons: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Plot choropleth of density scores.
    """
    if polygons.empty:
        print("No polygons available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    polygons.plot(
        column="density_score",
        cmap="OrRd",
        legend=True,
        legend_kwds={"label": "Density Score"},
        linewidth=0.1,
        ax=ax,
    )
    ax.set_title("KDE-based Booking Hotspots (Administrative Overlay)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    output_dir = ensure_output_dir("outputs/hotspots")
    figure_dir = ensure_output_dir("outputs/figures")

    df = load_clean_booking_locations()
    print(f"Loaded {len(df):,} cleaned bookings with location data.")

    kde, lat_mean = build_kde(df, bandwidth_km=20.0)
    bbox = {
        "min_lat": df["latitude"].min(),
        "max_lat": df["latitude"].max(),
        "min_lon": df["longitude"].min(),
        "max_lon": df["longitude"].max(),
    }

    municipalities_paths = [
        PROJECT_ROOT / "notebooks" / "eda" / "geo_obj" / "es_municipal_limits.shp",
        PROJECT_ROOT / "data" / "spain_municipalities.geojson",
    ]
    selected_path = next((path for path in municipalities_paths if path.exists()), None)
    polygons = load_admin_polygons(bbox, municipalities_path=selected_path)
    scored = evaluate_polygons(df, polygons, kde, lat_mean)

    scored.to_file(output_dir / "kde_hotspots.geojson", driver="GeoJSON")
    scored.drop(columns="geometry").to_csv(output_dir / "kde_hotspot_scores.csv", index=False)
    print(f"Saved KDE hotspot scores to {output_dir}.")

    plot_path = figure_dir / "kde_hotspots.png"
    plot_choropleth(scored, plot_path)
    print(f"Saved KDE hotspot choropleth to {plot_path}.")


if __name__ == "__main__":
    main()


