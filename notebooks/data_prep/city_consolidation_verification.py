"""
Interactive hexagon heatmap for city consolidation verification with temporal slider.

This script creates interactive Folium maps with H3 hexagonal spatial indexing
to visualize booking density at the hexagon level (resolution 8, ~1.2km edge length).

Features:
- Temporal slider to explore seasonal patterns week-by-week (weeks 1-52)
- Play button to animate through the year
- Three maps: original cities, consolidated cities, and difference map
"""

import sys
from pathlib import Path
import json
from collections import Counter

import pandas as pd
import numpy as np
import folium
from folium import plugins
from folium.plugins import HeatMap
import h3
import branca.colormap as cm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
from notebooks.data_prep.city_consolidation_v2 import consolidate_cities

# JavaScript files directory
JS_DIR = Path(__file__).parent / 'js'


def load_js_file(filename: str) -> str:
    """Load JavaScript file content."""
    js_path = JS_DIR / filename
    with open(js_path, 'r') as f:
        return f.read()


def load_booking_locations_with_cities(con) -> pd.DataFrame:
    """
    Load booking locations with city names and arrival week.
    
    Returns DataFrame with columns:
    - booking_id
    - hotel_id
    - city_original
    - latitude, longitude
    - arrival_week (1-52)
    - booking_count (always 1 per row)
    """
    print("   Querying booking locations...")
    query = """
        SELECT 
            b.id as booking_id,
            hl.hotel_id,
            hl.city as city_original,
            hl.latitude,
            hl.longitude,
            EXTRACT(WEEK FROM b.arrival_date) as arrival_week,
            1 as booking_count
        FROM bookings b
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE hl.city IS NOT NULL
          AND hl.city != ''
          AND hl.latitude IS NOT NULL
          AND hl.longitude IS NOT NULL
          AND hl.latitude BETWEEN 27 AND 44
          AND hl.longitude BETWEEN -18 AND 5
          AND b.status IN ('confirmed', 'Booked')
    """
    df = con.execute(query).fetchdf()
    
    print(f"   DEBUG: Raw query returned {len(df):,} rows")
    if len(df) > 0:
        print(f"   DEBUG: Sample row: {df.iloc[0].to_dict()}")
    
    # Convert week 0 to 52 (some databases use 0-based weeks)
    df.loc[df['arrival_week'] == 0, 'arrival_week'] = 52
    
    print(f"   Loaded {len(df):,} bookings from {df['hotel_id'].nunique() if len(df) > 0 else 0:,} hotels")
    if len(df) > 0:
        print(f"   Week range: {df['arrival_week'].min():.0f} - {df['arrival_week'].max():.0f}")
    
    return df


def assign_h3_hexagons(df: pd.DataFrame, resolution: int = 8) -> pd.DataFrame:
    """
    Assign H3 hexagon IDs to each location.
    
    Uses vectorized operations for performance.
    """
    print(f"   Assigning H3 hexagons (resolution {resolution})...")
    
    # Vectorized H3 assignment
    df['h3_hex'] = df.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
        axis=1
    )
    
    print(f"   Assigned {df['h3_hex'].nunique():,} unique hexagons")
    
    return df


def aggregate_by_hexagon_and_week(
    df: pd.DataFrame, 
    city_col: str
) -> pd.DataFrame:
    """
    Aggregate bookings by hexagon and week number.
    
    For each (hexagon, week), finds the dominant city (most bookings).
    """
    print(f"   Aggregating by hexagon and week (using '{city_col}')...")
    
    # Group by hexagon, week, and city
    grouped = df.groupby(['h3_hex', 'arrival_week', city_col]).agg({
        'booking_count': 'sum'
    }).reset_index()
    
    # For each (hexagon, week), find dominant city
    idx = grouped.groupby(['h3_hex', 'arrival_week'])['booking_count'].idxmax()
    hex_week_stats = grouped.loc[idx].copy()
    
    # Rename city column to 'dominant_city'
    hex_week_stats = hex_week_stats.rename(columns={city_col: 'dominant_city'})
    
    # Get hexagon centroids
    hex_week_stats['hex_center_lat'] = hex_week_stats['h3_hex'].apply(
        lambda h: h3.cell_to_latlng(h)[0]
    )
    hex_week_stats['hex_center_lon'] = hex_week_stats['h3_hex'].apply(
        lambda h: h3.cell_to_latlng(h)[1]
    )
    
    print(f"   Created {len(hex_week_stats):,} hexagon-week combinations")
    
    return hex_week_stats


def get_city_colors(df: pd.DataFrame, top_n: int = 50) -> dict:
    """
    Generate consistent colors for top N cities.
    
    Uses a hash-based approach for consistency across maps.
    """
    # Get top cities by total bookings
    city_bookings = df.groupby('dominant_city')['booking_count'].sum().sort_values(ascending=False)
    top_cities = city_bookings.head(top_n).index.tolist()
    
    # Generate colors using matplotlib colormap
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab20')
    
    city_colors = {}
    for i, city in enumerate(top_cities):
        if i < 20:
            # Use tab20 for first 20
            rgba = cmap(i / 20)
            city_colors[city] = f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}'
        else:
            # Use hash for remaining cities
            hash_val = hash(city) % 360
            city_colors[city] = f'hsl({hash_val}, 70%, 50%)'
    
    return city_colors


def create_temporal_hexagon_map(
    hex_week_df: pd.DataFrame,
    title: str,
    output_path: Path,
    city_colors: dict
) -> None:
    """
    Create Folium map with week-by-week slider.
    
    Features:
    - Slider to select week (1-52)
    - Play button to animate through the year
    - Hexagons colored by dominant city
    - Opacity based on booking count
    - Tooltips with city, bookings, week
    """
    print(f"   Creating temporal map: {title}")
    
    # Initialize map (centered to include both mainland and Canary Islands)
    m = folium.Map(
        location=[37.5, -5.0],
        zoom_start=5,
        tiles='CartoDB positron',
        control_scale=True
    )
    
    # Prepare temporal data structure for heatmap
    # Format: {week_num: [[lat, lon, intensity], ...]}
    temporal_heatmap_data = {}
    
    # Also prepare hexagon data for optional toggle
    temporal_hexagon_data = {}
    
    for week in range(1, 53):
        week_data = hex_week_df[hex_week_df['arrival_week'] == week]
        temporal_heatmap_data[week] = []
        temporal_hexagon_data[week] = []
        
        for _, row in week_data.iterrows():
            h3_hex = row['h3_hex']
            
            # Get hexagon center for heatmap
            center = h3.cell_to_latlng(h3_hex)
            intensity = float(row['booking_count'])
            temporal_heatmap_data[week].append([center[0], center[1], intensity])
            
            # Get hexagon boundary for optional hexagon view
            boundary = h3.cell_to_boundary(h3_hex)
            coords = [[lon, lat] for lat, lon in boundary]
            coords.append(coords[0])  # Close the polygon
            
            temporal_hexagon_data[week].append({
                'hex_id': h3_hex,
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords]
                },
                'properties': {
                    'city': row['dominant_city'],
                    'bookings': int(row['booking_count']),
                    'week': int(week)
                }
            })
    
    # Convert data to JSON-safe format
    city_colors_json = json.dumps(city_colors)
    temporal_heatmap_json = json.dumps(temporal_heatmap_data)
    temporal_hexagon_json = json.dumps(temporal_hexagon_data)
    
    # Get the map's variable name from Folium
    map_id = m.get_name()
    
    # Load JavaScript modules
    js_config = load_js_file('heatmap_config.js')
    js_renderer = load_js_file('map_renderer.js')
    js_controls = load_js_file('ui_controls.js')
    js_main = load_js_file('temporal_map.js')
    
    # Add custom JavaScript slider control
    slider_html = f"""
    <div id="week-slider-container" style="
        position: fixed;
        bottom: 50px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: white;
        padding: 15px 30px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        width: 700px;
    ">
        <div style="text-align: center; margin-bottom: 10px;">
            <strong style="font-size: 16px;">{title}</strong><br>
            <span style="font-size: 14px;">Week: <span id="week-display">1</span> | Points: <span id="hex-count">0</span> | Bookings: <span id="booking-count">0</span></span>
            <button id="play-btn" style="margin-left: 20px; padding: 5px 15px; cursor: pointer;">▶ Play</button>
            <button id="view-toggle" style="margin-left: 10px; padding: 5px 15px; cursor: pointer; font-size: 11px;">Switch to Hexagons</button>
        </div>
        <div style="margin-bottom: 15px;">
            <label style="font-size: 12px; color: #666; display: block; margin-bottom: 3px;">
                Week Timeline
            </label>
            <input type="range" id="week-slider" min="1" max="52" value="1" 
                   style="width: 100%;" step="1">
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                <span>Week 1 (Jan)</span>
                <span>Week 13 (Apr)</span>
                <span>Week 26 (Jul)</span>
                <span>Week 39 (Oct)</span>
                <span>Week 52 (Dec)</span>
            </div>
        </div>
        <div style="margin-top: 10px;">
            <label style="font-size: 12px; color: #666; display: block; margin-bottom: 3px;">
                Heatmap Radius: <span id="radius-value">15</span>px (Heatmap view only)
            </label>
            <input type="range" id="radius-slider" min="5" max="50" value="15" 
                   style="width: 100%;" step="1">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #999;">
                <span>Precise (5px)</span>
                <span>Balanced (25px)</span>
                <span>Diffuse (50px)</span>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        .leaflet-heatmap-layer {{
            transition: opacity 0.3s ease-in-out;
        }}
    </style>
    
    <!-- Configuration -->
    <script>
    {js_config}
    </script>
    
    <!-- Map Renderer -->
    <script>
    {js_renderer}
    </script>
    
    <!-- UI Controls -->
    <script>
    {js_controls}
    </script>
    
    <!-- Main Temporal Map Controller -->
    <script>
    {js_main}
    </script>
    
    <!-- Initialize with data -->
    <script>
    window.addEventListener('load', function() {{
        const heatmapData = {temporal_heatmap_json};
        const hexagonData = {temporal_hexagon_json};
        const cityColors = {city_colors_json};
        const mapId = '{map_id}';
        
        // Initialize the temporal map
        initTemporalMap(mapId, heatmapData, hexagonData, cityColors);
    }});
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(slider_html))
    
    # Add dual legend (heatmap gradient + city colors for hexagon view)
    top_cities = list(city_colors.keys())[:10]  # Show top 10 cities in legend
    legend_items = []
    for city in top_cities:
        color = city_colors[city]
        legend_items.append(f'<div><span style="display:inline-block; width:15px; height:15px; background-color:{color}; border:1px solid #333; margin-right:5px;"></span>{city}</div>')
    
    legend_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        max-height: 450px;
        overflow-y: auto;
        font-size: 11px;
    ">
        <strong style="font-size: 13px;">Heatmap Legend</strong><br>
        <div style="margin-top: 8px;">
            <div style="background: linear-gradient(to right, blue, cyan, lime, yellow, orange, red); 
                        width: 180px; height: 15px; border: 1px solid #333; border-radius: 3px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 9px; color: #666; margin-top: 2px;">
                <span>Low</span>
                <span>Booking Intensity</span>
                <span>High</span>
            </div>
        </div>
        <hr style="margin: 10px 0;">
        <strong style="font-size: 13px;">Top Cities (Hexagon View)</strong><br>
        <div style="margin-top: 5px;">
            {''.join(legend_items)}
        </div>
        <hr style="margin: 8px 0;">
        <div style="font-size: 10px; color: #666;">
            <strong>Note:</strong> Use toggle button to switch views
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(str(output_path))
    print(f"   ✓ Saved: {output_path.name}")


def create_difference_map(
    original_hex_df: pd.DataFrame,
    consolidated_hex_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Create static map showing where consolidation changed city assignments.
    
    Aggregates across all weeks to show overall changes.
    """
    print("   Creating difference map...")
    
    # Aggregate across all weeks (sum bookings per hexagon)
    original_agg = original_hex_df.groupby('h3_hex').agg({
        'dominant_city': lambda x: x.value_counts().index[0],  # Most common city
        'booking_count': 'sum'
    }).reset_index()
    
    consolidated_agg = consolidated_hex_df.groupby('h3_hex').agg({
        'dominant_city': lambda x: x.value_counts().index[0],
        'booking_count': 'sum'
    }).reset_index()
    
    # Merge on hexagon
    merged = original_agg.merge(
        consolidated_agg,
        on='h3_hex',
        suffixes=('_original', '_consolidated')
    )
    
    # Identify changes
    merged['city_changed'] = merged['dominant_city_original'] != merged['dominant_city_consolidated']
    
    # Initialize map (centered to include both mainland and Canary Islands)
    m = folium.Map(
        location=[37.5, -5.0],
        zoom_start=5,
        tiles='CartoDB positron',
        control_scale=True
    )
    
    # Add hexagons
    for _, row in merged.iterrows():
        h3_hex = row['h3_hex']
        boundary = h3.cell_to_boundary(h3_hex)
        coords = [[lat, lon] for lat, lon in boundary]
        
        if row['city_changed']:
            # Red for changed
            color = '#e74c3c'
            tooltip_text = (
                f"<strong>City Changed</strong><br>"
                f"{row['dominant_city_original']} → {row['dominant_city_consolidated']}<br>"
                f"Bookings: {row['booking_count_original']:.0f}"
            )
        else:
            # Green for unchanged
            color = '#27ae60'
            tooltip_text = (
                f"<strong>Unchanged</strong><br>"
                f"{row['dominant_city_original']}<br>"
                f"Bookings: {row['booking_count_original']:.0f}"
            )
        
        folium.Polygon(
            locations=coords,
            color='#333',
            weight=0.5,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            tooltip=tooltip_text
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    ">
        <strong>City Consolidation</strong><br>
        <span style="color: #e74c3c;">■</span> City Changed<br>
        <span style="color: #27ae60;">■</span> City Unchanged
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(str(output_path))
    print(f"   ✓ Saved: {output_path.name}")
    
    # Print statistics
    changed_count = merged['city_changed'].sum()
    total_count = len(merged)
    changed_pct = (changed_count / total_count) * 100
    
    changed_bookings = merged[merged['city_changed']]['booking_count_original'].sum()
    total_bookings = merged['booking_count_original'].sum()
    changed_bookings_pct = (changed_bookings / total_bookings) * 100
    
    print(f"\n   Difference Map Statistics:")
    print(f"   - Total hexagons: {total_count:,}")
    print(f"   - Hexagons with city changes: {changed_count:,} ({changed_pct:.1f}%)")
    print(f"   - Bookings affected: {changed_bookings:,.0f} ({changed_bookings_pct:.1f}%)")


def generate_comparison_report(
    original_hex_df: pd.DataFrame,
    consolidated_hex_df: pd.DataFrame,
    output_path: Path
) -> dict:
    """
    Generate statistics comparing original vs consolidated with seasonal insights.
    """
    print("\n   Generating comparison report...")
    
    # Overall statistics
    total_bookings_orig = original_hex_df['booking_count'].sum()
    total_bookings_cons = consolidated_hex_df['booking_count'].sum()
    
    # Week-by-week statistics
    week_stats_orig = original_hex_df.groupby('arrival_week')['booking_count'].sum()
    peak_week = week_stats_orig.idxmax()
    low_week = week_stats_orig.idxmin()
    
    # Seasonal aggregation
    summer_weeks = list(range(27, 40))  # Weeks 27-39 (Jul-Sep)
    winter_weeks = list(range(1, 13)) + list(range(40, 53))  # Weeks 1-12, 40-52
    
    summer_bookings = original_hex_df[original_hex_df['arrival_week'].isin(summer_weeks)]['booking_count'].sum()
    winter_bookings = original_hex_df[original_hex_df['arrival_week'].isin(winter_weeks)]['booking_count'].sum()
    
    summer_pct = (summer_bookings / total_bookings_orig) * 100
    winter_pct = (winter_bookings / total_bookings_orig) * 100
    
    # Top cities by season
    summer_cities = original_hex_df[original_hex_df['arrival_week'].isin(summer_weeks)].groupby('dominant_city')['booking_count'].sum().sort_values(ascending=False).head(5)
    winter_cities = original_hex_df[original_hex_df['arrival_week'].isin(winter_weeks)].groupby('dominant_city')['booking_count'].sum().sort_values(ascending=False).head(5)
    
    report = {
        'total_bookings': int(total_bookings_orig),
        'peak_week': int(peak_week),
        'peak_week_bookings': int(week_stats_orig[peak_week]),
        'low_week': int(low_week),
        'low_week_bookings': int(week_stats_orig[low_week]),
        'summer_bookings': int(summer_bookings),
        'summer_pct': float(summer_pct),
        'winter_bookings': int(winter_bookings),
        'winter_pct': float(winter_pct),
        'top_summer_cities': summer_cities.to_dict(),
        'top_winter_cities': winter_cities.to_dict()
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✓ Saved report: {output_path.name}")
    
    # Print summary
    print(f"\n   Seasonal Statistics:")
    print(f"   - Peak week: Week {peak_week} ({week_stats_orig[peak_week]:,} bookings)")
    print(f"   - Low week: Week {low_week} ({week_stats_orig[low_week]:,} bookings)")
    print(f"   - Summer (weeks 27-39): {summer_bookings:,} bookings ({summer_pct:.1f}%)")
    print(f"   - Winter (weeks 1-12, 40-52): {winter_bookings:,} bookings ({winter_pct:.1f}%)")
    
    return report


def main():
    """Main execution function."""
    print("=" * 80)
    print("CITY CONSOLIDATION VERIFICATION - HEXAGON HEATMAP WITH TEMPORAL SLIDER")
    print("=" * 80)
    
    # 1. Load data (without cleaning to preserve all location data)
    print("\n1. Loading data...")
    con = init_db()
    
    # 2. Load booking locations with weeks
    print("\n2. Loading booking locations with arrival weeks...")
    df = load_booking_locations_with_cities(con)
    
    # 3. Apply city consolidation
    print("\n3. Applying city consolidation...")
    city_mapping, city_stats = consolidate_cities(con, verbose=False)
    df['city_consolidated'] = df['city_original'].map(city_mapping)
    
    # Fill any unmapped cities with original
    df['city_consolidated'] = df['city_consolidated'].fillna(df['city_original'])
    
    print(f"   Mapped {len(city_mapping):,} cities")
    print(f"   Unique original cities: {df['city_original'].nunique():,}")
    print(f"   Unique consolidated cities: {df['city_consolidated'].nunique():,}")
    
    # 4. Assign H3 hexagons
    print("\n4. Assigning H3 hexagons...")
    df = assign_h3_hexagons(df, resolution=8)
    
    # 5. Aggregate by hexagon and week
    print("\n5. Aggregating by hexagon and week...")
    original_hex_df = aggregate_by_hexagon_and_week(df, 'city_original')
    consolidated_hex_df = aggregate_by_hexagon_and_week(df, 'city_consolidated')
    
    # 6. Generate consistent city colors
    print("\n6. Generating city color palette...")
    city_colors = get_city_colors(consolidated_hex_df, top_n=50)
    print(f"   Generated colors for top {len(city_colors)} cities")
    
    # 7. Create output directory
    output_dir = PROJECT_ROOT / "outputs" / "city_consolidation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 8. Create maps
    print("\n7. Creating interactive maps...")
    
    print("\n   a) Original city names (temporal)...")
    create_temporal_hexagon_map(
        original_hex_df,
        'Original City Names',
        output_dir / "heatmap_original_temporal.html",
        city_colors
    )
    
    print("\n   b) Consolidated city names (temporal)...")
    create_temporal_hexagon_map(
        consolidated_hex_df,
        'Consolidated City Names',
        output_dir / "heatmap_consolidated_temporal.html",
        city_colors
    )
    
    print("\n   c) Difference map (static)...")
    create_difference_map(
        original_hex_df,
        consolidated_hex_df,
        output_dir / "heatmap_difference.html"
    )
    
    # 9. Generate comparison report
    print("\n8. Generating comparison report...")
    report = generate_comparison_report(
        original_hex_df,
        consolidated_hex_df,
        output_dir / "hexagon_comparison_report.json"
    )
    
    # 10. Summary
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - heatmap_original_temporal.html")
    print("  - heatmap_consolidated_temporal.html")
    print("  - heatmap_difference.html")
    print("  - hexagon_comparison_report.json")
    print("\nOpen the HTML files in a browser to explore the interactive maps.")
    print("Use the slider to explore week-by-week patterns.")
    print("Click 'Play' to animate through the year.")


if __name__ == "__main__":
    main()


