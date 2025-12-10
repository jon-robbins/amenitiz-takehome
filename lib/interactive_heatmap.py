"""
Interactive temporal heatmap visualization for booking data.

This module provides a single function to create an interactive Folium heatmap
with temporal slider and dynamic radius control.
"""

from pathlib import Path
import json
from collections import Counter

import pandas as pd
import folium
from folium.plugins import HeatMap
import h3
import duckdb


def create_interactive_heatmap(
    con: duckdb.DuckDBPyConnection,
    output_path: str = None,
    title: str = "Booking Heatmap"
) -> folium.Map:
    """
    Create an interactive temporal heatmap from booking data.
    
    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection with bookings and hotel_location tables
    output_path : str, optional
        If provided, saves the map to this HTML file path
    title : str, default "Booking Heatmap"
        Title displayed on the map
    
    Returns
    -------
    folium.Map
        Interactive Folium map object that can be displayed in Jupyter
    
    Example
    -------
    >>> from lib.db import init_db
    >>> from lib.interactive_heatmap import create_interactive_heatmap
    >>> con = init_db()
    >>> m = create_interactive_heatmap(con, title="Spain Bookings")
    >>> m  # Display in Jupyter
    """
    
    print("Loading booking data...")
    
    # Load booking locations with arrival weeks
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
    
    # Convert week 0 to 52
    df.loc[df['arrival_week'] == 0, 'arrival_week'] = 52
    
    print(f"Loaded {len(df):,} bookings from {df['hotel_id'].nunique():,} hotels")
    
    # Assign H3 hexagons
    print("Assigning H3 hexagons...")
    df['h3_hex'] = df.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 8),
        axis=1
    )
    
    # Aggregate by hexagon and week
    print("Aggregating by hexagon and week...")
    hex_week_df = df.groupby(['h3_hex', 'arrival_week']).agg({
        'booking_count': 'sum'
    }).reset_index()
    
    # Prepare temporal heatmap data: {week_num: [[lat, lon, intensity], ...]}
    print("Preparing temporal data...")
    temporal_heatmap_data = {}
    
    for week in range(1, 53):
        week_data = hex_week_df[hex_week_df['arrival_week'] == week]
        temporal_heatmap_data[week] = []
        
        for _, row in week_data.iterrows():
            h3_hex = row['h3_hex']
            center = h3.cell_to_latlng(h3_hex)
            intensity = float(row['booking_count'])
            temporal_heatmap_data[week].append([center[0], center[1], intensity])
    
    # Create Folium map
    print("Creating interactive map...")
    m = folium.Map(
        location=[37.5, -5.0],
        zoom_start=5,
        tiles='CartoDB positron',
        control_scale=True
    )
    
    # Load JavaScript modules
    js_dir = Path(__file__).parent.parent / 'notebooks' / 'data_prep' / 'js'
    
    def load_js_file(filename: str) -> str:
        with open(js_dir / filename, 'r') as f:
            return f.read()
    
    js_config = load_js_file('heatmap_config.js')
    js_renderer = load_js_file('map_renderer.js')
    js_controls = load_js_file('ui_controls.js')
    js_main = load_js_file('temporal_map.js')
    
    # Convert data to JSON
    temporal_heatmap_json = json.dumps(temporal_heatmap_data)
    map_id = m.get_name()
    
    # Create HTML with controls and JavaScript
    slider_html = f"""
    <!-- Vertical Radius Slider (Left Side) -->
    <div id="radius-slider-container" style="
        position: fixed;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        z-index: 1000;
        background: white;
        padding: 10px 8px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        width: 50px;
        height: 220px;
        display: flex;
        flex-direction: column;
        align-items: center;
    ">
        <div style="font-size: 10px; font-weight: bold; color: #444; margin-bottom: 15px; text-align: center;">
            Radius
        </div>
        <div style="height: 120px; display: flex; align-items: center; justify-content: center; margin: 10px 0;">
            <input type="range" id="radius-slider" min="5" max="50" value="15" 
                   orient="vertical" 
                   style="width: 120px; height: 20px; transform: rotate(-90deg); transform-origin: center; margin: 0;" 
                   step="1">
        </div>
        <div style="font-size: 12px; font-weight: bold; color: #333; margin-top: 15px; text-align: center;">
            <span id="radius-value">15</span>px
        </div>
        <div style="font-size: 8px; color: #999; text-align: center; margin-top: 8px; line-height: 1.3;">
            5=Precise<br>50=Diffuse
        </div>
    </div>
    
    <!-- Week Timeline Controls (Bottom) -->
    <div id="week-slider-container" style="
        position: fixed;
        bottom: 10px;
        left: 70px;
        right: 10px;
        z-index: 1000;
        background: white;
        padding: 10px 16px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <div style="flex: 1;">
                <strong style="font-size: 13px;">{title}</strong>
                <span style="font-size: 11px; color: #666; margin-left: 12px;">
                    Week: <span id="week-display">1</span> | 
                    Points: <span id="hex-count">0</span> | 
                    Bookings: <span id="booking-count">0</span>
                </span>
            </div>
            <button id="play-btn" style="padding: 5px 16px; cursor: pointer; font-size: 12px; border-radius: 4px; border: 1px solid #ccc; background: white;">
                ▶ Play
            </button>
        </div>
        <input type="range" id="week-slider" min="1" max="52" value="1" 
               style="width: 100%; height: 5px; margin: 2px 0;" step="1">
        <div style="display: flex; justify-content: space-between; font-size: 9px; color: #999; margin-top: 2px;">
            <span>Week 1 (Jan)</span>
            <span>Week 13 (Apr)</span>
            <span>Week 26 (Jul)</span>
            <span>Week 39 (Oct)</span>
            <span>Week 52 (Dec)</span>
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
        const mapId = '{map_id}';
        
        // Initialize the temporal map (heatmap only mode)
        initTemporalMap(mapId, heatmapData, null, null);
    }});
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(slider_html))
    
    # Add legend (top right)
    legend_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-size: 11px;
    ">
        <strong style="font-size: 12px;">Booking Intensity</strong>
        <div style="margin-top: 6px;">
            <div style="background: linear-gradient(to right, blue, cyan, lime, yellow, orange, red); 
                        width: 160px; height: 12px; border: 1px solid #333; border-radius: 2px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 9px; color: #666; margin-top: 2px;">
                <span>Low</span>
                <span>High</span>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save if output path provided
    if output_path:
        m.save(str(output_path))
        print(f"Saved to: {output_path}")
    
    print("Complete!")
    return m

