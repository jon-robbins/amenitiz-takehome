# JavaScript Modules for Temporal Heatmap Visualization

This directory contains modular JavaScript code for the interactive temporal heatmap visualization.

## File Structure

### `heatmap_config.js`
Configuration constants for the visualization:
- `HEATMAP_CONFIG`: Heatmap layer settings (radius, blur, gradient colors)
- `HEXAGON_CONFIG`: Hexagon polygon styling (opacity, colors, borders)
- `ANIMATION_CONFIG`: Animation timing (transitions, play speed)

### `map_renderer.js`
Core rendering functions:
- `renderHeatmapView()`: Creates Leaflet heatmap layer for a given week
- `renderHexagonView()`: Creates GeoJSON hexagon layer with city colors and tooltips

### `ui_controls.js`
UI event handlers and controls:
- `updateDisplay()`: Updates week number, point count, and booking stats
- `initSlider()`: Sets up week slider with callback
- `initPlayButton()`: Initializes play/pause animation control
- `initViewToggle()`: Handles switching between heatmap and hexagon views

### `temporal_map.js`
Main controller that coordinates everything:
- `initTemporalMap()`: Entry point that initializes the entire visualization
- Manages state (current layer, view mode)
- Coordinates data, rendering, and UI controls

## Usage in Python

The JavaScript modules are loaded and injected into the HTML by the Python script:

```python
# Load JavaScript modules
js_config = load_js_file('heatmap_config.js')
js_renderer = load_js_file('map_renderer.js')
js_controls = load_js_file('ui_controls.js')
js_main = load_js_file('temporal_map.js')

# Inject into HTML with data
slider_html = f"""
    <script>{js_config}</script>
    <script>{js_renderer}</script>
    <script>{js_controls}</script>
    <script>{js_main}</script>
    <script>
        initTemporalMap(mapId, heatmapData, hexagonData, cityColors);
    </script>
"""
```

## Benefits of Modular Structure

1. **Maintainability**: Each file has a single responsibility
2. **Reusability**: Functions can be used in other visualizations
3. **Testability**: Individual modules can be tested separately
4. **Readability**: Clear separation of concerns (config, rendering, UI, control)
5. **Version Control**: Easier to track changes to specific functionality

## Modifying the Visualization

- **Change colors/sizes**: Edit `heatmap_config.js`
- **Adjust rendering logic**: Edit `map_renderer.js`
- **Add new controls**: Edit `ui_controls.js`
- **Change coordination logic**: Edit `temporal_map.js`

