# Interactive Heatmap Module

A simple, reusable function to create interactive temporal heatmaps from booking data.

## Quick Start

```python
from lib.db import init_db
from lib.interactive_heatmap import create_interactive_heatmap

# Load data
con = init_db()

# Create and display heatmap
m = create_interactive_heatmap(con, title="My Heatmap")
m  # Display in Jupyter
```

## Function Signature

```python
create_interactive_heatmap(
    con: duckdb.DuckDBPyConnection,
    output_path: str = None,
    title: str = "Booking Heatmap"
) -> folium.Map
```

### Parameters

- **con** (required): DuckDB connection with `bookings` and `hotel_location` tables
- **output_path** (optional): Path to save HTML file (e.g., `"outputs/map.html"`)
- **title** (optional): Title displayed on the map (default: `"Booking Heatmap"`)

### Returns

- **folium.Map**: Interactive map object that displays in Jupyter notebooks

## Features

### Interactive Controls

1. **Week Timeline Slider**
   - Explore bookings week-by-week (1-52)
   - Real-time map updates

2. **Play/Pause Button**
   - Animates through all 52 weeks automatically
   - Speed: 800ms per week

3. **Dynamic Radius Slider**
   - Adjust heatmap blob size: 5px to 50px
   - 5-10px: Precise locations
   - 15-25px: Balanced view
   - 30-50px: Regional trends

4. **Map Interactions**
   - Zoom, pan, scale bar
   - CartoDB Positron basemap (clean, minimal)

### Data Processing

- **Spatial Indexing**: H3 hexagons (resolution 8, ~1.2km)
- **Temporal Grouping**: Weekly aggregation (weeks 1-52)
- **Geographic Coverage**: Spain + Canary Islands
  - Latitude: 27°N to 44°N
  - Longitude: 18°W to 5°E

## Usage Examples

### Example 1: Display in Notebook

```python
from lib.db import init_db
from lib.interactive_heatmap import create_interactive_heatmap

con = init_db()
m = create_interactive_heatmap(con, title="Spain Bookings")
m  # Displays the map
```

### Example 2: Save to HTML

```python
m = create_interactive_heatmap(
    con,
    output_path="outputs/my_heatmap.html",
    title="Annual Booking Trends"
)
# Opens in browser: file:///path/to/outputs/my_heatmap.html
```

### Example 3: Multiple Maps

```python
# Summer bookings only
summer_con = # ... filter for summer data
m_summer = create_interactive_heatmap(
    summer_con,
    title="Summer Season (Jul-Sep)"
)

# Winter bookings only  
winter_con = # ... filter for winter data
m_winter = create_interactive_heatmap(
    winter_con,
    title="Winter Season (Dec-Feb)"
)
```

## Demo Notebook

See `notebooks/interactive_heatmap_demo.ipynb` for a complete walkthrough.

## Architecture

### Python Module
- **File**: `lib/interactive_heatmap.py`
- **Function**: `create_interactive_heatmap()`
- Handles: data loading, H3 aggregation, map creation

### JavaScript Modules
Located in `notebooks/data_prep/js/`:
- **heatmap_config.js**: Configuration constants
- **map_renderer.js**: Rendering functions
- **ui_controls.js**: Slider and button handlers
- **temporal_map.js**: Main controller

### Output
- Returns: Folium Map object
- Format: Interactive HTML with embedded JavaScript
- Size: ~500KB per map (includes all 52 weeks of data)

## Performance

- **Data Volume**: ~730K bookings, ~1,740 hexagons
- **Processing Time**: 5-10 seconds
- **Rendering**: Instant (client-side JavaScript)
- **Browser Compatibility**: Chrome, Firefox, Safari, Edge

## Troubleshooting

### Map doesn't display in Jupyter
- Ensure Jupyter has JavaScript enabled
- Try: `m.save("temp.html")` then open in browser

### "Module not found" error
- Check: `sys.path` includes project root
- Add: `sys.path.append('/path/to/amenitiz-takehome')`

### No data appears
- Verify database has `bookings` and `hotel_location` tables
- Check: geographic bounds include your data
- Confirm: `status IN ('confirmed', 'Booked')`

## Dependencies

- `folium` - Map visualization
- `h3` - Hexagonal spatial indexing
- `pandas` - Data manipulation
- `duckdb` - Database engine

All installed via Poetry: `poetry install`












