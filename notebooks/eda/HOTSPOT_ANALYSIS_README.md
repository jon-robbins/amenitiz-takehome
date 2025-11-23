# Geographic Hotspot Analysis - Section 3.1

## Overview

This analysis implements three complementary methodologies for detecting booking hotspots in Spain, addressing the issue that city names in the raw data have inconsistent formatting and don't capture metro-area effects (e.g., Badalona vs Barcelona).

## Files

### Core Analysis Scripts

1. **`hotspots_grid.py`** - Grid-based aggregation
   - Bins bookings into 0.1° lat/lon grid cells (~11km resolution)
   - Outputs: `grid_hotspots.csv`, `grid_hotspots_top50.csv`, `grid_hotspots.png`

2. **`hotspots_dbscan.py`** - Density-based clustering
   - Uses DBSCAN with haversine distance (eps=5km, min_samples=25)
   - Outputs: `dbscan_labeled_bookings.csv`, `dbscan_cluster_summary.csv`, `dbscan_hotspots.png`

3. **`hotspots_kde_admin.py`** - KDE with administrative overlay
   - Generates kernel density estimates over Spanish municipalities
   - Outputs: `kde_hotspot_scores.csv`, `kde_hotspots.geojson`, `kde_hotspots.png`

4. **`section_3_1_integrated.py`** - Integrated analysis (RECOMMENDED)
   - Combines all three methods with city-level analysis
   - Provides comprehensive summary statistics and recommendations
   - Outputs: `section_3_1_integrated.png` (9-panel visualization)

### Utilities

- **`utils/spatial.py`** - Shared helper functions for loading data and creating output directories

## Key Improvements Over Previous Version

### 1. Fixed Visualization Issues

**Problem**: DBSCAN plot showed Spain as tiny because outlier coordinates (North America, Northern Europe) were included.

**Solution**: 
- Added Spain geographic bounds filtering (35-44°N, 10°W-5°E)
- Filtered 38,382 outlier coordinates (5.4% of data)
- Set explicit axis limits for all plots

**Problem**: Grid plot had no map context, just scatter points.

**Solution**:
- Added sized scatter points (size = booking count)
- Added color mapping (color = avg revenue per booking)
- Added legend showing what point sizes represent
- Set consistent Spain bounds across all visualizations

### 2. Integrated Analysis

Created `section_3_1_integrated.py` that:
- Runs all three hotspot methods on the same cleaned dataset
- Generates a 9-panel comparative visualization
- Provides detailed summary statistics for each method
- Calculates market concentration (Herfindahl Index) for each approach
- Offers concrete recommendations for pricing model

### 3. Data Quality

All scripts now use:
```python
con = validate_and_clean(
    init_db(),
    rooms_to_exclude=['reception_hall'],
    exclude_missing_location_bookings=True
)
```

This ensures:
- Reception hall bookings excluded (not accommodation)
- Hotels with missing location data excluded
- Consistent cleaned dataset across all analyses

## Results Summary

### Dataset
- **714,729** total bookings with location data
- **676,347** bookings within Spain bounds (94.6%)
- **€193.2M** total revenue
- **1,480** named cities

### Method Comparison

| Method | Units | Median Size | HHI | Top 10 Capture |
|--------|-------|-------------|-----|----------------|
| Grid | 947 cells | 195 bookings | 0.0078 | 21.2% |
| DBSCAN | 752 clusters | 315 bookings | 0.0102 | 24.8% |
| City | 1,480 cities | 116 bookings | 0.0081 | 19.7% |

**Interpretation**: All methods show highly distributed demand (low HHI < 0.015). DBSCAN creates the most concentrated groupings (highest HHI), while city names are most granular.

### Top 5 Hotspots

**By Method:**
1. **Madrid** - 42,870 bookings (Grid: 40.45,-3.75 | DBSCAN: Cluster 9 | City: Madrid)
2. **Barcelona** - 19,494 bookings (Grid: 41.35,2.15 | DBSCAN: Cluster 53 | City: Barcelona)
3. **Sevilla** - 18,715 bookings (Grid: 37.35,-5.95 | DBSCAN: Cluster 0 | City: Sevilla)
4. **Valencia** - 16,893 bookings (Grid: 39.85,-4.05 | DBSCAN: Cluster 57 | City: Toledo)
5. **Galicia** - 16,434 bookings (Grid: 42.25,-8.65 | DBSCAN: Cluster 39 | City: Málaga)

## Recommendations for Pricing Model

### Hybrid Geographic Feature Strategy

1. **Primary**: Use `city` name as categorical feature
   - ✓ Interpretable for business users
   - ✓ Aligns with how customers search
   - ✗ 1,480 categories (high cardinality)

2. **Fallback**: Use DBSCAN `cluster_id` for missing/poor city data
   - ✓ Only 752 clusters (more manageable)
   - ✓ Captures metro-area effects
   - ✗ Requires recomputing if new hotels added

3. **Continuous**: Include raw `latitude` and `longitude`
   - ✓ Captures fine-grained location effects
   - ✓ Generalizes to new locations
   - ✗ May overfit to specific coordinates

4. **Engineered**: Create "metro tier" feature
   - Tier 1: Top 10 cities (19.7% of bookings)
   - Tier 2: Cities with >1,000 bookings
   - Tier 3: Cities with 100-1,000 bookings
   - Tier 4: Small markets (<100 bookings)

### Feature Engineering Examples

```python
# Option 1: One-hot encode top N cities, group rest as "Other"
top_20_cities = city_df.head(20)['city'].tolist()
df['city_grouped'] = df['city'].apply(lambda x: x if x in top_20_cities else 'Other')

# Option 2: Use DBSCAN cluster as proxy for metro area
df['metro_cluster'] = df['cluster_id'].apply(lambda x: x if x >= 0 else -1)

# Option 3: Create distance to nearest major city
major_cities = [(40.42, -3.70), (41.39, 2.17), (37.39, -5.99)]  # Madrid, Barcelona, Sevilla
df['dist_to_major_city'] = df.apply(lambda row: min([
    haversine(row['latitude'], row['longitude'], lat, lon) 
    for lat, lon in major_cities
]), axis=1)
```

## Visualization Outputs

### Individual Method Plots
- `grid_hotspots.png` - Scatter plot with sized/colored points showing grid cells
- `dbscan_hotspots.png` - Cluster visualization with noise points
- `kde_hotspots.png` - Choropleth of municipal density scores

### Integrated Plot (9 panels)
`section_3_1_integrated.png` contains:
- **Row 1**: Spatial visualizations (Grid, DBSCAN, Top Cities)
- **Row 2**: Comparative rankings (Top cells, clusters, cities)
- **Row 3**: Size distributions (histograms for each method)

## Running the Analysis

```bash
# Run individual methods
poetry run python notebooks/eda/hotspots_grid.py
poetry run python notebooks/eda/hotspots_dbscan.py
poetry run python notebooks/eda/hotspots_kde_admin.py

# Run integrated analysis (RECOMMENDED)
poetry run python notebooks/eda/section_3_1_integrated.py
```

## Next Steps

1. **Validate hotspot-city alignment**: Check if DBSCAN clusters align with named cities
2. **Temporal analysis**: Do hotspots shift seasonally?
3. **Price modeling**: Test which geographic feature(s) have strongest predictive power
4. **Choropleth visualization**: Use `kde_hotspots.geojson` in Folium/Kepler.gl for interactive maps

