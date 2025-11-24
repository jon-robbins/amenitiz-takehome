/**
 * Main temporal map controller
 * Coordinates data, rendering, and UI controls
 */

/**
 * Initialize the temporal map with all functionality
 * 
 * @param {string} mapId - Leaflet map variable name
 * @param {Object} heatmapData - Temporal heatmap data {week: [[lat, lon, intensity], ...]}
 * @param {Object|null} hexagonData - Optional hexagon data (null for heatmap-only mode)
 * @param {Object|null} cityColors - Optional city colors (null for heatmap-only mode)
 */
function initTemporalMap(mapId, heatmapData, hexagonData, cityColors) {
    // State
    let currentLayer = null;
    let viewMode = 'heatmap';
    let currentRadius = 15;
    const theMap = window[mapId];
    const heatmapOnly = (hexagonData === null || hexagonData === undefined);
    
    /**
     * Update map for a given week, view mode, and radius
     */
    function updateMap(week, newViewMode, newRadius) {
        // Update view mode if provided (only if not heatmap-only mode)
        if (!heatmapOnly && newViewMode !== undefined) {
            viewMode = newViewMode;
        }
        
        // Update radius if provided
        if (newRadius !== undefined) {
            currentRadius = newRadius;
        }
        
        // Remove previous layer
        if (currentLayer) {
            theMap.removeLayer(currentLayer);
        }
        
        let result;
        
        if (viewMode === 'heatmap' || heatmapOnly) {
            // Render heatmap view with current radius
            const weekData = heatmapData[week];
            result = renderHeatmapView(theMap, weekData, HEATMAP_CONFIG, currentRadius);
        } else {
            // Render hexagon view (only if hexagon data is provided)
            const weekData = hexagonData[week];
            result = renderHexagonView(theMap, weekData, cityColors, HEXAGON_CONFIG);
        }
        
        // Update current layer
        currentLayer = result.layer;
        
        // Update display
        updateDisplay(week, result.pointCount, result.totalBookings);
    }
    
    // Initialize UI controls
    initSlider(updateMap);
    initPlayButton(updateMap, ANIMATION_CONFIG.playIntervalMs);
    initRadiusSlider(updateMap);
    
    // Only initialize view toggle if hexagon data is provided
    if (!heatmapOnly) {
        initViewToggle(updateMap);
    }
    
    // Initialize with week 1
    updateMap(1);
}

