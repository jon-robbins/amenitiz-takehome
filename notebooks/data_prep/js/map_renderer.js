/**
 * Map rendering functions for heatmap and hexagon views
 */

/**
 * Render heatmap view for a given week
 */
function renderHeatmapView(theMap, weekData, config, customRadius) {
    if (!weekData || weekData.length === 0) {
        return { totalBookings: 0, pointCount: 0, layer: null };
    }
    
    // Calculate total bookings
    const totalBookings = weekData.reduce((sum, point) => sum + point[2], 0);
    
    // Use custom radius if provided, otherwise use config default
    const radius = customRadius !== undefined ? customRadius : config.radius;
    const blur = getBlurForRadius(radius);
    
    // Create heatmap layer with dynamic radius
    const heatmapConfig = {
        ...config,
        radius: radius,
        blur: blur
    };
    
    const layer = L.heatLayer(weekData, heatmapConfig).addTo(theMap);
    
    return {
        totalBookings: Math.round(totalBookings),
        pointCount: weekData.length,
        layer: layer
    };
}

/**
 * Render hexagon view for a given week
 */
function renderHexagonView(theMap, weekData, cityColors, config) {
    if (!weekData || weekData.length === 0) {
        return { totalBookings: 0, pointCount: 0, layer: null };
    }
    
    // Create GeoJSON features
    const features = weekData.map(item => ({
        type: 'Feature',
        geometry: item.geometry,
        properties: item.properties
    }));
    
    // Calculate totals
    const totalBookings = weekData.reduce((sum, item) => sum + item.properties.bookings, 0);
    
    // Find max bookings for this week to normalize opacity
    const maxBookings = Math.max(...weekData.map(item => item.properties.bookings));
    
    // Add new layer
    const layer = L.geoJSON({ type: 'FeatureCollection', features: features }, {
        style: feature => {
            const bookings = feature.properties.bookings;
            const city = feature.properties.city;
            
            // Get color for city (default to gray if not in top cities)
            const color = cityColors[city] || config.defaultColor;
            
            // Opacity based on booking count relative to max for this week
            const normalizedBookings = bookings / maxBookings;
            const opacity = config.fillOpacityMin + (normalizedBookings * (config.fillOpacityMax - config.fillOpacityMin));
            
            return {
                fillColor: color,
                fillOpacity: opacity,
                color: config.borderColor,
                weight: config.weight,
                opacity: config.borderOpacity
            };
        },
        onEachFeature: (feature, layer) => {
            const props = feature.properties;
            const tooltipContent = 
                '<div style="font-size: 12px;">' +
                '<strong style="font-size: 14px;">' + props.city + '</strong><br>' +
                '<strong>Bookings:</strong> ' + props.bookings + '<br>' +
                '<strong>Week:</strong> ' + props.week +
                '</div>';
            layer.bindTooltip(tooltipContent, { sticky: true });
        }
    }).addTo(theMap);
    
    return {
        totalBookings: totalBookings,
        pointCount: weekData.length,
        layer: layer
    };
}

