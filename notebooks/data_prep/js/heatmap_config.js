/**
 * Configuration for heatmap visualization
 */
const HEATMAP_CONFIG = {
    radius: 15,  // Default radius, can be changed dynamically
    blur: 20,    // Blur will be proportional to radius (radius * 1.33)
    maxZoom: 10,
    max: 1.0,
    gradient: {
        0.0: 'blue',
        0.2: 'cyan',
        0.4: 'lime',
        0.6: 'yellow',
        0.8: 'orange',
        1.0: 'red'
    }
};

/**
 * Calculate blur based on radius (maintains 1.33x ratio)
 */
function getBlurForRadius(radius) {
    return Math.round(radius * 1.33);
}

const HEXAGON_CONFIG = {
    fillOpacityMin: 0.2,
    fillOpacityMax: 0.9,
    weight: 0.5,
    borderOpacity: 0.8,
    borderColor: '#333',
    defaultColor: '#999999'
};

const ANIMATION_CONFIG = {
    fadeTransitionMs: 300,
    playIntervalMs: 800
};

