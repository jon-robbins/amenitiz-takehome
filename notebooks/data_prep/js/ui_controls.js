/**
 * UI control handlers for slider, play button, and view toggle
 */

/**
 * Update display elements with current week stats
 */
function updateDisplay(week, pointCount, totalBookings) {
    document.getElementById('week-display').textContent = week;
    document.getElementById('hex-count').textContent = pointCount.toLocaleString();
    document.getElementById('booking-count').textContent = totalBookings.toLocaleString();
}

/**
 * Initialize slider event listener
 */
function initSlider(updateMapCallback) {
    document.getElementById('week-slider').addEventListener('input', e => {
        updateMapCallback(parseInt(e.target.value));
    });
}

/**
 * Initialize play button with animation control
 */
function initPlayButton(updateMapCallback, intervalMs) {
    let isPlaying = false;
    let playInterval = null;
    
    document.getElementById('play-btn').addEventListener('click', function() {
        if (isPlaying) {
            clearInterval(playInterval);
            this.textContent = '▶ Play';
            isPlaying = false;
        } else {
            this.textContent = '⏸ Pause';
            isPlaying = true;
            playInterval = setInterval(() => {
                const slider = document.getElementById('week-slider');
                let week = parseInt(slider.value);
                if (week >= 52) {
                    week = 1;
                } else {
                    week++;
                }
                slider.value = week;
                updateMapCallback(week);
            }, intervalMs);
        }
    });
}

/**
 * Initialize view toggle button (heatmap <-> hexagon)
 */
function initViewToggle(updateMapCallback) {
    let viewMode = 'heatmap';
    
    document.getElementById('view-toggle').addEventListener('click', function() {
        if (viewMode === 'heatmap') {
            viewMode = 'hexagon';
            this.textContent = 'Switch to Heatmap';
        } else {
            viewMode = 'heatmap';
            this.textContent = 'Switch to Hexagons';
        }
        
        // Re-render current week with new view mode
        const currentWeek = parseInt(document.getElementById('week-slider').value);
        updateMapCallback(currentWeek, viewMode);
    });
    
    return viewMode;
}

/**
 * Initialize radius slider for dynamic heatmap size adjustment
 */
function initRadiusSlider(updateMapCallback) {
    let currentRadius = 15;
    
    document.getElementById('radius-slider').addEventListener('input', function(e) {
        currentRadius = parseInt(e.target.value);
        document.getElementById('radius-value').textContent = currentRadius;
        
        // Re-render current week with new radius
        const currentWeek = parseInt(document.getElementById('week-slider').value);
        updateMapCallback(currentWeek, undefined, currentRadius);
    });
    
    return currentRadius;
}

