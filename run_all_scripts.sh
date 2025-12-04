#!/bin/bash
cd /Users/jon/GitHub/amenitiz-takehome

files=(
"notebooks/eda/00_schema_exploration/occupancy_calculation_debug.py"
"notebooks/eda/01_data_quality/analyze_distance_features.py"
"notebooks/eda/01_data_quality/calculate_distance_features.py"
"notebooks/eda/02_descriptive_analysis/section_1_2_hotel_supply.py"
"notebooks/eda/02_descriptive_analysis/section_1_3_daily_price.py"
"notebooks/eda/02_descriptive_analysis/section_2_1_room_features.py"
"notebooks/eda/02_descriptive_analysis/section_2_2_capacity_policies.py"
"notebooks/eda/02_descriptive_analysis/section_4_1_seasonality.py"
"notebooks/eda/02_descriptive_analysis/section_4_2_popular_expensive.py"
"notebooks/eda/02_descriptive_analysis/section_4_3_booking_counts.py"
"notebooks/eda/03_spatial/hotspots/hotspots_dbscan.py"
"notebooks/eda/03_spatial/hotspots/hotspots_grid.py"
"notebooks/eda/03_spatial/hotspots/hotspots_kde_admin.py"
"notebooks/eda/03_spatial/hotspots/spatial_utils.py"
"notebooks/eda/03_spatial/section_3_1_integrated.py"
"notebooks/eda/04_pricing/section_5_1_lead_time.py"
"notebooks/eda/04_pricing/section_5_2_occupancy_pricing.py"
"notebooks/eda/04_pricing/section_6_1_room_features.py"
"notebooks/eda/04_pricing/section_7_1_occupancy_capacity.py"
"notebooks/eda/04_pricing/section_7_2_revpar.py"
"notebooks/eda/05_elasticity/feature_importance_validation.py"
"notebooks/eda/05_elasticity/gps_continuous_treatment.py"
"notebooks/eda/05_elasticity/matched_pairs_geographic.py"
"notebooks/eda/05_elasticity/matched_pairs_longitudinal.py"
"notebooks/eda/05_elasticity/matched_pairs_validated_features.py"
"notebooks/eda/05_elasticity/matched_pairs_validated.py"
"notebooks/eda/05_elasticity/matched_pairs_with_replacement.py"
"notebooks/eda/05_elasticity/over_under_pricing.py"
"notebooks/eda/eda_updated.py"
"notebooks/eda/geo_data/GeoSpain/geospain/geo.py"
)

for file in "${files[@]}"; do
    echo "=========================================="
    echo "Running: $file"
    echo "=========================================="
    poetry run python "$file" 2>&1 | head -30
    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "✓ SUCCESS"
    else
        echo "✗ FAILED (exit code: $exit_code)"
    fi
    echo ""
done
