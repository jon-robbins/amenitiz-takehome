# EDA Pipeline Structure

## Central Notebook
- **eda_updated.py** - Central notebook consolidating all EDA analyses

## Analysis Pipeline

The EDA is organized into a clear pipeline structure:

### 1. **01_data_quality/** - Data Validation and Feature Engineering
- `calculate_distance_features.py` - Calculates distance from coast and Madrid
- `analyze_distance_features.py` - Analyzes distance feature correlations with price

**Outputs:** `outputs/eda/spatial/data/`

### 2. **02_descriptive_analysis/** - Basic Statistics and Distributions
- `section_1_2_hotel_supply.py` - Hotel inventory and supply structure
- `section_1_3_daily_price.py` - Daily price distributions
- `section_2_1_room_features.py` - Room size and view analysis
- `section_2_2_capacity_policies.py` - Capacity and policy flags
- `section_4_1_seasonality.py` - Seasonal price patterns
- `section_4_2_popular_expensive.py` - Popular and expensive dates
- `section_4_3_booking_counts.py` - Booking volume trends

**Outputs:** `outputs/eda/descriptive_analysis/figures/`

### 3. **03_spatial/** - Geographic and Location Analysis
- `section_3_1_integrated.py` - Integrated spatial analysis
- `hotspots/` - Hotspot detection algorithms (DBSCAN, Grid, KDE)

**Outputs:** `outputs/eda/spatial/`

### 4. **04_pricing/** - Pricing Patterns and Opportunities
- `section_5_1_lead_time.py` - Lead time and booking window analysis
- `section_5_2_occupancy_pricing.py` - Occupancy-based pricing opportunities
- `section_6_1_room_features.py` - Room feature pricing premiums
- `section_7_1_occupancy_capacity.py` - Occupancy vs capacity analysis (Simpson's Paradox)
- `section_7_2_revpar.py` - RevPAR analysis

**Outputs:** `outputs/eda/pricing/figures/`

### 5. **05_elasticity/** - Price Elasticity Estimation
- `matched_pairs_geographic.py` - **FINAL METHOD** - Geographic matched pairs analysis

**Method:** Matches hotels within same geographic market segment (coastal/inland × Madrid/provincial) plus revenue quartile, using KNN on normalized features.

**Outputs:** 
- `outputs/eda/elasticity/figures/` - Visualizations
- `outputs/eda/elasticity/data/` - Matched pairs data

### 6. **summaries/** - Analysis Summaries and Documentation
- Comprehensive analysis summaries
- Executive summaries
- Methodology documentation
- Strategic frameworks

## Outputs Directory Structure

```
outputs/eda/
├── descriptive_analysis/figures/  # Section 1-4 visualizations
├── spatial/
│   ├── figures/                   # Spatial visualizations
│   └── data/                      # Distance features, hotspots
├── pricing/figures/               # Section 5-7 visualizations
└── elasticity/
    ├── figures/                   # Elasticity visualizations
    └── data/                      # Matched pairs results
```

## Dependencies

Scripts should be run in order:
1. Data quality (generates distance features)
2. Descriptive analysis (uses cleaned data)
3. Spatial analysis (uses distance features)
4. Pricing analysis (uses all previous analyses)
5. Elasticity (uses spatial data and pricing insights)

## Key Features

- **Single elasticity method**: Geographic matched pairs (most robust)
- **Organized outputs**: All outputs categorized by analysis type
- **Clear progression**: Pipeline flows from data quality → elasticity
- **Central notebook**: eda_updated.py provides consolidated view

