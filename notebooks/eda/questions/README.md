# EDA Questions

This directory contains individual Python scripts for each EDA question from the business case.

## Section 1: Supply Structure

### `section_1_2_hotel_supply.py`
Analyzes hotel-level supply structure:
- Distribution of room configurations per hotel
- Distribution of individual units per hotel
- Room type category mix by hotel
- Visualizations of supply patterns

### `section_1_3_daily_price.py`
Analyzes daily price per room-night:
- Overall price distribution and statistics
- Price by room type category
- Price vs stay length (length-of-stay discounts)
- Price vs room size
- Price vs number of guests

## Section 2: Room Features

### `section_2_1_room_features.py`
Analyzes room features (size, view):
- Room size distribution by room type
- View type distribution and pricing
- Feature combinations and their impact on price

### `section_2_2_capacity_policies.py`
Analyzes capacity and policy flags:
- Max occupancy distribution
- Policy flag analysis (children_allowed, events_allowed)
- Occupancy rate patterns
- Revenue implications of different policies

## Section 3: Location Analysis

### `section_3_1_location_analysis.py`
City-level supply and demand analysis:
- Top cities by bookings, revenue, price
- Hotel and room distribution by city
- Market concentration metrics
- Price variation across cities

### `section_3_1_integrated.py`
Integrated geographic hotspot analysis combining three methods:
- Grid-based aggregation (0.1° cells)
- DBSCAN clustering (5km radius)
- City-level analysis with TF-IDF name matching
- Comparative visualizations and recommendations

## Running Scripts

All scripts can be run from the project root:

```bash
cd /path/to/amenitiz-takehome

# Section 1
poetry run python notebooks/eda/questions/section_1_2_hotel_supply.py
poetry run python notebooks/eda/questions/section_1_3_daily_price.py

# Section 2
poetry run python notebooks/eda/questions/section_2_1_room_features.py
poetry run python notebooks/eda/questions/section_2_2_capacity_policies.py

# Section 3
poetry run python notebooks/eda/questions/section_3_1_location_analysis.py
poetry run python notebooks/eda/questions/section_3_1_integrated.py
```

## Dependencies

All scripts import from:
- `lib/db.py` - Database connection
- `lib/data_validator.py` - Data cleaning
- `notebooks/eda/hotspots/` - Spatial analysis utilities (Section 3 only)
- `notebooks/data_prep/city_consolidation.py` - City name matching (Section 3 only)

## Output

Scripts generate:
- Console output with statistics and insights
- Matplotlib visualizations (displayed interactively)
- Some scripts save outputs to `outputs/` directory

