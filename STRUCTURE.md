# Repository Structure

This document explains the organization of the amenitiz-takehome repository.

## Directory Layout

```
amenitiz-takehome/
├── data/                          # Raw CSV data files
│   ├── ds_booked_rooms.csv
│   ├── ds_bookings.csv
│   ├── ds_hotel_location.csv
│   └── ds_rooms.csv
│
├── docs/                          # Documentation
│   ├── background.md              # Business case background
│   └── Senior Data Scientist - Business Case (3).pdf
│
├── lib/                           # Shared utilities (notebooks + ML pipeline)
│   ├── __init__.py
│   ├── db.py                      # Database connection
│   ├── data_validator.py          # Data validation & cleaning (REFACTORED)
│   ├── DATA_VALIDATOR_REFACTOR.md # Refactoring documentation
│   └── README.md
│
├── notebooks/
│   ├── data_prep/                 # Data quality & preparation
│   │   ├── data_checks.py         # Quality assessment
│   │   ├── data_checks_example.py # Example using new API
│   │   ├── city_consolidation.py  # TF-IDF city name matching
│   │   ├── CITY_MATCHING_README.md
│   │   └── README.md
│   │
│   └── eda/                       # Exploratory Data Analysis
│       ├── eda_clustering.ipynb   # Main EDA notebook
│       │
│       ├── questions/             # One .py file per EDA question
│       │   ├── section_1_2_hotel_supply.py
│       │   ├── section_1_3_daily_price.py
│       │   ├── section_2_1_room_features.py
│       │   ├── section_2_2_capacity_policies.py
│       │   ├── section_3_1_location_analysis.py
│       │   ├── section_3_1_integrated.py
│       │   └── README.md
│       │
│       ├── hotspots/              # Geographic hotspot analysis
│       │   ├── hotspots_grid.py
│       │   ├── hotspots_dbscan.py
│       │   ├── hotspots_kde_admin.py
│       │   ├── spatial_utils.py
│       │   └── HOTSPOT_ANALYSIS_README.md
│       │
│       └── geo_data/              # Geographic resources
│           ├── GeoSpain/          # Spanish administrative boundaries
│           └── shapefiles/        # Municipal shapefiles
│
├── ml_pipeline/                   # ML pipeline (future)
│   ├── __init__.py
│   ├── main.py
│   └── preprocessing.py
│
├── outputs/                       # Analysis outputs
│   ├── figures/                   # Visualizations
│   └── hotspots/                  # Hotspot analysis results
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_pipeline.py
│
├── visualization/                 # Visualization utilities
│   ├── __init__.py
│   └── plots.py
│
├── pyproject.toml                 # Poetry dependencies
├── poetry.lock
├── README.md                      # Main project README
└── STRUCTURE.md                   # This file
```

## Key Principles

### 1. Clear Separation of Concerns

- **`lib/`**: Shared code used by both notebooks and ML pipeline
- **`notebooks/data_prep/`**: Data quality checks and preparation
- **`notebooks/eda/`**: Exploratory data analysis
- **`ml_pipeline/`**: Production ML code (future)

### 2. No "Utils" Confusion

Previously, there were multiple "utils" directories:
- `notebooks/utils/`
- `notebooks/eda/utils/`

Now, there's only **`lib/`** for shared utilities, with topic-specific helpers in their respective directories (e.g., `notebooks/eda/hotspots/spatial_utils.py`).

### 3. One File Per Question

EDA questions are in `notebooks/eda/questions/`, with one `.py` file per question. This makes it easy to:
- Run individual analyses
- Review code for specific questions
- Maintain and update analyses independently

### 4. Topic-Based Grouping

Related files are grouped together:
- Hotspot analysis methods in `notebooks/eda/hotspots/`
- Geographic data in `notebooks/eda/geo_data/`
- Data prep in `notebooks/data_prep/`

## Import Paths

### From Notebooks

```python
# Shared utilities
from lib.db import init_db
from lib.data_validator import validate_and_clean

# Data prep
from notebooks.data_prep.city_consolidation import create_city_mapping

# EDA utilities
from notebooks.eda.hotspots.spatial_utils import load_clean_booking_locations
```

### From ML Pipeline

```python
# ML pipeline can import from lib/
from lib.db import init_db
from lib.data_validator import validate_and_clean
```

## Running Scripts

### Data Preparation

```bash
# Check data quality
poetry run python notebooks/data_prep/data_checks.py

# Consolidate city names
poetry run python notebooks/data_prep/city_consolidation.py
```

### EDA Questions

```bash
# Section 1: Supply Structure
poetry run python notebooks/eda/questions/section_1_2_hotel_supply.py
poetry run python notebooks/eda/questions/section_1_3_daily_price.py

# Section 2: Room Features
poetry run python notebooks/eda/questions/section_2_1_room_features.py
poetry run python notebooks/eda/questions/section_2_2_capacity_policies.py

# Section 3: Location Analysis
poetry run python notebooks/eda/questions/section_3_1_location_analysis.py
poetry run python notebooks/eda/questions/section_3_1_integrated.py
```

### Hotspot Analysis

```bash
poetry run python notebooks/eda/hotspots/hotspots_grid.py
poetry run python notebooks/eda/hotspots/hotspots_dbscan.py
poetry run python notebooks/eda/hotspots/hotspots_kde_admin.py
```

## Workflow

1. **Data Preparation** (`notebooks/data_prep/`)
   - Run `data_checks.py` to assess quality
   - Use `validate_and_clean()` to clean data
   - Optionally run `city_consolidation.py` to standardize city names

2. **Exploratory Data Analysis** (`notebooks/eda/`)
   - Run individual question scripts in `questions/`
   - Or work interactively in `eda_clustering.ipynb`
   - Use hotspot analysis for geographic insights

3. **ML Pipeline** (`ml_pipeline/`)
   - Import cleaned data using `lib.db` and `lib.data_validator`
   - Build features and train models
   - Deploy to production

## Benefits of This Structure

- ✅ **Clear separation**: Data prep vs EDA vs ML pipeline
- ✅ **No confusion**: Single `lib/` for shared code
- ✅ **Easy navigation**: Logical grouping of related files
- ✅ **Maintainable**: One file per question, easy to update
- ✅ **Future-proof**: `lib/` accessible to both notebooks and production code
- ✅ **Clean root**: No scattered scripts
- ✅ **Documented**: README in each major directory

