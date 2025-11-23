# Data Preparation

This directory contains scripts for data quality assessment and preparation before EDA.

## Files

### `data_checks.py`
Data quality assessment script that:
- Checks for NULL values, negative prices, date inconsistencies
- Validates referential integrity between tables
- Reports percentage of problematic rows
- Can be run standalone or imported

**Usage:**
```bash
cd /path/to/amenitiz-takehome
poetry run python notebooks/data_prep/data_checks.py
```

### `city_consolidation.py`
TF-IDF-based city name matching to consolidate variations:
- Handles compound names ("Suburb, City" → "City")
- Normalizes case ("MADRID" → "Madrid")
- Fuzzy matches similar names ("Leon" → "León")
- Reduces 1,480 cities to 1,132 (23.5% reduction)

**Usage:**
```bash
poetry run python notebooks/data_prep/city_consolidation.py
```

**Outputs:**
- `outputs/hotspots/city_name_mapping.csv` - Full mapping dictionary
- `outputs/hotspots/city_consolidation_comparison.csv` - Before/after comparison

See `CITY_MATCHING_README.md` for detailed documentation.

## Workflow

1. **Data Quality Check**: Run `data_checks.py` to assess raw data quality
2. **Data Cleaning**: Use `lib/data_validator.py` functions to clean data
3. **City Consolidation**: Run `city_consolidation.py` to standardize city names
4. **EDA**: Proceed to `notebooks/eda/` for exploratory analysis

## Dependencies

All data prep scripts import from:
- `lib/db.py` - Database connection utilities
- `lib/data_validator.py` - Validation and cleaning functions

## Notes

- Data prep scripts modify in-memory DuckDB connections, not raw CSV files
- The `lib/data_validator.py` module will eventually be used by the ML pipeline
- City consolidation is optional but recommended for location-based analysis

