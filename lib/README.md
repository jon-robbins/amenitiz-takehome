# Shared Library

This directory contains shared utilities used by both notebooks and the ML pipeline.

## Files

### `db.py`
Database connection utilities:
- `init_db()` - Initialize DuckDB connection and load CSV data
- `get_connection()` - Alias for `init_db()`
- Handles type casting for all tables (bookings, booked_rooms, rooms, hotel_location)

**Usage:**
```python
from lib.db import init_db

con = init_db()  # Returns DuckDB connection with all tables loaded
```

### `data_validator.py` ⭐ RECENTLY REFACTORED

Data validation and cleaning using a unified Rule-based architecture.

**Key Components:**
- `Rule` - Unified dataclass for all data quality operations
- `CleaningConfig` - Self-documenting configuration (no summary() needed)
- `DataCleaner` - Applies rules based on config
- `DuckDBConnectionWrapper` - Wraps connection to convert None/empty strings to NaN

**New API (Recommended):**
```python
from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner

# Define configuration
config = CleaningConfig(
    exclude_reception_halls=True,      # Section 2.2: Not accommodation
    exclude_missing_location=True,     # Section 3.1: Can't analyze location
    impute_children_allowed=True,      # Section 2.2: From booking behavior
    verbose=True
)

# Apply cleaning
cleaner = DataCleaner(config)
con = cleaner.clean(init_db())

# Check what was done
for rule_name, rows_affected in cleaner.stats.items():
    print(f"{rule_name}: {rows_affected:,} rows")
```

**Old API (Still Supported):**
```python
from lib.data_validator import validate_and_clean

con = validate_and_clean(
    init_db(),
    verbose=True,
    rooms_to_exclude=['reception_hall'],
    exclude_missing_location_bookings=True
)
```

**Built-in Rules:**
- **Basic Validation** (15 rules): Negative/zero/NULL prices, invalid dates, overcrowded rooms, orphan bookings, etc.
- **Exclusions** (found during EDA): Reception halls, missing location data
- **Fixes**: Convert empty strings to NULL
- **Imputations** (found during EDA): children_allowed, events_allowed

**See Also:**
- `DATA_VALIDATOR_REFACTOR.md` - Full refactoring documentation
- `notebooks/data_prep/data_checks_example.py` - Usage examples

## Import Path

All notebooks and ML pipeline scripts should import from `lib`:

```python
from lib.db import init_db
from lib.data_validator import validate_and_clean, check_data_quality
```

## Design Rationale

This `lib/` directory consolidates what were previously multiple "utils" folders:
- `notebooks/utils/` (moved here)
- `notebooks/eda/utils/` (split between `lib/` and topic-specific folders)

Benefits:
- Single source of truth for shared code
- No confusion about which "utils" to import from
- Accessible to both notebooks and ML pipeline
- Future-proof for production deployment

