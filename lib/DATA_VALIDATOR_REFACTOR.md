# Data Validator Refactoring

## Summary

The `data_validator.py` module has been refactored to use a unified, configuration-driven architecture where ALL data quality operations (deletions, exclusions, imputations, fixes) use the same `Rule` format.

## Key Changes

### 1. Unified Rule Format

**Before**: Different formats for validation rules, exclusions, and imputations

**After**: Everything is a `Rule` with `check_query` and `action_query`

```python
@dataclass
class Rule:
    name: str
    check_query: str      # How many rows affected?
    action_query: str     # Fix the issue
    enabled: bool = True
```

**Examples:**
- Deletion: `Rule("Negative Price", "SELECT COUNT(*) ...", "DELETE FROM ...")`
- Exclusion: `Rule("Exclude Reception Halls", "SELECT COUNT(*) ...", "DELETE FROM ...")`
- Imputation: `Rule("Impute children_allowed", "SELECT COUNT(*) ...", "UPDATE ...")`
- Fix: `Rule("Fix Empty city", "SELECT COUNT(*) ...", "UPDATE ...")`

### 2. Self-Documenting Configuration

**Before**: Function parameters (`rooms_to_exclude`, `exclude_missing_location_bookings`)

**After**: `CleaningConfig` dataclass with descriptive field names

```python
@dataclass
class CleaningConfig:
    # Basic validation
    remove_negative_prices: bool = True
    remove_zero_prices: bool = True
    # ... 13 more basic rules
    
    # Exclusions (found during EDA)
    exclude_reception_halls: bool = False      # Section 2.2
    exclude_missing_location: bool = False     # Section 3.1
    
    # Fixes
    fix_empty_strings: bool = True
    
    # Imputations (found during EDA)
    impute_children_allowed: bool = True       # Section 2.2
    impute_events_allowed: bool = True         # Section 2.2
    
    verbose: bool = False
```

**Key insight**: The config fields ARE the documentation. No `summary()` method needed.

### 3. DataCleaner Class

**Before**: Single `validate_and_clean()` function with growing parameter list

**After**: `DataCleaner` class that builds rules from config

```python
class DataCleaner:
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.rules = self._build_rules()  # Build rules from config
        self.stats = {}
    
    def clean(self, con) -> DuckDBConnectionWrapper:
        # Apply all enabled rules
        # Track statistics
        # Return wrapped connection
```

### 4. Connection Wrapper (Unchanged)

`DuckDBConnectionWrapper` and `_DuckDBResultWrapper` remain the same - they convert `None` and empty strings to `NaN` in `fetchdf()` results.

## Usage

### New API (Recommended)

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

### Old API (Backward Compatible)

```python
from lib.db import init_db
from lib.data_validator import validate_and_clean

# Old API still works
con = validate_and_clean(
    init_db(),
    verbose=True,
    rooms_to_exclude=['reception_hall'],
    exclude_missing_location_bookings=True
)
```

## Adding New Rules During EDA

When you find a new data quality issue:

**Step 1**: Add field to `CleaningConfig` in `lib/data_validator.py`

```python
@dataclass
class CleaningConfig:
    # ... existing fields
    remove_duplicate_bookings: bool = False  # NEW: Section X.Y
```

**Step 2**: Add rule to `DataCleaner._build_rules()`

```python
if self.config.remove_duplicate_bookings:
    rules.append(Rule(
        "Remove Duplicate Bookings",
        "SELECT COUNT(*) - COUNT(DISTINCT ...) FROM bookings",
        "DELETE FROM bookings WHERE ..."
    ))
```

**Step 3**: Use it in your notebook

```python
config = CleaningConfig(
    remove_duplicate_bookings=True,  # ← New field
    verbose=True
)
cleaner = DataCleaner(config)
con = cleaner.clean(init_db())
```

Done! The config field documents when/why it was added.

## Benefits

### 1. Unified Format
Everything is a `Rule` - no special cases for deletions vs exclusions vs imputations

### 2. Self-Documenting
```python
config = CleaningConfig(
    exclude_reception_halls=True,      # ← Clear what it does
    exclude_missing_location=True,     # ← No summary() needed
    impute_children_allowed=True,      # ← Field name IS the doc
)
```

### 3. Easy to Extend
Found new issue in EDA? Add one config field + one rule. That's it.

### 4. Single File
No need to jump between files. Everything related to data cleaning is in `data_validator.py`.

### 5. Backward Compatible
Old `validate_and_clean()` function still works - it creates a `CleaningConfig` internally.

### 6. Trackable
`cleaner.stats` shows exactly what was done:
```python
{
    'Negative Price': 3,
    'Zero Price': 12464,
    'Exclude Reception Halls': 2225,
    'Impute children_allowed': 6913,
    ...
}
```

## File Structure

```
lib/
├── __init__.py
├── db.py
└── data_validator.py    # Everything in one file:
                          #   - Rule dataclass
                          #   - CleaningConfig dataclass
                          #   - DataCleaner class
                          #   - DuckDBConnectionWrapper
                          #   - validate_and_clean() (deprecated)
```

## Example Script

See `notebooks/data_prep/data_checks_example.py` for a complete example showing:
- How to create a config
- How to apply cleaning
- How to review results
- How to add new rules

## Migration Guide

### For Existing Scripts

**Option 1**: Keep using old API (no changes needed)
```python
con = validate_and_clean(init_db(), verbose=True, rooms_to_exclude=['reception_hall'])
```

**Option 2**: Migrate to new API (recommended)
```python
config = CleaningConfig(exclude_reception_halls=True, verbose=True)
con = DataCleaner(config).clean(init_db())
```

### For New Scripts

Always use the new API:
```python
from lib.data_validator import CleaningConfig, DataCleaner

config = CleaningConfig(
    exclude_reception_halls=True,
    exclude_missing_location=True,
    verbose=True
)
cleaner = DataCleaner(config)
con = cleaner.clean(init_db())
```

## Testing

All existing tests pass with backward compatibility:
- ✅ Old `validate_and_clean()` API works
- ✅ New `CleaningConfig` + `DataCleaner` API works
- ✅ Connection wrapper converts None→NaN
- ✅ All EDA scripts continue to work

## Future Enhancements

Possible future additions:
1. Rule priorities/ordering
2. Conditional rules (if X then Y)
3. Rule dependencies
4. Dry-run mode (check without applying)
5. Export config to YAML/JSON
6. Rule templates for common patterns

But for now, the simple approach is sufficient and easy to understand.

