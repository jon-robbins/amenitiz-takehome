# Testing Suite

Comprehensive pytest-based testing suite for `lib/db.py` and `lib/data_validator.py`.

## Overview

This testing suite ensures that data loading and cleaning functionality works correctly and prevents regressions from future changes.

## Test Structure

```
tests/
├── conftest.py                # Shared fixtures and test data
├── test_db.py                 # Database connection tests (21 tests)
├── test_data_validator.py     # Data validator core tests (26 tests)
├── test_cleaning_rules.py     # Individual rule tests (15 tests)
├── test_integration.py        # End-to-end integration tests (12 tests)
├── test_pipeline.py           # Legacy pipeline tests (2 tests)
└── README.md                  # This file
```

**Total: 76 tests**

## Running Tests

### Run All Tests

```bash
poetry run pytest
```

### Run Specific Test File

```bash
poetry run pytest tests/test_db.py
poetry run pytest tests/test_data_validator.py
poetry run pytest tests/test_cleaning_rules.py
poetry run pytest tests/test_integration.py
```

### Run With Verbose Output

```bash
poetry run pytest -v
```

### Run Specific Test

```bash
poetry run pytest tests/test_cleaning_rules.py::TestBasicValidationRules::test_remove_negative_prices
```

### Run With Coverage

```bash
# Generate coverage report
poetry run pytest --cov=lib --cov-report=html

# View HTML report
open htmlcov/index.html
```

### Run Only Fast Tests (Skip Integration)

```bash
poetry run pytest -m "not integration"
```

### Run Only Integration Tests

```bash
poetry run pytest -m integration
```

## Test Categories

### 1. Database Tests (`test_db.py`)

Tests for `lib/db.py` functionality:

**Basic Functionality (3 tests):**
- Connection creation
- In-memory database
- Query execution

**Table Loading (5 tests):**
- All 4 tables loaded
- Schema validation for each table

**Type Casting (6 tests):**
- Integer columns
- BIGINT columns
- DOUBLE columns
- DATE columns
- TIMESTAMP columns
- BOOLEAN columns

**NULL Handling (2 tests):**
- NULLIF logic
- Empty string preservation

**Data Integrity (3 tests):**
- Row counts
- No data corruption
- Foreign key relationships

**Edge Cases (2 tests):**
- Multiple connections
- Connection reusability

### 2. Data Validator Tests (`test_data_validator.py`)

Tests for `lib/data_validator.py` core components:

**Rule (2 tests):**
- Rule creation
- Enabled/disabled flag

**CleaningConfig (4 tests):**
- Default config creation
- Default values
- Custom values
- All boolean flags

**DataCleaner (6 tests):**
- Initialization
- Rule building
- Rule counting
- Wrapped connection return
- Stats tracking
- Verbose mode

**DuckDBConnectionWrapper (6 tests):**
- Wrapper creation
- Execute returns wrapper
- None to NaN conversion
- Empty string to NaN conversion
- Both conversions
- Method delegation

**Backward Compatibility (5 tests):**
- validate_and_clean() works
- Parameters mapping
- Reception hall exclusion
- check_data_quality() works
- Quality check structure

**Rule Execution (3 tests):**
- Check query execution
- Action query execution
- Enabled flag functionality

### 3. Cleaning Rules Tests (`test_cleaning_rules.py`)

Tests for individual cleaning rules:

**Basic Validation Rules (7 tests):**
- Negative prices
- Zero prices
- NULL prices
- NULL room_id
- NULL dates
- Negative stay
- Cancelled but active

**Exclusion Rules (1 test):**
- Reception halls

**Fix Rules (2 tests):**
- Empty strings in room_view
- Empty strings in city

**Imputation Rules (1 test):**
- children_allowed imputation

**Rule Ordering (2 tests):**
- Missing location phase ordering
- Orphan cleanup after exclusions

**Combined Rules (2 tests):**
- Multiple price rules
- Full cleaning pipeline

### 4. Integration Tests (`test_integration.py`)

End-to-end workflow tests:

**Full Pipeline (4 tests):**
- Load and clean workflow
- Idempotency
- Stats accuracy
- Quality checks pass

**Real-World Scenarios (4 tests):**
- EDA workflow
- ML pipeline workflow
- Incremental cleaning
- Different configs

**Performance (2 tests - marked as slow):**
- Reasonable completion time
- Multiple cleanings efficiency

**Data Consistency (2 tests):**
- Foreign keys remain valid
- No data corruption

## Test Fixtures

### Shared Fixtures (`conftest.py`)

- `raw_connection`: Empty DuckDB connection
- `sample_booked_rooms`: Sample booked_rooms data (10 rows)
- `sample_bookings`: Sample bookings data (12 rows)
- `sample_rooms`: Sample rooms data (5 rows)
- `sample_hotel_location`: Sample hotel_location data (8 rows)
- `test_db_with_data`: Fully loaded test database
- `expected_test_data_stats`: Expected statistics for test data

### Test Data Characteristics

Test data includes:
- 1 negative price
- 1 zero price
- 1 NULL price
- 1 extreme price (>€5000/night)
- 1 NULL room_id
- 1 NULL date
- 1 negative stay
- 1 cancelled but active booking
- 1 reception hall
- 2 hotels with missing location
- 2 empty string cities
- 2 empty string room_views
- 2 bookings with children

## Test Markers

Custom pytest markers for test organization:

- `@pytest.mark.integration`: Integration tests (slower)
- `@pytest.mark.slow`: Performance tests (very slow)
- `@pytest.mark.unit`: Unit tests (fast)

Configure in `pytest.ini`:

```ini
[pytest]
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow
    unit: marks tests as unit tests
```

## Success Metrics

✅ **76 tests pass** (excluding slow performance tests)  
✅ **All test categories covered**  
✅ **Tests run in <60 seconds** (excluding slow tests)  
✅ **Clear test failure messages**  
✅ **High code coverage** for `lib/db.py` and `lib/data_validator.py`

## Coverage Goals

Target coverage for critical modules:

- `lib/db.py`: 90%+
- `lib/data_validator.py`: 90%+

Run coverage report:

```bash
poetry run pytest --cov=lib --cov-report=term-missing
```

## Adding New Tests

### When to Add Tests

Add tests when:
1. Adding new cleaning rules
2. Modifying existing rules
3. Refactoring core functionality
4. Fixing bugs
5. Adding new features

### Test Template

```python
def test_new_rule(test_db_with_data):
    """Test that new rule works correctly."""
    # Count problematic data before
    before = test_db_with_data.execute(
        "SELECT COUNT(*) FROM table WHERE condition"
    ).fetchone()[0]
    assert before > 0, "Test data should have problematic rows"
    
    # Apply cleaning with only this rule enabled
    config = CleaningConfig(
        new_rule=True,
        # Disable all other rules
        verbose=False
    )
    cleaner = DataCleaner(config)
    clean_con = cleaner.clean(test_db_with_data)
    
    # Count problematic data after
    after = clean_con.execute(
        "SELECT COUNT(*) FROM table WHERE condition"
    ).fetchone()[0]
    
    assert after == 0, "All problematic rows should be removed"
    assert cleaner.stats.get('Rule Name', 0) == before
```

## Continuous Integration

### GitHub Actions (Future)

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.13'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run tests
        run: poetry run pytest --cov=lib --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Tests Fail After Code Changes

1. Run specific failing test with verbose output:
   ```bash
   poetry run pytest tests/test_file.py::test_name -v
   ```

2. Check if test data needs updating:
   - Review `conftest.py` fixtures
   - Ensure test data matches new requirements

3. Check if new rules need tests:
   - Add tests to `test_cleaning_rules.py`
   - Update integration tests if needed

### Slow Test Execution

1. Run without integration tests:
   ```bash
   poetry run pytest -m "not integration"
   ```

2. Run specific test file:
   ```bash
   poetry run pytest tests/test_db.py
   ```

3. Use pytest-xdist for parallel execution:
   ```bash
   poetry add --dev pytest-xdist
   poetry run pytest -n auto
   ```

### Coverage Too Low

1. Check which lines are not covered:
   ```bash
   poetry run pytest --cov=lib --cov-report=term-missing
   ```

2. Add tests for uncovered code paths
3. Focus on critical functionality first

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Test names should describe what they test
3. **Single Assertion**: Each test should test one thing
4. **Fast Tests**: Keep tests fast (use fixtures, avoid I/O)
5. **Readable**: Tests serve as documentation
6. **Maintainable**: Update tests when code changes

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)
- [pytest-cov](https://pytest-cov.readthedocs.io/)

