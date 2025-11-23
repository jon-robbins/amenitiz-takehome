# City Name Consolidation using TF-IDF & Cosine Similarity

## Problem

The raw booking data has inconsistent city naming:
- **Compound names**: "San Pere de Ribes, Barcelona" should map to "Barcelona"
- **Case variations**: "MADRID", "Madrid", "madrid" are all the same city
- **Suburbs vs metro**: "Badalona" is effectively part of Barcelona metro area
- **Typos & variations**: "València" vs "Valencia", "Leon" vs "León"

This creates **1,480 unique city names** when there should be far fewer distinct markets.

## Solution: TF-IDF + Cosine Similarity

### Approach

1. **Identify canonical cities**: Cities with ≥1,000 bookings become "canonical" (167 cities)
2. **Extract compound names**: If city contains comma, extract the part after comma (e.g., "Suburb, City" → "City")
3. **TF-IDF vectorization**: Convert city names to character n-gram vectors (1-3 chars)
4. **Cosine similarity matching**: Match each city to most similar canonical city (threshold: 0.6)
5. **Fallback**: If no match above threshold, keep original name

### Implementation

Located in `notebooks/eda/utils/city_matcher.py`:

```python
from notebooks.eda.utils.city_matcher import (
    create_city_mapping,
    apply_city_mapping,
)

# Create mapping
city_mapping = create_city_mapping(
    df,
    city_col="city",
    min_bookings_canonical=1000,
    similarity_threshold=0.6,
)

# Apply to dataframe
df = apply_city_mapping(
    df, 
    city_mapping, 
    city_col="city", 
    new_col="city_canonical"
)
```

## Results

### Overall Impact
- **Before**: 1,480 unique cities
- **After**: 1,132 unique cities
- **Reduction**: 348 cities (23.5%)
- **Mappings applied**: 360 name changes

### Top City Changes

| City | Before | After | Gain |
|------|--------|-------|------|
| Madrid | 42,870 | 47,530 | +4,660 |
| Málaga | 12,421 | 24,958 | +12,537 |
| Barcelona | 19,494 | 24,074 | +4,580 |
| Sevilla | 18,715 | 21,496 | +2,781 |
| Valencia | 5,871 | 12,247 | +6,376 |
| A Coruña | 4,679 | 11,231 | +6,552 |

**Key insight**: Major cities gained significant booking counts by absorbing suburbs and variations.

### Consolidation Categories

1. **Compound names** (212 mappings, 58.9%)
   - `Marbella, Málaga` → `Málaga`
   - `Santiago de Compostela, A Coruña` → `A Coruña`
   - `Nerja, Málaga` → `Málaga`

2. **Fuzzy matches** (125 mappings, 34.7%)
   - `Leon` → `León` (accent correction)
   - `València` → `Valencia` (language variation)
   - `El Burgo de Osma` → `Burgos` (similar name)

3. **Case normalization** (23 mappings, 6.4%)
   - `MADRID` → `Madrid`
   - `VALENCIA` → `Valencia`
   - `TARIFA` → `Tarifa`

## Example Consolidations

### Málaga Metro Area
Before consolidation, Málaga bookings were split across:
- Málaga: 12,421
- Marbella, Málaga: (various)
- Nerja, Málaga: (various)
- Estepona, Málaga: (various)

After consolidation:
- **Málaga: 24,958** (+12,537 bookings, +101% increase)

### Barcelona Metro Area
Before:
- Barcelona: 19,494
- Various Girona suburbs with "Girona" suffix

After:
- Barcelona: 24,074 (+4,580)
- Girona: 16,769 (NEW - consolidated from suburbs)

### Madrid Metro Area
Before:
- Madrid: 42,870
- Alcalá de Henares, Madrid: (various)

After:
- **Madrid: 47,530** (+4,660 bookings)

## Files Generated

### 1. `city_name_mapping.csv`
Complete mapping of all 1,480 original cities to canonical names.

Columns:
- `original_city`: Raw city name from data
- `canonical_city`: Matched canonical name
- `changed`: Boolean indicating if name was modified

### 2. `city_consolidation_comparison.csv`
Before/after comparison for top cities.

Columns:
- `city_before`: Original city name
- `bookings_before`: Booking count before consolidation
- `city_after`: Canonical city name
- `bookings_after`: Booking count after consolidation

## Usage in Analysis

### In Section 3.1 Integrated Analysis

The `section_3_1_integrated.py` script now automatically applies city consolidation:

```python
city_df, city_mapping = load_city_analysis(con, use_canonical=True)
```

This reduces the city count from 1,480 → 1,132 and increases market concentration (HHI) from 0.0081 → 0.0124, making the top cities more prominent.

### For Pricing Model

Recommended feature engineering:

```python
# Option 1: Use canonical city names directly
df['city'] = df['city'].map(city_mapping)

# Option 2: Create city tiers based on consolidated booking volume
city_tiers = {
    'Tier 1': ['Madrid', 'Barcelona', 'Málaga', 'Sevilla'],  # >20k bookings
    'Tier 2': [...],  # 10-20k bookings
    'Tier 3': [...],  # 5-10k bookings
    'Tier 4': [...],  # <5k bookings
}

# Option 3: One-hot encode top N canonical cities, group rest as "Other"
top_20_cities = city_df.head(20)['city'].tolist()
df['city_grouped'] = df['city'].apply(
    lambda x: city_mapping.get(x, x) if city_mapping.get(x, x) in top_20_cities else 'Other'
)
```

## Running the Scripts

### Demo Script (Recommended)
Shows clear before/after comparison:
```bash
poetry run python notebooks/eda/city_consolidation_demo.py
```

### Integrated Analysis
Runs full Section 3.1 with city consolidation:
```bash
poetry run python notebooks/eda/section_3_1_integrated.py
```

### Standalone Matcher
Just creates the mapping:
```bash
poetry run python notebooks/eda/utils/city_matcher.py
```

## Advantages over DBSCAN

| Aspect | TF-IDF Matching | DBSCAN Clustering |
|--------|----------------|-------------------|
| **Interpretability** | ✅ Clear name mappings | ❌ Arbitrary cluster IDs |
| **Business alignment** | ✅ Uses actual city names | ❌ Geographic clusters don't match admin boundaries |
| **Simplicity** | ✅ Simple string matching | ❌ Requires tuning eps/min_samples |
| **Stability** | ✅ Deterministic | ⚠️ Sensitive to parameter changes |
| **New data** | ✅ Easy to map new cities | ❌ Requires re-clustering |
| **Explainability** | ✅ Can show why cities matched | ❌ Hard to explain cluster membership |

## Limitations

1. **Threshold sensitivity**: similarity_threshold=0.6 is somewhat arbitrary
2. **False positives**: "El Burgo de Osma" → "Burgos" may be too aggressive
3. **Missing context**: Doesn't use geographic coordinates (could improve matching)
4. **Canonical selection**: min_bookings=1000 is arbitrary cutoff

## Future Improvements

1. **Add geographic validation**: Only match cities within X km of each other
2. **Manual overrides**: Allow curated list of known mappings (e.g., "Badalona" → "Barcelona")
3. **Hierarchical matching**: Match to province first, then city within province
4. **Interactive review**: Tool to review/approve/reject suggested mappings
5. **Confidence scores**: Return similarity scores for manual review of low-confidence matches

## Conclusion

TF-IDF city name matching successfully reduces city count by **23.5%** while maintaining interpretability and business alignment. This is a much simpler and more practical solution than DBSCAN clustering for the pricing model use case.

**Recommended for production use** with periodic manual review of mappings.

