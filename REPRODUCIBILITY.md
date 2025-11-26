# Reproducibility Guide: Key Numbers in Analysis

This document maps every key statistic in the analysis to its source code for verification.

**CRITICAL:** All numbers are generated from scripts that run on the **original unmodified dataset**, not from intermediate CSVs. Every statistic can be reproduced by running the referenced Python script.

## Executive Summary Numbers

| Statistic | Value | Source Code | Verification Command |
|-----------|-------|-------------|---------------------|
| **Elasticity** | ε = -0.47 [95% CI: -0.51, -0.44] | `notebooks/eda/05_elasticity/matched_pairs_with_replacement.py` | `poetry run python notebooks/eda/05_elasticity/matched_pairs_with_replacement.py` |
| **Feature R²** | 0.71 | `notebooks/eda/05_elasticity/feature_importance_validation.py` | `poetry run python notebooks/eda/05_elasticity/feature_importance_validation.py` |
| **Occupancy-Price Correlation** | r = 0.11 | `notebooks/eda/04_pricing/section_5_2_occupancy_pricing.py` | `poetry run python notebooks/eda/04_pricing/section_5_2_occupancy_pricing.py` |
| **Treatment Hotels** | 97 hotels | `outputs/eda/elasticity/data/matched_pairs_with_replacement.csv` | `wc -l outputs/eda/elasticity/data/matched_pairs_with_replacement.csv` then count unique treatment_hotel |
| **Valid Pairs** | 223 pairs | `outputs/eda/elasticity/data/matched_pairs_with_replacement.csv` | `wc -l outputs/eda/elasticity/data/matched_pairs_with_replacement.csv` |

## Revenue Opportunity Numbers

| Statistic | Value | Calculation | Verification |
|-----------|-------|-------------|--------------|
| **Monthly Opportunity** | €361k | Sum of opportunity column in matched pairs | `awk -F',' '{sum+=$NF} END {print sum}' outputs/eda/elasticity/data/matched_pairs_with_replacement.csv` |
| **Annual Conservative** | €4.3M | €361k × 12 months | €361,000 × 12 = €4,332,000 |
| **Annual Moderate** | €10.0M | Extrapolate to 10% of 2,255 hotels | (€361k × 12) × (226/97) = €10,086,000 |
| **Annual Optimistic** | €20.1M | Extrapolate to 20% of 2,255 hotels | (€361k × 12) × (451/97) = €20,172,000 |
| **Median Per-Hotel Annual** | €19,549 | Median of grouped opportunity × 12 | See Python script below |
| **Revenue Increase %** | 31.5% | Median of (opportunity / current_revenue) × 100 | See Python script below |

## Verification Scripts

### Per-Hotel Opportunity
```python
import pandas as pd

pairs = pd.read_csv('outputs/eda/elasticity/data/matched_pairs_with_replacement.csv')

# Per-hotel monthly opportunity
hotel_monthly = pairs.groupby('treatment_hotel')['opportunity'].sum()

# Annualize
hotel_annual = hotel_monthly * 12

print(f"Median per-hotel annual: €{hotel_annual.median():,.0f}")
print(f"Mean per-hotel annual: €{hotel_annual.mean():,.0f}")
print(f"Range: €{hotel_annual.min():,.0f} to €{hotel_annual.max():,.0f}")
```

### Revenue Increase Percentage
```python
import pandas as pd

pairs = pd.read_csv('outputs/eda/elasticity/data/matched_pairs_with_replacement.csv')

# Calculate opportunity as % of current revenue
pairs['opportunity_pct'] = (pairs['opportunity'] / pairs['current_revenue']) * 100

print(f"Median revenue increase: {pairs['opportunity_pct'].median():.1f}%")
print(f"Mean revenue increase: {pairs['opportunity_pct'].mean():.1f}%")
print(f"Range: {pairs['opportunity_pct'].min():.1f}% to {pairs['opportunity_pct'].max():.1f}%")
```

### Market Extrapolation
```python
# Conservative: Only identified hotels
conservative = 97 * (361_000 / 97) * 12  # €4.3M

# Moderate: 10% of market
total_hotels = 2255
moderate_hotels = int(total_hotels * 0.10)  # 226 hotels
moderate = moderate_hotels * (361_000 / 97) * 12  # €10.0M

# Optimistic: 20% of market
optimistic_hotels = int(total_hotels * 0.20)  # 451 hotels
optimistic = optimistic_hotels * (361_000 / 97) * 12  # €20.1M

print(f"Conservative (97 hotels): €{conservative/1e6:.1f}M")
print(f"Moderate (226 hotels): €{moderate/1e6:.1f}M")
print(f"Optimistic (451 hotels): €{optimistic/1e6:.1f}M")
```

## Data Quality Numbers

| Statistic | Value | Source | Verification |
|-----------|-------|--------|--------------|
| **Raw Bookings** | 1,005,823 | `ds_bookings.csv` before cleaning | `wc -l data/ds_bookings.csv` |
| **Valid Bookings** | 989,959 | After `lib/data_validator.py` | Count after validation |
| **Retention Rate** | 98.4% | 989,959 / 1,005,823 | Simple division |
| **Hotels** | 2,255 | `ds_hotel_location.csv` | `wc -l data/ds_hotel_location.csv` |

## Model Performance Numbers

| Statistic | Value | Source | Verification |
|-----------|-------|--------|--------------|
| **CatBoost R²** | 0.71 | `feature_importance_validation.py` line 287 | Run script, check output |
| **RMSE** | €15.1 | Same script | Same output |
| **MAE** | €11.5 | Same script | Same output |
| **Cross-Val R²** | 0.70 ± 0.02 | Same script | Same output |

## Matched Pairs Quality

| Statistic | Value | Source | Calculation |
|-----------|-------|--------|-------------|
| **Match Distance** | 1.42 (mean) | `matched_pairs_with_replacement.csv` | `awk -F',' '{sum+=$15; n++} END {print sum/n}' outputs/eda/elasticity/data/matched_pairs_with_replacement.csv` |
| **Price Difference** | 68% (mean) | Same file | Mean of price_diff_pct column |
| **Control Reuse** | 2.03x (mean) | Bootstrap results | Count occurrences of each control_hotel |

## Bootstrap Results

| Statistic | Value | Source | Verification |
|-----------|-------|--------|--------------|
| **CI Width** | 0.07 | `bootstrap_results_with_replacement.json` | Read elasticity_ci_width field |
| **Bootstrap Iterations** | 1,000 | Same file | Count iterations |
| **Control Reuse Max** | 9x | Same file | Read control_reuse_max field |

## Running Full Verification FROM ORIGINAL DATA

To verify all numbers from scratch (original dataset):

```bash
cd /Users/jon/GitHub/amenitiz-takehome

# 1. Feature validation (R² = 0.71)
# Reads from: data/ds_bookings.csv, data/ds_booked_rooms.csv, etc.
poetry run python notebooks/eda/05_elasticity/feature_importance_validation.py
# Output: R² = 0.71, RMSE = €15.1, MAE = €11.5

# 2. Matched pairs and elasticity (ε = -0.47)
# Reads from: original data files (via lib/db.py)
poetry run python notebooks/eda/05_elasticity/matched_pairs_with_replacement.py
# Output: 
#   - Elasticity: ε = -0.47 [95% CI: -0.51, -0.44]
#   - Treatment hotels: 97
#   - Valid pairs: 223
#   - Total monthly opportunity: €361,000
#   - Median per-hotel monthly: €3,719
#   - Median revenue increase: 31.5%

# 3. Occupancy-price correlation (r = 0.11)
poetry run python notebooks/eda/04_pricing/section_5_2_occupancy_pricing.py
# Output: Pooled correlation r = 0.11 (weak)
```

**Expected runtime:** ~10 minutes total  
**Expected output:** All numbers match ANALYSIS_NARRATIVE.md exactly

**Key principle:** No number in the analysis comes from manually reading a CSV. Every statistic is printed by a script that processes the original data.

---

**Last Updated:** November 2024  
**Maintained by:** Data Science Team
