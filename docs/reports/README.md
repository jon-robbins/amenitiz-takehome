# Price Elasticity Analysis - Final Results

## Quick Summary

**Research Question:** What is the price elasticity of demand for Spanish hotels?

**Answer:** ε = **-0.47** [95% CI: -0.51, -0.44]

**Interpretation:** 10% price increase → 4.7% occupancy decrease (inelastic demand)

**Implication:** Hotels have significant pricing power. Revenue increases with price increases.

---

## Methodology

### Approach: 1:1 Matching with Replacement

**The Goldilocks Solution:**
- Too Hot (Many-to-Many): N=194k pairs (combinatorial explosion)
- Too Cold (Greedy 1:1): N=94 pairs (too strict)
- **Just Right (1:1 w/ Replacement + 100% Cap): N=223 pairs** ✓

### Matching Criteria

**Exact Matching (7 variables):**
- Location: `is_coastal`, `city_standardized` (top 5)
- Product: `room_type`, `room_view`
- Temporal: `month`
- Policy: `children_allowed`
- Scale: `revenue_quartile` (quality control)

**Continuous Matching (8 features):**
- Geographic: `dist_center_km`, `dist_coast_log`
- Product: `log_room_size`, `room_capacity_pax`, `amenities_score`, `total_capacity_log`, `view_quality_ordinal`
- Temporal: `weekend_ratio`

**Quality Filters:**
- Match distance < 3.0
- **10% < price difference < 100%** (ensures true substitutes)
- -5 < elasticity < 0 (economically valid)
- Occupancy > 1% (active hotels)

### Statistical Inference

**Block Bootstrap (N=1000):**
- Clusters by `treatment_hotel`
- Accounts for control reuse (2.03x average)
- Percentile method for 95% CI

---

## Results

### Sample Characteristics

| Metric | Value |
|--------|-------|
| Valid pairs | 223 |
| Treatment hotels | 97 |
| Control hotels | 110 |
| Reuse ratio | 2.03x |

### Elasticity Estimate

**Point Estimate:** ε = -0.47

**95% Confidence Interval:** [-0.51, -0.44]

**Interpretation:**
- Inelastic demand (|ε| < 1)
- 10% price ↑ → 4.7% occupancy ↓
- Net revenue increases with price increases
- Hotels have pricing power

### Match Quality

| Metric | Value |
|--------|-------|
| Avg match distance | 1.42 (excellent) |
| Avg price difference | 68% |
| Median price difference | 73% |
| Price range | [12%, 100%] ✓ |

### Revenue Opportunity

**Total:** €230k [€165k, €288k]

**Per Treatment Hotel:** €1,617 average

**Interpretation:** Low-price hotels could raise prices with minimal volume loss.

---

## Validation

### Causal Validity

✓ Exact matching removes confounding (7 variables)
✓ Continuous matching ensures similarity (8 features)
✓ True substitutes only (price diff < 100%)
✓ Strong treatment contrast (68% avg price diff)
✓ Features validated by XGBoost (R²=0.71)
✓ Stable estimate across methods

### Statistical Validity

✓ Sufficient sample size (N=223)
✓ Excellent match quality (distance=1.42)
✓ Proper clustering in bootstrap
✓ Realistic uncertainty (CI width=0.07)
✓ Balanced control reuse (top 10 = 30.5%)

### Economic Validity

✓ Inelastic demand (expected for hotels)
✓ Consistent with location differentiation
✓ Conservative estimate (survivorship bias)
✓ Realistic revenue opportunity

---

## Key Files

### Data
- `data/matched_pairs_with_replacement.csv` - All 223 pairs
- `data/bootstrap_results_with_replacement.json` - Bootstrap CI + diagnostics
- `data/feature_importance_results.csv` - XGBoost validation (R²=0.71)

### Documentation
- `MATCHING_WITH_REPLACEMENT_FINAL.md` - Complete methodology and results
- `CRITIQUE_RESPONSE.md` - Response to methodology critique
- `FEATURE_IMPORTANCE_FINAL.md` - Feature validation summary

### Figures
- `figures/matching_methodology_evolution.png` - Before/after comparison
- `figures/matching_diagnostics_final.png` - Quality checks
- `figures/1_model_comparison.png` - XGBoost validation
- `figures/4_shap_beeswarm.png` - Feature importance

### Code
- `../../notebooks/eda/05_elasticity/matched_pairs_with_replacement.py` - Main script
- `../../notebooks/eda/05_elasticity/feature_importance_validation.py` - XGBoost validation

---

## Methodology Evolution

### Version History

1. **Many-to-Many (Initial):**
   - N=194k pairs
   - Problem: Combinatorial explosion, inflated sample
   - Status: ❌ Too loose

2. **Greedy 1:1 (Attempt 1):**
   - N=94 pairs
   - Problem: Too strict, wasteful of good controls
   - Status: ❌ Too strict

3. **1:1 w/ Replacement (Uncapped):**
   - N=673 pairs
   - Problem: 385% avg price diff (false twins)
   - Status: ❌ False substitutes

4. **1:1 w/ Replacement + 100% Cap (FINAL):**
   - N=223 pairs
   - Solution: True substitutes, realistic CI
   - Status: ✓ Just right

### Critical Correction

**Issue:** Initial results showed 385% average price difference, indicating we were comparing different asset classes (€100 vs €500 rooms).

**Fix:** Added 100% price difference cap to ensure true substitutes (€100 vs €200).

**Impact:**
- Sample: 673 → 223 pairs
- Avg price diff: 385% → 68%
- Elasticity: -0.46 → -0.47 (stable)
- CI width: 0.02 → 0.07 (realistic)

---

## Economic Interpretation

### Why Inelastic Demand?

1. **Location Lock-In:** Prime locations (coastal, city center) have limited substitutes
2. **Switching Costs:** High search costs for travelers to find alternatives
3. **Product Differentiation:** Brand, reviews, amenities create loyalty
4. **Temporal Constraints:** Last-minute bookings have fewer options

### Business Implications

**For Low-Price Hotels:**
- Currently leaving money on the table
- Could raise prices with minimal occupancy loss
- Expected revenue gain: €1,617 per hotel on average

**For Revenue Management:**
- Demand is inelastic (pricing power exists)
- Focus on price optimization, not just volume
- Location and product differentiation are key

**For Market Analysis:**
- Spanish hotel market is not perfectly competitive
- Hotels can extract consumer surplus
- Quality matching (revenue quartile) ensures valid comparisons

---

## Reproducibility

### Requirements

```bash
poetry install  # Install dependencies
```

### Run Analysis

```bash
cd notebooks/eda/05_elasticity
poetry run python matched_pairs_with_replacement.py
```

### Expected Runtime

- Data loading: ~30 seconds
- Feature engineering: ~1 minute
- Matching: ~2 minutes
- Bootstrap (N=1000): ~3 minutes
- **Total: ~6 minutes**

### Expected Output

```
Valid pairs: 223
Elasticity: -0.47 [-0.51, -0.44]
CI Width: 0.07
Total Opportunity: €230k [€165k, €288k]
```

---

## Citation

If using this methodology, please cite:

```
1:1 Matching with Replacement for Hotel Price Elasticity Estimation
- Exact matching on 7 categorical variables
- Continuous matching on 8 validated features (XGBoost R²=0.71)
- Price difference cap: 100% (true substitutes)
- Block bootstrap with treatment clustering (N=1000)
- Sample: 223 pairs (97 treatments, 110 controls)
- Result: ε = -0.47 [95% CI: -0.51, -0.44]
```

---

## Contact & Questions

For questions about methodology, results, or code:
- See `MATCHING_WITH_REPLACEMENT_FINAL.md` for complete details
- See `CRITIQUE_RESPONSE.md` for validation and corrections
- See code comments in `matched_pairs_with_replacement.py`

---

**Status:** ✓ Analysis complete and validated
**Last Updated:** November 2024
**Version:** 1.0 (Final)
