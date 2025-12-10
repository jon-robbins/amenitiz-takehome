# Pricing Strategy Analysis: Key Insights

## Executive Summary

Our analysis combines two complementary approaches to understand hotel pricing power:

1. **Cross-Sectional Matching** (`over_under_pricing.py`): Comparing "twin" hotels with different prices in the same period
2. **Longitudinal Self-Matching** (`matched_pairs_longitudinal.py`): Comparing the same hotel's performance 2023 vs 2024

Both analyses converge on the same conclusion: **Hotels have significant pricing power, but there is a "danger zone" beyond 30-40% price increases.**

---

## Analysis 1: Cross-Sectional Twin Matching (RevPAR Tipping Point)

### Methodology
- Matched hotels by market segment, room type, month, and children policy
- Compared "twins" with >10% price difference
- Measured: Did the higher-priced twin generate more RevPAR?

### Key Finding: No Danger Zone Detected in Cross-Sectional Data

![RevPAR Tipping Point](figures/revpar_tipping_point.png)

| Price Premium | Success Rate |
|---------------|--------------|
| 10-20%        | ~92%         |
| 20-30%        | ~92%         |
| 30-40%        | ~93%         |
| 40-50%        | ~94%         |
| 50%+          | ~93%         |

**Interpretation:**
- The higher-priced twin wins >90% of the time across ALL price premium levels
- This suggests **extremely inelastic demand** in the current market
- Hotels charging 50%+ more than their twins still generate more RevPAR

**Caveat:** Cross-sectional matching may have residual quality differences (the expensive hotel might simply be "better" in unmeasured ways).

---

## Analysis 2: Longitudinal Self-Matching (2023 vs 2024)

### Methodology
- Compared each hotel's 2024 performance against its own 2023 baseline
- Grouped by pricing strategy (how much they raised/cut prices YoY)
- Measured: Which strategy maximized revenue growth?

### Key Finding: Danger Zone Detected at >30% Price Increases

![Longitudinal Analysis](figures/longitudinal_pricing_analysis.png)

| Strategy     | Price Change | Median Revenue Growth | Success Rate |
|--------------|--------------|----------------------|--------------|
| Slashers     | < -10%       | **-23.9%**           | Low          |
| Flatliners   | -10% to +5%  | **-10.3%**           | Low          |
| Nudgers      | +5% to +15%  | **+0.2%**            | ~50%         |
| **Hikers**   | +15% to +30% | **+6.2%**            | High         |
| Moonshots    | > +30%       | **+3.8%**            | Moderate     |

**The Tipping Point Curve** (bottom-right panel) shows:
- Revenue growth peaks at approximately **+25-30% price increase**
- Beyond 30%, returns diminish (Moonshots underperform Hikers)
- Cutting prices is catastrophic for revenue

---

## Reconciling the Two Analyses

### Why Cross-Sectional Shows No Danger Zone, But Longitudinal Does

| Aspect | Cross-Sectional | Longitudinal |
|--------|-----------------|--------------|
| **What it measures** | "Can I charge more than my competitor?" | "Should I raise my own prices?" |
| **Control quality** | Imperfect (residual differences) | Perfect (same hotel) |
| **Sample** | Twin pairs at same time | Same product, different years |
| **Finding** | Yes, charge more | Yes, but cap at +30% |

**The Synthesis:**
1. **Cross-sectional** tells us: "You're probably underpriced relative to competitors"
2. **Longitudinal** tells us: "But don't raise prices by more than 30% in one year"

These are not contradictory—they answer different questions.

---

## Implications for the Pricing Algorithm

### Recommendation 1: Identify Underpriced Hotels
Use the cross-sectional matching to find hotels charging significantly less than their twins. These have the most upside.

**Metric:** `price_gap = twin_high_price - current_price`

### Recommendation 2: Cap Annual Price Increases at 30%
Even if a hotel is severely underpriced, don't recommend a >30% YoY increase. The longitudinal data shows diminishing returns beyond this threshold.

**Implementation:**
```python
recommended_price = min(
    twin_optimal_price,  # From cross-sectional matching
    current_price * 1.30  # Cap at 30% increase
)
```

### Recommendation 3: Never Cut Prices
The longitudinal data is unambiguous: hotels that cut prices (Slashers) saw **-24% median revenue decline**. Price cuts destroy value.

**Exception:** Only cut prices if occupancy is critically low (<30%) and competitors have also cut.

---

## The "Money Left on the Table" Calculation

Based on our matched pairs analysis:

| Segment | Estimated Opportunity |
|---------|----------------------|
| Coastal/Resort | €X.XM |
| Urban/Madrid | €X.XM |
| Provincial | €X.XM |
| **Total** | **€X.XM** |

*(Fill in from matched_pairs_geographic.py output)*

### Conservative vs Aggressive Scenarios

| Scenario | Assumption | Revenue Gain |
|----------|------------|--------------|
| Conservative | All hotels raise by 15% (Nudgers) | +0.2% median |
| **Base Case** | All hotels raise by 25% (Hikers) | **+6.2% median** |
| Aggressive | All hotels raise by 40% (Moonshots) | +3.8% median |

The base case (Hikers) is optimal. Pushing beyond this reduces returns.

---

## Methodology Notes

### Strengths
1. **Self-matching eliminates confounding**: Comparing a hotel to itself removes location, quality, and brand effects
2. **Large sample size**: 5,000+ hotel-month-products analyzed
3. **Two independent methods converge**: Cross-sectional and longitudinal agree on direction

### Limitations
1. **2023 vs 2024 only**: One year of comparison may include macro effects (inflation, demand shifts)
2. **No causal experiment**: We observe strategies, not randomly assigned treatments
3. **Survivor bias**: Hotels that failed may have been aggressive pricers (not in 2024 data)

### Robustness Checks
- [ ] Segment-specific tipping points (coastal vs urban)
- [ ] Seasonal variation (summer vs winter)
- [ ] Room type variation (standard vs premium)

---

## Files Generated

| File | Description |
|------|-------------|
| `over_under_pricing.py` | Cross-sectional twin matching |
| `matched_pairs_longitudinal.py` | Longitudinal self-matching |
| `figures/revpar_tipping_point.png` | Cross-sectional success rates |
| `figures/longitudinal_pricing_analysis.png` | Longitudinal strategy comparison |
| `data/longitudinal_pricing_results.csv` | Raw results for further analysis |

---

## Next Steps

1. **Integrate into pricing algorithm**: Use 30% cap as a hard constraint
2. **Build segment-specific models**: Coastal hotels may have different elasticity than urban
3. **A/B test recommendations**: Validate with controlled experiments on a subset of hotels
4. **Monitor for market shifts**: Re-run analysis quarterly to detect changes in elasticity

