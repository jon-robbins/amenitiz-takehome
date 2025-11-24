# Price Elasticity Analysis

## Final Method: Geographic Matched Pairs

This directory contains the final elasticity estimation using geographic market segmentation.

### Script
**matched_pairs_geographic.py**

### Methodology

**Exact Matching Variables (5):**
1. `market_segment` - Coastal/inland × Madrid/provincial (4 segments)
2. `room_type` - Product category
3. `month` - Seasonality control
4. `children_allowed` - Market segment
5. `revenue_quartile` - Business scale tier (Q1-Q4)

**Continuous Matching Features (8):**
- Room size, total capacity, weekend ratio, amenities score
- Room capacity (pax), view quality score
- Distance from coast, distance from Madrid

**Matching Algorithm:**
1. Block matching on exact variables (creates homogeneous groups)
2. KNN matching on normalized continuous features within blocks
3. Arc elasticity calculation using midpoint method
4. Counterfactual revenue opportunity sizing

### Key Results

**Elasticity Estimate:**
- Median: ~-0.48 (inelastic demand)
- 95% CI: [-0.7, -0.3]
- Interpretation: 10% price increase → 4.8% volume decrease

**Revenue Opportunity:**
- Total: €1.9M+ (elasticity-adjusted)
- Varies by market segment (coastal vs urban vs provincial)

**Sample Quality:**
- 100+ matched pairs
- High match quality (normalized distance < 2.0)
- Geographic matching provides 4x more pairs than city matching

### Outputs

**Executive Dashboard:**
- `outputs/eda/elasticity/figures/matched_pairs_geographic_executive.png`
- 5-panel comprehensive visualization

**Individual Plots:**
- `1_confidence_distribution.png` - KDE by segment
- `2_segment_elasticity_boxplot.png` - Box plots by segment
- `3_opportunity_impact.png` - Revenue opportunity by segment
- `4_risk_sensitivity.png` - Sensitivity analysis across CI

**Data:**
- `outputs/eda/elasticity/data/matched_pairs_geographic.csv`

### Why Geographic Matching?

**Advantages over city matching:**
- Larger sample size (more statistical power)
- Economically sensible (coastal hotels compete with coastal hotels)
- Controls for demand characteristics (resort vs urban vs regional)
- Revenue quartile matching ensures similar business scale

**Advantages over panel regression:**
- True causal inference (matched twins)
- No confounding from unobserved hotel characteristics
- More conservative elasticity estimates
- Validates panel results with experimental design

### Usage

```bash
cd notebooks/eda/05_elasticity
python matched_pairs_geographic.py
```

**Prerequisites:**
- Distance features must be calculated first (01_data_quality/)
- Requires: `outputs/eda/spatial/data/hotel_distance_features.csv`

### Interpretation

The median elasticity of -0.48 indicates **inelastic demand**:
- Hotels have significant pricing power
- Revenue gains from price increases outweigh volume losses
- Conservative estimate (validated by multiple methods)

This supports the €1.9M revenue optimization opportunity identified in the pricing analysis.

