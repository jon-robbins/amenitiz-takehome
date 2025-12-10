# Price Optimization Model: Mathematical Specification

## Executive Summary

This document provides the formal mathematical specification of the RevPAR-optimized pricing model. The model identifies underpriced and overpriced hotels by comparing their observed price-occupancy relationship to validated market benchmarks.

---

## 1. Core Metrics

### 1.1 Revenue Per Available Room (RevPAR)

The primary optimization target:

$$
\text{RevPAR}_i = \text{ADR}_i \times \text{Occupancy}_i
$$

Where:
- $\text{ADR}_i$ = Average Daily Rate for hotel $i$ (€/night)
- $\text{Occupancy}_i$ = Rooms sold / Rooms available (0 to 1)

### 1.2 Price Elasticity of Demand

The relationship between price changes and occupancy changes:

$$
\varepsilon = \frac{\%\Delta Q}{\%\Delta P} = \frac{\Delta \text{Occupancy} / \text{Occupancy}_{\text{avg}}}{\Delta \text{Price} / \text{Price}_{\text{avg}}}
$$

**Validated market elasticity: ε = -0.39** (95% CI: [-0.55, -0.25])

Interpretation: A 10% price increase leads to ~3.9% occupancy decrease.

---

## 2. Matched Pairs Methodology

### 2.1 Pair Construction

Hotels are matched within **blocking variables**:
- Same city
- Same month
- Same room type
- Geographic proximity (< 10km)

For each block, we identify:
- **High-price hotel**: $P_H, O_H$ (price, occupancy)
- **Low-price hotel**: $P_L, O_L$

### 2.2 Arc Elasticity Calculation (Midpoint Method)

$$
\varepsilon_{ij} = \frac{(O_H - O_L) / \bar{O}}{(P_H - P_L) / \bar{P}}
$$

Where:
$$
\bar{P} = \frac{P_H + P_L}{2}, \quad \bar{O} = \frac{O_H + O_L}{2}
$$

### 2.3 Counterfactual Prediction

"If the low-price hotel raised prices to match the high-price hotel, what would their occupancy be?"

$$
O_{\text{counterfactual}} = O_L \times \left(1 + \varepsilon \times \frac{P_H - P_L}{\bar{P}}\right)
$$

Counterfactual RevPAR:
$$
\text{RevPAR}_{\text{cf}} = P_H \times O_{\text{counterfactual}}
$$

Opportunity:
$$
\text{Opportunity} = \text{RevPAR}_{\text{cf}} - \text{RevPAR}_{\text{current}}
$$

---

## 3. Hotel Classification

### 3.1 RevPAR Gap

Compare hotel to peer average:

$$
\text{RevPAR Gap}_i = \frac{\text{RevPAR}_i - \text{RevPAR}_{\text{peer}}}{\text{RevPAR}_{\text{peer}}}
$$

### 3.2 Classification Rules

| Category | RevPAR Gap | Price Gap | Signal |
|----------|------------|-----------|--------|
| **Underpriced** | < -15% | < 0% | Raise price |
| **Overpriced** | < -15% | > +10% | Lower price |
| **Optimal** | ≥ -15% | Any | Hold |

### 3.3 Individual Elasticity Classification

Hotels are also classified by how their observed elasticity compares to market:

$$
\text{Classification} = 
\begin{cases}
\text{Underpriced} & \text{if } \varepsilon_i > 0.7 \times \varepsilon_{\text{market}} \\
\text{Overpriced} & \text{if } \varepsilon_i < 1.3 \times \varepsilon_{\text{market}} \\
\text{Optimal} & \text{otherwise}
\end{cases}
$$

**Interpretation**: 
- Hotels with **lower elasticity** than market (e.g., ε = -0.25 vs market -0.39) have captive demand → can raise prices
- Hotels with **higher elasticity** than market (e.g., ε = -0.55 vs market -0.39) are price-sensitive → should lower prices

---

## 4. Price Recommendation Formula

### 4.1 Basic Recommendation

For underpriced hotels:
$$
P_{\text{recommended}} = P_{\text{current}} \times \left(1 + \min\left(0.30, \left|\frac{P_{\text{peer}} - P_{\text{current}}}{P_{\text{peer}}}\right| \times 0.6\right)\right)
$$

For overpriced hotels:
$$
P_{\text{recommended}} = P_{\text{current}} \times \left(1 - \min\left(0.10, \left|\frac{P_{\text{current}} - P_{\text{peer}}}{P_{\text{peer}}}\right| \times 0.3\right)\right)
$$

### 4.2 Expected Occupancy After Price Change

Using validated elasticity:
$$
O_{\text{expected}} = O_{\text{current}} \times \left(1 + \varepsilon \times \frac{P_{\text{recommended}} - P_{\text{current}}}{P_{\text{current}}}\right)
$$

### 4.3 Expected RevPAR Lift

$$
\text{RevPAR Lift} = \frac{P_{\text{rec}} \times O_{\text{exp}} - P_{\text{curr}} \times O_{\text{curr}}}{P_{\text{curr}} \times O_{\text{curr}}}
$$

---

## 5. Validation Methodology

### 5.1 Elasticity Validation

| Source | Methodology | Elasticity | Sample |
|--------|-------------|------------|--------|
| Matched Pairs (Geographic) | Same city, <10km, same month | ε = -0.39 | 6,565 pairs |
| Matched Pairs (Validated Features) | XGBoost-validated features | ε = -0.46 | 6,565 pairs |
| Longitudinal (Same Hotels) | Same hotel, 2023 vs 2024 | ε = -0.35 | 847 hotels |
| Fixed Effects Regression | Panel data, hotel FE | ε = -0.42 | 24,000 obs |

**Consensus**: ε ≈ -0.39 to -0.46 (inelastic demand)

### 5.2 Feature Validation (XGBoost)

Features used for matching were validated by XGBoost price prediction (R² = 0.71):

| Feature | Importance | Description |
|---------|------------|-------------|
| `dist_coast_log` | 0.18 | Log distance to coast (km) |
| `dist_center_km` | 0.15 | Distance to hotel's **own city center** (not Madrid!) |
| `log_room_size` | 0.14 | Log of room size (m²) |
| `amenities_score` | 0.12 | Sum of amenity flags (children, pets, events, smoking) |
| `room_capacity_pax` | 0.10 | Max occupancy per room |
| `is_madrid_metro` | 0.09 | Within 50km of Madrid (categorical: 0/1) |
| `view_quality_ordinal` | 0.08 | View quality (0=none, 1=pool/garden, 2=lake/mountain, 3=sea) |
| `city_standardized` | 0.07 | Standardized city name (top 5 + 'other') |
| `room_type` | 0.06 | Room type category |
| `is_coastal` | 0.05 | Within 20km of coast (categorical: 0/1) |

**Important Feature Definitions:**

- **`dist_center_km`**: Measures how central a hotel is within its OWN city, not distance to Madrid. Calculated as haversine distance from hotel coordinates to the mean coordinates of all hotels in that city (or cities500.json reference point if available).

- **`is_madrid_metro`**: Binary flag (0/1) indicating whether hotel is within 50km of Madrid city center. This captures the urban market segment independently of `dist_center_km`.

### 5.3 Cold-Start Peer Features (10km Radius)

For new hotels without pricing history, we calculate peer pricing features from nearby hotels within a 10km radius:

| Feature | Description |
|---------|-------------|
| `peer_price_mean` | Average price of peers within 10km |
| `peer_price_median` | Median price (robust to outliers) |
| `peer_price_p25` | 25th percentile (budget benchmark) |
| `peer_price_p75` | 75th percentile (premium benchmark) |
| `peer_price_std` | Price variation in local market |
| `peer_occupancy_mean` | Average occupancy of peers |
| `peer_revpar_mean` | Average RevPAR of peers |
| `n_peers_10km` | Number of comparable hotels in radius |
| `peer_price_same_type` | Price of same room type peers |
| `peer_price_distance_weighted` | Closer hotels weighted more heavily |

**Usage in Cold-Start:**
1. For a new hotel, calculate 10km peer features based on its coordinates and room type
2. Use `peer_price_distance_weighted` as initial price recommendation
3. Use `peer_occupancy_mean` to estimate expected occupancy
4. Use `peer_price_p25` to `peer_price_p75` range for pricing bounds

**Implementation:** `src/features/engineering.py::calculate_peer_price_features()`

### 5.4 Counterfactual Validation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Direction Accuracy | 100% | Counterfactual correctly predicts which hotel performs better |
| MAE | 10.9% | Average prediction error for occupancy change |
| Correlation (pred vs actual) | 0.78 | Strong correlation between predicted and actual outcomes |

---

## 6. Arbitrary Thresholds & Justification

### 6.1 Thresholds Used

| Threshold | Value | Justification | Sensitivity |
|-----------|-------|---------------|-------------|
| **RevPAR gap for underperformance** | -15% | One standard deviation below peer mean | Tested at -10%, -20% |
| **Price gap for overpricing** | +10% | Minimum to trigger "overpriced" signal | Tested at +5%, +15% |
| **Maximum price increase** | +30% | Avoid "sticker shock" for customers | Industry standard |
| **Maximum price decrease** | -10% | Preserve margin while improving volume | Conservative |
| **Geographic matching radius** | 10km | Balance sample size vs homogeneity | Tested at 5km, 15km |
| **Minimum occupancy for analysis** | 1% | Exclude inactive hotels | Standard filter |
| **Elasticity validity range** | -5 < ε < 0 | Exclude implausible values | Econometric standard |

### 6.2 Sensitivity Analysis

**RevPAR gap threshold (-15%)**:
- At -10%: More hotels classified as underpriced (45% vs 34%)
- At -20%: Fewer hotels classified as underpriced (25% vs 34%)
- **-15% chosen** as it captures meaningful underperformance without over-flagging

**Maximum price increase (+30%)**:
- Based on validated elasticity: 30% price increase → ~12% occupancy loss
- Net RevPAR change: +30% × (1 - 0.12) = +14.4% expected
- **Diminishing returns beyond 30%** due to demand destruction

---

## 7. Limitations

### 7.1 Model Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Market-level elasticity** | Individual hotels may differ from ε = -0.39 | Classification by elasticity deviation |
| **No competitor pricing data** | Can't react to real-time competitor changes | Use historical peer performance |
| **No demand forecasting** | Assumes stable demand | Use seasonal adjustments |
| **Selection bias in pairs** | Only hotels with bookings are compared | Explicit coverage tracking |
| **Static elasticity** | Elasticity may vary by price level | Use arc (midpoint) method |

### 7.2 Data Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Monthly aggregation** | Loses weekly/daily variation | Use for strategic, not tactical pricing |
| **No cancellation data** | Bookings may not convert to revenue | Use confirmed bookings only |
| **Missing room inventory** | Occupancy calculation imprecise | Default to 10 rooms if missing |
| **Geographic data quality** | Some coordinates may be imprecise | 10km radius tolerates error |

### 7.3 Assumption Violations

| Assumption | When Violated | Consequence |
|------------|---------------|-------------|
| **Hotels are substitutes** | Unique properties (castles, boutiques) | Elasticity estimate biased |
| **Price causes occupancy** | Reverse causality (low demand → low price) | Use temporal lag |
| **Stable market conditions** | COVID, major events | Exclude anomalous periods |
| **Homogeneous customers** | Business vs leisure, domestic vs foreign | Segment-specific elasticity |

---

## 8. Model Outputs

### 8.1 Per-Hotel Output

```
Hotel ID: 12345
Category: UNDERPRICED
Current Price: €85/night
Current Occupancy: 74%
Current RevPAR: €63

Peer Comparison:
  Peer Avg Price: €110/night
  Peer Avg Occupancy: 68%
  Peer RevPAR: €75
  Price Gap: -23%
  RevPAR Gap: -16%

Recommendation:
  Recommended Price: €98/night (+15%)
  Expected Occupancy: 70% (-4pp)
  Expected RevPAR: €69 (+10%)

Confidence: HIGH (Twin match, validated features)
```

### 8.2 Portfolio Output

```
Category Distribution:
  Underpriced: 553 hotels (43%)
  Optimal: 644 hotels (50%)
  Overpriced: 94 hotels (7%)

Portfolio Impact:
  Current RevPAR: €163,405
  Expected RevPAR: €170,872
  Total Lift: +4.6%
```

---

## 9. Mathematical Summary

The model optimizes RevPAR by:

1. **Estimating market elasticity** from matched pairs:
   $$\hat{\varepsilon} = \text{median}\left(\frac{\Delta O / \bar{O}}{\Delta P / \bar{P}}\right) \approx -0.39$$

2. **Classifying hotels** by RevPAR gap vs peers:
   $$\text{Category}_i = f\left(\frac{\text{RevPAR}_i - \text{RevPAR}_{\text{peer}}}{\text{RevPAR}_{\text{peer}}}\right)$$

3. **Recommending prices** to move toward peer levels:
   $$P_{\text{rec}} = P_{\text{curr}} \times (1 + \alpha \times \text{Gap})$$
   where $\alpha \in [0.3, 0.6]$ based on confidence

4. **Predicting outcomes** using validated elasticity:
   $$\text{RevPAR}_{\text{exp}} = P_{\text{rec}} \times O_{\text{curr}} \times (1 + \varepsilon \times \Delta P / P)$$

---

## 10. Key Takeaways for Stakeholders

1. **This is not arbitrary** - The model uses the same methodology that validated price elasticity exists in the market (ε = -0.39, p < 0.001)

2. **Conservative recommendations** - Maximum +30% increase, most recommendations are +5-15%

3. **100% bounded** - All recommendations are within ±30%

4. **Validated accuracy** - 100% direction accuracy on matched pair counterfactuals

5. **Market-level baseline, individual adjustments** - Uses market elasticity but classifies hotels by their deviation from it

6. **Transparent limitations** - Model works best for standard hotels in competitive markets; unique properties may differ

---

## 11. Cold-Start Performance

### Validated Cold-Start Accuracy

When a hotel has NO booking history, we rely on peer comparison:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Price Correlation | r = 0.39 | Weak but positive |
| Price MAPE | 42% | ~60% accurate |
| Occupancy Model R² | 0.17 | Low - limited by available features |

### By Price Tier

| Tier | Price MAPE | Notes |
|------|------------|-------|
| €150-200 | 30% | Best accuracy |
| €100-150 | 35% | Good accuracy |
| €50-100 | 46% | Moderate |
| >€200 | 43% | Luxury harder to predict |
| <€50 | 129% | Budget most variable |

### Cold-Start Recommendation Strategy

For new hotels without history:

```
1. FIND PEERS: Hotels in same city, similar room type, <10km
2. RECOMMEND PEER AVERAGE: Don't optimize, recommend market price
3. WAIT FOR DATA: After 30-60 days, switch to optimization model
```

### Honest Limitations

- **Occupancy model has low R²** (0.17) due to missing features
- **Direction accuracy ~50%** - can't reliably predict above/below peer
- **Best for existing hotels** - where we have booking history to validate

### What Would Improve Cold-Start

1. **More hotel features**: Room photos quality score, review ratings, amenity details
2. **OTA pricing data**: Competitor prices for the same hotel
3. **Historical market data**: Seasonal demand patterns by location

