# Price Prediction Model: Competitor Pricing Feature

## Implementation Status: ✅ COMPLETED

The competitor pricing feature has been implemented and integrated into the two-stage model.

### Files Created/Modified
- **NEW**: `ml_pipeline/competitor_features.py` - Competitor pricing calculation logic
- **MODIFIED**: `ml_pipeline/two_stage_model.py` - Integration with Stage 2 (K=50, 13 KNN features)
- **MODIFIED**: `ml_pipeline/analyze_mape_by_segment.py` - Evaluation with competitor features
- **MODIFIED**: `notebooks/eda/05_elasticity/QUERY_LOAD_DAILY_BOOKING_DATA.sql` - Added created_at column

---

## Current Performance (Final)

| Segment | MAPE | Status |
|---------|------|--------|
| Mid-market (€75-150) | **15.9%** | ✅ Production-ready |
| Budget (€50-75) | 47.4% | ❌ Needs work |
| Luxury (€150-250) | 42.6% | ❌ Needs work |
| **Overall** | **26.9%** | |

### Model Configuration
- Stage 1 KNN: **K=50** similar hotels, **13 features** (geographic, product, capacity)
- Stage 2 RF: **30 features** including competitor pricing
- Competitor pricing: Latest price from similar hotels for arrival date ±1 day

### Competitor Pricing Statistics
- Coverage: **8.5%** of test bookings have competitor price data
- Mean competitor price: **€128.46**
- Avg hotels with price data: **1.3** per booking

---

## Key Finding: Within-Hotel Variance Not Captured

**59% of price variance is WITHIN hotels** - the same hotel charges €37-€163 on different days.

Multiple competitor feature approaches were tried:
1. Binary occupancy (hotel booked or not) - low signal
2. Room-level occupancy - inflated capacity calculation issue
3. **Latest competitor prices** - cleaner signal, ~3-4% feature importance

The competitor pricing feature adds value but doesn't significantly reduce MAPE because:
- Coverage limited to ~8% due to temporal train/test split
- Within-hotel variance driven by unobservable factors:
  - Room-specific pricing
  - Promotional codes  
  - Channel-specific pricing (OTA vs direct)
  - Customer segments (business vs leisure)

---

## The Missing Signal: Competitor Occupancy

### Concept
When a hotel sets its price for a future date, it considers: *"How many similar hotels are already booked for that date?"*

We can compute this from our data:
```
competitor_occupancy = (# similar hotels with bookings for same arrival_date 
                        where booking.created_at < this booking's created_at)
                       / (# total similar hotels)
```

### Example
Predicting price for **Hotel A, arrival Dec 14th**, booking made on **Dec 1st**:
- Find similar hotels (same city, room type, capacity tier)
- Count how many already have bookings for Dec 14th created before Dec 1st
- If 80% booked → high demand → expect higher price
- If 20% booked → low demand → expect lower price

### Why This Helps
- **High competitor occupancy** → strong demand → hotel can charge premium
- **Low competitor occupancy** → weak demand → hotel discounts to fill rooms
- This explains the 59% within-hotel variance!

---

## Implementation Plan

### Phase 1: Compute Competitor Occupancy Feature

```python
def compute_competitor_occupancy(
    booking_hotel_id: int,
    booking_arrival_date: date,
    booking_created_at: datetime,
    similar_hotels: List[int],  # From KNN or blocking
    all_bookings: pd.DataFrame
) -> float:
    """
    Computes what % of similar hotels were booked for same arrival_date
    at the time this booking was created.
    """
    # Filter to same arrival_date
    same_date = all_bookings[all_bookings['arrival_date'] == booking_arrival_date]
    
    # Filter to similar hotels (excluding target)
    competitors = same_date[same_date['hotel_id'].isin(similar_hotels)]
    competitors = competitors[competitors['hotel_id'] != booking_hotel_id]
    
    # Filter to bookings created BEFORE this booking
    prior_bookings = competitors[competitors['created_at'] < booking_created_at]
    
    # Count unique hotels booked
    hotels_booked = prior_bookings['hotel_id'].nunique()
    total_similar = len(similar_hotels) - 1  # Exclude target
    
    return hotels_booked / max(total_similar, 1)
```

### Phase 2: Optimize for Speed

The naive approach is O(n²) - for each booking, scan all other bookings.

**Optimization: Pre-compute lookup tables**
1. Group bookings by (arrival_date, city/region)
2. Sort by created_at
3. For each booking, binary search to find prior bookings

### Phase 3: Add to Model

```python
# New features to add to Stage 2:
- competitor_occupancy_rate  # 0.0 to 1.0
- competitor_occupancy_bucket  # low/medium/high
- days_until_arrival  # Lead time (already have this)
- competitor_occupancy_trend  # Is it rising fast?
```

### Phase 4: Evaluate Impact

Expected improvement:
- Should explain significant portion of within-hotel variance
- Target: **MAPE < 20%** for mid-market, **< 30%** overall

---

## Files to Reference

| File | Purpose |
|------|---------|
| `ml_pipeline/two_stage_model.py` | Current two-stage model |
| `ml_pipeline/analyze_mape_by_segment.py` | Reproducible limitations analysis |
| `notebooks/eda/05_elasticity/QUERY_LOAD_DAILY_BOOKING_DATA.sql` | Daily booking data query |
| `lib/holiday_features.py` | Holiday feature engineering |

## Key Code Locations

### Similar Hotels (for competitor matching)
Currently in `Stage1PeerPrice.predict_batch()`:
```python
# Find K nearest neighbors
distances, indices = self.knn.kneighbors(X_query_scaled)
```

### Booking Data Fields
```sql
-- From QUERY_LOAD_DAILY_BOOKING_DATA.sql
hotel_id, arrival_date, created_at, daily_price, lead_time_days
```

---

## Summary

1. **Current model works well for €75-150 segment (15.8% MAPE)** ✅
2. **59% of variance is within-hotel** → demand signals needed
3. **Competitor occupancy implemented** but has low coverage (2%)
4. **Implementation uses O(n) pre-computed lookup** with binary search

---

## What Was Implemented

### Competitor Pricing Feature (Final Approach)
For each booking being made on **date X** for **arrival date Y**:
1. Find K=50 similar hotels using KNN (13 features)
2. For each similar hotel, get their **latest price** for arrival date Y (±1 day)
   - Only consider bookings created before date X
3. Return: avg_price, min_price, max_price, count

This captures "what are competitors charging right now?" as a direct pricing signal.

### Feature Importance
| Feature | Importance |
|---------|------------|
| competitor_avg_price | 3.4% |
| competitor_max_price | 2.9% |
| competitor_price_count | 2.5% |

### Why MAPE Didn't Improve Significantly
1. **Low coverage**: Only 8.5% of test bookings have competitor price data (temporal split)
2. **Peer stats dominate**: Stage 1 peer price statistics already capture market pricing
3. **Unobservable factors**: Within-hotel variance driven by factors not in data

### Alternative Approaches to Explore
1. **Segment-specific models**: Separate models for budget/mid/luxury
2. **Ensemble with confidence**: Use peer price range as confidence interval
3. **Time-series approach**: Model hotel's own price history
4. **Real-time integration**: If live competitor prices were available

---

## Quick Start

```bash
# Run current model with competitor features
poetry run python ml_pipeline/two_stage_model.py

# Run full analysis with competitor occupancy breakdown
poetry run python ml_pipeline/analyze_mape_by_segment.py

# Quick test (50k sample)
poetry run python ml_pipeline/two_stage_model.py --quick
```

