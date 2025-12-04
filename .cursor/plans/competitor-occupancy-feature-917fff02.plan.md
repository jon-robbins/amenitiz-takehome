<!-- 917fff02-6489-4dd0-9541-48013a0085ef 6f46865d-a49d-438c-bf01-d5fe030a46f8 -->
# Add Peer Monthly Price Feature

## Goal

Improve daily price prediction by adding cluster/peer pricing from last month as a feature. This captures "what is the market rate for similar hotels?"

## Strategy Context

1. Build model that matches current hotel pricing behavior
2. Later: Add multiplier to recommend RevPAR-optimizing prices

## Feature to Add

### Peer Last Month Average Price

For each booking, get the average price of K=50 similar hotels (from KNN) during the **previous month**:

```python
# For a March 15 booking, get avg price of similar hotels in February
peer_last_month_avg = similar_hotels_feb['daily_price'].mean()
```

**Why last month?**

- No data leakage (we definitely have February data for a March booking)
- Captures recent market conditions
- Provides benchmark for pricing decisions

### Optional: Hotel's Own Last Month Average

For comparison, what THIS hotel charged last month:

```python
hotel_last_month_avg = this_hotel_feb['daily_price'].mean()
```

## Implementation

### Files to Modify

- [ml_pipeline/competitor_features.py](ml_pipeline/competitor_features.py) - Add `peer_last_month_avg` feature using existing KNN similar hotels mapping
- [ml_pipeline/two_stage_model.py](ml_pipeline/two_stage_model.py) - Add feature to Stage 2 feature list, add MAE metric

### Key Logic

```python
# In competitor_features.py
def get_peer_last_month_price(hotel_id, booking_month, similar_hotels, all_bookings):
    last_month = booking_month - 1  # Handle Jan -> Dec
    peer_prices = all_bookings[
        (all_bookings['hotel_id'].isin(similar_hotels)) &
        (all_bookings['month'] == last_month)
    ]['daily_price']
    return peer_prices.mean()
```

## Expected Impact

- Strong predictor: captures market rate for comparable hotels
- Should improve R² by explaining "between-hotel" variance
- Provides foundation for later multiplier-based recommendations

### To-dos

- [ ] Create competitor_features.py with blocking and occupancy calculation logic
- [ ] Integrate competitor features into two_stage_model.py Stage 2
- [ ] Run analyze_mape_by_segment.py to measure MAPE improvement by segment
- [ ] Create competitor_features.py with KNN-based occupancy calculation
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Create competitor_features.py with blocking and occupancy calculation logic
- [ ] Integrate competitor features into two_stage_model.py Stage 2
- [ ] Run analyze_mape_by_segment.py to measure MAPE improvement by segment
- [ ] Create competitor_features.py with KNN-based occupancy calculation
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Create competitor_features.py with blocking and occupancy calculation logic
- [ ] Integrate competitor features into two_stage_model.py Stage 2
- [ ] Run analyze_mape_by_segment.py to measure MAPE improvement by segment
- [ ] Create competitor_features.py with KNN-based occupancy calculation
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics
- [ ] Add peer_last_month_avg feature using KNN similar hotels
- [ ] Add hotel_last_month_avg for comparison
- [ ] Add MAE metric to evaluation output
- [ ] Run full model and compare metrics