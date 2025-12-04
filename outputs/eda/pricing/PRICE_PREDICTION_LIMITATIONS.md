# Price Prediction Model: Limitations Analysis

## Executive Summary

Our two-stage price prediction model achieves **26% overall MAPE**, but this masks significant variation by price segment:

| Price Segment | MAPE | Accuracy | Recommendation |
|---------------|------|----------|----------------|
| Budget (€50-75) | 44% | Poor | Exclude or separate model |
| **Mid-market (€75-150)** | **15%** | **Excellent** | **Primary use case** |
| Luxury (€150-250) | 36% | Poor | Exclude or separate model |

## Key Finding: Within-Hotel Price Variance

**59% of total price variance is WITHIN hotels** (same hotel charging different prices on different days).

This is the fundamental limitation: we can predict "Hotel X typically charges €100" but not "Hotel X is charging €150 today because of a local event."

### Missing Signals
- Real-time occupancy at booking time
- Competitor prices that day
- Local events (concerts, conferences, festivals)
- Demand signals (search volume, booking velocity)

## Error Analysis by Segment

### Budget Hotels (€50-75): 44% MAPE
- Likely includes hostels, guesthouses, B&Bs
- More erratic pricing patterns
- Seasonal closures, irregular availability
- Different pricing dynamics than traditional hotels

### Mid-Market (€75-150): 15% MAPE
- Traditional hotels with predictable pricing
- Model captures hotel characteristics well
- Lead time and seasonality patterns apply

### Luxury Hotels (€150-250): 36% MAPE
- Heavy dynamic pricing based on demand
- Special packages, VIP rates
- Event-driven pricing spikes
- More revenue management sophistication

## Reproducible Analysis

Run the following to verify these findings:

```bash
cd /Users/jon/GitHub/amenitiz-takehome
poetry run python ml_pipeline/analyze_mape_by_segment.py
```

## Implications

1. **Model is production-ready for €75-150 segment** (15% MAPE)
2. **Budget/Luxury segments need separate handling** or should show lower confidence
3. **Real-time demand signals** would be needed to improve luxury segment predictions

## Data Evidence

See `analyze_mape_by_segment.py` for full reproducible analysis.

