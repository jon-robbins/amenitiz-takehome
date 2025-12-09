#!/usr/bin/env python
"""
Generate price recommendations.

Usage:
    python entrypoint/recommend.py --hotel-id 123 --date 2025-01-15
    python entrypoint/recommend.py --validate  # Check distribution
    python entrypoint/recommend.py --demo      # Show examples
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.recommender.price_recommender import PriceRecommender


def main():
    parser = argparse.ArgumentParser(description='Generate price recommendations')
    parser.add_argument('--hotel-id', type=int, help='Hotel ID for recommendation')
    parser.add_argument('--date', type=str, default='2025-01-15', help='Target date')
    parser.add_argument('--validate', action='store_true', help='Validate recommendation distribution')
    parser.add_argument('--demo', action='store_true', help='Show demo recommendations')
    parser.add_argument('--quick', action='store_true', help='Quick fitting with sample data')
    args = parser.parse_args()
    
    # Fit recommender
    print("Fitting recommender...")
    recommender = PriceRecommender()
    recommender.fit(quick=args.quick)
    
    if args.validate:
        print("\n" + "=" * 70)
        print("VALIDATION: Recommendation Distribution")
        print("=" * 70)
        
        dist = recommender.get_recommendation_distribution(n_samples=200)
        
        print(f"\nSampled {dist['n_samples']} hotels:")
        print(f"  INCREASE: {dist['pct_increase']:.1f}% (avg {dist.get('avg_increase_pct', 0):+.1f}%)")
        print(f"  DECREASE: {dist['pct_decrease']:.1f}% (avg {dist.get('avg_decrease_pct', 0):+.1f}%)")
        print(f"  MAINTAIN: {dist['pct_maintain']:.1f}%")
        
        # Check if distribution is sensible
        if dist['pct_increase'] > 80:
            print("\n⚠️ WARNING: Too many increases - check diagnosis logic")
        elif dist['pct_decrease'] < 10:
            print("\n⚠️ WARNING: Very few decreases - may be missing overpriced hotels")
        else:
            print("\n✓ Distribution looks reasonable")
    
    elif args.demo:
        print("\n" + "=" * 70)
        print("DEMO: Sample Recommendations")
        print("=" * 70)
        
        hotel_ids = recommender.hotel_data['hotel_id'].unique()[:5]
        
        for hotel_id in hotel_ids:
            rec = recommender.recommend_price(int(hotel_id), args.date)
            print(f"\n{'─' * 50}")
            print(f"Hotel {hotel_id}")
            print(f"{'─' * 50}")
            print(f"Current: €{rec.current_price:.0f} | Peer: €{rec.peer_price:.0f} ({rec.price_premium_pct:+.0f}%)")
            print(f"Occupancy: {rec.actual_occupancy:.0%} actual, {rec.expected_occupancy:.0%} expected ({rec.occ_residual:+.0%})")
            print(f"\n→ {rec.direction.upper()}: €{rec.recommended_price:.0f} ({rec.change_pct:+.1f}%)")
            print(f"  {rec.reasoning}")
    
    elif args.hotel_id:
        print("\n" + "=" * 70)
        print(f"RECOMMENDATION: Hotel {args.hotel_id}")
        print("=" * 70)
        
        rec = recommender.recommend_price(args.hotel_id, args.date)
        
        print(f"\nDate: {rec.date}")
        print(f"Segment: {rec.market_segment}")
        print(f"\nCurrent State:")
        print(f"  Price: €{rec.current_price:.2f}")
        print(f"  Peer Avg: €{rec.peer_price:.2f} ({rec.price_premium_pct:+.1f}% premium)")
        print(f"  Occupancy: {rec.actual_occupancy:.1%} actual, {rec.expected_occupancy:.1%} expected")
        print(f"\nDiagnosis: {rec.direction.upper()}")
        print(f"  {rec.reasoning}")
        print(f"\nRecommendation:")
        print(f"  New Price: €{rec.recommended_price:.2f} ({rec.change_pct:+.1f}%)")
        print(f"  Confidence: {rec.confidence} ({rec.n_peers} peers)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

