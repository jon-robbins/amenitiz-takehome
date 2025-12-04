"""
Main entry point for the Price Recommendation ML pipeline.

Usage:
    python ml_pipeline/main.py train      # Train the model
    python ml_pipeline/main.py evaluate   # Evaluate the model
    python ml_pipeline/main.py recommend  # Run demo recommendations
    python ml_pipeline/main.py --help     # Show help
"""

import sys
import argparse
from pathlib import Path


def train_model():
    """Train the price recommendation model."""
    from ml_pipeline.train import main as train_main
    train_main()


def evaluate_model():
    """Evaluate the model on holdout data and matched pairs."""
    from ml_pipeline.evaluate import run_full_evaluation
    results = run_full_evaluation()
    return results


def run_demo():
    """Run demo recommendations for sample hotels."""
    from ml_pipeline.predict import recommend_price
    from lib.db import init_db
    
    print("=" * 80)
    print("PRICE RECOMMENDATION - DEMO")
    print("=" * 80)
    
    try:
        con = init_db()
        
        # Get sample hotels
        sample = con.execute("""
            SELECT DISTINCT b.hotel_id, hl.city 
            FROM bookings b 
            JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
            WHERE hl.city IS NOT NULL
            LIMIT 3
        """).fetchdf()
        
        date = '2024-12-15'
        
        for _, row in sample.iterrows():
            hotel_id = row['hotel_id']
            city = row['city']
            
            print(f"\n{'='*60}")
            print(f"Hotel {hotel_id} ({city})")
            print(f"Date: {date}")
            print(f"{'='*60}")
            
            for strategy in ['conservative', 'safe', 'optimal']:
                try:
                    result = recommend_price(hotel_id, date, strategy)
                    print(f"\n{strategy.upper()} Strategy:")
                    print(f"  Peer Price: €{result['peer_price']:.2f}")
                    print(f"  Recommended: €{result['recommended_price']:.2f} (+{result['price_deviation_pct']:.0f}%)")
                    print(f"  Expected RevPAR Lift: +{result['expected_revpar_lift_pct']:.1f}%")
                    print(f"  Confidence: {result['confidence']}")
                    print(f"  Segment: {result['market_segment']}")
                except Exception as e:
                    print(f"\n{strategy.upper()}: Error - {e}")
                    
    except FileNotFoundError:
        print("\n⚠️ Model not trained yet. Run: python ml_pipeline/main.py train")
    except Exception as e:
        print(f"\nError: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Price Recommendation ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_pipeline/main.py train      # Train the model
  python ml_pipeline/main.py evaluate   # Evaluate performance
  python ml_pipeline/main.py recommend  # Run demo recommendations
        """
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'recommend'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model()
    elif args.command == 'evaluate':
        evaluate_model()
    elif args.command == 'recommend':
        run_demo()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: show help
        print(__doc__)
        print("\nRun with --help for usage information.")
    else:
        main()
