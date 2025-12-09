#!/usr/bin/env python
"""
Train the occupancy prediction model.

Usage:
    python entrypoint/train.py
    python entrypoint/train.py --quick  # Fast training with sample
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.data.loader import load_hotel_month_data, get_clean_connection
from src.features.engineering import engineer_features, standardize_city
from src.models.occupancy import OccupancyModel


def main():
    parser = argparse.ArgumentParser(description='Train occupancy model')
    parser.add_argument('--quick', action='store_true', help='Quick training with sample')
    parser.add_argument('--output', type=str, default='outputs/models/occupancy_model.pkl',
                        help='Output path for model')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRAINING OCCUPANCY MODEL")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    con = get_clean_connection()
    df = load_hotel_month_data(con)
    
    # Engineer features
    print("\n2. Engineering features...")
    df['city_standardized'] = df['city'].apply(standardize_city)
    df = engineer_features(df)
    
    if args.quick:
        df = df.sample(n=min(5000, len(df)), random_state=42)
        print(f"   Using sample of {len(df):,} records for quick training")
    
    # Train model
    print("\n3. Training model...")
    model = OccupancyModel()
    model.fit(df)
    
    # Save model
    print("\n4. Saving model...")
    output_path = Path(args.output)
    model.save(output_path)
    
    # Print summary
    metrics = model.get_metrics()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}")
    print(f"Performance:")
    print(f"  - MAE: {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
    print(f"  - R²: {metrics['r2']:.4f}")
    print(f"  - Train/Test: {metrics['n_train']:,}/{metrics['n_test']:,}")


if __name__ == "__main__":
    main()

