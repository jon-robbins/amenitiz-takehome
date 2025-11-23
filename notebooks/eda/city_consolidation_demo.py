"""
Demonstration of TF-IDF city name consolidation.

Shows before/after comparison of city name matching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebooks.utils.db import init_db  # noqa: E402
from notebooks.utils.data_validator import validate_and_clean  # noqa: E402
from notebooks.eda.utils.city_matcher import (  # noqa: E402
    create_city_mapping,
    apply_city_mapping,
    print_mapping_examples,
)


def main() -> None:
    """Run city consolidation demo."""
    print("=" * 80)
    print("CITY NAME CONSOLIDATION - TF-IDF & COSINE SIMILARITY")
    print("=" * 80)

    # Load data
    print("\nLoading booking data...")
    con = validate_and_clean(
        init_db(),
        verbose=False,
        rooms_to_exclude=["reception_hall"],
        exclude_missing_location_bookings=True,
    )

    # Get city data with booking counts
    df = con.execute(
        """
        SELECT 
            hl.city,
            b.id as booking_id
        FROM bookings b
        JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
        WHERE b.status IN ('confirmed', 'Booked')
          AND hl.city IS NOT NULL
    """
    ).fetchdf()

    print(f"Loaded {len(df):,} bookings")
    print(f"Unique cities (before): {df['city'].nunique():,}")

    # Show top cities BEFORE consolidation
    print("\n" + "-" * 80)
    print("TOP 20 CITIES - BEFORE CONSOLIDATION")
    print("-" * 80)
    before_counts = df["city"].value_counts().head(20)
    for i, (city, count) in enumerate(before_counts.items(), 1):
        print(f"{i:2d}. {city:40s} {count:6,} bookings")

    # Create mapping
    print("\n" + "-" * 80)
    print("CREATING TF-IDF MAPPING")
    print("-" * 80)
    city_mapping = create_city_mapping(
        df,
        city_col="city",
        min_bookings_canonical=1000,
        similarity_threshold=0.6,
        verbose=True,
    )

    # Apply mapping
    df = apply_city_mapping(df, city_mapping, city_col="city", new_col="city_canonical")

    # Show top cities AFTER consolidation
    print("\n" + "-" * 80)
    print("TOP 20 CITIES - AFTER CONSOLIDATION")
    print("-" * 80)
    after_counts = df["city_canonical"].value_counts().head(20)
    for i, (city, count) in enumerate(after_counts.items(), 1):
        change = ""
        if city in before_counts.index:
            diff = count - before_counts[city]
            if diff > 0:
                change = f" (+{diff:,})"
        else:
            change = f" (NEW)"
        print(f"{i:2d}. {city:40s} {count:6,} bookings{change}")

    # Show interesting examples
    print("\n" + "-" * 80)
    print("EXAMPLE CONSOLIDATIONS")
    print("-" * 80)
    print_mapping_examples(city_mapping, n_examples=40)

    # Show impact by category
    print("\n" + "-" * 80)
    print("CONSOLIDATION IMPACT BY CATEGORY")
    print("-" * 80)

    changes = {k: v for k, v in city_mapping.items() if k != v}

    # Compound names (with comma)
    compound = {k: v for k, v in changes.items() if "," in k}
    print(f"\nCompound names (e.g., 'Suburb, City'): {len(compound)}")
    print("Examples:")
    for orig, canon in list(compound.items())[:5]:
        print(f"  • {orig} → {canon}")

    # Case normalization
    case_only = {
        k: v
        for k, v in changes.items()
        if k.lower() == v.lower() and k != v
    }
    print(f"\nCase normalization (e.g., 'MADRID' → 'Madrid'): {len(case_only)}")
    print("Examples:")
    for orig, canon in list(case_only.items())[:5]:
        print(f"  • {orig} → {canon}")

    # Fuzzy matches
    fuzzy = {
        k: v
        for k, v in changes.items()
        if "," not in k and k.lower() != v.lower()
    }
    print(f"\nFuzzy matches (similar names): {len(fuzzy)}")
    print("Examples:")
    for orig, canon in list(fuzzy.items())[:5]:
        print(f"  • {orig} → {canon}")

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "hotspots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mapping
    mapping_df = pd.DataFrame(
        list(city_mapping.items()), columns=["original_city", "canonical_city"]
    )
    mapping_df["changed"] = mapping_df["original_city"] != mapping_df["canonical_city"]
    mapping_df.to_csv(output_dir / "city_name_mapping.csv", index=False)

    # Save before/after comparison
    comparison = pd.DataFrame(
        {
            "city_before": before_counts.index,
            "bookings_before": before_counts.values,
        }
    )
    comparison["city_after"] = comparison["city_before"].map(city_mapping)
    comparison["bookings_after"] = comparison["city_after"].map(
        after_counts.to_dict()
    )
    comparison.to_csv(output_dir / "city_consolidation_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Cities before: {df['city'].nunique():,}")
    print(f"Cities after:  {df['city_canonical'].nunique():,}")
    print(f"Reduction:     {df['city'].nunique() - df['city_canonical'].nunique():,} ({(df['city'].nunique() - df['city_canonical'].nunique()) / df['city'].nunique() * 100:.1f}%)")
    print(f"\nTop city before: {before_counts.index[0]} ({before_counts.iloc[0]:,} bookings)")
    print(f"Top city after:  {after_counts.index[0]} ({after_counts.iloc[0]:,} bookings)")
    print(f"Gain:            +{after_counts.iloc[0] - before_counts.iloc[0]:,} bookings")
    print("\n✓ Saved outputs to:", output_dir)
    print("  - city_name_mapping.csv")
    print("  - city_consolidation_comparison.csv")


if __name__ == "__main__":
    main()

