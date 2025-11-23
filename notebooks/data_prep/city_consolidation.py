"""
City name consolidation using TF-IDF and cosine similarity.

This module provides functions to match and consolidate city name variations.
Includes a demo script that shows before/after comparison.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lib.db import init_db  # noqa: E402
from lib.data_validator import validate_and_clean  # noqa: E402


def clean_city_name(city: str) -> str:
    """
    Basic cleaning of city names.
    Removes extra whitespace, converts to lowercase for matching.
    """
    if pd.isna(city) or city == "":
        return ""
    # Remove extra whitespace and convert to lowercase
    cleaned = re.sub(r"\s+", " ", str(city).strip().lower())
    return cleaned


def extract_major_city_from_compound(city: str) -> str | None:
    """
    Extract major city from compound names like "San Pere de Ribes, Barcelona".
    Returns the part after the comma if it exists.
    """
    if pd.isna(city) or city == "":
        return None
    
    # Check for comma-separated format
    if "," in city:
        parts = city.split(",")
        # Return the last part (usually the major city)
        return parts[-1].strip()
    
    return None


def find_canonical_cities(
    df: pd.DataFrame,
    city_col: str = "city",
    min_bookings: int = 1000,
) -> list[str]:
    """
    Identify canonical cities based on booking volume.
    These are the "major" cities we want to map variations to.
    """
    city_counts = df[city_col].value_counts()
    canonical = city_counts[city_counts >= min_bookings].index.tolist()
    return canonical


def build_tfidf_matcher(
    canonical_cities: list[str],
    ngram_range: tuple[int, int] = (1, 3),
) -> tuple[TfidfVectorizer, np.ndarray]:
    """
    Build TF-IDF vectorizer and matrix for canonical cities.
    Uses character n-grams to handle typos and variations.
    """
    # Clean canonical city names
    cleaned_canonical = [clean_city_name(city) for city in canonical_cities]
    
    # Create TF-IDF vectorizer with character n-grams
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
    )
    
    # Fit and transform canonical cities
    tfidf_matrix = vectorizer.fit_transform(cleaned_canonical)
    
    return vectorizer, tfidf_matrix


def match_city_to_canonical(
    city: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    canonical_cities: list[str],
    threshold: float = 0.6,
) -> tuple[str | None, float]:
    """
    Match a city name to the most similar canonical city.
    
    Returns:
        (matched_city, similarity_score) or (None, 0.0) if no match above threshold
    """
    if pd.isna(city) or city == "":
        return None, 0.0
    
    # First, check if there's a compound name with comma
    extracted = extract_major_city_from_compound(city)
    if extracted:
        # Check if extracted city is in canonical list (case-insensitive)
        for canonical in canonical_cities:
            if clean_city_name(extracted) == clean_city_name(canonical):
                return canonical, 1.0  # Perfect match
    
    # Clean and vectorize the input city
    cleaned = clean_city_name(city)
    city_vector = vectorizer.transform([cleaned])
    
    # Compute cosine similarity with all canonical cities
    similarities = cosine_similarity(city_vector, tfidf_matrix).flatten()
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    if best_score >= threshold:
        return canonical_cities[best_idx], float(best_score)
    
    return None, float(best_score)


def create_city_mapping(
    df: pd.DataFrame,
    city_col: str = "city",
    min_bookings_canonical: int = 1000,
    similarity_threshold: float = 0.6,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Create a mapping from all city names to canonical city names.
    
    Args:
        df: DataFrame with city column
        city_col: Name of city column
        min_bookings_canonical: Minimum bookings to be considered a canonical city
        similarity_threshold: Minimum cosine similarity to match (0-1)
        verbose: Print mapping statistics
    
    Returns:
        Dictionary mapping original city names to canonical names
    """
    # Find canonical cities (high booking volume)
    canonical_cities = find_canonical_cities(
        df, city_col=city_col, min_bookings=min_bookings_canonical
    )
    
    if verbose:
        print(f"Found {len(canonical_cities)} canonical cities (>={min_bookings_canonical} bookings)")
        print(f"Top 10: {canonical_cities[:10]}")
    
    # Build TF-IDF matcher
    vectorizer, tfidf_matrix = build_tfidf_matcher(canonical_cities)
    
    # Get all unique city names
    unique_cities = df[city_col].dropna().unique()
    
    if verbose:
        print(f"\nMatching {len(unique_cities)} unique city names to canonical cities...")
    
    # Create mapping
    city_mapping = {}
    matched_count = 0
    
    for city in unique_cities:
        matched_city, score = match_city_to_canonical(
            city, vectorizer, tfidf_matrix, canonical_cities, similarity_threshold
        )
        
        if matched_city:
            city_mapping[city] = matched_city
            matched_count += 1
        else:
            # Keep original if no match
            city_mapping[city] = city
    
    if verbose:
        print(f"\nMatched {matched_count}/{len(unique_cities)} cities ({matched_count/len(unique_cities)*100:.1f}%)")
        print(f"Reduced from {len(unique_cities)} to {len(set(city_mapping.values()))} unique cities")
    
    return city_mapping


def apply_city_mapping(
    df: pd.DataFrame,
    city_mapping: dict[str, str],
    city_col: str = "city",
    new_col: str = "city_canonical",
) -> pd.DataFrame:
    """
    Apply city mapping to create a new canonical city column.
    """
    df = df.copy()
    df[new_col] = df[city_col].map(city_mapping)
    # Fill any unmapped values with original
    df[new_col] = df[new_col].fillna(df[city_col])
    return df


def print_mapping_examples(
    city_mapping: dict[str, str],
    n_examples: int = 20,
) -> None:
    """Print examples of city name mappings."""
    # Find mappings where original != canonical (actual changes)
    changes = {k: v for k, v in city_mapping.items() if k != v}
    
    if not changes:
        print("No city name changes found.")
        return
    
    print(f"\n--- Example Mappings (showing {min(n_examples, len(changes))} of {len(changes)} changes) ---")
    for i, (original, canonical) in enumerate(list(changes.items())[:n_examples]):
        print(f"  {original:40s} → {canonical}")


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
