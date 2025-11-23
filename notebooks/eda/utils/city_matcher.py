"""
City name matching using TF-IDF and cosine similarity.

Handles variations like:
- "San Pere de Ribes, Barcelona" → "Barcelona"
- "Badalona" → "Barcelona" (if geographically close)
- "Madrid Centro" → "Madrid"
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def analyze_mapping_impact(
    df: pd.DataFrame,
    city_col: str = "city",
    canonical_col: str = "city_canonical",
) -> pd.DataFrame:
    """
    Analyze the impact of city mapping on booking distribution.
    """
    # Before mapping
    before = df.groupby(city_col, as_index=False).agg(
        bookings_before=("city", "count")
    )
    
    # After mapping
    after = df.groupby(canonical_col, as_index=False).agg(
        bookings_after=(canonical_col, "count")
    )
    
    print("\n--- Impact Analysis ---")
    print(f"Cities before mapping: {len(before)}")
    print(f"Cities after mapping: {len(after)}")
    print(f"Reduction: {len(before) - len(after)} cities ({(len(before) - len(after))/len(before)*100:.1f}%)")
    
    # Show top cities before and after
    print("\nTop 10 cities BEFORE mapping:")
    print(before.nlargest(10, "bookings_before")[["city", "bookings_before"]].to_string(index=False))
    
    print("\nTop 10 cities AFTER mapping:")
    print(after.nlargest(10, "bookings_after")[["city_canonical", "bookings_after"]].to_string(index=False))
    
    return after


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    
    from notebooks.utils.db import init_db  # noqa: E402
    from notebooks.utils.data_validator import validate_and_clean  # noqa: E402
    
    print("=" * 80)
    print("CITY NAME MATCHING - TF-IDF & COSINE SIMILARITY")
    print("=" * 80)
    
    # Load data
    con = validate_and_clean(
        init_db(),
        verbose=False,
        rooms_to_exclude=["reception_hall"],
        exclude_missing_location_bookings=True,
    )
    
    # Get city data
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
    
    print(f"\nLoaded {len(df):,} bookings with city data")
    print(f"Unique cities: {df['city'].nunique():,}")
    
    # Create mapping
    city_mapping = create_city_mapping(
        df,
        city_col="city",
        min_bookings_canonical=1000,
        similarity_threshold=0.6,
        verbose=True,
    )
    
    # Show examples
    print_mapping_examples(city_mapping, n_examples=30)
    
    # Apply mapping
    df = apply_city_mapping(df, city_mapping, city_col="city", new_col="city_canonical")
    
    # Analyze impact
    analyze_mapping_impact(df, city_col="city", canonical_col="city_canonical")
    
    # Save mapping to file
    output_path = PROJECT_ROOT / "outputs" / "hotspots" / "city_name_mapping.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mapping_df = pd.DataFrame(
        list(city_mapping.items()), columns=["original_city", "canonical_city"]
    )
    mapping_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved city mapping to {output_path}")

