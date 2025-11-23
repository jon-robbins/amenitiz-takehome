"""
City name consolidation script.

Three-step approach:
1. Identify inconsistently named cities using TF-IDF + cosine similarity (>=0.95)
2. Match to the city with most bookings
3. For cities not in top 100, validate nearest city using lat/long
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner


def normalize_city_name(city: str) -> str:
    """Normalize city name for comparison."""
    if pd.isna(city) or city == "":
        return ""
    return re.sub(r"\s+", " ", str(city).strip().lower())


def get_region(lat: float, lon: float) -> str:
    """Determine geographic region from coordinates."""
    # Balearic Islands: lat 38.5-40.5, lon 1-4.5
    if 38.5 <= lat <= 40.5 and 1.0 <= lon <= 4.5:
        return "balearic"
    # Canary Islands: lat 27-30, lon -18 to -13
    elif 27.0 <= lat <= 30.0 and -18.0 <= lon <= -13.0:
        return "canary"
    # Mainland Spain
    else:
        return "mainland"


def consolidate_cities(con=None, verbose: bool = True) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Consolidate city names using three-step approach.
    
    Args:
        con: DuckDB connection (if None, creates new connection with cleaning)
        verbose: Print progress messages
    
    Returns:
        Tuple of (city_mapping, city_stats_df)
        - city_mapping: dict mapping original city names to canonical names
        - city_stats_df: DataFrame with city statistics (booking counts, coordinates)
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    vprint("=" * 80)
    vprint("CITY CONSOLIDATION - THREE-STEP APPROACH")
    vprint("=" * 80)

    # Load and clean data
    vprint("\n1. Loading and cleaning data...")
    
    if con is None:
        config = CleaningConfig(
            exclude_missing_location=True,
            verbose=False
        )
        cleaner = DataCleaner(config)
        con = cleaner.clean(init_db())
    else:
        # Wrap if needed
        from lib.data_validator import DuckDBConnectionWrapper
        if not isinstance(con, DuckDBConnectionWrapper):
            con = DuckDBConnectionWrapper(con)

    # Get all unique cities with booking counts
    vprint("\n2. Getting city statistics...")
    city_stats = con.execute("""
        SELECT 
            hl.city,
            COUNT(b.id) as booking_count,
            COUNT(DISTINCT hl.hotel_id) as hotel_count,
            AVG(hl.latitude) as avg_lat,
            AVG(hl.longitude) as avg_lon
        FROM hotel_location hl
        INNER JOIN bookings b ON hl.hotel_id = b.hotel_id
        WHERE hl.country = 'ES' 
          AND hl.city IS NOT NULL 
          AND hl.city != ''
          AND hl.latitude IS NOT NULL
          AND hl.longitude IS NOT NULL
        GROUP BY hl.city
        ORDER BY booking_count DESC
    """).fetchdf()

    vprint(f"  Total unique cities: {len(city_stats):,}")
    vprint(f"  Total bookings: {city_stats['booking_count'].sum():,}")

    # Get top 100 cities by booking count (canonical cities)
    top_100 = city_stats.head(100).copy()
    vprint(f"\n  Top 100 cities account for {top_100['booking_count'].sum():,} bookings")

    # Step 1: Identify inconsistently named cities using TF-IDF + cosine similarity
    vprint("\n" + "=" * 80)
    vprint("STEP 1: Identify Inconsistently Named Cities (TF-IDF + Cosine Similarity >= 0.95)")
    vprint("=" * 80)

    # Get all city names
    all_cities = city_stats['city'].tolist()
    all_cities_normalized = [normalize_city_name(city) for city in all_cities]

    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        lowercase=True,
        strip_accents="unicode"
    )
    tfidf_matrix = vectorizer.fit_transform(all_cities_normalized)

    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)  # Set self-similarity to 0

    # Find similar city groups (>= 0.95 similarity)
    # Use union-find approach to group similar cities
    threshold = 0.95
    city_groups = {}  # city_index -> set of similar city indices
    
    for i in range(len(all_cities)):
        # Find all cities similar to city i
        similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
        
        if len(similar_indices) > 0:
            # Create a group with all similar cities
            group = set([i] + list(similar_indices))
            
            # Merge with existing groups if there's overlap
            merged_group = group.copy()
            for idx in group:
                if idx in city_groups:
                    merged_group.update(city_groups[idx])
            
            # Update all cities in the merged group
            for idx in merged_group:
                city_groups[idx] = merged_group
    
    # For each group, find the city with most bookings (canonical)
    city_mapping_step1 = {}
    processed_groups = set()
    
    for group in city_groups.values():
        # Convert to tuple for set hashing
        group_tuple = tuple(sorted(group))
        if group_tuple in processed_groups:
            continue
        processed_groups.add(group_tuple)
        
        # Get all cities in this group with their booking counts
        group_cities = [
            {
                'city': all_cities[idx],
                'booking_count': city_stats.iloc[idx]['booking_count'],
                'index': idx
            }
            for idx in group
        ]
        
        # Sort by booking count (descending)
        group_cities.sort(key=lambda x: x['booking_count'], reverse=True)
        
        # The city with most bookings is the canonical one
        canonical = group_cities[0]
        
        # All other cities in the group map to the canonical one
        for city_info in group_cities[1:]:
            if city_info['city'] != canonical['city']:
                # Get similarity between this city and canonical
                sim = similarity_matrix[city_info['index']][canonical['index']]
                # If similarity is 0 (diagonal was set to 0), it means they're the same normalized name
                if sim == 0:
                    sim = 1.0
                city_mapping_step1[city_info['city']] = {
                    'canonical': canonical['city'],
                    'similarity': sim,
                    'original_bookings': city_info['booking_count'],
                    'canonical_bookings': canonical['booking_count']
                }

    vprint(f"\n  Found {len(city_mapping_step1):,} cities with similar names (>= {threshold} similarity)")
    vprint(f"\n  Example mappings:")
    for i, (original, info) in enumerate(list(city_mapping_step1.items())[:10]):
        print(f"    {original:40s} → {info['canonical']:40s} (sim: {info['similarity']:.3f}, "
              f"{info['original_bookings']:,} → {info['canonical_bookings']:,} bookings)")

    # Step 2: For cities in top 100, use direct match or Step 1 mapping
    vprint("\n" + "=" * 80)
    vprint("STEP 2: Direct Match for Top 100 Cities")
    vprint("=" * 80)

    # Create mapping: normalized name -> canonical name (from top 100)
    # But first, apply Step 1 mappings to resolve conflicts within top 100
    top_100_resolved = {}
    for city in top_100['city']:
        # Check if Step 1 found a mapping for this city
        if city in city_mapping_step1:
            canonical = city_mapping_step1[city]['canonical']
            # Only use if canonical is also in top 100 and has more bookings
            if canonical in top_100['city'].values:
                canonical_bookings = city_stats[city_stats['city'] == canonical]['booking_count'].iloc[0]
                city_bookings = city_stats[city_stats['city'] == city]['booking_count'].iloc[0]
                if canonical_bookings >= city_bookings:
                    top_100_resolved[city] = canonical
                else:
                    top_100_resolved[city] = city
            else:
                top_100_resolved[city] = city
        else:
            top_100_resolved[city] = city
    
    # Create normalized mapping for top 100 (using resolved names)
    top_100_normalized = {normalize_city_name(city): top_100_resolved[city] for city in top_100['city']}
    
    # Apply Step 1 mappings first, then check top 100
    final_mapping = {}
    for city in city_stats['city']:
        city_norm = normalize_city_name(city)
        
        # First, check if Step 1 found a similar city
        if city in city_mapping_step1:
            canonical = city_mapping_step1[city]['canonical']
            # Check if canonical is in top 100
            if normalize_city_name(canonical) in top_100_normalized:
                final_mapping[city] = top_100_normalized[normalize_city_name(canonical)]
            else:
                # Canonical not in top 100, keep original for Step 3
                final_mapping[city] = city
        # Check if it's in top 100 (direct match)
        elif city_norm in top_100_normalized:
            final_mapping[city] = top_100_normalized[city_norm]
        else:
            final_mapping[city] = city  # Keep original for Step 3

    cities_in_top100 = sum(1 for v in final_mapping.values() if normalize_city_name(v) in top_100_normalized)
    vprint(f"\n  Cities mapped to top 100: {cities_in_top100:,} / {len(final_mapping):,}")

    # Step 3: For cities not in top 100, validate nearest city using lat/long
    vprint("\n" + "=" * 80)
    vprint("STEP 3: Lat/Long Validation for Cities Not in Top 100")
    vprint("=" * 80)

    # Get cities not yet mapped to top 100
    unmapped_cities = [
        city for city, canonical in final_mapping.items()
        if normalize_city_name(canonical) not in top_100_normalized
    ]

    vprint(f"  Cities not in top 100: {len(unmapped_cities):,}")

    if len(unmapped_cities) > 0:
        # Get coordinates for top 100 cities
        top_100_coords = top_100.dropna(subset=["avg_lat", "avg_lon"]).copy()
        
        # Validate coordinates are within Spain's bounds
        top_100_coords = top_100_coords[
            (top_100_coords["avg_lat"] >= 35) & (top_100_coords["avg_lat"] <= 45) &
            (top_100_coords["avg_lon"] >= -11) & (top_100_coords["avg_lon"] <= 5)
        ].copy()

        # Build BallTree
        canonical_lat_lon = np.radians(
            np.c_[top_100_coords["avg_lat"].to_numpy(),
                  top_100_coords["avg_lon"].to_numpy()]
        )
        tree = BallTree(canonical_lat_lon, metric="haversine")

        # For each unmapped city, find nearest top 100 city
        max_distance_km = 50
        lat_long_matches = 0
        too_far_count = 0
        wrong_region_count = 0

        for city in unmapped_cities:
            city_data = city_stats[city_stats['city'] == city].iloc[0]
            city_lat = city_data['avg_lat']
            city_lon = city_data['avg_lon']
            
            # Validate coordinates
            if not (35 <= city_lat <= 45 and -11 <= city_lon <= 5):
                continue
            
            # Find nearest canonical city
            city_coord = np.radians([[city_lat, city_lon]])
            dist, idx = tree.query(city_coord, k=1)
            dist_km = dist[0][0] * 6371  # Convert to km
            
            matched_city_original = top_100_coords.iloc[idx[0][0]]['city']
            matched_lat = top_100_coords.iloc[idx[0][0]]['avg_lat']
            matched_lon = top_100_coords.iloc[idx[0][0]]['avg_lon']
            
            # Map to resolved canonical name (from Step 1/2)
            matched_city = top_100_resolved.get(matched_city_original, matched_city_original)
            
            # Check region match
            city_region = get_region(city_lat, city_lon)
            matched_region = get_region(matched_lat, matched_lon)
            same_region = city_region == matched_region
            
            # Only match if same region and within distance
            if same_region and dist_km <= max_distance_km:
                final_mapping[city] = matched_city
                lat_long_matches += 1
            elif not same_region:
                wrong_region_count += 1
            else:
                too_far_count += 1

        print(f"\n  Lat/long matches (same region, <{max_distance_km} km): {lat_long_matches:,}")
        print(f"  Rejected - different region: {wrong_region_count:,}")
        print(f"  Rejected - too far (>{max_distance_km} km): {too_far_count:,}")

    # Summary
    vprint("\n" + "=" * 80)
    vprint("SUMMARY")
    vprint("=" * 80)

    # Count unique canonical cities
    unique_canonical = len(set(final_mapping.values()))
    vprint(f"\n  Original cities: {len(final_mapping):,}")
    vprint(f"  Canonical cities: {unique_canonical:,}")
    vprint(f"  Reduction: {len(final_mapping) - unique_canonical:,} ({(len(final_mapping) - unique_canonical) / len(final_mapping) * 100:.1f}%)")

    # Show mapping statistics
    step1_count = sum(1 for city in final_mapping.keys() if city in city_mapping_step1)
    direct_top100 = sum(1 for city, canonical in final_mapping.items() 
                       if normalize_city_name(city) in top_100_normalized and city == canonical)
    lat_long_mapped = sum(1 for city in unmapped_cities 
                          if normalize_city_name(final_mapping[city]) in top_100_normalized)

    vprint(f"\n  Mapping breakdown:")
    vprint(f"    - Direct match to top 100: {direct_top100:,}")
    vprint(f"    - Step 1 (similarity >= 0.95): {step1_count:,}")
    vprint(f"    - Step 3 (lat/long validation): {lat_long_matches:,}")
    vprint(f"    - Kept original (not in top 100): {len(final_mapping) - direct_top100 - step1_count - lat_long_matches:,}")

    # Show examples
    vprint(f"\n  Example final mappings:")
    examples = [
        (city, canonical) for city, canonical in final_mapping.items()
        if city != canonical
    ][:20]
    for original, canonical in examples:
        orig_bookings = city_stats[city_stats['city'] == original]['booking_count'].iloc[0]
        canon_bookings = city_stats[city_stats['city'] == canonical]['booking_count'].iloc[0] if canonical in city_stats['city'].values else 0
        print(f"    {original:40s} → {canonical:40s} ({orig_bookings:,} → {canon_bookings:,} bookings)")

    # Save mapping
    output_dir = PROJECT_ROOT / "outputs" / "city_consolidation"
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_df = pd.DataFrame([
        {
            'original_city': city,
            'canonical_city': canonical,
            'changed': city != canonical,
            'original_bookings': city_stats[city_stats['city'] == city]['booking_count'].iloc[0],
            'canonical_bookings': city_stats[city_stats['city'] == canonical]['booking_count'].iloc[0] if canonical in city_stats['city'].values else 0
        }
        for city, canonical in final_mapping.items()
    ])
    mapping_df.to_csv(output_dir / "city_mapping_v2.csv", index=False)
    vprint(f"\n✓ Saved mapping to: {output_dir / 'city_mapping_v2.csv'}")
    
    # Return mapping and stats
    return final_mapping, city_stats


def main():
    """CLI entry point."""
    consolidate_cities(verbose=True)


if __name__ == "__main__":
    main()

