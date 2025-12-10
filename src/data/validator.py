"""
Data validation and cleaning using a unified Rule-based architecture.

All data quality operations (deletions, exclusions, imputations, fixes) use the same
Rule format with check_query and action_query. Some advanced cleaning (e.g., city name merging) is handled by custom methods.
"""

import duckdb
import logging
import numpy as np
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# For TF-IDF and cosine similarity
from typing import Optional, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 1. RULE DATACLASS (Unified Format)
# ============================================================================

@dataclass
class Rule:
    """
    Single data quality rule - works for deletions, exclusions, imputations, fixes.
    
    All operations follow the same pattern:
    1. Check query: How many rows are affected?
    2. Action query: Fix the issue
    """
    name: str
    check_query: str
    action_query: str
    enabled: bool = True

# ============================================================================
# 2. CLEANING CONFIG (Self-Documenting Configuration)
# ============================================================================

@dataclass
class CleaningConfig:
    """
    Configuration for data cleaning pipeline.
    
    Each field enables/disables a specific rule or set of rules.
    The config itself IS the documentation - field names describe what they do.
    """
    # Basic validation rules (always recommended)
    remove_negative_prices: bool = True
    remove_zero_prices: bool = True
    remove_low_prices: bool = True  # <€5/night
    remove_null_prices: bool = True
    remove_extreme_prices: bool = True  # >€5000/night
    remove_price_outliers: bool = True  # Remove top 2% and bottom 2% of prices
    remove_null_dates: bool = True
    remove_null_created_at: bool = True
    remove_negative_stay: bool = True
    remove_negative_lead_time: bool = True
    remove_null_occupancy: bool = True
    remove_overcrowded_rooms: bool = True
    remove_null_room_id: bool = True
    remove_null_booking_id: bool = True
    remove_null_hotel_id: bool = True
    remove_orphan_bookings: bool = True
    remove_null_status: bool = True
    remove_cancelled_but_active: bool = True
    remove_bookings_before_2023: bool = True
    remove_bookings_after_2024: bool = True
    
    # Exclusions (found during EDA)
    exclude_reception_halls: bool = False      # Section 2.2: Not accommodation
    exclude_missing_location: bool = False     # Section 3.1: Can't analyze location
    exclude_non_spain_hotels: bool = True      # Exclude hotels outside Spain bounding box
    
    # Data quality fixes
    fix_empty_strings: bool = True             # Convert '' to NULL
    
    # Imputations (found during EDA)
    impute_children_allowed: bool = True       # Section 2.2: From booking behavior
    impute_events_allowed: bool = True         # Section 2.2: From room_type

    # City name merging using tf-idf/cosine similarity
    match_city_names_with_tfidf: bool = False
    city_name_similarity_threshold: float = 0.97

    #if booked_rooms.room_view is empty, set it to NULL
    set_empty_room_view_to_no_view_str: bool = True
    
    # City name cleaning
    clean_suspicious_city_names: bool = True  # Remove fake/test city names
    
    # Impute missing coordinates from cities500.json
    impute_coordinates_from_cities500: bool = True
    
    # Logging
    verbose: bool = False

# ============================================================================
# 3. DATA CLEANER CLASS (Applies Rules)
# ============================================================================

class DataCleaner:
    """
    Applies data cleaning rules based on configuration.
    
    Usage:
        config = CleaningConfig(
            exclude_reception_halls=True,
            exclude_missing_location=True,
            match_city_names_with_tfidf=True,
            city_name_similarity_threshold=0.97,
            verbose=True
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(init_db())
    """
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.rules = self._build_rules()
        self.stats = {}
    
    def _build_rules(self) -> list[Rule]:
        """Build list of rules based on config."""
        rules = []
        # (Rules unchanged from previous implementation)
        # ===== BASIC VALIDATION RULES =====
        if self.config.remove_negative_prices:
            rules.append(Rule(
                "Negative Price",
                "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0",
                "DELETE FROM booked_rooms WHERE total_price < 0"
            ))
        
        if self.config.remove_zero_prices:
            rules.append(Rule(
                "Zero Price",
         "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0",
                "DELETE FROM booked_rooms WHERE total_price = 0"
            ))
        
        if self.config.remove_low_prices:
            rules.append(Rule(
                "Low Price (<€5/night)",
                "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 5",
                "DELETE FROM booked_rooms WHERE total_price < 5"
            ))
        
        if self.config.remove_null_prices:
            rules.append(Rule(
                "NULL Price",
         "SELECT COUNT(*) FROM booked_rooms WHERE total_price IS NULL",
                "DELETE FROM booked_rooms WHERE total_price IS NULL"
            ))
        
        if self.config.remove_extreme_prices:
            rules.append(Rule(
                "Extreme Price (>5k/night)",
         """SELECT COUNT(*) FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            WHERE (b.status IN ('confirmed', 'Booked'))
              AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
              AND (br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) > 3000""",
         """DELETE FROM booked_rooms WHERE id IN (
                SELECT br.id FROM booked_rooms br
                JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
                WHERE (b.status IN ('confirmed', 'Booked'))
                  AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
                         AND (br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) > 5000)"""
            ))
        
        # Note: Price outlier removal (top/bottom 2%) is handled in clean() method
        # because it requires computing percentiles first
    
        if self.config.remove_null_dates:
            rules.append(Rule(
                "NULL Dates",
         "SELECT COUNT(*) FROM bookings WHERE arrival_date IS NULL OR departure_date IS NULL",
                "DELETE FROM bookings WHERE arrival_date IS NULL OR departure_date IS NULL"
            ))
    
        if self.config.remove_null_created_at:
            rules.append(Rule(
                "NULL Created At",
         "SELECT COUNT(*) FROM bookings WHERE created_at IS NULL",
                "DELETE FROM bookings WHERE created_at IS NULL"
            ))
    
        if self.config.remove_negative_stay:
            rules.append(Rule(
                "Negative Stay",
         "SELECT COUNT(*) FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)",
                "DELETE FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)"
            ))
    
        if self.config.remove_negative_lead_time:
            rules.append(Rule(
                "Negative Lead Time",
         "SELECT COUNT(*) FROM bookings WHERE CAST(created_at AS DATE) > CAST(arrival_date AS DATE)",
                "DELETE FROM bookings WHERE CAST(created_at AS DATE) > CAST(arrival_date AS DATE)"
            ))
    
        if self.config.remove_null_occupancy:
            rules.append(Rule(
                "NULL Occupancy",
         "SELECT COUNT(*) FROM booked_rooms WHERE total_adult IS NULL AND total_children IS NULL",
                "DELETE FROM booked_rooms WHERE total_adult IS NULL AND total_children IS NULL"
            ))
    
        if self.config.remove_overcrowded_rooms:
            rules.append(Rule(
                "Overcrowded Room",
         """SELECT COUNT(*) FROM booked_rooms br
            JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
            WHERE (COALESCE(br.total_adult, 0) + COALESCE(br.total_children, 0)) > r.max_occupancy""",
         """DELETE FROM booked_rooms WHERE id IN (
                SELECT br.id FROM booked_rooms br
                JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
                       WHERE (COALESCE(br.total_adult, 0) + COALESCE(br.total_children, 0)) > r.max_occupancy)"""
            ))
    
        if self.config.remove_null_room_id:
            rules.append(Rule(
                "NULL Room ID",
                "SELECT COUNT(*) FROM booked_rooms WHERE room_id IS NULL",
                "DELETE FROM booked_rooms WHERE room_id IS NULL"
            ))
    
        if self.config.remove_null_booking_id:
            rules.append(Rule(
                "NULL Booking ID",
                "SELECT COUNT(*) FROM booked_rooms WHERE booking_id IS NULL",
                "DELETE FROM booked_rooms WHERE booking_id IS NULL"
            ))
    
        if self.config.remove_null_hotel_id:
            rules.append(Rule(
                "NULL Hotel ID",
                "SELECT COUNT(*) FROM bookings WHERE hotel_id IS NULL",
                "DELETE FROM bookings WHERE hotel_id IS NULL"
            ))
    
        if self.config.remove_orphan_bookings:
            rules.append(Rule(
                "Orphan Bookings",
                """SELECT COUNT(*) FROM bookings b 
                   LEFT JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
                   WHERE br.booking_id IS NULL""",
                "DELETE FROM bookings WHERE id NOT IN (SELECT DISTINCT CAST(booking_id AS BIGINT) FROM booked_rooms)"
            ))
    
        if self.config.remove_null_status:
            rules.append(Rule(
                "NULL Status",
                "SELECT COUNT(*) FROM bookings WHERE status IS NULL",
                "DELETE FROM bookings WHERE status IS NULL"
            ))
    
        if self.config.remove_cancelled_but_active:
            rules.append(Rule(
                "Cancelled but Active",
         """SELECT COUNT(*) FROM bookings 
            WHERE status IN ('confirmed', 'Booked') 
            AND cancelled_date IS NOT NULL""",
         """DELETE FROM bookings 
            WHERE status IN ('confirmed', 'Booked') 
                   AND cancelled_date IS NOT NULL"""
            ))
        
        if self.config.remove_bookings_before_2023:
            rules.append(Rule(
                "Bookings Before 2023",
                "SELECT COUNT(*) FROM bookings WHERE CAST(created_at AS DATE) < '2023-01-01'",
                "DELETE FROM bookings WHERE CAST(created_at AS DATE) < '2023-01-01'"
            ))
        if self.config.remove_bookings_after_2024:
            rules.append(Rule(
                "Bookings After 2024",
                "SELECT COUNT(*) FROM bookings WHERE CAST(created_at AS DATE) > '2024-12-31' or cast(arrival_date as date) > '2024-12-31'",
                "DELETE FROM bookings WHERE CAST(created_at AS DATE) > '2024-12-31' or cast(arrival_date as date) > '2024-12-31'"
            ))
        # ===== EXCLUSIONS =====
        if self.config.exclude_reception_halls:
            rules.append(Rule(
                "Exclude Reception Halls",
                "SELECT COUNT(*) FROM booked_rooms WHERE room_type = 'reception_hall'",
                "DELETE FROM booked_rooms WHERE room_type = 'reception_hall'"
            ))
            # Clean up orphan bookings after excluding reception halls
            rules.append(Rule(
                "Orphan Bookings (after exclusions)",
                """SELECT COUNT(*) FROM bookings b 
                   LEFT JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
                   WHERE br.booking_id IS NULL""",
                "DELETE FROM bookings WHERE id NOT IN (SELECT DISTINCT CAST(booking_id AS BIGINT) FROM booked_rooms)"
            ))
        
        if self.config.exclude_non_spain_hotels:
            # Spain bounding box: lat 35.5-44°N, lon -10-5°E
            # Canary Islands: lat 27.5-29.5°N, lon -18.5--13°W
            rules.append(Rule(
                "Exclude Non-Spain Hotels",
                """SELECT COUNT(DISTINCT b.hotel_id) FROM bookings b
                   JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                   WHERE b.status IN ('confirmed', 'Booked')
                     AND NOT (
                         -- Mainland Spain + Balearic Islands
                         (hl.latitude BETWEEN 35.5 AND 44.0 AND hl.longitude BETWEEN -10.0 AND 5.0)
                         OR
                         -- Canary Islands
                         (hl.latitude BETWEEN 27.5 AND 29.5 AND hl.longitude BETWEEN -18.5 AND -13.0)
                     )""",
                """DELETE FROM bookings WHERE hotel_id IN (
                    SELECT DISTINCT hl.hotel_id FROM hotel_location hl
                    WHERE NOT (
                        (hl.latitude BETWEEN 35.5 AND 44.0 AND hl.longitude BETWEEN -10.0 AND 5.0)
                        OR
                        (hl.latitude BETWEEN 27.5 AND 29.5 AND hl.longitude BETWEEN -18.5 AND -13.0)
                    )
                )"""
            ))
            rules.append(Rule(
                "Exclude Non-Spain Booked Rooms",
                """SELECT COUNT(*) FROM booked_rooms br
                   WHERE CAST(br.booking_id AS BIGINT) NOT IN (SELECT id FROM bookings)""",
                """DELETE FROM booked_rooms 
                   WHERE CAST(booking_id AS BIGINT) NOT IN (SELECT id FROM bookings)"""
            ))
        
        if self.config.exclude_missing_location:
            # Phase 1: Before empty string cleaning
            rules.append(Rule(
                "Exclude Missing Location (Phase 1)",
                """SELECT COUNT(*) FROM booked_rooms br
                   WHERE EXISTS (
                       SELECT 1 FROM bookings b
                       JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                       WHERE CAST(br.booking_id AS BIGINT) = b.id
                         AND b.status IN ('confirmed', 'Booked')
                         AND hl.city IS NULL 
                         AND (hl.latitude IS NULL OR hl.longitude IS NULL)
                   )""",
                """DELETE FROM booked_rooms
                   WHERE EXISTS (
                       SELECT 1 FROM bookings b
                       JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                       WHERE CAST(booked_rooms.booking_id AS BIGINT) = b.id
                         AND b.status IN ('confirmed', 'Booked')
                         AND hl.city IS NULL 
                         AND (hl.latitude IS NULL OR hl.longitude IS NULL)
                   )"""
            ))
            rules.append(Rule(
                "Exclude Missing Location Bookings (Phase 1)",
                """SELECT COUNT(DISTINCT b.id) FROM bookings b
                   JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                   WHERE hl.city IS NULL 
                     AND (hl.latitude IS NULL OR hl.longitude IS NULL)""",
                """DELETE FROM bookings
                   WHERE id IN (
                       SELECT DISTINCT b.id FROM bookings b
                       JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                       WHERE hl.city IS NULL 
                         AND (hl.latitude IS NULL OR hl.longitude IS NULL)
                   )"""
            ))
        
        # ===== DATA QUALITY FIXES =====
        if self.config.fix_empty_strings:
            # Note: room_view is handled separately by set_empty_room_view_to_no_view_str
            # to avoid conflicts. Only convert room_view to NULL if the no_view rule is disabled.
            if not self.config.set_empty_room_view_to_no_view_str:
                rules.append(Rule(
                    "Fix Empty room_view",
                    "SELECT COUNT(*) FROM booked_rooms WHERE room_view = ''",
                    "UPDATE booked_rooms SET room_view = NULL WHERE room_view = ''"
                ))
            rules.append(Rule(
                "Fix Empty city",
                "SELECT COUNT(*) FROM hotel_location WHERE city = ''",
                "UPDATE hotel_location SET city = NULL WHERE city = ''"
            ))
            rules.append(Rule(
                "Fix Empty country",
                "SELECT COUNT(*) FROM hotel_location WHERE country = ''",
                "UPDATE hotel_location SET country = NULL WHERE country = ''"
            ))
            rules.append(Rule(
                "Fix Empty address",
                "SELECT COUNT(*) FROM hotel_location WHERE address = ''",
                "UPDATE hotel_location SET address = NULL WHERE address = ''"
            ))
        
        # Phase 2 of missing location (after empty string cleaning)
        if self.config.exclude_missing_location:
            rules.append(Rule(
                "Exclude Missing Location (Phase 2)",
                """SELECT COUNT(*) FROM booked_rooms br
                   WHERE EXISTS (
                       SELECT 1 FROM bookings b
                       JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                       WHERE CAST(br.booking_id AS BIGINT) = b.id
                         AND b.status IN ('confirmed', 'Booked')
                         AND hl.city IS NULL 
                         AND (hl.latitude IS NULL OR hl.longitude IS NULL)
                   )""",
                """DELETE FROM booked_rooms
                   WHERE EXISTS (
                       SELECT 1 FROM bookings b
                       JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                       WHERE CAST(booked_rooms.booking_id AS BIGINT) = b.id
                         AND b.status IN ('confirmed', 'Booked')
                         AND hl.city IS NULL 
                         AND (hl.latitude IS NULL OR hl.longitude IS NULL)
                   )"""
            ))
            rules.append(Rule(
                "Exclude Missing Location Bookings (Phase 2)",
                """SELECT COUNT(DISTINCT b.id) FROM bookings b
                   JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                   WHERE hl.city IS NULL 
                     AND (hl.latitude IS NULL OR hl.longitude IS NULL)""",
                """DELETE FROM bookings
                   WHERE id IN (
                       SELECT DISTINCT b.id FROM bookings b
                       JOIN hotel_location hl ON b.hotel_id = hl.hotel_id
                       WHERE hl.city IS NULL 
                         AND (hl.latitude IS NULL OR hl.longitude IS NULL)
                   )"""
            ))
        
        # ===== IMPUTATIONS =====
        if self.config.impute_children_allowed:
            rules.append(Rule(
                "Impute children_allowed",
                """SELECT COUNT(*) FROM rooms 
                   WHERE id IN (SELECT DISTINCT room_id FROM booked_rooms WHERE total_children > 0)
                   AND (children_allowed = FALSE OR children_allowed IS NULL)""",
                """UPDATE rooms SET children_allowed = TRUE 
                   WHERE id IN (SELECT DISTINCT room_id FROM booked_rooms WHERE total_children > 0)"""
            ))
        
        if self.config.impute_events_allowed:
            rules.append(Rule(
                "Impute events_allowed",
                """SELECT COUNT(*) FROM rooms 
                   WHERE id IN (SELECT DISTINCT room_id FROM booked_rooms WHERE room_type = 'reception_hall')
                   AND (events_allowed = FALSE OR events_allowed IS NULL)""",
                """UPDATE rooms SET events_allowed = TRUE 
                   WHERE id IN (SELECT DISTINCT room_id FROM booked_rooms WHERE room_type = 'reception_hall')"""
            ))
        if self.config.set_empty_room_view_to_no_view_str:
            rules.append(Rule(
                "Set Empty Room View to 'no_view'",
                "SELECT COUNT(*) FROM booked_rooms WHERE room_view = ''",
                "UPDATE booked_rooms SET room_view = 'no_view' WHERE room_view = ''"
            ))
        # The city name merging is NOT handled by a Rule, but as a separate step.
        return rules
    
    def _merge_city_names_with_tfidf(self, con: duckdb.DuckDBPyConnection):
        """
        Merge city names that are highly similar using tf-idf and cosine similarity.

        Any two city names with cosine similarity above the threshold are merged to the most frequent one.
        """
        threshold = self.config.city_name_similarity_threshold
        if self.config.verbose:
            logger.info(f"Running city name merging with TF-IDF (threshold: {threshold})")

        # Step 1: Get all non-null city values and their frequencies
        df_city = con.execute("""
            SELECT LOWER(TRIM(city)) AS city, COUNT(*) AS freq
            FROM hotel_location
            WHERE city IS NOT NULL
            GROUP BY LOWER(TRIM(city))
        """).fetchdf()

        if df_city.empty or len(df_city) == 1:
            if self.config.verbose:
                logger.info("No city name standardization necessary.")
            return

        city_names = df_city["city"].tolist()
        freqs = dict(zip(df_city["city"], df_city["freq"]))
        n = len(city_names)

        # Step 2: Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        tfidf = vectorizer.fit_transform(city_names)

        # Step 3: Compute cosine similarity matrix
        sim_matrix = cosine_similarity(tfidf)

        # Step 4: Find groups to merge
        # We'll assign each city to its group, and for each group select the most frequent.
        # A naive connected-components graph clustering for similarities > threshold.

        visited = set()
        groups = []
        for i in range(n):
            if i in visited:
                continue
            group = set([i])
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop()
                # Compare against all remaining cities
                for j in range(n):
                    if j != curr and j not in visited and sim_matrix[curr, j] > threshold:
                        group.add(j)
                        queue.append(j)
                        visited.add(j)
            groups.append(group)

        # For each group, select representative as the most frequently occurring city name
        city_merge_map = {}
        for group in groups:
            group_cities = [city_names[idx] for idx in group]
            group_freqs = [freqs[city] for city in group_cities]
            # pick the most frequent
            rep_idx = group_cities.index(group_cities[np.argmax(group_freqs)])
            representative = group_cities[rep_idx]
            for idx in group:
                city_merge_map[city_names[idx]] = representative

        # Only perform changes if there are duplicates to merge
        merge_pairs = {c: r for c, r in city_merge_map.items() if c != r}
        if not merge_pairs:
            if self.config.verbose:
                logger.info("No city names above similarity threshold. No merge needed.")
            return

        # Step 5: Update the hotel_location table to merge city names
        # Loop over non-representative cities and update to representative
        # It's safe to use lower(trim(city)) as keys since that's how the map was built

        changes = 0
        for src, rep in merge_pairs.items():
            changes += con.execute(
                "UPDATE hotel_location SET city = ? WHERE lower(trim(city)) = ?",
                (rep, src)
            ).rowcount

        if self.config.verbose:
            logger.info(f"  • Merged {len(merge_pairs)} city name(s) into {len(groups)} groups, {changes} row(s) updated.")

    def _remove_price_outliers(self, con: duckdb.DuckDBPyConnection):
        """
        Remove top 2% and bottom 2% of booking prices (per-night rate).
        
        This helps eliminate data entry errors and extreme outliers that
        would skew the model's price predictions.
        """
        if self.config.verbose:
            logger.info("Removing price outliers (top/bottom 2%)...")
        
        # Compute per-night price and get percentiles
        percentiles = con.execute("""
            WITH per_night_prices AS (
                SELECT 
                    br.id,
                    br.total_price / GREATEST(
                        (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)), 1
                    ) AS price_per_night
                FROM booked_rooms br
                JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
                WHERE b.status IN ('confirmed', 'Booked')
                  AND br.total_price > 0
            )
            SELECT 
                PERCENTILE_CONT(0.02) WITHIN GROUP (ORDER BY price_per_night) AS p02,
                PERCENTILE_CONT(0.98) WITHIN GROUP (ORDER BY price_per_night) AS p98
            FROM per_night_prices
        """).fetchone()
        
        if percentiles[0] is None or percentiles[1] is None:
            if self.config.verbose:
                logger.info("  • No valid prices found for percentile calculation")
            return
        
        p02, p98 = percentiles
        
        if self.config.verbose:
            logger.info(f"  • Price per night range: €{p02:.2f} (P2) to €{p98:.2f} (P98)")
        
        # Count and delete outliers
        count_result = con.execute(f"""
            SELECT COUNT(*) FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            WHERE b.status IN ('confirmed', 'Booked')
              AND br.total_price > 0
              AND (
                  br.total_price / GREATEST(
                      (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)), 1
                  ) < {p02}
                  OR 
                  br.total_price / GREATEST(
                      (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)), 1
                  ) > {p98}
              )
        """).fetchone()[0]
        
        if count_result > 0:
            con.execute(f"""
                DELETE FROM booked_rooms WHERE id IN (
                    SELECT br.id FROM booked_rooms br
                    JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
                    WHERE b.status IN ('confirmed', 'Booked')
                      AND br.total_price > 0
                      AND (
                          br.total_price / GREATEST(
                              (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)), 1
                          ) < {p02}
                          OR 
                          br.total_price / GREATEST(
                              (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)), 1
                          ) > {p98}
                      )
                )
            """)
            self.stats["Price Outliers (Top/Bottom 2%)"] = count_result
            if self.config.verbose:
                logger.info(f"  ✓ Removed {count_result:,} price outliers")
    
    def _clean_malicious_data(self, con: duckdb.DuckDBPyConnection):
        """
        Clean malicious data patterns (XSS, SQL injection attempts) from text fields.
        """
        if self.config.verbose:
            logger.info("Cleaning malicious data patterns...")
        
        # Patterns that indicate malicious data (safe for SQL LIKE)
        malicious_patterns = [
            '%<script%', '%<img%', '%onerror%', '%onclick%', 
            '%javascript:%', '%alert(%', '%document.cookie%'
        ]
        
        fields = [('city', 'hotel_location'), ('zip', 'hotel_location'), ('address', 'hotel_location')]
        
        total_cleaned = 0
        for field, table in fields:
            for pattern in malicious_patterns:
                count = con.execute(f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE LOWER({field}) LIKE '{pattern}'
                """).fetchone()[0]
                
                if count > 0:
                    con.execute(f"""
                        UPDATE {table} SET {field} = NULL
                        WHERE LOWER({field}) LIKE '{pattern}'
                    """)
                    total_cleaned += count
        
        if total_cleaned > 0:
            self.stats["Malicious Data Cleaned"] = total_cleaned
            if self.config.verbose:
                logger.info(f"  ✓ Cleaned {total_cleaned} malicious data entries")
    
    def _clean_suspicious_city_names(self, con: duckdb.DuckDBPyConnection):
        """
        Clean suspicious city names that appear to be fake, test data, or data entry errors.
        
        Patterns detected:
        - Names with repeated words (e.g., "Very Cool Cool City")
        - Names that are too short (< 2 chars)
        - Names that are too long (> 50 chars)
        - Names containing numbers
        - Names containing suspicious words (test, sample, demo, etc.)
        """
        if self.config.verbose:
            logger.info("Cleaning suspicious city names...")
        
        suspicious_patterns = [
            # Repeated words
            ("Repeated words", r"(\b\w+\b).*\b\1\b"),
            # Too short
            ("Too short", None),  # Handled separately
            # Contains numbers
            ("Contains numbers", r"[0-9]"),
            # Suspicious words
            ("Suspicious words", r"\b(test|sample|demo|fake|example|xxx|null|none|unknown|asdf|qwerty)\b"),
        ]
        
        # Get all city names
        cities_df = con.execute("""
            SELECT DISTINCT city, hotel_id 
            FROM hotel_location 
            WHERE city IS NOT NULL
        """).fetchdf()
        
        if cities_df.empty:
            return
        
        import re
        
        suspicious_hotel_ids = set()
        suspicious_cities = []
        
        for _, row in cities_df.iterrows():
            city = row['city']
            hotel_id = row['hotel_id']
            
            if pd.isna(city) or city.strip() == '':
                continue
            
            city_lower = city.lower().strip()
            
            # Check for repeated words (like "Very Cool Cool City")
            words = city_lower.split()
            if len(words) != len(set(words)):
                suspicious_hotel_ids.add(hotel_id)
                suspicious_cities.append(city)
                continue
            
            # Too short (< 2 chars)
            if len(city_lower) < 2:
                suspicious_hotel_ids.add(hotel_id)
                suspicious_cities.append(city)
                continue
            
            # Too long (> 50 chars)
            if len(city_lower) > 50:
                suspicious_hotel_ids.add(hotel_id)
                suspicious_cities.append(city)
                continue
            
            # Contains digits
            if re.search(r'[0-9]', city_lower):
                suspicious_hotel_ids.add(hotel_id)
                suspicious_cities.append(city)
                continue
            
            # Suspicious words
            if re.search(r'\b(test|sample|demo|fake|example|xxx|null|none|asdf|qwerty|cool cool|very cool)\b', city_lower):
                suspicious_hotel_ids.add(hotel_id)
                suspicious_cities.append(city)
                continue
        
        if suspicious_hotel_ids:
            # Set suspicious city names to NULL
            hotel_id_list = ','.join(str(h) for h in suspicious_hotel_ids)
            con.execute(f"""
                UPDATE hotel_location 
                SET city = NULL 
                WHERE hotel_id IN ({hotel_id_list})
            """)
            
            self.stats["Suspicious City Names"] = len(suspicious_hotel_ids)
            if self.config.verbose:
                logger.info(f"  ✓ Set {len(suspicious_hotel_ids)} suspicious city names to NULL")
                if len(suspicious_cities) <= 10:
                    for city in suspicious_cities:
                        logger.info(f"    - '{city}'")
                else:
                    for city in suspicious_cities[:5]:
                        logger.info(f"    - '{city}'")
                    logger.info(f"    ... and {len(suspicious_cities) - 5} more")

    def _impute_city_from_nearest(self, con: duckdb.DuckDBPyConnection):
        """
        Impute city names for hotels that have coordinates but no city.
        Uses cities500.json to find the nearest city.
        """
        import json
        from pathlib import Path
        
        if self.config.verbose:
            logger.info("Imputing city names from nearest city (lat/lon lookup)...")
        
        # Load cities500.json
        project_root = Path(__file__).parent.parent.parent
        cities_path = project_root / 'data' / 'cities500.json'
        
        if not cities_path.exists():
            return
        
        with open(cities_path, 'r') as f:
            cities_data = json.load(f)
        
        spain_cities = [c for c in cities_data if c.get('country') == 'ES']
        
        # Find hotels with coords but no city
        hotels_need_city = con.execute("""
            SELECT hotel_id, latitude, longitude
            FROM hotel_location
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL
              AND (city IS NULL OR city = '')
        """).fetchdf()
        
        if len(hotels_need_city) == 0:
            if self.config.verbose:
                logger.info("  • No hotels need city imputation")
            return
        
        city_updates = 0
        for _, row in hotels_need_city.iterrows():
            hotel_id = row['hotel_id']
            lat, lon = row['latitude'], row['longitude']
            
            nearest = self._find_nearest_city(lat, lon, spain_cities)
            if nearest:
                con.execute(
                    "UPDATE hotel_location SET city = ? WHERE hotel_id = ?",
                    (nearest['name'], hotel_id)
                )
                city_updates += 1
        
        if city_updates > 0:
            self.stats["Imputed City (post-clean)"] = city_updates
            if self.config.verbose:
                logger.info(f"  ✓ Imputed city names for {city_updates} hotels")
    
    def _find_nearest_city(self, lat: float, lon: float, cities: list, max_dist_km: float = 50.0) -> Optional[dict]:
        """
        Find the nearest city from a list of cities based on lat/lon.
        
        Args:
            lat, lon: Coordinates to search from
            cities: List of city dicts with 'name', 'lat', 'lon' keys
            max_dist_km: Maximum distance to consider (default 50km)
            
        Returns:
            Nearest city dict or None if none within max_dist_km
        """
        if not cities:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for city in cities:
            # Simple Euclidean approximation (good enough for Spain)
            dlat = (city['lat'] - lat) * 111  # ~111km per degree latitude
            dlon = (city['lon'] - lon) * 85   # ~85km per degree longitude at Spain's latitude
            dist = (dlat**2 + dlon**2)**0.5
            
            if dist < min_dist and dist < max_dist_km:
                min_dist = dist
                nearest = city
        
        return nearest
    
    def _impute_coordinates_from_cities500(self, con: duckdb.DuckDBPyConnection):
        """
        Impute missing latitude/longitude from cities500.json using city name matching.
        
        Uses fuzzy matching to find the best matching city in Spain.
        """
        import json
        from pathlib import Path
        
        if self.config.verbose:
            logger.info("Imputing missing coordinates from cities500.json...")
        
        # Load cities500.json
        project_root = Path(__file__).parent.parent.parent
        cities_path = project_root / 'data' / 'cities500.json'
        
        if not cities_path.exists():
            if self.config.verbose:
                logger.info("  • cities500.json not found, skipping imputation")
            return
        
        with open(cities_path, 'r') as f:
            cities_data = json.load(f)
        
        # Filter to Spain cities (country='ES')
        spain_cities = [c for c in cities_data if c.get('country') == 'ES']
        
        if self.config.verbose:
            logger.info(f"  • Loaded {len(spain_cities):,} Spanish cities from cities500.json")
        
        # Build lookup dict by normalized city name
        city_lookup = {}
        for city in spain_cities:
            name = city['name'].lower().strip()
            # Keep the one with higher population if duplicate
            if name not in city_lookup or city.get('pop', 0) > city_lookup[name].get('pop', 0):
                city_lookup[name] = city
        
        # Also add common variations
        variations = {}
        for name, city in city_lookup.items():
            # Remove accents for matching
            import unicodedata
            normalized = ''.join(
                c for c in unicodedata.normalize('NFD', name)
                if unicodedata.category(c) != 'Mn'
            )
            if normalized != name and normalized not in city_lookup:
                variations[normalized] = city
        city_lookup.update(variations)
        
        # Get hotels with missing coordinates but valid city name
        missing_coords = con.execute("""
            SELECT hotel_id, city
            FROM hotel_location
            WHERE (latitude IS NULL OR longitude IS NULL)
              AND city IS NOT NULL
              AND city != ''
        """).fetchdf()
        
        if missing_coords.empty:
            if self.config.verbose:
                logger.info("  • No hotels with missing coordinates found")
            return
        
        if self.config.verbose:
            logger.info(f"  • Found {len(missing_coords)} hotels with missing coordinates")
        
        # Match and update
        updates = 0
        matched_cities = []
        
        for _, row in missing_coords.iterrows():
            hotel_id = row['hotel_id']
            city_name = row['city']
            
            if pd.isna(city_name):
                continue
            
            # Normalize city name
            city_normalized = city_name.lower().strip()
            
            # Try exact match first
            matched_city = city_lookup.get(city_normalized)
            
            # Try without accents
            if not matched_city:
                import unicodedata
                normalized = ''.join(
                    c for c in unicodedata.normalize('NFD', city_normalized)
                    if unicodedata.category(c) != 'Mn'
                )
                matched_city = city_lookup.get(normalized)
            
            # Try partial match (city name contains or is contained)
            if not matched_city:
                for lookup_name, lookup_city in city_lookup.items():
                    if lookup_name in city_normalized or city_normalized in lookup_name:
                        matched_city = lookup_city
                        break
            
            if matched_city:
                lat = matched_city['lat']
                lon = matched_city['lon']
                con.execute(
                    "UPDATE hotel_location SET latitude = ?, longitude = ? WHERE hotel_id = ?",
                    (lat, lon, hotel_id)
                )
                updates += 1
                matched_cities.append((city_name, matched_city['name']))
        
        self.stats["Imputed Coordinates"] = updates
        if self.config.verbose:
            logger.info(f"  ✓ Imputed coordinates for {updates} hotels")
            if updates > 0 and updates <= 10:
                for orig, matched in matched_cities:
                    logger.info(f"    - '{orig}' → '{matched}'")
            elif updates > 10:
                for orig, matched in matched_cities[:5]:
                    logger.info(f"    - '{orig}' → '{matched}'")
                logger.info(f"    ... and {updates - 5} more")
        
        # Second pass: impute city names from nearest city in cities500 for hotels with coords but no city
        if self.config.verbose:
            logger.info("  Imputing city names from nearest city (lat/lon lookup)...")
        
        hotels_with_coords_no_city = con.execute("""
            SELECT hotel_id, latitude, longitude
            FROM hotel_location
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL
              AND (city IS NULL OR city = '')
        """).fetchdf()
        
        if len(hotels_with_coords_no_city) > 0:
            city_updates = 0
            for _, row in hotels_with_coords_no_city.iterrows():
                hotel_id = row['hotel_id']
                lat, lon = row['latitude'], row['longitude']
                
                # Find nearest city
                nearest_city = self._find_nearest_city(lat, lon, spain_cities)
                if nearest_city:
                    con.execute(
                        "UPDATE hotel_location SET city = ? WHERE hotel_id = ?",
                        (nearest_city['name'], hotel_id)
                    )
                    city_updates += 1
            
            if city_updates > 0:
                self.stats["Imputed City from Coords"] = city_updates
                if self.config.verbose:
                    logger.info(f"  ✓ Imputed city names for {city_updates} hotels from nearest city")
        
        # Third pass: impute from zip codes using other hotels with same zip
        if self.config.verbose:
            logger.info("  Imputing coordinates from zip codes...")
        
        still_missing = con.execute("""
            SELECT hotel_id, zip
            FROM hotel_location
            WHERE (latitude IS NULL OR longitude IS NULL)
              AND zip IS NOT NULL
              AND zip != ''
        """).fetchdf()
        
        zip_updates = 0
        for _, row in still_missing.iterrows():
            hotel_id = row['hotel_id']
            zip_code = row['zip']
            
            # Skip invalid zip codes (must be alphanumeric only)
            if not zip_code or not str(zip_code).replace(' ', '').replace('-', '').isalnum():
                continue
            
            # Find another hotel with same zip that has coordinates (parameterized query)
            match = con.execute("""
                SELECT latitude, longitude, city
                FROM hotel_location
                WHERE zip = ?
                  AND latitude IS NOT NULL
                  AND longitude IS NOT NULL
                LIMIT 1
            """, [zip_code]).fetchone()
            
            if match:
                lat, lon, city = match
                con.execute(
                    "UPDATE hotel_location SET latitude = ?, longitude = ? WHERE hotel_id = ?",
                    (lat, lon, hotel_id)
                )
                zip_updates += 1
        
        if zip_updates > 0:
            self.stats["Imputed from Zip Code"] = zip_updates
            if self.config.verbose:
                logger.info(f"  ✓ Imputed coordinates from zip code for {zip_updates} hotels")

    def clean(self, con: duckdb.DuckDBPyConnection) -> 'DuckDBConnectionWrapper':
        """
        Apply all enabled rules to clean the data, including tf-idf city merge if enabled.
        Returns wrapped connection that converts None→NaN.
        """
        if self.config.verbose:
            logger.info(f"Applying {len(self.rules)} data cleaning rules...")
        
        # Impute missing coordinates BEFORE rules (so exclude_missing_location has a chance to keep imputed hotels)
        if getattr(self.config, "impute_coordinates_from_cities500", True):
            self._impute_coordinates_from_cities500(con)
        
        # Apply rules
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            affected = con.execute(rule.check_query).fetchone()[0]
            
            if affected > 0:
                con.execute(rule.action_query)
                self.stats[rule.name] = affected
                
                if self.config.verbose:
                    logger.info(f"  ✓ {rule.name}: {affected:,} rows")
            elif self.config.verbose:
                logger.info(f"  - {rule.name}: 0 rows")
        
        # Remove price outliers (top/bottom 2%) - requires computing percentiles
        if getattr(self.config, "remove_price_outliers", True):
            self._remove_price_outliers(con)
        
        # Clean malicious data (XSS, SQL injection attempts)
        self._clean_malicious_data(con)
        
        # Clean suspicious city names
        if getattr(self.config, "clean_suspicious_city_names", True):
            self._clean_suspicious_city_names(con)
        
        # Re-run nearest city imputation for hotels whose city was just cleaned
        if getattr(self.config, "impute_coordinates_from_cities500", True):
            self._impute_city_from_nearest(con)
        
        # Match/merge city names using tfidf if configured
        if getattr(self.config, "match_city_names_with_tfidf", False):
            self._merge_city_names_with_tfidf(con)

        # Log table stats
        if self.config.verbose:
            bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
            booked_rooms = con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
            logger.info(f"\nFinal: {bookings:,} bookings, {booked_rooms:,} booked_rooms")
        
        return DuckDBConnectionWrapper(con)

# ============================================================================
# 4. CONNECTION WRAPPER (None→NaN Conversion)
# ============================================================================

class DuckDBConnectionWrapper:
    """
    Wrapper for DuckDB connection that converts None to NaN in fetchdf() results.
    This ensures consistent pandas representation of missing values.
    """
    
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self._con = con
    
    def __getattr__(self, name: str):
        """Delegate all other attributes to the underlying connection."""
        attr = getattr(self._con, name)
        return attr
    
    def execute(self, *args, **kwargs):
        """
        Execute a query and return a wrapped result that converts None to NaN in fetchdf().
        """
        result = self._con.execute(*args, **kwargs)
        return _DuckDBResultWrapper(result)
    
    def fetchone(self, *args, **kwargs):
        """Delegate fetchone to underlying connection."""
        return self._con.fetchone(*args, **kwargs)
    
    def fetchall(self, *args, **kwargs):
        """Delegate fetchall to underlying connection."""
        return self._con.fetchall(*args, **kwargs)


class _DuckDBResultWrapper:
    """
    Wrapper for DuckDB query result that converts None to NaN in fetchdf().
    """
    
    def __init__(self, result):
        self._result = result
    
    def fetchdf(self, *args, **kwargs):
        """Fetch DataFrame and convert None and empty strings to NaN."""
        df = self._result.fetchdf(*args, **kwargs)
        # Convert None and empty strings to NaN for all object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Replace None with NaN, then empty strings with NaN
                df[col] = df[col].where(df[col].notna(), np.nan)
                df[col] = df[col].where(df[col] != '', np.nan)
        return df
    
    def fetchone(self, *args, **kwargs):
        """Delegate fetchone to underlying result."""
        return self._result.fetchone(*args, **kwargs)
    
    def fetchall(self, *args, **kwargs):
        """Delegate fetchall to underlying result."""
        return self._result.fetchall(*args, **kwargs)
    
    def __getattr__(self, name: str):
        """Delegate all other attributes to the underlying result."""
        return getattr(self._result, name)


# ============================================================================
# 5. BACKWARD COMPATIBILITY
# ============================================================================

def validate_and_clean(
    con: duckdb.DuckDBPyConnection,
    verbose: bool = False,
    rooms_to_exclude: list[str] | None = None,
    exclude_missing_location_bookings: bool = False
) -> DuckDBConnectionWrapper:
    """
    DEPRECATED: Use DataCleaner with CleaningConfig instead.
    
    This function is kept for backward compatibility with existing scripts.
    It only applies a SUBSET of available cleaning rules.
    
    For full control and to enable ALL cleaning rules, use the new API:
    
        from lib.data_validator import CleaningConfig, DataCleaner
        
        config = CleaningConfig(
            # Enable ALL cleaning rules
            remove_negative_prices=True,
            remove_zero_prices=True,
            remove_low_prices=True,
            remove_null_prices=True,
            remove_extreme_prices=True,
            remove_null_dates=True,
            remove_null_created_at=True,
            remove_negative_stay=True,
            remove_negative_lead_time=True,
            remove_null_occupancy=True,
            remove_overcrowded_rooms=True,
            remove_null_room_id=True,
            remove_null_booking_id=True,
            remove_null_hotel_id=True,
            remove_orphan_bookings=True,
            remove_null_status=True,
            remove_cancelled_but_active=True,
            remove_bookings_before_2023=True,
            remove_bookings_after_2024=True,
            exclude_reception_halls=True,
            exclude_missing_location=True,
            fix_empty_strings=True,
            impute_children_allowed=True,
            impute_events_allowed=True,
            match_city_names_with_tfidf=True,
            city_name_similarity_threshold=0.97,
            verbose=True
        )
        cleaner = DataCleaner(config)
        clean_con = cleaner.clean(init_db())
    
    WARNING: This function will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "validate_and_clean() is deprecated. Use CleaningConfig and DataCleaner instead. "
        "See function docstring for migration instructions.",
        DeprecationWarning,
        stacklevel=2
    )
    
    config = CleaningConfig(
        exclude_reception_halls='reception_hall' in (rooms_to_exclude or []),
        exclude_missing_location=exclude_missing_location_bookings,
        verbose=verbose
    )
    return DataCleaner(config).clean(con)


def check_data_quality(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Check data quality without modifying data.
    Returns dict with results for each rule.
    
    This function creates a default config and checks all rules.
    """
    config = CleaningConfig(
        # Enable all checks
        verbose=False
    )
    cleaner = DataCleaner(config)
    
    results = []
    total_failed = 0
    
    for rule in cleaner.rules:
        failed = con.execute(rule.check_query).fetchone()[0]
        # Get total count (approximate from rule name)
        if "booked_rooms" in rule.check_query:
            total = con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        elif "bookings" in rule.check_query:
            total = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        else:
            total = failed  # For UPDATE rules, just use failed count
        
        pct = (failed / total * 100) if total > 0 else 0
        
        results.append({
            'name': rule.name,
            'failed': failed,
            'total': total,
            'pct': pct
        })
        total_failed += failed
    
    return {
        'rules': results,
        'total_failed': total_failed,
        'checks_passed': sum(1 for r in results if r['failed'] == 0),
        'total_checks': len(cleaner.rules)
    }
