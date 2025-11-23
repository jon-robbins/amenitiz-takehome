import duckdb
import logging
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Single validation rule with check and clean queries."""
    name: str
    fail_query: str
    total_query: str
    delete_query: str


RULES = [
    # Price issues
    Rule("Negative Price", 
         "SELECT COUNT(*) FROM booked_rooms WHERE total_price < 0",
         "SELECT COUNT(*) FROM booked_rooms",
         "DELETE FROM booked_rooms WHERE total_price < 0"),
    
    Rule("Zero Price",
         "SELECT COUNT(*) FROM booked_rooms WHERE total_price = 0",
         "SELECT COUNT(*) FROM booked_rooms",
         "DELETE FROM booked_rooms WHERE total_price = 0"),
    
    Rule("NULL Price",
         "SELECT COUNT(*) FROM booked_rooms WHERE total_price IS NULL",
         "SELECT COUNT(*) FROM booked_rooms",
         "DELETE FROM booked_rooms WHERE total_price IS NULL"),
    
    Rule("Extreme Price (>5k/night)",
         """SELECT COUNT(*) FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            WHERE (b.status IN ('confirmed', 'Booked'))
              AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
              AND (br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) > 5000""",
         """SELECT COUNT(*) FROM booked_rooms br
            JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
            WHERE (b.status IN ('confirmed', 'Booked'))
              AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0""",
         """DELETE FROM booked_rooms WHERE id IN (
                SELECT br.id FROM booked_rooms br
                JOIN bookings b ON CAST(br.booking_id AS BIGINT) = b.id
                WHERE (b.status IN ('confirmed', 'Booked'))
                  AND (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE)) > 0
                  AND (br.total_price / (CAST(b.departure_date AS DATE) - CAST(b.arrival_date AS DATE))) > 5000)"""),
    
    # Date issues
    Rule("NULL Dates",
         "SELECT COUNT(*) FROM bookings WHERE arrival_date IS NULL OR departure_date IS NULL",
         "SELECT COUNT(*) FROM bookings",
         "DELETE FROM bookings WHERE arrival_date IS NULL OR departure_date IS NULL"),
    
    Rule("NULL Created At",
         "SELECT COUNT(*) FROM bookings WHERE created_at IS NULL",
         "SELECT COUNT(*) FROM bookings",
         "DELETE FROM bookings WHERE created_at IS NULL"),
    
    Rule("Negative Stay",
         "SELECT COUNT(*) FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)",
         "SELECT COUNT(*) FROM bookings",
         "DELETE FROM bookings WHERE CAST(departure_date AS DATE) <= CAST(arrival_date AS DATE)"),
    
    Rule("Negative Lead Time",
         "SELECT COUNT(*) FROM bookings WHERE CAST(created_at AS DATE) > CAST(arrival_date AS DATE)",
         "SELECT COUNT(*) FROM bookings",
         "DELETE FROM bookings WHERE CAST(created_at AS DATE) > CAST(arrival_date AS DATE)"),
    
    # Occupancy issues
    Rule("NULL Occupancy",
         "SELECT COUNT(*) FROM booked_rooms WHERE total_adult IS NULL AND total_children IS NULL",
         "SELECT COUNT(*) FROM booked_rooms",
         "DELETE FROM booked_rooms WHERE total_adult IS NULL AND total_children IS NULL"),
    
    Rule("Overcrowded Room",
         """SELECT COUNT(*) FROM booked_rooms br
            JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
            WHERE (COALESCE(br.total_adult, 0) + COALESCE(br.total_children, 0)) > r.max_occupancy""",
         "SELECT COUNT(*) FROM booked_rooms",
         """DELETE FROM booked_rooms WHERE id IN (
                SELECT br.id FROM booked_rooms br
                JOIN rooms r ON CAST(br.room_id AS BIGINT) = r.id
                WHERE (COALESCE(br.total_adult, 0) + COALESCE(br.total_children, 0)) > r.max_occupancy)"""),
    
    # Referential integrity
    Rule("NULL Room ID",
         "SELECT COUNT(*) FROM booked_rooms WHERE room_id IS NULL",
         "SELECT COUNT(*) FROM booked_rooms",
         "DELETE FROM booked_rooms WHERE room_id IS NULL"),
    
    Rule("NULL Booking ID",
         "SELECT COUNT(*) FROM booked_rooms WHERE booking_id IS NULL",
         "SELECT COUNT(*) FROM booked_rooms",
         "DELETE FROM booked_rooms WHERE booking_id IS NULL"),
    
    Rule("NULL Hotel ID",
         "SELECT COUNT(*) FROM bookings WHERE hotel_id IS NULL",
         "SELECT COUNT(*) FROM bookings",
         "DELETE FROM bookings WHERE hotel_id IS NULL"),
    
    Rule("Orphan Bookings",
         """SELECT COUNT(*) FROM bookings b 
            LEFT JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
            WHERE br.booking_id IS NULL""",
         "SELECT COUNT(*) FROM bookings",
         "DELETE FROM bookings WHERE id NOT IN (SELECT DISTINCT CAST(booking_id AS BIGINT) FROM booked_rooms)"),
    
    Rule("NULL Status",
         "SELECT COUNT(*) FROM bookings WHERE status IS NULL",
         "SELECT COUNT(*) FROM bookings",
         "DELETE FROM bookings WHERE status IS NULL"),
    
    Rule("Cancelled but Active",
         """SELECT COUNT(*) FROM bookings 
            WHERE status IN ('confirmed', 'Booked') 
            AND cancelled_date IS NOT NULL""",
         "SELECT COUNT(*) FROM bookings",
         """DELETE FROM bookings 
            WHERE status IN ('confirmed', 'Booked') 
            AND cancelled_date IS NOT NULL"""),
]


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
        # If it's a method we want to wrap, wrap it; otherwise return as-is
        if name == 'fetchdf':
            return self._fetchdf_wrapper
        if callable(attr):
            return attr
        return attr
    
    def _fetchdf_wrapper(self, *args, **kwargs):
        """Wrapper for fetchdf that converts None to NaN."""
        df = self._con.fetchdf(*args, **kwargs)
        # Convert None to NaN for all columns
        return df.replace([None], np.nan)
    
    # Explicitly delegate common methods for clarity
    def execute(self, *args, **kwargs):
        """Delegate execute to underlying connection."""
        return self._con.execute(*args, **kwargs)
    
    def fetchone(self, *args, **kwargs):
        """Delegate fetchone to underlying connection."""
        return self._con.fetchone(*args, **kwargs)
    
    def fetchall(self, *args, **kwargs):
        """Delegate fetchall to underlying connection."""
        return self._con.fetchall(*args, **kwargs)


def clean_data_quality_issues(con: duckdb.DuckDBPyConnection, verbose: bool = False) -> None:
    """
    Fix data quality issues that don't require deletion.
    Handles empty strings and standardizes missing values.
    """
    if verbose:
        logger.info("Fixing data quality issues...")
    
    # Replace empty room views with NULL (room_view is in booked_rooms table)
    # Note: SQL NULL values are automatically converted to pandas NaN when using fetchdf()
    con.execute("""
        UPDATE booked_rooms 
        SET room_view = NULL 
        WHERE room_view = ''
    """)
    
    # Replace empty strings in hotel_location text columns with NULL
    # Note: latitude/longitude are DOUBLE types, not strings, so they're already handled
    for col in ['city', 'country', 'address']:
        con.execute(f"""
            UPDATE hotel_location 
            SET {col} = NULL 
            WHERE {col} = ''
        """)
    
    if verbose:
        logger.info("Data quality issues fixed")


def impute_policy_flags(con: duckdb.DuckDBPyConnection, verbose: bool = False) -> None:
    """
    Impute policy flags based on actual booking behavior.
    
    Imputation logic:
    - children_allowed: TRUE if room has any bookings with children
    - events_allowed: TRUE if room_type is 'reception_hall'
    
    Note: pets_allowed and smoking_allowed cannot be imputed due to lack of data.
    """
    if verbose:
        logger.info("Imputing policy flags from booking behavior...")
    
    # Impute children_allowed based on bookings with children
    con.execute("""
        UPDATE rooms
        SET children_allowed = TRUE
        WHERE id IN (
            SELECT DISTINCT room_id
            FROM booked_rooms
            WHERE total_children > 0
        )
    """)
    
    # Impute events_allowed for reception halls
    con.execute("""
        UPDATE rooms
        SET events_allowed = TRUE
        WHERE id IN (
            SELECT DISTINCT room_id
            FROM booked_rooms
            WHERE room_type = 'reception_hall'
        )
    """)
    
    if verbose:
        children_imputed = con.execute("""
            SELECT COUNT(*) FROM rooms WHERE children_allowed = TRUE
        """).fetchone()[0]
        
        events_imputed = con.execute("""
            SELECT COUNT(*) FROM rooms WHERE events_allowed = TRUE
        """).fetchone()[0]
        
        logger.info(f"  children_allowed: {children_imputed:,} rooms marked TRUE")
        logger.info(f"  events_allowed: {events_imputed:,} rooms marked TRUE")



def check_data_quality(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Check data quality without modifying data.
    Returns dict with results for each rule.
    """
    results = []
    total_failed = 0
    
    for rule in RULES:
        failed = con.execute(rule.fail_query).fetchone()[0]
        total = con.execute(rule.total_query).fetchone()[0]
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
        'total_checks': len(RULES)
    }


def validate_and_clean(
    con: duckdb.DuckDBPyConnection, 
    verbose: bool = False,
    rooms_to_exclude: list[str] | None = None
) -> DuckDBConnectionWrapper:
    """
    Check data quality, drop bad rows, fix data issues, impute policy flags, and return wrapped connection.
    The returned connection will automatically convert None to NaN in fetchdf() results.
    Logs progress and summary when verbose=True.
    
    Args:
        con: DuckDB connection
        verbose: If True, log progress and summary
        rooms_to_exclude: Optional list of room_type values to exclude (e.g., ['reception_hall'])
    """
    if verbose:
        logger.info("Checking data quality...")
    
    # Check before
    before = check_data_quality(con)
    if verbose:
        logger.info(f"Found {before['total_failed']:,} problematic rows")
        logger.info(f"Quality: {before['checks_passed']}/{before['total_checks']} checks passed\n")
        
        # Show issues
        for r in before['rules']:
            if r['failed'] > 0:
                logger.info(f"  {r['name']}: {r['failed']:,} / {r['total']:,} ({r['pct']:.2f}%)")
    
    # Clean bad rows
    if verbose:
        logger.info("\nCleaning data...")
    total_deleted = 0
    for rule in RULES:
        failed = con.execute(rule.fail_query).fetchone()[0]
        if failed > 0:
            con.execute(rule.delete_query)
            total_deleted += failed
    
    # Exclude specific room types if requested
    if rooms_to_exclude:
        if verbose:
            logger.info(f"\nExcluding room types: {', '.join(rooms_to_exclude)}")
        
        for room_type in rooms_to_exclude:
            # Count before deletion
            count_before = con.execute("""
                SELECT COUNT(*) FROM booked_rooms WHERE room_type = ?
            """, [room_type]).fetchone()[0]
            
            # Delete booked_rooms with this room_type
            con.execute("""
                DELETE FROM booked_rooms WHERE room_type = ?
            """, [room_type])
            
            total_deleted += count_before
            
            if verbose:
                logger.info(f"  Removed {count_before:,} booked_rooms with room_type='{room_type}'")
        
        # Clean up orphan bookings after room exclusion
        orphan_count = con.execute("""
            SELECT COUNT(*) FROM bookings b 
            LEFT JOIN booked_rooms br ON b.id = CAST(br.booking_id AS BIGINT)
            WHERE br.booking_id IS NULL
        """).fetchone()[0]
        
        if orphan_count > 0:
            con.execute("""
                DELETE FROM bookings 
                WHERE id NOT IN (SELECT DISTINCT CAST(booking_id AS BIGINT) FROM booked_rooms)
            """)
            total_deleted += orphan_count
            
            if verbose:
                logger.info(f"  Removed {orphan_count:,} orphan bookings")
    
    # Fix data quality issues (empty strings, etc.)
    clean_data_quality_issues(con, verbose=verbose)
    
    # Impute policy flags from booking behavior
    impute_policy_flags(con, verbose=verbose)
    
    # Summary
    if verbose:
        bookings = con.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        booked_rooms = con.execute("SELECT COUNT(*) FROM booked_rooms").fetchone()[0]
        
        logger.info(f"\nDeleted {total_deleted:,} rows total")
        logger.info(f"Remaining: {bookings:,} bookings, {booked_rooms:,} booked_rooms")
        
        # Verify clean
        after = check_data_quality(con)
        logger.info(f"Quality after cleaning: {after['checks_passed']}/{after['total_checks']} checks passed")
    
    # Return wrapped connection that converts None to NaN
    return DuckDBConnectionWrapper(con)
