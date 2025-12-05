"""
Utility functions for loading SQL queries from .sql files.
"""
from pathlib import Path


def load_sql_file(sql_filename: str, relative_to_file: str | Path | None = None) -> str:
    """
    Load SQL query from a .sql file.
    
    Parameters
    ----------
    sql_filename : str
        Name of the SQL file (e.g., 'QUERY_LOAD_HOTEL_LOCATIONS.sql').
        Can include relative path from the calling file's directory.
    relative_to_file : str | Path | None, default=None
        Path to the file calling this function. If None, uses caller's __file__.
        Typically passed as __file__ from the calling module.
    
    Returns
    -------
    str
        Contents of the SQL file.
    
    Examples
    --------
    >>> # In calculate_distance_features.py
    >>> query = load_sql_file('QUERY_LOAD_HOTEL_LOCATIONS.sql', __file__)
    """
    if relative_to_file is None:
        raise ValueError(
            "relative_to_file must be provided. Use: load_sql_file('query.sql', __file__)"
        )
    
    # Get the directory of the calling file
    caller_dir = Path(relative_to_file).parent
    
    # Build path to SQL file
    sql_path = caller_dir / sql_filename
    
    if not sql_path.exists():
        raise FileNotFoundError(
            f"SQL file not found: {sql_path}\n"
            f"Expected location: {sql_path.absolute()}"
        )
    
    # Read and return SQL content
    return sql_path.read_text(encoding='utf-8')

