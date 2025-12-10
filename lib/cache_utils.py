"""
Utility functions for caching DataFrame results to CSV.

Provides a decorator to automatically cache function results to CSV files,
allowing expensive computations to be skipped if cached results exist.
"""

import functools
import os
from pathlib import Path
from typing import Callable, TypeVar, ParamSpec

import pandas as pd

P = ParamSpec('P')
R = TypeVar('R')


def cache_to_csv(
    csv_path: str,
    save_after_compute: bool = True
) -> Callable[[Callable[P, pd.DataFrame]], Callable[P, pd.DataFrame]]:
    """
    Decorator to cache DataFrame results to CSV.
    
    This decorator allows functions that return DataFrames to:
    - Automatically save results to CSV after computation
    - Check for existing CSV and load if environment variable is set
    
    The decorator checks the LOAD_FROM_CSV environment variable or
    a function-specific environment variable to determine if CSV should be loaded.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file for caching. Can be relative or absolute.
        Relative paths are resolved relative to project root.
    save_after_compute : bool, default=True
        If True, save function result to CSV after execution.
    
    Returns
    -------
    Decorated function that handles CSV caching.
    
    Environment Variables:
    - LOAD_FROM_CSV: If set to '1' or 'true', all cached functions will load from CSV
    - <FUNCTION_NAME>_LOAD_FROM_CSV: Function-specific override
    
    Examples
    --------
    >>> @cache_to_csv(csv_path="outputs/data.csv")
    ... def load_data() -> pd.DataFrame:
    ...     # ... expensive computation ...
    ...     return df
    """
    def decorator(func: Callable[P, pd.DataFrame]) -> Callable[P, pd.DataFrame]:
        # Resolve CSV path relative to project root (outside wrapper so it's accessible)
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.is_absolute():
            project_root = Path(__file__).parent.parent
            csv_path_obj = project_root / csv_path
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
            
            # Check if we should load from CSV
            # Check function-specific env var first, then global
            func_env_var = f"{func.__name__.upper()}_LOAD_FROM_CSV"
            load_from_csv = (
                os.getenv(func_env_var, '').lower() in ('1', 'true', 'yes') or
                os.getenv('LOAD_FROM_CSV', '').lower() in ('1', 'true', 'yes')
            )
            
            # Load from cache if requested and file exists
            if load_from_csv and csv_path_obj.exists():
                return pd.read_csv(csv_path_obj)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Save to CSV if requested
            if save_after_compute:
                if not isinstance(result, pd.DataFrame):
                    raise TypeError(
                        f"Function {func.__name__} must return pd.DataFrame "
                        f"to use @cache_to_csv decorator. Got {type(result)}."
                    )
                
                csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(csv_path_obj, index=False)
            
            return result
        
        # Store CSV path on wrapper for external access
        wrapper._csv_path = csv_path_obj
        wrapper._load_from_csv = lambda: (
            wrapper._csv_path.exists() and pd.read_csv(wrapper._csv_path)
            if wrapper._csv_path.exists() else None
        )
        
        return wrapper
    return decorator


def load_cached_csv(csv_path: str) -> pd.DataFrame | None:
    """
    Helper function to load a cached CSV file if it exists.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file (relative to project root or absolute).
    
    Returns
    -------
    pd.DataFrame | None
        Loaded DataFrame if file exists, None otherwise.
    """
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.is_absolute():
        project_root = Path(__file__).parent.parent
        csv_path_obj = project_root / csv_path
    
    if csv_path_obj.exists():
        return pd.read_csv(csv_path_obj)
    return None

