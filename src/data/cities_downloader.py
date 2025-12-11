"""
Auto-download cities500.json from GitHub if not present.

Source: https://github.com/lmfmaier/cities-json
License: CC BY 4.0

The file contains ~178K populated places with population >= 500,
including coordinates, country, admin regions, and population.
"""

import json
import urllib.request
from pathlib import Path
from typing import Optional


# GitHub raw URL for cities500.json
CITIES_URL = "https://raw.githubusercontent.com/lmfmaier/cities-json/master/cities500.json"

# Default local path
DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "cities500.json"


def ensure_cities500(target_path: Optional[Path] = None) -> Path:
    """
    Ensure cities500.json exists, downloading if necessary.
    
    Args:
        target_path: Where to save/find the file. Defaults to data/cities500.json
        
    Returns:
        Path to the cities500.json file
    """
    path = target_path or DEFAULT_PATH
    
    if path.exists():
        return path
    
    print(f"cities500.json not found at {path}")
    print(f"Downloading from {CITIES_URL}...")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(CITIES_URL, path)
        print(f"Downloaded cities500.json ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to download cities500.json: {e}")


def load_cities500(target_path: Optional[Path] = None) -> list:
    """
    Load cities500.json, downloading if necessary.
    
    Args:
        target_path: Where to find/save the file
        
    Returns:
        List of city dictionaries with keys: id, name, country, admin1, admin2, lat, lon, pop
    """
    path = ensure_cities500(target_path)
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Test download
    path = ensure_cities500()
    cities = load_cities500()
    print(f"Loaded {len(cities):,} cities")
    
    # Show sample
    spain_cities = [c for c in cities if c.get("country") == "ES"]
    print(f"Spain has {len(spain_cities):,} cities with pop >= 500")
    
    # Show largest Spanish cities
    spain_cities.sort(key=lambda x: int(x.get("pop", 0)), reverse=True)
    print("\nLargest Spanish cities:")
    for city in spain_cities[:10]:
        print(f"  {city['name']}: {int(city['pop']):,}")

