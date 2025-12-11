"""
Auto-download GSHHS and WDBII shapefiles on first use.

Downloads from GitHub releases and extracts only the files needed
for coastline distance calculations.
"""

import os
import tarfile
import requests
from pathlib import Path
from typing import Set


URL = "https://github.com/GenericMappingTools/gshhg-gmt/releases/download/2.3.7/gshhg-gmt-2.3.7.tar.gz"

# Files to keep after extraction
KEEP_FILES: Set[str] = {
    "GSHHS_shp/h/GSHHS_h_L1.dbf",
    "GSHHS_shp/h/GSHHS_h_L1.prj",
    "GSHHS_shp/h/GSHHS_h_L1.shp",
    "GSHHS_shp/h/GSHHS_h_L1.shx",
    "WDBII_shp/f/WDBII_border_f_L1.dbf",
    "WDBII_shp/f/WDBII_border_f_L1.prj",
    "WDBII_shp/f/WDBII_border_f_L1.shp",
    "WDBII_shp/f/WDBII_border_f_L1.shx",
}

# Default target directory (relative to project root)
DEFAULT_TARGET_DIR = Path("data/shapefiles")


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/data/shapefile_downloader.py to project root
    return Path(__file__).parent.parent.parent


def get_shapefile_dir() -> Path:
    """Get the shapefile directory path."""
    return get_project_root() / DEFAULT_TARGET_DIR


def shapefiles_exist() -> bool:
    """Check if required shapefiles already exist."""
    shapefile_dir = get_shapefile_dir()
    
    # Check if the main shapefile exists
    main_shp = shapefile_dir / "GSHHS_shp/h/GSHHS_h_L1.shp"
    return main_shp.exists()


def _download(archive_path: Path) -> None:
    """Download the shapefile archive."""
    print(f"Downloading shapefiles from {URL}...")
    print("(This may take a few minutes)")
    
    response = requests.get(URL, stream=True, timeout=300)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):  # 1MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r  Downloaded: {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")
    
    print(f"\n  Download complete: {archive_path}")


def _extract_only(archive_path: Path, target_dir: Path) -> None:
    """Extract only the needed files from the archive."""
    print("Extracting required shapefiles...")
    
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            # Normalize path inside tar (drop root folder like "gshhg-gmt-2.3.7/")
            parts = Path(member.name).parts
            if len(parts) > 1:
                rel = "/".join(parts[1:])  # Drop root folder
                if rel in KEEP_FILES:
                    # Extract into target_dir/rel
                    member.name = rel
                    tar.extract(member, path=target_dir)
                    print(f"  Extracted: {rel}")


def _cleanup(archive_path: Path, target_dir: Path) -> None:
    """Remove archive and any unnecessary files."""
    print("Cleaning up...")
    
    # Remove archive
    if archive_path.exists():
        archive_path.unlink()
        print(f"  Removed archive: {archive_path}")
    
    # Walk and delete anything not in KEEP_FILES
    for root, dirs, files in os.walk(target_dir, topdown=False):
        for f in files:
            full = Path(root) / f
            try:
                rel = str(full.relative_to(target_dir))
                if rel not in KEEP_FILES:
                    full.unlink()
            except ValueError:
                pass
        
        # Remove empty folders
        for d in dirs:
            p = Path(root) / d
            try:
                p.rmdir()
            except OSError:
                pass  # Not empty, skip


def ensure_shapefiles(target_dir: Path | None = None) -> Path:
    """
    Ensure shapefiles are downloaded and available.
    
    Downloads GSHHS and WDBII shapefiles if not already present.
    
    Args:
        target_dir: Optional custom target directory. If None, uses data/shapefiles/
    
    Returns:
        Path to the shapefile directory
    
    Example:
        >>> shapefile_dir = ensure_shapefiles()
        >>> coastline_path = shapefile_dir / "GSHHS_shp/h/GSHHS_h_L1.shp"
    """
    if target_dir is None:
        target_dir = get_shapefile_dir()
    
    target_dir = Path(target_dir)
    
    # Check if already exists
    main_shp = target_dir / "GSHHS_shp/h/GSHHS_h_L1.shp"
    if main_shp.exists():
        return target_dir
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and extract
    archive_path = target_dir / "gshhg.tar.gz"
    
    try:
        _download(archive_path)
        _extract_only(archive_path, target_dir)
        _cleanup(archive_path, target_dir)
        print(f"Shapefiles ready at: {target_dir}")
    except Exception as e:
        # Clean up on failure
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError(f"Failed to download shapefiles: {e}") from e
    
    return target_dir


def get_coastline_path() -> Path:
    """
    Get path to the GSHHS coastline shapefile, downloading if needed.
    
    Returns:
        Path to GSHHS_h_L1.shp (high-resolution coastlines, Level 1)
    """
    shapefile_dir = ensure_shapefiles()
    return shapefile_dir / "GSHHS_shp/h/GSHHS_h_L1.shp"


if __name__ == "__main__":
    # Test the downloader
    shapefile_dir = ensure_shapefiles()
    print(f"\nShapefile directory: {shapefile_dir}")
    print("\nContents:")
    for f in sorted(shapefile_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(shapefile_dir)}")

