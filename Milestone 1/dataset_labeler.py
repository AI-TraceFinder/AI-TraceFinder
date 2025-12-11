# label_optimized.py
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, UnidentifiedImageError
import pandas as pd
import sys

# Allowed image file extensions (lowercase)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def compute_md5(path: Path, chunk_size: int = 8192) -> Optional[str]:
    """Compute MD5 checksum streaming the file to avoid loading whole file."""
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def get_image_metadata(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str], Optional[int]]:
    """Return (width, height, mode_or_channels, format, dpi) for the image or Nones if unreadable."""
    try:
        with Image.open(path) as im:
            width, height = im.size
            fmt = im.format
            mode = im.mode
            # Try to get channel count if available (number of bands)
            try:
                channels = len(im.getbands())
                channels_or_mode = f"{channels}-ch ({mode})"
            except Exception:
                channels_or_mode = mode

            info = im.info or {}
            dpi_raw = info.get("dpi", None)
            if isinstance(dpi_raw, tuple):
                dpi = dpi_raw[0] or None
            else:
                dpi = dpi_raw

            return width, height, channels_or_mode, fmt, dpi
    except (UnidentifiedImageError, OSError):
        return None, None, None, None, None
    except Exception:
        return None, None, None, None, None


def scan_data_dir(data_dir: Path) -> List[Dict]:
    """
    Walk data_dir and return list of rows (dicts).
    Top-level folders under data/ are treated as categories.
    If images are directly inside category folder, scanner_model = 'unknown'.
    If images are in category/<scanner_model>/... scanner_model = first subfolder name.
    """
    rows: List[Dict] = []
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found at: {data_dir}")

    top_level = sorted([p for p in data_dir.iterdir()])
    print("Found top-level items in data/:", [p.name for p in top_level])

    for category_entry in top_level:
        if not category_entry.is_dir():
            # skip stray files that aren't directories
            continue
        category = category_entry.name

        for root, dirs, files in os_walk_generator(category_entry):
            root_path = Path(root)
            relpath = root_path.relative_to(category_entry)
            # compute scanner model: first directory name under category_entry (if any)
            parts = relpath.parts
            scanner_model = "unknown" if relpath == Path(".") else (parts[0] if parts else "unknown")

            for fname in files:
                fpath = root_path / fname
                if not fpath.is_file():
                    continue
                if fpath.suffix.lower() not in IMAGE_EXTS:
                    continue

                width, height, channels_or_mode, fmt, dpi = get_image_metadata(fpath)
                try:
                    file_size = fpath.stat().st_size
                except Exception:
                    file_size = None

                checksum = compute_md5(fpath)

                rows.append({
                    "image_path": str(fpath.resolve()),
                    "category": category,
                    "scanner_model": scanner_model,
                    "file_name": fname,
                    "width": width,
                    "height": height,
                    "channels_or_mode": channels_or_mode,
                    "format": fmt,
                    "dpi": dpi,
                    "file_size_bytes": file_size,
                    "md5": checksum
                })

    return rows


def os_walk_generator(top: Path):
    """
    Generator wrapper for os.walk that yields (root, dirs, files) with sorted lists
    for consistent output order across runs.
    """
    # using os.walk for scalibility, but convert to strings for compatibility
    import os
    for root, dirs, files in os.walk(top):
        dirs.sort()
        files.sort()
        yield root, dirs, files


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create image metadata CSV for dataset organized under data/")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory (overrides default project layout).")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (defaults to data/image_labels.csv).")
    args = parser.parse_args(argv)

    # Determine DATA_DIR relative to this script file (works regardless of cwd)
    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = THIS_DIR.parent
    default_data_dir = PROJECT_DIR / "data"

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: data directory not found at: {data_dir}")
        print("Please check that your project folder contains a 'data' folder or pass --data-dir to point to it.")
        sys.exit(1)

    output_csv = Path(args.output) if args.output else (data_dir / "image_labels.csv")

    print(f"Scanning images under: {data_dir}")
    rows = scan_data_dir(data_dir)

    df = pd.DataFrame(rows)
    # ensure output folder exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("\nSaved:", output_csv)
    print("Total image rows:", len(df))

    if df.empty:
        print("\nWARNING: No image files were found by the script.")
        print("Possible causes:")
        print(" - data/ folder empty or contains different folder names.")
        print(" - images use uncommon extensions (not jpg/png/tif/...).")
        print(" - images are stored in a different location.")
    else:
        print("\nSample rows:")
        print(df.head(10).to_string(index=False))

        print("\nCounts per category:")
        print(df["category"].value_counts().to_string())

        print("\nCounts per scanner_model (top 20):")
        print(df["scanner_model"].value_counts().head(20).to_string())


if __name__ == "__main__":
    import os
    main()
