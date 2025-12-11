##!/usr/bin/env python3
"""
scripts/create_labels.py

Optimized version of the original labeling script:
 - uses pathlib
 - streaming MD5 calculation
 - parallel file processing (ThreadPoolExecutor)
 - CLI options for data dir, output, workers
 - robust image metadata extraction with fallbacks
"""

from __future__ import annotations
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os
import sys

# Allowed image file extensions (lowercase)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def compute_md5(path: Path, chunk_size: int = 8192) -> Optional[str]:
    """Compute MD5 checksum in streaming fashion to avoid reading whole file into memory."""
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def get_image_metadata(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str], Optional[int]]:
    """
    Return (width, height, channels_or_mode, format, dpi) or None values if unable to read.
    channels_or_mode tries to present channels count and mode (e.g. '3-ch (RGB)').
    """
    try:
        with Image.open(path) as im:
            width, height = im.size
            fmt = im.format
            mode = im.mode
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
        # can't open as an image
        return None, None, None, None, None
    except Exception:
        return None, None, None, None, None


@dataclass
class FileTask:
    fpath: Path
    category: str
    scanner_model: str


def collect_file_tasks(data_dir: Path) -> List[FileTask]:
    """
    Walk the data directory and create FileTask objects for all image files.
    Top-level entries in data_dir are categories. Subfolders under each category are considered scanner-models.
    """
    tasks: List[FileTask] = []
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found at: {data_dir}")

    # Deterministic listing
    top_level = sorted(p for p in data_dir.iterdir())

    print("Found top-level items in data/:", [p.name for p in top_level])

    for category_entry in top_level:
        if not category_entry.is_dir():
            continue
        category = category_entry.name

        # Walk category_entry deterministically
        for root, dirs, files in os.walk(category_entry):
            dirs.sort()
            files.sort()
            root_path = Path(root)
            # compute relative path to categorize scanner_model
            try:
                rel = root_path.relative_to(category_entry)
            except Exception:
                rel = Path(".")
            if str(rel) == ".":
                scanner_model = "unknown"
            else:
                parts = rel.parts
                scanner_model = parts[0] if parts and parts[0] else "unknown"

            for fname in files:
                fpath = root_path / fname
                if not fpath.is_file():
                    continue
                if fpath.suffix.lower() not in IMAGE_EXTS:
                    continue
                tasks.append(FileTask(fpath=fpath, category=category, scanner_model=scanner_model))

    return tasks


def process_task(task: FileTask) -> Dict:
    """Process a single file task: read metadata, filesize, md5, return dict row for DataFrame."""
    fpath = task.fpath
    width, height, channels_or_mode, fmt, dpi = get_image_metadata(fpath)

    try:
        file_size = fpath.stat().st_size
    except Exception:
        file_size = None

    checksum = compute_md5(fpath)

    return {
        "image_path": str(fpath.resolve()),
        "category": task.category,
        "scanner_model": task.scanner_model,
        "file_name": fpath.name,
        "width": width,
        "height": height,
        "channels_or_mode": channels_or_mode,
        "format": fmt,
        "dpi": dpi,
        "file_size_bytes": file_size,
        "md5": checksum
    }


def write_results(rows: List[Dict], output_csv: Path) -> None:
    """Save rows to CSV and print diagnostics."""
    df = pd.DataFrame(rows)
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Create image_labels.csv for images organized under data/")
    p.add_argument("--data-dir", type=str, default=None, help="Path to data directory (defaults to ../data relative to this script).")
    p.add_argument("--output", type=str, default=None, help="Output CSV path (defaults to data/image_labels.csv).")
    p.add_argument("--workers", type=int, default=8, help="Number of worker threads for parallel processing (default: 8). Use 0 or 1 to disable concurrency.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = THIS_DIR.parent
    default_data_dir = PROJECT_DIR / "data"

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: data directory not found at: {data_dir}")
        print("Please ensure your project contains a 'data' folder with subfolders per category, or pass --data-dir.")
        sys.exit(1)

    output_csv = Path(args.output) if args.output else (data_dir / "image_labels.csv")

    print(f"Scanning images under: {data_dir}")
    tasks = collect_file_tasks(data_dir)

    rows: List[Dict] = []
    if not tasks:
        print("No candidate image files found. Exiting.")
        write_results(rows, output_csv)
        return

    # Parallel processing (threading is suitable because operations are I/O bound: file read + PIL open)
    workers = max(1, args.workers)
    if workers <= 1:
        # sequential
        for t in tasks:
            rows.append(process_task(t))
    else:
        with ThreadPoolExecutor(max_workers=workers) as exe:
            future_to_task = {exe.submit(process_task, t): t for t in tasks}
            for fut in as_completed(future_to_task):
                try:
                    row = fut.result()
                except Exception as exc:
                    # log and continue
                    t = future_to_task[fut]
                    print(f"Error processing {t.fpath}: {exc}")
                    continue
                rows.append(row)

    # Keep deterministic ordering (sort by image_path)
    rows.sort(key=lambda r: r.get("image_path", ""))

    write_results(rows, output_csv)


if __name__ == "__main__":
    main()
