import os
import cv2
import pandas as pd
import numpy as np
import re

print("=== Milestone 2: Feature Engineering ===")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS_CSV = os.path.join(
    BASE_DIR, "milestone1", "tracefinder_labels.csv"
)

IMAGES_DIR = os.path.join(
    BASE_DIR, "milestone1", "images"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR, "milestone2", "features.csv"
)

# --------------------------------------------------
# Load labels
# --------------------------------------------------
df = pd.read_csv(LABELS_CSV)

print(f"[INFO] Loaded {len(df)} image entries")
print(f"[INFO] CSV columns: {list(df.columns)}")

# --------------------------------------------------
# Prepare output
# --------------------------------------------------
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

header_written = False
processed = 0
skipped = 0
skipped_nan_dpi = 0

# --------------------------------------------------
# Helper: extract DPI from filename
# --------------------------------------------------
def extract_dpi_from_name(fname):
    numbers = re.findall(r"\d+", fname)
    for n in numbers:
        if int(n) in (150, 300, 600):
            return int(n)
    return None

# --------------------------------------------------
# Feature extraction loop
# --------------------------------------------------
for _, row in df.iterrows():

    scanner = str(row["scanner"]).strip()

    # ---- FIX: handle NaN DPI safely ----
    if pd.isna(row["dpi"]):
        skipped += 1
        skipped_nan_dpi += 1
        continue

    dpi_csv = int(float(row["dpi"]))

    scanner_dir = os.path.join(IMAGES_DIR, scanner)

    if not os.path.isdir(scanner_dir):
        skipped += 1
        continue

    img_path = None

    for fname in os.listdir(scanner_dir):
        if not fname.lower().endswith(".tif"):
            continue

        dpi_file = extract_dpi_from_name(fname)

        if dpi_file == dpi_csv:
            img_path = os.path.join(scanner_dir, fname)
            break

    if img_path is None:
        skipped += 1
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        skipped += 1
        continue

    # --------------------------------------------------
    # Feature extraction
    # --------------------------------------------------
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))

    edges = cv2.Canny(img, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)

    feature_row = {
        "scanner": scanner,
        "dpi": dpi_csv,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "edge_density": edge_density,
        "source_type": row.get("source_type", ""),
        "doc_id": row.get("doc_id", "")
    }

    pd.DataFrame([feature_row]).to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not header_written,
        index=False
    )

    header_written = True
    processed += 1

    if processed % 100 == 0:
        print(f"[INFO] Processed {processed}/{len(df)} images")

# --------------------------------------------------
# Summary
# --------------------------------------------------
print("=======================================")
print("[INFO] Feature extraction completed")
print(f"[INFO] Total processed: {processed}")
print(f"[INFO] Total skipped: {skipped}")
print(f"[INFO] Skipped due to NaN DPI: {skipped_nan_dpi}")
print(f"[INFO] Features saved to: {OUTPUT_CSV}")






