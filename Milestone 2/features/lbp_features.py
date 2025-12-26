import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: noise residual images
INPUT_DIR = os.path.join(BASE_DIR, "noise_images")

# Output: feature CSV
OUTPUT_DIR = os.path.join(BASE_DIR, "feature_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lbp_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LBP parameters
RADIUS = 1
POINTS = 8 * RADIUS
METHOD = "uniform"
BINS = POINTS + 2  # for uniform LBP

print("🔹 Starting LBP feature extraction...")

feature_rows = []

for file_name in os.listdir(INPUT_DIR):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        image_path = os.path.join(INPUT_DIR, file_name)

        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Skipped: {file_name}")
            continue

        # Compute LBP
        lbp = local_binary_pattern(img, POINTS, RADIUS, METHOD)

        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=BINS, range=(0, BINS), density=True)

        # Scanner label from filename (example: scannerA_doc1.jpg.tif)
        scanner_label = file_name.split("_")[0]

        row = [scanner_label] + hist.tolist()
        feature_rows.append(row)

        print(f"✅ LBP extracted: {file_name}")

# Create DataFrame
columns = ["scanner_model"] + [f"lbp_bin_{i}" for i in range(BINS)]
df = pd.DataFrame(feature_rows, columns=columns)

# Save CSV
df.to_csv(OUTPUT_FILE, index=False)

print("🎉 LBP feature extraction completed successfully!")
print(f"📁 Features saved to: {OUTPUT_FILE}")
