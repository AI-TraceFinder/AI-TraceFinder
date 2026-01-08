import os
import numpy as np
import cv2

# ============== CONFIG ==================
INPUT_DIR = "../Milestone-2/prnu_fingerprints"
OUTPUT_DIR = "dataset_all"
IMG_SIZE = 224
# ========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for scanner in os.listdir(INPUT_DIR):
    in_dir = os.path.join(INPUT_DIR, scanner)
    if not os.path.isdir(in_dir):
        continue

    out_dir = os.path.join(OUTPUT_DIR, scanner)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n🔄 Converting PRNU for: {scanner}")

    for file in os.listdir(in_dir):
        if not file.endswith(".npy"):
            continue

        prnu = np.load(os.path.join(in_dir, file))

        # Normalize PRNU to 0–255
        prnu = prnu - prnu.min()
        prnu = prnu / (prnu.max() + 1e-8)
        prnu = (prnu * 255).astype("uint8")

        # Resize for CNN
        prnu = cv2.resize(prnu, (IMG_SIZE, IMG_SIZE))

        out_name = file.replace(".npy", ".png")
        cv2.imwrite(os.path.join(out_dir, out_name), prnu)

        print(f"✅ Saved {out_name}")

print("\n🎯 PRNU PNG conversion completed successfully.")
