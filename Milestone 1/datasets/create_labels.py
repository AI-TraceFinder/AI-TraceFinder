import os
import csv
import hashlib
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "image_labels.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV Header
HEADERS = [
    "scanner_model",
    "file_name",
    "image_path",
    "category",
    "width",
    "height",
    "mode",
    "format",
    "dpi",
    "file_size",
    "md5_checksum"
]

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

rows = []

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
        image_path = os.path.join(IMAGE_DIR, file)

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                fmt = img.format
                dpi = img.info.get("dpi", "Not Available")

            file_size = os.path.getsize(image_path)
            checksum = get_md5(image_path)

            # Simple manual logic (change if needed)
            scanner_model = "Scanner_1" # example: scannerA_001.png
            category = "document"

            rows.append([
                scanner_model,
                file,
                image_path,
                category,
                width,
                height,
                mode,
                fmt,
                dpi,
                file_size,
                checksum
            ])

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Write CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(HEADERS)
    writer.writerows(rows)

print("✅ image_labels.csv created successfully")
