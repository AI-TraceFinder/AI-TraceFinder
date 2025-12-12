import os
import pandas as pd
from PIL import Image

BASE_DIR = "dataset_scanners_copy"   
output_csv = "scanner_labeled_dataset.csv"

data_rows = []

for scanner_model in os.listdir(BASE_DIR):
    scanner_path = os.path.join(BASE_DIR, scanner_model)

    if not os.path.isdir(scanner_path):
        continue

    for subtype in os.listdir(scanner_path):
        subtype_path = os.path.join(scanner_path, subtype)

        if not os.path.isdir(subtype_path):
            continue
        for img_file in os.listdir(subtype_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):

                img_path = os.path.join(subtype_path, img_file)

                try:
                    img = Image.open(img_path)

                    width, height = img.size
                    img_format = img.format
                    color_mode = img.mode

                    data_rows.append({
                        "scanner_model": scanner_model,
                        "subtype": subtype,
                        "file_name": img_file,
                        "file_path": img_path,
                        "resolution_width": width,
                        "resolution_height": height,
                        "format": img_format,
                        "color_mode": color_mode
                    })
                except Exception as e:
                    print("Error reading:", img_path, e)

df = pd.DataFrame(data_rows)
df.to_csv(output_csv, index=False)

print("✅ Labeled dataset created successfully!")
print("📄 Saved CSV as:", output_csv)
print("Total images processed:", len(df))
