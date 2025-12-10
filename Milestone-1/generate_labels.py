import os
import csv
import re

BASE_PATH = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Datasets"
DATASET_FOLDERS = ["Flatfield", "Official", "Wikipedia"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

rows = []

def extract_label(folder_name):
    clean = folder_name.split("-20")[0]
    match = re.match(r"(Canon\d+(?:-\d+)?|EpsonV\d+(?:-\d+)?|HP)", clean)
    return match.group(0) if match else clean

for dataset in DATASET_FOLDERS:
    dataset_path = os.path.join(BASE_PATH, dataset)

    for scanner_folder in os.listdir(dataset_path):
        scanner_path = os.path.join(dataset_path, scanner_folder)

        if not os.path.isdir(scanner_path):
            continue

        label = extract_label(scanner_folder)

        for root, _, files in os.walk(scanner_path):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    rows.append([label, file_path, dataset])

with open("labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "filepath", "dataset"])
    writer.writerows(rows)

print("✅ labels.csv generated successfully")
print(f"✅ Total samples: {len(rows)}")
