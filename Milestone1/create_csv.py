import os
import csv
base_path = "/Users/mithra/Desktop/Scanner-Forensics/dataset/raw/Official"
csv_file = "scanner_dataset.csv"
 
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["brand", "model", "image_path"])
    for brand in os.listdir(base_path):
        brand_path = os.path.join(base_path, brand)
        if not os.path.isdir(brand_path):
            continue
        for model in os.listdir(brand_path):
            model_path = os.path.join(brand_path, model)
            if not os.path.isdir(model_path):
                continue
            for image in os.listdir(model_path):
                image_path = os.path.join(model_path, image)
                writer.writerow([brand, model, image_path])

print(f"CSV file created: {csv_file}")
