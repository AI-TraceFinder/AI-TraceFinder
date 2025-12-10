import os
import csv
import base64
from PIL import Image
import io

base_path = "/Users/mithra/Desktop/Scanner-Forensics/dataset/raw/Official"
csv_file = "scanner_dataset_small.csv"

def encode_image_small(image_path, max_size=(400, 400), quality=40):
    # Open image
    img = Image.open(image_path)
    
    # Resize to make small
    img.thumbnail(max_size)
    
    # Save to memory buffer
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    
    # Base64 encode
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

count = 0

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["brand", "model", "image_name", "image_base64"])

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

                try:
                    encoded_img = encode_image_small(image_path)
                except Exception as e:
                    print(f"Skipping corrupted file: {image_path}")
                    continue

                writer.writerow([brand, model, image, encoded_img])

                count += 1
                print(f"Processed {count} images...", end="\r")

print(f"\nDONE: Optimized CSV created → {csv_file}")
