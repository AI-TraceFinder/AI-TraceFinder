import os
import cv2
import numpy as np
import tifffile
from PIL import Image

INPUT_ROOT = "/Users/mithra/Jeya_Mithra_Tracefinder-1/dataset_scanners_copy"
OUTPUT_ROOT = "preprocessed_images_optimized"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

TARGET_SIZE = 512   

for subdir, _, files in os.walk(INPUT_ROOT):
    for file in files:
        if file.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):

            input_path = os.path.join(subdir, file)
            rel_path = os.path.relpath(subdir, INPUT_ROOT)
            output_subdir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            try:
                img_array = tifffile.imread(input_path)

                if img_array.ndim == 3:
                    img_array = img_array[:, :, 0]

                img = img_array.astype(np.uint8)
                h, w = img.shape
                crop_size = min(h, w)
                start_x = (w - crop_size) // 2
                start_y = (h - crop_size) // 2
                img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]

                
                img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)

                img = cv2.GaussianBlur(img, (3, 3), 0)

                output_path = os.path.join(output_subdir, file)
                Image.fromarray(img).save(output_path)

            except Exception as e:
                print("Failed:", input_path, e)

print("✅ Optimized preprocessing completed!")
