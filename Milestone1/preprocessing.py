import os
import cv2
from skimage import io, img_as_float
import numpy as np

base_path =  "/content/drive/MyDrive/a"

INPUT_FOLDER = "/content/drive/MyDrive/a"
OUTPUT_FOLDER = "/content/drive/MyDrive/a/Output"
CSV_FILE = "/content/drive/MyDrive/a/labels.csv"


csv_file = "scanner_dataset_small.csv"

def resize_and_grayscale_dataset(input_dir, output_dir, target_size=(512, 512)):

    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0

    for root, dirs, files in os.walk(input_dir):
        # Create corresponding output subfolder structure
        rel_path = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel_path)
        os.makedirs(out_root, exist_ok=True)

        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue

            img_path = os.path.join(root, file)

            try:
                # Load as grayscale, convert to float [0,1], resize
                img = io.imread(img_path, as_gray=True)
                img = img_as_float(img)
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Save as grayscale PNG (lossless)
                out_path = os.path.join(out_root, f"{os.path.splitext(file)[0]}_gray_{target_size[1]}x{target_size[0]}.png")
                cv2.imwrite(out_path, (img_resized * 255).astype(np.uint8))

                processed_count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"Processed {processed_count} images. Saved to {output_dir}")

# Usage example
if __name__ == "__main__":
    INPUT_DATASET = "/content/drive/MyDrive/a/originals"
    OUTPUT_GRAY = "/content/drive/MyDrive/a/Output/"

    resize_and_grayscale_dataset(INPUT_DATASET, OUTPUT_GRAY, target_size=(512, 512))
    print("Dataset preprocessing complete!")
