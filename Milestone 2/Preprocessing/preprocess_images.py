import os
import cv2

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output folders
INPUT_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_images")

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

print("🔹 Starting image preprocessing...")

for file_name in os.listdir(INPUT_DIR):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        # Read image
        img = cv2.imread(input_path)

        if img is None:
            print(f"❌ Skipped (cannot read): {file_name}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize image
        resized = cv2.resize(gray, IMG_SIZE)

        # Normalize pixel values
        normalized = resized / 255.0

        # Convert back to uint8 for saving
        normalized_uint8 = (normalized * 255).astype("uint8")

        # Save processed image
        cv2.imwrite(output_path, normalized_uint8)

        print(f"✅ Processed: {file_name}")

print("🎉 Preprocessing completed successfully!")
