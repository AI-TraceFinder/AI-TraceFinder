import os
import cv2

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: processed images
INPUT_DIR = os.path.join(BASE_DIR, "processed_images")

# Output: noise residual images
OUTPUT_DIR = os.path.join(BASE_DIR, "noise_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔹 Starting noise residual extraction...")

for file_name in os.listdir(INPUT_DIR):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        # Read processed image (grayscale)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Skipped: {file_name}")
            continue

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Extract noise residual
        noise = cv2.subtract(img, blurred)

        # Normalize noise for visibility
        noise_norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)

        # Save noise image
        cv2.imwrite(output_path, noise_norm)

        print(f"✅ Noise extracted: {file_name}")

print("🎉 Noise residual extraction completed successfully!")
