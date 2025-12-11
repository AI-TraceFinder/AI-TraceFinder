import os
import cv2
import numpy as np

# Input folder (raw images)
input_folder = "C:\\Users\\HP\\Desktop\\INFOSYS\\dataset"

# Output folder (processed images will be saved here)
output_folder = "C:\\Users\\HP\\Desktop\\INFOSYS\\processed_dataset"
os.makedirs(output_folder, exist_ok=True)

# Function to preprocess images
def preprocess_image(input_path, output_path, size=(224, 224)):
    # Read image
    img = cv2.imread(input_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise image
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Resize image
    resized = cv2.resize(denoised, size)

    # Normalize (0 to 1)
    normalized = resized / 255.0

    # Convert back to uint8 for saving
    final_img = (normalized * 255).astype("uint8")

    # Save final processed image
    cv2.imwrite(output_path, final_img)

# Process all images inside dataset/
for scanner in os.listdir(input_folder):
    scanner_path = os.path.join(input_folder, scanner)

    if os.path.isdir(scanner_path):
        output_scanner_path = os.path.join(output_folder, scanner)
        os.makedirs(output_scanner_path, exist_ok=True)

        for file in os.listdir(scanner_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_img = os.path.join(scanner_path, file)
                output_img = os.path.join(output_scanner_path, file)

                preprocess_image(input_img, output_img)
                print("Processed:", file)
