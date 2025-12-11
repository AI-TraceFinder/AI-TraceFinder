import os
from PIL import Image
import tifffile

input_root = "dataset_scanners"
output_root = "preprocessed_images"

if not os.path.exists(output_root):
    os.makedirs(output_root)

for subdir, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            input_path = os.path.join(subdir, file)

            rel_path = os.path.relpath(subdir, input_root)
            output_subdir = os.path.join(output_root, rel_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Read TIFF using tifffile
            try:
                img_array = tifffile.imread(input_path)  # returns a numpy array
                # Convert to PIL Image for further processing
                if img_array.ndim == 2:
                    img = Image.fromarray(img_array)        # grayscale
                else:
                    img = Image.fromarray(img_array[:, :, :3])  # RGB (ignore alpha if exists)

                img = img.resize((256, 256))
                img = img.convert("L")  # convert to grayscale

                output_path = os.path.join(output_subdir, file)
                img.save(output_path)
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

print(f"All images preprocessed and saved in '{output_root}'")
