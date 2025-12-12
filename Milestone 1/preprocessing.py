import os
from PIL import Image
import tifffile

input_root = "/Users/mithra/Jeya_Mithra_Tracefinder-1/dataset_scanners_copy"
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

            try:
                img_array = tifffile.imread(input_path)

                if img_array.ndim == 2:
                    img = Image.fromarray(img_array)
                else:
                    img = Image.fromarray(img_array[:, :, :3])
                img = img.resize((1024, 1024), Image.LANCZOS)
                img = img.convert("L")

                output_path = os.path.join(output_subdir, file)
                img.save(output_path)

            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

print(f"All images preprocessed and saved in '{output_root}'")
