import os
import cv2
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
INPUT_FOLDER = "/content/drive/MyDrive/a/originals"
OUTPUT_FOLDER = "/content/drive/MyDrive/a/Output"
CSV_FILE = "/content/drive/MyDrive/a/labels.csv"
IMAGE_SIZE = (512, 512)
GRAYSCALE = True
NUM_PROCESSES = 8
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.resize(img, IMAGE_SIZE)
        if GRAYSCALE:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        filename = os.path.basename(image_path)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, img)
        return filename
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

image_paths = [os.path.join(root, f)
               for root, _, files in os.walk(INPUT_FOLDER)
               for f in files if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

if not image_paths:
    print("No images found in", INPUT_FOLDER)
else:
    processed_files = []
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {executor.submit(process_image, path): path for path in image_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = f.result()
            if result:
                processed_files.append(result)

    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    df = pd.DataFrame({"filename": processed_files})
    df.to_csv(CSV_FILE, index=False)

    print("Preprocessing complete!")
    print(f"Total images processed: {len(processed_files)}")
