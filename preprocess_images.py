import os
import cv2

# 🔧 CHANGE THESE PATHS IF NEEDED
BASE_PATH = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Datasets"
PROCESSED_PATH = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Processed"

DATASET_FOLDERS = ["Flatfield", "Official", "Wikipedia"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TARGET_SIZE = (512, 512)

for dataset in DATASET_FOLDERS:
    in_dataset = os.path.join(BASE_PATH, dataset)
    out_dataset = os.path.join(PROCESSED_PATH, dataset)

    if not os.path.exists(in_dataset):
        print(f"⚠️ Skipping missing folder: {in_dataset}")
        continue

    os.makedirs(out_dataset, exist_ok=True)

    print(f"\n📂 Processing dataset: {dataset}")
    print(f"   Input : {in_dataset}")
    print(f"   Output: {out_dataset}")

    # loop over scanner folders inside each dataset
    for scanner_folder in os.listdir(in_dataset):
        in_scanner = os.path.join(in_dataset, scanner_folder)
        if not os.path.isdir(in_scanner):
            continue

        out_scanner = os.path.join(out_dataset, scanner_folder)
        os.makedirs(out_scanner, exist_ok=True)

        print(f"\n   🔍 Scanner: {scanner_folder}")
        print(f"      In : {in_scanner}")
        print(f"      Out: {out_scanner}")

        # walk through all files inside this scanner folder
        for root, _, files in os.walk(in_scanner):
            for file in files:

                # ✅ Skip macOS hidden junk files like "._150.tif"
                if file.startswith("._"):
                    continue

                # ✅ Only process image files
                if not file.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                in_path = os.path.join(root, file)

                # read as GRAYSCALE
                img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"❌ Could not read: {in_path}")
                    continue

                # resize to 512x512
                img_resized = cv2.resize(img, TARGET_SIZE)

                # save to processed folder (same filename)
                out_path = os.path.join(out_scanner, file)
                cv2.imwrite(out_path, img_resized)

                print(f"✅ Saved: {out_path}")

print("\n🎉 Preprocessing completed!")
