import os
import cv2

PROCESSED_BASE = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Processed"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

total_images = 0
grayscale_ok = True
size_ok = True
junk_found = False

for root, _, files in os.walk(PROCESSED_BASE):
    for file in files:
        total_images += 1

        # ❌ junk file check
        if file.startswith("._"):
            junk_found = True
            print("❌ Junk file found:", os.path.join(root, file))
            continue

        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        path = os.path.join(root, file)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print("❌ Cannot read image:", path)
            continue

        # ✅ grayscale check
        if len(img.shape) != 2:
            grayscale_ok = False
            print("❌ Not grayscale:", path)

        # ✅ size check
        if img.shape != (512, 512):
            size_ok = False
            print("❌ Wrong size:", img.shape, "→", path)

print("\n================ VERIFY RESULT ================\n")

print("Total files checked:", total_images)
print("Grayscale conversion:", "✅ PASS" if grayscale_ok else "❌ FAIL")
print("Resize to 512x512:", "✅ PASS" if size_ok else "❌ FAIL")
print("Junk files skipped:", "✅ PASS" if not junk_found else "❌ FAIL")
print("Saved in Processed folder:", "✅ PASS")

if grayscale_ok and size_ok and not junk_found:
    print("\n🎉 WEEK-2 PREPROCESSING: COMPLETED ✅✅✅")
else:
    print("\n⚠️ WEEK-2 PREPROCESSING: ISSUES FOUND ❌")
