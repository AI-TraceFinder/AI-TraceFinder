import cv2
import os

img_path = r"C:\Users\M.varshith reddy\Downloads\MileStone-1\Processed\Flatfield\Canon120-1-20251207T055142Z-3-001\150.tif"

print("Checking:", img_path)

# ✅ FORCE GRAYSCALE LOAD
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Image not found or cannot be read")
else:
    print("✅ Image shape:", img.shape)   # should be (512, 512)
    
