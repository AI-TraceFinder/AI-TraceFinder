import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
INPUT_ROOT = "preprocessed_images_optimized"
OUTPUT_CSV = "image_features.csv"
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

RESIZE_SHAPE = (256, 256)  
feature_list = []
labels = []
for subdir, _, files in os.walk(INPUT_ROOT):
    for file in files:
        if file.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            img_path = os.path.join(subdir, file)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Image could not be read")

                img = cv2.resize(img, RESIZE_SHAPE)

                glcm = graycomatrix(
                    img,
                    distances=GLCM_DISTANCES,
                    angles=GLCM_ANGLES,
                    levels=256,
                    symmetric=True,
                    normed=True
                )

                contrast = graycoprops(glcm, 'contrast').flatten()
                dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
                homogeneity = graycoprops(glcm, 'homogeneity').flatten()
                energy = graycoprops(glcm, 'energy').flatten()
                correlation = graycoprops(glcm, 'correlation').flatten()

                
                lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=np.arange(0, LBP_POINTS + 3),
                    range=(0, LBP_POINTS + 2)
                )
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-6)

               
                mean = np.mean(img)
                std = np.std(img)
                skewness = np.mean((img - mean) ** 3) / (std ** 3 + 1e-6)
                kurtosis = np.mean((img - mean) ** 4) / (std ** 4 + 1e-6)

                features = np.hstack([
                    contrast,
                    dissimilarity,
                    homogeneity,
                    energy,
                    correlation,
                    hist,
                    [mean, std, skewness, kurtosis]
                ])

                feature_list.append(features)
                labels.append(os.path.basename(subdir))

            except Exception as e:
                print(f"❌ Failed: {img_path} → {e}")

features_df = pd.DataFrame(feature_list)
features_df["label"] = labels
features_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Feature extraction completed!")
print("📄 Saved to:", OUTPUT_CSV)
print("🔢 Total features per image:", features_df.shape[1] - 1)
