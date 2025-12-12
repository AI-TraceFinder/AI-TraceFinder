import os
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.feature import local_binary_pattern



# ------------------------------------------------------------
# 1️⃣  PRNU Noise Extraction
# ------------------------------------------------------------
def extract_prnu_noise(img):
    img = np.float32(img) / 255.0

    coeffs = pywt.wavedec2(img, "sym4", level=3)   # better for scanner noise
    coeffs_noise = list(coeffs)
    coeffs_noise[0] *= 0  # remove image content

    prnu = pywt.waverec2(coeffs_noise, "sym4")
    prnu = np.uint8(np.clip(prnu * 255, 0, 255))

    return prnu


# ------------------------------------------------------------
# 2️⃣  FFT Features
# ------------------------------------------------------------
def extract_fft_features(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = 20 * np.log(mag + 1)

    flat = mag_log.flatten()

    fft_feat = np.array([
        np.mean(flat),
        np.std(flat),
        np.max(flat),
        np.percentile(flat, 75),
        np.percentile(flat, 25),
        np.sum(flat ** 2)  # Energy
    ])

    return fft_feat


# ------------------------------------------------------------
# 3️⃣  LBP Texture Features
# ------------------------------------------------------------
def extract_lbp(img):
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=20, range=(0, 20))
    hist = hist.astype("float")
    hist = hist / (hist.sum() + 1e-6)
    return hist


# ------------------------------------------------------------
# 4️⃣  Edge Features (Sobel)
# ------------------------------------------------------------
def extract_edges(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    edge_feat = np.array([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude)
    ])

    return edge_feat


# ------------------------------------------------------------
# ⭐ 5️⃣  Combined Feature Extractor
# ------------------------------------------------------------
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image read failed: {img_path}")

    img = cv2.resize(img, (512, 512))

    prnu = extract_prnu_noise(img)
    fft_feat = extract_fft_features(prnu)
    lbp_feat = extract_lbp(img)
    edge_feat = extract_edges(img)

    full_vector = np.hstack([fft_feat, lbp_feat, edge_feat])
    return full_vector



# ------------------------------------------------------------
# ⭐ 6️⃣  PROCESS ENTIRE DATASET
# ------------------------------------------------------------
dataset_root = "preprocessed_images"     # Your dataset folder
output_csv = "features.csv"

rows = []

print("\n🔍 Extracting features from dataset...\n")

for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            img_path = os.path.join(root, file)

            label = os.path.basename(root)   # folder name = scanner label

            try:
                feats = extract_features(img_path)
                feats = feats.tolist()

                rows.append([img_path, label] + feats)
                print(f"✔ Processed {img_path}")

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")


# ------------------------------------------------------------
# ⭐ 7️⃣  SAVE FEATURES TO CSV
# ------------------------------------------------------------
columns = (
    ["image_path", "label"] +
    [f"fft_{i}" for i in range(6)] +
    [f"lbp_{i}" for i in range(20)] +
    [f"edge_{i}" for i in range(3)]
)

df = pd.DataFrame(rows, columns=columns)
df.to_csv(output_csv, index=False)

print("\n🎉 DONE!")
print(f"📁 Features saved to: {output_csv}")
