import os
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.feature import local_binary_pattern

IMG_SIZE = 512
DATASET_DIR = "/content/drive/MyDrive/a/Output/"
OUTPUT_FILE = "/content/drive/MyDrive/a/Output/features.csv"

# -------------------------------------------------
# Noise (PRNU-like) Extraction using Wavelets
# -------------------------------------------------
def get_noise_component(gray_img):
    gray_img = gray_img.astype(np.float32) / 255.0

    wavelet_result = pywt.wavedec2(gray_img, wavelet="sym4", level=3)

    approximation = wavelet_result[0]
    details = wavelet_result[1:]

    approximation[:] = 0  # suppress image content

    reconstructed = pywt.waverec2([approximation] + list(details), "sym4")
    reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)

    return reconstructed


# -------------------------------------------------
# FFT-based Statistical Features
# -------------------------------------------------
def compute_fft_features(img):
    freq_domain = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.log1p(np.abs(freq_domain))

    return [
        magnitude.mean(),
        magnitude.std(),
        magnitude.max(),
        np.percentile(magnitude, 75),
        np.percentile(magnitude, 25),
        np.sum(magnitude ** 2)
    ]


# -------------------------------------------------
# LBP Texture Histogram
# -------------------------------------------------
def compute_lbp_features(gray_img, bins=20):
    lbp_img = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp_img.flatten(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


# -------------------------------------------------
# Edge-based Features
# -------------------------------------------------
def compute_edge_features(gray_img):
    grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(grad_x, grad_y)

    return [
        edge_mag.mean(),
        edge_mag.std(),
        edge_mag.max()
    ]


# -------------------------------------------------
# Full Feature Extraction Pipeline
# -------------------------------------------------
def build_feature_vector(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise IOError("Unable to read image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    noise_img = get_noise_component(img)

    features = []
    features.extend(compute_fft_features(noise_img))
    features.extend(compute_lbp_features(img))
    features.extend(compute_edge_features(img))

    return features


# -------------------------------------------------
# Dataset Traversal and CSV Creation
# -------------------------------------------------
records = []

print("\n🔍 Starting feature extraction...\n")

for root, _, files in os.walk(DATASET_DIR):
    label = os.path.basename(root)

    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            img_path = os.path.join(root, fname)

            try:
                feature_vec = build_feature_vector(img_path)
                records.append([img_path, label] + feature_vec)
                print(f"✔ Processed: {img_path}")

            except Exception as err:
                print(f"❌ Failed: {img_path} → {err}")


columns = (
    ["image_path", "label"] +
    [f"fft_{i}" for i in range(6)] +
    [f"lbp_{i}" for i in range(20)] +
    [f"edge_{i}" for i in range(3)]
)

df = pd.DataFrame(records, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 Feature extraction completed!")
print(f"📁 Output saved as: {OUTPUT_FILE}")
