import matplotlib
matplotlib.use("TkAgg")   

import pandas as pd
import cv2
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv")

classes = df["label"].unique()

plt.figure(figsize=(10, 6))

for i, cls in enumerate(classes):
    path = df[df["label"] == cls].iloc[0]["image_path"]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    plt.subplot(1, len(classes), i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Scanner {cls}")
    plt.axis('off')

plt.tight_layout()
plt.show()
