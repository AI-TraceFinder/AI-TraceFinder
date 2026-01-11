import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from torchvision import transforms
from model import TraceFinderCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\Techie\Projects\TraceFinder_AI\milestone3\checkpoint_best.pth"

PATCH_SIZE = 224
STRIDE = 112
CONF_THRESHOLD = 0.75   # raised
STD_THRESHOLD = 20.0    # better than variance

CLASS_NAMES = [
    'Canon120-1', 'Canon120-2', 'Canon220',
    'Canon9000-1', 'Canon9000-2',
    'EpsonV370-1', 'EpsonV370-2',
    'EpsonV39-1', 'EpsonV39-2',
    'EpsonV550', 'HP'
]

# ------------------ MODEL ------------------
model = TraceFinderCNN(num_classes=len(CLASS_NAMES))
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ PATCH UTILS ------------------
def extract_patches(img):
    patches = []
    h, w = img.shape[:2]
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patches.append(img[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
    return patches

def is_informative(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return gray.std() > STD_THRESHOLD

# ------------------ DOCUMENT PREDICTION ------------------
def predict_document(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to load image")

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    patches = extract_patches(img)

    conf_sum = defaultdict(float)
    count = defaultdict(int)
    used = 0

    with torch.no_grad():
        for p in patches:
            if not is_informative(p):
                continue

            t = transform(p).unsqueeze(0).to(DEVICE)
            out = model(t)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)

            if conf.item() >= CONF_THRESHOLD:
                cls = pred.item()
                conf_sum[cls] += conf.item()
                count[cls] += 1
                used += 1

    if not conf_sum:
        return None, {}, 0

    # Normalize confidence
    final_scores = {
        k: conf_sum[k] / count[k]
        for k in conf_sum
    }

    final = max(final_scores, key=final_scores.get)
    return final, final_scores, used

# ------------------ TEST ------------------
TEST_IMAGE = r"C:\Techie\Projects\TraceFinder_AI_Dataset\Official\Canon120-1\150\s1_2.tif"

pred, scores, used = predict_document(TEST_IMAGE)

print("\nNormalized patch scores:")
for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{CLASS_NAMES[k]}: {v:.3f}")

print(f"\nUsed patches: {used}")
print("FINAL DOCUMENT PREDICTION:", CLASS_NAMES[pred])





