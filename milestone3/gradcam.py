import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from model import TraceFinderCNN
from dataset import TraceFinderDataset

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "checkpoint_best.pth"   # correct path
DATASET_ROOT = r"C:\Techie\Projects\TraceFinder_AI_Dataset"

TEST_IMAGE = r"C:\Techie\Projects\TraceFinder_AI_Dataset\Official\Canon120-1\150\s1_2.tif"
PATCH_SIZE = 224

# ---------------- DATA ----------------
dataset = TraceFinderDataset(DATASET_ROOT)
class_names = dataset.class_names

# ---------------- MODEL ----------------
model = TraceFinderCNN(num_classes=len(class_names))
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- GRAD-CAM HOOKS ----------------
gradients = None
activations = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Register on LAST conv block
target_layer = model.model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ---------------- LOAD IMAGE ----------------
img_gray = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise RuntimeError("Failed to load image")

img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
img_rgb = cv2.resize(img_rgb, (224, 224))

input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

# ---------------- FORWARD + BACKWARD ----------------
output = model(input_tensor)
prob = torch.softmax(output, dim=1)
conf, pred = torch.max(prob, 1)

model.zero_grad()
output[0, pred.item()].backward()

# ---------------- BUILD GRAD-CAM ----------------
weights = gradients.mean(dim=(2, 3))
cam = torch.sum(weights[:, :, None, None] * activations, dim=1)
cam = cam.squeeze().detach().cpu().numpy()

cam = np.maximum(cam, 0)
cam = cam / cam.max()

cam = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

# ---------------- SAVE OUTPUT ----------------
os.makedirs("results/gradcam", exist_ok=True)

cv2.imwrite("results/gradcam/original.png", img_rgb)
cv2.imwrite("results/gradcam/heatmap.png", heatmap)
cv2.imwrite("results/gradcam/overlay.png", overlay)

print("Grad-CAM generated successfully")
print(f"Predicted class: {class_names[pred.item()]}")
print(f"Confidence: {conf.item()*100:.2f}%")
print("Saved to: results/gradcam/")

