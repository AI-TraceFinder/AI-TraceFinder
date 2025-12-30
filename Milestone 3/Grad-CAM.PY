import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 128
DATASET_PATH = "/Users/mithra/Jeya_Mithra_Tracefinder-1/dataset_scanners_copy"
class ScannerCNN(nn.Module):
    def __init__(self, num_classes):
        super(ScannerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(128 * (INPUT_SIZE//8) * (INPUT_SIZE//8), 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        self.feature_maps = x
        self.feature_maps.retain_grad()   # <-- retain gradients for Grad-CAM
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
NUM_CLASSES = 4
class_names = ['Canon120', 'EpsonV39', 'EpsonV550', 'HP']
model = ScannerCNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('scanner_cnn_auto.pth', map_location=device))
model.eval()
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor.to(device), np.array(image)
def grad_cam(model, image_tensor, target_class=None):
    image_tensor.requires_grad = True
    outputs = model(image_tensor)
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()
    loss = outputs[0, target_class]
    
    model.zero_grad()
    loss.backward()

    gradients = model.feature_maps.grad
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    feature_maps = model.feature_maps.detach()[0]

    for i in range(feature_maps.shape[0]):
        feature_maps[i, :, :] *= pooled_gradients[i]

    heatmap = feature_maps.sum(dim=0).cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap, target_class
def show_gradcam(image_path):
    image_tensor, orig_image = preprocess_image(image_path)
    heatmap, pred_class = grad_cam(model, image_tensor)

    heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_image, 0.6, heatmap_color, 0.4, 0)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title(f"Original Image")
    plt.imshow(orig_image)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title(f"Grad-CAM: {class_names[pred_class]}")
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
def get_random_image(dataset_path):
    classes = [d for d in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not classes:
        raise ValueError(f"No class folders found in {dataset_path}")
    
    chosen_class = random.choice(classes)
    class_folder = os.path.join(dataset_path, chosen_class)
    images = []
    for root, _, files in os.walk(class_folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                images.append(os.path.join(root, f))
    
    if not images:
        raise ValueError(f"No image files found in {class_folder} or its subfolders")
    
    chosen_image = random.choice(images)
    return chosen_image
image_path = get_random_image(DATASET_PATH)
print(f"Selected image: {image_path}")
show_gradcam(image_path)
