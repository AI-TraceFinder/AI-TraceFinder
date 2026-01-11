import torch
import cv2
from torchvision import transforms
from model import TraceFinderCNN
from dataset import TraceFinderDataset

DATASET_ROOT = r"C:\Techie\Projects\TraceFinder_AI_Dataset"
MODEL_PATH = "tracefinder_resnet18.pth"
IMAGE_PATH = r"sample.tif"   # change this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset for class names
dataset = TraceFinderDataset(DATASET_ROOT)
class_names = dataset.class_names

model = TraceFinderCNN(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1).item()

print("Predicted class:", class_names[pred])
