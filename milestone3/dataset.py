import os
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class TraceFinderDataset(Dataset):
    def __init__(self, dataset_root, patches_per_image=12):
        self.dataset_root = dataset_root
        self.modes = ["Flatfield", "Official", "Wikipedia"]
        self.patches_per_image = patches_per_image

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.samples = []
        self.class_names = []
        self.class_to_idx = {}

        self._build_index()

    def _build_index(self):
        all_classes = set()

        for mode in self.modes:
            mode_dir = os.path.join(self.dataset_root, mode)
            if not os.path.isdir(mode_dir):
                continue
            for cls in os.listdir(mode_dir):
                if os.path.isdir(os.path.join(mode_dir, cls)):
                    all_classes.add(cls)

        self.class_names = sorted(all_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for mode in self.modes:
            mode_dir = os.path.join(self.dataset_root, mode)
            if not os.path.isdir(mode_dir):
                continue
            for cls in os.listdir(mode_dir):
                cls_dir = os.path.join(mode_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                label = self.class_to_idx[cls]
                for root, _, files in os.walk(cls_dir):
                    for f in files:
                        if f.lower().endswith(".tif"):
                            self.samples.append((os.path.join(root, f), label))

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img_path, label = self.samples[img_idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # High-pass filter (scanner noise)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.subtract(img, blur)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w, _ = img.shape
        ps = 128
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)

        patch = img[y:y + ps, x:x + ps]
        patch = self.transform(patch)

        return patch, label












