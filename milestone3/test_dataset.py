from dataset import TraceFinderDataset

DATASET_ROOT = r"C:\Techie\Projects\TraceFinder_AI_Dataset"

dataset = TraceFinderDataset(DATASET_ROOT)

print("[INFO] Number of images:", len(dataset))
print("[INFO] Classes:", dataset.class_names)

img, label = dataset[0]
print("[INFO] Image shape:", img.shape)
print("[INFO] Label index:", label)
print("[INFO] Label name:", dataset.class_names[label])

