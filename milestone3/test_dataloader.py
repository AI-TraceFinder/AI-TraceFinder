from dataloader import get_dataloader

DATASET_ROOT = r"C:\Techie\Projects\TraceFinder_AI_Dataset"

loader = get_dataloader(
    dataset_root=DATASET_ROOT,
    mode=None,        # 👈 IMPORTANT
    batch_size=2
)

for images, labels in loader:
    print("[INFO] Batch image shape:", images.shape)
    print("[INFO] Batch labels:", labels)
    break
