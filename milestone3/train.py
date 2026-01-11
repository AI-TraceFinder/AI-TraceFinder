import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import time
import os

from dataloader import get_dataloaders
from model import TraceFinderCNN
from dataset import TraceFinderDataset

# ---------------------------
# CONFIG
# ---------------------------
DATASET_ROOT = r"C:\Techie\Projects\TraceFinder_AI_Dataset"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 5e-5   # slightly lower for safe resume

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# ---------------------------
# DATASET
# ---------------------------
ds = TraceFinderDataset(DATASET_ROOT)
print("[DEBUG] Total images:", len(ds.samples))
print("[DEBUG] Classes:", ds.class_names)

train_loader, val_loader = get_dataloaders(
    DATASET_ROOT,
    batch_size=BATCH_SIZE
)

# ---------------------------
# MODEL
# ---------------------------
model = TraceFinderCNN(num_classes=len(ds.class_names)).to(device)

# Freeze early layers
for name, param in model.model.named_parameters():
    if any(k in name for k in ["layer2", "layer3", "layer4", "fc"]):
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# ---------------------------
# LOSS (CLASS WEIGHTS + LABEL SMOOTHING)
# ---------------------------
labels = [label for _, label in ds.samples]
counts = Counter(labels)
total = sum(counts.values())

class_weights = [total / counts[i] for i in range(len(counts))]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

# ---------------------------
# SCHEDULER
# ---------------------------
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.3,
    patience=3
)

# ---------------------------
# RESUME LOGIC
# ---------------------------
start_epoch = 0
best_val_acc = 0.0

if os.path.exists(CHECKPOINT_PATH):
    print("[INFO] Resuming from checkpoint")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)

    print(f"[INFO] Resumed from epoch {start_epoch}")
    print(f"[INFO] Best Val Acc so far: {best_val_acc:.2f}%")
else:
    print("[INFO] No checkpoint found. Training from scratch.")

# ---------------------------
# TRAINING LOOP
# ---------------------------
try:
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Training started")

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()
        correct = 0
        total_val = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                total_val += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = 100.0 * correct / total_val
        scheduler.step(val_acc)

        epoch_time = (time.time() - start_time) / 60.0

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f} min"
        )

        # ---------------------------
        # SAVE CHECKPOINT
        # ---------------------------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc
        }, CHECKPOINT_PATH)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict()
            }, BEST_MODEL_PATH)
            print(f"[SAVE] New best model saved ({best_val_acc:.2f}%)")

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Training stopped safely.")

print("\n[OK] Training completed.")
print(f"[OK] Best validation accuracy: {best_val_acc:.2f}%")








