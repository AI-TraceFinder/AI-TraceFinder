import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from infer_document import predict_document
from dataset import TraceFinderDataset

# ---------------- CONFIG ----------------
DATASET_ROOT = r"C:\Techie\Projects\TraceFinder_AI_Dataset"
SPLIT = "Official"   # Official / Flatfield / Wikipedia
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- DATASET INFO ----------------
dataset = TraceFinderDataset(DATASET_ROOT)
class_names = dataset.class_names

y_true = []
y_pred = []
skipped = 0
total_docs = 0

print("[INFO] Evaluating DOCUMENT-LEVEL accuracy...")
print(f"[INFO] Split: {SPLIT}")
print(f"[INFO] Classes: {len(class_names)}")

# ---------------- EVALUATION LOOP ----------------
for cls_name in os.listdir(os.path.join(DATASET_ROOT, SPLIT)):
    cls_dir = os.path.join(DATASET_ROOT, SPLIT, cls_name)
    if not os.path.isdir(cls_dir):
        continue

    true_label = class_names.index(cls_name)

    # Each subfolder contains documents
    for doc_folder in os.listdir(cls_dir):
        folder_path = os.path.join(cls_dir, doc_folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.lower().endswith(".tif"):
                continue

            total_docs += 1
            doc_path = os.path.join(folder_path, file)

            try:
                pred_label, _, used = predict_document(doc_path)

                # Skip if no valid patches
                if pred_label is None or used == 0:
                    skipped += 1
                    continue

                y_true.append(true_label)
                y_pred.append(pred_label)

            except Exception as e:
                skipped += 1
                print(f"[WARN] Failed on {doc_path}: {e}")

# ---------------- METRICS ----------------
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)

print("\n================ RESULTS ================")
print(f"Total documents found : {total_docs}")
print(f"Documents evaluated   : {len(y_true)}")
print(f"Documents skipped     : {skipped}")
print(f"\nDocument Accuracy     : {acc*100:.2f}%")
print(f"Weighted F1 Score     : {f1:.3f}")
print("========================================")

# ---------------- CONFUSION MATRIX ----------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    xticklabels=class_names,
    yticklabels=class_names,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Document-Level Confusion Matrix")
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "document_confusion_matrix.png"))
plt.show()

