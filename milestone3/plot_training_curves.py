import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = "training_logs.csv"

df = pd.read_csv(LOG_FILE)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df["epoch"], df["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.show()
