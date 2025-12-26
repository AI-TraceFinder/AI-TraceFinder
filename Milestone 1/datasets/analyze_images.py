import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "image_labels.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
REPORT_FILE = os.path.join(REPORT_DIR, "dataset_summary_by_scanner.csv")

os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

summary = df.groupby("scanner_model").agg(
    total_images=("file_name", "count"),
    avg_width=("width", "mean"),
    avg_height=("height", "mean")
).reset_index()

summary.to_csv(REPORT_FILE, index=False)

print("✅ dataset_summary_by_scanner.csv created successfully")
print(summary)
