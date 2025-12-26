🎯 Milestone 1: Dataset Collection & Analysis
Objective

The objective of Milestone 1 is to collect scanned document images from multiple scanner sources, generate a labeled dataset, and analyze basic image properties required for forensic scanner identification.

📂 Project Structure
TraceFinder/
│
├─ images/                       # Scanned document images
├─ datasets/
│   ├─ create_labels.py          # Script to generate labeled CSV
│   ├─ analyze_images.py         # Script to analyze image properties
│
├─ data/
│   └─ image_labels.csv          # Generated labeled dataset
│
├─ reports/
│   └─ dataset_summary_by_scanner.csv   # Scanner-wise summary report
│
└─ README.md

📑 Dataset Description

The dataset consists of scanned document images collected from multiple scanner devices (minimum 3–5 models).
The following categories are included:

Official document scans

Wikipedia document scans

Tampered images

Flatfield images

Original documents

These images represent scanned outputs required for forensic scanner identification.

🏷️ Labeled Dataset

The script create_labels.py automatically generates a labeled dataset with the following attributes:

Scanner model

File name

Image path

Category

Image width and height

Color mode

Image format

DPI (if available)

File size

MD5 checksum

The output file is saved as:

data/image_labels.csv

📊 Image Property Analysis

The script analyze_images.py performs basic dataset analysis, including:

Total number of images

Number of unique scanner models

Average image width and height per scanner

The analysis results are saved as:

reports/dataset_summary_by_scanner.csv

▶️ How to Run (Windows)
Step 1: Open Command Prompt
cd "D:\Infosys Intern\Milestone 1\TraceFinder"

Step 2: Generate labeled dataset
python datasets\create_labels.py

Step 3: Analyze dataset
python datasets\analyze_images.py

✅ Output Files Generated

data/image_labels.csv

reports/dataset_summary_by_scanner.csv

📌 Milestone 1 Status

✔ Dataset collected
✔ Dataset labeled
✔ Image properties analyzed
✔ Summary report generated     

