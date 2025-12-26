🎯 Milestone 2: Image Preprocessing & Feature Extraction
Objective

The objective of Milestone 2 is to preprocess scanned document images and extract scanner-specific features that can be used for forensic scanner identification. This milestone focuses on preparing the dataset for machine learning–based classification.

📂 Folder Structure
TraceFinder/
│
├─ images/                     # Sample scanned images
├─ processed_images/           # Preprocessed grayscale images
├─ noise_images/               # Noise residual images
│
├─ preprocessing/
│   └─ preprocess_images.py    # Image preprocessing script
│
├─ features/
│   ├─ noise_extraction.py     # Noise residual extraction
│   └─ lbp_features.py         # LBP feature extraction
│
├─ feature_data/
│   └─ lbp_features.csv        # Extracted feature vectors
│
└─ README_Milestone_2.md

🔧 Technologies Used

Python 3.10

OpenCV

NumPy

scikit-image

Pandas

🟢 Step 1: Image Preprocessing

Each scanned image is:

Converted to grayscale

Resized to 256 × 256 pixels

Normalized to ensure uniformity

📁 Output directory:

processed_images/

Command:
python preprocessing/preprocess_images.py

🟢 Step 2: Noise Residual Extraction

Scanner-specific noise patterns are extracted by:

Applying Gaussian blur

Subtracting the blurred image from the original

This isolates intrinsic scanner artifacts.

📁 Output directory:

noise_images/

Command:
python features/noise_extraction.py

🟢 Step 3: Texture Feature Extraction (LBP)

Local Binary Pattern (LBP) is used to capture texture information from noise residual images:

LBP histograms are computed for each image

Features are normalized and stored as numerical vectors

📁 Output file:

feature_data/lbp_features.csv

Command:
python features/lbp_features.py

📊 Output Description

processed_images/ – Preprocessed grayscale images

noise_images/ – Extracted scanner noise residuals

lbp_features.csv – Final feature dataset for ML models

Each row in the CSV represents:

Scanner label

Corresponding LBP histogram features

✅ Milestone 2 Status

✔ Image preprocessing completed
✔ Noise residual extraction completed
✔ LBP texture features extracted
✔ Feature dataset generated

🎉 Milestone 2 successfully completed

🔜 Next Milestone

Milestone 3 will focus on:

Machine learning model training

Scanner classification

Performance evaluation

👤 Author

Venkatajalapathi S
TraceFinder Project