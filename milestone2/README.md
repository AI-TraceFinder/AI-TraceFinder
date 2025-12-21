# Milestone 2 – Feature Extraction

## Overview

Milestone 2 builds on the dataset and preprocessing pipeline established in Milestone 1. The focus of this milestone is **extracting meaningful features from preprocessed images** and organizing them in a structured format suitable for machine learning models.

This milestone validates that the project has moved beyond raw data handling and into **quantitative representation of image content**.

---

## Folder Structure

```
milestone2/
├── feature_extraction.py
├── utils.py            (if applicable)
├── features.csv        (generated, ignored by git)
```

---

## Objectives

* Load preprocessed images from the dataset
* Extract relevant numerical features from images
* Aggregate extracted features into a structured format
* Prepare data for downstream model training and evaluation

---

## Work Completed

* Implemented feature extraction pipeline
* Computed image-level features from processed images
* Stored extracted features in a CSV file for analysis
* Verified correctness and consistency of extracted features

---

## Output

* **features.csv**: Tabular representation of extracted image features (ignored in version control)

---

## How to Run

```bash
python milestone2/feature_extraction.py
```

---

## Outcome

Milestone 2 confirms that image data can be transformed into machine-learning-ready numerical features, enabling the next stage of the project: model development and experimentation.

---

## Status

✅ Milestone 2 Completed
