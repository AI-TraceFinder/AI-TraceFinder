# Milestone 1 – Dataset Setup & Preprocessing

## Overview

Milestone 1 establishes the **foundation of the AI-TraceFinder project** by focusing on dataset preparation and preprocessing validation. The primary objective of this milestone is to ensure that raw image data can be systematically organized, processed, and verified before advancing to feature extraction and model development stages.

This milestone demonstrates that the end-to-end data pipeline—from raw input acquisition to generation of processed outputs—is functional, reproducible, and aligned with the project’s objectives.

---

## Folder Structure

```
milestone1/
├── sample_input/
│   └── raw_sample.tif
├── sample_output/
│   └── processed_sample.tif
├── dataset_setup.py
```

---

## Milestone 1 Objectives

* Define a clear and scalable dataset directory structure
* Validate handling of raw `.tif` image files
* Implement a preprocessing script to transform raw images
* Verify correctness of preprocessing using sample input and output

---

## Work Completed

* Dataset directory structure finalized
* Preprocessing logic implemented in `dataset_setup.py`
* Raw input image successfully processed
* Output image generated and stored in the designated directory
* Pipeline tested to ensure consistency and correctness

---

## How to Run

```bash
python milestone1/dataset_setup.py
```

## Outcome

Milestone 1 confirms that the dataset and preprocessing pipeline are correctly set up and ready to support downstream tasks such as feature extraction, model training, and evaluation.

---

## Status

✅ Milestone 1 Completed


