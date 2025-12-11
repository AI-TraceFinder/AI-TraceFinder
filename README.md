AI-TraceFinder is a forensic machine learning system designed to identify the source scanner device used to scan a document or image.

Every scanner introduces unique patterns such as noise, texture, and frequency artifacts.
This project uses these patterns to detect which scanner model produced a scanned image.

This system is useful in:

Digital forensics

Document authentication

Legal evidence verification

Objectives:

Identify the source scanner from scanned document images

Extract scanner-specific artifacts such as noise, PRNU, texture, and FFT signatures

Train ML/CNN models to classify scanner brands/models

Visualize model accuracy, confusion matrix, and important features

Build a simple UI to upload an image and get prediction
