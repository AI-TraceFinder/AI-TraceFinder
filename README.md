# 🔍 AI-TraceFinder – Scanner Source Identification  
### *By Nageswari Mettukuru*  

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-Private-red?style=for-the-badge)

---

## ✨ **Project Overview**

AI-TraceFinder is a digital forensics project focused on **identifying the source scanner** (Canon, Epson, HP, etc.) based on the noise signatures and texture patterns present in scanned document images.

Each scanner leaves a **unique noise fingerprint**, and this project aims to capture, analyze, and classify those fingerprints using image processing + machine learning.

This repository contains all work related to **Milestones 1–3**.

---

## 📁 **Repository Structure**

📦 Project Root
│── 📄 README.md
│── 📄 .gitignore
│
├── 📂 Milestone-1
│ ├── 📄 Milestone-1 Report.pdf
│ ├── 🧪 check_grayscale.py
│ ├── 🏷️ generate_labels.py
│ ├── 📊 labels.csv
│ ├── ⚙️ preprocess_images.py
│ ├── 🧹 verify_week2.py
│ ├── 📂 sample output images
│ ├── 🖼️ input image.png
│ ├── 🖼️ output image.png
│
├── 📂 Datasets (ignored in Git)
├── 📂 Processed (ignored in Git)



---

## 🧩 **Milestone-1 Summary: Dataset Labeling & Preprocessing**

Milestone-1 focuses on preparing the dataset for modeling:

### ✔️ **Tasks Completed**
- Organized raw datasets into scanner-specific folders  
- Generated **labels.csv** with correct class labels  
- Converted all images into **512×512 grayscale**  
- Removed corrupted & hidden macOS files (like `._abc.tif`)  
- Created a **Processed/** directory with clean standardized data  
- Verified preprocessing consistency  

---

## 🖼️ **Milestone-1 Sample Input & Output Images**

### 📥 **Input Image (Original Scanned Document)**
![Input Image](Milestone-1/sample%20output%20images/input%20image.png)

### 📤 **Output Image (After Grayscale + Resize Preprocessing)**
![Output Image](Milestone-1/sample%20output%20images/output%20image.png)

---

## 🚀 **Milestones Roadmap**

### ✅ **Milestone-1 — Preprocessing & Labeling**
- Dataset structuring  
- Label generation  
- Standardized grayscale + resizing  
- Preprocessing validation  

### 🔄 **Milestone-2 — Feature Extraction (Coming Next)**
- Noiseprint extraction  
- Spatial / frequency domain features  
- Texture descriptors (LBP, GLCM, Wavelets)  

### 🔮 **Milestone-3 — Classification Model**
- Train ML/DL models  
- Evaluate accuracy per scanner  
- Build final inference pipeline  

---

## 🛠️ **Technologies Used**
- Python  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- Git & GitHub  

---

## 📊 **GitHub Stats**

![Stats](https://github-readme-stats.vercel.app/api?username=Nageswari00530&show_icons=true&theme=radical)
![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=Nageswari00530&layout=compact&theme=radical)

---

## 🤝 **Contributions**
This is a private academic project under the **AI-TraceFinder** organization.  
Contributions are internal only.

---

## 💬 **Contact**
📧 Email: *nageswarimettukuru606@gmail.com*  
📌 GitHub: [Nageswari00530](https://github.com/Nageswari00530)

---

⭐ *If you like this project, don’t forget to Star the repository!* ⭐
