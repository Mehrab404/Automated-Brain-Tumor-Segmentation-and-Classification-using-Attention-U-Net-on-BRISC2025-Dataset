# 🧠 Automated Brain Tumor Segmentation and Classification using Attention U-Net (BRISC2025)

## 📌 Project Overview

This project implements a multi-task Deep Learning framework for the automated analysis of Brain MRI scans using the **BRISC2025** dataset. It tackles two critical diagnostic challenges:

1. **Segmentation:** Precise pixel-wise delineation of tumor regions using **Attention U-Net**.
2. **Classification:** Categorizing tumors into **Glioma, Meningioma, Pituitary, or No Tumor** classes.

The project demonstrates that integrating **Attention Gates** into the U-Net architecture significantly improves segmentation performance by suppressing irrelevant background noise and focusing on tumor boundaries.

## 📂 Repository Structure

* `Report_Automated Brain Tumor Segmentation and Classification.pdf`: **Project Report**. Includes the project report and its findings.
* `Exploratory Data Analysis (EDA).ipynb`: **Exploratory Data Analysis**. Includes data loading, pixel intensity distribution analysis, class balance visualization, and sample MRI rendering.
* `Model_code.ipynb`: **Core Implementation**. Contains the PyTorch code for:
* **U-Net** & **Attention U-Net** architectures.
* Classification CNN backbone.
* Training loops with hybrid loss (Dice + BCE).
* Evaluation metrics and visualization of predicted masks.



## 🔬 Methodology

### 1. Segmentation (Attention U-Net)

We utilize an **Attention U-Net**, which modifies the standard U-Net by adding **Attention Gates (AGs)** at the skip connections.

* **Encoder:** Captures context features (downsampling).
* **Attention Gates:** Learns to weigh the importance of features from the encoder before passing them to the decoder, filtering out non-tumor regions.
* **Decoder:** Enables precise localization (upsampling).
* **Loss Function:** `0.2 * BCE + 0.8 * Dice` to handle class imbalance.

### 2. Classification

A parallel CNN branch classifies the input MRI slice into one of four categories using a Softmax output layer.

## 📊 Performance Results

| Model | Task | Metric | Score |
| --- | --- | --- | --- |
| **Attention U-Net** | Segmentation | **IoU (Intersection over Union)** | **0.7525** |
| **Attention U-Net** | Segmentation | Dice Score | 0.8585 |
| **CNN Classifier** | Classification | **Accuracy** | **97.30%** |

### Per-Class F1 Scores:

* **Glioma:** 96.99%
* **Meningioma:** 95.75%
* **Pituitary:** 98.94%
* **No Tumor:** 98.35%

## 🛠️ Installation & Requirements

To run this project locally or on Colab, ensure you have the following dependencies:

```bash
pip install torch torchvision tensorflow opencv-python matplotlib seaborn scikit-learn pandas kagglehub

```

## 🚀 How to Run

1. **Dataset Setup:** The notebook automatically downloads the BRISC2025 dataset via `kagglehub`.
2. **Run EDA:** Execute `Exploratory Data Analysis (EDA).ipynb` to visualize the dataset and understand class distributions.
3. **Train Models:** Open the Final Notebook. Run the cells sequentially to:
* Preprocess images (Resize to 256x256, Normalize).
* Train the Attention U-Net and Classifier.
* Visualize the segmentation masks against the Ground Truth.



---

*Created by Mehrabul Islam*
