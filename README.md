# Multi-Label Image Classification Using Vision Transformers

This project implements an end-to-end pipeline for multi-label image classification using Vision Transformers (ViTs). The model is designed to classify images into multiple classes (1-8) simultaneously, with each class independently predicted. The implementation is optimized for Apple Silicon GPUs (M1-Pro) and structured into modular components for data processing, model building, training, and evaluation.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Structure](#code-structure)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Acknowledgments](#acknowledgments)

---

## **Overview**

Multi-label image classification involves predicting multiple labels for a single image, where each label indicates the presence or absence of a specific class. This project leverages Vision Transformers (ViTs) to capture both local and global dependencies in the image.

Key highlights:
- **Multi-label Classification**: Each image can belong to one or more of eight classes.
- **Vision Transformer Architecture**: Customized ViT with 8 attention heads and a sigmoid activation layer for independent class predictions.
- **Optimized for Apple Silicon GPUs**: Utilizes the MPS backend for efficient training on Mac M1-Pro.

---

## **Features**

- **Custom Vision Transformer Model**:
  - 8 attention heads to capture patterns for each class.
  - Output layer with 8 neurons and sigmoid activation for multi-label classification.
  
- **Data Augmentation**:
  - Random horizontal flips, rotations, and color jittering during training.

- **Metrics**:
  - F1 score (macro, micro, and sample-based), precision, recall, and validation loss.

- **Visualization**:
  - Training history plots (loss and F1 score).
  - Class-wise probability visualization during inference.

---

## **Installation**

### Prerequisites
Ensure you have the following installed:
- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- NumPy
- Matplotlib
- scikit-learn

### Steps
1. Clone the repository:
git clone https://github.com/your-repo/multi-label-vit.git
cd multi-label-vit

2. Install dependencies:
```pip install -r requirements.txt```
