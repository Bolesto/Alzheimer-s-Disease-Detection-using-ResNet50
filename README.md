# Alzheimer-s-Disease-Detection-using-ResNet50
This repository contains my project for CET313 – Artificial Intelligence at the University of Sunderland. The project applies deep learning for medical image classification, specifically detecting Alzheimer’s Disease from MRI scans using a pretrained ResNet50 model.

# Project Overview
Goal: Build a deep learning pipeline to classify brain MRI scans as demented or non-demented.

Dataset: OASIS (Open Access Series of Imaging Studies)
 – over 80,000 MRI samples.

Model: Pretrained ResNet50 CNN I tuned for MRI slices.

Platform: Training and experimentation performed on Google Colab Pro with an NVIDIA A100 GPU.

# Methodology
1-Data Preprocessing
  Image resizing → 224x224 to match ResNet50 input.
  Grayscale conversion.
  Normalisation of pixel values.
  Train/test split with class balancing techniques.

2-Handling Class Imbalance
  Undersampling of majority class (non-demented).
  Augmentation of minority classes (mild/moderate dementia).
  Final ratio improved from 1:137 → 1:3, reducing bias.

3-Model Training
  Pretrained ResNet50 from Torchvision.
  Loss functions tested (class-weighted vs default).
  Final training: batch size 256, ~30 minutes per run on A100 GPU (initial 4 hours).
  
4-Evaluation Metrics
  Training/validation accuracy per epoch.
  Confusion matrix.
  Classification report (precision, recall, F1-score).

# Results
Validation Accuracy: >99%
Errors: Misclassified only 12 samples out of 3,784 in the final run.

# Key Insights:
Preprocessing and class balancing were critical.
Google Colab hardware (A100 GPU) drastically reduced training time.
Future work → testing on more balanced datasets and comparing ResNet50 vs other CNN/INN architectures.

# Repository Structure
├── CET313_AI_Assignment_Bi11ca.ipynb   # Jupyter Notebook with full pipeline
├── CET313_Report.docx                  # Detailed report
├── README.md                           # Project documentation

# How to Run
Clone the repo:git clone https://github.com/<your-username>/Alzheimers-Detection-ResNet50.git
cd Alzheimers-Detection-ResNet50
Open the Jupyter Notebook in Google Colab or locally.
Ensure the dataset is downloaded from Kaggle OASIS MRI Dataset (Link: https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
and paths are updated in the notebook.
Run all cells to train and evaluate the model.


# Tech Stack
Python
PyTorch
Torchvision
Google Colab Pro
Matplotlib / Seaborn (visualization)


# Author

Belal Ghonem
📧 Email: belalghonem007@gmail.com
💼 LinkedIn: [Your Profile Link]
🎓 B.Sc. Computer Science, University of Sunderland
