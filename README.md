# 🐦 Bird Species Classification using CNN and K-Fold Cross-Validation

This project implements an efficient and scalable deep learning pipeline for **classifying bird species** from images. The model is trained using **5-Fold Stratified Cross-Validation** to ensure robustness and generalization.

---

## 📌 Features

- 🔍 **Stratified K-Fold Cross-Validation** for balanced performance evaluation
- 🧠 **Custom Efficient CNN** architecture with < 40M parameters
- 🧪 **Smart training strategy** with early stopping, adaptive learning rate, and dropout
- 🧾 **Comprehensive classification report** and confusion matrix
- 📈 **Training/validation history tracking** with plots

---

## 🗂️ Project Structure

```plaintext
Bird-Classification/
├── DATA/ # Directory for training data
├── col671-a3-part1.ipynb # Main notebook (model training and evaluation)
└── README.md # Project documentation
```


---

## 🧪 Dataset

- Dataset used: **[Kaggle Birds Dataset](https://www.kaggle.com/competitions/identify-the-birds/data)**
- Folder format: `train/<class_name>/<image_files>`
- Classes: 10 bird species, inferred from folder names

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

---

## 🏗️ Model Architecture
Custom CNN with:
- 5 convolutional blocks (Conv-BatchNorm-ReLU-MaxPool)
- AdaptiveAvgPooling to (7x7)
- Fully connected layers with dropout
- Total Trainable Parameters: 35.4M

---

## 🚀 Training Pipeline

- Input image size: 224x224
- Data augmentations: random crop, flip, rotation, color jitter
- Optimizer: AdamW with weight decay
- Learning rate scheduler: ReduceLROnPlateau
- Early stopping: patience of 5 epochs
- Final evaluation includes confusion matrix and classification report
  
---

## 📊 Performance (5-Fold CV)

- Overall Accuracy: 85.71%
- Precision: 86%
- Recall: 85%
- F1-score: 85%

---
