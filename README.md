# Breast Cancer Detection Using Deep Learning

This project focuses on detecting breast cancer by classifying tumors as **malignant** or **benign** using a deep learning model built with TensorFlow and Keras. The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**, utilizing diagnostic features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

---

## 🧠 Problem Statement

Early and accurate detection of breast cancer is critical to improving patient outcomes. This project aims to develop a machine learning model that can automatically predict whether a tumor is malignant or benign based on clinical measurements, thereby assisting in faster and more reliable diagnosis.

---

## 📂 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features**: 30 numeric diagnostic features (e.g., radius, texture, perimeter, area, smoothness, etc.)
- **Target**: 
  - `0` → Malignant  
  - `1` → Benign

---

## 🧪 Tech Stack

- Python
- NumPy, Pandas, Scikit-learn
- TensorFlow / Keras
- Matplotlib & Seaborn (for visualization)
- Google Colab / Jupyter Notebook

---

## 🚀 Model Architecture

- Input Layer: Flattens the 30 features into a 1D array.
- Hidden Layers: 20 neurons with ReLU activation for non-linearity.
- Output Layer: 2 neurons with sigmoid activation for binary classification (Benign or Malignant).
- Loss Function:sparse_categorical_crossentropy
- Optimizer: Adam
- Metrics: Accuracy

---

## 📈 Training Highlights

- Data standardization using `StandardScaler`
- Train-test split: 80/20
- Batch size: 32  
- Epochs: 10
- Validation Split: 20%
- Accuracy: High test accuracy on unseen data

---

## 🔍 Prediction

The model accepts 30 input features and outputs probabilities for both classes. Based on the highest confidence, it classifies the tumor and displays a clear diagnosis:

```python
The model predicts:
→ Benign tumor 
