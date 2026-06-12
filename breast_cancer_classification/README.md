
# 🎗️ Breast Cancer Classification

A deep learning project that classifies breast tumors as **malignant** or **benign** using a neural network built with **TensorFlow / Keras** on the Wisconsin Breast Cancer dataset.

---

## 📋 Overview

This project trains a feedforward neural network on 30 cellular features extracted from digitized images of breast mass biopsies. The model learns to distinguish malignant from benign tumors and supports inference on new patient data.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `sklearn.datasets.load_breast_cancer` |
| **Samples** | 569 |
| **Features** | 30 (mean, standard error, and worst values of 10 cell nucleus measurements) |
| **Target** | `label` — `0` = malignant, `1` = benign |

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## 🔁 Workflow

1. **Load Data** — fetch the breast cancer dataset from scikit-learn.
2. **Build DataFrame** — convert to pandas with feature names and append `label` column.
3. **Explore** — inspect shape, info, class distribution, and group means.
4. **Split Features** — `X` = 30 features, `Y` = label.
5. **Train/Test Split** — 80% train / 20% test (`random_state=2`).
6. **Standardize** — apply `StandardScaler` (fit on train, transform on test).
7. **Build Model** — Sequential network:
   - `Flatten(30)` → `Dense(20, relu)` → `Dense(2, sigmoid)`
8. **Compile & Train** — Adam optimizer, sparse categorical crossentropy, 10 epochs, 10% validation split.
9. **Visualize** — plot training/validation accuracy and loss curves.
10. **Evaluate** — accuracy on the test set.
11. **Predict** — classify a single patient sample as malignant or benign.

---

## 🤖 Model Architecture

```
Flatten(30) → Dense(20, ReLU) → Dense(2, Sigmoid)
```

**Optimizer:** Adam
**Loss:** Sparse Categorical Crossentropy
**Metric:** Accuracy

---

## 🚀 Usage

```bash
jupyter notebook Untitled11.ipynb
```

Run all cells. The final cell demonstrates inference on a single example.

---

## 📁 Project Structure

```
.
├── Untitled11.ipynb   # Main notebook (consider renaming to breast_cancer_classification.ipynb)
└── README.md
```

---

## 📝 Notes

- **Rename the notebook** — `Untitled11.ipynb` is the default Colab name; something like `breast_cancer_classification.ipynb` is more meaningful.
- The output layer uses `sigmoid` with 2 units — `softmax` is conventionally preferred for multi-class classification.
- For only 569 samples, a smaller network (or classical models like SVM/Logistic Regression) often performs just as well with less risk of overfitting.
- Always standardize new inputs using the **fitted** scaler from training (already done correctly in the inference cell).

---

## 📄 License

Free to use for learning and personal projects.
