# 😷 Face Mask Detection

A deep learning project that classifies images as **with mask** or **without mask** using a custom Convolutional Neural Network (CNN) built with TensorFlow / Keras.

---

## 📋 Overview

This project trains a CNN from scratch on a labeled dataset of faces with and without masks. The model learns spatial features (edges, textures, mask shapes) through stacked convolutional layers and predicts whether a person in a new image is wearing a mask.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) (Kaggle) |
| **With mask** | 3,725 images |
| **Without mask** | 3,828 images |
| **Total** | 7,553 images |
| **Image size (after resize)** | 128 × 128 × 3 (RGB) |
| **Split** | 80% train / 20% test |

**Labels:** `with_mask` → 1, `without_mask` → 0

---

## ⚙️ Requirements

```bash
pip install numpy matplotlib pillow opencv-python scikit-learn tensorflow kaggle
```

You'll also need a Kaggle API token (`kaggle.json`) to download the dataset.

---

## 🔁 Workflow

1. **Setup Kaggle API** — place `kaggle.json` in `~/.kaggle/` with proper permissions.
2. **Download & Extract** — fetch and unzip the Face Mask dataset.
3. **Load Filenames** — list images in `with_mask/` and `without_mask/` directories.
4. **Create Labels** — assign `1` to with-mask, `0` to without-mask.
5. **Image Preprocessing:**
   - Open each image with PIL
   - Resize to 128 × 128
   - Convert to RGB
   - Convert to NumPy array
6. **Build Arrays** — combine all images into `X` and labels into `y`.
7. **Train/Test Split** — 80% train / 20% test (`random_state=2`).
8. **Normalize** — scale pixel values from `[0, 255]` to `[0, 1]`.
9. **Build CNN** — stacked Conv2D + MaxPooling layers with Dropout for regularization.
10. **Compile & Train** — Adam optimizer, 5 epochs with 10% validation split.
11. **Evaluate** — accuracy and loss on the test set.
12. **Visualize** — plot training/validation loss and accuracy curves.
13. **Predict** — classify a custom image loaded via OpenCV.

---

## 🤖 Model Architecture

```
Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(128, ReLU) → Dropout(0.5)
    ↓
Dense(64, ReLU) → Dropout(0.5)
    ↓
Dense(2, Sigmoid)
```

**Optimizer:** Adam
**Loss:** Sparse Categorical Crossentropy
**Metric:** Accuracy

---

## 🚀 Usage

```bash
jupyter notebook face_mask_detection.ipynb
```

1. Place `kaggle.json` in the working directory.
2. Run all cells to download data, train, and evaluate.
3. When prompted, enter a path to your own image for prediction.

---

## 📁 Project Structure

```
.
├── face_mask_detection.ipynb   # Main notebook
├── kaggle.json                 # Kaggle API token
├── data/
│   ├── with_mask/
│   └── without_mask/
└── README.md
```

---

## 📝 Notes

- **Output activation mismatch:** the final layer uses `sigmoid` with 2 units, but `softmax` is the standard choice for multi-class classification with `sparse_categorical_crossentropy`. Switching to `softmax` typically gives better-calibrated probabilities.
- **OpenCV reads BGR**, but the model was trained on RGB. Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before prediction for accurate results.
- **Hardcoded label counts** (`3725`, `3828`) — fragile if the dataset changes. Use `len(with_mask_files)` and `len(without_mask_files)` instead.
- **5 epochs may underfit** — try 10–15 epochs with `EarlyStopping` for better accuracy.
- **Memory-heavy** — loading 7,553 images at 128×128×3 as a single NumPy array uses ~1.2GB RAM. For larger datasets, use `tf.data.Dataset` with batched loading.
- For real-time detection, pair this classifier with a face detector (e.g., Haar Cascade, MTCNN, or YOLO).

---

## 📄 License

Free to use for learning and personal projects.
