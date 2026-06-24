# 🌿 Plant Disease Prediction (CNN)

A deep learning project that classifies plant leaf images into **38 disease categories** (across multiple crops) using a custom Convolutional Neural Network built with TensorFlow / Keras.

---

## 📋 Overview

This project trains a CNN from scratch on the **PlantVillage** dataset to identify plant diseases from leaf images. It supports inference on new images and saves both the trained model and class label mappings for downstream deployment.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (Kaggle) |
| **Variants** | `color`, `grayscale`, `segmented` (this project uses `color`) |
| **Classes** | 38 (healthy + diseased leaves across crops like Apple, Tomato, Potato, Corn, Grape, etc.) |
| **Image size (after resize)** | 224 × 224 × 3 (RGB) |
| **Split** | 80% train / 20% validation |

---

## ⚙️ Requirements

```bash
pip install numpy matplotlib pillow tensorflow kaggle
```

You'll also need a Kaggle API token (`kaggle.json`) to download the dataset.

---

## 🔁 Workflow

1. **Set Random Seeds** — fix `random`, `numpy`, and `tensorflow` seeds for reproducibility.
2. **Setup Kaggle API** — load credentials from `kaggle.json` and configure environment variables.
3. **Download & Extract** — fetch and unzip the PlantVillage dataset.
4. **Inspect Data** — list folders for `color`, `grayscale`, and `segmented` variants.
5. **Configure** — set `img_size = 224`, `batch_size = 32`, base directory to `color`.
6. **Data Generator** — `ImageDataGenerator` with `rescale=1./255` and `validation_split=0.2`.
7. **Build Generators** — `flow_from_directory` for training and validation subsets.
8. **Build CNN** — stacked Conv2D + MaxPooling layers, Flatten, Dense head.
9. **Compile & Train** — Adam optimizer, categorical crossentropy, 5 epochs.
10. **Evaluate** — validation accuracy and loss.
11. **Visualize** — plot accuracy and loss curves for train vs. validation.
12. **Save Class Indices** — export `class_indices.json` mapping index → class name.
13. **Inference Helpers** — `load_and_preprocess_image()` and `predict_image_class()` functions.
14. **Predict** — classify a sample leaf image.
15. **Save Model** — export trained model as `plant_disease_prediction_model.h5`.

---

## 🤖 Model Architecture

```
Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(256, ReLU)
    ↓
Dense(38, Softmax)
```

**Optimizer:** Adam
**Loss:** Categorical Crossentropy
**Metric:** Accuracy

---

## 🚀 Usage

### Train

```bash
jupyter notebook plant_disease_prediction_cnn.ipynb
```

1. Place `kaggle.json` in the working directory.
2. Run all cells to download data, train, and evaluate.
3. The notebook saves `plant_disease_prediction_model.h5` and `class_indices.json`.

### Predict on a new image

```python
from tensorflow.keras.models import load_model
import json

model = load_model('plant_disease_prediction_model.h5')
class_indices = json.load(open('class_indices.json'))
class_indices = {int(k): v for k, v in class_indices.items()}

predicted = predict_image_class(model, 'leaf.jpg', class_indices)
print("Predicted:", predicted)
```

---

## 📁 Project Structure

```
.
├── plant_disease_prediction_cnn.ipynb   # Main notebook
├── kaggle.json                          # Kaggle API token
├── plantvillage dataset/
│   ├── color/
│   ├── grayscale/
│   └── segmented/
├── plant_disease_prediction_model.h5    # Saved model
├── class_indices.json                   # Index → class name mapping
└── README.md
```

---

## 📝 Notes

- **`!cat ~/.kaggle/kaggle.json`** is in the notebook — remove this before pushing to GitHub. It will leak your Kaggle API key publicly.
- **5 epochs is light** for 38 classes — extending to 15–25 epochs (with `EarlyStopping`) typically improves accuracy significantly.
- **No data augmentation** — adding `rotation_range`, `horizontal_flip`, `zoom_range`, etc. in `ImageDataGenerator` would improve generalization.
- **Consider transfer learning** — MobileNetV2, EfficientNetB0, or ResNet50 pretrained on ImageNet would likely give 95%+ accuracy with fewer epochs.
- The `color` variant is used; experimenting with `segmented` (background-removed) may help the model focus on leaf features only.
- Saving as `.h5` works, but the newer Keras `.keras` format is recommended for newer TensorFlow versions.

---

## 📄 License

Free to use for learning and personal projects.
