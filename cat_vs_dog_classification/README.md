# 🐱🐶 Cat vs Dog Classification (Transfer Learning)

A deep learning project that classifies images as **cat** or **dog** using **transfer learning** with **MobileNetV2** pretrained on ImageNet.

---

## 📋 Overview

Instead of training a CNN from scratch, this project leverages **MobileNetV2**'s pretrained feature extractor (frozen) and adds a lightweight classification head on top. The result: high accuracy on a small subset of data with minimal training time.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [Microsoft Cats vs Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) (Kaggle) |
| **Classes** | 2 (Cat, Dog) |
| **Subset used** | 1,000 cats + 1,000 dogs = 2,000 images |
| **Image size** | 224 × 224 × 3 (RGB) |
| **Split** | 80% train / 20% test |

---

## ⚙️ Requirements

```bash
pip install numpy matplotlib pillow opencv-python scikit-learn tensorflow kaggle
```

You'll also need a Kaggle API token (`kaggle.json`) to download the dataset.

---

## 🔁 Workflow

1. **Setup Kaggle API** — place `kaggle.json` in `~/.kaggle/` with proper permissions.
2. **Download Dataset** — fetch and unzip the Microsoft Cats vs Dogs dataset.
3. **Inspect Data** — count images per class and visualize a sample.
4. **Load Subset** — read 1,000 cats + 1,000 dogs using PIL, resize to 224×224, convert to RGB, skip corrupted files.
5. **Label Encoding** — Cat → 0, Dog → 1.
6. **Train/Test Split** — 80% train / 20% test (`random_state=2`).
7. **Normalize** — scale pixel values from `[0, 255]` to `[0, 1]`.
8. **Build Model:**
   - **MobileNetV2** (pretrained on ImageNet, frozen) as feature extractor
   - `GlobalAveragePooling2D` to flatten features
   - `Dense(1, sigmoid)` for binary output
9. **Compile** — Adam optimizer, binary crossentropy loss.
10. **Train** — 5 epochs on scaled training data.
11. **Evaluate** — accuracy on the test set.
12. **Predict** — classify a custom image loaded via OpenCV.

---

## 🤖 Model Architecture

```
MobileNetV2 (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dense(1, Sigmoid)
```

**Optimizer:** Adam
**Loss:** Binary Crossentropy
**Metric:** Accuracy

---

## 🚀 Usage

```bash
jupyter notebook cat_vs_dog_transfer_learning_.ipynb
```

1. Place `kaggle.json` in the working directory.
2. Run all cells to download data, train, and evaluate.
3. When prompted, enter a path to your own image for prediction.

---

## 📁 Project Structure

```
.
├── cat_vs_dog_transfer_learning_.ipynb   # Main notebook
├── kaggle.json                           # Kaggle API token
├── PetImages/
│   ├── Cat/
│   └── Dog/
└── README.md
```

---

## 📝 Notes

- **Prediction logic bug:** the final cell uses `np.argmax` on a single sigmoid output, which always returns `0`. Since the model uses `Dense(1, sigmoid)`, use a threshold instead:
  ```python
  input_pred_label = 1 if input_prediction[0][0] > 0.5 else 0
  ```
- **OpenCV reads BGR**, while the model was trained on RGB. Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before prediction for correct results.
- **Only 2,000 of ~25,000 images** are used to keep training fast. Use the full dataset (via `image_dataset_from_directory`, already set up) for production-grade accuracy.
- The dataset contains a few corrupted images — the `try/except` block handles them silently.
- For higher accuracy, **unfreeze** the top few MobileNetV2 layers and fine-tune with a low learning rate.

---

## 📄 License

Free to use for learning and personal projects.
