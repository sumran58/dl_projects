# 🔢 MNIST Digit Classification

A deep learning project that classifies handwritten digits (0–9) using a fully connected neural network built with **TensorFlow / Keras**, with custom image inference via OpenCV.

---

## 📋 Overview

This project trains a feedforward neural network on the classic **MNIST** dataset of 28×28 grayscale handwritten digits, evaluates it on a held-out test set, visualizes performance with a confusion matrix, and runs inference on a custom image loaded from disk.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `keras.datasets.mnist` |
| **Training samples** | 60,000 |
| **Test samples** | 10,000 |
| **Image size** | 28 × 28 (grayscale) |
| **Classes** | 10 (digits 0–9) |

---

## ⚙️ Requirements

```bash
pip install numpy matplotlib seaborn opencv-python pillow tensorflow
```

---

## 🔁 Workflow

1. **Load Data** — fetch MNIST via `keras.datasets`.
2. **Visualize** — plot a sample image with its label.
3. **Normalize** — scale pixel values from `[0, 255]` to `[0, 1]`.
4. **Build Model** — Sequential network:
   - `Flatten(28×28)` → `Dense(50, relu)` → `Dense(50, relu)` → `Dense(10, sigmoid)`
5. **Compile** — Adam optimizer, sparse categorical crossentropy loss.
6. **Train** — 10 epochs with 10% validation split.
7. **Evaluate** — accuracy on the test set.
8. **Predict** — convert probabilities to labels via `argmax`.
9. **Visualize Results** — confusion matrix as a Seaborn heatmap.
10. **Custom Image Inference** — load `MNIST_digit.png` with OpenCV, convert to grayscale, resize to 28×28, normalize, reshape, and predict.

---

## 🤖 Model Architecture

```
Flatten(28×28) → Dense(50, ReLU) → Dense(50, ReLU) → Dense(10, Sigmoid)
```

**Optimizer:** Adam
**Loss:** Sparse Categorical Crossentropy
**Metric:** Accuracy

---

## 🚀 Usage

```bash
jupyter notebook mnist_digit_classification.ipynb
```

To test on your own digit, save it as `MNIST_digit.png` (white digit on dark background, ideally) and run the inference cell.

---

## 📁 Project Structure

```
.
├── mnist_digit_classification.ipynb   # Main notebook
├── MNIST_digit.png                    # Optional custom test image
└── README.md
```

---

## 📝 Notes

- The output layer uses `sigmoid` — `softmax` is more appropriate for multi-class classification and usually gives better calibrated probabilities.
- Custom images should match MNIST's style: white digit on black background, centered. The current pipeline doesn't invert colors, so dark-on-light input will misclassify.
- For higher accuracy (~99%+), consider a CNN architecture (`Conv2D` + `MaxPooling2D` layers).

---

## 📄 License

Free to use for learning and personal projects.
