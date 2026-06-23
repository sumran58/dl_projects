# Fashion Item Classifier

A Streamlit web app that classifies clothing images into 10 categories using a trained CNN on the Fashion MNIST dataset.

## Demo

Upload any clothing image (JPG/PNG) and the model predicts which of 10 fashion categories it belongs to.

## Classes

`T-shirt/top` · `Trouser` · `Pullover` · `Dress` · `Coat` · `Sandal` · `Shirt` · `Sneaker` · `Bag` · `Ankle boot`

## Tech Stack

- **TensorFlow / Keras** — CNN model (trained on Fashion MNIST)
- **Streamlit** — web interface
- **Pillow** — image loading and preprocessing
- **NumPy** — array operations

## Project Structure

```
fashion_mnist/
├── app.py                          # Streamlit app
├── trained_model/
│   └── trained_fashion_mnist_model.h5   # Trained Keras model
├── requirements.txt
└── README.md
```

## Setup

Use a virtual environment to avoid dependency conflicts:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

1. User uploads an image via the file uploader.
2. The image is resized to 28×28, converted to grayscale, normalized to `[0, 1]`, and reshaped to `(1, 28, 28, 1)`.
3. The CNN predicts class probabilities; `argmax` selects the label.
4. The predicted category is displayed.

## requirements.txt

```
streamlit
tensorflow
pillow
numpy
```

## Notes

- The model file (`.h5`) must be saved and loaded with **compatible Keras versions**. Models saved with Keras 3.x include a `quantization_config` field that older Keras cannot deserialize — keep training and inference environments on the same major version.
- For best results, upload images with a plain background and the item centered, similar to the Fashion MNIST style.
