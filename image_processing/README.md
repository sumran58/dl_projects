# 🖼️ Image Processing Basics

A beginner-friendly notebook covering essential image processing operations in Python using **Matplotlib**, **PIL**, and **OpenCV**.

---

## 📋 Overview

This notebook walks through the fundamental image operations every computer vision project relies on: loading images, inspecting their array structure, resizing, converting color spaces, and saving the result back to disk — using three of the most common Python imaging libraries side by side.

---

## ⚙️ Requirements

```bash
pip install numpy matplotlib pillow opencv-python
```

---

## 🔁 Workflow

1. **Download Image** — fetch a sample dog image with `wget`.
2. **Load with Matplotlib** — `mpimg.imread()` returns a NumPy array (height × width × channels).
3. **Inspect** — print shape, type, and pixel values.
4. **Display** — render the image inline with `plt.imshow()`.
5. **Resize with PIL** — open with `Image.open()`, resize to 100×100, and save.
6. **Reload & Verify** — load the resized image and confirm dimensions.
7. **Convert to Grayscale with OpenCV** — read with `cv2.imread()`, convert via `cv2.cvtColor(BGR2GRAY)`.
8. **Save Output** — write grayscale image with `cv2.imwrite()`.

---

## 🛠️ Libraries Compared

| Library | Used For |
|---|---|
| **Matplotlib** | Reading images as arrays, inline display in notebooks |
| **PIL (Pillow)** | Image objects, resizing, format conversion, saving |
| **OpenCV (cv2)** | Color space conversion, computer vision operations |

---

## 🚀 Usage

```bash
jupyter notebook image_processing.ipynb
```

Run cells top to bottom. The `wget` cell downloads the sample image; replace the URL or path to use your own.

---

## 📁 Project Structure

```
.
├── image_processing.ipynb   # Main notebook
├── dog.jpg                  # Sample image (downloaded)
├── dog_image_resize.jpg     # Resized output
├── dog_grascale.jpg         # Grayscale output
└── README.md
```

---

## 📝 Notes

- **OpenCV uses BGR**, not RGB. Display with `cv2_imshow` (Colab) or convert via `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before using `plt.imshow()`.
- `cv2_imshow` is **Colab-only** — for local runs use `cv2.imshow()` followed by `cv2.waitKey(0)`.
- The downloaded image filename in the `wget` cell may not match `dog.jpg` automatically; rename or use the `-O dog.jpg` flag.

---

## 📄 License

Free to use for learning and personal projects.
