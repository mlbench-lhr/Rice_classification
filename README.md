# 🌾 GrainVision PRO — Rice Grain Analysis

GrainVision PRO is an AI-powered rice grain analysis tool that automatically detects, measures, and classifies rice grains from a single photo. It combines instance segmentation, CNN-based classification, and coin-based real-world measurement calibration to deliver detailed quality and variety reports through an interactive web interface.

Upload a photo of rice grains next to a 5 PKR coin, and GrainVision PRO will:

- Detect and outline every individual grain in the image
- Measure each grain's length and width in millimetres
- Classify each grain's quality as **Full**, **Broken**, **Rejected**, **Weak**, or **Fatty**
- Further sub-classify broken grains by size (**SG**, **B1**, **B2**, **G1**)
- Identify the rice **variety** of full grains (1121, C9, 386, Supri, Super Kernel, Super Fine, etc.)
- Generate color-coded visualizations, distribution charts, and downloadable summary tables

---

## ✨ Features

- **Automatic grain segmentation** using Cellpose-SAM
- **Real-world mm measurements** calibrated via a 5 PKR reference coin (no manual scale needed)
- **5-class quality classification** (Full / Broken / Rejected / Weak / Fatty) via a DINOv2 + MLP classifier
- **Broken grain sub-classification** (SG / B1 / B2 / G1) based on relative grain length
- **7-class variety classification** for full grains (optional second model)
- **Two analysis modes**, each triggered by its own button in the UI:
  - 🔍 **Quality Test** — full quality grading + measurements
  - 🌾 **Full Grain Types** — variety identification for full grains only
- **Interactive results viewer** — zoom, pan, fullscreen, and toggle between the classified overlay and the original photo
- **Downloadable outputs** — per-grain measurement table, summary table (with Urdu translation), distribution charts, and result images
- **HEIC/HEIF support** for iPhone photos
- **GPU acceleration, multi-threading, and batched inference with Test-Time Augmentation (TTA)** for fast, robust predictions
- **Sample image gallery** for quick testing without uploading your own photo

---

## 🧠 How It Works

1. **Preprocessing** — the uploaded photo is cleaned up (contrast/sharpness enhancement) and, if needed, converted from HEIC/HEIF.
2. **Segmentation** — Cellpose-SAM detects every grain in the image and produces a pixel-level mask for each one.
3. **Coin detection & calibration** — the 5 PKR coin is located using color, shape, and feature matching, and used to convert pixel measurements into millimetres.
4. **Grain measurement** — each grain's length, width, aspect ratio, and area are computed from its mask.
5. **Quality classification** — a tight, shape-exact crop of each grain is passed through a CNN to classify it as Full, Broken, Rejected, Weak, or Fatty.
6. **Broken sub-classification** — broken grains are bucketed into SG/B1/B2/G1 based on their length as a percentage of a reference full-grain length.
7. **Variety classification** *(optional second pass)* — full grains are re-cropped and passed through a second CNN to identify the rice variety.
8. **Visualization & reporting** — results are rendered as color-coded overlay images, bar charts, length-distribution histograms, and exportable tables.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web interface | [Gradio](https://gradio.app/) |
| Grain segmentation | [Cellpose-SAM](https://github.com/MouseLand/cellpose) |
| Quality classifier | DINOv2 (frozen backbone) + MLP head |
| Variety classifier | EfficientNet-B0 |
| Image processing | OpenCV, Pillow, NumPy, SciPy |
| Deep learning | PyTorch, Hugging Face `transformers` |
| Charts & tables | Matplotlib, Pandas |
| HEIC support | pillow-heif |

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- A CUDA-capable GPU is recommended (the app will fall back to CPU automatically if none is available)
- Conda (recommended for managing the environment)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/grainvision-pro.git
cd grainvision-pro

# Create and activate the environment
conda create -n cellpose python=3.10
conda activate cellpose

# Install dependencies
pip install -r requirements.txt
```

### Model checkpoints

GrainVision PRO requires two trained model checkpoints:

| Checkpoint | Purpose | Config variable |
|---|---|---|
| 5-class quality model (`.pt`) | Full / Broken / Rejected / Weak / Fatty classification | `CNN_CHECKPOINT_PATH` |
| 7-class variety model (`.pt`) | Rice variety classification | `SECOND_CNN_CHECKPOINT_PATH` |

Update these paths at the top of `app.py` to point to your local checkpoint files. If a checkpoint fails to load, the corresponding feature is disabled and the app continues running (with a warning printed to the console).

---

## 🚀 Usage

```bash
conda activate cellpose
python app.py
```

The app will launch a local Gradio server and print a URL (and a public share link) in the console. Open it in your browser, then:

1. Upload a photo of rice grains next to a 5 PKR coin (or click one of the sample images).
2. Click **🔍 Quality test** for quality grading and measurements, or **🌾 Full grain types** for variety identification.
3. Explore the results in the **Combined View**, **Detail View**, and **Distribution Chart** tabs.
4. Download the per-grain measurement table or the summary table as an image if needed.

---

## ⚙️ Configuration

Key settings can be adjusted at the top of `app.py`:

| Setting | Description |
|---|---|
| `CNN_CHECKPOINT_PATH` | Path to the 5-class quality model checkpoint |
| `SECOND_CNN_CHECKPOINT_PATH` | Path to the 7-class variety model checkpoint |
| `COIN_REAL_MM` | Real-world diameter (mm) of the reference coin |
| `BROKEN_SUBCLASSES` | Size percentage ranges for SG / B1 / B2 / G1 |
| `CNN_USE_TTA` | Enable/disable Test-Time Augmentation |
| `SAVE_CNN_DEBUG_CROPS` / `SAVE_VARIETY_DEBUG_CROPS` | Save every classified grain crop to disk for debugging |
| `SAMPLE_IMAGES_DIR` | Folder of sample images shown in the UI |

---

## 📁 Project Structure

```
grainvision-pro/
├── app.py                  # Main application (UI + full pipeline)
├── fonts/                  # Auto-downloaded Urdu font (cached)
├── sample_images/          # Sample photos shown in the UI
├── cnn_debug_crops/        # Debug crops from the quality classifier (optional)
├── variety_debug_crops/    # Debug crops from the variety classifier (optional)
└── requirements.txt
```

---

## ⚠️ Notes & Limitations

- Measurements are only in millimetres when the reference coin is successfully detected in the photo; otherwise measurements fall back to pixels.
- Classification accuracy depends on photo resolution — very low-resolution photos with many small grains may reduce accuracy.
- The CNN input cropping method must match the cropping method used to generate the training dataset. If a model is retrained with a different cropping approach, the cropping function in `app.py` must be updated accordingly.

---

## 📄 License

Add your license here (e.g., MIT, Apache 2.0, or proprietary/internal use only).

---

## 🙌 Acknowledgements

- [Cellpose-SAM](https://github.com/MouseLand/cellpose) for grain segmentation
- [DINOv2](https://github.com/facebookresearch/dinov2) for feature extraction
- [Gradio](https://gradio.app/) for the web interface

---

## 👤 Author

**Muhammad Hamza Khalid**
📧 m.hamzakhalid22@gmail.com

Developed and maintained by Muhammad Hamza Khalid.
