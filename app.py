# 31 July 2026  (tight exact-mask CNN crops)
# * The CNN input crop is now produced by crop_grain_tight_native(), which cuts
#   each grain TIGHT to its EXACT Cellpose mask (no polygon simplification, no
#   6% pad, no 1px dilation) and pads to a centred square -- identical to the
#   training cropper (batch_crop_grains_5class_tight.py). This replaces the old
#   extract_grain_polygon_native + crop_grain_from_polygon path at the two CNN
#   call sites (quality pass + variety pass), so inference crops now match what
#   the retrained classifier was trained on. Segmentation, measurement, coin
#   calibration, drawing, tables, and the whole UI are UNCHANGED. The old crop
#   functions remain defined but are no longer used for CNN input.
#
# 08 July 2026  (performance overhaul — GPU + multi-threading)
#
# Modifications in THIS revision (on top of the 07 July 2026 masking/outline/
# summary revision):
# * NO CLASSIFICATION LOGIC CHANGED. Every patch below produces pixel-identical
#   crops, identical CNN inputs, and identical thresholds/rules to the previous
#   revision. Only HOW those results are computed changed.
# * WINDOWED GRAIN CROPPING -- crop_grain_from_mask (which used to run
#   cv2.cvtColor + a full-canvas `masks == label` scan + a full-canvas dilate
#   for EVERY grain) is replaced by crop_grain_from_mask_windowed, which uses
#   the grain's own bounding-box slice (already computed once via
#   scipy.ndimage.find_objects) so it only ever touches that grain's own
#   small window of the image instead of the entire canvas.
# * WINDOWED MASK BLEND / OUTLINE -- _blend_grain_mask and _draw_grain_outline
#   (used by both Tab 1 and Tab 2, called once per grain per visual) are
#   replaced by _blend_grain_mask_fast / _draw_grain_outline_fast, which
#   operate on a numpy VIEW of the grain's bounding box instead of scanning
#   `masks == label` across the full H x W canvas. Writes go straight back
#   into the canvas array (no copy), so visuals are byte-identical to before.
# * PARALLEL CPU WORK -- a shared ThreadPoolExecutor (sized to CPU core count)
#   is used to: (a) extract every grain's crop in parallel, and (b) run the
#   CNN's PIL-based preprocessing transform for every crop in parallel. This
#   is pure CPU-bound work (image decode/resize/normalize) that was
#   previously done in a single-threaded Python for-loop; running it across
#   threads (cv2/PIL/numpy release the GIL during their C work) lets all
#   cores work at once with zero change to the actual transform math.
# * GPU THROUGHPUT -- torch.backends.cudnn.benchmark is enabled once at
#   startup (auto-tunes convolution kernels for the model's fixed input
#   shape -- this only affects HOW the conv is computed on GPU, never the
#   numeric result for a fixed eval-mode model), batches are pinned to page-
#   locked memory before transfer, and .to(device, non_blocking=True) is used
#   so the CPU->GPU copy overlaps with other work instead of blocking. Model
#   remains in eval() mode throughout, so BatchNorm/Dropout behavior (and
#   therefore predictions) are unaffected by batch size or composition.
# * Everything else -- Cellpose segmentation, coin detection (heuristic +
#   ORB), CNN checkpoint/weights, broken sub-class height rule, drawing
#   colors/alphas/thresholds, and the overall Tab 1/2/3/4 pipeline -- is
#   completely unchanged from the previous revision.
#
# 12 July 2026  (HEIC/HEIF upload support)
# * iPhone photos are frequently uploaded as .HEIC/.HEIF, which plain
#   PIL/OpenCV/cellpose.io cannot decode -- this raised
#   PIL.UnidentifiedImageError deep inside tif_view()/imread() before any
#   segmentation/classification logic ever ran.
# * Added an optional pillow-heif import (register_heif_opener) and a new
#   convert_heic_if_needed() helper, called at the very top of run_analysis(),
#   which converts any .heic/.heif upload to a normal JPEG on disk BEFORE it
#   reaches tif_view/imread. Every other file type (.jpg/.png/.tif/etc.)
#   passes through unchanged -- no classification/segmentation logic touched.
# * If pillow-heif isn't installed, HEIC uploads now fail with a clear
#   gr.Error telling the user to `pip install pillow-heif`, instead of a raw
#   traceback.
#
# 13 July 2026  (measurement/summary table rework + Urdu summary + image
#                export buttons)   [see original for full detail]
#
# 24 July 2026  (UI: sample previews)
# * The quick-test sample strip now lives directly under the Preview box in
#   the left column (better on mobile, where the left column stacks on top),
#   and clicking a sample now updates the Preview box in place -- the same
#   behaviour uploading an image already had. No analysis/segmentation/
#   classification logic touched.

#==============================================================================( Code )=====================================================================================

"""
GrainVision PRO — Rice Grain Analysis
Cellpose-SAM segmentation  +  CNN 5-class classifier (full / broken /
rejected / weak / fatty)  +  height-based sub-classification of BROKEN
grains only, calibrated in mm via a 5 PKR reference coin.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import sys
import math
import glob
import socket
import base64
import datetime
import tempfile
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from scipy.ndimage import find_objects
from PIL import Image, ImageDraw, ImageFont

# ══════════════════════════════════════════════════════════════════════════════
#  HEIC / HEIF support (iPhone photo uploads)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    print("[heic] pillow-heif loaded -- .HEIC/.HEIF uploads supported")
except ImportError:
    pillow_heif = None
    print("[heic][warn] pillow-heif not installed -- HEIC/HEIF uploads (common "
          "for iPhone photos) will fail. Install with: pip install pillow-heif")

HEIC_EXTS = {".heic", ".heif"}

from cellpose import models
from cellpose.io import imread, imsave
from huggingface_hub import hf_hub_download

import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms
from torchvision.models import efficientnet_b0, efficientnet_b1, resnet50

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
#  URDU TEXT RENDERING (for the summary table's downloaded PNG image)
# ══════════════════════════════════════════════════════════════════════════════
import urllib.request

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
os.makedirs(FONT_DIR, exist_ok=True)
URDU_FONT_PATH = os.path.join(FONT_DIR, "NotoNastaliqUrdu-Regular.ttf")
URDU_FONT_URL  = ("https://raw.githubusercontent.com/google/fonts/main/ofl/"
                   "notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf")

_WINDOWS_URDU_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\JameelNooriNastaleeqRegular.ttf",
    r"C:\Windows\Fonts\jameel noori nastaleeq regular.ttf",
    r"C:\Windows\Fonts\Urdu Typesetting.ttf",
    r"C:\Windows\Fonts\urdutype.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]

_LATIN_FONT_PATH = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")

_urdu_font_path_cache = None


def _ensure_urdu_font():
    global _urdu_font_path_cache
    if _urdu_font_path_cache is not None:
        return _urdu_font_path_cache

    if os.path.exists(URDU_FONT_PATH) and os.path.getsize(URDU_FONT_PATH) > 10_000:
        _urdu_font_path_cache = URDU_FONT_PATH
        return _urdu_font_path_cache

    try:
        print(f"[urdu] downloading Noto Nastaliq Urdu font (one-time, cached "
              f"in '{FONT_DIR}')...")
        urllib.request.urlretrieve(URDU_FONT_URL, URDU_FONT_PATH)
        if os.path.getsize(URDU_FONT_PATH) > 10_000:
            print("[urdu] font downloaded OK")
            _urdu_font_path_cache = URDU_FONT_PATH
            return _urdu_font_path_cache
    except Exception as e:
        print(f"[urdu][warn] could not download Noto Nastaliq Urdu font: {e}")

    for cand in _WINDOWS_URDU_FONT_CANDIDATES:
        if os.path.exists(cand):
            print(f"[urdu] using system font fallback: {cand}")
            _urdu_font_path_cache = cand
            return _urdu_font_path_cache

    print("[urdu][warn] no Urdu-capable font found (download failed and no "
          "system fallback located) -- downloaded table images may show Urdu "
          "as boxes. Fix by manually placing a .ttf font (e.g. Noto Nastaliq "
          f"Urdu) at:\n    {URDU_FONT_PATH}")
    _urdu_font_path_cache = None
    return None


_RAQM_LAYOUT = getattr(ImageFont, "LAYOUT_RAQM", None)
if _RAQM_LAYOUT is None:
    _layout_enum = getattr(ImageFont, "Layout", None)
    _RAQM_LAYOUT = getattr(_layout_enum, "RAQM", None) if _layout_enum else None

_RAQM_WARNED = False
_font_cache = {}


def _load_font(path, size):
    global _RAQM_WARNED
    if path is None:
        return ImageFont.load_default()
    key = (path, size)
    if key in _font_cache:
        return _font_cache[key]
    try:
        if _RAQM_LAYOUT is None:
            raise RuntimeError("no Layout.RAQM / LAYOUT_RAQM attribute on this Pillow build")
        font = ImageFont.truetype(path, size, layout_engine=_RAQM_LAYOUT)
    except Exception as e:
        if not _RAQM_WARNED:
            print(f"[urdu][warn] Raqm text-shaping engine not available in this "
                  f"Pillow build ({e}) -- Urdu letters will render isolated/"
                  f"disconnected instead of properly joined. Fix with:\n"
                  f"    pip install --upgrade --force-reinstall Pillow")
            _RAQM_WARNED = True
        font = ImageFont.truetype(path, size)
    _font_cache[key] = font
    return font


def _is_arabic_script(s):
    return any('\u0600' <= ch <= '\u06FF' for ch in s)


def _font_for_text(text, size):
    if _is_arabic_script(text):
        return _load_font(_ensure_urdu_font(), size)
    return _load_font(_LATIN_FONT_PATH, size)


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE — shared thread pool for CPU-bound work
# ══════════════════════════════════════════════════════════════════════════════
CPU_WORKERS = max(4, (os.cpu_count() or 4))
_EXECUTOR = ThreadPoolExecutor(max_workers=CPU_WORKERS)
print(f"[perf] CPU thread pool: {CPU_WORKERS} workers")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FULL_CLASS_KEY  = "FULL"
FULL_CLASS_NAME = "Full"
FULL_CLASS_COLOR_RGB = (0, 87, 255)
FULL_CLASS_COLOR_HEX = "#0057FF"

BROKEN_CLASS_KEY  = "BROKEN"
BROKEN_CLASS_NAME = "Broken"
BROKEN_CLASS_COLOR_RGB = (255, 215, 0)
BROKEN_CLASS_COLOR_HEX = "#FFD700"

OTHER_CLASSES = [
    {"key": "REJECTED", "name": "Rejection",  "color_rgb": (214,   0, 158)},
    {"key": "WEAK",     "name": "Weak",       "color_rgb": (  0, 200,  83)},
    {"key": "FATTY",    "name": "Fatty",      "color_rgb": (255, 107,   0)},
]
for _oc in OTHER_CLASSES:
    _r, _g, _b = _oc["color_rgb"]
    _oc["color_hex"] = f"#{_r:02X}{_g:02X}{_b:02X}"

DIM_COLOR_RGB = (110, 110, 110)

REJECTED_OUTLINE_COLOR_RGB = (0, 0, 0)

MASK_FILL_ALPHA = 0.40

MASK_FILL_ALPHA_BY_CLASS = {
    "full":     MASK_FILL_ALPHA,
    "broken":   MASK_FILL_ALPHA,
    "rejected": MASK_FILL_ALPHA,
    "weak":     MASK_FILL_ALPHA,
    "fatty":    MASK_FILL_ALPHA,
}

SHOW_GRAIN_LABELS = False

SHOW_GRAIN_OUTLINE = True
OUTLINE_ONLY_CLASS_NAME = "Rejection"

BROKEN_SUBCLASSES = [
    {"key": "SG", "name": "SG",  "min_pct": 75.0,  "max_pct": 90.0,  "color_rgb": (250, 204,  21)},
    {"key": "B1", "name": "B1",  "min_pct": 55.0,  "max_pct": 74.0,  "color_rgb": ( 59, 130, 246)},
    {"key": "B2", "name": "B2",  "min_pct": 26.0,  "max_pct": 54.0,  "color_rgb": (239,  68,  68)},
    {"key": "G1", "name": "G1",  "min_pct":  0.0,  "max_pct": 25.0,  "color_rgb": (168,  85, 247)},
]
for _bc in BROKEN_SUBCLASSES:
    _r, _g, _b = _bc["color_rgb"]
    _bc["color_hex"] = f"#{_r:02X}{_g:02X}{_b:02X}"

_BROKEN_SUBCLASS_KEYS = {bc["key"] for bc in BROKEN_SUBCLASSES}

SUBCLASS_FALLBACK_REF_MM = 7.5

CNN_CHECKPOINT_PATH = r"D:\hamza\rice\29Jul26\dataset\runs\dinob_mlp_v1\best_dino_mlp.pt"

SECOND_CNN_CHECKPOINT_PATH = r"D:\hamza\rice\5classTraining\dataset\runs\secondclassifier2\best_efficientnet_b0.pt"
SAVE_VARIETY_DEBUG_CROPS = True
VARIETY_DEBUG_CROPS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variety_debug_crops")

VARIETY_PALETTE_RGB = [
    (0, 87, 255), (255, 215, 0), (214, 0, 158), (0, 200, 83),
    (255, 107, 0), (34, 211, 238), (236, 72, 153), (163, 230, 53),
    (129, 140, 248), (248, 113, 113), (45, 212, 191), (250, 204, 21),
]

def _normalize_variety_key(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")

VARIETY_COLOR_BY_NAME_RGB = {
    _normalize_variety_key("1121"):         (0, 102, 255),
    _normalize_variety_key("C9"):           (0, 255, 0),
    _normalize_variety_key("386"):          (255, 0, 0),
    _normalize_variety_key("Supri"):        (255, 136, 0),
    _normalize_variety_key("Super Kernel"): (153, 0, 255),
    _normalize_variety_key("Superkernel"):  (153, 0, 255),
    _normalize_variety_key("Super Fine"):   (0, 255, 255),
    _normalize_variety_key("Weak"):         (255, 0, 170),
}

def _variety_color_for(name, idx):
    key = _normalize_variety_key(name)
    if key in VARIETY_COLOR_BY_NAME_RGB:
        return VARIETY_COLOR_BY_NAME_RGB[key]
    return VARIETY_PALETTE_RGB[idx % len(VARIETY_PALETTE_RGB)]

SAVE_CNN_DEBUG_CROPS = True
CNN_DEBUG_CROPS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_debug_crops")

SAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

CNN_CLASS_INDEX_ORDER = ["full", "broken", "rejected", "weak", "fatty"]

_CLASS_COLOR_BY_NAME_RGB = {
    "full":     FULL_CLASS_COLOR_RGB,
    "broken":   BROKEN_CLASS_COLOR_RGB,
    **{oc["key"].lower(): oc["color_rgb"] for oc in OTHER_CLASSES},
}
_CLASS_COLOR_BY_NAME_HEX = {
    "full":     FULL_CLASS_COLOR_HEX,
    "broken":   BROKEN_CLASS_COLOR_HEX,
    **{oc["key"].lower(): oc["color_hex"] for oc in OTHER_CLASSES},
}

BACKBONE_INPUT_SIZE = {"efficientnet_b0": 224, "efficientnet_b1": 240, "resnet50": 224}

CNN_CROP_PAD_FRAC = 0.06
CNN_BATCH_SIZE    = 128

COIN_REAL_MM = 18.0

SUB_CLASS_ORDER = (
    [FULL_CLASS_KEY] + [bc["key"] for bc in BROKEN_SUBCLASSES]
    + [oc["key"] for oc in OTHER_CLASSES] + ["BR"]
)
SUB_COLORS = {FULL_CLASS_KEY: FULL_CLASS_COLOR_RGB, "BR": (148, 163, 184)}
SUB_COLORS.update({bc["key"]: bc["color_rgb"] for bc in BROKEN_SUBCLASSES})
SUB_COLORS.update({oc["key"]: oc["color_rgb"] for oc in OTHER_CLASSES})
SUB_HEX = {FULL_CLASS_KEY: FULL_CLASS_COLOR_HEX, "BR": "#94A3B8"}
SUB_HEX.update({bc["key"]: bc["color_hex"] for bc in BROKEN_SUBCLASSES})
SUB_HEX.update({oc["key"]: oc["color_hex"] for oc in OTHER_CLASSES})
SUB_NAMES = {FULL_CLASS_KEY: FULL_CLASS_NAME, "BR": "Unclassified (broken)"}
SUB_NAMES.update({bc["key"]: bc["name"] for bc in BROKEN_SUBCLASSES})
SUB_NAMES.update({oc["key"]: oc["name"] for oc in OTHER_CLASSES})
C_COIN_BOX = (180, 180, 180)

OUT_MAX_DIM = 1400
OUT_JPEG_Q  = 83


# ══════════════════════════════════════════════════════════════════════════════
#  CNN MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn_model(backbone: str, num_classes: int = 5) -> nn.Module:
    if backbone == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif backbone == "efficientnet_b1":
        m = efficientnet_b1(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif backbone == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return m


# ── DINOv2 frozen backbone + MLP head (Tab 1 quality classifier) ──────────────
# Matches train_dino_mlp.py: frozen DINOv2-B features (CLS ⊕ patch-mean) → MLP.
# Wrapped as a single nn.Module so classify_crops_cnn's `model(batch)` call and
# the CNN_TRANSFORM (ImageNet resize+normalize) work unchanged — the EfficientNet
# path and the 2nd (variety) model are not affected.
class _MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class DinoMLPClassifier(nn.Module):
    def __init__(self, num_classes, pooling="cls_mean", hidden=512,
                 dino_model="facebook/dinov2-base"):
        super().__init__()
        from transformers import AutoModel  # lazy: only needed for a DINOv2 checkpoint
        self.backbone = AutoModel.from_pretrained(dino_model)
        self.pooling = pooling
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        hdim = self.backbone.config.hidden_size
        in_dim = hdim * (2 if pooling == "cls_mean" else 1)
        self.head = _MLPHead(in_dim, num_classes, hidden=hidden)

    def forward(self, x):
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        last = out.last_hidden_state
        cls = last[:, 0]
        patch_mean = last[:, 1:].mean(dim=1)
        if self.pooling == "cls":
            feat = cls
        elif self.pooling == "mean":
            feat = patch_mean
        else:
            feat = torch.cat([cls, patch_mean], dim=1)
        return self.head(feat.float())


def _resolve_class_order(ckpt, fallback_order):
    for key in ("classes", "class_names", "idx_to_class"):
        val = ckpt.get(key)
        if isinstance(val, dict):
            order = [val[i].lower() for i in sorted(val, key=lambda k: int(k))]
            return order, key
        if isinstance(val, (list, tuple)) and val:
            return [str(v).lower() for v in val], key

    val = ckpt.get("class_to_idx")
    if isinstance(val, dict) and val:
        order = [name.lower() for name, _ in sorted(val.items(), key=lambda kv: kv[1])]
        return order, "class_to_idx"

    return [n.lower() for n in fallback_order], "CONFIG fallback (CNN_CLASS_INDEX_ORDER)"


def load_cnn_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # --- DINOv2 frozen + MLP checkpoint (from train_dino_mlp.py) ---
    # Detected by its head_state_dict/dino_model keys. Loaded as the wrapper
    # module above; the EfficientNet/ResNet branch below is untouched.
    if "head_state_dict" in ckpt and "dino_model" in ckpt:
        class_order, source = _resolve_class_order(ckpt, CNN_CLASS_INDEX_ORDER)
        num_classes = len(class_order)
        pooling    = ckpt.get("pooling", "cls_mean")
        input_size = ckpt.get("input_size", 224)
        hidden     = ckpt.get("hidden", 512)
        dino_model = ckpt.get("dino_model", "facebook/dinov2-base")
        m = DinoMLPClassifier(num_classes, pooling=pooling, hidden=hidden,
                              dino_model=dino_model).to(device)
        m.head.load_state_dict(ckpt["head_state_dict"])
        m.eval()
        print(f"CNN checkpoint loaded (DINOv2 frozen + MLP): {path}")
        print(f"  dino={dino_model}  pooling={pooling}  input_size={input_size}  "
              f"num_classes={num_classes}  val_acc={ckpt.get('val_acc', 'N/A')}")
        print(f"  class order (index -> name), from {source}: "
              f"{ {i: n for i, n in enumerate(class_order)} }")
        print("  >>> If this order looks wrong, retrain/save with the correct class order.")
        return m, "dinov2_mlp", input_size, class_order

    # --- EfficientNet / ResNet checkpoint (original behaviour) ---
    backbone   = ckpt.get("backbone", "efficientnet_b0")
    input_size = ckpt.get("input_size", BACKBONE_INPUT_SIZE.get(backbone, 224))

    class_order, source = _resolve_class_order(ckpt, CNN_CLASS_INDEX_ORDER)
    num_classes = len(class_order)

    m = build_cnn_model(backbone, num_classes=num_classes).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    print(f"CNN checkpoint loaded: {path}")
    print(f"  backbone={backbone}  input_size={input_size}  num_classes={num_classes}  "
          f"val_acc={ckpt.get('val_acc', 'N/A')}  phase={ckpt.get('phase', 1)}")
    print(f"  class order (index -> name), from {source}: "
          f"{ {i: n for i, n in enumerate(class_order)} }")
    print("  >>> If this order looks wrong, fix CNN_CLASS_INDEX_ORDER at the top "
          "of the file (or retrain with a checkpoint that saves its class order).")
    return m, backbone, input_size, class_order


CNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if CNN_DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"[perf] CUDA device: {torch.cuda.get_device_name(0)} | cudnn.benchmark=True")
else:
    print("[perf] No CUDA device found -- running on CPU.")

try:
    CNN_MODEL, CNN_BACKBONE, CNN_INPUT_SIZE, CNN_CLASS_ORDER_RESOLVED = load_cnn_checkpoint(
        CNN_CHECKPOINT_PATH, CNN_DEVICE)

    CNN_CLASS_NAMES = {i: name for i, name in enumerate(CNN_CLASS_ORDER_RESOLVED)}
    CNN_COLOR_RGB = {i: _CLASS_COLOR_BY_NAME_RGB.get(name, DIM_COLOR_RGB)
                      for i, name in CNN_CLASS_NAMES.items()}
    CNN_COLOR_HEX = {i: _CLASS_COLOR_BY_NAME_HEX.get(name, "#6E6E6E")
                      for i, name in CNN_CLASS_NAMES.items()}
    CNN_NAME_TO_ID = {name: i for i, name in CNN_CLASS_NAMES.items()}
    FULL_ID     = CNN_NAME_TO_ID.get("full")
    BROKEN_ID   = CNN_NAME_TO_ID.get("broken")
    REJECTED_ID = CNN_NAME_TO_ID.get("rejected")
    WEAK_ID     = CNN_NAME_TO_ID.get("weak")
    FATTY_ID    = CNN_NAME_TO_ID.get("fatty")
    OTHER_ID_TO_KEY = {}
    if REJECTED_ID is not None:
        OTHER_ID_TO_KEY[REJECTED_ID] = "REJECTED"
    if WEAK_ID is not None:
        OTHER_ID_TO_KEY[WEAK_ID] = "WEAK"
    if FATTY_ID is not None:
        OTHER_ID_TO_KEY[FATTY_ID] = "FATTY"

    _CNN_MEAN = [0.485, 0.456, 0.406]
    _CNN_STD  = [0.229, 0.224, 0.225]
    CNN_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_CNN_MEAN, _CNN_STD),
    ])
except Exception as e:
    print(f"⚠️  CNN checkpoint failed to load: {e}")
    print(f"   Tab 1 (CNN Full vs Broken) will be unavailable. "
          f"Check CNN_CHECKPOINT_PATH at the top of this file.")
    CNN_MODEL = None
    CNN_CLASS_NAMES = {i: n.lower() for i, n in enumerate(CNN_CLASS_INDEX_ORDER)}
    CNN_COLOR_RGB   = {i: _CLASS_COLOR_BY_NAME_RGB.get(n, DIM_COLOR_RGB) for i, n in CNN_CLASS_NAMES.items()}
    CNN_COLOR_HEX   = {i: _CLASS_COLOR_BY_NAME_HEX.get(n, "#6E6E6E") for i, n in CNN_CLASS_NAMES.items()}
    CNN_NAME_TO_ID  = {n: i for i, n in CNN_CLASS_NAMES.items()}
    FULL_ID, BROKEN_ID = CNN_NAME_TO_ID.get("full"), CNN_NAME_TO_ID.get("broken")
    REJECTED_ID, WEAK_ID, FATTY_ID = (CNN_NAME_TO_ID.get("rejected"),
                                       CNN_NAME_TO_ID.get("weak"), CNN_NAME_TO_ID.get("fatty"))
    OTHER_ID_TO_KEY = {i: k for i, k in ((REJECTED_ID, "REJECTED"), (WEAK_ID, "WEAK"), (FATTY_ID, "FATTY"))
                       if i is not None}

CNN_DISPLAY_NAME_BY_ID = {}
if FULL_ID is not None:
    CNN_DISPLAY_NAME_BY_ID[FULL_ID] = FULL_CLASS_NAME
if BROKEN_ID is not None:
    CNN_DISPLAY_NAME_BY_ID[BROKEN_ID] = BROKEN_CLASS_NAME
for _oc in OTHER_CLASSES:
    _cid = CNN_NAME_TO_ID.get(_oc["key"].lower())
    if _cid is not None:
        CNN_DISPLAY_NAME_BY_ID[_cid] = _oc["name"]


# ══════════════════════════════════════════════════════════════════════════════
#  SECOND CNN MODEL LOADING -- 7-class GRAIN VARIETY/TYPE classifier
# ══════════════════════════════════════════════════════════════════════════════
try:
    CNN_MODEL_2, CNN_BACKBONE_2, CNN_INPUT_SIZE_2, CNN_CLASS_ORDER_2 = load_cnn_checkpoint(
        SECOND_CNN_CHECKPOINT_PATH, CNN_DEVICE)

    CNN_CLASS_NAMES_2 = {i: name for i, name in enumerate(CNN_CLASS_ORDER_2)}
    CNN_COLOR_RGB_2 = {i: _variety_color_for(name, i)
                        for i, name in CNN_CLASS_NAMES_2.items()}
    CNN_COLOR_HEX_2 = {i: f"#{r:02X}{g:02X}{b:02X}" for i, (r, g, b) in CNN_COLOR_RGB_2.items()}
    CNN_NAME_TO_ID_2 = {name: i for i, name in CNN_CLASS_NAMES_2.items()}
    CNN_DISPLAY_NAME_BY_ID_2 = {i: name.replace("_", " ").title() for i, name in CNN_CLASS_NAMES_2.items()}

    _CNN_MEAN_2 = [0.485, 0.456, 0.406]
    _CNN_STD_2  = [0.229, 0.224, 0.225]
    CNN_TRANSFORM_2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CNN_INPUT_SIZE_2, CNN_INPUT_SIZE_2)),
        transforms.ToTensor(),
        transforms.Normalize(_CNN_MEAN_2, _CNN_STD_2),
    ])
except Exception as e:
    print(f"⚠️  2nd (7-class variety) CNN checkpoint failed to load: {e}")
    print(f"   The '🌾 Full grain types' button will be unavailable. "
          f"Check SECOND_CNN_CHECKPOINT_PATH at the top of this file.")
    CNN_MODEL_2 = None
    CNN_CLASS_NAMES_2 = {}
    CNN_COLOR_RGB_2 = {}
    CNN_COLOR_HEX_2 = {}
    CNN_NAME_TO_ID_2 = {}
    CNN_DISPLAY_NAME_BY_ID_2 = {}
    CNN_TRANSFORM_2 = None
    CNN_INPUT_SIZE_2 = 224


def crop_grain_from_mask(img_rgb: np.ndarray, mask_bool: np.ndarray,
                          pad_frac: float = CNN_CROP_PAD_FRAC):
    """Original full-canvas-scan version. Kept for reference / fallback use only."""
    ys, xs = np.where(mask_bool)
    if len(xs) < 5:
        return None
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    gw, gh = x2 - x1 + 1, y2 - y1 + 1
    if gw < 6 or gh < 6:
        return None

    pad = int(round(max(gw, gh) * pad_frac))
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)

    patch_bgr  = img_bgr[y1p:y2p+1, x1p:x2p+1].copy()
    patch_mask = mask_u8[y1p:y2p+1, x1p:x2p+1]
    patch_bgr[patch_mask == 0] = (0, 0, 0)

    ph, pw = patch_bgr.shape[:2]
    side = max(ph, pw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    oy, ox = (side - ph) // 2, (side - pw) // 2
    square[oy:oy+ph, ox:ox+pw] = patch_bgr
    return square


def extract_grain_polygon_native(masks_analysis, label, sl, sr_scale, epsilon_frac=0.005):
    """Returns this grain's polygon as an (N,2) float32 array in NATIVE image coords."""
    y1, y2 = sl[0].start, sl[0].stop
    x1, x2 = sl[1].start, sl[1].stop
    local = masks_analysis[y1:y2, x1:x2]
    bin_mask = (local == label).astype(np.uint8) * 255
    if not bin_mask.any():
        return None

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 4:
        return None
    perim = cv2.arcLength(cnt, True)
    epsilon = epsilon_frac * perim
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) < 3:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        approx = np.array([[[x_, y_]], [[x_+w_, y_]],
                            [[x_+w_, y_+h_]], [[x_, y_+h_]]], dtype=np.int32)

    pts_local = approx.reshape(-1, 2).astype(np.float32)

    pts_analysis = pts_local.copy()
    pts_analysis[:, 0] += x1
    pts_analysis[:, 1] += y1

    pts_native = pts_analysis / sr_scale

    h_native, w_native = masks_analysis.shape[0] // sr_scale, masks_analysis.shape[1] // sr_scale
    pts_native[:, 0] = np.clip(pts_native[:, 0], 0.0, w_native)
    pts_native[:, 1] = np.clip(pts_native[:, 1], 0.0, h_native)

    return pts_native


def crop_grain_from_polygon(img_native_rgb, pts_native, pad_frac=None):
    """Crops tight to a precise polygon, black background, padded to square."""
    if pad_frac is None:
        pad_frac = CNN_CROP_PAD_FRAC
    h, w = img_native_rgb.shape[:2]
    x1f = float(pts_native[:, 0].min()); x2f = float(pts_native[:, 0].max())
    y1f = float(pts_native[:, 1].min()); y2f = float(pts_native[:, 1].max())
    gw, gh = x2f - x1f, y2f - y1f
    if gw < 6 or gh < 6:
        return None

    pad = max(gw, gh) * pad_frac
    x1p = max(0, int(round(x1f - pad))); y1p = max(0, int(round(y1f - pad)))
    x2p = min(w - 1, int(round(x2f + pad))); y2p = min(h - 1, int(round(y2f + pad)))

    patch_bgr = cv2.cvtColor(img_native_rgb[y1p:y2p+1, x1p:x2p+1], cv2.COLOR_RGB2BGR).copy()

    pts_local = pts_native.copy()
    pts_local[:, 0] -= x1p
    pts_local[:, 1] -= y1p
    pts_local_int = np.round(pts_local).astype(np.int32)

    mask_u8 = np.zeros(patch_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_u8, [pts_local_int], 255)
    mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)

    patch_bgr[mask_u8 == 0] = (0, 0, 0)

    ph, pw = patch_bgr.shape[:2]
    side = max(ph, pw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    oy, ox = (side - ph) // 2, (side - pw) // 2
    square[oy:oy+ph, ox:ox+pw] = patch_bgr
    return square


def crop_grain_from_mask_windowed(img_rgb: np.ndarray, masks_full: np.ndarray,
                                   label: int, sl,
                                   pad_frac: float = CNN_CROP_PAD_FRAC):
    """[perf] Bit-identical output to crop_grain_from_mask, windowed."""
    h, w = img_rgb.shape[:2]
    y1, y2 = sl[0].start, sl[0].stop - 1
    x1, x2 = sl[1].start, sl[1].stop - 1
    gw, gh = x2 - x1 + 1, y2 - y1 + 1
    if gw < 6 or gh < 6:
        return None

    pad = int(round(max(gw, gh) * pad_frac))
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

    patch_bgr = cv2.cvtColor(img_rgb[y1p:y2p+1, x1p:x2p+1], cv2.COLOR_RGB2BGR).copy()

    local_labels = masks_full[y1p:y2p+1, x1p:x2p+1]
    mask_u8 = (local_labels == label).astype(np.uint8) * 255
    mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)

    patch_bgr[mask_u8 == 0] = (0, 0, 0)

    ph, pw = patch_bgr.shape[:2]
    side = max(ph, pw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    oy, ox = (side - ph) // 2, (side - pw) // 2
    square[oy:oy+ph, ox:ox+pw] = patch_bgr
    return square


# ══════════════════════════════════════════════════════════════════════════════
#  TIGHT exact-mask CNN crop  (matches batch_crop_grains_5class_tight.py)
#  The classifier is now trained on tight, shape-exact crops: the exact Cellpose
#  mask, no polygon simplification, no pad, no dilation, padded to a centred
#  square. Inference MUST crop the same way or the model sees a different input
#  distribution than it trained on. Keep these knobs in lockstep with the
#  training cropper.
# ══════════════════════════════════════════════════════════════════════════════
TIGHT_CROP_PAD_FRAC      = 0.0    # 0.0 = tight to the grain (training default)
TIGHT_CROP_MASK_ERODE_PX = 0      # set 1 if a thin background rim survives
TIGHT_CROP_SMOOTH_EDGES  = True   # area-downscale 2x mask -> native + threshold
TIGHT_CROP_MIN_GRAIN_PX  = 6


def crop_grain_tight_native(img_native_rgb, masks_analysis, label, sl, sr_scale):
    """
    Crop ONE grain tight to its EXACT Cellpose mask out of the RAW native image,
    black out everything outside the grain, and pad to a centred square. No pad,
    no dilation, no polygon simplification — the same approach as the training
    cropper, so the CNN input at inference matches the CNN input at training.

    masks_analysis : label mask at sr_scale × native resolution (uint16)
    sl             : (y-slice, x-slice) for this label, in masks_analysis coords
    Returns a BGR square (what classify_crops_cnn expects), or None.
    """
    H, W = img_native_rgb.shape[:2]

    y1a, y2a = sl[0].start, sl[0].stop
    x1a, x2a = sl[1].start, sl[1].stop
    local2x = (masks_analysis[y1a:y2a, x1a:x2a] == label).astype(np.uint8) * 255
    if not local2x.any():
        return None

    lh = max(1, int(round((y2a - y1a) / sr_scale)))
    lw = max(1, int(round((x2a - x1a) / sr_scale)))
    if TIGHT_CROP_SMOOTH_EDGES:
        local_native = cv2.resize(local2x, (lw, lh), interpolation=cv2.INTER_AREA)
        _, local_native = cv2.threshold(local_native, 127, 255, cv2.THRESH_BINARY)
    else:
        local_native = cv2.resize(local2x, (lw, lh), interpolation=cv2.INTER_NEAREST)

    if TIGHT_CROP_MASK_ERODE_PX > 0:
        local_native = cv2.erode(local_native, np.ones((3, 3), np.uint8),
                                 iterations=int(TIGHT_CROP_MASK_ERODE_PX))

    # tighten to the true mask extent (this is what removes background)
    ys, xs = np.where(local_native > 0)
    if ys.size == 0:
        return None
    ty1, ty2, tx1, tx2 = ys.min(), ys.max(), xs.min(), xs.max()
    local_native = local_native[ty1:ty2 + 1, tx1:tx2 + 1]
    gh, gw = local_native.shape
    if gw < TIGHT_CROP_MIN_GRAIN_PX or gh < TIGHT_CROP_MIN_GRAIN_PX:
        return None

    ny1 = int(round(y1a / sr_scale)) + ty1
    nx1 = int(round(x1a / sr_scale)) + tx1

    pad = int(round(max(gw, gh) * TIGHT_CROP_PAD_FRAC)) if TIGHT_CROP_PAD_FRAC > 0 else 0
    ny1p = max(0, ny1 - pad); nx1p = max(0, nx1 - pad)
    ny2p = min(H, ny1 + gh + pad); nx2p = min(W, nx1 + gw + pad)

    patch_rgb = img_native_rgb[ny1p:ny2p, nx1p:nx2p]
    if patch_rgb.size == 0:
        return None
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR).copy()

    # place the exact mask onto the patch; clip to whatever fits at image borders
    ph, pw = patch_bgr.shape[:2]
    off_y, off_x = ny1 - ny1p, nx1 - nx1p
    avail_h = min(gh, ph - off_y)
    avail_w = min(gw, pw - off_x)
    if avail_h < TIGHT_CROP_MIN_GRAIN_PX or avail_w < TIGHT_CROP_MIN_GRAIN_PX:
        return None
    m = np.zeros((ph, pw), np.uint8)
    m[off_y:off_y + avail_h, off_x:off_x + avail_w] = local_native[:avail_h, :avail_w]
    patch_bgr[m == 0] = (0, 0, 0)

    # pad to a centred square (preserves aspect ratio → no shape distortion)
    side = max(ph, pw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    oy, ox = (side - ph) // 2, (side - pw) // 2
    square[oy:oy + ph, ox:ox + pw] = patch_bgr
    return square


CNN_USE_TTA = True

@torch.no_grad()
def classify_crops_cnn(crops_bgr, model=None, transform=None, batch_size=None, use_tta=None):
    """Same math as before, parallelized preprocessing + pinned/non-blocking transfer + TTA."""
    if model is None:
        model = CNN_MODEL
    if transform is None:
        transform = CNN_TRANSFORM
    if batch_size is None:
        batch_size = CNN_BATCH_SIZE
    if use_tta is None:
        use_tta = CNN_USE_TTA

    if model is None or not crops_bgr:
        return [(0, 0.0)] * len(crops_bgr)

    def _to_tensor(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return transform(img_rgb)

    tensors = list(_EXECUTOR.map(_to_tensor, crops_bgr))

    if use_tta:
        tensors_hflip = [t.flip(dims=[2]) for t in tensors]
        tensors_vflip = [t.flip(dims=[1]) for t in tensors]
        all_tensors = tensors + tensors_hflip + tensors_vflip
        n = len(tensors)
    else:
        all_tensors = tensors
        n = len(tensors)

    all_probs = []
    use_pin = CNN_DEVICE.type == "cuda"
    for i in range(0, len(all_tensors), batch_size):
        batch = torch.stack(all_tensors[i:i + batch_size])
        if use_pin:
            batch = batch.pin_memory()
        batch_t = batch.to(CNN_DEVICE, non_blocking=True)
        with autocast(device_type="cuda" if CNN_DEVICE.type == "cuda" else "cpu"):
            out = model(batch_t)
            probs = torch.softmax(out, dim=1).float()
        all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0)

    if use_tta:
        avg_probs = (all_probs[:n] + all_probs[n:2*n] + all_probs[2*n:3*n]) / 3.0
    else:
        avg_probs = all_probs

    preds = avg_probs.argmax(dim=1).numpy()
    confs = avg_probs.max(dim=1).values.numpy()
    results = [(int(p), float(c)) for p, c in zip(preds, confs)]
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Cellpose model
# ══════════════════════════════════════════════════════════════════════════════
try:
    _fpath = hf_hub_download(repo_id="mouseland/cellpose-sam", filename="cpsam")
    cp_model = models.CellposeModel(gpu=True, pretrained_model=_fpath)
    print(f"Cellpose model loaded: {_fpath}")
except Exception as e:
    print(f"Cellpose model load failed: {e}"); sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Image utilities
# ══════════════════════════════════════════════════════════════════════════════

def normalize99(img):
    X = img.copy().astype(np.float32)
    p1, p99 = np.percentile(X, 1), np.percentile(X, 99)
    return (X - p1) / (1e-10 + p99 - p1)

def image_resize(img, resize=1000):
    ny, nx = img.shape[:2]
    if max(ny, nx) > resize:
        if ny > nx:
            nx = int(nx / ny * resize); ny = resize
        else:
            ny = int(ny / nx * resize); nx = resize
        img = cv2.resize(img, (nx, ny))
    return img.astype(np.uint8)

def tif_view(filepath):
    _, ext = os.path.splitext(filepath)
    if ext.lower() in ['.tiff', '.tif']:
        img = imread(filepath)
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
        elif img.ndim == 3:
            imin = int(np.argmin(img.shape))
            if imin < 2:
                axes = [i for i in range(3) if i != imin] + [imin]
                img  = np.transpose(img, axes)
        Ly, Lx = img.shape[:2]
        imgi = np.zeros((Ly, Lx, 3), dtype=img.dtype)
        imgi[:, :, :min(3, img.shape[-1])] = img[:, :, :min(3, img.shape[-1])]
        imsave(filepath, imgi)
    return filepath


def convert_heic_if_needed(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in HEIC_EXTS:
        return filepath
    if pillow_heif is None:
        raise gr.Error("This is a HEIC/HEIF photo (common for iPhone camera "
                        "shots). Install support with:  pip install pillow-heif "
                        " -- then restart the app and try again.")
    print(f"[heic] converting {filepath} -> JPEG")
    img = Image.open(filepath).convert("RGB")
    out_path = os.path.splitext(filepath)[0] + "_converted.jpg"
    img.save(out_path, "JPEG", quality=95)
    return out_path

def prepare_input_image_for_preview(filepath):
    if filepath is None:
        return None

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in HEIC_EXTS:
        return filepath

    if pillow_heif is None:
        return None

    img = Image.open(filepath).convert("RGB")
    img.thumbnail((1000, 1000), Image.LANCZOS)
    out_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    img.save(out_path, "JPEG", quality=85)
    return out_path

def enhance_resolution(img_np):
    blur  = cv2.GaussianBlur(img_np, (0, 0), sigmaX=1.5)
    sharp = cv2.addWeighted(img_np, 1.6, blur, -0.6, 0)
    lab   = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB).astype(np.uint8)

def save_compressed(img_rgb, save_path, max_dim=OUT_MAX_DIM, quality=OUT_JPEG_Q):
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = save_path.rsplit('.', 1)[0] + '.jpg'
    cv2.imwrite(out, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out

def px_to_mm(px, coin_diam_px):
    if coin_diam_px is None or coin_diam_px == 0:
        return None
    return px * COIN_REAL_MM / coin_diam_px

# ══════════════════════════════════════════════════════════════════════════════
#  ORB FEATURE-MATCHING COIN CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════

COIN_ORB_MIN_MATCHES = 30

# ------------------------------------------------------------------------
# <-- PASTE YOUR ORIGINAL _COIN_TEMPLATE_B64 STRING HERE (unchanged from
#     your previous version). Left as an empty string per your request --
#     add it back exactly as it was; nothing about it needs to change.
#     While it's empty, coin detection automatically falls back to the
#     heuristic-only path (see detect_coin_diameter_px below).
# ------------------------------------------------------------------------
_COIN_TEMPLATE_B64 = (
    "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"
    "yI3IjciNyB0oF8gydWam6QAAAABJRU5ErkJggg=="
)


def _load_coin_template():
    try:
        if not _COIN_TEMPLATE_B64:
            raise ValueError("empty coin template (paste _COIN_TEMPLATE_B64)")
        raw = base64.b64decode(_COIN_TEMPLATE_B64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        tpl_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if tpl_bgr is None:
            raise ValueError("failed to decode embedded coin template image")
        tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(tpl_gray, None)
        print(f"[coin] ORB template loaded: {len(kp)} keypoints")
        return orb, des
    except Exception as e:
        print(f"⚠️  Coin ORB template failed to load: {e}")
        print("   Falling back to heuristic-only coin detection (less robust "
              "against lone-grain false positives).")
        return cv2.ORB_create(nfeatures=500), None


_COIN_ORB, _COIN_TEMPLATE_DES = _load_coin_template()
_COIN_BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def orb_match_score(candidate_rgb):
    if _COIN_TEMPLATE_DES is None or candidate_rgb is None or candidate_rgb.size == 0:
        return 0
    cand_bgr = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR)
    cand_gray = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2GRAY)

    th, tw = 190, 150
    ch, cw = cand_gray.shape[:2]
    if ch < 20 or cw < 20:
        return 0
    scale = ((th / ch) + (tw / cw)) / 2.0
    if 0.4 < scale < 3.0:
        cand_gray = cv2.resize(cand_gray, (int(cw * scale), int(ch * scale)))

    kp2, des2 = _COIN_ORB.detectAndCompute(cand_gray, None)
    if des2 is None or len(kp2) < 4:
        return 0

    matches = _COIN_BF_MATCHER.knnMatch(_COIN_TEMPLATE_DES, des2, k=2)
    good = 0
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good += 1
    return good


_COIN_HSV_LO = np.array([10, 25, 70])
_COIN_HSV_HI = np.array([45, 200, 245])

_RIM_HUE_LO, _RIM_HUE_HI, _RIM_VAL_MAX = 90, 150, 130

COIN_MIN_SCORE = 0.42

COIN_HIGH_CONF_COLOR_FILL = 0.55
COIN_HIGH_CONF_MAX_GAP    = 0.03


def _coin_circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1:
        return 0.0, area
    return float(4 * np.pi * area / (peri ** 2)), area


def _score_coin_candidate(gray, hsv, mask_color, cx, cy, r, img_area):
    h, w = gray.shape[:2]
    if r < 8:
        return 0.0, {"reject": "too_small"}

    circ_area = np.pi * r * r
    if circ_area < img_area * 0.0006 or circ_area > img_area * 0.12:
        return 0.0, {"reject": "size_out_of_range"}

    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r), min(h, cy + r)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0, {"reject": "out_of_bounds"}

    disc = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(disc, (cx - x0, cy - y0), r, 255, -1)
    disc_bool = disc > 0
    n_disc = int(disc_bool.sum())
    if n_disc < 30:
        return 0.0, {"reject": "too_few_px"}

    local_color_mask = mask_color[y0:y1, x0:x1] > 0
    color_fill = float((local_color_mask & disc_bool).sum()) / n_disc

    local_hsv = hsv[y0:y1, x0:x1]
    disc_pixels = local_hsv[disc_bool]
    sat_std = float(disc_pixels[:, 1].std())
    val_std = float(disc_pixels[:, 2].std())
    uniformity = 1.0 - min(1.0, (sat_std + val_std) / 2.0 / 90.0)

    local_gray = gray[y0:y1, x0:x1]
    very_dark = local_gray < 45
    dark_gap_frac = float((very_dark & disc_bool).sum()) / n_disc

    ring_outer = np.zeros_like(disc)
    cv2.circle(ring_outer, (cx - x0, cy - y0), r, 255, -1)
    ring_inner = np.zeros_like(disc)
    cv2.circle(ring_inner, (cx - x0, cy - y0), max(1, int(r * 0.82)), 255, -1)
    ring_band = (ring_outer > 0) & (ring_inner == 0)
    n_ring = int(ring_band.sum())
    rim_frac = 0.0
    if n_ring > 10:
        local_rim_mask = (
            (local_hsv[:, :, 0] >= _RIM_HUE_LO) & (local_hsv[:, :, 0] <= _RIM_HUE_HI) &
            (local_hsv[:, :, 2] <= _RIM_VAL_MAX)
        )
        rim_frac = float((local_rim_mask & ring_band).sum()) / n_ring

    score = (
        0.40 * color_fill +
        0.25 * uniformity +
        0.20 * max(0.0, 1.0 - dark_gap_frac * 4.0) +
        0.15 * min(1.0, rim_frac * 2.0)
    )

    details = {
        "color_fill": round(color_fill, 3), "uniformity": round(uniformity, 3),
        "dark_gap_frac": round(dark_gap_frac, 3), "rim_frac": round(rim_frac, 3),
        "score": round(score, 3),
    }

    if color_fill < 0.35:
        return 0.0, {**details, "reject": "low_color_fill"}
    if dark_gap_frac > 0.22:
        return 0.0, {**details, "reject": "too_gappy_likely_grain_cluster"}

    return score, details


def detect_coin_diameter_px(img_np, min_score=COIN_MIN_SCORE, debug=False):
    h, w = img_np.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    mask_color = cv2.inRange(hsv, _COIN_HSV_LO, _COIN_HSV_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, k, iterations=2)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, k, iterations=1)

    candidates = []

    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        circ, area = _coin_circularity(c)
        if area < img_area * 0.0006 or area > img_area * 0.12:
            continue
        if circ < 0.45:
            continue
        (ccx, ccy), cr = cv2.minEnclosingCircle(c)
        candidates.append((int(ccx), int(ccy), int(cr)))

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    min_r = max(10, int(min(h, w) * 0.015))
    max_r = max(50, int(min(h, w) * 0.12))
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 6,
        param1=80, param2=32, minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        for ccx, ccy, cr in np.round(circles[0]).astype(int):
            candidates.append((int(ccx), int(ccy), int(cr)))

    if not candidates:
        if debug:
            print("[coin] no candidates found -> coin not detected")
        return None, None

    scored = []
    for (ccx, ccy, cr) in candidates:
        score, details = _score_coin_candidate(gray, hsv, mask_color, ccx, ccy, cr, img_area)
        if debug:
            print(f"[coin] candidate cx={ccx} cy={ccy} r={cr} -> {details}")
        if score >= min_score:
            scored.append((score, ccx, ccy, cr, details))

    if not scored:
        if debug:
            print(f"[coin] no candidate cleared heuristic threshold {min_score} -> coin not detected")
        return None, None

    scored.sort(key=lambda t: -t[0])

    for score, cx, cy, r, details in scored:
        if (details.get("color_fill", 0) >= COIN_HIGH_CONF_COLOR_FILL and
                details.get("dark_gap_frac", 1) <= COIN_HIGH_CONF_MAX_GAP):
            if debug:
                print(f"[coin] ACCEPTED (high-confidence, no ORB needed) cx={cx} cy={cy} r={r} "
                      f"heuristic_score={score:.3f} color_fill={details['color_fill']:.3f} "
                      f"dark_gap_frac={details['dark_gap_frac']:.3f}")
            return 2.0 * r, (cx - r, cy - r, 2 * r, 2 * r)

    if _COIN_TEMPLATE_DES is None:
        score, cx, cy, r, details = scored[0]
        if debug:
            print(f"[coin] ORB template not loaded -> heuristic-only fallback "
                  f"ACCEPTED cx={cx} cy={cy} r={r} heuristic_score={score:.3f} "
                  f"(add the real _COIN_TEMPLATE_B64 for more robust confirmation)")
        return 2.0 * r, (cx - r, cy - r, 2 * r, 2 * r)

    for score, cx, cy, r, details in scored:
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r), min(h, cy + r)
        crop = img_np[y0:y1, x0:x1]
        orb_matches = orb_match_score(crop)
        if debug:
            print(f"[coin] ORB check (borderline) cx={cx} cy={cy} r={r} heuristic_score={score:.3f} "
                  f"orb_matches={orb_matches} (need >= {COIN_ORB_MIN_MATCHES})")
        if orb_matches >= COIN_ORB_MIN_MATCHES:
            if debug:
                print(f"[coin] ACCEPTED (borderline + ORB confirmed) cx={cx} cy={cy} r={r} "
                      f"heuristic_score={score:.3f} orb_matches={orb_matches}")
            return 2.0 * r, (cx - r, cy - r, 2 * r, 2 * r)

    if debug:
        print("[coin] no candidate reached high-confidence tier or passed ORB confirmation -> coin not detected")
    return None, None

def run_cellpose(img, max_iter, flow_threshold, cellprob_threshold):
    masks, flows, _ = cp_model.eval(
        img, niter=int(max_iter),
        flow_threshold=float(flow_threshold), cellprob_threshold=float(cellprob_threshold))
    return masks, flows


def measure_grains(masks, img_np, run_cnn=True, cnn_img_np=None, sr_scale_for_cnn=None,
                    cnn_model=None, cnn_transform=None, cnn_class_names=None, cnn_debug_dir=None,
                    save_debug_crops=None):
    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    slices    = find_objects(masks)
    results   = []

    use_polygon_cnn_source = cnn_img_np is not None and sr_scale_for_cnn is not None

    for idx, sl in enumerate(slices):
        if sl is None:
            continue
        label = idx + 1
        sub   = masks[sl]
        lys, lxs = np.where(sub == label)
        if len(lxs) < 5:
            continue

        gxs = (lxs + sl[1].start).astype(np.float32)
        gys = (lys + sl[0].start).astype(np.float32)
        pts  = np.column_stack([gxs, gys])
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), angle = rect
        h_px    = float(max(rw, rh))
        w_px    = float(min(rw, rh))
        area_px = float(len(lxs))
        hull    = cv2.convexHull(pts.astype(np.float32))
        hull_a  = cv2.contourArea(hull)
        solidity = area_px / hull_a if hull_a > 0 else 1.0

        gxs_int = gxs.astype(int)
        gys_int = gys.astype(int)

        results.append({
            "label": label, "h_px": h_px, "w_px": w_px,
            "aspect": round(h_px / w_px, 2) if w_px > 0 else 0,
            "area_px": area_px, "solidity": round(solidity, 3),
            "centroid_x": int(cx), "centroid_y": int(cy), "top_y": int(gys.min()),
            "rect_angle": angle,
            "_slice": sl,
            "_gxs_int": gxs_int, "_gys_int": gys_int, "_gray_ref": gray_full,
        })

    model_to_use   = cnn_model if cnn_model is not None else CNN_MODEL
    names_to_use   = cnn_class_names if cnn_class_names is not None else CNN_CLASS_NAMES
    debug_dir      = cnn_debug_dir if cnn_debug_dir is not None else CNN_DEBUG_CROPS_DIR
    do_save_debug  = save_debug_crops if save_debug_crops is not None else SAVE_CNN_DEBUG_CROPS

    if run_cnn and model_to_use is not None and results:
        if use_polygon_cnn_source:
            def _extract(g):
                # TIGHT exact-mask crop — matches the retrained classifier's inputs
                return crop_grain_tight_native(cnn_img_np, masks, g["label"],
                                               g["_slice"], sr_scale_for_cnn)
        else:
            def _extract(g):
                return crop_grain_from_mask_windowed(img_np, masks, g["label"], g["_slice"])

        crops = list(_EXECUTOR.map(_extract, results))

        valid_idx = [i for i, c in enumerate(crops) if c is not None]
        valid_crops = [crops[i] for i in valid_idx]
        cnn_preds = classify_crops_cnn(valid_crops, model=model_to_use, transform=cnn_transform)

        for vi, (cls_id, conf) in zip(valid_idx, cnn_preds):
            results[vi]["cnn_class"]      = names_to_use[cls_id]
            results[vi]["cnn_class_id"]   = cls_id
            results[vi]["cnn_confidence"] = round(conf, 4)

        if do_save_debug:
            os.makedirs(debug_dir, exist_ok=True)
            for vi, crop_bgr in zip(valid_idx, valid_crops):
                g = results[vi]
                cls_folder = os.path.join(debug_dir, g["cnn_class"])
                os.makedirs(cls_folder, exist_ok=True)
                fname = f"grain{g['label']:04d}_conf{g['cnn_confidence']:.2f}.png"
                cv2.imwrite(os.path.join(cls_folder, fname), crop_bgr)
            print(f"[debug] saved {len(valid_idx)} CNN input crop(s) -> {debug_dir}")

        for i, g in enumerate(results):
            if "cnn_class" not in g:
                g["cnn_class"]      = "n/a"
                g["cnn_class_id"]   = -1
                g["cnn_confidence"] = 0.0

    return results


def classify_sub_class(grain_data, coin_diam_px):
    if not grain_data:
        return grain_data, None

    has_mm = coin_diam_px is not None and coin_diam_px > 0

    for g in grain_data:
        g["height_mm"] = round(px_to_mm(g["h_px"], coin_diam_px), 3) if has_mm else None
        g["width_mm"]  = round(px_to_mm(g["w_px"], coin_diam_px), 3) if has_mm else None

    full_heights_mm = [g["height_mm"] for g in grain_data
                        if g.get("cnn_class_id") == FULL_ID and g["height_mm"] is not None]
    reference_mm = (max(full_heights_mm) if full_heights_mm
                     else (SUBCLASS_FALLBACK_REF_MM if has_mm else None))

    for g in grain_data:
        cid = g.get("cnn_class_id", -1)
        g["pct_of_ref"] = None

        if cid == FULL_ID:
            g["sub_class"] = FULL_CLASS_KEY
            continue

        if cid in OTHER_ID_TO_KEY:
            g["sub_class"] = OTHER_ID_TO_KEY[cid]
            continue

        if cid != BROKEN_ID or not has_mm:
            g["sub_class"] = "BR"
            continue

        pct = 100.0 * g["height_mm"] / reference_mm if reference_mm else 0.0
        g["pct_of_ref"] = round(pct, 1)

        cls = BROKEN_SUBCLASSES[-1]["key"]
        for i, bc in enumerate(BROKEN_SUBCLASSES):
            if i == 0 and pct > bc["max_pct"]:
                cls = bc["key"]
                break
            if bc["min_pct"] <= pct <= bc["max_pct"]:
                cls = bc["key"]
                break
        g["sub_class"] = cls

    return grain_data, reference_mm


def zoom_to_grains(img_np, masks, coin_bbox, pad_frac=0.03):
    ys, xs = np.where(masks > 0)
    if len(xs) == 0:
        return img_np, (0, 0)
    if coin_bbox is not None:
        cx0, cy0, cw, ch = coin_bbox
        m   = int(max(cw, ch) * 0.3)
        ok  = ~((xs >= cx0 - m) & (xs <= cx0 + cw + m) & (ys >= cy0 - m) & (ys <= cy0 + ch + m))
        xs_g, ys_g = xs[ok], ys[ok]
    else:
        xs_g, ys_g = xs, ys
    if len(xs_g) == 0:
        xs_g, ys_g = xs, ys
    H, W = img_np.shape[:2]
    pad  = int(max(H, W) * pad_frac)
    x0   = max(0, xs_g.min() - pad); x1 = min(W - 1, xs_g.max() + pad)
    y0   = max(0, ys_g.min() - pad); y1 = min(H - 1, ys_g.max() + pad)
    return img_np[y0:y1 + 1, x0:x1 + 1], (x0, y0)

def _base_canvas(img_np):
    c = np.clip(normalize99(img_np), 0, 1)
    c = (c * 255).astype(np.uint8)
    if c.ndim == 2:
        c = np.stack([c] * 3, axis=-1)
    return c


def save_raw_overlay(img_np, box, save_path):
    if not save_path:
        return None
    x0, y0, x1, y1 = box
    raw_canvas = _base_canvas(img_np)
    crop = raw_canvas[y0:y1, x0:x1]
    return save_compressed(crop, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

OUTLINE_THICK_FOCUS   = 4
OUTLINE_THICK_DIM     = 2
LABEL_FONT_SCALE      = 0.70
LABEL_FONT_THICKNESS  = 1
LEGEND_TITLE_SCALE    = 0.60
LEGEND_ROW_SCALE      = 0.58
LEGEND_SUB_SCALE      = 0.42


def _draw_grain_outline(canvas, masks, label, color, thickness=OUTLINE_THICK_FOCUS):
    mask_u8 = (masks == label).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (12, 12, 16), thickness + 2, cv2.LINE_AA)
    cv2.drawContours(canvas, contours, -1, color, thickness, cv2.LINE_AA)


def _blend_grain_mask(canvas, masks, label, color, alpha):
    if alpha <= 0:
        return
    m = masks == label
    if not np.any(m):
        return
    region = canvas[m].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    blended = region * (1.0 - alpha) + color_arr * alpha
    canvas[m] = np.clip(blended, 0, 255).astype(np.uint8)


def _draw_grain_outline_fast(canvas, masks, label, sl, color, thickness=OUTLINE_THICK_FOCUS):
    if sl is None:
        _draw_grain_outline(canvas, masks, label, color, thickness)
        return
    H, W = canvas.shape[:2]
    halo = thickness + 4
    y0 = max(0, sl[0].start - halo); y1 = min(H, sl[0].stop + halo)
    x0 = max(0, sl[1].start - halo); x1 = min(W, sl[1].stop + halo)
    if y1 <= y0 or x1 <= x0:
        return
    mask_u8 = (masks[y0:y1, x0:x1] == label).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    sub_canvas = canvas[y0:y1, x0:x1]
    cv2.drawContours(sub_canvas, contours, -1, (12, 12, 16), thickness + 2, cv2.LINE_AA)
    cv2.drawContours(sub_canvas, contours, -1, color, thickness, cv2.LINE_AA)


def _blend_grain_mask_fast(canvas, masks, label, sl, color, alpha):
    if alpha <= 0:
        return
    if sl is None:
        _blend_grain_mask(canvas, masks, label, color, alpha)
        return
    local_m = masks[sl] == label
    if not np.any(local_m):
        return
    sub_canvas = canvas[sl]
    region = sub_canvas[local_m].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    blended = region * (1.0 - alpha) + color_arr * alpha
    sub_canvas[local_m] = np.clip(blended, 0, 255).astype(np.uint8)


def _place_label(canvas, text, anchor_xy, color, placed_boxes,
                  font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=LABEL_FONT_SCALE,
                  thickness=LABEL_FONT_THICKNESS,
                  pad=5, bg_alpha=0.88, text_color=(255, 255, 255)):
    H, W = canvas.shape[:2]
    ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w, box_h = tw + pad * 2, th + baseline + pad * 2

    offsets = [(0, -6 - th)]
    for dist in (4, 16, 30, 46, 64, 84, 106):
        offsets += [
            (0, dist + th), (dist, -th), (-box_w - dist, -th),
            (dist, -dist - th), (-box_w - dist, -dist - th),
            (dist, dist), (-box_w - dist, dist),
            (0, -dist - th - 10),
        ]

    def _overlaps(bx):
        x0, y0, x1, y1 = bx
        for px0, py0, px1, py1 in placed_boxes:
            if x0 < px1 and x1 > px0 and y0 < py1 and y1 > py0:
                return True
        return False

    chosen = None
    for ox, oy in offsets:
        x0 = ax + ox
        y0 = ay + oy
        x1, y1 = x0 + box_w, y0 + box_h
        if x0 < 2 or y0 < 2 or x1 > W - 2 or y1 > H - 2:
            continue
        if not _overlaps((x0, y0, x1, y1)):
            chosen = (x0, y0, x1, y1)
            break

    if chosen is None:
        x0 = int(np.clip(ax - box_w // 2, 2, max(2, W - box_w - 2)))
        y0 = int(np.clip(ay - th - 10, 2, max(2, H - box_h - 2)))
        chosen = (x0, y0, x0 + box_w, y0 + box_h)

    x0, y0, x1, y1 = chosen
    sub = canvas[y0:y1, x0:x1]
    if sub.shape[0] == box_h and sub.shape[1] == box_w:
        bg = np.full_like(sub, (14, 14, 18))
        canvas[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - bg_alpha, bg, bg_alpha, 0)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (x0 + pad, y1 - pad - baseline), font, font_scale,
                text_color, thickness, cv2.LINE_AA)

    lcx, lcy = (x0 + x1) // 2, (y0 + y1) // 2
    if (lcx - ax) ** 2 + (lcy - ay) ** 2 > 26 ** 2:
        ex = x0 if ax < x0 else (x1 if ax > x1 else ax)
        ey = y0 if ay < y0 else (y1 if ay > y1 else ay)
        cv2.line(canvas, (ax, ay), (ex, ey), color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (ax, ay), 3, color, -1, cv2.LINE_AA)

    placed_boxes.append(chosen)
    return chosen


def _solid_panel(canvas, x0, y0, x1, y1, border_color=(140, 140, 156)):
    sub = canvas[y0:y1, x0:x1]
    bg  = np.full_like(sub, (14, 17, 24))
    canvas[y0:y1, x0:x1] = cv2.addWeighted(sub, 0.04, bg, 0.96, 0)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border_color, 2, cv2.LINE_AA)


def _alpha_for_class_id(cid):
    name = CNN_CLASS_NAMES.get(cid, "")
    return MASK_FILL_ALPHA_BY_CLASS.get(name, MASK_FILL_ALPHA)


def _outline_target_class_id():
    for cid, name in CNN_DISPLAY_NAME_BY_ID.items():
        if name == OUTLINE_ONLY_CLASS_NAME:
            return cid
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Combined view
# ══════════════════════════════════════════════════════════════════════════════

def build_combined_image(img_np, masks, grain_data, coin_bbox, save_path,
                          slice_by_label=None, raw_save_path=None):
    canvas = _base_canvas(img_np)
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    slice_by_label = slice_by_label or {}

    def _color_for(cid):
        return CNN_COLOR_RGB.get(cid, DIM_COLOR_RGB)

    for g in grain_data:
        cid = g.get("cnn_class_id", -1)
        _blend_grain_mask_fast(canvas, masks, g["label"], slice_by_label.get(g["label"]),
                                _color_for(cid), _alpha_for_class_id(cid))

    outline_cid = _outline_target_class_id() if SHOW_GRAIN_OUTLINE else None
    if outline_cid is not None:
        for g in grain_data:
            if g.get("cnn_class_id", -1) == outline_cid:
                _draw_grain_outline_fast(canvas, masks, g["label"], slice_by_label.get(g["label"]),
                                          REJECTED_OUTLINE_COLOR_RGB, OUTLINE_THICK_FOCUS)

    if coin_bbox is not None:
        cx0, cy0, cw, ch = coin_bbox
        cv2.rectangle(canvas, (cx0, cy0), (cx0 + cw, cy0 + ch), C_COIN_BOX, 2, cv2.LINE_AA)
        cv2.putText(canvas, "5 PKR coin", (cx0, max(14, cy0 - 8)), fnt, 0.5, C_COIN_BOX, 2, cv2.LINE_AA)

    n_full     = sum(1 for g in grain_data if g.get("cnn_class_id") == FULL_ID)
    n_broken   = sum(1 for g in grain_data if g.get("cnn_class_id") == BROKEN_ID)
    total_fb   = n_full + n_broken
    pct_broken = 100.0 * n_broken / total_fb if total_fb else 0.0
    total      = len(grain_data)

    legend_rows = [(FULL_ID, FULL_CLASS_NAME, FULL_CLASS_COLOR_RGB, n_full),
                   (BROKEN_ID, BROKEN_CLASS_NAME, BROKEN_CLASS_COLOR_RGB, n_broken)]
    for oc in OTHER_CLASSES:
        cid = CNN_NAME_TO_ID.get(oc["key"].lower())
        n = sum(1 for g in grain_data if g.get("cnn_class_id") == cid)
        legend_rows.append((cid, oc["name"], oc["color_rgb"], n))

    leg_x, leg_y = 14, 14
    leg_w = 330
    row_h = 40
    leg_h = 46 + row_h * len(legend_rows) + 34
    _solid_panel(canvas, leg_x, leg_y, leg_x + leg_w, leg_y + leg_h)

    cv2.putText(canvas, "All classes: Full / Broken / Weak / Rejection / Fatty",
                (leg_x + 16, leg_y + 26), fnt, 0.52, (245, 245, 250), 2, cv2.LINE_AA)
    cv2.line(canvas, (leg_x + 12, leg_y + 38), (leg_x + leg_w - 12, leg_y + 38),
              (90, 90, 100), 1, cv2.LINE_AA)

    ly = leg_y + 68
    sw = 28
    for cid, name, color, n in legend_rows:
        cv2.rectangle(canvas, (leg_x + 16, ly - 20), (leg_x + 16 + sw, ly + 4), color, -1)
        cv2.rectangle(canvas, (leg_x + 16, ly - 20), (leg_x + 16 + sw, ly + 4), (235, 235, 235), 2, cv2.LINE_AA)
        tag = "  (outlined)" if (outline_cid is not None and cid == outline_cid) else ""
        cv2.putText(canvas, f"{name}  —  {n}{tag}", (leg_x + 16 + sw + 14, ly),
                    fnt, LEGEND_ROW_SCALE, (245, 245, 250), 2, cv2.LINE_AA)
        ly += row_h

    cv2.putText(canvas, f"Total {total}   |   Broken {pct_broken:.1f}% of Full+Broken",
                (leg_x + 16, ly), fnt, LEGEND_SUB_SCALE, (205, 205, 215), 1, cv2.LINE_AA)

    placed_boxes = [(leg_x, leg_y, leg_x + leg_w, leg_y + leg_h)]

    if SHOW_GRAIN_LABELS:
        labelled = list(grain_data)
        labelled.sort(key=lambda g: -g["area_px"])
        for g in labelled:
            cid = g.get("cnn_class_id", -1)
            color = _color_for(cid)
            text = CNN_DISPLAY_NAME_BY_ID.get(cid, "?")
            anchor = (g["centroid_x"], g["top_y"] - 4)
            _place_label(canvas, text, anchor, color, placed_boxes)

    cropped, (ox, oy) = zoom_to_grains(canvas, masks, coin_bbox)
    ch_, cw_ = cropped.shape[:2]
    raw_path = save_raw_overlay(img_np, (ox, oy, ox + cw_, oy + ch_), raw_save_path)
    return save_compressed(cropped, save_path), n_full, n_broken, pct_broken, raw_path


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Sub-classification
# ══════════════════════════════════════════════════════════════════════════════

_OTHER_KEYS = {oc["key"] for oc in OTHER_CLASSES}


def build_class_image(img_np, masks, grain_data, coin_bbox, save_path, coin_diam_px,
                       ref_height_mm, slice_by_label=None, raw_save_path=None):
    canvas = _base_canvas(img_np)
    H_img, W_img = canvas.shape[:2]
    has_mm = coin_diam_px is not None
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    slice_by_label = slice_by_label or {}

    def _color_for(sc):
        return DIM_COLOR_RGB if sc in _OTHER_KEYS else SUB_COLORS.get(sc, (200, 200, 200))

    for g in grain_data:
        sc = g.get("sub_class", BROKEN_SUBCLASSES[-1]["key"])
        if sc == FULL_CLASS_KEY:
            continue
        _blend_grain_mask_fast(canvas, masks, g["label"], slice_by_label.get(g["label"]),
                                _color_for(sc), MASK_FILL_ALPHA)

    broken_keys = [bc["key"] for bc in BROKEN_SUBCLASSES]
    focus_classes = broken_keys + ["BR"]

    if SHOW_GRAIN_OUTLINE:
        for g in grain_data:
            sc = g.get("sub_class", BROKEN_SUBCLASSES[-1]["key"])
            if sc == FULL_CLASS_KEY:
                continue
            thick = OUTLINE_THICK_FOCUS if sc in focus_classes else OUTLINE_THICK_DIM
            _draw_grain_outline_fast(canvas, masks, g["label"], slice_by_label.get(g["label"]),
                                      _color_for(sc), thick)

    ref_grain = None
    if has_mm and ref_height_mm:
        full_grains = [g for g in grain_data if g.get("cnn_class_id") == FULL_ID]
        if full_grains:
            ref_grain = max(full_grains, key=lambda g: g.get("height_mm", 0) or 0)

    if coin_bbox is not None:
        cx0, cy0, cw, ch = coin_bbox
        cv2.rectangle(canvas, (cx0, cy0), (cx0 + cw, cy0 + ch), C_COIN_BOX, 2, cv2.LINE_AA)

    cnt   = {c: sum(1 for g in grain_data if g.get("sub_class") == c) for c in SUB_CLASS_ORDER}
    total = len(grain_data)
    rows_to_show = [c for c in SUB_CLASS_ORDER
                     if c not in _OTHER_KEYS and (c != "BR" or cnt[c] > 0)]

    label_for = {FULL_CLASS_KEY: f"{FULL_CLASS_NAME} grain (not broken) -- shown unmasked",
                 "BR": "Broken grain, no coin -> n/a"}
    for bc in BROKEN_SUBCLASSES:
        label_for[bc["key"]] = f"Broken, {bc['min_pct']:.0f}-{bc['max_pct']:.0f}% of ref height"

    title_h, row_h, pad_in, panel_w = 42, 46, 16, 360
    footer_h = 30 if (not has_mm or ref_height_mm) else 0
    panel_h  = title_h + row_h * len(rows_to_show) + footer_h + pad_in
    px0, py0 = 14, 14
    px1, py1 = px0 + panel_w, py0 + panel_h
    _solid_panel(canvas, px0, py0, px1, py1)

    broken_label = "/".join(bc["key"] for bc in BROKEN_SUBCLASSES)
    cv2.putText(canvas, f"Sub-class: {broken_label}  ({FULL_CLASS_NAME} shown unmasked)", (px0 + 16, py0 + 28),
                fnt, LEGEND_TITLE_SCALE, (245, 245, 250), 2, cv2.LINE_AA)
    cv2.line(canvas, (px0 + 12, py0 + title_h), (px1 - 12, py0 + title_h), (90, 90, 100), 1, cv2.LINE_AA)

    ly = py0 + title_h + 32
    sw = 30
    for cls in rows_to_show:
        col = SUB_COLORS[cls]
        cv2.rectangle(canvas, (px0 + 16, ly - 22), (px0 + 16 + sw, ly + 4), col, -1)
        cv2.rectangle(canvas, (px0 + 16, ly - 22), (px0 + 16 + sw, ly + 4), (235, 235, 235), 2, cv2.LINE_AA)
        if cls == FULL_CLASS_KEY:
            cv2.line(canvas, (px0 + 16, ly - 9), (px0 + 16 + sw, ly - 9), (14, 17, 24), 2, cv2.LINE_AA)
        pct_str = f"{100 * cnt[cls] / total:.0f}%" if total else "0%"
        cv2.putText(canvas, f"{SUB_NAMES.get(cls, cls)}  —  {cnt[cls]}  ({pct_str})", (px0 + 16 + sw + 14, ly),
                    fnt, LEGEND_ROW_SCALE, (245, 245, 250), 2, cv2.LINE_AA)
        cv2.putText(canvas, label_for[cls], (px0 + 16 + sw + 14, ly + 18),
                    fnt, LEGEND_SUB_SCALE, (185, 185, 195), 1, cv2.LINE_AA)
        ly += row_h

    if not has_mm:
        cv2.putText(canvas, f"No coin detected -> {broken_label} skipped", (px0 + 16, ly + 8),
                    fnt, LEGEND_SUB_SCALE, (250, 204, 21), 1, cv2.LINE_AA)
    elif ref_height_mm:
        ref_note = "tallest full grain" if ref_grain is not None else f"{SUBCLASS_FALLBACK_REF_MM:.1f}mm fallback (no full grains)"
        cv2.putText(canvas, f"Ref height: {ref_height_mm:.1f}mm ({ref_note})", (px0 + 16, ly + 8),
                    fnt, LEGEND_SUB_SCALE, (196, 181, 253), 1, cv2.LINE_AA)

    n_other = sum(1 for g in grain_data if g.get("sub_class") in _OTHER_KEYS)
    if n_other:
        ly += 26
        cv2.putText(canvas, f"Rejected/Weak/Fatty: {n_other} (dimmed -- see Tab 1)", (px0 + 16, ly + 8),
                    fnt, LEGEND_SUB_SCALE, (175, 175, 175), 1, cv2.LINE_AA)

    placed_boxes = [(px0, py0, px1, py1)]

    if SHOW_GRAIN_LABELS:
        labelled = [g for g in grain_data if g.get("sub_class") in focus_classes]
        labelled.sort(key=lambda g: -g["area_px"])
        for g in labelled:
            sc  = g["sub_class"]
            col = SUB_COLORS.get(sc, (200, 200, 200))
            if has_mm and g.get("height_mm") is not None and sc != "BR":
                text = f"{SUB_NAMES.get(sc, sc)} {g['height_mm']:.1f}mm"
            else:
                text = SUB_NAMES.get(sc, sc)
            anchor = (g["centroid_x"], g["top_y"] - 4)
            _place_label(canvas, text, anchor, col, placed_boxes)

    if SHOW_GRAIN_LABELS and ref_grain is not None:
        rx, ry = ref_grain["centroid_x"], max(12, ref_grain["top_y"] - 10)
        _place_label(canvas, f"REF {ref_grain['height_mm']:.1f}mm", (rx, ry), (196, 181, 253), placed_boxes)

    cropped, (ox, oy) = zoom_to_grains(canvas, masks, coin_bbox)
    ch, cw = cropped.shape[:2]
    x0, y0, x1, y1 = ox, oy, ox + cw, oy + ch
    panel_x0, panel_y0 = max(0, px0 - 6), max(0, py0 - 6)
    panel_x1, panel_y1 = min(W_img, px1 + 6), min(H_img, py1 + 6)
    x0 = min(x0, panel_x0); y0 = min(y0, panel_y0)
    x1 = max(x1, panel_x1); y1 = max(y1, panel_y1)
    if (x0, y0, x1, y1) != (ox, oy, ox + cw, oy + ch):
        cropped = canvas[y0:y1, x0:x1]

    raw_path = save_raw_overlay(img_np, (x0, y0, x1, y1), raw_save_path)
    return save_compressed(cropped, save_path), raw_path


def build_chart(grain_data, ref_height_mm, coin_diam_px, save_path):
    total  = len(grain_data)
    has_mm = coin_diam_px is not None
    unit   = "mm" if has_mm else "px"
    def u(px): return px_to_mm(px, coin_diam_px) if has_mm else px
    sc = {c: sum(1 for g in grain_data if g.get("sub_class") == c) for c in SUB_CLASS_ORDER}

    broken_keys = [bc["key"] for bc in BROKEN_SUBCLASSES]
    other_keys  = [oc["key"] for oc in OTHER_CLASSES]
    chart_order = ([FULL_CLASS_KEY] + broken_keys if has_mm else [FULL_CLASS_KEY, "BR"]) + other_keys
    chart_labels = [SUB_NAMES.get(c, c) for c in chart_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0F172A")
    ax = axes[0]; ax.set_facecolor("#1E293B")
    x_pos = np.arange(len(chart_order))
    bars = ax.bar(x_pos, [sc[c] for c in chart_order],
                  color=[tuple(v / 255 for v in SUB_COLORS[c]) for c in chart_order],
                  edgecolor="#94A3B8", lw=0.8, width=0.55)
    ax.set_xticks(x_pos); ax.set_xticklabels(chart_labels, fontweight="bold")
    for bar, cls in zip(bars, chart_order):
        n = sc[cls]; pct = 100 * n / total if total else 0
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{n}\n({pct:.0f}%)", ha="center", va="bottom",
                    color="#F1F5F9", fontsize=10, fontweight="bold")
    ax.set_title("Grain Class Counts — all classes", color="#F1F5F9", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", color="#CBD5E1", fontsize=10)
    ax.tick_params(colors="#94A3B8", labelsize=11)
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    ax.set_ylim(0, max([sc[c] for c in chart_order], default=1) * 1.35 + 2)
    ax.grid(axis="y", color="#334155", lw=0.5, linestyle=":")

    ax2 = axes[1]; ax2.set_facecolor("#1E293B")
    bins = min(25, max(5, total // 4))
    for cls, lbl in zip(chart_order, chart_labels):
        ll = [u(g["h_px"]) for g in grain_data if g.get("sub_class") == cls]
        if ll:
            cp = tuple(v / 255 for v in SUB_COLORS[cls])
            ax2.hist(ll, bins=bins, color=cp, alpha=0.78, label=f"{lbl} ({len(ll)})",
                     edgecolor="#1E293B", lw=0.4)

    if has_mm and ref_height_mm:
        for bc in BROKEN_SUBCLASSES:
            thr_mm = bc["min_pct"] / 100.0 * ref_height_mm
            col_hex = bc["color_hex"]
            ax2.axvline(thr_mm, color=col_hex, lw=1.5, linestyle="--", alpha=0.85,
                        label=f"{bc['name']} floor ({bc['min_pct']:.0f}%)={thr_mm:.1f}{unit}")
        ax2.axvline(ref_height_mm, color="#A78BFA", lw=2.0, linestyle="-.",
                    label=f"Ref ({FULL_CLASS_NAME} grain)={ref_height_mm:.1f}{unit}")
    ax2.set_title(f"Length Distribution ({unit})", color="#F1F5F9", fontsize=13, fontweight="bold")
    ax2.set_xlabel(f"Length ({unit})", color="#CBD5E1", fontsize=10)
    ax2.set_ylabel("Count", color="#CBD5E1", fontsize=10)
    ax2.tick_params(colors="#94A3B8", labelsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor("#334155")
    ax2.legend(facecolor="#1E293B", labelcolor="white", fontsize=8, loc="upper right", framealpha=0.9)
    ax2.grid(axis="y", color="#334155", lw=0.5, linestyle=":")

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
    return save_path


MERGE_SOLIDITY_FLAG_THRESHOLD = 0.82


# ══════════════════════════════════════════════════════════════════════════════
#  Table building — trimmed measurement table + Urdu summary table
# ══════════════════════════════════════════════════════════════════════════════

_URDU_SUMMARY_LABELS = {
    "total":      "کل دانے",
    "full":       "مکمل دانے (Full)",
    "broken":     "ٹوٹے ہوئے دانے (Broken)",
    "weak":       "کمزور دانے (Weak)",
    "fatty":      "چوبا (Fatty)",
    "rejected":   "ریجیکشن",
    "sub_header": "— ٹوٹے ہوئے دانوں کی اقسام —",
    "sg":            "Short Grain",
    "b1":            "B1",
    "b2":            "B2",
    "g1":            "گولی (G1)",
    "unclassified":  "    • غیر متعین (کوائن نہیں ملا)",
    "metric_col":    "میٹرک",
    "value_col":     "ویلیو",
}


def build_tables(grain_data, ref_height_mm, coin_diam_px):
    has_mm = coin_diam_px is not None
    lc = "Length (mm)" if has_mm else "Length (px)"
    wc = "Width (mm)"  if has_mm else "Width (px)"

    rows = []
    for g in grain_data:
        h_mm = px_to_mm(g["h_px"], coin_diam_px) if has_mm else None
        w_mm = px_to_mm(g["w_px"], coin_diam_px) if has_mm else None
        cnn_cls   = g.get("cnn_class", "n/a")
        rice_type = cnn_cls.capitalize() if cnn_cls != "n/a" else "N/A"

        sub_key = g.get("sub_class", "")
        if sub_key in _BROKEN_SUBCLASS_KEYS or sub_key == "BR":
            broken_sub_cls = SUB_NAMES.get(sub_key, sub_key)
        else:
            broken_sub_cls = "—"

        rows.append({
            "Ref #": g["label"],
            lc: round(h_mm, 2) if h_mm else round(g["h_px"], 1),
            wc: round(w_mm, 2) if w_mm else round(g["w_px"], 1),
            "Aspect Ratio": round(g["aspect"], 2),
            "Rice Type": rice_type,
            "Broken Sub-class (SG/B1/B2/G1)": broken_sub_cls,
        })
    grain_df = pd.DataFrame(rows)

    total = len(grain_data)

    def _pct(n):
        return f"{n} ({100 * n / total:.1f}%)" if total else "0"

    n_full     = sum(1 for g in grain_data if g.get("sub_class") == FULL_CLASS_KEY)
    n_br       = sum(1 for g in grain_data if g.get("sub_class") == "BR")
    broken_subclass_keys = [bc["key"] for bc in BROKEN_SUBCLASSES]
    n_broken_total = sum(1 for g in grain_data if g.get("sub_class") in broken_subclass_keys) + n_br
    n_weak     = sum(1 for g in grain_data if g.get("sub_class") == "WEAK")
    n_fatty    = sum(1 for g in grain_data if g.get("sub_class") == "FATTY")
    n_rejected = sum(1 for g in grain_data if g.get("sub_class") == "REJECTED")

    L = _URDU_SUMMARY_LABELS
    srows = [
        {L["metric_col"]: L["total"],    L["value_col"]: str(total)},
        {L["metric_col"]: L["full"],     L["value_col"]: _pct(n_full)},
        {L["metric_col"]: L["broken"],   L["value_col"]: _pct(n_broken_total)},
        {L["metric_col"]: L["weak"],     L["value_col"]: _pct(n_weak)},
        {L["metric_col"]: L["fatty"],    L["value_col"]: _pct(n_fatty)},
        {L["metric_col"]: L["rejected"], L["value_col"]: _pct(n_rejected)},
        {L["metric_col"]: L["sub_header"], L["value_col"]: ""},
    ]

    _sub_label_by_key = {"SG": L["sg"], "B1": L["b1"], "B2": L["b2"], "G1": L["g1"]}
    for bc in BROKEN_SUBCLASSES:
        n = sum(1 for g in grain_data if g.get("sub_class") == bc["key"])
        label = _sub_label_by_key.get(bc["key"], bc["name"])
        srows.append({L["metric_col"]: f"    • {label}", L["value_col"]: _pct(n)})

    if n_br:
        srows.append({L["metric_col"]: L["unclassified"], L["value_col"]: _pct(n_br)})

    summary_df = pd.DataFrame(srows)
    return grain_df, summary_df


def _pil_text_size(draw, text, font):
    if not text:
        return 0, 0
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def dataframe_to_table_image(df: pd.DataFrame, title: str) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[("" if v is None else str(v)) for v in row] for row in df.astype(object).values.tolist()]

    BG            = (15, 23, 42)
    HEADER_BG     = (49, 46, 129)
    ROW_BG_A      = (30, 41, 59)
    ROW_BG_B      = (17, 24, 39)
    GRID_COLOR    = (51, 65, 85)
    HEADER_TEXT   = (241, 245, 249)
    CELL_TEXT     = (226, 232, 240)
    TITLE_TEXT    = (241, 245, 249)

    TITLE_SIZE  = 24
    HEADER_SIZE = 18
    CELL_SIZE   = 17
    ROW_H       = 44
    HEADER_H    = 50
    TITLE_H     = 60
    CELL_PAD_X  = 20
    OUTER_PAD   = 14

    tmp = Image.new("RGB", (10, 10))
    d0  = ImageDraw.Draw(tmp)

    col_widths = []
    for ci, col in enumerate(cols):
        w, _ = _pil_text_size(d0, col, _font_for_text(col, HEADER_SIZE))
        for r in rows:
            cw, _ = _pil_text_size(d0, r[ci], _font_for_text(r[ci], CELL_SIZE))
            w = max(w, cw)
        col_widths.append(w + CELL_PAD_X * 2)

    table_w = sum(col_widths)
    total_w = table_w + OUTER_PAD * 2
    total_h = OUTER_PAD + TITLE_H + HEADER_H + ROW_H * len(rows) + OUTER_PAD

    img = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(img)

    title_font = _font_for_text(title, TITLE_SIZE)
    tw, th = _pil_text_size(draw, title, title_font)
    draw.text(((total_w - tw) / 2, OUTER_PAD + (TITLE_H - th) / 2), title,
               font=title_font, fill=TITLE_TEXT)

    x0 = OUTER_PAD
    y = OUTER_PAD + TITLE_H

    draw.rectangle([x0, y, x0 + table_w, y + HEADER_H], fill=HEADER_BG)
    x = x0
    for ci, col in enumerate(cols):
        font = _font_for_text(col, HEADER_SIZE)
        tw, th = _pil_text_size(draw, col, font)
        draw.text((x + (col_widths[ci] - tw) / 2, y + (HEADER_H - th) / 2 - 1),
                   col, font=font, fill=HEADER_TEXT)
        x += col_widths[ci]
    y += HEADER_H

    for ri, row in enumerate(rows):
        bg = ROW_BG_A if ri % 2 == 0 else ROW_BG_B
        draw.rectangle([x0, y, x0 + table_w, y + ROW_H], fill=bg)
        x = x0
        for ci, val in enumerate(row):
            font = _font_for_text(val, CELL_SIZE)
            tw, th = _pil_text_size(draw, val, font)
            draw.text((x + (col_widths[ci] - tw) / 2, y + (ROW_H - th) / 2 - 1),
                       val, font=font, fill=CELL_TEXT)
            x += col_widths[ci]
        y += ROW_H

    x = x0
    top = OUTER_PAD + TITLE_H
    bottom = y
    for w in col_widths[:-1]:
        x += w
        draw.line([(x, top), (x, bottom)], fill=GRID_COLOR, width=1)
    draw.line([(x0, top + HEADER_H), (x0 + table_w, top + HEADER_H)], fill=GRID_COLOR, width=2)
    draw.rectangle([x0, top, x0 + table_w, bottom], outline=GRID_COLOR, width=1)

    out_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    img.save(out_path)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _quality_mode_captions():
    L = _URDU_SUMMARY_LABELS

    def _chip(key, hex_color, fallback_name):
        label = L.get(key.lower(), fallback_name)
        return (f'<span style="color:{hex_color};font-weight:700;'
            f'direction:ltr;unicode-bidi:isolate;">■ {label}</span>')

    combined_html = f'''<p class="compact-legend">
        {_chip(FULL_CLASS_KEY, FULL_CLASS_COLOR_HEX, FULL_CLASS_NAME)}
        &nbsp;&nbsp;
        {_chip(BROKEN_CLASS_KEY, BROKEN_CLASS_COLOR_HEX, BROKEN_CLASS_NAME)}
        &nbsp;&nbsp;
        {"&nbsp;&nbsp;".join(_chip(oc["key"], oc["color_hex"], oc["name"]) for oc in OTHER_CLASSES)}
    </p>'''
    class_html = f'''<p class="compact-legend">
        {"&nbsp;&nbsp;".join(
            f'<span style="color:{bc["color_hex"]};font-weight:700;">■ {L.get(bc["key"].lower(), bc["name"])}</span>'
            f'&nbsp;{bc["min_pct"]:.0f}-{bc["max_pct"]:.0f}%'
            for bc in BROKEN_SUBCLASSES
        )}
        &nbsp;&nbsp;<span style="font-weight:700;color:#CBD5E1;">□ {L["full"]} (unmasked)</span>
    </p>'''
    return combined_html, class_html


def _variety_mode_captions():
    legend_bits = "&nbsp;&nbsp;".join(
        f'<span style="color:{CNN_COLOR_HEX_2.get(cid, "#6E6E6E")};font-weight:700;">■ {name}</span>'
        for cid, name in sorted(CNN_DISPLAY_NAME_BY_ID_2.items())
    )
    html = f'''<p class="compact-legend">
        {legend_bits if legend_bits else "<i>7-class variety model not loaded.</i>"}
        &nbsp;&nbsp;<span style="color:#9CA3AF;font-weight:700;">■ خارج شدہ (Excluded, not {FULL_CLASS_NAME})</span>
    </p>'''
    return html, html


def run_analysis(single_img):
    if single_img is None:
        raise gr.Error("Upload an image first.")

    single_img = convert_heic_if_needed(single_img)

    sr_scale = 2
    base     = os.path.splitext(single_img)[0]

    fp        = tif_view(single_img)
    img_input = imread(fp)
    orig_h, orig_w = img_input.shape[:2]

    img_enh   = enhance_resolution(img_input)
    img_model = image_resize(img_enh, resize=1600)
    masks_model, _ = run_cellpose(img_model, 250, 0.4, 0.0)
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}  {masks_model.max()} grains")

    masks_orig = cv2.resize(masks_model.astype("uint16"), (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST).astype("uint16")

    coin_diam_model, coin_bbox_model = detect_coin_diameter_px(img_model, debug=True)
    if coin_diam_model is not None:
        sx = orig_w / img_model.shape[1]
        sy = orig_h / img_model.shape[0]
        coin_diam_px = coin_diam_model * (sx + sy) / 2
        bx, by, bw, bh = coin_bbox_model
        coin_bbox = (int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy))
    else:
        coin_diam_px = None; coin_bbox = None

    if sr_scale > 1:
        img_analysis = enhance_resolution(
            cv2.resize(img_input, (orig_w * sr_scale, orig_h * sr_scale), interpolation=cv2.INTER_LANCZOS4))
        masks_analysis = cv2.resize(masks_model.astype("uint16"), (orig_w * sr_scale, orig_h * sr_scale),
                                    interpolation=cv2.INTER_NEAREST).astype("uint16")
    else:
        img_analysis   = img_enh
        masks_analysis = masks_orig

    raw_grains = measure_grains(masks_analysis, img_analysis, run_cnn=True,
                                 cnn_img_np=img_input, sr_scale_for_cnn=sr_scale)

    MIN_RELIABLE_NATIVE_GRAIN_PX = 80
    if raw_grains:
        native_long_axis_px = [g["h_px"] / sr_scale for g in raw_grains]
        avg_native_px = sum(native_long_axis_px) / len(native_long_axis_px)
        frac_below = sum(1 for p in native_long_axis_px if p < MIN_RELIABLE_NATIVE_GRAIN_PX) / len(native_long_axis_px)
        if avg_native_px < MIN_RELIABLE_NATIVE_GRAIN_PX:
            print(f"[quality] ⚠ LOW RESOLUTION WARNING: average grain is only "
                  f"~{avg_native_px:.0f}px (long axis) at native camera resolution "
                  f"({frac_below:.0%} of grains below the ~{MIN_RELIABLE_NATIVE_GRAIN_PX}px "
                  f"guideline). {img_input.shape[1]}x{img_input.shape[0]} photo with "
                  f"{len(raw_grains)} grains means each grain gets very few real "
                  f"camera pixels -- classification accuracy is likely reduced.")
        else:
            print(f"[quality] Average grain native resolution: ~{avg_native_px:.0f}px "
                  f"(long axis) -- OK")

    if coin_bbox is not None:
        cx0, cy0, cw, ch = coin_bbox
        margin = int(max(cw, ch) * 0.35)
        raw_grains = [
            g for g in raw_grains
            if not (cx0 - margin <= g["centroid_x"] // sr_scale <= cx0 + cw + margin
                    and cy0 - margin <= g["centroid_y"] // sr_scale <= cy0 + ch + margin)
        ]

    for g in raw_grains:
        g["h_px"]       /= sr_scale
        g["w_px"]       /= sr_scale
        g["area_px"]    /= sr_scale ** 2
        g["centroid_x"]  = int(g["centroid_x"] / sr_scale)
        g["centroid_y"]  = int(g["centroid_y"] / sr_scale)
        g["top_y"]       = int(g["top_y"]      / sr_scale)
        g["_gxs_int"]   = (g["_gxs_int"] / sr_scale).astype(int)
        g["_gys_int"]   = (g["_gys_int"] / sr_scale).astype(int)

    grain_data, ref_height_mm = classify_sub_class(raw_grains, coin_diam_px)

    if sr_scale > 1:
        disp_img   = cv2.resize(img_input, (orig_w * sr_scale, orig_h * sr_scale), interpolation=cv2.INTER_LANCZOS4)
        disp_masks = cv2.resize(masks_orig.astype("uint16"), (orig_w * sr_scale, orig_h * sr_scale),
                                interpolation=cv2.INTER_NEAREST).astype("uint16")
        dc = None if coin_bbox is None else (
            coin_bbox[0] * sr_scale, coin_bbox[1] * sr_scale, coin_bbox[2] * sr_scale, coin_bbox[3] * sr_scale)
        disp_grains = []
        for g in grain_data:
            dg = dict(g)
            dg["centroid_x"] *= sr_scale; dg["centroid_y"] *= sr_scale; dg["top_y"] *= sr_scale
            dg["w_px"] *= sr_scale; dg["h_px"] *= sr_scale
            dg["_gxs_int"] = (dg["_gxs_int"] * sr_scale); dg["_gys_int"] = (dg["_gys_int"] * sr_scale)
            disp_grains.append(dg)
    else:
        disp_img, disp_masks, dc, disp_grains = img_input, masks_orig, coin_bbox, grain_data

    disp_slices = find_objects(disp_masks)
    disp_slice_by_label = {i + 1: sl for i, sl in enumerate(disp_slices) if sl is not None}

    ts             = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_combined = base + f"_{ts}_combined.jpg"
    fname_class    = base + f"_{ts}_class.jpg"
    fname_chart    = base + f"_{ts}_chart.jpg"
    fname_combined_raw = base + f"_{ts}_combined_raw.jpg"
    fname_class_raw    = base + f"_{ts}_class_raw.jpg"

    fname_combined, n_full, n_broken, pct_broken, raw_combined = build_combined_image(
        disp_img, disp_masks, disp_grains, dc, fname_combined, disp_slice_by_label,
        raw_save_path=fname_combined_raw)
    fname_class, raw_class = build_class_image(
        disp_img, disp_masks, disp_grains, dc, fname_class, coin_diam_px, ref_height_mm,
        disp_slice_by_label, raw_save_path=fname_class_raw)
    build_chart(grain_data, ref_height_mm, coin_diam_px, fname_chart)

    grain_df, summary_df = build_tables(grain_data, ref_height_mm, coin_diam_px)
    sc = {c: sum(1 for g in grain_data if g.get("sub_class") == c) for c in SUB_CLASS_ORDER}
    coin_msg = (f"Coin ✓ {coin_diam_px:.0f}px → {COIN_REAL_MM / coin_diam_px * 1000:.2f}µm/px"
                if coin_diam_px else "Coin not detected (measurements in px)")
    if coin_diam_px:
        sub_msg = " ".join(f"{bc['name']}:{sc[bc['key']]}" for bc in BROKEN_SUBCLASSES)
    else:
        sub_msg = f"No coin -> sub-class skipped ({FULL_CLASS_NAME}:{sc[FULL_CLASS_KEY]} BR:{sc['BR']})"
    other_msg = " ".join(f"{oc['name']}:{sc[oc['key']]}" for oc in OTHER_CLASSES)
    status = (f"✅ {masks_model.max()} grains | "
              f"CNN Full:{n_full} Broken:{n_broken} ({pct_broken:.1f}% broken) | "
              f"Sub-class {sub_msg} | {other_msg} | {coin_msg}")
    print(f"[coin] {'detected, diam_px=%.1f' % coin_diam_px if coin_diam_px else 'NOT detected -> no mm calibration, no grains excluded'}")
    print(status)
    cap_combined_html, cap_class_html = _quality_mode_captions()
    return (fname_combined, fname_class, fname_chart, grain_df, summary_df,
            cap_combined_html, cap_class_html, raw_combined, raw_class)


# ══════════════════════════════════════════════════════════════════════════════
#  2nd PIPELINE -- "🌾 Full grain types" button (7-class variety classifier)
# ══════════════════════════════════════════════════════════════════════════════

NOT_FULL_EXCLUDED_LABEL = "Not Full (excluded)"


def classify_full_grains_variety(grain_data, masks, cnn_img_np, sr_scale_for_cnn,
                                   full_id, debug_dir=None, save_debug=None):
    for g in grain_data:
        g["variety_class_id"]   = -1
        g["variety_class"]      = NOT_FULL_EXCLUDED_LABEL
        g["variety_confidence"] = 0.0

    if CNN_MODEL_2 is None or not grain_data:
        return grain_data

    full_grains = [g for g in grain_data if g.get("cnn_class_id") == full_id]
    if not full_grains:
        return grain_data

    def _extract(g):
        # TIGHT exact-mask crop — matches the retrained classifier's inputs
        return crop_grain_tight_native(cnn_img_np, masks, g["label"],
                                       g["_slice"], sr_scale_for_cnn)

    crops = list(_EXECUTOR.map(_extract, full_grains))
    valid_idx   = [i for i, c in enumerate(crops) if c is not None]
    valid_crops = [crops[i] for i in valid_idx]

    variety_preds = classify_crops_cnn(valid_crops, model=CNN_MODEL_2, transform=CNN_TRANSFORM_2)

    for vi, (cls_id, conf) in zip(valid_idx, variety_preds):
        g = full_grains[vi]
        g["variety_class_id"]   = cls_id
        g["variety_class"]      = CNN_CLASS_NAMES_2.get(cls_id, "n/a")
        g["variety_confidence"] = round(conf, 4)

    do_save_debug = save_debug if save_debug is not None else SAVE_VARIETY_DEBUG_CROPS
    dbg_dir       = debug_dir if debug_dir is not None else VARIETY_DEBUG_CROPS_DIR
    if do_save_debug and valid_idx:
        os.makedirs(dbg_dir, exist_ok=True)
        for vi, crop_bgr in zip(valid_idx, valid_crops):
            g = full_grains[vi]
            cls_folder = os.path.join(dbg_dir, g["variety_class"])
            os.makedirs(cls_folder, exist_ok=True)
            fname = f"grain{g['label']:04d}_conf{g['variety_confidence']:.2f}.png"
            cv2.imwrite(os.path.join(cls_folder, fname), crop_bgr)
        print(f"[debug] saved {len(valid_idx)} variety CNN input crop(s) -> {dbg_dir}")

    return grain_data


def build_variety_image(img_np, masks, grain_data, coin_bbox, save_path, slice_by_label=None,
                         raw_save_path=None):
    canvas = _base_canvas(img_np)
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    slice_by_label = slice_by_label or {}

    def _color_for(g):
        vcid = g.get("variety_class_id", -1)
        if vcid == -1:
            return DIM_COLOR_RGB
        return CNN_COLOR_RGB_2.get(vcid, DIM_COLOR_RGB)

    for g in grain_data:
        _blend_grain_mask_fast(canvas, masks, g["label"], slice_by_label.get(g["label"]),
                                _color_for(g), MASK_FILL_ALPHA)

    if coin_bbox is not None:
        cx0, cy0, cw, ch = coin_bbox
        cv2.rectangle(canvas, (cx0, cy0), (cx0 + cw, cy0 + ch), C_COIN_BOX, 2, cv2.LINE_AA)
        cv2.putText(canvas, "5 PKR coin", (cx0, max(14, cy0 - 8)), fnt, 0.5, C_COIN_BOX, 2, cv2.LINE_AA)

    total      = len(grain_data)
    n_excluded = sum(1 for g in grain_data if g.get("variety_class_id", -1) == -1)
    n_full     = total - n_excluded

    legend_rows = []
    for cid in sorted(CNN_DISPLAY_NAME_BY_ID_2.keys()):
        name = CNN_DISPLAY_NAME_BY_ID_2[cid]
        n = sum(1 for g in grain_data if g.get("variety_class_id") == cid)
        legend_rows.append((cid, name, CNN_COLOR_RGB_2.get(cid, DIM_COLOR_RGB), n))

    leg_x, leg_y = 14, 14
    leg_w = 340
    row_h = 36
    leg_h = 46 + row_h * (max(1, len(legend_rows)) + 1) + 30
    _solid_panel(canvas, leg_x, leg_y, leg_x + leg_w, leg_y + leg_h)

    cv2.putText(canvas, f"Grain Types (variety classifier -- {FULL_CLASS_NAME} grains only)",
                (leg_x + 16, leg_y + 26), fnt, 0.48, (245, 245, 250), 2, cv2.LINE_AA)
    cv2.line(canvas, (leg_x + 12, leg_y + 38), (leg_x + leg_w - 12, leg_y + 38),
              (90, 90, 100), 1, cv2.LINE_AA)

    ly = leg_y + 64
    sw = 26
    for cid, name, color, n in legend_rows:
        cv2.rectangle(canvas, (leg_x + 16, ly - 18), (leg_x + 16 + sw, ly + 4), color, -1)
        cv2.rectangle(canvas, (leg_x + 16, ly - 18), (leg_x + 16 + sw, ly + 4), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{name}  —  {n}", (leg_x + 16 + sw + 14, ly),
                    fnt, LEGEND_ROW_SCALE, (245, 245, 250), 2, cv2.LINE_AA)
        ly += row_h

    cv2.rectangle(canvas, (leg_x + 16, ly - 18), (leg_x + 16 + sw, ly + 4), DIM_COLOR_RGB, -1)
    cv2.rectangle(canvas, (leg_x + 16, ly - 18), (leg_x + 16 + sw, ly + 4), (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Excluded (not {FULL_CLASS_NAME})  —  {n_excluded}", (leg_x + 16 + sw + 14, ly),
                fnt, LEGEND_ROW_SCALE, (245, 245, 250), 2, cv2.LINE_AA)
    ly += row_h

    cv2.putText(canvas, f"Total {total} grains  |  {n_full} {FULL_CLASS_NAME} scored by variety CNN",
                (leg_x + 16, ly), fnt, LEGEND_SUB_SCALE, (205, 205, 215), 1, cv2.LINE_AA)

    cropped, (ox, oy) = zoom_to_grains(canvas, masks, coin_bbox)
    ch_, cw_ = cropped.shape[:2]
    raw_path = save_raw_overlay(img_np, (ox, oy, ox + cw_, oy + ch_), raw_save_path)
    return save_compressed(cropped, save_path), raw_path


def build_variety_chart(grain_data, save_path):
    full_grains = [g for g in grain_data if g.get("variety_class_id", -1) != -1]
    total = len(full_grains)
    order = sorted(CNN_DISPLAY_NAME_BY_ID_2.keys()) or [0]
    labels = [CNN_DISPLAY_NAME_BY_ID_2.get(c, "?") for c in order]
    counts = [sum(1 for g in full_grains if g.get("variety_class_id") == c) for c in order]
    colors = [tuple(v / 255 for v in CNN_COLOR_RGB_2.get(c, DIM_COLOR_RGB)) for c in order]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0F172A")
    ax.set_facecolor("#1E293B")
    x_pos = np.arange(len(order))
    bars = ax.bar(x_pos, counts, color=colors, edgecolor="#94A3B8", lw=0.8, width=0.55)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontweight="bold", rotation=20, ha="right")
    for bar, n in zip(bars, counts):
        pct = 100 * n / total if total else 0
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{n}\n({pct:.0f}%)", ha="center", va="bottom",
                    color="#F1F5F9", fontsize=10, fontweight="bold")
    n_excluded = len(grain_data) - total
    subtitle = f"{total} {FULL_CLASS_NAME} grain(s) scored"
    if n_excluded:
        subtitle += f"  ·  {n_excluded} non-{FULL_CLASS_NAME} grain(s) excluded"
    ax.set_title(f"Grain Type Counts (7-class variety classifier)\n{subtitle}",
                 color="#F1F5F9", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", color="#CBD5E1", fontsize=10)
    ax.tick_params(colors="#94A3B8", labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")
    ax.set_ylim(0, max(counts, default=1) * 1.35 + 2)
    ax.grid(axis="y", color="#334155", lw=0.5, linestyle=":")

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
    return save_path


def build_variety_tables(grain_data):
    rows = []
    for g in grain_data:
        vcid = g.get("variety_class_id", -1)
        conf = g.get("variety_confidence", 0.0)
        if vcid == -1:
            grain_type = NOT_FULL_EXCLUDED_LABEL
        else:
            v_cls = g.get("variety_class", "n/a")
            grain_type = CNN_DISPLAY_NAME_BY_ID_2.get(vcid, v_cls.replace("_", " ").title())
        rows.append({
            "Ref #": g["label"],
            "Grain Type": grain_type,
            "Confidence": f"{conf * 100:.1f}%" if conf else "—",
        })
    grain_df = pd.DataFrame(rows)

    full_grains = [g for g in grain_data if g.get("variety_class_id", -1) != -1]
    total       = len(full_grains)
    n_excluded  = len(grain_data) - total

    def _pct(n, denom):
        return f"{n} ({100 * n / denom:.1f}%)" if denom else "0"

    L = _URDU_SUMMARY_LABELS
    srows = [
        {L["metric_col"]: "کل دانے (تصویر)", L["value_col"]: str(len(grain_data))},
        {L["metric_col"]: f"{FULL_CLASS_NAME} دانے (قسم کی درجہ بندی)", L["value_col"]: str(total)},
    ]
    for cid in sorted(CNN_DISPLAY_NAME_BY_ID_2.keys()):
        name = CNN_DISPLAY_NAME_BY_ID_2[cid]
        n = sum(1 for g in full_grains if g.get("variety_class_id") == cid)
        srows.append({L["metric_col"]: name, L["value_col"]: _pct(n, total)})
    srows.append({L["metric_col"]: f"خارج شدہ ({FULL_CLASS_NAME} نہیں)", L["value_col"]: _pct(n_excluded, len(grain_data))})
    summary_df = pd.DataFrame(srows)

    return grain_df, summary_df


def run_variety_analysis(single_img):
    if single_img is None:
        raise gr.Error("Upload an image first.")
    if CNN_MODEL is None:
        raise gr.Error("The 5-class quality model failed to load. It's required as "
                        "stage 1 of this pipeline. Check CNN_CHECKPOINT_PATH.")
    if CNN_MODEL_2 is None:
        raise gr.Error("The 7-class grain-type model failed to load. Check "
                        "SECOND_CNN_CHECKPOINT_PATH at the top of this file.")
    if FULL_ID is None:
        raise gr.Error(f"The quality model's checkpoint has no '{FULL_CLASS_NAME}' class.")

    single_img = convert_heic_if_needed(single_img)

    sr_scale = 2
    base     = os.path.splitext(single_img)[0]

    fp        = tif_view(single_img)
    img_input = imread(fp)
    orig_h, orig_w = img_input.shape[:2]

    img_enh   = enhance_resolution(img_input)
    img_model = image_resize(img_enh, resize=1600)
    masks_model, _ = run_cellpose(img_model, 250, 0.4, 0.0)
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}  {masks_model.max()} grains (grain-type pass)")

    masks_orig = cv2.resize(masks_model.astype("uint16"), (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST).astype("uint16")

    coin_diam_model, coin_bbox_model = detect_coin_diameter_px(img_model, debug=True)
    if coin_diam_model is not None:
        sx = orig_w / img_model.shape[1]
        sy = orig_h / img_model.shape[0]
        bx, by, bw, bh = coin_bbox_model
        coin_bbox = (int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy))
    else:
        coin_bbox = None

    if sr_scale > 1:
        img_analysis = enhance_resolution(
            cv2.resize(img_input, (orig_w * sr_scale, orig_h * sr_scale), interpolation=cv2.INTER_LANCZOS4))
        masks_analysis = cv2.resize(masks_model.astype("uint16"), (orig_w * sr_scale, orig_h * sr_scale),
                                    interpolation=cv2.INTER_NEAREST).astype("uint16")
    else:
        img_analysis   = img_enh
        masks_analysis = masks_orig

    raw_grains = measure_grains(masks_analysis, img_analysis, run_cnn=True,
                                 cnn_img_np=img_input, sr_scale_for_cnn=sr_scale)

    if coin_bbox is not None:
        cx0, cy0, cw, ch = coin_bbox
        margin = int(max(cw, ch) * 0.35)
        raw_grains = [
            g for g in raw_grains
            if not (cx0 - margin <= g["centroid_x"] // sr_scale <= cx0 + cw + margin
                    and cy0 - margin <= g["centroid_y"] // sr_scale <= cy0 + ch + margin)
        ]

    for g in raw_grains:
        g["centroid_x"] = int(g["centroid_x"] / sr_scale)
        g["centroid_y"] = int(g["centroid_y"] / sr_scale)
        g["top_y"]      = int(g["top_y"] / sr_scale)

    raw_grains = classify_full_grains_variety(raw_grains, masks_analysis, img_input,
                                               sr_scale, FULL_ID)

    grain_data = raw_grains

    if sr_scale > 1:
        disp_img   = cv2.resize(img_input, (orig_w * sr_scale, orig_h * sr_scale), interpolation=cv2.INTER_LANCZOS4)
        disp_masks = cv2.resize(masks_orig.astype("uint16"), (orig_w * sr_scale, orig_h * sr_scale),
                                interpolation=cv2.INTER_NEAREST).astype("uint16")
        dc = None if coin_bbox is None else (
            coin_bbox[0] * sr_scale, coin_bbox[1] * sr_scale, coin_bbox[2] * sr_scale, coin_bbox[3] * sr_scale)
        disp_grains = []
        for g in grain_data:
            dg = dict(g)
            dg["centroid_x"] *= sr_scale; dg["centroid_y"] *= sr_scale; dg["top_y"] *= sr_scale
            disp_grains.append(dg)
    else:
        disp_img, disp_masks, dc, disp_grains = img_input, masks_orig, coin_bbox, grain_data

    disp_slices = find_objects(disp_masks)
    disp_slice_by_label = {i + 1: sl for i, sl in enumerate(disp_slices) if sl is not None}

    ts            = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_variety = base + f"_{ts}_grain_types.jpg"
    fname_chart   = base + f"_{ts}_grain_types_chart.jpg"
    fname_variety_raw = base + f"_{ts}_grain_types_raw.jpg"

    fname_variety, raw_variety = build_variety_image(disp_img, disp_masks, disp_grains, dc,
                                                       fname_variety, disp_slice_by_label,
                                                       raw_save_path=fname_variety_raw)
    build_variety_chart(grain_data, fname_chart)

    grain_df, summary_df = build_variety_tables(grain_data)

    full_grains = [g for g in grain_data if g.get("variety_class_id", -1) != -1]
    n_excluded  = len(grain_data) - len(full_grains)
    counts_msg = " ".join(
        f"{CNN_DISPLAY_NAME_BY_ID_2[cid]}:{sum(1 for g in full_grains if g.get('variety_class_id') == cid)}"
        for cid in sorted(CNN_DISPLAY_NAME_BY_ID_2.keys())
    )
    status = (f"✅ Grain-type classifier: {len(grain_data)} grains total | "
              f"{len(full_grains)} {FULL_CLASS_NAME} (scored) | {n_excluded} excluded "
              f"(not {FULL_CLASS_NAME}) | {counts_msg}")
    print(status)

    cap_html, _ = _variety_mode_captions()
    return (fname_variety, fname_variety, fname_chart, grain_df, summary_df,
            cap_html, cap_html, raw_variety, raw_variety)


# ══════════════════════════════════════════════════════════════════════════════
#  Table -> image download handlers
# ══════════════════════════════════════════════════════════════════════════════

def download_grain_table_image(grain_df):
    if grain_df is None or len(grain_df) == 0:
        raise gr.Error("Run an analysis first -- there's no measurement table to export yet.")
    return dataframe_to_table_image(grain_df, "Per-grain Measurements")


def download_summary_table_image(summary_df):
    if summary_df is None or len(summary_df) == 0:
        raise gr.Error("Run an analysis first -- there's no summary table to export yet.")
    return dataframe_to_table_image(summary_df, "خلاصہ (Summary)")


# ══════════════════════════════════════════════════════════════════════════════
#  Sample / preview images for the UI (click-to-load, no upload needed)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_THUMB_DIR  = os.path.join(SAMPLE_IMAGES_DIR, ".thumbs")
SAMPLE_THUMB_MAX  = 320
SAMPLE_THUMB_Q    = 70


def _build_thumb_if_needed(src_path, thumb_dir):
    os.makedirs(thumb_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    thumb_path = os.path.join(thumb_dir, base + ".jpg")

    try:
        if os.path.exists(thumb_path) and os.path.getmtime(thumb_path) >= os.path.getmtime(src_path):
            return thumb_path
        img = cv2.imread(src_path)
        if img is None:
            print(f"[samples] could not read '{src_path}' -- skipping")
            return None
        h, w = img.shape[:2]
        scale = SAMPLE_THUMB_MAX / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        cv2.imwrite(thumb_path, img, [cv2.IMWRITE_JPEG_QUALITY, SAMPLE_THUMB_Q])
        return thumb_path
    except Exception as e:
        print(f"[samples] failed to build thumbnail for '{src_path}': {e}")
        return None


def _list_sample_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    try:
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"[samples] created empty folder '{folder}' -- drop .jpg/.png "
                  f"test images in there and restart to enable click-to-test samples.")
            return []
        originals = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))
        )
        if not originals:
            fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Binclassification", "sample_images")
            if os.path.isdir(fallback):
                fb_originals = sorted(
                    os.path.join(fallback, f) for f in os.listdir(fallback)
                    if f.lower().endswith(exts) and os.path.isfile(os.path.join(fallback, f))
                )
                if fb_originals:
                    print(f"[samples] no local samples, using fallback folder '{fallback}'")
                    originals = fb_originals
        pairs = []
        for orig in originals:
            thumb = _build_thumb_if_needed(orig, SAMPLE_THUMB_DIR)
            if thumb is not None:
                pairs.append((thumb, orig))
        print(f"[samples] {len(pairs)} sample image(s) ready (thumbnails cached in "
              f"'{SAMPLE_THUMB_DIR}')")
        return pairs
    except Exception as e:
        print(f"[samples] failed to list/prepare '{folder}': {e}")
        return []


SAMPLE_PAIRS       = _list_sample_images(SAMPLE_IMAGES_DIR)
SAMPLE_THUMB_PATHS = [t for t, _ in SAMPLE_PAIRS]
SAMPLE_THUMB_TO_ORIGINAL = {t: o for t, o in SAMPLE_PAIRS}


# ══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate",
                       font=gr.themes.GoogleFont("Inter"))
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

.gradio-container {
    background: radial-gradient(circle at 12% -10%, #2e1065 0%, #0f172a 45%, #05070d 100%) !important;
}

#title-bar {
    padding: 22px 26px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #5b21b6 100%);
    border-radius: 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(91, 33, 182, 0.35);
    border: 1px solid rgba(167, 139, 250, 0.25);
}
#title-bar .gv-badge {
    display: inline-block; margin-top: 8px; padding: 3px 10px;
    background: rgba(167, 139, 250, 0.16); border: 1px solid rgba(167, 139, 250, 0.4);
    border-radius: 999px; font-size: 0.72rem; color: #DDD6FE; font-weight: 600;
    letter-spacing: .02em;
}

#run-btn {
    margin-top: 14px; width: 100%; height: 54px;
    font-size: 1.08rem; font-weight: 700; border-radius: 12px;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.45);
    transition: transform .15s ease, box-shadow .15s ease;
}
#run-btn:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(124, 58, 237, 0.55); }

#run-btn2 {
    margin-top: 10px; width: 100%; height: 54px;
    font-size: 1.08rem; font-weight: 700; border-radius: 12px;
    background: linear-gradient(135deg, #0d9488, #059669) !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(5, 150, 105, 0.45);
    transition: transform .15s ease, box-shadow .15s ease;
}
#run-btn2:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(5, 150, 105, 0.55); }

footer  { display: none !important; }

.compact-legend {
    font-size: 0.83rem; color: #CBD5E1; line-height: 1.7;
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(100, 116, 139, 0.25);
    border-radius: 10px; padding: 10px 14px; margin: 6px 0 10px;
}
.compact-legend span { padding: 0 1px; }

.gv-section-title {
    font-size: 1.05rem; font-weight: 800; color: #E9D5FF;
    margin: 26px 0 12px; letter-spacing: .01em;
    display: flex; align-items: center; gap: 8px;
}
.gv-subsection-title {
    font-size: 0.9rem; font-weight: 700; color: #CBD5E1; margin: 0 0 8px;
}

.gv-samples-hint {
    margin-top: 12px; font-size: 0.85rem; color: #94A3B8;
}

.gv-note-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(20,184,166,0.06));
    border: 1px solid rgba(167, 139, 250, 0.35);
    border-left: 4px solid #7C3AED;
    border-radius: 10px;
    padding: 12px 16px; margin-top: 12px;
    font-size: 0.82rem; color: #CBD5E1; line-height: 1.75;
}
.gv-note-box b { color: #E9D5FF; }
.gv-note-box .gv-chip {
    display: inline-block; padding: 1px 7px; border-radius: 6px;
    font-weight: 700; font-size: 0.78rem;
}

.gv-table-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; margin-bottom: 6px;
}

.gv-holder-row { display: none !important; }

.gv-viewer {
    position: relative;
    width: 100%;
    height: 500px;
    overflow: hidden;
    border-radius: 12px;
    background: #0b1020;
    border: 1px solid rgba(148, 163, 184, 0.18);
    touch-action: none;
}
.gv-viewer-stage {
    width: 100%; height: 100%;
    display: flex; align-items: center; justify-content: center;
}
.gv-viewer-img {
    max-width: 100%; max-height: 100%;
    object-fit: contain;
    user-select: none;
    -webkit-user-drag: none;
    cursor: grab;
    will-change: transform;
    transform-origin: center center;
}
.gv-viewer-img:active { cursor: grabbing; }
.gv-viewer-placeholder {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: #64748B; font-size: 0.85rem; pointer-events: none;
}
.gv-overlay-btn {
    position: absolute; top: 10px; right: 10px; z-index: 5;
    background: rgba(15, 23, 42, 0.85); color: #F1F5F9;
    border: 1px solid rgba(148, 163, 184, 0.45); border-radius: 8px;
    padding: 6px 14px; font-size: 0.78rem; font-weight: 700;
    cursor: pointer; backdrop-filter: blur(4px);
    transition: background .15s ease, transform .15s ease, border-color .15s ease;
}
.gv-overlay-btn:hover { transform: translateY(-1px); border-color: rgba(167, 139, 250, 0.6); }
.gv-overlay-btn.gv-overlay-active {
    background: linear-gradient(135deg, #0d9488, #059669);
    border-color: rgba(45, 212, 191, 0.6);
}
.gv-viewer-hint {
    position: absolute; bottom: 8px; left: 10px; z-index: 5;
    font-size: 0.72rem; color: rgba(226, 232, 240, 0.65);
    pointer-events: none; letter-spacing: .01em;
}

.gv-viewer-tools {
    position: absolute; top: 10px; left: 10px; z-index: 5;
    display: flex; gap: 6px;
}
.gv-tool-btn {
    width: 30px; height: 30px; padding: 0;
    display: flex; align-items: center; justify-content: center;
    background: rgba(15, 23, 42, 0.85); color: #F1F5F9;
    border: 1px solid rgba(148, 163, 184, 0.45); border-radius: 8px;
    font-size: 0.95rem; line-height: 1; cursor: pointer;
    backdrop-filter: blur(4px);
    transition: background .15s ease, transform .15s ease, border-color .15s ease;
}
.gv-tool-btn:hover { transform: translateY(-1px); border-color: rgba(167, 139, 250, 0.6); }
.gv-tool-btn:active { transform: translateY(0); }

.gv-viewer:fullscreen,
.gv-viewer:-webkit-full-screen {
    width: 100vw !important;
    height: 100vh !important;
    border-radius: 0;
}

@media (max-width: 900px) {
    .gradio-container .gr-row {
        flex-wrap: wrap;
        gap: 16px;
    }
    .gradio-container .gr-row > .gr-column {
        min-width: auto !important;
        width: 100% !important;
    }
    .gradio-container .gr-row > .gr-column:nth-child(2) {
        margin-top: 0 !important;
    }
    .gv-viewer {
        height: 360px;
    }
    .gv-viewer-placeholder {
        font-size: 0.95rem;
    }
    .gv-table-toolbar {
        flex-direction: column;
        align-items: stretch;
        gap: 10px;
    }
}
"""

def _viewer_html(viewer_id):
    return f'''
    <div id="{viewer_id}" class="gv-viewer">
      <div class="gv-viewer-stage">
        <img class="gv-viewer-img" draggable="false" data-out-src="" data-raw-src="" />
      </div>
      <div class="gv-viewer-placeholder">Run an analysis to see results here</div>
      <div class="gv-viewer-tools">
        <button type="button" class="gv-tool-btn gv-expand-btn" title="Enlarge / open fullscreen">⛶</button>
        <button type="button" class="gv-tool-btn gv-download-btn" title="Download this image">⬇</button>
      </div>
      <button type="button" class="gv-overlay-btn" title="Toggle between the predicted mask and the original photo">🔍 Overlay</button>
      <div class="gv-viewer-hint">pinch to zoom · drag to pan</div>
    </div>'''

_OVERLAY_VIEWER_JS = """
() => {
  function initViewer(viewerId, holdOutId, holdRawId) {
    const root = document.getElementById(viewerId);
    if (!root) return false;
    const holdOut = document.getElementById(holdOutId);
    const holdRaw = document.getElementById(holdRawId);
    if (!holdOut || !holdRaw) return false;
    if (root.dataset.gvInit === "1") return true;
    root.dataset.gvInit = "1";

    const img = root.querySelector(".gv-viewer-img");
    const btn = root.querySelector(".gv-overlay-btn");
    const expandBtn = root.querySelector(".gv-expand-btn");
    const downloadBtn = root.querySelector(".gv-download-btn");
    const placeholder = root.querySelector(".gv-viewer-placeholder");

    let scale = 1, tx = 0, ty = 0;
    let dragging = false, startX = 0, startY = 0, moved = false;
    let pinchZoom = false, pinchDist = 0, pinchStartScale = 1;
    let showingRaw = false;

    function clampPan() {
      const baseW = img.offsetWidth;
      const baseH = img.offsetHeight;
      if (!baseW || !baseH) return;
      const maxX = Math.max(0, (baseW * scale - baseW) / 2);
      const maxY = Math.max(0, (baseH * scale - baseH) / 2);
      tx = Math.min(maxX, Math.max(-maxX, tx));
      ty = Math.min(maxY, Math.max(-maxY, ty));
    }

    function applyTransform() {
      img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    }
    function resetTransform() {
      scale = 1; tx = 0; ty = 0; applyTransform();
    }

    function getDistance(touches) {
      const dx = touches[0].clientX - touches[1].clientX;
      const dy = touches[0].clientY - touches[1].clientY;
      return Math.hypot(dx, dy);
    }
    function getCenter(touches) {
      return {
        x: (touches[0].clientX + touches[1].clientX) / 2,
        y: (touches[0].clientY + touches[1].clientY) / 2,
      };
    }

    root.addEventListener("wheel", (e) => {
      if (!img.getAttribute("data-out-src")) return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      scale = Math.min(10, Math.max(1, scale * factor));
      if (scale === 1) { tx = 0; ty = 0; }
      clampPan();
      applyTransform();
    }, { passive: false });

    root.addEventListener("mousedown", (e) => {
      if (!img.getAttribute("data-out-src")) return;
      dragging = true; moved = false;
      startX = e.clientX - tx; startY = e.clientY - ty;
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      tx = e.clientX - startX; ty = e.clientY - startY;
      moved = true;
      clampPan();
      applyTransform();
    });
    window.addEventListener("mouseup", () => { dragging = false; });

    root.addEventListener("touchstart", (e) => {
      if (!img.getAttribute("data-out-src")) return;
      if (e.touches.length === 1) {
        dragging = true;
        pinchZoom = false;
        startX = e.touches[0].clientX - tx;
        startY = e.touches[0].clientY - ty;
      } else if (e.touches.length === 2) {
        dragging = false;
        pinchZoom = true;
        pinchDist = getDistance(e.touches);
        pinchStartScale = scale;
      }
    }, { passive: false });

    root.addEventListener("touchmove", (e) => {
      if (!img.getAttribute("data-out-src")) return;
      if (pinchZoom && e.touches.length === 2) {
        e.preventDefault();
        const newDist = getDistance(e.touches);
        const factor = newDist / pinchDist;
        const nextScale = Math.min(10, Math.max(1, pinchStartScale * factor));
        const rect = root.getBoundingClientRect();
        const center = getCenter(e.touches);
        const dx = center.x - rect.left;
        const dy = center.y - rect.top;
        const ratio = nextScale / scale;
        tx = (tx - dx) * ratio + dx;
        ty = (ty - dy) * ratio + dy;
        scale = nextScale;
        clampPan();
        applyTransform();
      } else if (dragging && e.touches.length === 1) {
        e.preventDefault();
        tx = e.touches[0].clientX - startX;
        ty = e.touches[0].clientY - startY;
        moved = true;
        clampPan();
        applyTransform();
      }
    }, { passive: false });

    root.addEventListener("touchend", (e) => {
      if (e.touches.length < 2) pinchZoom = false;
      if (e.touches.length === 1) {
        dragging = true;
        startX = e.touches[0].clientX - tx;
        startY = e.touches[0].clientY - ty;
      } else {
        dragging = false;
      }
    });

    const decodeCache = new Map();
    function preload(src) {
      if (!src || decodeCache.has(src)) return;
      const pre = new Image();
      pre.decoding = "async";
      pre.src = src;
      decodeCache.set(src, pre);
    }

    btn.addEventListener("click", () => {
      const outSrc = img.getAttribute("data-out-src");
      const rawSrc = img.getAttribute("data-raw-src");
      if (!outSrc) return;
      showingRaw = !showingRaw;
      const nextSrc = (showingRaw && rawSrc) ? rawSrc : outSrc;
      const cached = decodeCache.get(nextSrc);
      if (cached && cached.complete) {
        img.src = nextSrc;
      } else {
        preload(nextSrc);
        img.src = nextSrc;
      }
      if (showingRaw && rawSrc) {
        btn.textContent = "🎭 Show Masks";
        btn.classList.add("gv-overlay-active");
      } else {
        showingRaw = false;
        btn.textContent = "🔍 Overlay";
        btn.classList.remove("gv-overlay-active");
      }
    });

    expandBtn.addEventListener("click", () => {
      if (!img.getAttribute("data-out-src")) return;
      const isFs = document.fullscreenElement || document.webkitFullscreenElement;
      if (!isFs) {
        const req = root.requestFullscreen || root.webkitRequestFullscreen
                    || root.msRequestFullscreen;
        if (req) req.call(root);
      } else {
        const exit = document.exitFullscreen || document.webkitExitFullscreen
                     || document.msExitFullscreen;
        if (exit) exit.call(document);
      }
    });
    ["fullscreenchange", "webkitfullscreenchange"].forEach((evt) => {
      root.addEventListener(evt, () => { resetTransform(); });
    });

    downloadBtn.addEventListener("click", () => {
      const src = img.src;
      if (!src) return;
      const a = document.createElement("a");
      a.href = src;
      a.download = `${viewerId}${showingRaw ? "_raw" : "_output"}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    function currentSrc(holder) {
      const im = holder.querySelector("img");
      return im ? im.src : "";
    }

    function watch(holder, isRaw) {
      let lastSrc = "";
      const sync = () => {
        const src = currentSrc(holder);
        if (!src || src === lastSrc) return;
        lastSrc = src;
        preload(src);
        if (isRaw) {
          img.setAttribute("data-raw-src", src);
          if (showingRaw) img.src = src;
        } else {
          img.setAttribute("data-out-src", src);
          if (placeholder) placeholder.style.display = "none";
          showingRaw = false;
          img.src = src;
          btn.textContent = "🔍 Overlay";
          btn.classList.remove("gv-overlay-active");
          resetTransform();
        }
      };
      sync();
      const obs = new MutationObserver(sync);
      obs.observe(holder, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
    }
    watch(holdOut, false);
    watch(holdRaw, true);
    return true;
  }

  function boot() {
    const a = initViewer("gv-viewer-combined", "hold-combined-out", "hold-combined-raw");
    const b = initViewer("gv-viewer-class", "hold-class-out", "hold-class-raw");
    if (!a || !b) setTimeout(boot, 400);
  }
  setTimeout(boot, 300);
}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Sample-strip click handler (shared by the left-column gallery)
# ══════════════════════════════════════════════════════════════════════════════
import shutil

def _pick_sample_bottom(evt: gr.SelectData):
    """Called when a thumbnail is clicked. Returns the ORIGINAL full-res path so
    it loads into the uploader exactly like a manual upload would."""
    try:
        thumb_path = None
        if isinstance(evt.value, dict):
            thumb_path = evt.value.get("image", {}).get("path") or evt.value.get("path")
        elif isinstance(evt.value, (list, tuple)) and evt.value:
            thumb_path = evt.value[0]
        elif isinstance(evt.value, str):
            thumb_path = evt.value

        original = None
        if thumb_path is not None:
            original = SAMPLE_THUMB_TO_ORIGINAL.get(thumb_path)
            if original is None:
                base = os.path.basename(thumb_path)
                for t, o in SAMPLE_THUMB_TO_ORIGINAL.items():
                    if os.path.basename(t) == base or os.path.splitext(os.path.basename(o))[0] == os.path.splitext(base)[0]:
                        original = o
                        break

        if original is None and 0 <= evt.index < len(SAMPLE_PAIRS):
            original = SAMPLE_PAIRS[evt.index][1]

        if original is None or not os.path.exists(original):
            print(f"[samples] could not resolve clicked sample (index={evt.index}, "
                  f"value={evt.value!r})")
            return None

        # Make sure the file is somewhere Gradio can serve from. If the original
        # lives outside cwd / the sample dir, copy it into the sample dir.
        cwd = os.getcwd()
        abs_orig = os.path.abspath(original)
        allowed = abs_orig.startswith(os.path.abspath(SAMPLE_IMAGES_DIR)) or abs_orig.startswith(cwd)
        if not allowed:
            os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
            dst = os.path.join(SAMPLE_IMAGES_DIR, os.path.basename(abs_orig))
            try:
                if not os.path.exists(dst):
                    shutil.copy2(abs_orig, dst)
                abs_orig = os.path.abspath(dst)
            except Exception as e:
                print(f"[samples] could not copy sample into servable dir: {e}")

        print(f"[samples] loaded sample -> {abs_orig}")
        return abs_orig
    except Exception as e:
        print(f"[samples] click handler error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Blocks
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="GrainVision PRO", theme=THEME, css=CSS) as demo:
    gr.HTML(f"""
    <div id="title-bar">
      <div style="font-size:1.7rem;font-weight:900;color:#F5F3FF;letter-spacing:-.01em;">
        🌾 GrainVision <span style="color:#C4B5FD;">PRO</span>
      </div>
      <div style="color:#DDD6FE;font-size:0.92rem;margin-top:4px;">
        Cellpose-SAM segmentation · CNN quality &amp; variety classification ·
        mm calibration via a 5&nbsp;PKR reference coin
      </div>
      <span class="gv-badge">Full / Broken / Weak / Rejection / Fatty &nbsp;·&nbsp; SG · B1 · B2 · G1</span>
    </div>
    """)

    with gr.Row():
        # ── LEFT COLUMN ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            inp_image = gr.Image(label="Upload rice image", type="filepath", height=280)
            inp_image_preview = gr.Image(label="Preview", interactive=False, height=280)

            # keep the preview in sync with whatever's in the uploader
            inp_image.change(prepare_input_image_for_preview,
                              inputs=[inp_image], outputs=[inp_image_preview])

            run_btn = gr.Button("🔍 Quality test (Full / Broken / …)", elem_id="run-btn")
            run_btn2 = gr.Button("🌾 Full grain types (variety)", elem_id="run-btn2")

        # ── RIGHT COLUMN ─────────────────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🎨 Combined View"):
                    cap_combined = gr.HTML()
                    gr.HTML(_viewer_html("gv-viewer-combined"))
                    with gr.Row(elem_classes=["gv-holder-row"]):
                        out_combined     = gr.Image(elem_id="hold-combined-out", visible=True)
                        hold_combined_raw = gr.Image(elem_id="hold-combined-raw", visible=True)

                with gr.Tab("🔎 Detail View"):
                    cap_class = gr.HTML()
                    gr.HTML(_viewer_html("gv-viewer-class"))
                    with gr.Row(elem_classes=["gv-holder-row"]):
                        out_class     = gr.Image(elem_id="hold-class-out", visible=True)
                        hold_class_raw = gr.Image(elem_id="hold-class-raw", visible=True)

                with gr.Tab("📊 Distribution Chart"):
                    out_chart = gr.Image(label="Distribution", height=520)

    # ── SUMMARY TABLE ────────────────────────────────────────────────────────
    gr.HTML('<div class="gv-section-title">📋 خلاصہ — Summary</div>')
    with gr.Row(elem_classes=["gv-table-toolbar"]):
        gr.HTML('<div class="gv-subsection-title">Class breakdown for this sample</div>')
        dl_summary_img = gr.DownloadButton("🖼️ Download summary as image", size="sm")
    summary_table = gr.Dataframe(interactive=False, wrap=True)
    gr.HTML(f"""
    <div class="gv-note-box">
      <b>How broken grains are sub-classified.</b> A reference height is taken from
      the tallest detected <b>{FULL_CLASS_NAME}</b> grain (or a
      {SUBCLASS_FALLBACK_REF_MM:.1f}&nbsp;mm fallback when a coin is present but no
      full grains are found). Each broken grain is then bucketed by its height as a
      percentage of that reference:
      &nbsp;<span class="gv-chip" style="background:{BROKEN_SUBCLASSES[0]['color_hex']};color:#1e1b4b;">SG 75–90%</span>
      &nbsp;<span class="gv-chip" style="background:{BROKEN_SUBCLASSES[1]['color_hex']};color:#f8fafc;">B1 55–74%</span>
      &nbsp;<span class="gv-chip" style="background:{BROKEN_SUBCLASSES[2]['color_hex']};color:#f8fafc;">B2 26–54%</span>
      &nbsp;<span class="gv-chip" style="background:{BROKEN_SUBCLASSES[3]['color_hex']};color:#f8fafc;">G1 0–25%</span>.
      Without a coin, no mm calibration is possible, so broken grains stay
      <b>unclassified (BR)</b>.
    </div>
    """)

    # ── PER-GRAIN TABLE ──────────────────────────────────────────────────────
    gr.HTML('<div class="gv-section-title">🔬 Per-grain Measurements</div>')
    with gr.Row(elem_classes=["gv-table-toolbar"]):
        gr.HTML('<div class="gv-subsection-title">One row per detected grain</div>')
        dl_grain_img = gr.DownloadButton("🖼️ Download table as image", size="sm")
    grain_table = gr.Dataframe(interactive=False, wrap=True)

    # ── SAMPLE IMAGES (bottom of the page) ───────────────────────────────────
    gr.HTML('<div class="gv-section-title">🖼️ Sample Images</div>')
    if SAMPLE_THUMB_PATHS:
        gr.HTML('<div class="gv-samples-hint">Don\'t have a photo handy? '
                'Choose a sample image from here by <b>clicking</b> any thumbnail '
                'below — it will load into the uploader and preview at the top of '
                'the page, ready for you to run a Quality test or Full grain '
                'types analysis.</div>')
        sample_gallery = gr.Gallery(
            value=SAMPLE_THUMB_PATHS,
            label=None,
            show_label=False,
            columns=6,
            rows=1,
            height="auto",
            object_fit="cover",
            allow_preview=False,
            elem_id="gv-sample-gallery",
        )
        # click a sample -> load original into the uploader (top) -> refresh the
        # preview (the .then() is what makes the Preview update in-place, since a
        # programmatic set of inp_image doesn't always re-fire .change)
        sample_gallery.select(
            _pick_sample_bottom,
            inputs=None,
            outputs=[inp_image],
        ).then(
            prepare_input_image_for_preview,
            inputs=[inp_image],
            outputs=[inp_image_preview],
        )
    else:
        gr.HTML('<div class="gv-samples-hint">💡 No sample images yet. Drop a few '
                f'.jpg/.png images into <code>{SAMPLE_IMAGES_DIR}</code> and '
                'restart the app — they\'ll appear here as click-to-load '
                'thumbnails.</div>')

    # ── WIRING ───────────────────────────────────────────────────────────────
    run_btn.click(
        run_analysis,
        inputs=[inp_image],
        outputs=[out_combined, out_class, out_chart, grain_table, summary_table,
                 cap_combined, cap_class, hold_combined_raw, hold_class_raw],
    )

    run_btn2.click(
        run_variety_analysis,
        inputs=[inp_image],
        outputs=[out_combined, out_class, out_chart, grain_table, summary_table,
                 cap_combined, cap_class, hold_combined_raw, hold_class_raw],
    )

    dl_grain_img.click(download_grain_table_image, inputs=[grain_table], outputs=[dl_grain_img])
    dl_summary_img.click(download_summary_table_image, inputs=[summary_table], outputs=[dl_summary_img])

    demo.load(None, None, None, js=_OVERLAY_VIEWER_JS)


# ══════════════════════════════════════════════════════════════════════════════
#  Launch
# ══════════════════════════════════════════════════════════════════════════════
def _free_port(start=7860, end=7870):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


if __name__ == "__main__":
    PORT = _free_port(7860, 7870)

    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(PORT)
        print(f"[ngrok] public URL: {public_url}")
    except Exception as e:
        print(f"[ngrok] not started ({e}) -- using Gradio's own share link instead.")

    demo.launch(server_name="0.0.0.0", server_port=PORT, share=True, show_error=True)