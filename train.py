"""
train.py
========
SolarScan FYP — YOLOv8 Model Training Script
University of Malaya

DATASET: dataset2/   (129 images from Roboflow export)
CLASSES (7):
  0 - Hotspot           : hotspot present (general)
  1 - Junction_box      : junction box anomaly
  2 - Mild              : single cell, minor hotspot
  3 - Moderate          : single cell, moderate hotspot
  4 - Moderate-Critical : diode failure, whole panel affected
  5 - Moderate-High     : single cell, extremely hot
  6 - healthy_panel     : no defect, panel is clean

FOLDER LAYOUT (Roboflow export format):
  dataset2/
    data.yaml
    train/images/   train/labels/
    valid/images/   valid/labels/
    test/images/    test/labels/

HOW TO RUN:
  python train.py

REQUIREMENTS:
  pip install ultralytics torch
"""

from ultralytics import YOLO
import os
import torch

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
MODEL_WEIGHTS = "yolov8s.pt"
DATA_CONFIG   = "dataset2/data.yaml"   # data.yaml lives INSIDE dataset2/
EPOCHS        = 150                    # boosted for small dataset (129 images)
IMAGE_SIZE    = 640
BATCH_SIZE    = 8
WORKERS       = 2
PROJECT_NAME  = "runs/train"
RUN_NAME      = "solarscan_v1"
FREEZE_LAYERS = 10
PATIENCE      = 30                     # more patience for small dataset
LEARNING_RATE = 0.001

# ─────────────────────────────────────────────
# CLASS NAMES — must match data.yaml order exactly
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Hotspot",           # 0
    "Junction_box",      # 1
    "Mild",              # 2
    "Moderate",          # 3
    "Moderate-Critical", # 4
    "Moderate-High",     # 5
    "healthy_panel",     # 6
]


def check_dataset():
    """Verify the Roboflow export folder structure exists."""
    print("\n📋 Checking dataset2/ structure...")
    folders = {
        "Train images" : "dataset2/train/images",
        "Val images"   : "dataset2/valid/images",
        "Train labels" : "dataset2/train/labels",
        "Val labels"   : "dataset2/valid/labels",
    }
    all_good = True
    for name, path in folders.items():
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"  ✅ {name}: {count} files")
        else:
            print(f"  ❌ {name}: NOT FOUND → {path}")
            all_good = False

    if not all_good:
        print("\n  ⚠️  Expected layout:")
        print("     dataset2/")
        print("       data.yaml")
        print("       train/images/   train/labels/")
        print("       valid/images/   valid/labels/")
        print("       test/images/    test/labels/")
        print("\n  Make sure you placed the Roboflow export INTO a dataset2/ folder.")
        return False

    # Check labels exist
    label_train = "dataset2/train/labels"
    txt_files = [f for f in os.listdir(label_train) if f.endswith(".txt")]
    if not txt_files:
        print("\n  ❌ No .txt annotation files in dataset2/train/labels/")
        print("     Re-export from Roboflow in YOLOv8 format.")
        return False

    print(f"\n  ✅ Found {len(txt_files)} label files in train/labels/")
    return True


def train():
    print("=" * 60)
    print("  SolarScan — YOLOv8 Training Script")
    print("  Dataset : dataset2/ (129 images, Roboflow export)")
    print("  Classes : 7")
    for i, name in enumerate(CLASS_NAMES):
        print(f"            {i} - {name}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  🖥️  Device: {device.upper()}")
    if device == "cpu":
        print("  ⚠️  No GPU detected. Training will be slow.")
        print("     Consider Google Colab (free GPU) for faster results.")

    if not os.path.exists(DATA_CONFIG):
        print(f"\n  ❌ data.yaml not found at: {DATA_CONFIG}")
        print("     Copy data.yaml into your dataset2/ folder.")
        return

    if not check_dataset():
        print("\n  ❌ Fix dataset issues above before training.")
        return

    print(f"\n  📦 Loading base model: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)

    print(f"  🚀 Starting training ({EPOCHS} epochs)...\n")
    results = model.train(
        data        = DATA_CONFIG,
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        workers     = WORKERS,
        project     = PROJECT_NAME,
        name        = RUN_NAME,
        device      = device,
        patience    = PATIENCE,
        lr0         = LEARNING_RATE,
        freeze      = FREEZE_LAYERS,

        # ── Augmentation (critical for only 129 images) ──────────────
        augment     = True,
        mosaic      = 1.0,        # 4-image mosaic (great for small datasets)
        mixup       = 0.15,       # blend two images — extra diversity
        copy_paste  = 0.1,        # copy-paste augmentation
        fliplr      = 0.5,
        flipud      = 0.2,
        degrees     = 15.0,       # slightly more rotation
        translate   = 0.15,
        scale       = 0.6,
        shear       = 3.0,
        erasing     = 0.4,
        perspective = 0.0005,

        # ── Preserve INFERNO colormap (no HSV shifts) ─────────────────
        hsv_h       = 0.0,
        hsv_s       = 0.0,
        hsv_v       = 0.0,

        save        = True,
        save_period = 10,
        plots       = True,
        verbose     = True,
    )

    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print("=" * 60)

    best_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    if os.path.exists(best_path):
        print(f"\n  📁 Best model saved: {best_path}")
    else:
        print(f"\n  ⚠️  Check: {PROJECT_NAME}/{RUN_NAME}/weights/")

    try:
        map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
        print(f"\n  📊 mAP@0.5: {map50}")
        print("     > 0.65 = Good for FYP with 129 images")
        print("     > 0.45 = Acceptable — collect more data")
        print("     < 0.30 = Re-check annotations in Roboflow")
    except Exception:
        print("\n  📊 Check runs/train/ for training plots.")

    print("\n  Next steps:")
    print("    1. python evaluate.py")
    print("    2. streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
