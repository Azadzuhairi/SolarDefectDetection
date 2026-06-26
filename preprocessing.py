"""
preprocessing.py
================
SolarScan FYP — IR Thermal Image Preprocessor
University of Malaya

WHAT THIS DOES:
  1. Reads IR thermal images from dataset2/train, valid, test
  2. Applies CLAHE (contrast enhancement for thermal gradients)
  3. Converts to INFERNO colormap (grayscale → BGR → INFERNO)
  4. Saves processed images back to the same folders (overwrites originals)

DATASET LAYOUT (Roboflow export format):
  dataset2/
    data.yaml
    train/images/   ← processed here
    valid/images/   ← processed here
    test/images/    ← processed here

HOW TO RUN:
  python preprocessing.py

  Run this ONCE before train.py.
  Do NOT run it twice — it will re-process already-processed images.

REQUIREMENTS:
  pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import os

# ─────────────────────────────────────────────
# SETTINGS — Roboflow export folder layout
# ─────────────────────────────────────────────
IMAGE_DIRS = [
    "dataset2/train/images",
    "dataset2/valid/images",
    "dataset2/test/images",
]
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# CLAHE settings
CLAHE_CLIP_LIMIT = 2.0     # Higher = stronger local contrast boost
CLAHE_TILE_GRID  = (8, 8)  # Local region size for adaptive equalization


def apply_clahe(gray_image):
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.
    Enhances local thermal gradients without blowing out hotspot areas.
    """
    clahe = cv2.createCLAHE(
        clipLimit    = CLAHE_CLIP_LIMIT,
        tileGridSize = CLAHE_TILE_GRID
    )
    return clahe.apply(gray_image)


def apply_inferno_colormap(gray_image):
    """
    Convert grayscale thermal image to INFERNO colormap.
    INFERNO: black → purple → red → orange → yellow → white.
    Gives YOLOv8 meaningful RGB channels since it was pretrained on RGB.
    """
    import matplotlib.cm as cm

    normalized   = gray_image.astype(np.float32) / 255.0
    inferno      = cm.get_cmap("inferno")
    colored_rgba = inferno(normalized)
    colored_rgb  = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
    colored_bgr  = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    return colored_bgr


def preprocess_image(image_path):
    """
    Full pipeline for one image:
      Load → Grayscale → CLAHE → INFERNO → Save (overwrite original)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ⚠️  Could not read: {image_path} — skipping.")
        return False

    gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe_image = apply_clahe(gray)
    inferno_img = apply_inferno_colormap(clahe_image)
    cv2.imwrite(image_path, inferno_img)
    return True


def preprocess_folder(folder_path):
    """Process all supported images inside one folder."""
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder not found: {folder_path} — skipping.")
        print(f"       Make sure your Roboflow export is placed inside dataset2/")
        return 0, 0

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if len(image_files) == 0:
        print(f"  ℹ️  No images in: {folder_path}")
        return 0, 0

    success, fail = 0, 0
    for idx, filename in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, filename)
        print(f"  [{idx:3d}/{len(image_files)}] {filename} ...", end=" ")
        if preprocess_image(image_path):
            print("✅")
            success += 1
        else:
            print("❌")
            fail += 1

    return success, fail


def main():
    print("=" * 60)
    print("  SolarScan — IR Thermal Preprocessor")
    print("  Dataset: dataset2/ (Roboflow export layout)")
    print("  Steps:   CLAHE → INFERNO colormap")
    print("=" * 60)

    # Check if dataset2 exists at all
    if not os.path.exists("dataset2"):
        print("\n  ❌ dataset2/ folder not found!")
        print("     Place your Roboflow export folder and rename it to dataset2/")
        print("     Expected structure:")
        print("       dataset2/")
        print("         data.yaml")
        print("         train/images/")
        print("         valid/images/")
        print("         test/images/")
        return

    total_success, total_fail = 0, 0

    for folder in IMAGE_DIRS:
        print(f"\n📂 {folder}")
        s, f = preprocess_folder(folder)
        total_success += s
        total_fail    += f

    print("\n" + "=" * 60)
    print(f"  ✅ Processed : {total_success} images")
    if total_fail > 0:
        print(f"  ❌ Failed    : {total_fail} images")
    print("=" * 60)

    if total_success > 0:
        print("\n  ⚠️  Images have been overwritten with INFERNO-colormap versions.")
        print("  Do NOT run preprocessing.py again — it will double-process.")
        print("\n  Next step: python train.py")
    else:
        print("\n  ⚠️  No images were processed.")
        print("     Check that your dataset2/ folder contains images.")


if __name__ == "__main__":
    main()
