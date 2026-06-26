"""
evaluate.py
===========
SolarScan FYP — Model Evaluation Script
University of Malaya

CLASSES (7):
  Hotspot | Junction_box | Mild | Moderate |
  Moderate-Critical | Moderate-High | healthy_panel

DATASET: dataset2/test/images/

HOW TO RUN:
  python evaluate.py

REQUIREMENTS:
  pip install ultralytics matplotlib seaborn
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
MODEL_PATH  = "runs/train/solarscan_v1/weights/best.pt"
DATA_CONFIG = "dataset2/data.yaml"
OUTPUT_DIR  = "runs/evaluate"
IMAGE_SIZE  = 640
CONF        = 0.25
IOU         = 0.50

# Must match data.yaml names order exactly
CLASS_NAMES = [
    "Hotspot",           # 0
    "Junction_box",      # 1
    "Mild",              # 2
    "Moderate",          # 3
    "Moderate-Critical", # 4
    "Moderate-High",     # 5
    "healthy_panel",     # 6
]


def save_confusion_matrix(metrics):
    try:
        matrix = metrics.confusion_matrix.matrix
        labels = CLASS_NAMES + ["background"]

        plt.figure(figsize=(11, 9))
        sns.heatmap(
            matrix,
            annot     = True,
            fmt       = ".0f",
            cmap      = "YlOrRd",
            xticklabels = labels,
            yticklabels = labels,
            linewidths  = 0.5,
        )
        plt.title("SolarScan — Confusion Matrix (7 classes)", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("Actual Class",    fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  🖼️  Confusion matrix saved: {save_path}")

    except Exception as e:
        print(f"\n  ⚠️  Could not save confusion matrix: {e}")
        print(f"     YOLOv8 auto-saves one in {OUTPUT_DIR}/results/ anyway.")


def run_evaluation():
    print("=" * 60)
    print("  SolarScan — Model Evaluation")
    print("  Dataset : dataset2/test/images/")
    print("  Classes : 7")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"\n  ❌ Model not found: {MODEL_PATH}")
        print("     Run train.py first.")
        return

    test_dir = "dataset2/test/images"
    if not os.path.exists(test_dir):
        print(f"\n  ⚠️  Test folder not found: {test_dir}")
        return

    test_images = [f for f in os.listdir(test_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not test_images:
        print(f"\n  ⚠️  No images in: {test_dir}")
        return

    print(f"\n  Found {len(test_images)} test images.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"  Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("  Running evaluation...\n")
    metrics = model.val(
        data    = DATA_CONFIG,
        imgsz   = IMAGE_SIZE,
        conf    = CONF,
        iou     = IOU,
        split   = "test",
        project = OUTPUT_DIR,
        name    = "results",
        verbose = True,
        plots   = True,
    )

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    try:
        p       = metrics.results_dict.get("metrics/precision(B)", "N/A")
        r       = metrics.results_dict.get("metrics/recall(B)",    "N/A")
        map50   = metrics.results_dict.get("metrics/mAP50(B)",     "N/A")
        map5095 = metrics.results_dict.get("metrics/mAP50-95(B)",  "N/A")

        def fmt(v, k):
            return f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"

        print(fmt(p,       "Precision      "))
        print(fmt(r,       "Recall         "))
        print(fmt(map50,   "mAP@0.5        "))
        print(fmt(map5095, "mAP@0.5:0.95   "))

        print("\n  Score guide for 129-image dataset:")
        print("     mAP@0.5 > 0.65  → Good result for FYP")
        print("     mAP@0.5 > 0.45  → Acceptable, collect more data")
        print("     mAP@0.5 < 0.30  → Re-check annotations in Roboflow")

        # Per-class breakdown if available
        try:
            print("\n  Per-class AP@0.5:")
            for i, name in enumerate(CLASS_NAMES):
                ap = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') else "N/A"
                bar = "█" * int(ap * 20) if isinstance(ap, float) else ""
                print(f"    {name:<20} {ap:.3f}  {bar}" if isinstance(ap, float) else f"    {name:<20} N/A")
        except Exception:
            pass

    except Exception as e:
        print(f"  Could not extract metrics: {e}")

    save_confusion_matrix(metrics)
    print(f"\n  All results saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
