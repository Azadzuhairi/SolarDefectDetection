"""
predict.py
==========
SolarScan FYP — YOLOv8 Inference Script
University of Malaya

CLASSES (7):
  Hotspot | Junction_box | Mild | Moderate |
  Moderate-Critical | Moderate-High | healthy_panel

DEFAULT SOURCE: dataset2/test/images/

HOW TO RUN:
  python predict.py
  python predict.py --source dataset2/test/images/panel_001.jpg
  python predict.py --source dataset2/test/images/

REQUIREMENTS:
  pip install ultralytics opencv-python
"""

import argparse
import os
import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
MODEL_PATH     = "runs/train/solarscan_v1/weights/best.pt"
OUTPUT_DIR     = "runs/predict"
CONF_THRESHOLD = 0.25

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

# Color per class (BGR for OpenCV)
CLASS_COLORS = {
    "Hotspot"           : (0,   165, 255),   # Orange
    "Junction_box"      : (255, 100, 50),    # Blue
    "Mild"              : (0,   220, 220),   # Yellow
    "Moderate"          : (0,   128, 255),   # Light orange
    "Moderate-Critical" : (0,   0,   160),   # Dark red
    "Moderate-High"     : (0,   0,   220),   # Red
    "healthy_panel"     : (0,   200, 0),     # Green
}

# Recommended action per class
CLASS_ACTION = {
    "Hotspot"           : "Investigate — confirm defect type",
    "Junction_box"      : "Inspect junction box — check wiring & seals",
    "Mild"              : "Monitor — re-inspect next maintenance cycle",
    "Moderate"          : "Schedule inspection soon",
    "Moderate-Critical" : "IMMEDIATE action — likely bypass diode failure",
    "Moderate-High"     : "Urgent inspection required",
    "healthy_panel"     : "No action needed",
}


def run_inference(source):
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ❌ Model not found: {MODEL_PATH}")
        print("     Run train.py first.")
        return

    print("=" * 65)
    print("  SolarScan — Defect Detection (predict.py)")
    print("  Classes: Hotspot | Junction_box | Mild | Moderate |")
    print("           Moderate-Critical | Moderate-High | healthy_panel")
    print("=" * 65)
    print(f"\n  📦 Loading: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect images
    if os.path.isdir(source):
        exts   = (".jpg", ".jpeg", ".png", ".bmp")
        images = [os.path.join(source, f) for f in sorted(os.listdir(source))
                  if f.lower().endswith(exts)]
        print(f"  📂 Folder: {len(images)} images in {source}")
    else:
        images = [source]
        print(f"  🖼️  Single image: {source}")

    if not images:
        print("  ⚠️  No images found.")
        return

    print("-" * 65)

    for idx, image_path in enumerate(images, start=1):
        print(f"\n  [{idx}/{len(images)}] {os.path.basename(image_path)}")

        image = cv2.imread(image_path)
        if image is None:
            print("    ⚠️  Could not read — skipping.")
            continue

        results    = model(image_path, conf=CONF_THRESHOLD, verbose=False)
        detections = results[0].boxes

        if detections is None or len(detections) == 0:
            print("    ℹ️  No detections.")
        else:
            print(f"    🔍 {len(detections)} detection(s):")

            for i, box in enumerate(detections):
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                color      = CLASS_COLORS.get(class_name, (200, 200, 200))
                action     = CLASS_ACTION.get(class_name, "N/A")

                print(f"       [{i+1}] {class_name} | {confidence:.2%} confidence")
                print(f"            Box: ({x1},{y1}) → ({x2},{y2})")
                print(f"            Action: {action}")

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.0%}"
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, image)
        print(f"    💾 Saved: {out_path}")

    print("\n" + "=" * 65)
    print(f"  ✅ Done! Outputs in: {OUTPUT_DIR}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="SolarScan — Defect Detection")
    parser.add_argument(
        "--source",
        type    = str,
        default = "dataset2/test/images",
        help    = "Path to image file or folder (default: dataset2/test/images)"
    )
    args = parser.parse_args()
    run_inference(source=args.source)


if __name__ == "__main__":
    main()
