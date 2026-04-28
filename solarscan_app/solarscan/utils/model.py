"""
utils/model.py
Handles YOLOv8 model loading and inference.
Returns structured results compatible with all pages.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import streamlit as st
from datetime import datetime

# ── defect class definitions ──────────────────────────────────────────────────
CLASSES = [
    "hotspot_minor",
    "hotspot_moderate",
    "hotspot_severe",
    "bypass_diode_fault",
    "soiling_ir",
    "crack_minor",
    "crack_severe",
    "delamination",
    "discoloration",
    "broken_cell",
    "soiling_rgb",
    "snail_trail",
    "pid_defect",
]

SEVERITY = {
    "hotspot_minor":      ("LOW",    "#22c55e"),
    "hotspot_moderate":   ("MEDIUM", "#f59e0b"),
    "hotspot_severe":     ("HIGH",   "#ef4444"),
    "bypass_diode_fault": ("HIGH",   "#ef4444"),
    "soiling_ir":         ("LOW",    "#22c55e"),
    "crack_minor":        ("LOW",    "#22c55e"),
    "crack_severe":       ("HIGH",   "#ef4444"),
    "delamination":       ("MEDIUM", "#f59e0b"),
    "discoloration":      ("MEDIUM", "#f59e0b"),
    "broken_cell":        ("HIGH",   "#ef4444"),
    "soiling_rgb":        ("LOW",    "#22c55e"),
    "snail_trail":        ("MEDIUM", "#f59e0b"),
    "pid_defect":         ("HIGH",   "#ef4444"),
}

WEIGHTS_PATH = Path("runs/pv_defect/exp/weights/best.pt")


@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached so it only loads once)."""
    try:
        from ultralytics import YOLO
        if WEIGHTS_PATH.exists():
            model = YOLO(str(WEIGHTS_PATH))
            return model, "trained"
        else:
            model = YOLO("yolov8n.pt")
            return model, "pretrained"
    except ImportError:
        return None, "unavailable"


def run_inference(image: Image.Image, conf: float = 0.25) -> dict:
    """
    Run YOLOv8 inference on a PIL image.
    Returns a structured result dict.
    """
    model, status = load_model()

    # ── model not available → return mock result for UI demo ──
    if model is None or status == "pretrained":
        return _mock_result(image, status)

    img_array = np.array(image)
    results = model.predict(img_array, conf=conf, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_id    = int(box.cls)
        cls_name  = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
        conf_val  = float(box.conf)
        xyxy      = box.xyxy[0].tolist()
        sev, col  = SEVERITY.get(cls_name, ("UNKNOWN", "#888"))
        detections.append({
            "class":      cls_name,
            "confidence": round(conf_val * 100),
            "severity":   sev,
            "color":      col,
            "bbox":       xyxy,
        })

    detections.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["severity"], 3))

    return {
        "status":     "trained" if status == "trained" else "pretrained",
        "detections": detections,
        "image":      image,
        "annotated":  r.plot() if detections else np.array(image),
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "counts": {
            "HIGH":   sum(1 for d in detections if d["severity"] == "HIGH"),
            "MEDIUM": sum(1 for d in detections if d["severity"] == "MEDIUM"),
            "LOW":    sum(1 for d in detections if d["severity"] == "LOW"),
        }
    }


def _mock_result(image: Image.Image, status: str) -> dict:
    """
    Returns a placeholder result when model is not trained yet.
    Used for UI demonstration.
    """
    return {
        "status":     status,
        "detections": [],
        "image":      image,
        "annotated":  np.array(image),
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "counts":     {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "warning":    "⚠️ Model not trained yet. Annotate your dataset first, then train to see real detections.",
    }


def get_severity_badge_html(sev: str) -> str:
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-med", "LOW": "badge-low"}.get(sev, "badge-ok")
    return f'<span class="badge {cls}">{sev}</span>'
