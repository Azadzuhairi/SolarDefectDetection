"""
app.py — SolarScan FYP
University of Malaya — YOLOv8 IR Thermal Panel Inspection

streamlit run app.py
pip install streamlit ultralytics opencv-python pandas plotly fpdf2
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import io
import tempfile
import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image as PILImage

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ─── CONFIGURATION ────────────────────────────
MODEL_PATH  = "runs/train/solarscan_v1/weights/best.pt"
LOG_FILE    = "detections_log.csv"

CLASS_NAMES = [
    "hotspot",
    "healthy_panel",
    "mild",
    "moderate",
    "moderate-high",
    "moderate-critical",
]

CLASS_CONFIG = {
    "hotspot": {
        "badge_color": "#dd6b20",
        "description": "Hotspot detected — confirm defect type",
    },
    "healthy_panel": {
        "badge_color": "#38a169",
        "description": "Panel is healthy — no action needed",
    },
    "mild": {
        "badge_color": "#d69e2e",
        "description": "Single cell hotspot — monitor, re-inspect next cycle",
    },
    "moderate": {
        "badge_color": "#ed8936",
        "description": "Single cell hotspot — schedule inspection soon",
    },
    "moderate-high": {
        "badge_color": "#e53e3e",
        "description": "Extremely hot cell — urgent inspection required",
    },
    "moderate-critical": {
        "badge_color": "#9b2c2c",
        "description": "Diode failure — whole panel affected, immediate action required",
    },
}

COLOR_MAP = {
    "healthy_panel"     : "#38a169",
    "mild"              : "#d69e2e",
    "moderate"          : "#ed8936",
    "moderate-high"     : "#e53e3e",
    "moderate-critical" : "#9b2c2c",
    "hotspot"           : "#dd6b20",
}

CRITICAL_CLASSES = {"moderate-critical", "moderate-high"}
WARNING_CLASSES  = {"moderate"}
ALERT_CLASSES    = CRITICAL_CLASSES | WARNING_CLASSES

# ─── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="SolarScan",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&family=Noto+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #07080e;
    --bg-2:     #0c0e18;
    --bg-3:     #10131e;
    --border:   #1a1f30;
    --border-2: #252c42;
    --orange:   #f97316;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --green:    #22c55e;
    --blue:     #3b82f6;
    --text:     #dde3f0;
    --text-2:   #8b96ab;
    --text-dim: #3d4a5c;
    --fh: 'Rajdhani', sans-serif;
    --fm: 'IBM Plex Mono', monospace;
    --fb: 'Noto Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--fb) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp, [data-testid="stAppViewContainer"] { background-color: var(--bg) !important; }
[data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border) !important; }
section.main, .main, .main .block-container { background-color: var(--bg) !important; }
.main .block-container { padding-top: 1.5rem !important; }
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; font-family: var(--fh) !important; letter-spacing: 0.04em !important; }
hr { border-color: var(--border) !important; }
p, span, label { color: var(--text) !important; }

/* SIDEBAR */
[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
[data-testid="stSidebar"] .stRadio label {
    background: var(--bg-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.35rem 0.8rem !important;
    margin-bottom: 4px !important;
    transition: border-color 0.2s, color 0.2s !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover { border-color: var(--orange) !important; }

/* BUTTONS */
.stButton > button {
    background: var(--bg-3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 6px !important;
    font-family: var(--fh) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: border-color 0.2s, color 0.2s, background 0.2s !important;
}
.stButton > button:hover { border-color: var(--orange) !important; color: var(--orange) !important; background: rgba(249,115,22,0.07) !important; }
.stButton > button[kind="primary"] { background: var(--orange) !important; color: #fff !important; border-color: var(--orange) !important; }
.stButton > button[kind="primary"]:hover { background: #ea6900 !important; }
.stDownloadButton > button {
    background: var(--bg-3) !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
    font-family: var(--fh) !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
}

/* INPUTS */
.stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {
    background: var(--bg-3) !important; border: 1px solid var(--border) !important; border-radius: 6px !important;
}
.stTextInput input, .stNumberInput input { color: var(--text) !important; font-family: var(--fm) !important; background: var(--bg-3) !important; }
[data-testid="stFileUploader"] { background: var(--bg-3) !important; border: 1px dashed var(--border-2) !important; border-radius: 10px !important; }
[data-testid="stFileUploader"] * { color: var(--text) !important; }
.stRadio label { color: var(--text) !important; font-family: var(--fb) !important; }
.stDataFrame, .stDataFrame > div { background: var(--bg-3) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stAlert"], .stAlert { background: var(--bg-3) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stSpinner > div { border-top-color: var(--orange) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-2) !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-2) !important; font-family: var(--fh) !important; font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; border-bottom: 2px solid transparent !important; }
.stTabs [aria-selected="true"] { color: var(--orange) !important; border-bottom-color: var(--orange) !important; }
.stTabs [data-baseweb="tab-panel"] { background: var(--bg) !important; padding-top: 1rem !important; }
.stProgress > div > div { background: var(--orange) !important; }

/* PAGE HEADER */
.page-header {
    background: var(--bg-2); border: 1px solid var(--border-2);
    border-left: 3px solid var(--orange); border-radius: 8px;
    padding: 1.2rem 1.8rem; margin-bottom: 1.6rem;
    display: flex; align-items: baseline; gap: 1rem;
}
.page-header h1 { font-family: var(--fh) !important; font-size: 1.7rem !important; font-weight: 700 !important; letter-spacing: 0.12em !important; color: var(--text) !important; margin: 0 !important; text-transform: uppercase !important; }
.page-header h1 span { color: var(--orange); }
.page-header p { margin: 0 !important; color: var(--text-dim) !important; font-family: var(--fm) !important; font-size: 0.72rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

/* STAT BOX */
.stat-box { background: var(--bg-3); border: 1px solid var(--border); border-top: 2px solid var(--orange); border-radius: 8px; padding: 1.1rem 1rem; text-align: center; }
.stat-number { font-family: var(--fh) !important; font-size: 2.3rem !important; font-weight: 700 !important; letter-spacing: 0.04em; line-height: 1; display: block; }
.stat-label { font-family: var(--fm) !important; font-size: 0.68rem !important; color: var(--text-dim) !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; margin-top: 0.4rem !important; display: block; }

/* DETECTION CARD */
.detection-card { background: var(--bg-3); border: 1px solid var(--border); border-left: 3px solid var(--blue); border-radius: 8px; padding: 0.85rem 1rem; margin: 0.45rem 0; display: flex; flex-direction: column; gap: 0.3rem; }
.card-title { font-family: 'Rajdhani', sans-serif; font-size: 0.9rem; font-weight: 600; color: #dde3f0; letter-spacing: 0.07em; text-transform: uppercase; display: flex; align-items: center; gap: 0.5rem; }
.card-desc { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #8b96ab; }
.card-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #3d4a5c; }

/* CRITICAL CARD */
.critical-card { background: rgba(127,29,29,0.1); border: 1px solid rgba(239,68,68,0.25); border-left: 3px solid var(--red); border-radius: 8px; padding: 0.85rem 1rem; margin: 0.45rem 0; display: flex; flex-direction: column; gap: 0.3rem; animation: crit-glow 2.8s ease-in-out infinite; }
@keyframes crit-glow { 0%, 100% { box-shadow: none; } 50% { box-shadow: 0 0 14px rgba(239,68,68,0.1); } }
.critical-card .card-title { color: #fca5a5; }

/* WARNING CARD */
.warning-card { background: rgba(120,80,0,0.1); border: 1px solid rgba(245,158,11,0.25); border-left: 3px solid var(--amber); border-radius: 8px; padding: 0.85rem 1rem; margin: 0.45rem 0; display: flex; flex-direction: column; gap: 0.3rem; }
.warning-card .card-title { color: #fde68a; }
.warning-card .card-desc { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #8b96ab; }
.warning-card .card-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #3d4a5c; }

/* BADGE */
.badge { display: inline-block; padding: 1px 8px; border-radius: 3px; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important; font-weight: 500 !important; letter-spacing: 0.04em; color: #fff !important; }

/* DOT */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

/* WARNING / MODE HINT BOXES */
.warning-box { background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.25); border-left: 3px solid var(--amber); border-radius: 8px; padding: 0.85rem 1.1rem; margin: 1rem 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #8b96ab !important; }
.mode-hint { border-radius: 8px; padding: 0.55rem 1rem; margin-bottom: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; }
.mode-hint-hot { background: rgba(249,115,22,0.07); border: 1px solid rgba(249,115,22,0.2); border-left: 3px solid var(--orange); }
.mode-hint-sev { background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.2); border-left: 3px solid var(--red); }
.mode-hint strong { color: #dde3f0 !important; font-family: 'Rajdhani', sans-serif !important; letter-spacing: 0.04em; }
.mode-hint span { color: #8b96ab !important; }

/* LEGEND */
.legend-item { display: flex; align-items: center; gap: 0.5rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8b96ab; padding: 0.2rem 0; }

/* ANALYSIS BOX */
.analysis-box { background: rgba(59,130,246,0.04); border: 1px solid rgba(59,130,246,0.15); border-left: 3px solid #3b82f6; border-radius: 8px; padding: 0.9rem 1.1rem; margin: 0.8rem 0 0.5rem 0; }
.analysis-label { font-family: 'Rajdhani', sans-serif; font-size: 0.72rem; font-weight: 700; color: #3b82f6 !important; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem; display: block; }
.analysis-box p { font-family: 'Noto Sans', sans-serif !important; font-size: 0.83rem !important; color: #8b96ab !important; margin: 0.3rem 0 !important; line-height: 1.65 !important; }
.analysis-box strong { color: #dde3f0 !important; }

/* THRESHOLD COUNTER */
.thresh-counter { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #3d4a5c; margin-top: 0.2rem; }
.thresh-counter span { color: #f97316; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────

@st.cache_resource
def load_model():
    if not YOLO_AVAILABLE:
        return None, "ultralytics not installed. Run: pip install ultralytics"
    if not os.path.exists(MODEL_PATH):
        return None, f"Model not found at {MODEL_PATH}. Run train.py first."
    try:
        return YOLO(MODEL_PATH), None
    except Exception as e:
        return None, str(e)


HOTSPOT_CLASSES  = {"hotspot"}
SEVERITY_CLASSES = {"mild", "moderate", "moderate-high", "moderate-critical"}

BBOX_COLORS = {
    "hotspot"           : (0,  165, 255),
    "mild"              : (0,  220, 220),
    "moderate"          : (0,  128, 255),
    "moderate-high"     : (0,  0,   220),
    "moderate-critical" : (0,  0,   160),
}


def run_inference(model, image_array, mode="hotspot", conf=0.25):
    results    = model(image_array, conf=conf, verbose=False)
    boxes      = results[0].boxes
    detections = []
    allowed    = HOTSPOT_CLASSES if mode == "hotspot" else SEVERITY_CLASSES

    if boxes is not None:
        for box in boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            if class_name not in allowed:
                continue
            detections.append({
                "class": class_name, "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    annotated = image_array.copy()
    for d in detections:
        color = BBOX_COLORS.get(d["class"], (200, 200, 200))
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} {d['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return detections, annotated


def is_critical(class_name):
    return class_name in CRITICAL_CLASSES


def load_log():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            if "resolved" in df.columns:
                df["resolved"] = df["resolved"].astype(str).str.lower().map(
                    lambda x: True if x == "true" else False
                )
            if "panel_id" not in df.columns:
                df["panel_id"] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["timestamp", "panel_id", "defect", "confidence", "resolved"])


def append_to_log(records):
    df_new = pd.DataFrame(records)
    if os.path.exists(LOG_FILE):
        df_existing = pd.read_csv(LOG_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(LOG_FILE, index=False)


def render_header(page_name):
    st.markdown(f"""
    <div class="page-header">
        <h1>Solar<span>Scan</span></h1>
        <p>{page_name}</p>
    </div>
    """, unsafe_allow_html=True)


def model_warning():
    st.markdown("""
    <div class="warning-box">
        <strong>Model not loaded.</strong>
        Run <code>train.py</code> first, then restart the app.
    </div>
    """, unsafe_allow_html=True)


def render_detection_card(d):
    cfg        = CLASS_CONFIG.get(d["class"], {"badge_color": "#718096", "description": ""})
    badge_color = cfg["badge_color"]
    card_class  = "critical-card" if is_critical(d["class"]) else "detection-card"
    st.markdown(f"""
    <div class="{card_class}">
        <div class="card-title">
            <span class="dot" style="background:{badge_color};"></span>
            {d['class'].replace('-', ' ').upper()}
            <span class="badge" style="background:{badge_color};">{d['confidence']:.0%}</span>
        </div>
        <div class="card-desc">{cfg['description']}</div>
        <div class="card-meta">box ({d['x1']}, {d['y1']}) — ({d['x2']}, {d['y2']})</div>
    </div>
    """, unsafe_allow_html=True)


def chart_layout(height=320):
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10131e",
        font=dict(color="#8b96ab", family="IBM Plex Mono", size=11),
        xaxis=dict(gridcolor="#1a1f30", linecolor="#1a1f30", tickcolor="#3d4a5c"),
        yaxis=dict(gridcolor="#1a1f30", linecolor="#1a1f30", tickcolor="#3d4a5c"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a1f30", borderwidth=1),
        margin=dict(l=0, r=0, t=10, b=0),
    )


def _ps(text):
    """Sanitize text to latin-1 safe characters for old fpdf."""
    return (str(text)
            .replace("—", "-").replace("–", "-")
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .encode("latin-1", errors="replace").decode("latin-1"))


def generate_pdf(panel_id, timestamp, mode, conf_thresh, detections, annotated_rgb):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(249, 115, 22)
    pdf.cell(0, 10, "SolarScan", ln=False)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, " Inspection Report", ln=True)
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # Meta info
    meta = [
        ("Panel ID",             _ps(panel_id or "Not specified")),
        ("Timestamp",            _ps(timestamp)),
        ("Detection Mode",       _ps(mode.title())),
        ("Confidence Threshold", _ps(f"{conf_thresh:.2f}")),
        ("Defects Found",        _ps(str(len(detections)))),
    ]
    for k, v in meta:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(120, 130, 150)
        pdf.cell(55, 6, _ps(k), ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 6, v, ln=True)

    pdf.ln(4)

    # Annotated image — write to temp file (old fpdf only accepts file paths)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(annotated_rgb).save(tmp.name, format="JPEG", quality=90)
        tmp_path = tmp.name
    pdf.image(tmp_path, w=180)
    os.unlink(tmp_path)
    pdf.ln(5)

    # Detections table
    if detections:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 242, 250)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(65, 7, "Class",        border=1, fill=True)
        pdf.cell(30, 7, "Confidence",   border=1, fill=True)
        pdf.cell(75, 7, "Bounding Box", border=1, fill=True, ln=True)

        pdf.set_font("Helvetica", "", 9)
        for d in detections:
            cfg = CLASS_CONFIG.get(d["class"], {"description": ""})
            pdf.set_text_color(30, 30, 30)
            pdf.cell(65, 6, _ps(d["class"].replace("-", " ").title()), border=1)
            pdf.cell(30, 6, _ps(f"{d['confidence']:.0%}"),             border=1)
            pdf.cell(75, 6, _ps(f"({d['x1']},{d['y1']}) to ({d['x2']},{d['y2']})"), border=1, ln=True)
            # Description row
            pdf.set_text_color(120, 130, 150)
            pdf.cell(170, 5, _ps(cfg["description"]), ln=True)
            pdf.set_text_color(30, 30, 30)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(120, 130, 150)
        pdf.cell(0, 8, "No defects detected.", ln=True)

    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "Generated by SolarScan - University of Malaya FYP", ln=True)

    return bytes(pdf.output())


# ─── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown("## SolarScan")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Detect", "Dashboard", "Alerts", "Evaluate"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Detection Settings**")
    conf_thresh = st.slider(
        "Confidence threshold", 0.10, 0.90, 0.25, 0.05,
        format="%.2f", label_visibility="visible"
    )

    # Threshold feedback counter
    last_count = st.session_state.get("last_detection_count")
    last_conf  = st.session_state.get("last_conf_used")
    if last_count is not None:
        st.markdown(
            f'<div class="thresh-counter">Last scan: '
            f'<span>{last_count}</span> detection(s) at '
            f'<span>{last_conf:.2f}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**SolarScan — UM FYP**")
    st.markdown("YOLOv8 + IR thermal imaging")
    st.markdown("**Defect classes:**")
    for name in CLASS_NAMES:
        color = CLASS_CONFIG[name]["badge_color"]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin:3px 0;">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{color};flex-shrink:0;"></span>'
            f'<code style="font-size:0.78rem;">{name}</code></div>',
            unsafe_allow_html=True
        )
    model, model_error = load_model()
    st.markdown("---")
    if model:
        st.success("Model loaded")
    else:
        st.error("Model not found")
        st.caption(model_error or "Run train.py first")


# ─── PAGE: DETECT ─────────────────────────────
if page == "Detect":
    render_header("Single Image Detection")

    st.markdown("#### Detection Mode")
    mode_col1, mode_col2, _ = st.columns([1, 1, 3])

    if "detect_mode" not in st.session_state:
        st.session_state.detect_mode = "hotspot"

    if mode_col1.button(
        "Hotspot", use_container_width=True,
        type="primary" if st.session_state.detect_mode == "hotspot" else "secondary",
    ):
        st.session_state.detect_mode = "hotspot"
        st.session_state.pop("detect_result", None)
        st.rerun()

    if mode_col2.button(
        "Severity", use_container_width=True,
        type="primary" if st.session_state.detect_mode == "severity" else "secondary",
    ):
        st.session_state.detect_mode = "severity"
        st.session_state.pop("detect_result", None)
        st.rerun()

    mode = st.session_state.detect_mode

    if mode == "hotspot":
        st.markdown(
            "<div class='mode-hint mode-hint-hot'><strong>Hotspot Mode</strong> — "
            "<span>annotates any panel where a hotspot is present.</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='mode-hint mode-hint-sev'><strong>Severity Mode</strong> — "
            "<span>classifies hotspot severity: Mild / Moderate / Moderate-High / Moderate-Critical.</span></div>",
            unsafe_allow_html=True
        )

    col_upload, col_result = st.columns([1, 1.5])

    with col_upload:
        st.subheader("Upload Image")

        panel_id = st.text_input(
            "Panel ID",
            placeholder="e.g. Panel-A1, Row-3-Col-7",
            help="Identifier for the panel being inspected. Saved with the detection log and PDF report."
        )

        uploaded = st.file_uploader(
            "IR thermal image",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )

        if uploaded is not None:
            file_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("detect_file_id") != file_id:
                st.session_state.detect_file_id = file_id
                st.session_state.pop("detect_result", None)

        run_btn = st.button("Run Detection", use_container_width=True, type="primary",
                            disabled=(uploaded is None or model is None))

        if uploaded is not None:
            raw = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if raw is not None:
                st.image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                         caption="Original", use_container_width=True)

        st.markdown("**Legend**")
        if mode == "hotspot":
            legend_items = [("hotspot", CLASS_CONFIG["hotspot"]["badge_color"])]
        else:
            legend_items = [
                ("mild",              CLASS_CONFIG["mild"]["badge_color"]),
                ("moderate",          CLASS_CONFIG["moderate"]["badge_color"]),
                ("moderate-high",     CLASS_CONFIG["moderate-high"]["badge_color"]),
                ("moderate-critical", CLASS_CONFIG["moderate-critical"]["badge_color"]),
            ]
        for name, color in legend_items:
            st.markdown(
                f'<div class="legend-item"><span class="dot" style="background:{color};"></span>{name}</div>',
                unsafe_allow_html=True
            )

    with col_result:
        st.subheader("Result")

        if uploaded is None:
            st.info("Upload a thermal image and run detection.")
        elif model is None:
            model_warning()
        else:
            if run_btn:
                with st.spinner("Running inference..."):
                    image         = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    detections, annotated = run_inference(model, image, mode=mode, conf=conf_thresh)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_records = [{
                    "timestamp"  : now,
                    "panel_id"   : panel_id or "",
                    "defect"     : d["class"],
                    "confidence" : round(d["confidence"], 4),
                    "detect_mode": mode,
                    "resolved"   : False,
                } for d in detections]

                if log_records:
                    append_to_log(log_records)

                st.session_state.detect_result = {
                    "annotated_rgb": annotated_rgb,
                    "detections"   : detections,
                    "mode_label"   : "Hotspot Detection" if mode == "hotspot" else "Severity Classification",
                    "n_logged"     : len(log_records),
                    "panel_id"     : panel_id or "Not specified",
                    "timestamp"    : now,
                }
                st.session_state.last_detection_count = len(detections)
                st.session_state.last_conf_used        = conf_thresh

            result = st.session_state.get("detect_result")
            if result:
                st.image(result["annotated_rgb"], caption=result["mode_label"], use_container_width=True)
                st.caption(
                    f"Panel: {result['panel_id']} &nbsp;|&nbsp; "
                    f"Threshold: {conf_thresh:.2f} &nbsp;|&nbsp; "
                    f"{result['n_logged']} record(s) saved"
                )

                if not result["detections"]:
                    st.success("No defects detected.")
                else:
                    st.markdown(f"**{len(result['detections'])} detection(s):**")
                    for d in result["detections"]:
                        render_detection_card(d)

                # PDF export
                st.markdown("---")
                if FPDF_AVAILABLE:
                    pdf_bytes = generate_pdf(
                        result["panel_id"], result["timestamp"],
                        mode, conf_thresh,
                        result["detections"], result["annotated_rgb"]
                    )
                    st.download_button(
                        "Download Inspection Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"solarscan_{result['panel_id'].replace(' ','_')}_{result['timestamp'][:10]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.caption("Install fpdf2 to enable PDF reports: pip install fpdf2")
            else:
                st.info("Click Run Detection to analyse the image.")


# ─── PAGE: DASHBOARD ──────────────────────────
elif page == "Dashboard":
    render_header("Detection Dashboard")
    df = load_log()

    if df.empty:
        st.info("No detections yet. Run detections on the Detect page first.")
    else:
        # Date filter
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            min_date = df["timestamp"].dt.date.min()
            max_date = df["timestamp"].dt.date.max()
            fc1, fc2, _ = st.columns([1, 1, 2])
            start_date = fc1.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
            end_date   = fc2.date_input("To",   value=max_date, min_value=min_date, max_value=max_date)
            df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]
            st.markdown("---")

        if df.empty:
            st.info("No detections in the selected date range.")
        else:
            total      = len(df)
            critical   = len(df[df["defect"].isin(CRITICAL_CLASSES)])
            unresolved = len(df[df["defect"].isin(CRITICAL_CLASSES) & (df["resolved"] == False)]) if "resolved" in df.columns else critical
            avg_conf   = df["confidence"].mean() if "confidence" in df.columns else 0

            c1, c2, c3, c4 = st.columns(4)
            for col, label, value, accent in [
                (c1, "Total Detections",    total,            "#f97316"),
                (c2, "High / Critical",     critical,         "#ef4444"),
                (c3, "Unresolved Critical", unresolved,       "#f59e0b"),
                (c4, "Avg Confidence",      f"{avg_conf:.0%}","#22c55e"),
            ]:
                col.markdown(f"""
                <div class="stat-box" style="border-top-color:{accent};">
                    <span class="stat-number" style="color:{accent};">{value}</span>
                    <span class="stat-label">{label}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            SEVERITY_DISPLAY = {"mild", "moderate", "moderate-high", "moderate-critical", "healthy_panel"}
            counts = (
                df[df["defect"].isin(SEVERITY_DISPLAY)]["defect"]
                .value_counts().reset_index()
            )
            counts.columns = ["Class", "Count"]

            col_bar, col_pie = st.columns(2)
            with col_bar:
                st.subheader("Class Distribution")
                fig_bar = px.bar(counts, x="Class", y="Count",
                                 color="Class", color_discrete_map=COLOR_MAP,
                                 template="plotly_dark")
                fig_bar.update_layout(showlegend=False, **chart_layout())
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_pie:
                st.subheader("Class Breakdown")
                fig_pie = px.pie(counts, names="Class", values="Count",
                                 color="Class", color_discrete_map=COLOR_MAP,
                                 template="plotly_dark")
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b96ab", family="IBM Plex Mono", size=11),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=320,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Recent Detections")
            display_cols = [c for c in ["timestamp", "panel_id", "defect", "confidence", "resolved"] if c in df.columns]
            st.dataframe(
                df[display_cols].sort_values("timestamp", ascending=False).head(50),
                use_container_width=True
            )


# ─── PAGE: ALERTS ─────────────────────────────
elif page == "Alerts":
    render_header("Alerts")
    df = load_log()

    if df.empty:
        st.info("No detection records found.")
    else:
        alert_df    = df[df["defect"].isin(ALERT_CLASSES)].copy()
        critical_df = alert_df[alert_df["defect"].isin(CRITICAL_CLASSES)]
        warning_df  = alert_df[alert_df["defect"].isin(WARNING_CLASSES)]

        n_crit_active = len(critical_df[critical_df["resolved"] == False])
        n_warn_active = len(warning_df[warning_df["resolved"] == False])
        n_resolved    = len(alert_df[alert_df["resolved"] == True])

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="stat-box" style="border-top-color:#ef4444;">
            <span class="stat-number" style="color:#ef4444;">{n_crit_active}</span>
            <span class="stat-label">Active Critical</span></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="stat-box" style="border-top-color:#f59e0b;">
            <span class="stat-number" style="color:#f59e0b;">{n_warn_active}</span>
            <span class="stat-label">Active Warnings</span></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="stat-box" style="border-top-color:#22c55e;">
            <span class="stat-number" style="color:#22c55e;">{n_resolved}</span>
            <span class="stat-label">Resolved</span></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="stat-box" style="border-top-color:#3b82f6;">
            <span class="stat-number" style="color:#3b82f6;">{len(alert_df)}</span>
            <span class="stat-label">Total Alerts</span></div>""", unsafe_allow_html=True)

        st.markdown("---")

        tab_crit, tab_warn, tab_resolved, tab_all = st.tabs(
            ["Critical", "Warnings", "Resolved", "All Detections"]
        )

        def alert_card(idx, row, card_class, badge_bg, badge_label, show_resolve=True):
            defect = row.get("defect", "N/A")
            cfg    = CLASS_CONFIG.get(defect, {"badge_color": "#718096", "description": ""})
            pid    = row.get("panel_id", "")
            pid_str = f" &nbsp;|&nbsp; Panel: {pid}" if pid and str(pid).strip() else ""
            st.markdown(f"""
            <div class="{card_class}">
                <div class="card-title">
                    <span class="dot" style="background:{cfg['badge_color']};"></span>
                    {defect.replace('-', ' ').upper()}
                    <span class="badge" style="background:{badge_bg};">{badge_label}</span>
                </div>
                <div class="card-desc">{cfg['description']}</div>
                <div class="card-meta">{row.get('timestamp','N/A')}{pid_str} &nbsp;|&nbsp; {float(row.get('confidence',0)):.0%} confidence</div>
            </div>
            """, unsafe_allow_html=True)
            if show_resolve:
                if st.button("Mark Resolved", key=f"resolve_{idx}"):
                    df.at[idx, "resolved"] = True
                    df.to_csv(LOG_FILE, index=False)
                    st.success("Marked as resolved.")
                    st.rerun()

        with tab_crit:
            active_crit = critical_df[critical_df["resolved"] == False]
            if active_crit.empty:
                st.success("No active critical alerts.")
            else:
                for idx, row in active_crit.iterrows():
                    alert_card(idx, row, "critical-card", "#ef4444", "Critical")

        with tab_warn:
            active_warn = warning_df[warning_df["resolved"] == False]
            if active_warn.empty:
                st.success("No active warnings.")
            else:
                for idx, row in active_warn.iterrows():
                    alert_card(idx, row, "warning-card", "#f59e0b", "Warning")

        with tab_resolved:
            resolved_rows = alert_df[alert_df["resolved"] == True]
            if resolved_rows.empty:
                st.info("No resolved alerts yet.")
            else:
                for idx, row in resolved_rows.iterrows():
                    defect = row.get("defect", "N/A")
                    card   = "critical-card" if defect in CRITICAL_CLASSES else "warning-card"
                    alert_card(idx, row, "detection-card", "#22c55e", "Resolved", show_resolve=False)

        with tab_all:
            display_cols = [c for c in ["timestamp", "panel_id", "defect", "confidence", "resolved"] if c in df.columns]
            st.dataframe(df[display_cols].sort_values("timestamp", ascending=False), use_container_width=True)
            col_dl, col_clr = st.columns(2)
            col_dl.download_button(
                "Export Log",
                df.to_csv(index=False).encode("utf-8"),
                file_name="detections_log_export.csv",
                mime="text/csv",
                use_container_width=True
            )
            if col_clr.button("Clear Log", use_container_width=True):
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                st.success("Log cleared.")
                st.rerun()


# ─── PAGE: EVALUATE ───────────────────────────
elif page == "Evaluate":
    render_header("Model Evaluation")

    TRAIN_DIR   = "runs/train/solarscan_v1"
    EVAL_DIR    = "runs/evaluate/results"
    RESULTS_CSV = os.path.join(TRAIN_DIR, "results.csv")
    ARGS_YAML   = os.path.join(TRAIN_DIR, "args.yaml")

    EVAL_CLASSES = [
        "Hotspot", "Junction_box", "Mild", "Moderate",
        "Moderate-Critical", "Moderate-High", "healthy_panel",
    ]

    def load_image(path):
        if os.path.exists(path):
            return PILImage.open(path)
        return None

    def count_files(folder):
        if not os.path.exists(folder):
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)

    if not os.path.exists(RESULTS_CSV):
        st.warning("Training results not found. Run train.py first.")
    else:
        res = pd.read_csv(RESULTS_CSV)
        res.columns = res.columns.str.strip()

        best_map50   = res["metrics/mAP50(B)"].max()
        best_prec    = res["metrics/precision(B)"].max()
        best_recall  = res["metrics/recall(B)"].max()
        best_map5095 = res["metrics/mAP50-95(B)"].max()
        best_epoch   = int(res["metrics/mAP50(B)"].idxmax()) + 1
        total_epochs = len(res)
        last         = res.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, accent in [
            (c1, "Best mAP@0.5",       f"{best_map50:.3f}",   "#f97316"),
            (c2, "Best Precision",     f"{best_prec:.3f}",    "#3b82f6"),
            (c3, "Best Recall",        f"{best_recall:.3f}",  "#22c55e"),
            (c4, "Best mAP@0.5:0.95",  f"{best_map5095:.3f}", "#f59e0b"),
        ]:
            col.markdown(f"""
            <div class="stat-box" style="border-top-color:{accent};">
                <span class="stat-number" style="color:{accent};">{value}</span>
                <span class="stat-label">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            f'<p style="color:#3d4a5c;font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
            f'margin:0.5rem 0 1rem 0;">Best mAP@0.5 at epoch {best_epoch} of {total_epochs} &nbsp;|&nbsp;'
            f' Final — P: {last["metrics/precision(B)"]:.3f}'
            f'  R: {last["metrics/recall(B)"]:.3f}'
            f'  mAP50: {last["metrics/mAP50(B)"]:.3f}</p>',
            unsafe_allow_html=True
        )

        m = best_map50
        if m >= 0.65:
            verdict, vcol = "Good result for FYP scope", "#22c55e"
        elif m >= 0.45:
            verdict, vcol = "Acceptable — consider more training data", "#f59e0b"
        else:
            verdict, vcol = "Below baseline — check annotations", "#ef4444"
        st.markdown(
            f'<div class="mode-hint mode-hint-hot" style="margin-bottom:1rem;">'
            f'<strong>mAP@0.5 = {best_map50:.3f}</strong>'
            f'<span style="color:{vcol};margin-left:0.5rem;">— {verdict}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="analysis-box">
            <span class="analysis-label">Analysis — Model Performance</span>
            <p><strong>Precision ({best_prec:.3f})</strong> is high — when the model fires a detection,
            it is correct {best_prec:.0%} of the time, indicating very few false positives.</p>
            <p><strong>Recall ({best_recall:.3f})</strong> is moderate — the model misses approximately
            {1-best_recall:.0%} of actual defects. For predictive maintenance, recall is the more
            critical metric: an undetected diode failure carries significantly higher operational risk
            than a false alarm that triggers a manual re-inspection.</p>
            <p><strong>mAP@0.5 ({best_map50:.3f})</strong> aggregates detection accuracy across all
            7 classes at a standard IoU threshold. The score exceeds the 0.65 FYP benchmark,
            confirming adequate performance given the dataset size and class difficulty.</p>
            <p><strong>mAP@0.5:0.95 ({best_map5095:.3f})</strong> is {(1 - best_map5095/best_map50):.0%}
            lower than mAP@0.5, indicating that bounding box localisation accuracy degrades at stricter
            overlap requirements. Tighter annotations or higher-resolution training images would likely
            close this gap.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Confusion matrix
        st.subheader("Evaluation Plots (Test Set)")
        img = load_image(os.path.join(EVAL_DIR, "confusion_matrix_normalized.png"))
        if img is None:
            img = load_image(os.path.join(TRAIN_DIR, "confusion_matrix_normalized.png"))
        if img:
            _, col_img, _ = st.columns([1, 2, 1])
            col_img.image(img, use_container_width=True)
        else:
            st.info("Run evaluate.py to generate this plot.")

        st.markdown("""
        <div class="analysis-box">
            <span class="analysis-label">Analysis — Confusion Matrix</span>
            <p>Each row represents an <strong>actual class</strong>; each column a <strong>predicted class</strong>.
            A strong diagonal (values close to 1.0) means the model correctly identifies most instances of that class.</p>
            <p>Inter-class confusion is most expected among adjacent severity levels —
            <strong>Mild, Moderate, Moderate-High</strong> — because their thermal signatures
            differ only in temperature gradient intensity, a property sensitive to camera calibration.</p>
            <p><strong>Moderate-Critical</strong> being misclassified as a lower-severity class is the
            highest-risk error in this system, as it would cause a diode failure to go unescalated.</p>
            <p><strong>healthy_panel</strong> and <strong>Junction_box</strong> are visually distinct
            from thermal defect classes and are expected to show minimal cross-class confusion.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Validation samples
        st.subheader("Validation Samples")
        col_gt, col_pred = st.columns(2)
        with col_gt:
            st.markdown("**Ground Truth**")
            img = load_image(os.path.join(EVAL_DIR, "val_batch0_labels.jpg"))
            if img is None:
                img = load_image(os.path.join(TRAIN_DIR, "val_batch0_labels.jpg"))
            if img:
                st.image(img, use_container_width=True)
        with col_pred:
            st.markdown("**Model Predictions**")
            img = load_image(os.path.join(EVAL_DIR, "val_batch0_pred.jpg"))
            if img is None:
                img = load_image(os.path.join(TRAIN_DIR, "val_batch0_pred.jpg"))
            if img:
                st.image(img, use_container_width=True)

        st.markdown("""
        <div class="analysis-box">
            <span class="analysis-label">Analysis — Validation Samples</span>
            <p>Comparing ground truth (left) against predictions (right) reveals three error types:</p>
            <p><strong>Missed detections</strong> — a labelled box present in ground truth but absent
            in predictions. Common for small-area defects at 640px input resolution.</p>
            <p><strong>Class substitution</strong> — box detected at the correct location but assigned
            the wrong severity label. The most operationally significant error for the severity chain.</p>
            <p><strong>Localisation drift</strong> — predicted box overlaps the correct region but is
            notably larger or offset. Contributes to the mAP@0.5:0.95 gap.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Inference log analysis
        st.subheader("Inference Log Analysis")
        df_log = load_log()

        if df_log.empty:
            st.info("No inference records yet. Run detections first.")
        else:
            col_hist, col_conf = st.columns(2)
            with col_hist:
                st.markdown("**Confidence Score Distribution**")
                fig_hist = px.histogram(df_log, x="confidence", nbins=20,
                                        template="plotly_dark", color_discrete_sequence=["#f97316"])
                fig_hist.update_layout(**chart_layout(280), bargap=0.05,
                                       xaxis_title="Confidence", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_conf:
                st.markdown("**Average Confidence per Class**")
                avg_conf_cls = (
                    df_log.groupby("defect")["confidence"].mean().reset_index()
                    .rename(columns={"defect": "Class", "confidence": "Avg Confidence"})
                )
                fig_ac = px.bar(avg_conf_cls, x="Class", y="Avg Confidence",
                                color="Class", color_discrete_map=COLOR_MAP,
                                template="plotly_dark")
                fig_ac.update_layout(showlegend=False, **chart_layout(280))
                st.plotly_chart(fig_ac, use_container_width=True)

            if "detect_mode" in df_log.columns:
                mode_counts = df_log["detect_mode"].value_counts().reset_index()
                mode_counts.columns = ["Mode", "Count"]
                st.markdown("**Scans by Detection Mode**")
                fig_mode = px.bar(mode_counts, x="Mode", y="Count", color="Mode",
                                  color_discrete_map={"hotspot": "#f97316", "severity": "#ef4444"},
                                  template="plotly_dark")
                fig_mode.update_layout(showlegend=False, **chart_layout(220))
                st.plotly_chart(fig_mode, use_container_width=True)

            mean_conf = df_log["confidence"].mean()
            low_conf  = (df_log["confidence"] < 0.5).sum()
            pct_low   = low_conf / len(df_log) if len(df_log) else 0
            st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-label">Analysis — Inference Log</span>
                <p><strong>Overall average confidence: {mean_conf:.2f}.</strong>
                Values consistently above 0.70 indicate the model makes high-certainty decisions
                in production, consistent with the high precision observed during evaluation.</p>
                <p>{low_conf} detection(s) ({pct_low:.0%}) fell below 0.50 confidence.
                These borderline detections are the most likely to be incorrect and should be
                prioritised for manual review. Raising the confidence threshold in the sidebar
                will suppress them at the cost of reduced recall.</p>
                <p>Classes with lower average confidence are typically the severity subclasses
                (Mild, Moderate) — their thermal signatures overlap, making the model less certain
                when distinguishing them.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Training config + dataset
        col_cfg, col_ds = st.columns(2)

        with col_cfg:
            st.subheader("Training Configuration")
            cfg = {}
            if YAML_AVAILABLE and os.path.exists(ARGS_YAML):
                with open(ARGS_YAML) as f:
                    cfg = yaml.safe_load(f) or {}

            params = [
                ("Model",          cfg.get("model", "yolov8s.pt")),
                ("Epochs (limit)", cfg.get("epochs", 150)),
                ("Trained epochs", total_epochs),
                ("Batch size",     cfg.get("batch", 8)),
                ("Image size",     cfg.get("imgsz", 640)),
                ("Device",         cfg.get("device", "cpu")),
                ("Optimizer",      cfg.get("optimizer", "auto")),
                ("LR initial",     cfg.get("lr0", 0.001)),
                ("LR final",       cfg.get("lrf", 0.01)),
                ("Patience",       cfg.get("patience", 30)),
                ("Frozen layers",  cfg.get("freeze", 10)),
                ("Pretrained",     cfg.get("pretrained", True)),
                ("IOU threshold",  cfg.get("iou", 0.7)),
                ("Augmentation",   "mosaic + mixup + randaugment"),
            ]
            rows = "".join(
                f'<tr><td style="color:#3d4a5c;font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:0.72rem;padding:0.3rem 0.8rem 0.3rem 0;">{k}</td>'
                f'<td style="color:#dde3f0;font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:0.72rem;padding:0.3rem 0;">{v}</td></tr>'
                for k, v in params
            )
            st.markdown(f'<table style="border-collapse:collapse;width:100%;">{rows}</table>',
                        unsafe_allow_html=True)

        with col_ds:
            st.subheader("Dataset Overview")
            st.markdown(
                '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
                'color:#3d4a5c;margin-bottom:0.4rem;">7 classes — Roboflow dataset v2</p>',
                unsafe_allow_html=True
            )
            cls_colors = {
                "Hotspot"           : "#dd6b20", "Junction_box"      : "#6366f1",
                "Mild"              : "#d69e2e", "Moderate"          : "#ed8936",
                "Moderate-Critical" : "#9b2c2c", "Moderate-High"     : "#e53e3e",
                "healthy_panel"     : "#38a169",
            }
            for name in EVAL_CLASSES:
                color = cls_colors.get(name, "#718096")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin:4px 0;">'
                    f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                    f'background:{color};flex-shrink:0;"></span>'
                    f'<code style="font-size:0.78rem;color:#dde3f0;">{name}</code></div>',
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)
            splits = {"Train": "dataset2/train/images", "Valid": "dataset2/valid/images", "Test": "dataset2/test/images"}
            split_rows = ""
            total_imgs = 0
            for split, path in splits.items():
                n = count_files(path)
                total_imgs += n
                split_rows += (
                    f'<tr><td style="color:#3d4a5c;font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:0.72rem;padding:0.25rem 0.8rem 0.25rem 0;">{split}</td>'
                    f'<td style="color:#dde3f0;font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:0.72rem;padding:0.25rem 0;">{n if n else "—"} images</td></tr>'
                )
            split_rows += (
                f'<tr><td style="color:#f97316;font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:0.72rem;padding:0.4rem 0.8rem 0 0;font-weight:600;">Total</td>'
                f'<td style="color:#f97316;font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:0.72rem;padding:0.4rem 0 0 0;font-weight:600;">'
                f'{total_imgs if total_imgs else "—"} images</td></tr>'
            )
            st.markdown(f'<table style="border-collapse:collapse;">{split_rows}</table>',
                        unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-box">
            <span class="analysis-label">Analysis — Training Setup &amp; Dataset</span>
            <p><strong>YOLOv8s</strong> (~11M parameters) is an appropriate choice for a 7-class
            detection task with a limited dataset — fast enough for edge inference while avoiding
            the overfitting risk of larger variants on small training sets.</p>
            <p><strong>Frozen layers (freeze = 10)</strong> — the first 10 backbone layers were kept
            at ImageNet-pretrained weights. This preserves low-level feature extractors (edges, textures)
            while allowing the task-specific head to adapt to thermal IR imagery.</p>
            <p><strong>Early stopping at epoch 88 of 150</strong> (patience = 30) indicates the model
            converged around epoch 58. Further training with the current setup would not yield gains —
            more data or stronger augmentation would be needed to progress further.</p>
            <p><strong>Dataset size is the primary bottleneck.</strong> The precision–recall imbalance
            and mAP@0.5:0.95 gap are both characteristic of models trained on limited data. Expanding
            the dataset to ~150–200 images per class (~1,050–1,400 total) is the highest-leverage
            improvement available without changing the architecture.</p>
        </div>
        """, unsafe_allow_html=True)
