"""
SolarScan — PV Panel Defect Detection App
Main entry point
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="SolarScan — PV Defect Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── inject global CSS ──────────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── session state defaults ─────────────────────────────────────────────────────
defaults = {
    "page": "detect",
    "notifications": [
        {"id": 1, "type": "high",  "title": "Critical hotspot detected — Panel B2", "msg": "Confidence 94% · Immediate inspection recommended", "read": False},
        {"id": 2, "type": "high",  "title": "Severe crack detected — Panel B3",     "msg": "Confidence 87% · Panel replacement advised",         "read": False},
        {"id": 3, "type": "med",   "title": "Delamination worsening — Panel F6",    "msg": "Compared to last scan — severity increased",          "read": False},
        {"id": 4, "type": "ok",    "title": "Batch scan complete — 12 images",      "msg": "2 hours ago · 3 defects found",                       "read": True},
    ],
    "scan_history": [],
    "batch_files": [],
    "conf_threshold": 0.25,
    "image_type": "Auto-detect",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">☀️</div>
        <div>
            <div class="logo-title">SolarScan</div>
            <div class="logo-sub">PV Defect Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav_items = [
        ("🔍", "detect",    "Detect"),
        ("📊", "dashboard", "Dashboard"),
        ("📁", "batch",     "Batch Upload"),
        ("🗺️",  "map",       "Map View"),
        ("🔀", "compare",   "Compare"),
        ("🔔", "alerts",    "Alerts"),
    ]

    for icon, key, label in nav_items:
        unread = ""
        if key == "alerts":
            count = sum(1 for n in st.session_state.notifications if not n["read"])
            if count:
                unread = f" 🔴 {count}"
        active = "nav-active" if st.session_state.page == key else ""
        if st.button(f"{icon}  {label}{unread}", key=f"nav_{key}",
                     use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#888;padding:0 8px;'>
        Farm A · Block 3<br>
        Last sync: just now
    </div>
    """, unsafe_allow_html=True)

# ── route to page ──────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "detect":
    from pages.detect import render
    render()
elif page == "dashboard":
    from pages.dashboard import render
    render()
elif page == "batch":
    from pages.batch import render
    render()
elif page == "map":
    from pages.map_view import render
    render()
elif page == "compare":
    from pages.compare import render
    render()
elif page == "alerts":
    from pages.alerts import render
    render()
