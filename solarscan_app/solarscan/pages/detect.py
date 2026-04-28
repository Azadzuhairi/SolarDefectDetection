"""pages/detect.py — Single image upload & detection"""

import streamlit as st
from PIL import Image
import io
from utils.model import run_inference, get_severity_badge_html


def render():
    st.markdown('<p class="page-title">🔍 Detect Defects</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Upload a single IR or RGB image to run defect detection</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    # ── LEFT: upload controls ──────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Upload image</div>', unsafe_allow_html=True)

        img_type = st.radio(
            "Image type",
            ["IR (Infrared)", "RGB (Real)", "Auto-detect"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.image_type = img_type

        uploaded = st.file_uploader(
            "Drop image here",
            type=["jpg", "jpeg", "png", "tiff", "tif"],
            label_visibility="collapsed",
        )

        conf = st.slider(
            "Confidence threshold",
            min_value=0.05, max_value=0.95,
            value=st.session_state.conf_threshold,
            step=0.05,
            format="%.2f",
        )
        st.session_state.conf_threshold = conf

        run = st.button("🔍  Analyze image", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── model status hint ──
        from utils.model import WEIGHTS_PATH
        if WEIGHTS_PATH.exists():
            st.success("✅ Trained model loaded", icon="✅")
        else:
            st.warning("⚠️ No trained weights found. Train your model first.", icon="⚠️")

    # ── RIGHT: results ─────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Detection preview</div>', unsafe_allow_html=True)

        if uploaded and run:
            image = Image.open(uploaded).convert("RGB")
            with st.spinner("Running inference..."):
                result = run_inference(image, conf=conf)

            # show warning if untrained
            if "warning" in result:
                st.warning(result["warning"])

            # show annotated image
            st.image(result["annotated"], use_container_width=True)

            # counts
            c = result["counts"]
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 High",   c["HIGH"])
            m2.metric("🟡 Medium", c["MEDIUM"])
            m3.metric("🟢 Low",    c["LOW"])

            st.markdown("---")

            if result["detections"]:
                st.markdown('<div class="card-label">Detected defects</div>', unsafe_allow_html=True)
                for d in result["detections"]:
                    badge = get_severity_badge_html(d["severity"])
                    dot_cls = {"HIGH": "dot-high", "MEDIUM": "dot-med", "LOW": "dot-low"}.get(d["severity"], "dot-low")
                    st.markdown(f"""
                    <div class="defect-row">
                        <div class="defect-dot {dot_cls}"></div>
                        <span class="defect-name">{d['class'].replace('_', ' ').title()}</span>
                        {badge}
                        <span class="defect-conf">{d['confidence']}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # save to history
                st.session_state.scan_history.insert(0, {
                    "name":      uploaded.name,
                    "type":      img_type,
                    "timestamp": result["timestamp"],
                    "counts":    c,
                    "detections": result["detections"],
                    "severity":  "HIGH" if c["HIGH"] else ("MEDIUM" if c["MEDIUM"] else "LOW"),
                })

                # export buttons
                st.markdown("---")
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.button("📄 PDF report", use_container_width=True)
                with e2:
                    import json
                    json_data = json.dumps(result["detections"], indent=2)
                    st.download_button("📥 JSON", json_data, file_name="defects.json",
                                       mime="application/json", use_container_width=True)
                with e3:
                    st.button("💾 Save", use_container_width=True)
            else:
                st.info("No defects detected above the confidence threshold.")

        elif uploaded and not run:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
            st.caption("Press **Analyze image** to run detection")
        else:
            st.markdown("""
            <div class="upload-hint">
                Upload an image on the left<br>then press Analyze
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── recent scans ───────────────────────────────────────────────────────────
    if st.session_state.scan_history:
        st.markdown("---")
        st.markdown('<div class="card-label">Recent scans</div>', unsafe_allow_html=True)
        for scan in st.session_state.scan_history[:5]:
            sev = scan["severity"]
            badge = get_severity_badge_html(sev)
            c = scan["counts"]
            st.markdown(f"""
            <div class="file-row">
                <span style="font-weight:500;flex:1;">{scan['name']}</span>
                <span style="color:#6b7280;font-size:11px;">{scan['timestamp']}</span>
                <span style="color:#6b7280;font-size:11px;margin:0 8px;">
                    🔴{c['HIGH']} 🟡{c['MEDIUM']} 🟢{c['LOW']}
                </span>
                {badge}
            </div>
            """, unsafe_allow_html=True)
