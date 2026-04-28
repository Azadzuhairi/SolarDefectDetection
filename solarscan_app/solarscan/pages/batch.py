"""pages/batch.py — Batch upload & process multiple images"""

import streamlit as st
from PIL import Image
from utils.model import run_inference, get_severity_badge_html
import time


def render():
    st.markdown('<p class="page-title">📁 Batch Upload</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Upload and analyze multiple IR or RGB images at once</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Upload multiple images</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop images here",
            type=["jpg", "jpeg", "png", "tiff", "tif"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        conf = st.slider(
            "Confidence threshold",
            0.05, 0.95,
            value=st.session_state.conf_threshold,
            step=0.05,
            format="%.2f",
            key="batch_conf",
        )

        run_all = st.button("▶  Run all", type="primary", use_container_width=True,
                            disabled=not uploaded_files)
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_files:
            st.markdown(f'<div class="card-label">{len(uploaded_files)} files queued</div>',
                        unsafe_allow_html=True)
            for f in uploaded_files:
                size_kb = round(f.size / 1024)
                st.markdown(f"""
                <div class="file-row">
                    <span style="font-weight:500;flex:1;">{f.name}</span>
                    <span style="color:#6b7280;font-size:11px;">{size_kb} KB</span>
                    <span class="badge badge-ok">QUEUED</span>
                </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Results</div>', unsafe_allow_html=True)

        if run_all and uploaded_files:
            batch_results = []
            progress = st.progress(0, text="Processing...")
            status_area = st.empty()

            for i, f in enumerate(uploaded_files):
                status_area.markdown(f"**Processing:** `{f.name}`")
                image = Image.open(f).convert("RGB")
                result = run_inference(image, conf=conf)
                result["filename"] = f.name
                batch_results.append(result)

                # add to session history
                c = result["counts"]
                sev = "HIGH" if c["HIGH"] else ("MEDIUM" if c["MEDIUM"] else "LOW")
                st.session_state.scan_history.insert(0, {
                    "name":       f.name,
                    "type":       "Batch",
                    "timestamp":  result["timestamp"],
                    "counts":     c,
                    "detections": result["detections"],
                    "severity":   sev,
                })

                pct = round((i + 1) / len(uploaded_files) * 100)
                progress.progress(pct / 100, text=f"Processing... {pct}%")
                time.sleep(0.1)

            status_area.empty()
            progress.empty()

            # summary
            total_high = sum(r["counts"]["HIGH"]   for r in batch_results)
            total_med  = sum(r["counts"]["MEDIUM"] for r in batch_results)
            total_low  = sum(r["counts"]["LOW"]    for r in batch_results)

            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 High",   total_high)
            m2.metric("🟡 Medium", total_med)
            m3.metric("🟢 Low",    total_low)

            st.markdown("---")

            for res in batch_results:
                c   = res["counts"]
                sev = "HIGH" if c["HIGH"] else ("MEDIUM" if c["MEDIUM"] else "LOW")
                badge = get_severity_badge_html(sev)
                nd  = c["HIGH"] + c["MEDIUM"] + c["LOW"]
                st.markdown(f"""
                <div class="file-row">
                    <span style="font-weight:500;flex:1;">{res['filename']}</span>
                    <span style="color:#6b7280;font-size:11px;">{nd} defect{'s' if nd != 1 else ''}</span>
                    {badge}
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            import json
            export_data = [
                {"file": r["filename"], "detections": r["detections"]}
                for r in batch_results
            ]
            st.download_button(
                "📥 Export all results (JSON)",
                json.dumps(export_data, indent=2),
                file_name="batch_results.json",
                mime="application/json",
                use_container_width=True,
            )

        else:
            st.markdown("""
            <div class="upload-hint" style="margin-top:1rem;">
                Results will appear here after processing
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
