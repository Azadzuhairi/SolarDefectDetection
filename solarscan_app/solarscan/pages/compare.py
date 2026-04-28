"""pages/compare.py — Before vs After image comparison"""

import streamlit as st
from PIL import Image
from utils.model import run_inference, get_severity_badge_html


def render():
    st.markdown('<p class="page-title">🔀 Compare</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Upload two scans of the same panel to compare defect changes over time</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="compare-header">BEFORE</div>', unsafe_allow_html=True)
        before_file = st.file_uploader(
            "Before image",
            type=["jpg","jpeg","png","tiff","tif"],
            key="before_upload",
            label_visibility="collapsed",
        )
        before_date = st.date_input("Scan date (before)", key="before_date")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="compare-header">AFTER</div>', unsafe_allow_html=True)
        after_file = st.file_uploader(
            "After image",
            type=["jpg","jpeg","png","tiff","tif"],
            key="after_upload",
            label_visibility="collapsed",
        )
        after_date = st.date_input("Scan date (after)", key="after_date")
        st.markdown('</div>', unsafe_allow_html=True)

    conf = st.session_state.conf_threshold
    run_compare = st.button("🔀  Compare scans", type="primary",
                             disabled=not (before_file and after_file))

    if run_compare and before_file and after_file:
        with st.spinner("Analyzing both images..."):
            img_before = Image.open(before_file).convert("RGB")
            img_after  = Image.open(after_file).convert("RGB")
            res_before = run_inference(img_before, conf=conf)
            res_after  = run_inference(img_after,  conf=conf)

        # ── side by side annotated images ──────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-label">Annotated results</div>', unsafe_allow_html=True)
        ic1, ic2 = st.columns(2, gap="medium")
        with ic1:
            st.markdown(f'<div class="compare-header">BEFORE — {before_date}</div>', unsafe_allow_html=True)
            st.image(res_before["annotated"], use_container_width=True)
        with ic2:
            st.markdown(f'<div class="compare-header">AFTER — {after_date}</div>', unsafe_allow_html=True)
            st.image(res_after["annotated"],  use_container_width=True)

        # ── change summary ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-label">Change summary</div>', unsafe_allow_html=True)

        cb = res_before["counts"]
        ca = res_after["counts"]

        m1, m2, m3, m4 = st.columns(4)
        def delta_metric(col, label, before, after, bad_direction="up"):
            diff = after - before
            col.metric(label, after, delta=diff,
                       delta_color="inverse" if bad_direction == "up" else "normal")

        delta_metric(m1, "🔴 High",   cb["HIGH"],   ca["HIGH"])
        delta_metric(m2, "🟡 Medium", cb["MEDIUM"], ca["MEDIUM"])
        delta_metric(m3, "🟢 Low",    cb["LOW"],    ca["LOW"])

        total_before = cb["HIGH"] + cb["MEDIUM"] + cb["LOW"]
        total_after  = ca["HIGH"] + ca["MEDIUM"] + ca["LOW"]
        delta_metric(m4, "Total defects", total_before, total_after)

        # ── detailed diff ───────────────────────────────────────────────────────
        st.markdown("---")
        dc1, dc2 = st.columns(2, gap="medium")

        with dc1:
            st.markdown('<div class="card-label">Before defects</div>', unsafe_allow_html=True)
            if res_before["detections"]:
                for d in res_before["detections"]:
                    badge = get_severity_badge_html(d["severity"])
                    dot   = {"HIGH":"dot-high","MEDIUM":"dot-med","LOW":"dot-low"}.get(d["severity"],"dot-low")
                    st.markdown(f"""
                    <div class="defect-row">
                        <div class="defect-dot {dot}"></div>
                        <span class="defect-name">{d['class'].replace('_',' ').title()}</span>
                        {badge}
                        <span class="defect-conf">{d['confidence']}%</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No defects in before scan")

        with dc2:
            st.markdown('<div class="card-label">After defects</div>', unsafe_allow_html=True)
            if res_after["detections"]:
                for d in res_after["detections"]:
                    badge = get_severity_badge_html(d["severity"])
                    dot   = {"HIGH":"dot-high","MEDIUM":"dot-med","LOW":"dot-low"}.get(d["severity"],"dot-low")
                    st.markdown(f"""
                    <div class="defect-row">
                        <div class="defect-dot {dot}"></div>
                        <span class="defect-name">{d['class'].replace('_',' ').title()}</span>
                        {badge}
                        <span class="defect-conf">{d['confidence']}%</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No defects in after scan")

    elif before_file and not run_compare:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.image(Image.open(before_file), use_container_width=True, caption="Before")
        if after_file:
            with c2:
                st.image(Image.open(after_file), use_container_width=True, caption="After")
