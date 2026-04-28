"""pages/dashboard.py — Stats overview & charts"""

import streamlit as st
import json


DEFECT_TYPES = [
    ("Hotspot severe",    14, "bar-fill-red"),
    ("Crack severe",      10, "bar-fill-red"),
    ("Delamination",       7, "bar-fill-amber"),
    ("Hotspot moderate",   5, "bar-fill-amber"),
    ("Soiling",            4, "bar-fill-green"),
    ("Snail trail",        2, "bar-fill-amber"),
    ("Crack minor",        2, "bar-fill-green"),
]
TOTAL_DEFECTS = sum(x[1] for x in DEFECT_TYPES)

WEEKLY = [
    ("Mon", 28), ("Tue", 35), ("Wed", 42),
    ("Thu", 30), ("Fri", 55), ("Sat", 38), ("Sun", 20),
]
MAX_WEEKLY = max(v for _, v in WEEKLY)

SEVERITY_DIST = [
    ("HIGH",   24, "#ef4444"),
    ("MEDIUM", 14, "#f59e0b"),
    ("LOW",     6, "#22c55e"),
]


def render():
    st.markdown('<p class="page-title">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Overview of all scanned PV panels and detected defects</p>', unsafe_allow_html=True)

    # ── top stats ──────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-lbl">Total scanned</div>
            <div class="stat-num">248</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-lbl">Defects found</div>
            <div class="stat-num" style="color:#ef4444;">44</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-lbl">Healthy panels</div>
            <div class="stat-num" style="color:#1D9E75;">204</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-lbl">Avg confidence</div>
            <div class="stat-num">82%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    # ── defects by type ────────────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Defects by type</div>', unsafe_allow_html=True)

        bars_html = ""
        for name, count, bar_cls in DEFECT_TYPES:
            pct = round(count / TOTAL_DEFECTS * 100)
            bars_html += f"""
            <div class="bar-row">
                <span class="bar-label">{name}</span>
                <div class="bar-track">
                    <div class="{bar_cls}" style="width:{pct}%;"></div>
                </div>
                <span class="bar-count">{count}</span>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── weekly scans ───────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Weekly scans</div>', unsafe_allow_html=True)

        # simple HTML bar chart
        chart_html = '<div style="display:flex;align-items:flex-end;gap:6px;height:100px;margin-bottom:6px;">'
        for day, val in WEEKLY:
            h = round(val / MAX_WEEKLY * 90)
            is_peak = val == MAX_WEEKLY
            color = "#1D9E75" if is_peak else "#2a3045"
            border = "2px solid #1D9E75" if is_peak else "none"
            chart_html += f"""
            <div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;">
                <span style="font-size:10px;color:#6b7280;">{val}</span>
                <div style="width:100%;height:{h}px;background:{color};border-radius:4px 4px 0 0;border-top:{border};"></div>
                <span style="font-size:10px;color:#6b7280;">{day}</span>
            </div>"""
        chart_html += "</div>"
        st.markdown(chart_html, unsafe_allow_html=True)

        st.markdown("---")

        # severity distribution
        st.markdown('<div class="card-label">Severity distribution</div>', unsafe_allow_html=True)
        for sev, count, color in SEVERITY_DIST:
            pct = round(count / 44 * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
                <span style="font-size:12px;color:#d1d5db;width:60px;">{sev}</span>
                <div class="bar-track" style="flex:1;">
                    <div style="height:100%;width:{pct}%;background:{color};border-radius:3px;"></div>
                </div>
                <span style="font-size:11px;color:#6b7280;width:40px;text-align:right;">{count} ({pct}%)</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── recent activity from session ───────────────────────────────────────────
    if st.session_state.scan_history:
        st.markdown("---")
        st.markdown('<div class="card-label">Session scans</div>', unsafe_allow_html=True)
        from utils.model import get_severity_badge_html
        for scan in st.session_state.scan_history[:8]:
            c = scan["counts"]
            badge = get_severity_badge_html(scan["severity"])
            st.markdown(f"""
            <div class="file-row">
                <span style="font-weight:500;flex:1;">{scan['name']}</span>
                <span style="color:#6b7280;font-size:11px;">{scan['timestamp']}</span>
                <span style="color:#6b7280;font-size:11px;margin:0 8px;">
                    🔴{c['HIGH']} 🟡{c['MEDIUM']} 🟢{c['LOW']}
                </span>
                {badge}
            </div>""", unsafe_allow_html=True)
