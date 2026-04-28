"""pages/map_view.py — Solar farm panel grid map"""

import streamlit as st
import random

# ── farm grid data (8x8 = 64 panels) ──────────────────────────────────────────
GRID_ROWS = 8
GRID_COLS = 8

CELL_STATUS = {
    "CLEAR":    ("#9FE1CB", "#085041"),
    "LOW":      ("#C0DD97", "#27500A"),
    "MEDIUM":   ("#FAC775", "#633806"),
    "HIGH":     ("#F09595", "#791F1F"),
    "CRITICAL": ("#E24B4A", "#501313"),
}

# predefined layout — in your real app this comes from scan results
PANEL_DATA = [
    ["CLEAR","CLEAR","HIGH","CLEAR","CLEAR","MEDIUM","CLEAR","CLEAR"],
    ["CLEAR","CRITICAL","CRITICAL","CLEAR","CLEAR","CLEAR","LOW","CLEAR"],
    ["CLEAR","CLEAR","CLEAR","MEDIUM","CLEAR","CLEAR","CLEAR","HIGH"],
    ["CLEAR","CLEAR","CLEAR","CLEAR","CLEAR","CLEAR","CLEAR","CLEAR"],
    ["LOW","CLEAR","CLEAR","CLEAR","HIGH","CLEAR","CLEAR","CLEAR"],
    ["CLEAR","CLEAR","MEDIUM","CLEAR","CLEAR","CLEAR","CLEAR","CLEAR"],
    ["CLEAR","CLEAR","CLEAR","CLEAR","CLEAR","LOW","CLEAR","CLEAR"],
    ["CLEAR","CLEAR","CLEAR","CLEAR","CLEAR","CLEAR","CLEAR","CLEAR"],
]

COUNTS = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "CLEAR": 0}
for row in PANEL_DATA:
    for s in row:
        COUNTS[s] += 1


def render():
    st.markdown('<p class="page-title">🗺️ Map View</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Visual grid of the solar farm — click a panel to see details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Farm A — Block 3 (8 × 8 grid)</div>', unsafe_allow_html=True)

        # build the grid as HTML
        grid_html = '<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:5px;padding:12px;background:#131720;border-radius:10px;margin-bottom:10px;">'
        for r, row in enumerate(PANEL_DATA):
            for c, status in enumerate(row):
                bg, tc = CELL_STATUS[status]
                panel_id = f"{chr(65+r)}{c+1}"
                title = f"Panel {panel_id}: {status}"
                grid_html += f'<div title="{title}" style="aspect-ratio:1;background:{bg};border-radius:4px;cursor:pointer;border:1px solid rgba(0,0,0,.15);"></div>'
        grid_html += "</div>"

        # legend
        grid_html += '<div style="display:flex;gap:14px;flex-wrap:wrap;">'
        labels = [("CRITICAL","#E24B4A"),("HIGH","#F09595"),("MEDIUM","#FAC775"),("LOW","#C0DD97"),("CLEAR","#9FE1CB")]
        for lbl, col in labels:
            grid_html += f'<div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#9ca3af;"><div style="width:10px;height:10px;background:{col};border-radius:3px;"></div>{lbl}</div>'
        grid_html += "</div>"

        st.markdown(grid_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Summary</div>', unsafe_allow_html=True)

        for status, (bg, tc) in CELL_STATUS.items():
            count = COUNTS[status]
            pct   = round(count / 64 * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <div style="width:10px;height:10px;background:{bg};border-radius:3px;flex-shrink:0;"></div>
                <span style="font-size:12px;color:#d1d5db;flex:1;">{status}</span>
                <div class="bar-track" style="width:80px;">
                    <div style="height:100%;width:{pct}%;background:{bg};border-radius:3px;"></div>
                </div>
                <span style="font-size:11px;color:#6b7280;width:28px;text-align:right;">{count}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        critical_panels = [
            f"{chr(65+r)}{c+1}"
            for r, row in enumerate(PANEL_DATA)
            for c, s in enumerate(row)
            if s in ("CRITICAL", "HIGH")
        ]

        st.markdown('<div class="card-label">Panels needing attention</div>', unsafe_allow_html=True)
        for pid in critical_panels:
            status = next(
                PANEL_DATA[ord(pid[0])-65][int(pid[1:])-1]
                for _ in [None]
            )
            bg, _ = CELL_STATUS[status]
            st.markdown(f"""
            <div class="file-row" style="margin-bottom:5px;">
                <div style="width:8px;height:8px;background:{bg};border-radius:50%;"></div>
                <span style="font-weight:500;">Panel {pid}</span>
                <span class="badge {'badge-high' if status=='CRITICAL' else 'badge-high'}">{status}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── filter controls ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">Filter by severity</div>', unsafe_allow_html=True)
    fcols = st.columns(5)
    filters = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    for i, f in enumerate(filters):
        with fcols[i]:
            st.button(f, use_container_width=True, key=f"filter_{f}")
