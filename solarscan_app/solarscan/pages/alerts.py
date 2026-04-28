"""pages/alerts.py — Notification & alert system"""

import streamlit as st


ICONS = {
    "high": ("🔴", "notif-high"),
    "med":  ("🟡", "notif-med"),
    "ok":   ("✅", "notif-ok"),
}


def render():
    st.markdown('<p class="page-title">🔔 Alerts</p>', unsafe_allow_html=True)

    unread = sum(1 for n in st.session_state.notifications if not n["read"])
    st.markdown(f'<p class="page-sub">{unread} unread alert{"s" if unread != 1 else ""}</p>',
                unsafe_allow_html=True)

    # ── action buttons ─────────────────────────────────────────────────────────
    a1, a2, _ = st.columns([1, 1, 4])
    with a1:
        if st.button("✅  Mark all read", use_container_width=True):
            for n in st.session_state.notifications:
                n["read"] = True
            st.rerun()
    with a2:
        if st.button("🗑️  Clear all", use_container_width=True):
            st.session_state.notifications = []
            st.rerun()

    st.markdown("---")

    if not st.session_state.notifications:
        st.info("No alerts. All panels healthy! ✅")
        return

    # ── notification list ──────────────────────────────────────────────────────
    for i, notif in enumerate(st.session_state.notifications):
        icon, icon_cls = ICONS.get(notif["type"], ("ℹ️", "notif-ok"))
        read_cls = "notif-read" if notif["read"] else ""

        st.markdown(f"""
        <div class="notif-item {read_cls}">
            <div class="notif-icon {icon_cls}" style="font-size:16px;">{icon}</div>
            <div style="flex:1;">
                <div class="notif-title">{notif['title']}</div>
                <div class="notif-msg">{notif['msg']}</div>
            </div>
            {'<span class="badge badge-ok">READ</span>' if notif["read"] else
             ('<span class="badge badge-high">NEW</span>' if notif["type"]=="high"
              else '<span class="badge badge-med">NEW</span>')}
        </div>""", unsafe_allow_html=True)

        if not notif["read"]:
            if st.button("Mark as read", key=f"read_{i}", use_container_width=False):
                st.session_state.notifications[i]["read"] = True
                st.rerun()

    # ── alert settings ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-label">Alert settings</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns(2, gap="medium")
    with sc1:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Notify me when</div>', unsafe_allow_html=True)
        st.checkbox("HIGH severity defect detected",  value=True)
        st.checkbox("MEDIUM severity defect detected", value=True)
        st.checkbox("LOW severity defect detected",    value=False)
        st.checkbox("Batch scan complete",             value=True)
        st.checkbox("Defect worsened since last scan", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Notification method</div>', unsafe_allow_html=True)
        st.checkbox("In-app notifications", value=True)
        st.checkbox("Email alerts",         value=False)
        st.text_input("Email address", placeholder="you@example.com")
        st.checkbox("SMS alerts",           value=False)
        st.text_input("Phone number",   placeholder="+60 12-345 6789")
        st.markdown('</div>', unsafe_allow_html=True)
