"""
Approved Companies page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import persistence
from core.email_drafter import draft_outreach_email

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 1rem 0 0.25rem 0;">
        <h1 style="margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">
            ✅ Approved Companies
        </h1>
        <p style="margin: 6px 0 0 0; color: #5e6680; font-size: 0.875rem;">
            Staffing prospects you've approved — persisted locally in data/decisions.json, survives restarts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

approved = persistence.get_by_status("approved")

if not approved:
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    st.info("No companies approved yet. Approve prospects from the Discovery page and they'll show up here.")
    st.stop()

ranked = sorted(approved.items(), key=lambda kv: -(kv[1].get("score") or 0))

TIER_COLOR = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
TIER_BADGE = {
    "High":   ("#166534", "#dcfce7", "#22c55e"),
    "Medium": ("#78350f", "#fef3c7", "#f59e0b"),
    "Low":    ("#7f1d1d", "#fee2e2", "#ef4444"),
}

# ── Summary metrics ───────────────────────────────────────────────────────────
st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.metric("Total approved", len(approved))
high_count = sum(1 for _, v in ranked if v.get("tier") == "High")
col2.metric("High-tier approved", high_count)

st.divider()

# ── Company cards ─────────────────────────────────────────────────────────────
for key, record in ranked:
    tier = record.get("tier")
    tier_icon = TIER_COLOR.get(tier, "⚪")
    score = record.get("score")

    score_html = (
        f'<span style="color:#94a3b8; font-size:0.85rem; margin-left:8px;">Score </span>'
        f'<span style="color:#e2e8f0; font-weight:700; font-size:0.95rem;">{score}</span>'
        f'<span style="color:#64748b; font-size:0.85rem;">/100</span>'
        if score is not None else ""
    )

    if tier and tier in TIER_BADGE:
        bg, txt_col, border = TIER_BADGE[tier]
        badge_html = (
            f'<span style="background:{bg}; color:{txt_col}; border:1px solid {border}; '
            f'border-radius:5px; padding:2px 9px; font-size:0.76rem; font-weight:600; '
            f'margin-left:10px;">{tier}</span>'
        )
    else:
        badge_html = ""

    contacts = record.get("contacts") or []

    with st.container(border=True):

        # ── Card header ──────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem;">
                <span style="font-size:1.15rem;">{tier_icon}</span>
                <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">
                    {record['company_name']}
                </span>
                {score_html}
                {badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Meta row ─────────────────────────────────────────────────────────
        meta_cols = st.columns(3)
        meta_cols[0].markdown(f"**Website**  \n{record.get('website') or 'Unknown'}")
        meta_cols[1].markdown(f"**Approved at**  \n{record.get('decided_at', 'Unknown')}")
        meta_cols[2].markdown(f"**Urgency**  \n{record.get('urgency') or 'Unknown'}")

        if record.get("reasoning"):
            st.markdown(
                f'<div style="margin:0.4rem 0; font-size:0.85rem; color:#cbd5e1; '
                f'background:#0f172a; border-left:3px solid #22c55e; padding:0.5rem 0.75rem; '
                f'border-radius:0 5px 5px 0;">'
                f'<strong style="color:#86efac;">Qualification:</strong> {record["reasoning"]}</div>',
                unsafe_allow_html=True,
            )

        if record.get("recommended_outreach"):
            st.markdown(
                f'<div style="margin:0.4rem 0; font-size:0.85rem; background:#0f1f3d; '
                f'border-left:3px solid #3b82f6; padding:0.5rem 0.75rem; border-radius:0 5px 5px 0;">'
                f'<strong style="color:#93c5fd;">Recommendation:</strong> {record["recommended_outreach"]} '
                f'<span style="color:#64748b; font-style:italic;">'
                f'(Contact: {record.get("recommended_contact", "Unknown")})</span></div>',
                unsafe_allow_html=True,
            )

        if contacts:
            st.markdown(
                '<div style="font-size:0.8rem; font-weight:600; color:#7b82a0; '
                'text-transform:uppercase; letter-spacing:0.06em; margin: 0.5rem 0 0.3rem 0;">'
                'Decision Makers</div>',
                unsafe_allow_html=True,
            )
            for c in contacts:
                c_email = c.get("email", "Unknown")
                st.markdown(
                    f'<div style="font-size:0.84rem; padding:3px 0; color:#c9cdd7;">'
                    f'<strong style="color:#e2e8f0;">{c.get("name", "Unknown")}</strong>'
                    f' — <span style="color:#94a3b8;">{c.get("role") or c.get("matched_persona", "Unknown role")}</span>'
                    f' · <span style="color:#60a5fa;">{c_email}</span></div>',
                    unsafe_allow_html=True,
                )

        if record.get("feedback"):
            st.markdown(
                f'<div style="margin-top:0.5rem; font-size:0.82rem; color:#7b82a0; '
                f'background:#111318; border:1px solid #1f2636; padding:0.4rem 0.6rem; '
                f'border-radius:5px;">📝 {record["feedback"]}</div>',
                unsafe_allow_html=True,
            )

        # ── Action buttons ───────────────────────────────────────────────────
        action_cols = st.columns([1, 1])

        with action_cols[0]:
            if st.button("↩️ Move to Rejected", key=f"unapprove_{key}"):
                persistence.save_decision(
                    company_key=key,
                    status="rejected",
                    company_name=record["company_name"],
                    website=record.get("website"),
                    score=record.get("score"),
                    tier=record.get("tier"),
                    reasoning=record.get("reasoning"),
                    recommended_contact=record.get("recommended_contact"),
                    recommended_outreach=record.get("recommended_outreach"),
                    urgency=record.get("urgency"),
                    contacts=record.get("contacts"),
                    feedback=record.get("feedback"),
                )
                st.rerun()

        with action_cols[1]:
            if st.button("✉️ Draft Mail", key=f"draft_mail_{key}", type="primary"):
                st.session_state[f"show_draft_{key}"] = True
                st.session_state[f"draft_result_{key}"] = None
                st.session_state[f"draft_error_{key}"] = None
                st.rerun()

        # ── Draft panel ──────────────────────────────────────────────────────
        if st.session_state.get(f"show_draft_{key}"):

            # Generate once, cache in session state
            if st.session_state.get(f"draft_result_{key}") is None \
                    and not st.session_state.get(f"draft_error_{key}"):
                with st.spinner("🤖 Generating email draft..."):
                    try:
                        result = draft_outreach_email(record)
                        st.session_state[f"draft_result_{key}"] = result
                    except Exception as exc:
                        st.session_state[f"draft_error_{key}"] = str(exc)

            if st.session_state.get(f"draft_error_{key}"):
                st.error(f"Draft failed: {st.session_state[f'draft_error_{key}']}")

            draft = st.session_state.get(f"draft_result_{key}")
            if draft:
                st.markdown(
                    '<div style="margin-top:0.75rem; background:#0a0f1a; border:1px solid #1e3a5f; '
                    'border-radius:8px; padding:1.1rem 1.3rem;">',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div style="font-size:0.75rem; font-weight:600; color:#60a5fa; '
                    'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">'
                    '✉️ Generated Email Draft</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:0.82rem; color:#7b82a0; margin-bottom:2px;">Subject</div>'
                    f'<div style="font-size:1rem; font-weight:600; color:#e2e8f0; '
                    f'margin-bottom:1rem;">{draft["subject"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div style="font-size:0.82rem; color:#7b82a0; margin-bottom:6px;">Body</div>',
                    unsafe_allow_html=True,
                )
                # Render body with newlines preserved
                body_html = draft["body"].replace("\n", "<br>")
                st.markdown(
                    f'<div style="font-size:0.88rem; color:#c9cdd7; line-height:1.7;">'
                    f'{body_html}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                close_col, regen_col = st.columns([1, 1])
                with regen_col:
                    if st.button("🔄 Regenerate", key=f"regen_{key}"):
                        st.session_state[f"draft_result_{key}"] = None
                        st.session_state[f"draft_error_{key}"] = None
                        st.rerun()
                with close_col:
                    if st.button("✖ Close", key=f"close_draft_{key}"):
                        st.session_state.pop(f"show_draft_{key}", None)
                        st.session_state.pop(f"draft_result_{key}", None)
                        st.session_state.pop(f"draft_error_{key}", None)
                        st.rerun()

st.divider()

# ── Export table ──────────────────────────────────────────────────────────────
rows = [
    {
        "Company": r["company_name"],
        "Website": r.get("website"),
        "Score": r.get("score"),
        "Tier": r.get("tier"),
        "Recommended Contact": r.get("recommended_contact"),
        "Next Action": r.get("recommended_outreach"),
        "Urgency": r.get("urgency"),
        "Approved At": r.get("decided_at"),
        "Reasoning": r.get("reasoning"),
    }
    for _, r in ranked
]
df = pd.DataFrame(rows)
st.dataframe(df, width='stretch', hide_index=True)
st.download_button(
    "⬇️ Download Approved CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="approved_companies.csv",
    mime="text/csv",
    width='stretch',
)