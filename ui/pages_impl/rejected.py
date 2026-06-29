"""
Rejected Companies page.

Same persistence model as approved.py — reads from
core/persistence.py's local JSON file, not session state, so rejections
made in past sessions are still visible here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import persistence  # noqa: E402

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 1rem 0 0.25rem 0;">
        <h1 style="margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">
            ❌ Rejected Companies
        </h1>
        <p style="margin: 6px 0 0 0; color: #5e6680; font-size: 0.875rem;">
            Prospects you've passed on — persisted locally in data/decisions.json, survives restarts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

rejected = persistence.get_by_status("rejected")

if not rejected:
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    st.info("No companies rejected yet. Reject prospects from the Discovery page and they'll show up here.")
    st.stop()

ranked = sorted(rejected.items(), key=lambda kv: -(kv[1].get("score") or 0))

TIER_BADGE = {
    "High":   ("#166534", "#dcfce7", "#22c55e"),
    "Medium": ("#78350f", "#fef3c7", "#f59e0b"),
    "Low":    ("#7f1d1d", "#fee2e2", "#ef4444"),
}

# ── Summary metric ────────────────────────────────────────────────────────────
st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
st.metric("Total rejected", len(rejected))
st.divider()

# ── Company cards ─────────────────────────────────────────────────────────────
for key, record in ranked:
    tier = record.get("tier")
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

    with st.container(border=True):
        # Card header
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem;">
                <span style="font-size:1.15rem;">⚪</span>
                <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">
                    {record['company_name']}
                </span>
                {score_html}
                {badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Meta row
        meta_cols = st.columns(3)
        meta_cols[0].markdown(f"**Website**  \n{record.get('website') or 'Unknown'}")
        meta_cols[1].markdown(f"**Rejected at**  \n{record.get('decided_at', 'Unknown')}")
        meta_cols[2].markdown(f"**Tier at rejection**  \n{record.get('tier') or 'Unknown'}")

        if record.get("reasoning"):
            st.markdown(
                f'<div style="margin:0.4rem 0; font-size:0.85rem; color:#cbd5e1; '
                f'background:#1a0e0e; border-left:3px solid #ef4444; padding:0.5rem 0.75rem; '
                f'border-radius:0 5px 5px 0;">'
                f'<strong style="color:#fca5a5;">Qualification (at rejection):</strong> {record["reasoning"]}</div>',
                unsafe_allow_html=True,
            )

        if record.get("feedback"):
            st.markdown(
                f'<div style="margin-top:0.5rem; font-size:0.82rem; color:#7b82a0; '
                f'background:#111318; border:1px solid #1f2636; padding:0.4rem 0.6rem; '
                f'border-radius:5px;">📝 {record["feedback"]}</div>',
                unsafe_allow_html=True,
            )

        if st.button("↩️ Move to Approved", key=f"unreject_{key}"):
            persistence.save_decision(
                company_key=key,
                status="approved",
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

st.divider()

# ── Export table ──────────────────────────────────────────────────────────────
rows = [
    {
        "Company": r["company_name"],
        "Website": r.get("website"),
        "Score": r.get("score"),
        "Tier": r.get("tier"),
        "Rejected At": r.get("decided_at"),
        "Reasoning": r.get("reasoning"),
        "Notes": r.get("feedback"),
    }
    for _, r in ranked
]
df = pd.DataFrame(rows)
st.dataframe(df, width='stretch', hide_index=True)
st.download_button(
    "⬇️ Download Rejected CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="rejected_companies.csv",
    mime="text/csv",
    width='stretch',
)