"""
Discovery page — the main pipeline dashboard: sidebar ICP configuration,
Run Discovery button, execution status/trace, company cards, and the
approve/reject/edit/re-score controls.

This is functionally the original single-page app, with one addition:
every Approve/Reject/Edit decision is now ALSO written to
core/persistence.py's local JSON file (data/decisions.json), so the
Approved Companies and Rejected Companies pages can show it even after a
page refresh or app restart, not just within this session.

Design notes on how this UI drives the LangGraph graph, since
Streamlit's rerun-on-every-interaction execution model doesn't map onto
a long-running graph invoke() the same way a backend server would:

- The compiled graph and all run state live in st.session_state, so they
  survive reruns triggered by widget interaction.
- "Run Discovery" invokes the graph ONCE per click. Because every agent
  routes back through the Planner (see core/graph.py), one invoke() call
  walks the ENTIRE pipeline for this cycle — market trigger through
  human approval — and stops automatically once the Planner has nothing
  left to do (or hits its own circuit breakers). We don't need a manual
  step-by-step loop; LangGraph's Command-based routing already handles
  the full multi-agent walk in one invoke().
- The Human Approval Agent only marks recommendations "pending" — it does
  NOT block. This UI is what actually gates moving forward: clicking
  Approve/Reject/Edit/Re-score calls
  agents.human_approval.apply_user_decision() directly against session
  state (for the in-memory graph state) AND core.persistence.save_decision
  / remove_decision (for the durable on-disk record), then (for
  "Re-score" only) the next "Run Discovery" click re-invokes the graph,
  which the Planner will route back through Qualification ->
  Recommendation -> Human Approval for that company since its
  qualification/recommendation entries were cleared.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load .env BEFORE importing anything that reads os.environ at construction
# time (SearchClient/LLMClient read their keys lazily inside __init__, but
# loading early here means the sidebar's key-presence indicators are
# accurate from the very first render).
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from agents.human_approval import apply_user_decision  # noqa: E402
from config.schema import RunConfig  # noqa: E402
from core.graph import build_graph, run_graph  # noqa: E402
from core.state import empty_state  # noqa: E402
from core import persistence  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "icp_config.yaml"


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def _init_session():
    if "platform_state" not in st.session_state:
        st.session_state.platform_state = None
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0
    if "last_error" not in st.session_state:
        st.session_state.last_error = None


_init_session()


def _persist_decision(state: dict, key: str, status: str, feedback: str = "") -> None:
    """Mirror a decision into the durable on-disk store, pulling whatever
    context is available in current state so the Approved/Rejected pages
    have enough to display without needing the original graph state."""
    company_by_key = {c["key"]: c for c in state.get("validated_companies", [])}
    company = company_by_key.get(key, {})
    qual = state.get("qualification", {}).get(key, {})
    rec = state.get("recommendations", {}).get(key, {})
    contacts = state.get("contact_enrichment", {}).get(key) or state.get("decision_makers", {}).get(key) or []

    persistence.save_decision(
        company_key=key,
        status=status,
        company_name=company.get("name", key),
        website=company.get("website"),
        score=qual.get("score"),
        tier=qual.get("tier"),
        reasoning=qual.get("reasoning"),
        recommended_contact=rec.get("recommended_contact"),
        recommended_outreach=rec.get("recommended_outreach"),
        urgency=rec.get("urgency"),
        contacts=contacts,
        feedback=feedback or None,
    )


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #e2e6f0; letter-spacing: -0.3px;">
            🎯 ICP Configuration
        </div>
        <div style="font-size: 0.75rem; color: #5e6680; margin-top: 4px; line-height: 1.4;">
            Edit here, or in config/icp_config.yaml — sidebar values override the file for this session.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    default_cfg = RunConfig.from_yaml(str(DEFAULT_CONFIG_PATH))
except Exception as e:
    st.sidebar.error(f"Could not load config/icp_config.yaml: {e}")
    default_cfg = None

industry = st.sidebar.text_input("Industry", value=default_cfg.industry if default_cfg else "Technology")

employee_size_min = st.sidebar.number_input(
    "Minimum employee size (ICP)",
    min_value=1,
    value=default_cfg.icp.employee_size_min if default_cfg else 100,
    step=10,
)

employee_size_max = st.sidebar.number_input(
    "Maximum employee size (ICP) — 0 means no cap",
    min_value=0,
    value=default_cfg.icp.employee_size_max if (default_cfg and default_cfg.icp.employee_size_max) else 0,
    step=500,
    help="Set this to exclude mega-caps (Amazon, Google, etc.) that run their own in-house TA orgs and rarely buy staffing services.",
)

locations_str = st.sidebar.text_input(
    "Locations (comma-separated)",
    value=", ".join(default_cfg.icp.locations) if default_cfg else "United States",
)

hiring_focus_str = st.sidebar.text_area(
    "Hiring focus roles (comma-separated)",
    value=", ".join(default_cfg.icp.hiring_focus) if default_cfg else "Software Engineers, Backend Engineers, AI Engineers, Data Scientists",
    height=80,
)

hiring_threshold = st.sidebar.number_input(
    "Hiring threshold (min. estimated open roles)",
    min_value=0,
    value=default_cfg.hiring_threshold if default_cfg else 20,
    step=5,
)

target_personas_str = st.sidebar.text_area(
    "Target personas, in priority order (comma-separated)",
    value=", ".join(default_cfg.target_personas) if default_cfg else "Head of Talent Acquisition, VP Engineering, HR Director, CTO",
    height=80,
)

max_companies = st.sidebar.slider("Max companies per run", min_value=3, max_value=50, value=default_cfg.max_companies if default_cfg else 15)

search_depth = st.sidebar.selectbox(
    "Search depth",
    options=["basic", "advanced"],
    index=0 if (not default_cfg or default_cfg.search_depth == "basic") else 1,
    help="'advanced' costs more Tavily credits but searches more thoroughly.",
)

st.sidebar.divider()

tavily_key_present = bool(os.environ.get("TAVILY_API_KEY"))
groq_key_present = bool(os.environ.get("GROQ_API_KEY"))

st.sidebar.markdown(
    f"""
    <div style="margin-bottom: 0.75rem;">
        <div style="font-size: 0.8rem; font-weight: 600; color: #7b82a0; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-bottom: 0.5rem;">API Keys (.env)</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 0.85rem;">{'✅' if tavily_key_present else '❌'}</span>
            <span style="font-size: 0.82rem; color: {'#86efac' if tavily_key_present else '#fca5a5'};">
                Tavily {'connected' if tavily_key_present else '— missing, see .env.example'}
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 0.85rem;">{'✅' if groq_key_present else '❌'}</span>
            <span style="font-size: 0.82rem; color: {'#86efac' if groq_key_present else '#fca5a5'};">
                Groq {'connected' if groq_key_present else '— missing, see .env.example'}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()
run_clicked = st.sidebar.button("▶️ Run Discovery", type="primary", width='stretch')
reset_clicked = st.sidebar.button("🔄 Reset Run", width='stretch')


def _build_run_config() -> dict:
    return RunConfig.from_dict(
        {
            "industry": industry,
            "icp": {
                "employee_size_min": int(employee_size_min),
                "employee_size_max": int(employee_size_max) if employee_size_max else None,
                "locations": [s.strip() for s in locations_str.split(",") if s.strip()],
                "hiring_focus": [s.strip() for s in hiring_focus_str.split(",") if s.strip()],
            },
            "hiring_threshold": int(hiring_threshold),
            "target_personas": [s.strip() for s in target_personas_str.split(",") if s.strip()],
            "max_companies": int(max_companies),
            "search_depth": search_depth,
        }
    ).model_dump()


if reset_clicked:
    st.session_state.platform_state = None
    st.session_state.run_count = 0
    st.session_state.last_error = None
    st.rerun()

if run_clicked:
    if not (tavily_key_present and groq_key_present):
        st.session_state.last_error = (
            "Missing API key(s) — add TAVILY_API_KEY and GROQ_API_KEY to your .env file before running."
        )
    else:
        try:
            cfg_dict = _build_run_config()
        except Exception as e:
            st.session_state.last_error = f"Invalid configuration: {e}"
            cfg_dict = None

        if cfg_dict:
            if st.session_state.platform_state is None:
                st.session_state.platform_state = empty_state(cfg_dict)
            else:
                # allow sidebar edits mid-run (e.g. raising the threshold)
                # without throwing away discovered companies/memory
                st.session_state.platform_state["config"] = cfg_dict

            with st.spinner("Planner is orchestrating agents — this can take a minute or two..."):
                try:
                    st.session_state.platform_state = run_graph(
                        st.session_state.graph, st.session_state.platform_state
                    )
                    st.session_state.run_count += 1
                    st.session_state.last_error = None
                except Exception as e:
                    st.session_state.last_error = f"Run failed: {e}"


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="padding: 1.2rem 0 0.5rem 0;">
        <h1 style="margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">
            Staffing Prospect Intelligence
        </h1>
        <p style="margin: 6px 0 0 0; color: #5e6680; font-size: 0.875rem;">
            Planner-driven multi-agent discovery of companies actively hiring — staffing sales prospects, not job listings.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

state = st.session_state.platform_state

if state is None:
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    st.info("⚙️ Configure your ICP in the sidebar and click **▶️ Run Discovery** to start.")
    st.stop()

# ---- Execution status metrics ----
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
status_col1.metric("Companies discovered", len(state.get("discovered_companies", [])))
status_col2.metric(
    "Above hiring threshold",
    len([k for k, v in state.get("hiring_intel", {}).items()
         if v.get("estimated_total_openings", 0) >= state["config"].get("hiring_threshold", 0)])
)
status_col3.metric("Qualified prospects", len(state.get("qualification", {})))
status_col4.metric("Run cycles", st.session_state.run_count)

st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

if state.get("done"):
    st.success("✓ Planner reports no remaining work for this configuration.")
else:
    st.warning("⟳ Run not yet complete — click **Run Discovery** again to continue (e.g. if a company needs re-scoring).")

with st.expander("🧭 Planner Decisions & Execution Trace", expanded=False):
    for entry in state.get("plan_log", [])[-40:]:
        st.text(f"[{entry['timestamp']}] {entry['agent']}: {entry['summary']}")

if state.get("errors"):
    with st.expander(f"⚠️ Errors ({len(state['errors'])})", expanded=False):
        for err in state["errors"][-20:]:
            company_part = f" (company: {err['company']})" if err.get("company") else ""
            st.text(f"[{err['agent']}]{company_part}: {err['error']}")

st.divider()

# ---- Company cards ----
st.markdown(
    "<h2 style='font-size: 1.2rem; font-weight: 600; color: #c4cad9; margin-bottom: 0.8rem;'>Company Prospects</h2>",
    unsafe_allow_html=True,
)

company_by_key = {c["key"]: c for c in state.get("validated_companies", [])}
intel = state.get("hiring_intel", {})
enrichment = state.get("company_enrichment", {})
qualification = state.get("qualification", {})
recommendations = state.get("recommendations", {})
decision_makers = state.get("decision_makers", {})
contact_enrichment = state.get("contact_enrichment", {})
approval_status = state.get("approval_status", {})

TIER_COLOR = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
TIER_BADGE = {
    "High":   ("#166534", "#dcfce7", "#22c55e"),  # bg, text, border
    "Medium": ("#78350f", "#fef3c7", "#f59e0b"),
    "Low":    ("#7f1d1d", "#fee2e2", "#ef4444"),
}

# Sort: qualified+scored companies first (by score desc), then everything else
scored_keys = sorted(qualification.keys(), key=lambda k: -qualification[k].get("score", 0))
other_keys = [k for k in company_by_key if k not in qualification]
ordered_keys = scored_keys + other_keys

if not ordered_keys:
    st.info("No companies processed yet this run.")

for key in ordered_keys:
    company = company_by_key.get(key, {"name": key})
    hiring = intel.get(key, {})
    enr = enrichment.get(key, {})
    qual = qualification.get(key)
    rec = recommendations.get(key)
    contacts = contact_enrichment.get(key) or decision_makers.get(key) or []

    tier = qual.get("tier") if qual else None
    tier_icon = TIER_COLOR.get(tier, "⚪")
    score = qual.get("score") if qual else None

    # Build score/tier badge HTML
    if tier and tier in TIER_BADGE:
        bg, txt_col, border = TIER_BADGE[tier]
        badge_html = (
            f'<span style="background:{bg}; color:{txt_col}; border:1px solid {border}; '
            f'border-radius:5px; padding:2px 9px; font-size:0.76rem; font-weight:600; '
            f'margin-left:10px;">{tier}</span>'
        )
    else:
        badge_html = ""

    score_html = (
        f'<span style="color:#94a3b8; font-size:0.85rem; margin-left:8px;">Score </span>'
        f'<span style="color:#e2e8f0; font-weight:700; font-size:0.95rem;">{score}</span>'
        f'<span style="color:#64748b; font-size:0.85rem;">/100</span>'
        if score is not None else ""
    )

    with st.container(border=True):
        # Card header
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem;">
                <span style="font-size:1.15rem;">{tier_icon}</span>
                <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">
                    {company.get('name', key)}
                </span>
                {score_html}
                {badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Meta row
        meta_cols = st.columns(4)
        meta_cols[0].markdown(f"**Website**  \n{company.get('website', 'Unknown')}")
        meta_cols[1].markdown(f"**HQ**  \n{enr.get('headquarters') or company.get('headquarters', 'Unknown')}")
        meta_cols[2].markdown(f"**Employees**  \n{enr.get('employee_count') or company.get('employee_estimate', 'Unknown')}")
        meta_cols[3].markdown(
            f"**Industry**  \n"
            f"{(enr.get('description', '') and ' '.join(enr['description'].split()[:6]) + '...') or company.get('industry', 'Unknown')}"
        )

        if hiring:
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            hire_cols = st.columns(5)
            hire_cols[0].metric("Est. Total Openings", hiring.get("estimated_total_openings", 0))
            hire_cols[1].metric("Engineering", hiring.get("engineering_jobs", 0))
            hire_cols[2].metric("AI", hiring.get("ai_jobs", 0))
            hire_cols[3].metric("Backend", hiring.get("backend_jobs", 0))
            hire_cols[4].metric("Data", hiring.get("data_jobs", 0))
            st.caption(
                f"Trend: {hiring.get('hiring_trend', 'Unknown')} · "
                f"Basis: {hiring.get('total_openings_basis', 'estimated')} · "
                f"Source quality: {hiring.get('source_quality', 'unknown')} · "
                f"Confidence: {hiring.get('confidence', 'Unknown')}"
            )
            if hiring.get("notes"):
                st.caption(f"📝 {hiring['notes']}")

        if enr.get("growth_signals") or enr.get("funding_stage"):
            growth_bits = []
            if enr.get("funding_stage"):
                growth_bits.append(f"Funding: {enr['funding_stage']}")
            if enr.get("global_offices"):
                growth_bits.append(f"Offices: {', '.join(enr['global_offices'])}")
            if enr.get("growth_signals"):
                growth_bits.append("; ".join(enr["growth_signals"]))
            st.markdown(
                f'<div style="margin:0.4rem 0; font-size:0.85rem; color:#86efac;">'
                f'📈 Growth signals: {" · ".join(growth_bits)}</div>',
                unsafe_allow_html=True,
            )

        if qual:
            st.markdown(
                f'<div style="margin:0.4rem 0; font-size:0.85rem; color:#cbd5e1; '
                f'background:#0f172a; border-left:3px solid #3b82f6; padding:0.5rem 0.75rem; '
                f'border-radius:0 5px 5px 0;">'
                f'<strong style="color:#93c5fd;">Qualification:</strong> {qual["reasoning"]}</div>',
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
                email = c.get("email", "Unknown")
                email_tag = ""
                if c.get("email_is_pattern_guess"):
                    email_tag = " *(inferred pattern, unverified)*"
                elif c.get("email_is_verified"):
                    email_tag = " *(verified ✓)*"
                linkedin = c.get("linkedin_url", "")
                linkedin_part = f' · <a href="{linkedin}" style="color:#60a5fa;">LinkedIn</a>' if linkedin else ""
                st.markdown(
                    f'<div style="font-size:0.84rem; padding:3px 0; color:#c9cdd7;">'
                    f'<strong style="color:#e2e8f0;">{c.get("name", "Unknown")}</strong>'
                    f' — <span style="color:#94a3b8;">{c.get("role") or c.get("matched_persona", "Unknown role")}</span>'
                    f' · <span style="color:#60a5fa;">{email}</span><span style="color:#64748b;">{email_tag}</span>'
                    f'{linkedin_part}</div>',
                    unsafe_allow_html=True,
                )

        if rec:
            priority_colors = {"High": "#f59e0b", "Medium": "#60a5fa", "Low": "#94a3b8"}
            p_color = priority_colors.get(rec.get("priority"), "#94a3b8")
            st.markdown(
                f'<div style="margin:0.5rem 0; font-size:0.85rem; background:#0f1f3d; '
                f'border-left:3px solid {p_color}; padding:0.5rem 0.75rem; border-radius:0 5px 5px 0;">'
                f'<strong style="color:{p_color};">{rec["priority"]} priority</strong>'
                f' — {rec["recommended_outreach"]}'
                f' <span style="color:#64748b; font-style:italic;">'
                f'(Contact: {rec["recommended_contact"]}, {rec["urgency"]})</span></div>',
                unsafe_allow_html=True,
            )

            status = approval_status.get(key, "pending")

            # Status badge
            status_colors = {
                "approved": ("#166534", "#22c55e"),
                "rejected": ("#7f1d1d", "#ef4444"),
                "pending":  ("#1e3a5f", "#60a5fa"),
                "edited":   ("#1a1a1a", "#a3a3a3"),
            }
            s_bg, s_fg = status_colors.get(status, ("#1e3a5f", "#60a5fa"))
            st.markdown(
                f'<div style="margin-bottom:0.5rem;">'
                f'<span style="background:{s_bg}; color:{s_fg}; border:1px solid {s_fg}40; '
                f'border-radius:5px; padding:2px 9px; font-size:0.75rem; font-weight:600;">'
                f'{status.upper()}</span>'
                + (f' <span style="color:#64748b; font-size:0.8rem;">— {state["user_feedback"][key]}</span>'
                   if key in state.get("user_feedback", {}) else "")
                + "</div>",
                unsafe_allow_html=True,
            )

            btn_cols = st.columns(4)
            if btn_cols[0].button("✅ Approve", key=f"approve_{key}"):
                update = apply_user_decision(state, key, "approved")
                state.update(update)
                _persist_decision(state, key, "approved")
                st.rerun()
            if btn_cols[1].button("❌ Reject", key=f"reject_{key}"):
                update = apply_user_decision(state, key, "rejected")
                state.update(update)
                _persist_decision(state, key, "rejected")
                st.rerun()
            if btn_cols[2].button("✏️ Edit", key=f"edit_{key}"):
                st.session_state[f"editing_{key}"] = True
            if btn_cols[3].button("🔁 Re-score", key=f"rescore_{key}"):
                update = apply_user_decision(state, key, "rescore")
                state.update(update)
                # the company's qualification/recommendation are being
                # cleared so it can be re-scored — any prior on-disk
                # decision for it is now stale, so remove it rather than
                # leave an outdated approve/reject sitting in the
                # Approved/Rejected pages until the next decision is made.
                persistence.remove_decision(key)
                st.rerun()

            if st.session_state.get(f"editing_{key}"):
                feedback = st.text_area("Edit notes / feedback", key=f"feedback_{key}")
                if st.button("Save edit", key=f"save_edit_{key}"):
                    update = apply_user_decision(state, key, "edited", feedback)
                    state.update(update)
                    _persist_decision(state, key, "edited", feedback)
                    st.session_state[f"editing_{key}"] = False
                    st.rerun()

st.divider()

# ---- Final approval table + export ----
st.markdown(
    "<h2 style='font-size: 1.2rem; font-weight: 600; color: #c4cad9; margin-bottom: 0.5rem;'>Approval Summary & Export</h2>",
    unsafe_allow_html=True,
)

rows = []
for key in scored_keys:
    company = company_by_key.get(key, {})
    qual = qualification.get(key, {})
    rec = recommendations.get(key, {})
    hiring = intel.get(key, {})
    contacts = contact_enrichment.get(key) or decision_makers.get(key) or []
    primary_contact = contacts[0] if contacts else {}
    rows.append(
        {
            "Company": company.get("name", key),
            "Website": company.get("website", ""),
            "Score": qual.get("score"),
            "Tier": qual.get("tier"),
            "Est. Openings": hiring.get("estimated_total_openings"),
            "Trend": hiring.get("hiring_trend"),
            "Recommended Contact": rec.get("recommended_contact"),
            "Contact Name": primary_contact.get("name", "Unknown"),
            "Contact Email": primary_contact.get("email", "Unknown"),
            "Priority": rec.get("priority"),
            "Next Action": rec.get("recommended_outreach"),
            "Urgency": rec.get("urgency"),
            "Approval Status": approval_status.get(key, "pending"),
            "Reasoning": qual.get("reasoning"),
        }
    )

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="staffing_prospects.csv",
        mime="text/csv",
        width='stretch',
    )
else:
    st.caption("No qualified prospects yet — run discovery to populate this table.")