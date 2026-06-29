"""
Human Approval Agent.

Per the spec: before finalizing recommendations, display companies +
score + reason + contacts + recommended action, and let the user
Approve / Reject / Edit / Re-score.

Implementation note: LangGraph has a native `interrupt()` primitive for
pausing a graph mid-run for human input, but that requires a checkpointer
and resume-by-thread-id, which fights Streamlit's rerun-on-every-
interaction execution model more than it helps here. Instead, this agent
implements the gate directly against state: it marks newly-recommended
companies "pending" in approval_status, and the Streamlit UI (ui/app.py)
writes the user's Approve/Reject/Edit/Re-score decision straight into
approval_status / user_feedback between graph invocations. This keeps the
UI in full control of pacing (one run can stop after Recommendation,
show the table, and only re-invoke the graph if the user clicks
"Re-score") while the Planner still treats "needs_human_approval" as a
real unmet need it routes to, per its routing table.

"Re-score" is handled by deleting the company's qualification +
recommendation entries so the Planner naturally re-routes it through
Qualification -> Recommendation -> Human Approval again on the next
invocation — no special-case graph logic needed.
"""

from __future__ import annotations

from agents.base import agent_node
from core.state import State


@agent_node("human_approval")
def human_approval_agent(state: State) -> dict:
    """
    On each pass, this agent's only job is to ensure every company with a
    recommendation has an approval_status entry (defaulting to "pending"
    so the UI knows to show it). It does NOT block execution itself —
    Streamlit's `ui/app.py` is responsible for stopping the run loop once
    there are pending approvals, and for re-invoking the graph after the
    user acts. This keeps the agent itself simple and testable without a
    UI attached.
    """
    approvals = dict(state.get("approval_status", {}))
    newly_pending = 0

    for key in state.get("recommendations", {}):
        if key not in approvals:
            approvals[key] = "pending"
            newly_pending += 1

    return {
        "approval_status": approvals,
        "_summary": f"Human Approval: {newly_pending} new companies awaiting approval "
        f"({sum(1 for v in approvals.values() if v == 'pending')} total pending)",
    }


def apply_user_decision(state: State, company_key: str, decision: str, feedback: str = "") -> dict:
    """
    Called directly by the Streamlit UI (not a graph node) when the user
    clicks Approve/Reject/Edit/Re-score for one company. Returns a state
    update dict the UI merges into its persisted session state before the
    next graph invocation.

    decision: one of "approved", "rejected", "edited", "rescore"
    """
    approvals = dict(state.get("approval_status", {}))
    feedback_map = dict(state.get("user_feedback", {}))

    if decision == "rescore":
        # Clear downstream results so the Planner re-routes this company
        # through Qualification -> Recommendation -> Human Approval again.
        qual = dict(state.get("qualification", {}))
        recs = dict(state.get("recommendations", {}))
        qual.pop(company_key, None)
        recs.pop(company_key, None)
        approvals.pop(company_key, None)
        if feedback:
            feedback_map[company_key] = feedback
        return {
            "qualification": qual,
            "recommendations": recs,
            "approval_status": approvals,
            "user_feedback": feedback_map,
        }

    approvals[company_key] = decision
    if feedback:
        feedback_map[company_key] = feedback
    return {"approval_status": approvals, "user_feedback": feedback_map}
