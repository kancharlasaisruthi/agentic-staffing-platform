"""
Shared state threaded through the LangGraph graph.

Every agent reads from and writes to this single structure. LangGraph
merges partial dict returns from each node into this state automatically
(standard reducer behavior: dict keys get overwritten, list keys we manage
manually via explicit merge helpers in core/memory.py so we don't lose
history when multiple agents append).

Design note: we deliberately keep this as a flat TypedDict rather than
nested Pydantic models inside state, because LangGraph's state merging
works on the top-level dict keys. Nested objects (CompanyRecord etc.) are
still pydantic-validated — see core/models.py — they're just stored as
plain dicts inside this TypedDict's lists for serialization simplicity
(important for Streamlit caching and eventual checkpointing).
"""

from __future__ import annotations

import re
from typing import Any, Optional, TypedDict


class AgentLogEntry(TypedDict):
    agent: str
    summary: str
    timestamp: str


class State(TypedDict, total=False):
    # ---- run configuration (set once, read-only after init) ----
    config: dict  # validated RunConfig.model_dump()

    # ---- planner bookkeeping ----
    plan_log: list[AgentLogEntry]          # human-readable trace of planner decisions
    agents_run: list[str]                  # which agents have executed at least once
    next_agent: Optional[str]               # set by planner, consumed by router
    done: bool                               # planner sets True when ready for final report

    # ---- shared memory (dedup across agents) ----
    seen_companies: dict[str, dict]          # company_key -> minimal record, populated by Discovery
    enriched_companies: dict[str, dict]        # company_key -> full enrichment, populated by Enrichment
    company_outputs: dict[str, dict]             # company_key -> {agent_name: result_dict} per-agent history
    query_cache: dict[str, list]                   # search query string -> raw Tavily results (dedup searches)

    # ---- pipeline data (each agent appends/updates) ----
    triggers: list[dict]                # Market Trigger Agent output
    discovered_companies: list[dict]      # Company Discovery Agent output
    validated_companies: list[dict]         # Company Validation Agent output
    hiring_intel: dict[str, dict]              # company_key -> hiring intelligence record
    company_enrichment: dict[str, dict]          # company_key -> firmographic record
    decision_makers: dict[str, list]               # company_key -> list of contact dicts
    contact_enrichment: dict[str, list]              # company_key -> enriched contact dicts
    qualification: dict[str, dict]                     # company_key -> {score, tier, reasoning}
    recommendations: dict[str, dict]                     # company_key -> {priority, next_action, ...}

    # ---- human-in-the-loop ----
    approval_status: dict[str, str]   # company_key -> "pending" | "approved" | "rejected" | "edited"
    user_feedback: dict[str, str]       # company_key -> free-text edit/rejection reason

    # ---- final output ----
    final_report: list[dict]   # fully assembled per-company report rows

    # ---- errors (never silently swallowed) ----
    errors: list[dict]   # {"agent": str, "company": Optional[str], "error": str}


def empty_state(config: dict) -> State:
    """Construct a fresh State for a new run."""
    return State(
        config=config,
        plan_log=[],
        agents_run=[],
        next_agent=None,
        done=False,
        seen_companies={},
        enriched_companies={},
        company_outputs={},
        query_cache={},
        triggers=[],
        discovered_companies=[],
        validated_companies=[],
        hiring_intel={},
        company_enrichment={},
        decision_makers={},
        contact_enrichment={},
        qualification={},
        recommendations={},
        approval_status={},
        user_feedback={},
        final_report=[],
        errors=[],
    )


def company_key(name: str, website: Optional[str] = None) -> str:
    """
    Stable dedup key for a company across agents.
    Prefer domain (strip protocol/www/path) since names collide
    ("Acme" vs "Acme Inc" vs "ACME") far more than domains do.
    """
    if website:
        domain = (
            website.lower()
            .replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .split("/")[0]
            .strip()
        )
        if domain:
            return domain
    return name.strip().lower().replace(",", "").replace(".", "")


def parse_employee_estimate(estimate: Optional[str]) -> Optional[int]:
    """
    Extract the first number from an employee-count string like '500-1000',
    '5000+', '1,200', or '50000', returning the LOWER bound for ranges.

    Shared helper used by both company_validation.py (rejecting companies
    over employee_size_max) and qualification.py (the company-size
    sub-score) — both originally had their own inline digit-extraction
    using "".join(ch for ch in s if ch.isdigit()), which is WRONG for
    range strings: stripping non-digit characters from "500-1000" before
    joining concatenates the digits into "5001000" instead of reading
    "500" as the first number. That bug was caught directly in testing
    (see tests/test_smoke.py) — every call site should use this function
    instead of re-implementing digit extraction locally.
    """
    if not estimate:
        return None
    match = re.search(r"\d[\d,]*", estimate)
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None
