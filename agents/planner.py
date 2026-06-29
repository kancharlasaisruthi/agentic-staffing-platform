"""
Planner Agent.

Per the spec, the Planner:
- reads the user configuration
- determines which agents should execute
- skips unnecessary agents (e.g. if discovery already produced 0
  companies above threshold, don't bother running Decision Maker)
- reuses previous memory whenever possible (never re-discovers a company
  already in seen_companies for this run)
- never performs enrichment itself — it only routes
- outputs an execution plan (the plan_log) as it goes

Implementation choice: rather than a single fixed linear chain, the
Planner is a small rule engine that inspects state after every agent
completes and decides the next node by checking, in priority order,
"is there a company missing X piece of information that a not-yet-run
agent can supply?". This is what makes the graph genuinely planner-driven
rather than a relabeled sequential pipeline — e.g. if Market Trigger
finds zero usable companies, the Planner routes straight to Company
Discovery instead of wasting a Validation pass on nothing, and if a
company already has hiring_intel from a previous run (memory reuse), the
Planner skips Hiring Intelligence for that company specifically.

The routing table below encodes the spec's "Planner Logic" section
directly: each entry is (need, agent, state_key_that_satisfies_need).
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.types import Command

from agents.base import agent_node
from core.memory import append_plan_log
from core.state import State

logger = logging.getLogger("platform.planner")

# Circuit breaker: if an agent has failed this many times for the SAME
# unmet need without making progress, the planner stops retrying it and
# moves on (or ends the run) rather than looping until the graph's
# recursion limit kills the whole run with no output at all. This was
# added after a real bug was caught in testing: an agent that raises on
# every call (e.g. missing API key) previously caused an infinite
# planner<->agent loop, because the routing "need" check only looked at
# whether the desired data existed — not whether we'd already tried and
# failed to produce it.
MAX_CONSECUTIVE_FAILURES_PER_AGENT = 3

# Ordered per the spec's architecture diagram. The Planner walks this list
# top to bottom each cycle and routes to the first agent whose "is_needed"
# check returns True. This preserves a sensible default order while still
# being driven by actual state rather than a hardcoded sequence — any step
# can be skipped, and the Planner re-evaluates from the top after every
# agent run rather than blindly advancing an index.
AgentName = Literal[
    "market_trigger",
    "company_discovery",
    "company_validation",
    "hiring_intelligence",
    "company_enrichment",
    "decision_maker",
    "contact_enrichment",
    "qualification",
    "recommendation",
    "human_approval",
    "__end__",
]


def _needs_market_trigger(state: State) -> bool:
    # Only run once per session — if we already have triggers OR the user
    # configuration signals they already know which companies to target
    # (not modeled yet, always run once for now), skip on repeat.
    return "market_trigger" not in state.get("agents_run", [])


def _needs_company_discovery(state: State) -> bool:
    return len(state.get("discovered_companies", [])) == 0


def _needs_company_validation(state: State) -> bool:
    discovered = state.get("discovered_companies", [])
    validated_keys = {c["key"] for c in state.get("validated_companies", [])}
    return any(c["key"] not in validated_keys for c in discovered)


def _qualifying_companies(state: State) -> list[dict]:
    """Companies that passed validation — these are the only ones later
    agents should spend budget on."""
    return [c for c in state.get("validated_companies", []) if c.get("is_valid", False)]


def _needs_hiring_intelligence(state: State) -> bool:
    intel = state.get("hiring_intel", {})
    return any(c["key"] not in intel for c in _qualifying_companies(state))


def _above_hiring_threshold(state: State) -> list[dict]:
    """Companies whose estimated hiring meets the configured threshold —
    only these proceed to enrichment/decision-maker/qualification, per the
    spec's intent that we're sales-prospecting, not cataloguing everyone."""
    threshold = state.get("config", {}).get("hiring_threshold", 0)
    intel = state.get("hiring_intel", {})
    out = []
    for c in _qualifying_companies(state):
        rec = intel.get(c["key"])
        if rec and rec.get("estimated_total_openings", 0) >= threshold:
            out.append(c)
    return out


def _needs_company_enrichment(state: State) -> bool:
    enrichment = state.get("company_enrichment", {})
    return any(c["key"] not in enrichment for c in _above_hiring_threshold(state))


def _needs_decision_maker(state: State) -> bool:
    dm = state.get("decision_makers", {})
    return any(c["key"] not in dm for c in _above_hiring_threshold(state))


def _needs_contact_enrichment(state: State) -> bool:
    ce = state.get("contact_enrichment", {})
    dm = state.get("decision_makers", {})
    for c in _above_hiring_threshold(state):
        if dm.get(c["key"]) and c["key"] not in ce:
            return True
    return False


def _needs_qualification(state: State) -> bool:
    qual = state.get("qualification", {})
    return any(c["key"] not in qual for c in _above_hiring_threshold(state))


def _needs_recommendation(state: State) -> bool:
    rec = state.get("recommendations", {})
    qual = state.get("qualification", {})
    # qual is keyed by company_key already (string -> dict), so we just
    # need keys present in qualification but missing from recommendations.
    return any(key not in rec for key in qual.keys())


def _needs_human_approval(state: State) -> bool:
    recs = state.get("recommendations", {})
    approvals = state.get("approval_status", {})
    if not recs:
        return False
    return any(k not in approvals for k in recs)


def _consecutive_failures(state: State, agent_name: str) -> int:
    """
    Total failures recorded for this specific agent so far in the run.

    Originally implemented as "trailing run length from the end of the
    error log" — that broke as soon as OTHER agents were also failing
    and interleaved their errors in between this agent's attempts, which
    is exactly the realistic failure mode (e.g. every agent fails because
    the API key is missing). Counting total occurrences for this agent
    name, regardless of position, is the correct and simpler check: we
    don't actually care whether the failures were contiguous, only that
    this specific agent has failed at least N times and should stop being
    retried.
    """
    return sum(1 for entry in state.get("errors", []) if entry.get("agent") == agent_name)


# Backstop circuit breaker independent of per-agent counting: if the
# planner itself has been invoked this many times in one run, something
# is wrong (e.g. every single agent is failing) and we should end the run
# with whatever partial state exists rather than approach LangGraph's
# recursion_limit, which raises and discards all partial output instead
# of returning it.
MAX_PLANNER_INVOCATIONS = 80


def _planner_invocation_count(state: State) -> int:
    return sum(1 for entry in state.get("plan_log", []) if entry.get("agent") == "planner")


# Ordered (need_check, agent_name) pairs — this IS the spec's "Planner
# Logic" section, encoded directly.
ROUTING_TABLE: list[tuple] = [
    (_needs_market_trigger, "market_trigger"),
    (_needs_company_discovery, "company_discovery"),
    (_needs_company_validation, "company_validation"),
    (_needs_hiring_intelligence, "hiring_intelligence"),
    (_needs_company_enrichment, "company_enrichment"),
    (_needs_decision_maker, "decision_maker"),
    (_needs_contact_enrichment, "contact_enrichment"),
    (_needs_qualification, "qualification"),
    (_needs_recommendation, "recommendation"),
    (_needs_human_approval, "human_approval"),
]


def planner(state: State) -> Command[AgentName]:
    """
    The single decision point in the graph. Every agent (including
    human_approval) routes back here when it finishes, and the planner
    decides what's next based on current state — never a hardcoded
    successor. This is what makes the graph planner-driven rather than a
    relabeled linear chain: removing or reordering an agent in
    ROUTING_TABLE changes behavior with no graph edits.

    Includes a circuit breaker: an agent that has failed
    MAX_CONSECUTIVE_FAILURES_PER_AGENT times in total is skipped even if
    its "need" check still returns True, so a persistently broken agent
    (bad API key, network down, etc.) ends the run gracefully with
    whatever partial output exists instead of looping until the graph's
    recursion limit aborts with nothing usable at all. A second, global
    backstop (MAX_PLANNER_INVOCATIONS) catches the pathological case where
    many different agents are all failing for the same root cause (e.g.
    no API keys configured at all) — each individually under its own
    failure threshold, but collectively keeping the planner busy forever.
    """
    if _planner_invocation_count(state) >= MAX_PLANNER_INVOCATIONS:
        summary = (
            f"Planner: hit global safety limit of {MAX_PLANNER_INVOCATIONS} planning "
            "cycles for this run — ending now with partial results rather than risking "
            "a recursion-limit abort that would discard all output"
        )
        logger.warning(summary)
        log_update = append_plan_log(state, "planner", summary)
        return Command(goto="__end__", update={"next_agent": None, "done": True, **log_update})

    for needs_fn, agent_name in ROUTING_TABLE:
        try:
            if not needs_fn(state):
                continue
            if _consecutive_failures(state, agent_name) >= MAX_CONSECUTIVE_FAILURES_PER_AGENT:
                summary = (
                    f"Planner: {agent_name} has failed "
                    f"{MAX_CONSECUTIVE_FAILURES_PER_AGENT}+ times in a row — "
                    "skipping it for the rest of this run rather than retrying indefinitely"
                )
                logger.warning(summary)
                log_update = append_plan_log(state, "planner", summary)
                # fall through to the NEXT routing-table entry instead of
                # returning, so other needed agents still get a chance to run
                state = {**state, **log_update}
                continue
            summary = f"Planner: routing to {agent_name} ({needs_fn.__name__})"
            logger.info(summary)
            log_update = append_plan_log(state, "planner", summary)
            return Command(goto=agent_name, update={"next_agent": agent_name, **log_update})
        except Exception:  # noqa: BLE001
            logger.exception("Planner routing check %s raised — skipping", needs_fn.__name__)
            continue

    summary = "Planner: no remaining work — all needs satisfied (or stuck agents skipped), routing to final report"
    logger.info(summary)
    log_update = append_plan_log(state, "planner", summary)
    return Command(goto="__end__", update={"next_agent": None, "done": True, **log_update})
