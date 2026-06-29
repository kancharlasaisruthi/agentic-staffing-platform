"""
Builds and compiles the LangGraph StateGraph.

Every agent node returns control to the planner (the only node with
conditional routing logic), and the planner decides the next node via
`Command(goto=...)`. This is the structural piece that makes "Adding a
new agent" (see README) a one-line change here plus a routing-table entry
in agents/planner.py — no other graph surgery required.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.company_discovery import company_discovery_agent
from agents.company_enrichment import company_enrichment_agent
from agents.company_validation import company_validation_agent
from agents.contact_enrichment import contact_enrichment_agent
from agents.decision_maker import decision_maker_agent
from agents.hiring_intelligence import hiring_intelligence_agent
from agents.human_approval import human_approval_agent
from agents.market_trigger import market_trigger_agent
from agents.planner import planner
from agents.qualification import qualification_agent
from agents.recommendation import recommendation_agent
from core.state import State

# (node_name, node_fn) — node_name MUST match the AgentName literal and the
# routing-table agent_name strings in agents/planner.py exactly.
AGENT_NODES = [
    ("market_trigger", market_trigger_agent),
    ("company_discovery", company_discovery_agent),
    ("company_validation", company_validation_agent),
    ("hiring_intelligence", hiring_intelligence_agent),
    ("company_enrichment", company_enrichment_agent),
    ("decision_maker", decision_maker_agent),
    ("contact_enrichment", contact_enrichment_agent),
    ("qualification", qualification_agent),
    ("recommendation", recommendation_agent),
    ("human_approval", human_approval_agent),
]


def build_graph():
    graph = StateGraph(State)

    graph.add_node("planner", planner)
    for name, fn in AGENT_NODES:
        graph.add_node(name, fn)
        # Every agent always returns to the planner — the planner is the
        # sole router. This is what makes it planner-driven rather than a
        # sequential chain: there is exactly one decision point, and it
        # re-evaluates full state after every single agent call.
        graph.add_edge(name, "planner")

    graph.add_edge(START, "planner")
    # planner's own routing to "__end__" or any agent name is handled via
    # the Command(goto=...) return value inside agents/planner.py, not via
    # add_conditional_edges — LangGraph 1.x supports returning Command
    # directly from a node for dynamic routing.

    return graph.compile()


# Each planner<->agent cycle costs 2 graph steps (planner, then the chosen
# agent). agents/planner.py's own MAX_PLANNER_INVOCATIONS backstop (80) is
# the circuit breaker we actually want to hit in pathological cases — it
# ends gracefully with partial results. LangGraph's recursion_limit must
# therefore be set comfortably ABOVE 2 * MAX_PLANNER_INVOCATIONS, or
# LangGraph's own limit fires first and raises GraphRecursionError,
# discarding all partial state instead of returning it. This was caught
# directly in testing — see the planner module's failure-handling tests.
RECOMMENDED_RECURSION_LIMIT = 250


def run_graph(compiled_graph, initial_state: dict) -> dict:
    """Thin convenience wrapper so every caller (CLI, UI, tests) gets the
    safe recursion_limit by default instead of risking LangGraph's
    much-lower built-in default of 25."""
    return compiled_graph.invoke(initial_state, config={"recursion_limit": RECOMMENDED_RECURSION_LIMIT})
