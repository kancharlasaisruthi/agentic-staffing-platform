"""
Shared-memory helper functions.

These are the functions agents call instead of touching state dicts
directly, so the "avoid duplicate searches / avoid processing the same
company twice" rule from the spec is enforced in one place rather than
re-implemented (inconsistently) inside every agent.
"""

from __future__ import annotations

from typing import Optional

from core.state import State, company_key


def has_been_discovered(state: State, name: str, website: Optional[str] = None) -> bool:
    return company_key(name, website) in state.get("seen_companies", {})


def has_been_enriched(state: State, key: str) -> bool:
    return key in state.get("enriched_companies", {})


def mark_discovered(state: State, key: str, record: dict) -> dict:
    """Returns the partial state update to merge — does not mutate in place,
    since LangGraph nodes should return updates rather than mutate state,
    to keep the graph's data flow explicit and debuggable."""
    seen = dict(state.get("seen_companies", {}))
    seen[key] = record
    return {"seen_companies": seen}


def mark_enriched(state: State, key: str, record: dict) -> dict:
    enriched = dict(state.get("enriched_companies", {}))
    enriched[key] = record
    return {"enriched_companies": enriched}


def record_agent_output(state: State, key: str, agent_name: str, output: dict) -> dict:
    """Append/overwrite a single agent's output for a single company in the
    company_outputs ledger, without clobbering other agents' entries for
    that same company."""
    outputs = {k: dict(v) for k, v in state.get("company_outputs", {}).items()}
    outputs.setdefault(key, {})
    outputs[key][agent_name] = output
    return {"company_outputs": outputs}


def get_cached_search(state: State, query: str) -> Optional[list]:
    return state.get("query_cache", {}).get(query.strip().lower())


def cache_search(state: State, query: str, results: list) -> dict:
    cache = dict(state.get("query_cache", {}))
    cache[query.strip().lower()] = results
    return {"query_cache": cache}


def log_error(state: State, agent: str, error: str, company: Optional[str] = None) -> dict:
    errors = list(state.get("errors", []))
    errors.append({"agent": agent, "company": company, "error": error})
    return {"errors": errors}


def append_plan_log(state: State, agent: str, summary: str) -> dict:
    import datetime

    log = list(state.get("plan_log", []))
    log.append(
        {
            "agent": agent,
            "summary": summary,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
    )
    return {"plan_log": log}


def mark_agent_run(state: State, agent: str) -> dict:
    run = list(state.get("agents_run", []))
    if agent not in run:
        run.append(agent)
    return {"agents_run": run}
