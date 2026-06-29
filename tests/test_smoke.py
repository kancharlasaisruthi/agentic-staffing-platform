"""
Smoke tests covering logic that does NOT require live API keys:
- config validation
- state/memory helpers
- planner routing logic (including the circuit breaker — this directly
  encodes the infinite-loop bug found during development, so it never
  regresses)
- qualification scoring rubric
- contact enrichment's no-hallucination helpers
- graph construction (compiles without error)

Agents that call SearchClient/LLMClient directly are NOT unit tested here
since that requires real TAVILY_API_KEY / GROQ_API_KEY — see
tests/test_live_smoke.py (skipped by default) for an opt-in live test
once you have keys configured.

Run with: pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agents.company_validation import _exceeds_employee_max, _is_known_megacap
from agents.contact_enrichment import _domain_from_website, _split_name
from agents.planner import (
    MAX_PLANNER_INVOCATIONS,
    ROUTING_TABLE,
    _consecutive_failures,
    _planner_invocation_count,
    planner,
)
from agents.qualification import _score_company, _tier_for_score
from config.schema import RunConfig
from core.graph import build_graph
from core.state import company_key, empty_state, parse_employee_estimate


# ---- config ----


def test_run_config_validates_minimal_spec_example():
    raw = {
        "industry": "Technology",
        "icp": {
            "employee_size_min": 100,
            "locations": ["United States"],
            "hiring_focus": ["Software Engineers", "AI Engineers"],
        },
        "hiring_threshold": 20,
        "target_personas": ["Head of Talent Acquisition", "VP Engineering"],
    }
    cfg = RunConfig.from_dict(raw)
    assert cfg.industry == "Technology"
    assert cfg.icp.employee_size_min == 100
    assert cfg.hiring_threshold == 20


def test_run_config_rejects_empty_hiring_focus():
    raw = {
        "industry": "Technology",
        "icp": {"employee_size_min": 100, "hiring_focus": []},
        "hiring_threshold": 20,
        "target_personas": ["CTO"],
    }
    with pytest.raises(Exception):
        RunConfig.from_dict(raw)


def test_run_config_rejects_empty_personas():
    raw = {
        "industry": "Technology",
        "icp": {"employee_size_min": 100, "hiring_focus": ["Engineers"]},
        "hiring_threshold": 20,
        "target_personas": [],
    }
    with pytest.raises(Exception):
        RunConfig.from_dict(raw)


# ---- state / company_key ----


def test_company_key_prefers_domain_over_name():
    assert company_key("Acme Inc", "https://www.acme.com/careers") == "acme.com"
    assert company_key("Acme Inc", None) == "acme inc"
    # Two name variants with the same domain must dedup to the same key
    assert company_key("Acme", "acme.com") == company_key("Acme, Inc.", "https://acme.com/jobs")


# ---- graph construction ----


def test_graph_compiles_with_all_agent_nodes():
    g = build_graph()
    nodes = set(g.get_graph().nodes.keys())
    expected = {name for _, name in ROUTING_TABLE} | {"planner", "__start__", "__end__"}
    assert expected.issubset(nodes)


# ---- planner routing & circuit breaker ----


def test_planner_routes_to_market_trigger_first_on_empty_state():
    config = {
        "industry": "Technology",
        "icp": {"employee_size_min": 100, "locations": ["US"], "hiring_focus": ["Engineers"]},
        "hiring_threshold": 20,
        "target_personas": ["CTO"],
    }
    state = empty_state(config)
    cmd = planner(state)
    assert cmd.goto == "market_trigger"


def test_planner_ends_when_everything_satisfied():
    config = {
        "industry": "Technology",
        "icp": {"employee_size_min": 100, "locations": ["US"], "hiring_focus": ["Engineers"]},
        "hiring_threshold": 20,
        "target_personas": ["CTO"],
    }
    state = empty_state(config)
    key = company_key("Acme", "acme.com")
    state["agents_run"] = [name for _, name in ROUTING_TABLE]
    state["discovered_companies"] = [{"key": key, "name": "Acme", "website": "acme.com"}]
    state["validated_companies"] = [{"key": key, "name": "Acme", "website": "acme.com", "is_valid": True}]
    state["hiring_intel"] = {key: {"estimated_total_openings": 50}}
    state["company_enrichment"] = {key: {}}
    state["decision_makers"] = {key: [{"name": "Jane Doe"}]}
    state["contact_enrichment"] = {key: [{"name": "Jane Doe"}]}
    state["qualification"] = {key: {"score": 80, "tier": "High", "reasoning": "x"}}
    state["recommendations"] = {key: {"priority": "High"}}
    state["approval_status"] = {key: "pending"}

    cmd = planner(state)
    assert cmd.goto == "__end__"
    assert cmd.update["done"] is True


def test_consecutive_failures_counts_per_agent_regardless_of_interleaving():
    """
    Regresion test for the real bug found during development: the
    original implementation only counted a TRAILING run of same-agent
    errors, which silently undercounted failures whenever other agents'
    errors were interleaved (e.g. every agent failing for the same root
    cause, like a missing API key). This must count total occurrences,
    not trailing-run length.
    """
    state = empty_state({})
    state["errors"] = [
        {"agent": "market_trigger", "error": "x"},
        {"agent": "company_discovery", "error": "y"},  # interleaved different agent
        {"agent": "market_trigger", "error": "x"},
        {"agent": "company_validation", "error": "z"},  # interleaved different agent
        {"agent": "market_trigger", "error": "x"},
    ]
    assert _consecutive_failures(state, "market_trigger") == 3


def test_planner_skips_agent_after_failure_ceiling():
    config = {
        "industry": "Technology",
        "icp": {"employee_size_min": 100, "locations": ["US"], "hiring_focus": ["Engineers"]},
        "hiring_threshold": 20,
        "target_personas": ["CTO"],
    }
    state = empty_state(config)
    state["errors"] = [{"agent": "market_trigger", "error": "boom"} for _ in range(3)]
    cmd = planner(state)
    # market_trigger has failed 3x (>= MAX_CONSECUTIVE_FAILURES_PER_AGENT),
    # so the planner must move PAST it to the next unmet need rather than
    # routing to it again.
    assert cmd.goto != "market_trigger"


def test_planner_global_backstop_ends_run_with_partial_results():
    """
    Regression test for the actual infinite-loop bug encountered during
    development: even with per-agent circuit breakers, many DIFFERENT
    agents each failing under their own threshold could keep the planner
    busy forever. The global invocation backstop must end the run.
    """
    config = {
        "industry": "Technology",
        "icp": {"employee_size_min": 100, "locations": ["US"], "hiring_focus": ["Engineers"]},
        "hiring_threshold": 20,
        "target_personas": ["CTO"],
    }
    state = empty_state(config)
    state["plan_log"] = [{"agent": "planner", "summary": "x", "timestamp": "x"} for _ in range(MAX_PLANNER_INVOCATIONS)]
    cmd = planner(state)
    assert cmd.goto == "__end__"
    assert cmd.update["done"] is True


# ---- qualification scoring rubric ----


def test_qualification_scores_strong_prospect_as_high_tier():
    hiring = {
        "estimated_total_openings": 84,
        "engineering_jobs": 60,
        "ai_jobs": 10,
        "backend_jobs": 20,
        "data_jobs": 5,
        "hiring_locations": ["San Francisco", "London", "Dublin"],
        "hiring_trend": "Increasing",
    }
    enrichment = {
        "funding_stage": "Series C",
        "funding_total": "$500M",
        "investors": ["Sequoia"],
        "global_offices": ["London", "Dublin"],
        "growth_signals": ["opened London office", "doubled engineering team"],
        "recent_news": ["raised Series C"],
    }
    company = {"employee_estimate": "5000+"}
    icp = {"employee_size_min": 100}

    result = _score_company(hiring, enrichment, company, icp, threshold=20)
    assert result["score"] >= 70
    assert _tier_for_score(result["score"]) == "High"


def test_qualification_scores_weak_prospect_as_low_tier():
    hiring = {
        "estimated_total_openings": 22,
        "engineering_jobs": 3,
        "ai_jobs": 0,
        "backend_jobs": 1,
        "data_jobs": 0,
        "hiring_locations": ["Remote"],
        "hiring_trend": "Unknown",
    }
    enrichment = {}
    company = {"employee_estimate": "50"}
    icp = {"employee_size_min": 100}

    result = _score_company(hiring, enrichment, company, icp, threshold=20)
    assert result["score"] < 45
    assert _tier_for_score(result["score"]) == "Low"


def test_qualification_score_never_exceeds_100():
    # Pathological input designed to max out every sub-score
    hiring = {
        "estimated_total_openings": 1000,
        "engineering_jobs": 1000,
        "ai_jobs": 1000,
        "backend_jobs": 1000,
        "data_jobs": 1000,
        "hiring_locations": ["A", "B", "C", "D", "E", "F"],
        "hiring_trend": "Increasing",
    }
    enrichment = {
        "funding_stage": "Series Z",
        "funding_total": "$10B",
        "investors": ["a", "b", "c"],
        "global_offices": ["A", "B", "C", "D", "E"],
        "growth_signals": ["a", "b", "c", "d"],
        "recent_news": ["a", "b", "c"],
    }
    company = {"employee_estimate": "100000"}
    icp = {"employee_size_min": 100}

    result = _score_company(hiring, enrichment, company, icp, threshold=20)
    assert result["score"] <= 100.0


# ---- contact enrichment no-hallucination helpers ----


def test_split_name_handles_single_name_safely():
    assert _split_name("Cher") is None


def test_split_name_extracts_first_and_last():
    assert _split_name("Jane Doe") == ("jane", "doe")
    assert _split_name("Mary Jane Watson") == ("mary", "watson")


def test_domain_from_website_strips_protocol_and_path():
    assert _domain_from_website("https://www.acme.com/careers") == "acme.com"
    assert _domain_from_website(None) is None
    assert _domain_from_website("acme.com") == "acme.com"


# ---- parse_employee_estimate (regression test for a real bug) ----


def test_parse_employee_estimate_takes_lower_bound_of_range():
    """
    Regression test for a real bug found during development: the
    original inline digit-extraction did
    "".join(ch for ch in s if ch.isdigit()) on a range string like
    '500-1000', which strips the dash and CONCATENATES the digits into
    '5001000' (five million, not five hundred) instead of reading '500'
    as the first number. parse_employee_estimate must return the lower
    bound of a range, not a mashed-together mega-number.
    """
    assert parse_employee_estimate("500-1000") == 500
    assert parse_employee_estimate("5000+") == 5000
    assert parse_employee_estimate("1,200") == 1200
    assert parse_employee_estimate("1,000-5,000") == 1000
    assert parse_employee_estimate("50000") == 50000


def test_parse_employee_estimate_handles_missing_or_unparseable():
    assert parse_employee_estimate(None) is None
    assert parse_employee_estimate("") is None
    assert parse_employee_estimate("a lot of people") is None


# ---- known megacap exclusion & employee_size_max enforcement ----


def test_known_megacap_detected_by_name_or_domain():
    assert _is_known_megacap("Amazon", None) is True
    assert _is_known_megacap("Amazon.com Inc", "https://www.amazon.com/careers") is True
    assert _is_known_megacap("Google", "google.com") is True
    assert _is_known_megacap("Myntra", "myntra.com") is True
    assert _is_known_megacap("Acme Robotics", "acmerobotics.io") is False


def test_exceeds_employee_max_uses_lower_bound_of_range():
    """
    Regression test for the same digit-mashing bug as
    test_parse_employee_estimate_takes_lower_bound_of_range, exercised
    through the actual call site that depends on it: a company reporting
    '500-1000' employees must NOT be excluded against a cap of 2000, even
    though naive digit-mashing would compute 5,001,000 and wrongly
    exclude it.
    """
    assert _exceeds_employee_max("5000+", 2000) is True
    assert _exceeds_employee_max("500-1000", 2000) is False
    assert _exceeds_employee_max("5000+", None) is False
    assert _exceeds_employee_max(None, 2000) is False
    assert _exceeds_employee_max("a lot of people", 2000) is False


def test_qualification_penalizes_company_over_employee_size_max():
    """A company well above employee_size_max should score LOW on the
    company_size sub-score even though it would otherwise score HIGH on
    every other dimension — this is what stops an Amazon/Google-scale
    company from registering as a 'High' tier staffing prospect just
    because it has huge absolute hiring volume."""
    hiring = {
        "estimated_total_openings": 500,
        "engineering_jobs": 400,
        "ai_jobs": 50,
        "backend_jobs": 100,
        "data_jobs": 50,
        "hiring_locations": ["Seattle", "Austin", "Dublin"],
        "hiring_trend": "Increasing",
    }
    enrichment = {"employee_count": "1,500,000", "funding_stage": None, "global_offices": ["Dublin"]}
    company = {}
    icp = {"employee_size_min": 100, "employee_size_max": 2000}

    result = _score_company(hiring, enrichment, company, icp, threshold=20)
    assert result["breakdown"]["company_size"] == 2.0
