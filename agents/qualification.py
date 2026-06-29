"""
Qualification Agent — the second agent you asked to be built with real
depth, alongside Hiring Intelligence.

Scores each company as a staffing-sales prospect using an explicit,
weighted, fully explainable rubric (not just an LLM vibe-score with no
audit trail). The LLM is used only to write the human-readable reasoning
string from the already-computed rubric inputs — the score itself is
computed deterministically in Python so it's debuggable, consistent
across runs, and tunable without touching prompts.

Rubric (weights sum to 100, mirroring the spec's listed factors):
- Hiring Volume            (25) — estimated_total_openings vs threshold
- Technical Hiring Mix     (15) — share of openings that are eng/AI/data/backend
- Company Size             (10) — employee count vs ICP minimum
- Growth Signals           (15) — recent_news / growth_signals presence + hiring_trend
- Funding                  (15) — funding_stage/funding_total presence and recency-ish signal
- Expansion                (10) — global_offices count, new-office triggers
- Hiring Locations Breadth (5)  — number of distinct hiring locations
- Engineering Demand       (5)  — engineering_jobs raw count, rewards absolute scale

Tiering: High >= 70, Medium >= 45, Low otherwise — configurable via
QUALIFICATION_TIERS if you want different cutoffs per vertical later.
"""

from __future__ import annotations

from agents.base import agent_node
from core.llm import LLMClient
from core.memory import record_agent_output
from core.state import State, parse_employee_estimate

QUALIFICATION_TIERS = [(70, "High"), (45, "Medium"), (0, "Low")]

REASONING_SYSTEM_PROMPT = """You are a staffing sales analyst. You are \
given a pre-computed qualification score breakdown for a company (each \
sub-score and what drove it). Write a concise 1-3 sentence human-readable \
explanation of WHY this company scored what it did, citing the specific \
numbers/facts given — similar in style to:

"Hiring 84 engineers. Opened London office. Series C funding. Hiring \
across three countries."

Be specific and factual. Do not add any claim not present in the input \
data. Return JSON exactly in this form:
{ "reasoning": "string" }
"""


def _tier_for_score(score: float) -> str:
    for cutoff, tier in QUALIFICATION_TIERS:
        if score >= cutoff:
            return tier
    return "Low"


def _score_company(hiring: dict, enrichment: dict, company: dict, icp: dict, threshold: int) -> dict:
    breakdown = {}

    # Hiring Volume (25) — scaled against threshold, capped at 2x threshold for full marks
    total = hiring.get("estimated_total_openings", 0) or 0
    volume_ratio = min(total / max(threshold, 1) / 2, 1.0) if threshold else min(total / 50, 1.0)
    breakdown["hiring_volume"] = round(volume_ratio * 25, 1)

    # Technical Hiring Mix (15) — share of total that's eng/ai/backend/data
    tech_jobs = sum(
        hiring.get(k, 0) or 0 for k in ["engineering_jobs", "ai_jobs", "backend_jobs", "data_jobs"]
    )
    tech_ratio = min(tech_jobs / total, 1.0) if total else 0.0
    breakdown["technical_mix"] = round(tech_ratio * 15, 1)

    # Company Size (10) — meets ICP minimum, penalized for exceeding the
    # configured maximum (if any). This is a second line of defense
    # alongside Company Validation's harder exclusion check
    # (agents/company_validation.py) — Validation runs on Discovery-stage
    # estimates which are often vague/missing, while this runs on firmer
    # Enrichment-stage data, so a mega-cap that slipped through Validation
    # under an ambiguous early estimate still gets scored down here once
    # better data is available.
    emp_min = icp.get("employee_size_min", 0)
    emp_max = icp.get("employee_size_max")
    employee_count_str = enrichment.get("employee_count") or company.get("employee_estimate")
    emp_num = parse_employee_estimate(employee_count_str)

    size_score = 5.0  # default partial credit if unknown — unverified isn't the same as disqualifying
    if emp_num is not None:
        if emp_max and emp_num > emp_max:
            # well outside the configured ceiling — score this as low
            # as an undersized company would score, rather than
            # letting sheer scale read as a positive signal
            size_score = 2.0
        else:
            size_score = 10.0 if emp_num >= emp_min else (5.0 if emp_num >= emp_min * 0.5 else 2.0)
    breakdown["company_size"] = size_score

    # Growth Signals (15)
    growth_signals = enrichment.get("growth_signals") or []
    recent_news = enrichment.get("recent_news") or []
    trend = hiring.get("hiring_trend", "Unknown")
    growth_score = 0.0
    growth_score += min(len(growth_signals) * 4, 8)
    growth_score += min(len(recent_news) * 2, 4)
    growth_score += {"Increasing": 3, "Stable": 1, "Decreasing": 0, "Unknown": 0.5}.get(trend, 0)
    breakdown["growth_signals"] = round(min(growth_score, 15), 1)

    # Funding (15)
    funding_score = 0.0
    if enrichment.get("funding_stage"):
        funding_score += 9
    if enrichment.get("funding_total"):
        funding_score += 4
    if enrichment.get("investors"):
        funding_score += 2
    breakdown["funding"] = round(min(funding_score, 15), 1)

    # Expansion (10)
    offices = enrichment.get("global_offices") or []
    expansion_score = min(len(offices) * 3, 10)
    breakdown["expansion"] = round(expansion_score, 1)

    # Hiring Locations Breadth (5)
    locations = hiring.get("hiring_locations") or []
    breakdown["location_breadth"] = round(min(len(locations) * 1.5, 5), 1)

    # Engineering Demand absolute scale (5)
    eng_jobs = hiring.get("engineering_jobs", 0) or 0
    breakdown["engineering_demand"] = round(min(eng_jobs / 20, 1.0) * 5, 1)

    total_score = round(sum(breakdown.values()), 1)
    return {"breakdown": breakdown, "score": min(total_score, 100.0)}


@agent_node("qualification")
def qualification_agent(state: State) -> dict:
    config = state.get("config", {})
    icp = config.get("icp", {})
    threshold = config.get("hiring_threshold", 0)
    intel = state.get("hiring_intel", {})
    enrichment_data = state.get("company_enrichment", {})
    company_by_key = {c["key"]: c for c in state.get("validated_companies", [])}

    llm = LLMClient()
    qual = dict(state.get("qualification", {}))
    memory_updates: dict = {}
    processed = 0

    for key, hiring in intel.items():
        if key in qual:
            continue  # memory reuse
        if hiring.get("estimated_total_openings", 0) < threshold:
            continue  # below threshold — not a qualified prospect, no score needed

        company = company_by_key.get(key, {})
        enrichment = enrichment_data.get(key, {})

        scored = _score_company(hiring, enrichment, company, icp, threshold)
        tier = _tier_for_score(scored["score"])

        reasoning_input = (
            f"Company: {company.get('name', key)}\n"
            f"Score: {scored['score']}/100, Tier: {tier}\n"
            f"Breakdown: {scored['breakdown']}\n"
            f"Hiring data: total_openings={hiring.get('estimated_total_openings')}, "
            f"engineering_jobs={hiring.get('engineering_jobs')}, trend={hiring.get('hiring_trend')}, "
            f"locations={hiring.get('hiring_locations')}\n"
            f"Enrichment: funding_stage={enrichment.get('funding_stage')}, "
            f"global_offices={enrichment.get('global_offices')}, growth_signals={enrichment.get('growth_signals')}"
        )
        reasoning_result = llm.extract_json(REASONING_SYSTEM_PROMPT, reasoning_input, max_tokens=300)
        reasoning = (
            reasoning_result.get("reasoning")
            if reasoning_result and reasoning_result.get("reasoning")
            else f"Scored {scored['score']}/100 ({tier}) based on {hiring.get('estimated_total_openings', 0)} estimated open roles and available growth/funding signals."
        )

        qual[key] = {
            "score": scored["score"],
            "tier": tier,
            "breakdown": scored["breakdown"],
            "reasoning": reasoning,
        }
        memory_updates.update(record_agent_output(state, key, "qualification", qual[key]))
        processed += 1

    return {
        "qualification": qual,
        **memory_updates,
        "_summary": f"Qualification: scored {processed} companies above hiring threshold",
    }
