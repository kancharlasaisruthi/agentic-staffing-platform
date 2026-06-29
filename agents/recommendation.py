"""
Recommendation Agent.

Translates a qualification score + tier into a concrete sales next-action:
priority, recommended contact (best-available decision maker by persona
priority), recommended outreach angle, and urgency — matching the spec's
Stripe-style worked example. Rule-based mapping from tier -> urgency, with
a short LLM call to phrase the specific outreach angle using the actual
hiring/growth facts for that company (so the "Next Action" line is
grounded, not generic boilerplate).
"""

from __future__ import annotations

from agents.base import agent_node
from core.llm import LLMClient
from core.memory import record_agent_output
from core.state import State

TIER_TO_URGENCY = {
    "High": "Reach out within 24 hours",
    "Medium": "Reach out within this week",
    "Low": "Add to nurture list — revisit in 30 days",
}

OUTREACH_SYSTEM_PROMPT = """You are a staffing sales strategist. Given a \
company's hiring facts and qualification tier, write ONE concise sentence \
recommending the specific outreach angle a staffing/recruiting firm should \
use when contacting this company — grounded in the actual facts given, \
not generic. Example style: "Lead with contract-to-hire engineering \
staffing given their 84 open engineering roles across three offices."

Return JSON exactly in this form:
{ "recommended_outreach": "string" }
"""


def _best_contact(contacts: list[dict], personas: list[str]) -> dict | None:
    if not contacts:
        return None
    persona_rank = {p.lower(): i for i, p in enumerate(personas)}
    ranked = sorted(
        contacts,
        key=lambda c: persona_rank.get((c.get("matched_persona") or c.get("role") or "").lower(), len(personas)),
    )
    return ranked[0]


@agent_node("recommendation")
def recommendation_agent(state: State) -> dict:
    config = state.get("config", {})
    personas = config.get("target_personas", [])
    contacts_by_company = state.get("contact_enrichment", {})
    dm_by_company = state.get("decision_makers", {})
    hiring_intel = state.get("hiring_intel", {})
    company_by_key = {c["key"]: c for c in state.get("validated_companies", [])}

    llm = LLMClient()
    recs = dict(state.get("recommendations", {}))
    memory_updates: dict = {}
    processed = 0

    for key, qual in state.get("qualification", {}).items():
        if key in recs:
            continue  # memory reuse

        tier = qual.get("tier", "Low")
        company = company_by_key.get(key, {})
        hiring = hiring_intel.get(key, {})

        enriched_contacts = contacts_by_company.get(key) or dm_by_company.get(key) or []
        best = _best_contact(enriched_contacts, personas)
        recommended_contact = best.get("matched_persona") or best.get("role") if best else (
            personas[0] if personas else "Hiring decision maker"
        )

        outreach_input = (
            f"Company: {company.get('name', key)}\n"
            f"Tier: {tier}, Score: {qual.get('score')}\n"
            f"Qualification reasoning: {qual.get('reasoning')}\n"
            f"Total openings: {hiring.get('estimated_total_openings')}, "
            f"engineering jobs: {hiring.get('engineering_jobs')}, locations: {hiring.get('hiring_locations')}"
        )
        outreach_result = llm.extract_json(OUTREACH_SYSTEM_PROMPT, outreach_input, max_tokens=200)
        recommended_outreach = (
            outreach_result.get("recommended_outreach")
            if outreach_result and outreach_result.get("recommended_outreach")
            else f"Lead with a staffing proposal tailored to their {hiring.get('estimated_total_openings', 0)} open roles."
        )

        recs[key] = {
            "priority": tier,
            "reason": qual.get("reasoning", ""),
            "recommended_contact": recommended_contact,
            "recommended_outreach": recommended_outreach,
            "urgency": TIER_TO_URGENCY.get(tier, TIER_TO_URGENCY["Low"]),
        }
        memory_updates.update(record_agent_output(state, key, "recommendation", recs[key]))
        processed += 1

    return {
        "recommendations": recs,
        **memory_updates,
        "_summary": f"Recommendation: generated {processed} next-action recommendations",
    }
