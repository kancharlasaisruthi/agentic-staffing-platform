"""
Company Enrichment Agent.

Collects firmographic detail: description, revenue estimate, employee
count, funding/investors, tech stack, HQ/offices, recent news, growth
signals. Single search + single extraction pass per company (lighter
weight, per priority — Hiring Intelligence and Qualification are the deep
agents). Only runs for companies that cleared the hiring threshold, so we
don't spend enrichment budget on companies that won't be recommended
anyway.
"""

from __future__ import annotations

from agents.base import agent_node
from core.llm import LLMClient
from core.memory import mark_enriched, record_agent_output
from core.search import SearchClient
from core.state import State

EXTRACTION_SYSTEM_PROMPT = """You are a B2B sales research analyst \
building a firmographic profile of ONE company. Given raw web search \
results, extract what is actually supported by the text. Use null for \
anything not found — never invent funding amounts, revenue figures, or \
investor names.

Return JSON exactly in this form:
{
  "description": "1-2 sentence description of what the company does",
  "revenue_estimate": "string or null, e.g. '$50M-$100M' if stated/estimated in sources",
  "employee_count": "string or null",
  "funding_stage": "string or null, e.g. 'Series C'",
  "funding_total": "string or null",
  "investors": ["list of investor names if mentioned, else empty list"],
  "tech_stack": ["list of technologies mentioned, else empty list"],
  "cloud_provider": "string or null",
  "headquarters": "string or null",
  "global_offices": ["list of additional office locations if mentioned"],
  "recent_news": ["1-3 short factual bullet points about recent company news"],
  "growth_signals": ["1-3 short bullet points indicating growth, e.g. 'opened Austin office in 2025'"]
}
"""


@agent_node("company_enrichment")
def company_enrichment_agent(state: State) -> dict:
    config = state.get("config", {})
    search = SearchClient()
    llm = LLMClient()

    enrichment = dict(state.get("company_enrichment", {}))
    memory_updates: dict = {}
    processed = 0

    threshold = config.get("hiring_threshold", 0)
    intel = state.get("hiring_intel", {})

    for c in state.get("validated_companies", []):
        if not c.get("is_valid"):
            continue
        key = c["key"]
        rec = intel.get(key)
        if not rec or rec.get("estimated_total_openings", 0) < threshold:
            continue  # don't enrich companies below the hiring threshold
        if key in enrichment:
            continue  # memory reuse

        query = f'"{c["name"]}" company profile funding revenue tech stack headquarters'
        if c.get("website"):
            query += f' {c["website"]}'
        results = search.search(query, max_results=8, search_depth=config.get("search_depth", "basic"))

        if not results:
            enrichment[key] = {
                "description": None,
                "revenue_estimate": None,
                "employee_count": c.get("employee_estimate"),
                "funding_stage": None,
                "funding_total": None,
                "investors": [],
                "tech_stack": [],
                "cloud_provider": None,
                "headquarters": c.get("headquarters"),
                "global_offices": [],
                "recent_news": [],
                "growth_signals": [],
            }
        else:
            text = "\n\n".join(f"URL: {r['url']}\nCONTENT: {r['content'][:600]}" for r in results)
            extracted = llm.extract_json(EXTRACTION_SYSTEM_PROMPT, text, max_tokens=1000)
            enrichment[key] = extracted or {}

        memory_updates.update(mark_enriched(state, key, enrichment[key]))
        memory_updates.update(record_agent_output(state, key, "company_enrichment", enrichment[key]))
        processed += 1

    return {
        "company_enrichment": enrichment,
        **memory_updates,
        "_summary": f"Company Enrichment: enriched {processed} companies above hiring threshold",
    }
