"""
Market Trigger Agent.

Discovers signals indicating a company may need staffing services: funding
announcements, expansion, new offices, rapid hiring, acquisitions, tech
migration. This is intentionally a single-pass search + extraction agent
(per your "stub the periphery, go deep on Hiring Intelligence and
Qualification" priority) — it produces a list of (company, trigger,
source, confidence) tuples that seed Company Discovery, but does not
itself do multi-round investigation.

Deepening this later: add per-trigger-type searches (a dedicated funding-
news query, a dedicated expansion-news query) instead of one combined
query, and run them in parallel.
"""

from __future__ import annotations

from agents.base import agent_node
from core.llm import LLMClient
from core.search import SearchClient
from core.state import State

EXTRACTION_SYSTEM_PROMPT = """You are a B2B sales research analyst looking \
for companies that show signals they may need to hire many people soon \
(and therefore may need staffing/recruiting help). Given raw web search \
results, extract a list of distinct COMPANIES (not job postings, not \
articles) mentioned, with the signal that suggests a hiring need.

Return JSON of the form:
{
  "triggers": [
    {
      "company": "string - company name as written in the source",
      "trigger_type": "one of: funding, expansion, new_office, rapid_hiring, \
team_growth, product_launch, acquisition, tech_migration, other_growth",
      "trigger_description": "one sentence describing the specific signal",
      "source_url": "the URL this came from",
      "confidence": "High, Medium, or Low based on how directly the source \
supports a hiring-need signal"
    }
  ]
}

Only include companies where the source text genuinely supports the \
trigger — do not infer a funding round or expansion that isn't stated.
"""


@agent_node("market_trigger")
def market_trigger_agent(state: State) -> dict:
    config = state.get("config", {})
    industry = config.get("industry", "Technology")
    locations = ", ".join(config.get("icp", {}).get("locations", ["United States"]))
    focus = ", ".join(config.get("icp", {}).get("hiring_focus", []))

    search = SearchClient()
    llm = LLMClient()

    queries = [
        f"{industry} companies funding announcement {locations} 2026",
        f"{industry} companies hiring surge expansion {focus} {locations}",
        f"{industry} company new office opening rapid hiring {locations}",
    ]

    all_results = []
    for q in queries:
        results = search.search(q, max_results=6, search_depth=config.get("search_depth", "basic"), topic="news")
        all_results.extend(results)

    if not all_results:
        return {
            "triggers": [],
            "_summary": "Market Trigger: no search results returned for trigger queries",
        }

    combined_text = "\n\n".join(
        f"URL: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:600]}" for r in all_results[:18]
    )

    extracted = llm.extract_json(EXTRACTION_SYSTEM_PROMPT, combined_text, max_tokens=2000)
    triggers = extracted.get("triggers", []) if extracted else []

    existing = list(state.get("triggers", []))
    existing.extend(triggers)

    return {
        "triggers": existing,
        "_summary": f"Market Trigger: found {len(triggers)} signals across {len(queries)} queries",
    }
