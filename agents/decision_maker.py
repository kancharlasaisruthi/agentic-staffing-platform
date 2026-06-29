"""
Decision Maker Agent.

Finds staffing-relevant buyer personas at each qualifying company, in the
priority order given by config.target_personas (defaults to the spec's
Head of TA > TA Manager > HR Director > VP Eng > CTO ordering if config
doesn't override it).

Honesty constraint: this agent has NO access to a people-data API
(Apollo/PDL/ZoomInfo/Crunchbase contacts), so "finding" a decision maker
here means finding a plausible named person via public web search (often
a LinkedIn profile page surfaced by Tavily) — not verifying through a
dedicated people-search product. Every record is tagged with how it was
found so the Recommendation/UI layer can represent confidence honestly
rather than implying a verified org-chart lookup happened.
"""

from __future__ import annotations

from agents.base import agent_node
from core.llm import LLMClient
from core.memory import record_agent_output
from core.search import SearchClient
from core.state import State

EXTRACTION_SYSTEM_PROMPT = """You are identifying named individuals who \
hold specific job titles at ONE company, for B2B sales outreach \
targeting. Given web search results (often LinkedIn page titles/snippets), \
extract people whose role matches one of the target personas provided.

Only include a person if their name AND role are both actually present in \
the source text — do not guess a plausible-sounding name. If no named \
individual is found for a persona, omit that persona rather than \
inventing someone.

Return JSON exactly in this form:
{
  "decision_makers": [
    {
      "name": "string",
      "role": "string - their actual title as found",
      "matched_persona": "which target persona this best matches",
      "linkedin_url": "string or null",
      "source_url": "string",
      "reason_selected": "one sentence on why this persona/person matters for staffing outreach"
    }
  ]
}
"""


@agent_node("decision_maker")
def decision_maker_agent(state: State) -> dict:
    config = state.get("config", {})
    personas = config.get(
        "target_personas",
        ["Head of Talent Acquisition", "Talent Acquisition Manager", "HR Director", "VP Engineering", "CTO"],
    )
    threshold = config.get("hiring_threshold", 0)
    intel = state.get("hiring_intel", {})

    search = SearchClient()
    llm = LLMClient()

    dm_state = dict(state.get("decision_makers", {}))
    memory_updates: dict = {}
    processed = 0

    for c in state.get("validated_companies", []):
        if not c.get("is_valid"):
            continue
        key = c["key"]
        rec = intel.get(key)
        if not rec or rec.get("estimated_total_openings", 0) < threshold:
            continue
        if key in dm_state:
            continue

        persona_query = " OR ".join(f'"{p}"' for p in personas[:4])
        results = search.search(
            f'"{c["name"]}" ({persona_query}) linkedin',
            max_results=8,
            search_depth=config.get("search_depth", "basic"),
            include_domains=["linkedin.com"],
        )
        if not results:
            results = search.search(
                f'"{c["name"]}" {personas[0]} OR {personas[-1]}',
                max_results=6,
                search_depth=config.get("search_depth", "basic"),
            )

        if not results:
            dm_state[key] = []
        else:
            text = "\n\n".join(f"URL: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:400]}" for r in results)
            prompt_with_personas = f"TARGET PERSONAS (priority order): {', '.join(personas)}\n\n{text}"
            extracted = llm.extract_json(EXTRACTION_SYSTEM_PROMPT, prompt_with_personas, max_tokens=1200)
            dm_state[key] = extracted.get("decision_makers", []) if extracted else []

        memory_updates.update(record_agent_output(state, key, "decision_maker", {"contacts": dm_state[key]}))
        processed += 1

    return {
        "decision_makers": dm_state,
        **memory_updates,
        "_summary": f"Decision Maker: searched {processed} companies for target personas",
    }
