"""
Contact Enrichment Agent.

Attempts to enrich decision-maker contacts with corporate email, phone,
LinkedIn profile, and a confidence score. Per the spec: "If unavailable,
return Unknown. Never hallucinate."

Without a paid people-data API (Apollo/PDL/Clearbit/ProxyCurl), there is
no reliable way to retrieve a verified personal email or direct phone
number for a named individual from public web search alone. So this
agent is deliberately conservative:

- LinkedIn URL: passed through from Decision Maker Agent if present, or a
  best-effort search.
- Email: we only ever output an INFERRED PATTERN (e.g.
  "first.last@company.com"), explicitly labeled as a pattern guess with a
  "low"/"medium" confidence — never presented as a verified individual
  address unless the exact address was actually found verbatim in a
  search result (rare, but possible from a public bio/press page).
- Phone: always "Unknown" unless a number is explicitly published
  (e.g. a public company directory page) — never inferred.

This is the agent to swap out first when you add a paid provider — see
README "Plugging in paid enrichment later".
"""

from __future__ import annotations

import re

from agents.base import agent_node
from core.llm import LLMClient
from core.memory import record_agent_output
from core.search import SearchClient
from core.state import State

EXTRACTION_SYSTEM_PROMPT = """You are doing best-effort, no-hallucination \
contact enrichment for a named individual at a company, using ONLY the \
provided web search content. 

Rules:
- Only output an email address as "found_email" if that EXACT address \
literally appears in the source text. Otherwise found_email must be null.
- Only output "found_phone" if a phone number literally appears \
associated with this person/company in the source text. Otherwise null.
- Only output "company_domain" if the company's website domain is \
evident from the source text (e.g. from a LinkedIn URL, a press mention, \
or an email domain seen elsewhere).

Return JSON exactly in this form:
{
  "found_email": "string or null - ONLY if verbatim in source",
  "found_phone": "string or null - ONLY if verbatim in source",
  "company_domain": "string or null"
}
"""

COMMON_PATTERNS = [
    "{first}.{last}@{domain}",
    "{first}{last}@{domain}",
    "{f}{last}@{domain}",
]


def _domain_from_website(website: str | None) -> str | None:
    if not website:
        return None
    return (
        website.lower()
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .split("/")[0]
        .strip()
    ) or None


def _split_name(name: str) -> tuple[str, str] | None:
    parts = [p for p in re.split(r"\s+", name.strip()) if p.isalpha()]
    if len(parts) < 2:
        return None
    return parts[0].lower(), parts[-1].lower()


@agent_node("contact_enrichment")
def contact_enrichment_agent(state: State) -> dict:
    config = state.get("config", {})
    search = SearchClient()
    llm = LLMClient()

    contact_state = dict(state.get("contact_enrichment", {}))
    memory_updates: dict = {}
    processed = 0

    company_by_key = {c["key"]: c for c in state.get("validated_companies", [])}

    for key, contacts in state.get("decision_makers", {}).items():
        if key in contact_state:
            continue  # memory reuse
        if not contacts:
            contact_state[key] = []
            continue

        company = company_by_key.get(key, {})
        domain = _domain_from_website(company.get("website"))

        enriched_contacts = []
        for person in contacts:
            name = person.get("name", "")
            results = search.search(
                f'"{name}" "{company.get("name", "")}" email contact',
                max_results=4,
                search_depth=config.get("search_depth", "basic"),
            )
            text = "\n\n".join(f"URL: {r['url']}\nCONTENT: {r['content'][:400]}" for r in results) or "No results."
            extracted = llm.extract_json(EXTRACTION_SYSTEM_PROMPT, text, max_tokens=400) or {}

            found_email = extracted.get("found_email")
            found_domain = extracted.get("company_domain") or domain
            found_phone = extracted.get("found_phone")

            inferred_pattern = None
            confidence = "low"
            if not found_email and found_domain:
                split = _split_name(name)
                if split:
                    first, last = split
                    inferred_pattern = COMMON_PATTERNS[0].format(first=first, last=last, domain=found_domain)
                    confidence = "low"  # pattern guess, not verified

            email_value = found_email or inferred_pattern or "Unknown"
            if found_email:
                confidence = "high"

            enriched_contacts.append(
                {
                    **person,
                    "email": email_value,
                    "email_is_verified": bool(found_email),
                    "email_is_pattern_guess": bool(inferred_pattern and not found_email),
                    "phone": found_phone or "Unknown",
                    "contact_confidence": confidence,
                }
            )

        contact_state[key] = enriched_contacts
        memory_updates.update(record_agent_output(state, key, "contact_enrichment", {"contacts": enriched_contacts}))
        processed += 1

    return {
        "contact_enrichment": contact_state,
        **memory_updates,
        "_summary": f"Contact Enrichment: processed {processed} companies' contacts (best-effort, no paid provider)",
    }
