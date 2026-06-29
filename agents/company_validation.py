"""
Company Validation Agent.

Verifies each discovered company: does it appear to actually exist
(website reachable / indexed), is hiring activity recent, does it
plausibly match the ICP (industry, size signal, location). Removes
duplicates (belt-and-suspenders on top of Discovery's own dedup, since a
company can also be re-surfaced through later trigger expansion).

This agent does NOT call an LLM for free-text reasoning per company —
that's intentionally cheap/fast (per your "stub the periphery" priority):
it does one quick verification search per company and a simple rule-based
confidence score. Hiring Intelligence (next agent) is where the deep,
LLM-driven analysis happens.
"""

from __future__ import annotations

import re

from agents.base import agent_node
from core.search import SearchClient
from core.state import State, parse_employee_estimate

# Backstop exclusion list for unambiguous mega-caps that run their own
# massive in-house talent acquisition organizations and essentially never
# buy contract staffing services the way a 100-2000 employee scale-up
# does. This exists because employee_estimate at the Discovery stage is
# often a vague string ("1000+", missing entirely) that the numeric
# employee_size_max check below can't reliably catch — Discovery doesn't
# have firm enrichment data yet, that only firms up later in the
# pipeline. Qualification (agents/qualification.py) ALSO penalizes
# oversized companies via the rubric, so this isn't the only line of
# defense, but it's the cheapest place to stop a known-bad case before
# spending hiring-intelligence/enrichment/decision-maker budget on it.
# Edit this list freely — it is plain data, not logic.
KNOWN_MEGACAPS = {
    "amazon", "amazon.com", "google", "alphabet", "google.com", "alphabet.com",
    "microsoft", "microsoft.com", "apple", "apple.com", "meta", "meta.com",
    "facebook", "facebook.com", "netflix", "netflix.com", "myntra", "myntra.com",
    "flipkart", "flipkart.com", "walmart", "walmart.com", "ibm", "ibm.com",
    "oracle", "oracle.com", "salesforce", "salesforce.com", "sap", "sap.com",
    "tcs", "tcs.com", "infosys", "infosys.com", "wipro", "wipro.com",
    "accenture", "accenture.com", "cognizant", "cognizant.com",
}


def _is_known_megacap(name: str, website: str | None) -> bool:
    name_norm = re.sub(r"[^a-z0-9.]", "", name.lower())
    domain_norm = None
    if website:
        domain_norm = (
            website.lower().replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        )
    return name_norm in KNOWN_MEGACAPS or (domain_norm in KNOWN_MEGACAPS if domain_norm else False)


def _exceeds_employee_max(employee_estimate: str | None, employee_size_max: int | None) -> bool:
    """Best-effort numeric check against employee_size_max. Returns False
    (don't exclude) whenever the estimate is missing or unparseable —
    Discovery-stage estimates are often vague strings like '1000+' or
    None, so we don't want a parsing failure to silently reject valid
    companies. This is a coarse pre-filter; Qualification's rubric does
    the more nuanced size scoring once Enrichment has firmer numbers.

    Uses the LOWER bound of a range (e.g. '500-1000' -> 500) as the
    conservative choice, so we only exclude when even the low end clearly
    blows past the cap, rather than over-excluding ambiguous ranges that
    might dip under it.
    """
    if not employee_size_max:
        return False
    estimate = parse_employee_estimate(employee_estimate)
    if estimate is None:
        return False
    return estimate > employee_size_max


def _quick_confidence(name: str, results: list[dict], icp_locations: list[str]) -> tuple[bool, float, str]:
    if not results:
        return False, 0.0, "No corroborating search results found — cannot confirm company exists"

    text_blob = " ".join(r["content"] for r in results).lower()
    name_mentioned = name.lower() in text_blob or any(name.lower() in r["title"].lower() for r in results)
    location_match = any(loc.lower() in text_blob for loc in icp_locations) if icp_locations else True
    has_careers_or_jobs = any(
        kw in text_blob for kw in ["careers", "we're hiring", "open positions", "join our team", "open roles"]
    )

    score = 0.3 * name_mentioned + 0.3 * has_careers_or_jobs + 0.2 * location_match + 0.2 * (len(results) >= 3)
    is_valid = name_mentioned and (has_careers_or_jobs or location_match)

    reasons = []
    if name_mentioned:
        reasons.append("company name corroborated by search")
    if has_careers_or_jobs:
        reasons.append("active hiring language found")
    if location_match:
        reasons.append("location matches ICP")
    if not reasons:
        reasons.append("insufficient corroborating signal")

    return is_valid, round(score, 2), "; ".join(reasons)


@agent_node("company_validation")
def company_validation_agent(state: State) -> dict:
    config = state.get("config", {})
    icp_locations = config.get("icp", {}).get("locations", [])
    employee_size_max = config.get("icp", {}).get("employee_size_max")
    search = SearchClient()

    validated = list(state.get("validated_companies", []))
    validated_keys = {c["key"] for c in validated}

    checked = 0
    excluded_megacap = 0
    excluded_oversize = 0

    for c in state.get("discovered_companies", []):
        if c["key"] in validated_keys:
            continue  # never re-validate the same company

        # Cheap pre-checks BEFORE spending a search call: known mega-caps
        # and companies already reporting an employee estimate over the
        # configured cap are rejected immediately, since no amount of
        # corroborating search evidence should make them a fit for an ICP
        # with an employee_size_max set.
        if _is_known_megacap(c["name"], c.get("website")):
            validated.append(
                {
                    **c,
                    "is_valid": False,
                    "validation_confidence": 1.0,
                    "validation_reason": "Excluded: known mega-cap company with its own in-house TA org "
                    "(see KNOWN_MEGACAPS in agents/company_validation.py — edit that list to change this).",
                }
            )
            validated_keys.add(c["key"])
            checked += 1
            excluded_megacap += 1
            continue

        if _exceeds_employee_max(c.get("employee_estimate"), employee_size_max):
            validated.append(
                {
                    **c,
                    "is_valid": False,
                    "validation_confidence": 0.9,
                    "validation_reason": f"Excluded: employee_estimate ({c.get('employee_estimate')}) "
                    f"exceeds configured employee_size_max ({employee_size_max}).",
                }
            )
            validated_keys.add(c["key"])
            checked += 1
            excluded_oversize += 1
            continue

        query = f'"{c["name"]}" careers hiring jobs'
        if c.get("website"):
            query += f' {c["website"]}'
        results = search.search(query, max_results=5, search_depth="basic")

        is_valid, confidence, reason = _quick_confidence(c["name"], results, icp_locations)

        validated.append(
            {
                **c,
                "is_valid": is_valid,
                "validation_confidence": confidence,
                "validation_reason": reason,
            }
        )
        validated_keys.add(c["key"])
        checked += 1

    valid_count = sum(1 for c in validated if c.get("is_valid"))
    return {
        "validated_companies": validated,
        "_summary": (
            f"Company Validation: checked {checked} companies, {valid_count}/{len(validated)} valid so far "
            f"({excluded_megacap} excluded as known mega-caps, {excluded_oversize} excluded for exceeding employee_size_max)"
        ),
    }
