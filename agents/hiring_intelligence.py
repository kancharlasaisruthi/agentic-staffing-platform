"""
Hiring Intelligence Agent — THE most important agent per the spec, and the
one you asked to be built with real depth.

Goal: produce a company-level hiring summary (never individual job
listings) covering total open positions, role-family breakdowns
(engineering/AI/backend/frontend/devops/data), hiring trend, growth rate,
and hiring locations — with an explicit confidence rating and a clear
statement of which numbers are "observed" (we found enumerable listings)
vs "estimated" (we inferred from partial signal), per the spec's
instruction to "estimate intelligently using all available evidence" when
exact numbers aren't available, and to never just output raw job
listings.

Design, source-by-source:

1. ATS boards (Greenhouse/Lever/Ashby) — these are public-facing and
   often list every open role with a job-family/department tag, so when
   we can reach a company's board, this is the highest-confidence source.
   We run a dedicated search per company restricted to these domains.
2. LinkedIn Jobs / company careers page — searched for corroboration and
   role-count language ("50+ open roles", "hiring across engineering").
3. Indeed / Glassdoor — explicitly evidence-only per the spec. We still
   search them (they often have the most complete count of listings for
   companies not on a major ATS) but we tag every such result and the
   LLM extraction step is told to treat them as lower-trust corroboration,
   never as the basis for the headline "Estimated Total" if better sources
   exist.
4. Company news / blog — for hiring-trend language ("doubling the
   engineering team", "plans to hire 200 by year end").

All raw text from steps 1-4 for one company is combined into a single
LLM extraction call with a detailed schema and explicit estimation rules,
modeled directly on the spec's Microsoft worked example.
"""

from __future__ import annotations

import logging

from agents.base import agent_node
from core.llm import LLMClient
from core.memory import record_agent_output
from core.search import EVIDENCE_ONLY_DOMAINS, SearchClient
from core.state import State

logger = logging.getLogger("platform.hiring_intelligence")

ATS_DOMAINS = ["boards.greenhouse.io", "jobs.lever.co", "jobs.ashbyhq.com", "wellfound.com"]

EXTRACTION_SYSTEM_PROMPT = """You are a staffing-industry hiring analyst. \
Given raw web search content about ONE specific company's open job \
postings, careers page, and hiring-related news, produce a COMPANY-LEVEL \
hiring summary. You must NEVER list individual job postings in your \
output — only company-level counts and summaries.

Distinguish two kinds of numbers:
- "observed": you can point to actual enumerated listings or an explicit \
stated count (e.g. a Greenhouse board showing 46 open roles, or an \
article stating "120 open positions").
- "estimated": no explicit count exists, so you are inferring a \
reasonable figure from partial evidence (e.g. seeing 8 distinct role \
titles mentioned across sources and inferring there are likely more \
unlisted). Estimated numbers should be conservative and should never be \
presented with false precision — round to a sensible figure.

If there is truly no usable evidence for a field, use 0 for counts and \
state that explicitly in the notes — do not fabricate a plausible-looking \
number.

Indeed and Glassdoor content, where present in the input, is marked as \
EVIDENCE-ONLY and should be used only to corroborate or sanity-check \
counts from other sources — never as the primary basis for the headline \
estimated_total_openings if better sources (ATS boards, company site, \
LinkedIn) are available. If Indeed/Glassdoor is the ONLY source available, \
you may use it but must set source_quality to "low" and say so in notes.

Return JSON exactly in this form:
{
  "estimated_total_openings": <int>,
  "total_openings_basis": "observed" or "estimated",
  "engineering_jobs": <int>,
  "ai_jobs": <int>,
  "backend_jobs": <int>,
  "frontend_jobs": <int>,
  "devops_jobs": <int>,
  "data_jobs": <int>,
  "hiring_locations": ["city/region strings"],
  "hiring_trend": "Increasing", "Stable", "Decreasing", or "Unknown",
  "growth_rate_note": "one sentence on growth rate/pace if evidence supports it, else null",
  "primary_sources_used": ["list of source types actually used, e.g. 'Greenhouse board', 'company careers page'"],
  "source_quality": "high", "medium", or "low",
  "confidence": "High", "Medium", or "Low",
  "notes": "1-3 sentences explaining the basis for the estimate and any caveats"
}
"""


def _search_company_hiring(search: SearchClient, name: str, website: str | None, search_depth: str) -> tuple[list, list]:
    """Returns (primary_results, evidence_only_results) for one company."""
    base = f'"{name}"'
    if website:
        base += f" {website}"

    primary: list = []
    # 1. ATS boards — highest confidence source when reachable
    primary.extend(
        search.search(
            f"{base} careers open positions",
            max_results=6,
            search_depth=search_depth,
            include_domains=ATS_DOMAINS,
        )
    )
    # 2. Company careers page / LinkedIn jobs language
    primary.extend(
        search.search(
            f"{base} careers \"open roles\" OR \"we're hiring\" engineering",
            max_results=6,
            search_depth=search_depth,
        )
    )
    # 3. Hiring-trend / growth news
    primary.extend(
        search.search(
            f"{base} hiring engineering team growth 2026",
            max_results=4,
            search_depth=search_depth,
            topic="news",
        )
    )
    # 4. Indeed/Glassdoor — evidence only, explicitly separated
    evidence_only = search.search(
        f"{base} jobs openings",
        max_results=5,
        search_depth=search_depth,
        include_domains=EVIDENCE_ONLY_DOMAINS,
    )

    return primary, evidence_only


def _build_extraction_input(name: str, primary: list[dict], evidence_only: list[dict]) -> str:
    parts = [f"COMPANY: {name}\n"]
    if primary:
        parts.append("=== PRIMARY SOURCES ===")
        for r in primary[:14]:
            parts.append(f"URL: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:700]}\n")
    if evidence_only:
        parts.append("=== EVIDENCE-ONLY SOURCES (Indeed/Glassdoor — corroboration only) ===")
        for r in evidence_only[:8]:
            parts.append(f"URL: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:500]}\n")
    if not primary and not evidence_only:
        parts.append("No search results were found for this company.")
    return "\n".join(parts)


def _fallback_record(reason: str) -> dict:
    return {
        "estimated_total_openings": 0,
        "total_openings_basis": "estimated",
        "engineering_jobs": 0,
        "ai_jobs": 0,
        "backend_jobs": 0,
        "frontend_jobs": 0,
        "devops_jobs": 0,
        "data_jobs": 0,
        "hiring_locations": [],
        "hiring_trend": "Unknown",
        "growth_rate_note": None,
        "primary_sources_used": [],
        "source_quality": "low",
        "confidence": "Low",
        "notes": reason,
    }


@agent_node("hiring_intelligence")
def hiring_intelligence_agent(state: State) -> dict:
    config = state.get("config", {})
    search_depth = config.get("search_depth", "basic")
    search = SearchClient()
    llm = LLMClient()

    intel = dict(state.get("hiring_intel", {}))
    memory_updates: dict = {}
    processed = 0

    for c in state.get("validated_companies", []):
        if not c.get("is_valid"):
            continue
        key = c["key"]
        if key in intel:
            continue  # memory reuse — never re-run hiring intel for a company we already have

        try:
            primary, evidence_only = _search_company_hiring(search, c["name"], c.get("website"), search_depth)
        except Exception:
            logger.exception("Search failed for %s", c["name"])
            intel[key] = _fallback_record("Search failed for this company; no data available.")
            processed += 1
            continue

        extraction_input = _build_extraction_input(c["name"], primary, evidence_only)
        result = llm.extract_json(EXTRACTION_SYSTEM_PROMPT, extraction_input, max_tokens=1200)

        if result is None:
            result = _fallback_record("LLM extraction failed; falling back to zero/Unknown rather than guessing.")
        else:
            # Defensive normalization — never trust the LLM blindly on types
            for int_field in [
                "estimated_total_openings",
                "engineering_jobs",
                "ai_jobs",
                "backend_jobs",
                "frontend_jobs",
                "devops_jobs",
                "data_jobs",
            ]:
                try:
                    result[int_field] = int(result.get(int_field, 0) or 0)
                except (TypeError, ValueError):
                    result[int_field] = 0

        intel[key] = result
        memory_updates.update(record_agent_output(state, key, "hiring_intelligence", result))
        processed += 1

    return {
        "hiring_intel": intel,
        **memory_updates,
        "_summary": f"Hiring Intelligence: processed {processed} companies (total tracked: {len(intel)})",
    }
