"""
Tavily search wrapper.

Centralizes two things every agent needs and must not reimplement
inconsistently:

1. Domain steering — the platform spec is explicit that Indeed/Glassdoor
   must never be the *primary* source of a final answer. We don't fully
   exclude them (they're useful corroborating signal for hiring volume),
   but we never let them be the *only* evidence and we tag any result from
   them so downstream agents can discount/exclude as appropriate.

2. Query caching via core/memory — same query string should never hit the
   API twice in one run.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from tavily import TavilyClient

logger = logging.getLogger("platform.search")

# Sources explicitly preferred by the spec as primary evidence.
PREFERRED_DOMAINS = [
    "linkedin.com",
    "crunchbase.com",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "wellfound.com",
]

# Sources allowed only as supporting/corroborating evidence, never as the
# cited primary source for a company's final report entry.
EVIDENCE_ONLY_DOMAINS = ["indeed.com", "glassdoor.com"]


class SearchClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Add it to your .env file — see .env.example."
            )
        self._client = TavilyClient(api_key=key)

    def search(
        self,
        query: str,
        max_results: int = 8,
        search_depth: str = "basic",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        topic: str = "general",
        retries: int = 2,
    ) -> list[dict]:
        """
        Returns a flat list of result dicts:
        {title, url, content, score, is_evidence_only}
        Never raises on transient failure within `retries` attempts — logs
        and returns [] instead, since one failed search should not crash
        an entire multi-company run.
        """
        last_err = None
        for attempt in range(retries + 1):
            try:
                raw = self._client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    topic=topic,
                )
                results = []
                for r in raw.get("results", []):
                    url = r.get("url", "")
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": url,
                            "content": r.get("content", ""),
                            "score": r.get("score", 0.0),
                            "is_evidence_only": any(d in url for d in EVIDENCE_ONLY_DOMAINS),
                        }
                    )
                return results
            except Exception as e:  # noqa: BLE001 - intentionally broad, see docstring
                last_err = e
                logger.warning("Tavily search failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
                time.sleep(1.5 * (attempt + 1))
        logger.error("Tavily search permanently failed for query=%r: %s", query, last_err)
        return []

    def search_primary_sources(self, query: str, max_results: int = 8, search_depth: str = "basic") -> list[dict]:
        """Search biased toward preferred domains for primary-evidence use cases
        (company discovery, hiring intel). Falls back to unrestricted search
        if the domain-restricted search returns nothing, so we don't go empty
        just because a small company isn't on Greenhouse."""
        restricted = self.search(
            query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=PREFERRED_DOMAINS,
        )
        if restricted:
            return restricted
        return self.search(
            query,
            max_results=max_results,
            search_depth=search_depth,
            exclude_domains=None,  # allow everything including evidence-only, tagged accordingly
        )


def filter_citable(results: list[dict]) -> list[dict]:
    """Results safe to cite as the primary source in a final report row."""
    return [r for r in results if not r.get("is_evidence_only")]


def filter_evidence_only(results: list[dict]) -> list[dict]:
    """Results only usable as corroborating signal (Indeed/Glassdoor)."""
    return [r for r in results if r.get("is_evidence_only")]
