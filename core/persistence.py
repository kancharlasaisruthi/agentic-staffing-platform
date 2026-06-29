"""
Local JSON persistence for human approval decisions.

This is DELIBERATELY separate from core/state.py's in-memory State —
that dict lives only for the duration of one Streamlit session (or one
run_cli.py process) and is lost on refresh/restart. This module is the
durable record of what a human actually decided, so:

- Approved/rejected companies survive a page refresh, a Streamlit
  restart, or even closing and reopening your browser days later.
- The "Approved Companies" / "Rejected Companies" pages (ui/pages/) read
  straight from this file rather than from session state, so they work
  even if you land on those pages without having run a fresh discovery
  cycle in the current session.

Storage shape — a single file, one row per company, status as a field
(rather than two separate files) — chosen because a company can move
between approved/rejected/pending over time (e.g. re-scored and
re-approved later), and a single source of truth avoids two files
silently drifting out of sync with each other.

File: data/decisions.json
{
  "<company_key>": {
    "company_name": str,
    "website": str | None,
    "score": float | None,
    "tier": str | None,
    "status": "approved" | "rejected" | "edited",
    "reasoning": str | None,
    "recommended_contact": str | None,
    "recommended_outreach": str | None,
    "urgency": str | None,
    "contacts": list[dict],
    "feedback": str | None,
    "decided_at": str (ISO timestamp),
  },
  ...
}
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from pathlib import Path
from typing import Literal

logger = logging.getLogger("platform.persistence")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DECISIONS_FILE = DATA_DIR / "decisions.json"


_lock = threading.Lock()

Status = Literal["approved", "rejected", "edited"]


def _ensure_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DECISIONS_FILE.exists():
        DECISIONS_FILE.write_text("{}")


def _read_all() -> dict:
    _ensure_file()
    try:
        return json.loads(DECISIONS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("decisions.json unreadable (%s) — treating as empty rather than crashing", e)
        return {}


def _write_all(data: dict) -> None:
    _ensure_file()
    # write to a temp file then replace, so a crash mid-write can't leave
    # a half-written/corrupt JSON file behind
    tmp_path = DECISIONS_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str))
    tmp_path.replace(DECISIONS_FILE)


def save_decision(
    company_key: str,
    status: Status,
    company_name: str,
    website: str | None = None,
    score: float | None = None,
    tier: str | None = None,
    reasoning: str | None = None,
    recommended_contact: str | None = None,
    recommended_outreach: str | None = None,
    urgency: str | None = None,
    contacts: list[dict] | None = None,
    feedback: str | None = None,
) -> dict:
    """Persist one company's approve/reject/edit decision to disk.
    Overwrites any prior decision for the same company_key — the file
    always reflects the MOST RECENT decision per company, not a history
    of every decision ever made (re-approving after a re-score is meant
    to replace the old entry, not append a duplicate)."""
    with _lock:
        data = _read_all()
        data[company_key] = {
            "company_name": company_name,
            "website": website,
            "score": score,
            "tier": tier,
            "status": status,
            "reasoning": reasoning,
            "recommended_contact": recommended_contact,
            "recommended_outreach": recommended_outreach,
            "urgency": urgency,
            "contacts": contacts or [],
            "feedback": feedback,
            "decided_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        _write_all(data)
        return data[company_key]


def remove_decision(company_key: str) -> None:
    """Used when a company is sent back for re-scoring — its prior
    approve/reject decision is no longer valid once the underlying
    qualification changes, so we clear it rather than leave a stale
    decision on disk."""
    with _lock:
        data = _read_all()
        if company_key in data:
            del data[company_key]
            _write_all(data)


def get_by_status(status: Status) -> dict:
    return {k: v for k, v in _read_all().items() if v.get("status") == status}


def get_all() -> dict:
    return _read_all()
