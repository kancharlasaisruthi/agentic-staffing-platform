"""
core/email_drafter.py
─────────────────────
Drafts a cold-outreach email using Groq LLM and returns subject + body.
"""

from __future__ import annotations

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


_SYSTEM = """You are an expert B2B sales development representative at a technical staffing agency.
Write a short, personalised cold-outreach email to a decision maker at a prospect company.

Rules:
- Subject line: punchy, under 10 words, no clickbait
- Body: 3-4 short paragraphs, conversational but professional
- Open with a specific observation about the company (hiring surge, growth, funding)
- Connect it to a staffing pain point
- One sentence on the agency value proposition
- Close with a single low-friction CTA (15-min call)
- Do NOT invent contact details

Return ONLY two lines, exactly like this (no extra text, no JSON, no markdown):
SUBJECT: <subject line here>
BODY:
<email body here>
"""

_HUMAN = """Company: {company_name}
Website: {website}
Score: {score}/100  Tier: {tier}  Urgency: {urgency}
Recommended contact role: {recommended_contact}
Contact name: {contact_name}
Outreach strategy: {recommended_outreach}
Qualification reasoning: {reasoning}
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def draft_outreach_email(record: dict) -> dict[str, str]:
    """
    Returns {"subject": "...", "body": "..."}.
    Raises RuntimeError on LLM failure.
    """
    contacts = record.get("contacts") or []
    primary = contacts[0] if contacts else {}
    contact_name = primary.get("name", "the hiring leader")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    chain = _PROMPT | llm

    try:
        response = chain.invoke({
            "company_name":         record.get("company_name", "the company"),
            "website":              record.get("website") or "N/A",
            "score":                record.get("score") or "N/A",
            "tier":                 record.get("tier") or "N/A",
            "urgency":              record.get("urgency") or "N/A",
            "recommended_contact":  record.get("recommended_contact") or "decision maker",
            "contact_name":         contact_name,
            "recommended_outreach": record.get("recommended_outreach") or "personalised email",
            "reasoning":            record.get("reasoning") or "Strong hiring activity detected.",
        })
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    raw = response.content.strip()

    # Parse the two-line format: SUBJECT: ... \n BODY: \n ...
    subject = ""
    body = ""
    if "SUBJECT:" in raw and "BODY:" in raw:
        subject_part, body_part = raw.split("BODY:", 1)
        subject = subject_part.replace("SUBJECT:", "").strip()
        body = body_part.strip()
    else:
        # Fallback: treat first line as subject, rest as body
        lines = raw.splitlines()
        subject = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

    return {"subject": subject, "body": body}