"""
Thin wrapper around the Groq API for the structured-extraction calls
agents need (turning search result text into typed JSON).

Kept deliberately separate from any LangChain LLM abstraction — agents call
`extract_json(...)` directly with a schema description and get a dict back,
with retries and JSON-repair built in. This keeps each agent's prompt and
parsing logic visible and debuggable in its own file rather than hidden
behind a framework abstraction.

This file is the ONLY place that talks to the LLM provider's SDK directly
— every agent calls LLMClient.extract_json(...) and has no knowledge of
which provider is behind it. Switching providers again later means editing
only this file.

Groq-specific notes (differences from a typical Anthropic-style wrapper):
- Groq's SDK is OpenAI-compatible: `client.chat.completions.create(...)`,
  not `client.messages.create(...)`. The system prompt is a normal entry
  in the `messages` list with role="system", not a separate `system=`
  kwarg.
- The response text lives at `resp.choices[0].message.content` (a plain
  string), not a list of typed content blocks.
- Token limit kwarg is `max_completion_tokens`, not `max_tokens`.
- Groq supports native JSON mode via `response_format={"type":
  "json_object"}`, which guarantees syntactically valid JSON (though not
  schema conformance) — this makes our own fence-stripping/brace-
  extraction fallback mostly a safety net rather than the primary path,
  but we keep it since JSON mode doesn't catch every edge case (e.g. a
  model occasionally still wrapping output despite the mode).
- JSON mode requires the literal word "json" to appear somewhere in the
  prompt or Groq returns a 400 error — our system prompt always mentions
  "JSON object" explicitly, so this is satisfied by construction.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from groq import Groq

logger = logging.getLogger("platform.llm")

# Pick any current Groq-hosted model. llama-3.3-70b-versatile is a solid
# default for structured extraction (good instruction-following, fast,
# inexpensive). Swap for a larger model (e.g. a 120b variant) if you find
# extraction quality on messy search results needs to be more reliable.
MODEL = "llama-3.3-70b-versatile"
from dotenv import load_dotenv
import os

load_dotenv()


class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env file — see .env.example."
            )
        self._client = Groq(api_key=key)

    def extract_json(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1500,
        retries: int = 2,
    ) -> Optional[dict]:
        """
        Calls the model with instructions to return ONLY JSON, parses the
        response, and repairs common formatting issues (markdown fences,
        leading/trailing prose). Returns None (never raises, never invents
        data) if parsing fails after retries — callers must handle None by
        falling back to "Unknown"/low-confidence rather than guessing.

        `max_tokens` is kept as the parameter name for compatibility with
        every existing agent call site (none needed to change) — it maps
        onto Groq's `max_completion_tokens` internally.
        """
        full_system = (
            system_prompt
            + "\n\nCRITICAL: Respond with ONLY a single valid JSON object. "
            "No markdown code fences, no preamble, no explanation text. "
            "If information is not present in the provided content, use "
            "null or \"Unknown\" for that field — never invent or guess a "
            "specific-sounding value (e.g. a name, email, or number) that "
            "is not actually supported by the content."
        )
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=MODEL,
                    max_completion_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_content},
                    ],
                )
                text = resp.choices[0].message.content or ""
                return _parse_json_loose(text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("LLM extraction failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
        logger.error("LLM extraction permanently failed: %s", last_err)
        return None


def _parse_json_loose(text: str) -> Optional[dict]:
    text = text.strip()
    # strip markdown fences if the model added them despite instructions
    # and despite JSON mode (rare, but seen with some open-weight models)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # last resort: grab the largest {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON even after extracting brace block")
    return None
