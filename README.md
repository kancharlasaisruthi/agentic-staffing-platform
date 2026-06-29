# Staffing Prospect Intelligence Platform

A reusable, planner-driven agentic platform that discovers companies actively
hiring and qualifies them as staffing/recruiting sales prospects — not a job
search tool. The final output of every run is a ranked list of **companies**,
never job postings.

Built with **LangGraph** (planner + specialized agent nodes over shared
state), **Tavily** for live web search, and **Groq** for fast LLM-based
extraction. No paid enrichment APIs (Clearbit/Apollo/PDL/Crunchbase) are
wired in yet — see "Extending" below for where to plug them in.

## Why this design

- **Planner-driven, not a fixed pipeline.** The Planner Agent inspects shared
  state after every step and decides what's missing, then routes to the
  next agent via LangGraph's `Command(goto=...)` mechanism. Add a new agent,
  register it with the planner's routing table, and it's part of the loop —
  no graph rewiring.
- **Config, not code, defines the business domain.** `config/icp_config.yaml`
  holds the ICP, industry, hiring thresholds, target personas. Point the
  same platform at a different vertical (e.g. healthcare staffing) by
  editing YAML, not Python.
- **Shared memory prevents duplicate work.** `core/memory.py` is a single
  state object threaded through the whole graph: companies seen, companies
  enriched, per-company agent outputs, a search-query cache. Every agent
  checks memory before calling out to the web.
- **Depth where it matters.** Per your priorities, **Hiring Intelligence**
  and **Qualification** are fully implemented with real multi-source search,
  LLM-based extraction, and an explainable scoring rubric. The other agents
  (Market Trigger, Decision Maker, Contact Enrichment, Company Enrichment,
  Recommendation) are implemented but lighter-weight — functional, single
  search pass, simpler extraction — so they're obvious to deepen later
  without restructuring anything.

## Architecture

```
User config (YAML)
       │
       ▼
  Planner Agent ──────────────────────────────────┐
       │  (reads shared state, decides next agent) │
       ▼                                           │
  Market Trigger Agent  → companies + triggers      │
       │                                            │
       ▼                                            │
  Company Discovery Agent → company records          │
       │                                            │
       ▼                                            │
  Company Validation Agent → confidence + dedup      │
       │                                            │
       ▼                                            │
  Hiring Intelligence Agent → open-role estimates ◄──┤ (deepest agent)
       │                                            │
       ▼                                            │
  Company Enrichment Agent → firmographics           │
       │                                            │
       ▼                                            │
  Decision Maker Agent → named buyer roles           │
       │                                            │
       ▼                                            │
  Contact Enrichment Agent → email/LinkedIn (best-effort)
       │                                            │
       ▼                                            │
  Qualification Agent → score + reasoning ◄──────────┤ (deepest agent)
       │                                            │
       ▼                                            │
  Recommendation Agent → next action                │
       │                                            │
       ▼                                            │
  Human Approval Agent ──────────────────────────────┘
       │ (loops back to Planner until user approves/edits/rejects)
       ▼
  Final Report
```

The Planner is the only node with conditional routing logic; every other
node is a straight function that reads state in and returns updates.

## Project layout

```
staffing-platform/
├── agents/
│   ├── base.py                 # AgentResult, retry/logging decorator
│   ├── planner.py               # Planner Agent — routing brain
│   ├── market_trigger.py
│   ├── company_discovery.py
│   ├── company_validation.py
│   ├── hiring_intelligence.py   # deep — primary agent
│   ├── company_enrichment.py
│   ├── decision_maker.py
│   ├── contact_enrichment.py
│   ├── qualification.py         # deep — scoring rubric
│   ├── recommendation.py
│   └── human_approval.py
├── core/
│   ├── state.py                 # shared State TypedDict
│   ├── memory.py                 # dedup / cache helpers over state
│   ├── search.py                  # Tavily wrapper, domain allow/deny lists
│   ├── llm.py                      # Groq client wrapper for extraction
│   ├── persistence.py               # local JSON store for approve/reject decisions
│   └── graph.py                       # builds & compiles the LangGraph
├── config/
│   ├── icp_config.yaml               # the ONLY file you edit to retarget domains
│   └── schema.py                       # pydantic validation of the config
├── data/
│   └── decisions.json                  # created at runtime — approved/rejected companies (gitignored)
├── ui/
│   ├── app.py                            # entry point — sets up top navigation between pages
│   └── pages_impl/
│       ├── discovery.py                    # main pipeline dashboard (sidebar, run button, company cards)
│       ├── approved.py                      # read-only view of approved companies (from data/decisions.json)
│       └── rejected.py                       # read-only view of rejected companies (from data/decisions.json)
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: add TAVILY_API_KEY and GROQ_API_KEY
```

Get a Tavily key at https://tavily.com (free tier available).
Get a free Groq key at https://console.groq.com.

## Run

```bash
streamlit run ui/app.py
```

The app opens with three pages, switchable via the row of tabs at the top
of the page (not the sidebar — the sidebar is reserved for ICP
configuration):

- **Discovery** — the main dashboard. Configure your ICP, click
  **Run Discovery**, review company cards, and Approve / Reject / Edit /
  Re-score each prospect.
- **Approved Companies** — every company you've approved, read from
  `data/decisions.json` on disk. This persists across page refreshes and
  app restarts — it is NOT tied to the current browser session.
- **Rejected Companies** — same idea, for rejected companies. Each entry
  has a "Move to Approved" button if you change your mind later.

Approving or rejecting a company on the Discovery page immediately writes
to `data/decisions.json`, so the Approved/Rejected pages reflect it right
away. Clicking **Re-score** on a company clears its prior decision from
disk (since the score that decision was based on is being recalculated)
and removes it from whichever list it was in until a new decision is made.

Edit `config/icp_config.yaml` first, or change values in the sidebar at
runtime (sidebar edits override the YAML for that session only).

## Extending to a new business domain

Everything domain-specific lives in `config/icp_config.yaml`:

```yaml
industry: "Technology"
icp:
  employee_size_min: 100
  locations: ["United States"]
  hiring_focus: ["Software Engineers", "Backend Engineers", "AI Engineers"]
hiring_threshold: 20
target_personas:
  - "Head of Talent Acquisition"
  - "VP Engineering"
```

To retarget at, say, healthcare staffing: change `industry`, `hiring_focus`
to clinical roles, and `target_personas` to "Director of Nursing" / "VP
Clinical Operations". No agent code changes.

## Adding a new agent

1. Write a function `def my_agent(state: State) -> dict` in `agents/`.
2. Register it in `core/graph.py`'s node list and in the Planner's
   `AGENT_REGISTRY` (agents/planner.py) with the state-key it's responsible
   for populating and the state-key(s) it depends on.
3. The Planner will pick it up automatically — no manual edge wiring beyond
   the one line in `graph.py` that adds the node.

## Plugging in paid enrichment later

`agents/contact_enrichment.py` and `agents/decision_maker.py` are the two
agents that would benefit most from a real data provider (Apollo, PDL,
Clearbit, ProxyCurl, Crunchbase API). Each has a single `_fetch_via_*`
function — swap its body for an API call and the rest of the agent
(confidence scoring, "Unknown" fallback, no-hallucination guarantee) needs
no changes.

## Excluding mega-caps (Amazon, Google, etc.)

By default `icp.employee_size_min` has no upper bound, which means
companies the size of Amazon or Google can show up as "qualified"
prospects — technically true (they are hiring), but not useful leads,
since companies that size run their own large in-house talent acquisition
organizations and essentially never buy contract staffing services.

Two settings control this:

- **`icp.employee_size_max`** in `config/icp_config.yaml` (or the sidebar
  field "Maximum employee size (ICP)") — companies whose reported
  employee count clearly exceeds this are excluded outright in Company
  Validation, and any that slip through with an ambiguous early estimate
  still get penalized in the Qualification score once Enrichment has
  firmer numbers.
- **`KNOWN_MEGACAPS`** in `agents/company_validation.py` — a small,
  editable set of unambiguous mega-cap names/domains (Amazon, Google,
  Microsoft, Meta, etc.) excluded regardless of what employee-count
  string Discovery happened to find, since that string is often missing
  or vague at the Discovery stage. This is plain data — add or remove
  companies freely; it's not logic you need to touch elsewhere.

## Known limitations (please read before relying on output)

- **No LinkedIn or Crunchbase scraping.** Both are heavily bot-protected.
  Tavily search will surface LinkedIn/Crunchbase *pages* in results (title,
  snippet, URL) but the agents do not log in or scrape behind auth. Treat
  any LinkedIn URLs in output as "likely the right page" not "verified."
- **No verified emails or phones without a paid provider.** Contact
  Enrichment will return an inferred email *pattern* (e.g.
  `first.last@company.com`) with a low confidence flag, and "Unknown" for
  phone, rather than ever inventing a real-looking address.
- **Hiring counts are estimates**, built from how many distinct open roles
  Tavily-sourced pages mention, not a scrape of a live ATS API. The agent
  is explicit in its output about which counts are "observed" vs "estimated."
- **Greenhouse/Lever/Ashby public job boards** are scrape-friendly (they're
  meant to be public), so these tend to be the most reliable hiring-count
  source. Indeed/Glassdoor results are deliberately excluded from primary
  evidence per the platform's rules and only referenced as corroborating
  signal counts, never as the cited source in the final report.
- **Re-running discovery in the same session won't find NEW companies**
  if the Planner considers every "need" already satisfied (Market Trigger
  and Company Discovery each only run once per run unless their outputs
  are cleared). Click **Reset Run** to start a fresh discovery cycle with
  new searches — but expect a different mix of companies each time, since
  search results aren't deterministic across calls.
