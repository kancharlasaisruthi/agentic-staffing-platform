"""
Shared scaffolding every agent module uses.

Per the spec: "Include logging, retries, error handling, and modular
architecture." Rather than re-implement try/except/log in each of the 11
agent files, the @agent_node decorator wraps a node function so that:

- entry/exit is logged with timing
- exceptions are caught, logged to state["errors"], and converted into a
  no-op state update (so one company's failure doesn't crash the whole run)
- the plan_log gets a human-readable trace entry automatically

Agents that need their own retry logic around a specific call (e.g. a
flaky search) still implement that locally — this decorator handles the
node-level "don't crash the graph" concern, not per-call retries (those
live in core/search.py and core/llm.py).
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from core.memory import append_plan_log, log_error, mark_agent_run
from core.state import State

logger = logging.getLogger("platform.agents")


def agent_node(name: str) -> Callable:
    def decorator(fn: Callable[[State], dict]) -> Callable[[State], dict]:
        @functools.wraps(fn)
        def wrapper(state: State) -> dict:
            start = time.monotonic()
            logger.info("[%s] starting", name)
            try:
                update = fn(state) or {}
                elapsed = time.monotonic() - start
                logger.info("[%s] finished in %.1fs", name, elapsed)
                log_update = append_plan_log(
                    state, name, update.get("_summary", f"{name} completed in {elapsed:.1f}s")
                )
                update.pop("_summary", None)
                # IMPORTANT: this is what lets the planner's
                # "_needs_market_trigger"-style once-per-run checks ever
                # turn False. Without recording here, agents_run stays
                # empty forever and the planner loops on the same agent
                # indefinitely (this was a real bug caught during testing,
                # not a hypothetical one — see graph smoke test).
                run_update = mark_agent_run(state, name)
                # merge all updates without clobbering each other
                merged = dict(update)
                merged["plan_log"] = log_update["plan_log"]
                merged["agents_run"] = run_update["agents_run"]
                return merged
            except Exception as e:  # noqa: BLE001 - node-level safety net by design
                elapsed = time.monotonic() - start
                logger.exception("[%s] failed after %.1fs", name, elapsed)
                err_update = log_error(state, agent=name, error=str(e))
                plan_update = append_plan_log(state, name, f"{name} FAILED: {e}")
                run_update = mark_agent_run(state, name)
                return {
                    **err_update,
                    "plan_log": plan_update["plan_log"],
                    "agents_run": run_update["agents_run"],
                }

        return wrapper

    return decorator
