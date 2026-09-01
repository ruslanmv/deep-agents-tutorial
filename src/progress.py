"""Run an agent and show what it is doing while it does it.

`agent.invoke(...)` returns nothing until the whole run finishes. On a local
8B model that can be many minutes — planning, delegating, several searches,
then synthesis — and a silent terminal is indistinguishable from a hang. Every
lab here used to look frozen for exactly that reason.

`run_with_progress()` is a drop-in replacement: same input, same returned state,
but it streams the run and prints each model and tool call as it happens. If it
stops printing, *that* is a hang — and the last line tells you where.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

# Ollama has to pull the weights into VRAM before the first token. This is the
# single longest pause in a run and the one most likely to be mistaken for a
# freeze, so say so up front.
_FIRST_CALL_HINT = (
    "the first model call also loads the weights into VRAM — "
    "30-60s is normal, longer on a cold cache"
)


def _text_of(msg: Any) -> str:
    """Best-effort plain text of a message, whatever shape its content is."""
    text = getattr(msg, "text", None)
    if not isinstance(text, str):
        text = str(getattr(msg, "content", "") or "")
    return text.strip()


def final_text(messages: list) -> str:
    """The last assistant message that actually said something.

    `messages[-1].content` is not safe to print. A local model quite happily
    ends a run with an empty assistant turn — you get blank lines where the
    summary should be. deepagents walks backwards for exactly this reason when
    it collects a sub-agent's reply; so do we.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage):
            text = _text_of(msg)
            if text:
                return text
    return ""


def _tool_names(msg: Any) -> list[str]:
    return [c["name"] for c in (getattr(msg, "tool_calls", None) or [])]


def run_with_progress(agent: Any, payload: dict, *, label: str = "run") -> dict:
    """Stream `agent` over `payload`, printing progress; return the final state.

    Args:
        agent: A compiled agent from `create_deep_agent()`.
        payload: Exactly what you would have passed to `agent.invoke()`.
        label: Shown in the header line.

    Returns:
        The final state dict — the same thing `invoke()` would have returned,
        so `result["messages"]` and `result["files"]` work unchanged.
    """
    t0 = time.perf_counter()
    print(f"[{label}] streaming — every line below appears as it happens")
    print(f"          ({_FIRST_CALL_HINT})\n")

    final: dict | None = None
    pending: dict[str, float] = {}   # tool name -> when the model asked for it
    steps = 0

    for mode, chunk in agent.stream(payload, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
            continue

        for node, update in (chunk or {}).items():
            for msg in (update or {}).get("messages") or []:
                now = time.perf_counter() - t0

                if isinstance(msg, AIMessage):
                    calls = _tool_names(msg)
                    if calls:
                        steps += 1
                        for name in calls:
                            pending[name] = now
                            extra = ""
                            # A `task` call is a whole sub-agent run happening
                            # inside one tool call. Nothing streams out of it,
                            # so warn that this gap is expected to be long.
                            if name == "task":
                                extra = "  (a sub-agent runs here — expect a long pause)"
                            print(f"  {now:7.1f}s  ask  {name}{extra}")
                    elif _text_of(msg):
                        print(f"  {now:7.1f}s  ---  the agent gave its final answer")

                elif isinstance(msg, ToolMessage):
                    name = msg.name or "?"
                    started = pending.pop(name, None)
                    took = f"  ({now - started:.1f}s)" if started is not None else ""
                    body = str(msg.content).replace("\n", " ")
                    flag = "ERR " if body.startswith("Error") else "got "
                    print(f"  {now:7.1f}s  {flag}{name}{took}")
                    if flag == "ERR ":
                        print(f"           {body[:160]}")

    total = time.perf_counter() - t0
    print(f"\n[{label}] finished in {total:.1f}s over {steps} tool-calling turns\n")
    return final or {}
