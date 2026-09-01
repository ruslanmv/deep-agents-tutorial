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

# Past this, a single orchestrator turn is not "a local model being slow", it is
# a model that did not fit on the GPU. The reference run does a whole turn in
# about three seconds.
_SLOW_TURN_SECONDS = 25.0


def _residency_warning() -> str:
    """A diagnosis, if the loaded model is not actually on the GPU.

    Called once, right after the first model call, because that is the moment
    the weights are resident and `/api/ps` can be believed. Returns "" when
    everything is fine, when there is no GPU to speak of, or when we cannot
    tell — this is a hint, never a blocker.
    """
    import os  # noqa: PLC0415

    if os.environ.get("DEEP_AGENT_BACKEND", "ollama").strip().lower() not in {
        "",
        "ollama",
    }:
        return ""  # a gateway backend has no local /api/ps to ask
    try:
        from local_model import gpu_residency  # noqa: PLC0415
    except ImportError:
        return ""

    info = gpu_residency()
    if not info or info["pct"] >= 99:
        return ""

    gib = 1024**3
    pct = info["pct"]
    where = (
        "entirely on the CPU"
        if pct <= 1
        else f"only {pct:.0f}% on the GPU — the rest is on the CPU"
    )
    ctx = os.environ.get("DEEP_AGENT_NUM_CTX", "24576")
    try:
        smaller = max(4096, int(ctx) // 2)
    except ValueError:
        smaller = 12288
    return (
        f"\n  [slow] {info['name']} is {where} "
        f"({info['on_gpu'] / gib:.1f} of {info['total'] / gib:.1f} GiB in VRAM).\n"
        f"         That is why this is crawling — expect minutes per step, not "
        f"seconds.\n"
        f"         Cheapest fixes first, in .env or in front of the command:\n"
        f"           {f'DEEP_AGENT_NUM_CTX={smaller}':<28}"
        f"# smaller KV cache, frees VRAM\n"
        f"           {'DEEP_AGENT_MODEL=qwen3:4b':<28}"
        f"# smaller weights (make model MODEL=qwen3:4b)\n"
        f"           {'OLLAMA_KV_CACHE_TYPE=q8_0':<28}"
        f"# on the Ollama server — halves the cache\n"
        f"         Close whatever else is using VRAM, then `make doctor` "
        f"(check 6b) to confirm.\n"
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
    pending: dict[tuple, float] = {}   # (scope, tool name) -> when it was asked for
    steps = 0
    sub_steps = 0            # model turns that happened inside sub-agents
    diagnosed = False        # the residency hint is printed at most once
    last_reply_at = 0.0      # when the previous model turn came back

    # subgraphs=True is what makes a `task` call stop being a black box. Without
    # it a sub-agent is one tool call that returns minutes later having told you
    # nothing — which is exactly the run you cannot debug.
    stream = agent.stream(
        payload, stream_mode=["updates", "values"], subgraphs=True
    )
    for namespace, mode, chunk in stream:
        inside = bool(namespace)
        if mode == "values":
            # Only the outer graph's state is the run's result; a sub-agent's
            # values would otherwise overwrite it with its own scratch state.
            if not inside:
                final = chunk
            continue

        # Indent anything happening inside a sub-agent, and keep its pending
        # tool calls in their own namespace so timings do not cross over.
        scope = namespace[0].split(":")[0] if inside else ""
        mark = "  ↳ " if inside else ""

        for node, update in (chunk or {}).items():
            for msg in (update or {}).get("messages") or []:
                now = time.perf_counter() - t0

                if isinstance(msg, AIMessage):
                    # The first reply means the weights are loaded, so this is
                    # the earliest moment /api/ps can be believed — and the
                    # earliest we can tell someone their run will take ten
                    # minutes instead of one.
                    first = not diagnosed
                    if first:
                        diagnosed = True
                        warning = _residency_warning()
                        if warning:
                            print(warning)

                    # How long the model spent on this turn. Skipped for the
                    # first one, which legitimately includes the weight load.
                    think = now - last_reply_at
                    slow = ""
                    if not first and think > _SLOW_TURN_SECONDS:
                        slow = f"   [{think:.0f}s on this turn — not normal]"
                    last_reply_at = now

                    calls = _tool_names(msg)
                    if calls:
                        if inside:
                            sub_steps += 1
                        else:
                            steps += 1
                        for i, name in enumerate(calls):
                            pending[(scope, name)] = now
                            extra = ""
                            # A `task` call is a whole sub-agent run happening
                            # inside one tool call. Its own turns are printed
                            # indented underneath, as they happen.
                            if name == "task":
                                extra = "  (a sub-agent runs here)"
                            # The timing note belongs to the turn, not to each
                            # tool it asked for.
                            print(f"  {now:7.1f}s  {mark}ask  {name}{extra}"
                                  f"{slow if i == 0 else ''}")
                    elif _text_of(msg):
                        what = ("the sub-agent reported back" if inside
                                else "the agent gave its final answer")
                        print(f"  {now:7.1f}s  {mark}---  {what}{slow}")

                elif isinstance(msg, ToolMessage):
                    name = msg.name or "?"
                    started = pending.pop((scope, name), None)
                    took = f"  ({now - started:.1f}s)" if started is not None else ""
                    body = str(msg.content).replace("\n", " ")
                    flag = "ERR " if body.startswith("Error") else "got "
                    print(f"  {now:7.1f}s  {mark}{flag}{name}{took}")
                    if flag == "ERR ":
                        print(f"           {body[:160]}")

    total = time.perf_counter() - t0
    inner = f", plus {sub_steps} inside sub-agents" if sub_steps else ""
    print(f"\n[{label}] finished in {total:.1f}s over {steps} tool-calling "
          f"turns{inner}\n")
    return final or {}
