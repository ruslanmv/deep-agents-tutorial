"""Lab 5: the same model, reached through an OllaBridge gateway.

[OllaBridge](https://github.com/ruslanmv/ollabridge) puts one OpenAI-compatible
URL in front of everything you can reach — the Ollama on this machine, a GPU box
under your desk, your own OpenAI or Anthropic keys. Your code stops caring where
the model lives.

A deep agent is nothing but tool calls, so the one thing that matters is whether
those survive the trip. Current OllaBridge carries them; an OpenAI-compatible
proxy that does not will accept your `tools` field, ignore it, and answer in
prose with nothing in the logs. This script measures rather than assumes:

  1. checks the gateway is up and lists what it publishes
  2. runs a plain chat turn through it
  3. probes whether tool calling survives the trip
  4. if it does, runs a real deep agent through the gateway and prints the
     tools it used; if it doesn't, says exactly what failed and why

Written this way it works as a compatibility check for any OpenAI-compatible
gateway, not just this one.

Nothing to configure. Start the gateway in the mode that trusts this machine,
then run the lab:

    ollabridge start --auth-mode local-trust --host 127.0.0.1
    make ollabridge

`local-trust` skips the key check for loopback callers, so there is no `.env`
entry and no key to copy. Set `OLLABRIDGE_URL` or `OLLABRIDGE_API_KEY` only if
your gateway is elsewhere or was started with `--auth-mode required`.
"""

from dotenv import load_dotenv

load_dotenv()

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_model import (  # noqa: E402
    DEFAULT_MODEL,
    build_ollabridge_model,
    describe,
    ollabridge_api_key,
)

GATEWAY = os.environ.get("OLLABRIDGE_URL", "http://localhost:11435/v1").rstrip("/")
# OllaBridge's default auth mode is `local-trust`, which waves through anything
# arriving over loopback — so there is nothing to configure. This falls back to
# a stand-in when no key is set.
API_KEY = ollabridge_api_key()
MODEL = os.environ.get("DEEP_AGENT_MODEL", DEFAULT_MODEL)


def ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def bad(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")


def note(msg: str) -> None:
    print(f"  \033[33mNOTE\033[0m  {msg}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def step_1_gateway() -> bool:
    """Is the gateway up, and does it publish our model?"""
    section("1. Gateway")
    req = urllib.request.Request(
        f"{GATEWAY}/models", headers={"Authorization": f"Bearer {API_KEY}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            bad(f"{GATEWAY}/models returned HTTP {exc.code} — the key was checked.")
            note(
                "So this gateway is not in `local-trust` mode, or you are not "
                "reaching it over loopback. Easiest fix: restart it with "
                "`ollabridge start --auth-mode local-trust --host 127.0.0.1`. "
                "Otherwise put the key OllaBridge printed at startup into "
                "OLLABRIDGE_API_KEY."
            )
        else:
            bad(f"{GATEWAY}/models returned HTTP {exc.code}.")
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        bad(
            f"cannot reach {GATEWAY} ({exc}). Start it with "
            "`ollabridge start --auth-mode local-trust --host 127.0.0.1`, or "
            "point OLLABRIDGE_URL at wherever yours is listening."
        )
        return False

    published = [m.get("id", "?") for m in body.get("data", [])]
    ok(f"reachable — publishes {len(published)} model(s): {', '.join(published) or 'none'}")
    if API_KEY == "local-trust":
        ok("no API key needed — the gateway trusted this loopback request")

    if MODEL in published:
        ok(f"{MODEL} is published through the gateway")
    else:
        note(
            f"{MODEL} is not in that list. OllaBridge only serves what you have "
            "switched on per model in its dashboard, and it may drop the ':tag' "
            "part of the name. Adjust DEEP_AGENT_MODEL to match a name above."
        )
    return True


def step_2_chat() -> bool:
    """A plain chat turn. This is what OllaBridge is built for."""
    section("2. Plain chat through the gateway")
    try:
        model = build_ollabridge_model()
        print(f"        {describe(model)}")
        reply = model.invoke("Reply with exactly the word: bridged")
    except Exception as exc:  # noqa: BLE001 — report whatever comes back
        bad(f"{type(exc).__name__}: {exc}")
        return False

    text = (reply.text if callable(getattr(reply, "text", None)) else None) or str(
        reply.content
    )
    ok(f"the model answered through OllaBridge: {text.strip()[:60]!r}")
    return True


def step_3_tool_probe() -> bool:
    """The question that decides whether a deep agent can use this path."""
    section("3. Does tool calling survive the gateway?")

    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"sunny in {city}"

    try:
        model = build_ollabridge_model()
        reply = model.bind_tools([get_weather]).invoke(
            "What is the weather in Copenhagen? Use the tool."
        )
    except Exception as exc:  # noqa: BLE001
        bad(f"the request was rejected outright — {type(exc).__name__}: {exc}")
        note(
            "An assistant turn carrying tool_calls has content=None. A gateway "
            "whose message schema requires a string rejects it outright."
        )
        return False

    calls = getattr(reply, "tool_calls", None) or []
    if calls:
        ok(f"tool calls come back: {calls[0].get('name')}({calls[0].get('args')})")
        return True

    bad("no tool call came back — the model answered in prose instead")
    note(
        "The gateway accepted the request and dropped the tool definitions, so "
        "the model was never told the tools existed. That is a missing feature "
        "in the gateway, not a misconfiguration here."
    )
    return False


def step_4_deep_agent() -> bool:
    """Reached once step 3 confirms tool calls survive the gateway."""
    section("4. A deep agent, through the gateway")

    from collections import Counter

    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.backends.utils import create_file_data
    from langchain.agents.middleware import TodoListMiddleware
    from langchain_core.messages import AIMessage

    agent = create_deep_agent(
        model=build_ollabridge_model(),
        tools=[],
        system_prompt=(
            "You are a meticulous auditor. Use the tools rather than guessing."
        ),
        backend=StateBackend(),
        middleware=[TodoListMiddleware()],
    )
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call write_todos with a one-item plan, then grep for "
                            "TODO, then write what you found to /FINDINGS.md."
                        ),
                    }
                ],
                "files": {
                    "/app/billing.py": create_file_data(
                        "def total(x):\n    # TODO: handle refunds\n    return x\n"
                    )
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 — the failure is the finding
        bad(f"the agent loop broke — {type(exc).__name__}: {str(exc)[:200]}")
        if "valid string" in str(exc) or "422" in str(exc):
            note(
                "A gateway can forward `tools` and still fail here: the second "
                "turn sends the assistant's own tool_calls back, and that "
                "message has content=None. Carrying tool calls needs both the "
                "`tools` field on the way in AND a message schema that allows "
                "null content plus tool_calls/tool_call_id."
            )
        return False

    calls = [
        c["name"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for c in (m.tool_calls or [])
    ]
    if not calls:
        bad("the agent made no tool calls")
        return False
    ok(f"{len(calls)} tool calls through the gateway")
    print("        order: " + " -> ".join(calls))
    for name, n in Counter(calls).most_common():
        print(f"        {name:<12} x{n}")
    return True


if __name__ == "__main__":
    print(f"OllaBridge demo — gateway={GATEWAY} model={MODEL}")

    if not step_1_gateway():
        sys.exit(1)

    chat_ok = step_2_chat()
    tools_ok = step_3_tool_probe()

    if tools_ok:
        agent_ok = step_4_deep_agent()
    else:
        section("4. A deep agent, through the gateway")
        note("skipped — step 3 says this version cannot carry tool calls.")
        note(
            "Upgrade the gateway to a version that forwards tool calls, then "
            "re-run this script."
        )
        note(
            "Meanwhile keep the agent labs on the direct backend: set "
            "DEEP_AGENT_BACKEND=ollama (or leave it unset) and run `make deep`."
        )
        agent_ok = False

    print("\n" + "=" * 68)
    print(f"  chat through OllaBridge     {'yes' if chat_ok else 'no'}")
    print(f"  tool calls through OllaBridge {'yes' if tools_ok else 'no'}")
    print(f"  deep agent through OllaBridge {'yes' if agent_ok else 'no'}")
    if chat_ok and not tools_ok:
        print(
            "\n  Verdict: this gateway serves the model fine for chat, but does\n"
            "  not forward tool calls, which is all a deep agent does. Upgrade\n"
            "  the gateway, or keep the agent on the direct Ollama backend."
        )
    elif chat_ok and tools_ok and agent_ok:
        print(
            "\n  Verdict: this gateway carries a full deep agent. You can point\n"
            "  every lab at it with DEEP_AGENT_BACKEND=ollabridge."
        )
    print("=" * 68)
    # Chat working is the bar for a passing run today; tool support is reported,
    # not required, so this stays usable in CI as OllaBridge evolves.
    sys.exit(0 if chat_ok else 1)
