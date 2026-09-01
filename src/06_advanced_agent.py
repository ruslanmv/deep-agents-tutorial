"""Lab 6: middleware and harness settings — the two ways to change the harness.

Everything in the earlier labs took `create_deep_agent()` as given. This one
opens it up. There are two levers, and the difference between them is the
lesson:

* **Middleware** wraps individual calls. It sees each model call and each tool
  call as it happens, and can watch, rewrite, or refuse it. Use it for
  behaviour: auditing, guardrails, retries, injecting context.

* **Harness settings** (a `HarnessProfile`) are declarative defaults attached
  to a *model*. They change what the agent is even offered — its system prompt
  suffix, which tools it can see, how those tools are described. Use them for
  policy that should hold everywhere that model is used.

Rule of thumb: if it's "watch or intervene per call", write middleware. If it's
"this model should never see that tool", write a profile.

Run it:

    make advanced
"""

from dotenv import load_dotenv

load_dotenv()

import os
import sys
import time
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepagents import (  # noqa: E402
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import StateBackend  # noqa: E402
from deepagents.backends.utils import create_file_data  # noqa: E402
from langchain.agents.middleware import TodoListMiddleware  # noqa: E402
from langchain.agents.middleware.types import AgentMiddleware  # noqa: E402

from local_model import build_local_model, describe  # noqa: E402
from progress import final_text, run_with_progress  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Middleware: watch every model call, and police every tool call.
# ---------------------------------------------------------------------------


class ToolAuditMiddleware(AgentMiddleware):
    """Records what the model was offered and what it did, and blocks a deny-list.

    Two hooks, which between them cover most of what middleware is for:

    `wrap_model_call` sits around each model call. `request.tools` is the tool
    list the model is about to be shown, so this is where you observe or edit
    what the agent can see this turn.

    `wrap_tool_call` sits around each tool execution. Call `handler(request)`
    to let it through; return a `ToolMessage` yourself to refuse it. That makes
    it the natural place for a guardrail — the model asked, and you said no.
    """

    # `name` is how deepagents identifies middleware in the stack. Reuse an
    # existing name and yours replaces that entry instead of being appended.
    name = "ToolAuditMiddleware"

    def __init__(self, *, blocked: frozenset[str] = frozenset()) -> None:
        super().__init__()
        self.blocked = blocked
        self.offered: list[list[str]] = []
        self.calls: list[tuple[str, float]] = []
        self.refused: list[str] = []
        self.grep_description = ""

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        names = []
        for t in request.tools or []:
            name = t.name if hasattr(t, "name") else str(t.get("name", "?"))
            names.append(name)
            # Proof the profile's description override reached the request.
            if name == "grep" and not self.grep_description:
                self.grep_description = getattr(t, "description", "") or ""
        self.offered.append(sorted(names))
        return handler(request)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        name = request.tool_call.get("name", "?")
        if name in self.blocked:
            self.refused.append(name)
            from langchain_core.messages import ToolMessage

            return ToolMessage(
                content=(
                    f"Refused: '{name}' is blocked by policy. Do not try it again; "
                    f"work with the other tools."
                ),
                tool_call_id=request.tool_call["id"],
                name=name,
            )
        started = time.perf_counter()
        result = handler(request)
        self.calls.append((name, (time.perf_counter() - started) * 1000))
        return result


# ---------------------------------------------------------------------------
# 2. Harness settings: declarative defaults attached to the model.
# ---------------------------------------------------------------------------

HOUSE_RULES = """<house_rules>
Prefer glob and grep to narrow before you read a file.
Never claim a file's contents without reading it first.
</house_rules>"""


def profile_key() -> str:
    """The key a profile must be filed under to match the model we build.

    Profiles are looked up by model. When you pass a *string* spec to
    `create_deep_agent`, that string is the key. When you pass a pre-built
    instance — which is what `build_local_model()` does — deepagents derives the
    key from the model itself, and falls back to the provider name.

    That provider changes with the backend, which is easy to miss: the direct
    path builds a `ChatOllama` whose provider is `"ollama"`, but
    `DEEP_AGENT_BACKEND=ollabridge` builds a `ChatOpenAI` whose provider is
    `"openai"` — the gateway speaks the OpenAI API, so that is what it is. Key
    it wrong and nothing fails; deepagents logs `No harness profile matched
    pre-built model …` and quietly uses defaults, so the tool exclusions and the
    prompt suffix below simply never take effect.
    """
    backend = os.environ.get("DEEP_AGENT_BACKEND", "ollama").strip().lower()
    return "openai" if backend == "ollabridge" else "ollama"


def register_house_profile() -> str:
    """Attach a profile to our model and return the key it was filed under."""
    key = profile_key()
    register_harness_profile(
        key,
        HarnessProfile(
            # Appended after your own system_prompt: USER -> BASE -> SUFFIX.
            system_prompt_suffix=HOUSE_RULES,
            # Withheld from the model. Note this hides them at model-call time
            # rather than unregistering the handlers.
            excluded_tools=frozenset({"execute", "delete"}),
            # Small models take tool descriptions literally, so say the thing
            # that actually trips them up.
            tool_description_overrides={
                "grep": (
                    "Search file contents for LITERAL text (not a regex). "
                    "Use this to find which files matter before reading any."
                ),
            },
            # We give this agent a named sub-agent below, so the catch-all
            # general-purpose one is just another way for it to go wandering.
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    return key


TASK = """Audit the project and write /FINDINGS.md.

1. Call write_todos with a short plan.
2. Call grep for "TODO".
3. Call delete on /app/legacy.py — this SHOULD be refused, keep going afterwards.
4. Call write_file to write /FINDINGS.md with what you found.

Reply with one sentence."""

PROJECT = {
    "/app/billing.py": "def total(x):\n    # TODO: refunds ignored\n    return x\n",
    "/app/legacy.py": "# dead code nobody deleted\n",
}


def main() -> None:
    key = register_house_profile()
    model = build_local_model()
    audit = ToolAuditMiddleware(blocked=frozenset({"delete"}))

    print(f"[advanced agent] {describe(model)}")
    print(f"[harness profile registered under {key!r}]\n")

    agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a careful auditor.",
        backend=StateBackend(),
        # User middleware is spliced in after the built-in core stack and
        # before the tail, so this sees every model and tool call.
        middleware=[TodoListMiddleware(), audit],
    )

    result = run_with_progress(
        agent,
        {
            "messages": [{"role": "user", "content": TASK}],
            "files": {p: create_file_data(c) for p, c in PROJECT.items()},
        },
        label="advanced agent",
    )

    # Not messages[-1].content: a local model often ends on an empty turn.
    print(final_text(result["messages"]) or
          "[the model ended with an empty message — the work it did is below]")

    print("\n--- what the harness settings did ---")
    offered = audit.offered[0] if audit.offered else []
    print(f"  tools in the request when it passed our middleware: {len(offered)}")
    print(f"    {', '.join(offered)}")
    print(f"  'task' present: {'task' in offered}  <- general_purpose_subagent(enabled=False)")
    print(f"  grep description: {audit.grep_description[:64]!r}")
    print(
        "  note: 'delete' is still in that list. excluded_tools is enforced by\n"
        "  a middleware the profile appends AFTER ours, i.e. closer to the\n"
        "  model — so we observe the request before it is filtered. Where you\n"
        "  sit in the stack decides what you see."
    )

    print("\n--- what the middleware saw ---")
    for name, ms in audit.calls:
        print(f"  {name:<12} {ms:6.1f} ms")
    if audit.refused:
        print(f"  refused by the guard: {', '.join(audit.refused)}")
    else:
        print("  the guard was never triggered (the model never asked for delete)")

    findings = result.get("files", {}).get("/FINDINGS.md")
    if findings:
        print(f"\n===== /FINDINGS.md =====\n{findings['content'][:400]}")


if __name__ == "__main__":
    main()
