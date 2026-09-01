"""Lab 4: the built-in toolbox, on a project that fits on one screen.

The other labs research the web, which needs a Tavily key and gives you a
different answer every run. This one needs no API key and no network at all.

We hand the agent a tiny fake project in its virtual filesystem and ask for an
audit. To do the job it has to reach for most of the built-in toolbox:

    write_todos   plan the audit
    ls            see what it was given
    glob          find the Python files
    grep          find the TODO/FIXME markers
    read_file     read the files that had markers
    task          hand a file to the reviewer sub-agent
    write_file    write AUDIT.md
    edit_file     amend it afterwards

At the end we print which tools were actually called, and how often. That
tally is the point of the lab: you get to see the harness working instead of
taking my word for it.
"""

from dotenv import load_dotenv

load_dotenv()

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepagents import create_deep_agent  # noqa: E402
from deepagents.backends import StateBackend  # noqa: E402
from deepagents.backends.utils import create_file_data  # noqa: E402
from langchain.agents.middleware import TodoListMiddleware  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from local_model import build_local_model, describe  # noqa: E402
from progress import final_text, run_with_progress  # noqa: E402

# ---------------------------------------------------------------------------
# The "project". Small on purpose: you can hold all of it in your head, so
# you can check the agent's homework.
# ---------------------------------------------------------------------------
PROJECT = {
    "/app/billing.py": '''\
def total(items, discount=0):
    """Sum line items and apply a discount."""
    # TODO: refunds are ignored, negative totals are possible
    return sum(i["price"] for i in items) * (1 - discount)


def charge(customer, amount):
    # FIXME: no retry, a network blip loses the charge silently
    return gateway.post("/charge", {"id": customer, "amount": amount})
''',
    "/app/users.py": '''\
def display_name(user):
    return user.get("nickname") or user["email"].split("@")[0]


def is_admin(user):
    # TODO: role check is a string compare, should use the roles table
    return user.get("role") == "admin"
''',
    "/app/utils.py": '''\
def slugify(text):
    return text.strip().lower().replace(" ", "-")
''',
    "/README.md": "# Demo billing service\n\nA deliberately small project for Lab 4.\n",
}

AUDIT_TASK = """Audit the project in this filesystem and write /AUDIT.md.

Steps:
1. Call write_todos to plan the audit.
2. Call glob with pattern "**/*.py" to list the Python files.
3. Call grep with pattern "TODO" and then "FIXME" to find the markers.
4. For each file that has a marker, call task with subagent_type="reviewer"
   and tell it which file to review.
5. Call write_file to write /AUDIT.md with a "| File | Line | Issue |" markdown
   table of every marker, then a two-sentence summary underneath.

Reply with one short paragraph. Do not paste the report into your reply."""

REVIEWER_PROMPT = """You review exactly one file and report what is wrong with it.

1. Call read_file on the file you were told to review.
2. Reply with one bullet per problem: the line number, and what is wrong in
   plain language. Mention real bugs even if there is no TODO comment on them.

Be brief. The agent that called you only sees your final reply."""


def summarise_tool_use(messages) -> None:
    """Print which tools the agent called, in order and as a tally."""
    calls = [
        c["name"]
        for m in messages
        if isinstance(m, AIMessage)
        for c in (m.tool_calls or [])
    ]
    errors = [
        m.name
        for m in messages
        if isinstance(m, ToolMessage) and str(m.content).startswith("Error")
    ]

    print(f"\n[the agent made {len(calls)} tool calls]")
    print("  order: " + " -> ".join(calls))
    for name, n in Counter(calls).most_common():
        print(f"  {name:<12} x{n}")
    if errors:
        print(f"  (errors from: {', '.join(sorted(set(errors)))})")


def main() -> None:
    model = build_local_model()
    print(f"[toolbox agent] {describe(model)}")
    print(f"[project: {len(PROJECT)} files, no network needed]\n")

    reviewer = {
        "name": "reviewer",
        "description": (
            "Reviews a single source file and reports its problems. Use one "
            "call per file that needs reviewing."
        ),
        "system_prompt": REVIEWER_PROMPT,
        # No tools key: the reviewer inherits the file tools it needs.
        # No model key: it reuses the model already loaded in VRAM.
    }

    agent = create_deep_agent(
        model=model,
        tools=[],  # no custom tools at all — this lab is the built-ins only
        system_prompt=(
            "You are a meticulous code auditor. Work through your plan one "
            "step at a time and use the tools rather than guessing at file "
            "contents."
        ),
        subagents=[reviewer],
        backend=StateBackend(),
        middleware=[TodoListMiddleware()],
    )

    # Seed the virtual filesystem. Build the values with create_file_data() —
    # a bare {"content": ...} dict is missing the timestamps that `glob` sorts
    # on, and glob then fails with a cryptic KeyError while every other tool
    # keeps working.
    result = run_with_progress(
        agent,
        {
            "messages": [{"role": "user", "content": AUDIT_TASK}],
            "files": {p: create_file_data(c) for p, c in PROJECT.items()},
        },
        label="toolbox agent",
    )

    # Not messages[-1].content: a local model often ends on an empty turn.
    print(final_text(result["messages"]) or
          "[the model ended with an empty message — the work it did is below]")
    summarise_tool_use(result["messages"])

    audit = result.get("files", {}).get("/AUDIT.md") or result.get("files", {}).get(
        "AUDIT.md"
    )
    if audit:
        print(f"\n===== AUDIT.md =====\n{audit['content']}")
    else:
        print("\n[no AUDIT.md was written — see the tool calls above]")


if __name__ == "__main__":
    main()
