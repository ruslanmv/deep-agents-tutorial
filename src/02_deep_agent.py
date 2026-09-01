"""Step 2: A deep agent — planning, virtual filesystem, sub-agents — on Ollama.

`create_deep_agent()` does not build a special graph. It builds the same
model-tools loop as step 1 and wraps it in middleware that supplies file tools,
a `task` tool for delegation, and automatic context compaction.

Two things here are specific to running locally, and both are load-bearing:

* `TodoListMiddleware` is passed explicitly. As of `deepagents` 0.7.x the
  planning tool is NOT in the default stack for non-OpenAI-harness models, so
  without this line the agent has no `write_todos` and a system prompt telling
  it to plan describes a tool that does not exist.
* The orchestrator and the sub-agent deliberately share one model instance.
  `deepagents` lets each sub-agent name its own model, but on a single 16 GB
  GPU two models means Ollama evicting and reloading weights on every
  delegation. One resident model is dramatically faster.
"""

from dotenv import load_dotenv

load_dotenv()

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepagents import create_deep_agent  # noqa: E402
from deepagents.backends import StateBackend  # noqa: E402
from langchain.agents.middleware import TodoListMiddleware  # noqa: E402

from local_model import build_local_model, describe  # noqa: E402
from progress import final_text, run_with_progress  # noqa: E402
from search import internet_search, require_key  # noqa: E402

TASK = (
    "Research the top 3 open-source LLM inference engines in 2026, "
    "compare their strengths, and write a short report to /report.md."
)

# Local models follow short, concrete, imperative instructions far better than
# the discursive prompts that work on frontier models. Every sentence here is
# an instruction, and the file names are spelled out.
ORCHESTRATOR_PROMPT = """You are an expert research orchestrator.

Every path in this filesystem is absolute and starts with "/". Use exactly the
paths given below — do not invent a directory.

Follow this procedure exactly:
1. Call write_todos with one todo per engine, plus a final "synthesise" todo.
2. Call task with subagent_type="researcher" exactly THREE times, once per
   engine. Tell each researcher its engine and the exact file to write:
   /notes_1.md for the first, /notes_2.md for the second, /notes_3.md for the third.
3. Call read_file on /notes_1.md, /notes_2.md and /notes_3.md.
4. Call write_file to write /report.md, using what you just read.
5. Reply with a three-sentence summary. Do not paste the report into your reply.

Never run a web search yourself — always delegate research to the researcher."""

RESEARCHER_PROMPT = """You are a focused researcher investigating exactly one question.

1. Call internet_search one to three times for the question you were given.
2. Call write_file to save your findings. Use the EXACT absolute path you were
   given — it starts with "/". Write short markdown bullets with source URLs.
3. Reply with a compact summary of your findings.

Writing the file is not optional: the agent that called you will read it back.
It only sees your final reply, never your searches."""


def main() -> None:
    # Before the first model call. A deep agent only reaches internet_search
    # inside a sub-agent, so without this the failure surfaces late and deep.
    require_key()

    # One model instance, shared by the orchestrator and the sub-agent.
    model = build_local_model()
    backend = StateBackend()
    print(f"[deep agent] {describe(model)}\n")

    research_subagent = {
        "name": "researcher",
        "description": (
            "Investigates a single focused research question and writes its "
            "findings to a file. Use for every web-research sub-question."
        ),
        "system_prompt": RESEARCHER_PROMPT,
        "tools": [internet_search],
        # No "model" key: the sub-agent inherits the parent's instance.
    }

    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=[research_subagent],
        backend=backend,
        # Restores write_todos. See the module docstring.
        middleware=[TodoListMiddleware()],
    )

    result = run_with_progress(
        agent, {"messages": [{"role": "user", "content": TASK}]}, label="deep agent"
    )

    # Not messages[-1].content: a local model often ends on an empty turn.
    print(final_text(result["messages"]) or
          "[the model ended with an empty message — the work it did is below]")

    # Inspect the virtual filesystem the agent used as working memory.
    # `files` maps a path to a FileData dict — {"content", "encoding",
    # "created_at", "modified_at"} — not to a bare string.
    files = result.get("files", {})
    print(f"\n[virtual filesystem: {len(files)} file(s)]")
    for path, data in sorted(files.items()):
        content = data["content"] if isinstance(data, dict) else str(data)
        print(f"\n===== {path} ({len(content)} chars) =====\n{content[:500]}")

    # The orchestrator's own context stayed small even though the sub-agents
    # ran many searches. That gap is the entire point of a deep agent.
    print(f"\n[orchestrator messages: {len(result['messages'])}]")


if __name__ == "__main__":
    main()
