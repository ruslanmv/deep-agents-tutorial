"""Step 3: Observing the local deep agent with Langfuse.

One callback handler traces every LLM call, tool call and sub-agent run. The
`task` tool forwards nothing explicitly — LangGraph seeds each sub-run from the
ambient parent config, so the handler reaches sub-agents on its own and their
work shows up as nested spans.

With a local model this trace is more useful than in the cloud, not less: it is
how you see that a step was slow because Ollama reloaded the weights, or that
the model looped because its plan scrolled out of a too-small context window.
"""

from dotenv import load_dotenv

load_dotenv()

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepagents import create_deep_agent  # noqa: E402
from langchain.agents.middleware import TodoListMiddleware  # noqa: E402

import keys  # noqa: E402
from local_model import build_local_model, describe  # noqa: E402
from search import internet_search, require_key  # noqa: E402


def main() -> None:
    # `.env.example` ships `pk-lf-...` / `sk-lf-...`, and a bare emptiness test
    # accepts those — so anyone who ran `make env` without editing .env got a
    # full run that died on an opaque "Failed to export span batch code: 403".
    # keys.require() treats a placeholder as absent.
    keys.require(
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        service="Langfuse",
        url="https://cloud.langfuse.com",
    )
    # This lab researches the web too, so it needs the search key as well.
    require_key()

    from langfuse.langchain import CallbackHandler

    model = build_local_model()
    print(f"[observed agent] {describe(model)}\n")

    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=(
            "You are a research agent. Plan with write_todos, research, "
            "then answer concisely."
        ),
        middleware=[TodoListMiddleware()],
    )

    handler = CallbackHandler()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize what deep agents are in 3 bullets.",
                }
            ]
        },
        config={"callbacks": [handler]},
    )
    print(result["messages"][-1].content)
    print(
        f"\n[trace sent to {os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')}]"
    )


if __name__ == "__main__":
    main()
