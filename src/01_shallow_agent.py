"""Step 1: A shallow agent — the classic reactive tool loop, running locally.

Works for short tasks; degrades on long multi-step work because all state
lives in the context window. On a local model that ceiling arrives far sooner
than it does in the cloud: 32k tokens, not 200k.
"""

from dotenv import load_dotenv

load_dotenv()

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain.agents import create_agent  # noqa: E402

from local_model import build_local_model, describe  # noqa: E402
from search import internet_search, require_key  # noqa: E402


def main() -> None:
    # Before the first model call, not several tool calls into the run.
    require_key()
    model = build_local_model()
    print(f"[shallow agent] {describe(model)}\n")

    agent = create_agent(
        model=model,
        tools=[internet_search],
        system_prompt="You are a helpful research assistant.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is PagedAttention in vLLM?"}]}
    )
    print(result["messages"][-1].content)

    # The whole run lives here. Watch this number when you move on to the
    # multi-part task in the blog post: every raw tool result is still in it.
    print(f"\n[messages in context: {len(result['messages'])}]")


if __name__ == "__main__":
    main()
