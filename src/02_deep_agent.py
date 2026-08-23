"""
Step 2: A deep agent — planning, virtual filesystem, sub-agents.
create_deep_agent() bundles the harness: todo planning, file tools
(ls, read_file, write_file, edit_file), and a `task` tool that
delegates work to sub-agents with isolated context.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> dict:
    """Run a web search."""
    return tavily.search(query, max_results=max_results, topic=topic)


research_subagent = {
    "name": "researcher",
    "description": "Delegates focused research on a single sub-question.",
    "system_prompt": (
        "You are a focused researcher. Investigate exactly one question, "
        "search the web, and write your findings to a file."
    ),
    "tools": [internet_search],
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[internet_search],
    system_prompt=(
        "You are an expert research agent. Plan your work with the todo tool, "
        "delegate sub-questions to the researcher sub-agent, save intermediate "
        "notes to files, and write the final answer to `report.md`."
    ),
    subagents=[research_subagent],
)

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Research the top 3 open-source LLM inference engines in 2026, "
                "compare their strengths, and write a short report to report.md."
            ),
        }]
    })

    # Print the agent's final message
    print(result["messages"][-1].content)

    # Inspect the virtual filesystem the agent used as working memory
    for path, content in result.get("files", {}).items():
        print(f"\n===== {path} =====\n{content[:500]}")
