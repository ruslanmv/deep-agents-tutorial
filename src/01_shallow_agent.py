"""
Step 1: A shallow agent — the classic reactive tool loop.
Works for short tasks; degrades on long multi-step work because
all state lives in the context window.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from tavily import TavilyClient
from langchain.agents import create_agent

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(query: str, max_results: int = 3) -> dict:
    """Run a web search and return results."""
    return tavily.search(query, max_results=max_results)


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[internet_search],
    system_prompt="You are a helpful research assistant.",
)

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What is PagedAttention in vLLM?"}]
    })
    print(result["messages"][-1].content)
