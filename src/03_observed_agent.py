"""
Step 3: Observing the deep agent with Langfuse (bridge to Blog 1).
One callback handler traces every LLM call, tool call and sub-agent run.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from tavily import TavilyClient
from deepagents import create_deep_agent
from langfuse.langchain import CallbackHandler

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(query: str, max_results: int = 3) -> dict:
    """Run a web search."""
    return tavily.search(query, max_results=max_results)


agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[internet_search],
    system_prompt="You are a research agent. Plan, research, then answer.",
)

if __name__ == "__main__":
    handler = CallbackHandler()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Summarize what deep agents are in 3 bullets."}]},
        config={"callbacks": [handler]},
    )
    print(result["messages"][-1].content)
