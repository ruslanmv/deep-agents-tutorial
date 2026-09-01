"""The one hosted dependency left in the stack: web search.

The model runs on your GPU, but the agent still needs to reach the internet.
Tavily's free tier covers this tutorial. The tool returns a trimmed dict rather
than Tavily's full payload — on a 32k local window, raw search JSON is the
single fastest way to fill the context you are trying to conserve.
"""

from __future__ import annotations

import os
from typing import Literal

from tavily import TavilyClient

import keys

_client: TavilyClient | None = None

SERVICE = "Tavily"
URL = "https://tavily.com"


def require_key() -> None:
    """Check the key at startup, before the agent makes its first model call.

    Without this the first `internet_search` fails several model calls into the
    run, and a one-line `.env` mistake arrives as a traceback from inside the
    agent loop.
    """
    keys.require("TAVILY_API_KEY", service=SERVICE, url=URL)


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if keys.missing("TAVILY_API_KEY"):
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Copy .env.example to .env and add a "
                f"free key from {URL} — the LLM is local, but web search is not."
            )
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"].strip())
    return _client


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> dict:
    """Run a web search and return titled results with short content snippets."""
    raw = _get_client().search(query, max_results=max_results, topic=topic)
    # Keep title/url/content only. Dropping Tavily's scores, raw HTML and
    # images cuts the tokens per search by roughly an order of magnitude.
    return {
        "query": query,
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:1200],
            }
            for r in raw.get("results", [])
        ],
    }
