# Deep Agents Tutorial

Companion repository for the blog post **"Building Deep Agents in Python: From Shallow Loops to Autonomous Systems"** by Ruslan Magana Vsevolodovna — [ruslanmv.com](https://ruslanmv.com).

Build a deep agent — explicit planning, virtual filesystem, sub-agent delegation — with the [`deepagents`](https://github.com/langchain-ai/deepagents) library, and trace it with Langfuse.

## Contents

| File | Blog Section |
|---|---|
| `src/01_shallow_agent.py` | The classic tool loop and where it breaks |
| `src/02_deep_agent.py` | Full deep agent: planning + files + researcher sub-agent |
| `src/03_observed_agent.py` | Tracing every step with the Langfuse callback |
| `2026-08-17-Building-Deep-Agents-in-Python.md` | The blog post (Jekyll / Minimal Mistakes, drop into `_posts/`) |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY + TAVILY_API_KEY (+ Langfuse keys for 03)
python src/02_deep_agent.py
```

## Requirements

- Python 3.11+
- Anthropic (or OpenAI) API key
- Tavily API key (free tier) for web search

Note: deep agent runs make many LLM calls (planning + sub-agents) — expect higher cost than a single completion.
