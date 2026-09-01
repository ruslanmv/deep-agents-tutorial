# Team talk: introducing deep agents

`deep-agents-intro.pptx` — three slides, built for a 15-minute slot with questions at the end.
Presenter: Ruslan Magana Vsevolodovna.

## Timing

| Slide | Minutes | Job |
|---|---|---|
| 1 — Deep Agents | ~4 | The problem everyone has already felt. Short tasks work, long ones quietly fail. |
| 2 — The same task, two architectures | ~7 | The heart of the talk. ReAct on the left, deep agent on the right. |
| 3 — What this means for us | ~4 | When to reach for it, what transfers, and the ask. |

Every slide has speaker notes with the points to land and the honest caveats — open the
notes pane in PowerPoint, or `markitdown deep-agents-intro.pptx` to read them as text.

## The argument

The talk deliberately does **not** pitch deep agents as a default. It says:

1. Below roughly ten steps, a plain loop is simpler, cheaper and easier to debug.
2. The transferable skill is context discipline — notes to files, messy work to
   sub-agents, narrow before you read. That helps any agent we build.
3. It can be evaluated on our own hardware first: a 12 GB laptop GPU, no API key,
   no per-token bill.

Leading with the limitation is what buys credibility for the cases where it genuinely helps.

## If someone asks

- **"Isn't this just RAG / prompt engineering?"** No — the state lives *outside* the
  context window. That's the whole mechanism.
- **"What does it cost?"** More than a single completion: planning plus sub-agents means
  many more model calls, so runs are slower. Locally you pay in time, not tokens.
- **"Is it reliable?"** On very long tasks, not yet. Keep a human in the loop for
  anything consequential.

## Backing material

The runnable walkthrough is the repository this folder sits in — see the root `README.md`.
`make toolbox` is the quickest live demo if anyone wants to see an agent actually work:
no API key, no network, and it prints the tools the agent chose.
