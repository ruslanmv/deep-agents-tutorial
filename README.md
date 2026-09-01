# Deep Agents on Your Own GPU

Companion code for the blog post **"Deep Agents on Your Own GPU: A Hands-On Lab"** by Ruslan Magana Vsevolodovna — [ruslanmv.com](https://ruslanmv.com).

Build an agent that plans, takes notes and delegates — the architecture behind tools like Claude Code — running entirely on your own graphics card with [Ollama](https://ollama.com) and the [`deepagents`](https://github.com/langchain-ai/deepagents) library.

**No API key for the model.** A free Tavily key is the only cloud service, and only because the agent needs to search the web.

> The post is the tutorial; this README is the reference. Tables, defaults and gotchas live here so the post can stay readable.

## Start here

```bash
make setup     # uv sync — installs into ./.venv, touches nothing else
make env       # copies .env.example to .env (then add your TAVILY_API_KEY)
make model     # ollama pull qwen3:8b (about 5 GB)
make doctor    # checks your machine can do this — get this green first
```

Then pick a lab. `make help` lists everything.

You'll need [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/download), Python 3.11+, and a free [Tavily](https://tavily.com) key.

## The labs

| Target | File | What it shows |
|---|---|---|
| `make shallow` | `01_shallow_agent.py` | The plain ReAct loop, and where it breaks |
| `make deep` | `02_deep_agent.py` | Plan + files + a researcher sub-agent |
| `make toolbox` | `04_toolbox_agent.py` | The built-in tools, auditing a tiny project — **no API key** |
| `make observed` | `03_observed_agent.py` | Tracing a whole run with Langfuse |
| `make ollabridge` | `05_ollabridge_demo.py` | The same model through an OllaBridge gateway — **nothing to configure** |
| `make advanced` | `06_advanced_agent.py` | Custom middleware and harness settings — **no API key** |

Supporting files: `local_model.py` (the settings that make local deep agents work), `search.py` (token-trimmed web search), `progress.py` (streams a line per model and tool call, so a slow run doesn't look like a hang), `keys.py` (checks hosted-service keys at startup, and treats a `.env.example` placeholder as missing), `00_doctor.py` (eight preflight checks), `Modelfile.example` (bake `num_ctx` into your own tag).

The post's diagrams live in `assets/images/posts/2026-08-23-Deep-Agents-Ollama/` as hand-written SVG. They carry their own `prefers-color-scheme` styles, so they follow the reader's light or dark theme with no JavaScript and no external assets.

### A real run, for calibration

`make deep` on the reference machine — RTX 4080 Laptop, 12 GB, `qwen3:8b` at 24576 ctx:

```
      3.0s  ask  write_todos
      3.9s  ask  task     (sub-agent: 9.6s)
     15.5s  ask  task     (sub-agent: 11.5s)
     29.2s  ask  task     (sub-agent: 9.3s)
     40.2s  ask  read_file
     47.2s  ask  write_file
[deep agent] finished in 49.1s over 6 tool-calling turns
[virtual filesystem: 3 file(s)]   [orchestrator messages: 14]
```

**49 seconds, 14 messages in the orchestrator, three delegations it never saw the inside of.** What it got wrong is worth knowing too: the researchers invented their own file paths instead of the `/notes_1.md` the prompt asked for, one skipped its write entirely, the orchestrator read one note back instead of three, and one sub-agent drifted onto models instead of engines. An 8B model gets the shape right and the details approximate — verify what it wrote rather than trusting it followed instructions. See the post's "What an 8B model does and doesn't obey".

**Quickest way to see the point:** `make toolbox`. No key, no network. It seeds a four-file fake project into the agent's filesystem, asks for a code audit, and prints the tools it reached for:

```
order: write_todos -> ls -> glob -> grep -> grep -> task -> write_file -> edit_file
```

Nobody wrote that sequence. Plan, get oriented, narrow by name, narrow by content, delegate the reading, write the result, tidy up.

## Hardware

The floor is an **RTX 4080 Laptop GPU (12 GB)** — everything is sized to fit it. Bigger cards just get a longer context window.

| Card | VRAM | `DEEP_AGENT_NUM_CTX` |
|---|---|---|
| **RTX 4080 Laptop** (floor) | **12 GB** | **24576** (default) |
| RTX 4080 / 5080 Desktop, 5080 Laptop, 4090 Laptop | 16 GB | 32768 |
| RTX 4090 Desktop, 5090 Laptop | 24 GB | 40960 |
| RTX 5090 Desktop | 32 GB | 40960, and try a bigger model |

VRAM is consumed by the model weights *and* by the context window's KV cache, which grows with `num_ctx`. `make doctor` reads the real numbers off your card rather than trusting this table.

No NVIDIA card? It runs on CPU. Slowly, but every lab completes — `make doctor` warns rather than failing, and still exits 0. Budget minutes per step instead of seconds, and consider `DEEP_AGENT_MODEL=qwen3:4b`. (If you *do* have a card and the model still lands 0% on it, that's a real failure and doctor says so: something spilled.)

## Models

Default is `qwen3:8b` (~5.2 GB). Change with `make model MODEL=qwen3:14b` plus `DEEP_AGENT_MODEL` in `.env`.

| Model | Weights (Q4) | Good for |
|---|---|---|
| `qwen3:4b` | ~2.6 GB | Tight on memory |
| **`qwen3:8b`** | **~5.2 GB** | **Default — best fit for 12 GB** |
| `llama3.1:8b` | ~4.9 GB | Alternative at the same size |
| `qwen3:14b` | ~9.3 GB | 16 GB and up |
| `gpt-oss:20b` | ~13 GB | 24 GB and up |

The hard requirement is **tool calling**. A deep agent is nothing but tool calls, and many small models answer a tool-call prompt with prose instead. That's check 6 of `make doctor`.

## Three settings that fail silently

All in `src/local_model.py`. Each one is quiet when wrong, which is what makes them expensive.

| Setting | Why it matters |
|---|---|
| `num_ctx` | Ollama's default window is small and it **truncates without erroring** — the agent forgets its plan with no traceback |
| `profile={"max_input_tokens": num_ctx}` | `deepagents` decides when to summarise from `model.profile`. `ChatOllama` publishes none, so the fallback trigger is **170,000 tokens** — on a 24k window it never fires. This moves it to 85% of the real window |
| `keep_alive` | A run is dozens of short calls; without this Ollama reloads the model each time |

**Looks frozen after the banner?** The labs stream a line per model and tool call, so you should see movement within a minute. If the lines have stopped, run `make doctor`: check 6b loads the model and measures how much of it landed on the GPU. Below 100% means the remainder is on the CPU and the whole run crawls — lower `DEEP_AGENT_NUM_CTX` or pick a smaller model. A long silent gap on a `task` line is normal: a whole sub-agent runs inside that one tool call and nothing streams out of it.

Also: use **one model instance** for the main agent and all its sub-agents. Two models on one card means Ollama swapping weights on every delegation.

## The built-in toolbox

`create_deep_agent()` provides all of these before you add any tools of your own.

| Tool | Use it for |
|---|---|
| `write_todos` | Anything with more than about three steps |
| `ls` | Getting oriented at the start |
| `glob` | "Which files should I even look at?" |
| `grep` | Finding the few files that matter (literal text, **not** regex) |
| `read_file` | Reading only what `grep` pointed at |
| `write_file` | Saving notes and final output |
| `edit_file` | Small fixes without rewriting the file |
| `delete` | Cleaning up scratch files |
| `task` | Anything that would make a mess in the main context |
| `execute` | Shell commands — needs a sandbox backend, errors on `StateBackend` |

**The pattern worth stealing:** `glob` → `grep` → `read_file`. Narrow by name, narrow by content, then read only what's left. Reading everything and letting the model sort it out fills a 24k window with files it didn't need. Say so in your system prompt — models don't always do it unprompted.

**Seeding files:** build the values with `create_file_data()` from `deepagents.backends.utils`. A bare `{"content": ...}` dict works for every tool except `glob`, which sorts by modification time and fails with a cryptic `'modified_at'` error.

## Middleware and harness settings

Two ways to change the harness. `make advanced` demonstrates both, no API key needed.

**Middleware** wraps individual calls. You write a class with the hooks you need:

| Hook | When |
|---|---|
| `before_agent` / `after_agent` | Once per run |
| `before_model` / `after_model` | Around each model call |
| `wrap_model_call` | *Around* the model call — read or edit `request.tools`, the prompt |
| `wrap_tool_call` | *Around* a tool execution — time it, log it, or refuse it |

Not calling `handler()` in `wrap_tool_call` is the whole guardrail: the tool never runs and your `ToolMessage` becomes the result. Every hook has an async twin (`awrap_model_call`, …).

Your middleware is spliced in **after** the built-in core stack and **before** the tail. Reusing an existing `name` **replaces** that entry in place instead of appending — so `name = "SummarizationMiddleware"` swaps out the built-in compaction and keeps its position.

**Harness settings** (`HarnessProfile`) are declarative defaults attached to a *model*:

| Field | Effect |
|---|---|
| `base_system_prompt` / `system_prompt_suffix` | Prompt assembles as `USER → BASE → SUFFIX` |
| `excluded_tools` | Tool names withheld from the model |
| `tool_description_overrides` | Rewrite a built-in tool's description |
| `extra_middleware` / `excluded_middleware` | Add or remove stack entries |
| `general_purpose_subagent` | Tune or disable the default catch-all sub-agent |

**The key you register under matters.** Pass a model *string* and that string is the key; pass a pre-built instance (what `build_local_model()` does) and the key is derived from the model, falling back to the provider. For `ChatOllama` that provider is `ollama`, so `register_harness_profile("ollama", ...)` covers every local model here. Mismatch and deepagents logs `No harness profile matched pre-built model …` rather than failing silently.

**And the provider changes with the backend.** `DEEP_AGENT_BACKEND=ollabridge` builds a `ChatOpenAI`, whose provider is `openai` — the gateway speaks the OpenAI API, so that is what it is. A profile registered under `ollama` then matches nothing, and the exclusions and prompt suffix quietly do not apply. `06_advanced_agent.py` picks its key from the active backend for exactly this reason.

Two guard rails: `excluded_middleware` refuses to strip required scaffolding (`FilesystemMiddleware`, `SubAgentMiddleware`) at the moment you declare the profile, and an exclusion matching nothing raises rather than being ignored.

Rule of thumb: *"this model should never see that tool"* → harness setting. *"watch or intervene per call"* → middleware.

## Using an OllaBridge gateway

[OllaBridge](https://github.com/ruslanmv/ollabridge) fronts every model you can reach — local Ollama, remote GPUs, your own provider keys — behind one OpenAI-compatible URL. This repo supports it as a backend, with no code changes and nothing to configure:

```bash
pip install ollabridge
ollabridge start --auth-mode local-trust --host 127.0.0.1   # one terminal
make ollabridge                                             # another
```

Verified against **OllaBridge 0.1.7** from PyPI, which carries tool calls end to end. `local-trust` skips the key check for requests from this machine, and `--host 127.0.0.1` keeps it listening here only — so there's no `.env` entry and no key to copy. `make ollabridge` sets `DEEP_AGENT_BACKEND=ollabridge` itself for that one command, and `build_local_model()` returns a `ChatOpenAI` pointed at the gateway instead of a `ChatOllama`. Same model, same GPU, different front door. The script verifies chat, then tool calling, then runs a real deep agent through the gateway.

Two optional overrides, both `.env`:

- `DEEP_AGENT_BACKEND=ollabridge` — send *every* lab through the gateway, not just `make ollabridge`
- `OLLABRIDGE_URL` / `OLLABRIDGE_API_KEY` — for a gateway on another host or port, or one started with `--auth-mode required`. Step 1 reports `HTTP 401 — the key was checked` when you need the key

Two things worth knowing about any OpenAI-compatible proxy:

- **Tool calling has to survive the trip.** A proxy that accepts `tools` and ignores it returns HTTP 200 and a confident prose answer with nothing in the logs. Current OllaBridge carries tool calls; step 3 of `make ollabridge` confirms whichever gateway you actually have.
- **`num_ctx` cannot be sent.** The OpenAI chat API has no such field, so bake the window into the model instead: `ollama create qwen3-agent -f Modelfile.example`, then set `DEEP_AGENT_MODEL=qwen3-agent`. Worth doing even if you never use a gateway.

## Notes on `deepagents` 0.7.x

Two behaviours worth knowing, both handled here:

- **`write_todos` isn't in the default middleware stack** on most models. Pass `middleware=[TodoListMiddleware()]` or the planning tool simply doesn't exist.
- **`result["files"]` maps paths to `FileData` dicts**, not strings. Read `data["content"]`; slicing the dict raises `TypeError: unhashable type: 'slice'`.

A deep agent run makes many model calls, so expect it to take noticeably longer than a single completion — locally you feel every one.
