# Troubleshooting

Almost everything that goes wrong with a local deep agent is one of a handful of
things, and none of them announce themselves. Each entry below is a symptom
first, because that's what you actually have when you go looking.

Run `make doctor` before anything else — eight checks, and it measures your
machine rather than guessing.

## The agent explains what it *would* do and never calls a tool

The model can't do tool calls. A deep agent is nothing but tool calls, so this
is fatal rather than cosmetic. Switch to `qwen3:8b` and confirm with check 6 of
`make doctor`.

## It forgets the task halfway through

Your context window is too small and Ollama truncated it without saying so —
no error, no warning, just an agent that has lost its own plan. Raise
`DEEP_AGENT_NUM_CTX`; checks 4 and 5 tell you what fits on your card.

## Answers get worse the longer it runs

The `profile` is missing, so auto-compaction never fires. `deepagents` sizes
its compaction threshold from `model.profile`, and `ChatOllama` publishes none,
so the fallback trigger is 170,000 tokens — on a 24k window it can never
happen. Add `profile={"max_input_tokens": num_ctx}`.

## Every step pauses for a few seconds

Ollama is unloading the model between calls. Set `keep_alive`, which
`.env.example` already does. If delegating to a sub-agent is *especially* slow,
you have two models fighting over VRAM — use one instance for the orchestrator
and all its sub-agents.

## Everything crawls and the GPU looks idle

You've spilled into system RAM. Lower `num_ctx`, or set
`OLLAMA_KV_CACHE_TYPE=q8_0` to roughly halve the KV cache. Check 6b of
`make doctor` measures exactly how much of the model landed on the GPU.

## `cannot reach Ollama`

The daemon isn't running: `ollama serve`, or start the desktop app.

## It printed the banner and then nothing

Almost always it is working, not stuck. A deep-agent run on a local 8B takes
minutes, and the labs stream a line per model call and per tool call so you can
watch it move. A long silent gap on a `task` line is normal — a whole sub-agent
runs inside that one tool call and nothing streams out of it.

If the lines have genuinely stopped, run `make doctor`: check 6b loads the
model and reports how much of it landed on the GPU. Anything below 100% means
the rest is running on the CPU and the whole run goes at that speed.

## `TAVILY_API_KEY is not set`, or a 403 exporting traces

You ran `make env` and never edited `.env`, so the placeholders are still in
it. The labs check for this before the first model call, and a placeholder
counts as missing — `pk-lf-...` is not empty, so an emptiness test waves it
through, which is exactly the bug this guards against.

## No NVIDIA card at all

It still runs, on the CPU. Every lab completes, but budget minutes per step
instead of seconds, and `DEEP_AGENT_MODEL=qwen3:4b` makes it noticeably less
painful. `make doctor` warns and exits 0 in this case rather than failing — CPU
is a supported configuration, just a slow one.

## The gateway lab returns HTTP 401

Your OllaBridge gateway checked the key, so it isn't in `local-trust` mode or
you aren't reaching it over loopback. Restart it with
`ollabridge start --auth-mode local-trust --host 127.0.0.1`, or put the key it
printed at startup into `OLLABRIDGE_API_KEY`.

## A harness profile seems to do nothing

The profile key falls back to the model's *provider*, and that's a property of
the client, not the weights. `ChatOllama` is `ollama`; the OllaBridge backend
builds a `ChatOpenAI`, whose provider is `openai`. Register under the wrong key
and deepagents logs `No harness profile matched pre-built model …` and quietly
uses defaults. Read that line as "your profile is not being applied".

## Two error messages worth recognising on sight

- **`TypeError: unhashable type: 'slice'`** — you sliced a `FileData` dict.
  `result["files"]` maps paths to dicts, not strings; read `data["content"]`.
- **`write_todos not found`** — you forgot `middleware=[TodoListMiddleware()]`.
  As of `deepagents` 0.7.x the planning tool is not in the default stack for
  most models, so a system prompt telling the agent to plan is describing a
  tool that does not exist.

## `glob failed: 'modified_at'`

You seeded the virtual filesystem with bare `{"content": ...}` dicts. `glob`
sorts by modification time and needs the timestamps. Build the values with
`create_file_data()` from `deepagents.backends.utils` — every other tool works
without it, so this one fails alone and cryptically.
