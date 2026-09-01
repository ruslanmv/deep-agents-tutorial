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

## It runs, but each step takes minutes (WSL especially)

The symptom is a run that never errors and never ends:

```
     50.8s  ask  write_todos
     69.4s  ask  task  (a sub-agent runs here — expect a long pause)
    248.9s  got task  (179.5s)
    520.0s  got task  (239.4s)
```

That can be ten times slower than the same task on the same card. There are two
causes, and they need telling apart before you change anything — `make doctor`
now separates them:

* **Check 6b** — the model is not on the GPU. Fixable by freeing VRAM.
* **Check 6c** — the model *is* on the GPU and generating at a healthy rate, but
  every turn is spending hundreds of tokens on a reasoning block you never see.

The second one is the trap, because every other check passes. qwen3 is a hybrid
reasoning model; `reasoning=False` asks Ollama to switch that off via its
`think` parameter, but that needs a recent Ollama and a model that honours it.
When it is ignored, each turn silently generates a `<think>` block before the
answer — and a deep-agent run is dozens of turns. Check 6c sends one probe with
thinking off and fails if a reasoning block comes back anyway.

The quick manual checks:

```bash
ollama ps           # recent versions show a PROCESSOR column: "100% GPU" / "100% CPU"
ollama --version    # `think` needs a recent build; update if this is old
```

The labs help too: after the first model call they print a `[slow]` block if the
model did not land on the GPU, they flag any turn over 25 seconds inline, and
they now stream what happens *inside* a sub-agent, so a `task` call that takes
nine minutes shows you the turns it spent them on rather than being a black box.

**On WSL there are two Ollamas, and it is easy to talk to the wrong one.** If
Ollama is installed on Windows it uses your NVIDIA card. If it is also
installed *inside* WSL, that copy is CPU-only unless you have set up the WSL
CUDA runtime — and `localhost:11434` inside WSL may reach either. That is the
usual cause of a run that used to be fast and now crawls.

To see which you are hitting:

```bash
# inside WSL
curl -s localhost:11434/api/ps            # is anything loaded here at all?
nvidia-smi                                # if this fails, a WSL-side Ollama is CPU-only
pgrep -a ollama                           # is there a WSL-side daemon running?
```

If there is a CPU-only daemon in WSL, either stop it and point at the Windows
one, or install the CUDA runtime for WSL. To use the Windows daemon, start it
on Windows with `OLLAMA_HOST=0.0.0.0:11434` and then, in WSL:

```bash
# .env — mirrored networking makes localhost work; otherwise use the host IP
OLLAMA_HOST=localhost:11434
# fallback when localhost does not reach Windows:
#   OLLAMA_HOST=$(ip route show default | awk '{print $3}'):11434
```

Either form is fine — `localhost`, `localhost:11434` and `http://localhost:11434`
all work. The one to avoid is a scheme with **no** port (`http://localhost`),
which means port 80; the repo now repairs that rather than failing on it.

**If it is the right daemon and still on the CPU,** the model did not fit. In
order of cost:

```bash
DEEP_AGENT_NUM_CTX=12288        # halve the KV cache — usually enough on 12 GB
DEEP_AGENT_MODEL=qwen3:4b       # smaller weights (make model MODEL=qwen3:4b)
OLLAMA_KV_CACHE_TYPE=q8_0       # set on the Ollama server, halves the cache again
```

Close whatever else is holding VRAM first — a browser with hardware
acceleration, a game, another model still resident from an earlier run
(`ollama stop <model>`).

One more WSL note that is *not* the cause here but costs you elsewhere: working
out of `/mnt/c/...` goes through the Windows filesystem bridge and is slow. It
does not affect inference, but `uv sync` and git will be noticeably quicker from
a native path like `~/deep-agents-tutorial`.

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
