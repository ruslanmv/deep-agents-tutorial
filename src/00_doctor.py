"""Step 0: preflight. Run this before anything else.

Checks, in the order that they will bite you:

  1. NVIDIA driver present, and how much VRAM the GPU actually has.
  2. Ollama daemon reachable.
  3. The configured model is pulled.
  4. The model's trained context length is >= the num_ctx you asked for.
  5. num_ctx fits in VRAM alongside the weights.
  6. The model can really emit a tool call — the gate most small models fail.
 6b. Where the model actually landed: GPU, or spilled to the CPU.
 6c. How fast it generates, and whether it is burning hidden thinking tokens.
  7. deepagents assembles a graph against it, with the expected tools.

Exits non-zero on the first hard failure, so `make doctor` is usable in CI.
"""

from dotenv import load_dotenv

load_dotenv()

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_model import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    build_local_model,
    normalize_base_url,
)

# Resolved the same way the labs resolve it, so the doctor cannot end up
# probing a different daemon than the one the agents will actually use.
OLLAMA_HOST = normalize_base_url(os.environ.get("OLLAMA_HOST", ""))
MODEL = os.environ.get("DEEP_AGENT_MODEL", DEFAULT_MODEL)
NUM_CTX = int(os.environ.get("DEEP_AGENT_NUM_CTX", DEFAULT_NUM_CTX))

GIB = 1024**3
_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def warn(msg: str) -> None:
    print(f"  \033[33mWARN\033[0m  {msg}")
    _warnings.append(msg)


def fail(msg: str, *, fatal: bool = False) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")
    _failures.append(msg)
    if fatal:
        summary()
        sys.exit(1)


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def _api(path: str, payload: dict | None = None, timeout: int = 15) -> dict:
    url = f"{OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# --------------------------------------------------------------------------
# 1. GPU
# --------------------------------------------------------------------------
def check_gpu() -> int | None:
    """Return total VRAM in MiB, or None when it cannot be determined."""
    section("1. GPU")
    if not shutil.which("nvidia-smi"):
        warn(
            "nvidia-smi not found. Ollama will fall back to CPU, which works "
            "but makes a deep agent run take many minutes per step."
        )
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        warn(f"nvidia-smi failed: {exc}")
        return None

    total_mib = None
    for line in out.splitlines():
        name, total, used, driver = (p.strip() for p in line.split(","))
        total_mib = int(total)
        ok(f"{name} — {total_mib} MiB total, {used} MiB in use, driver {driver}")
    return total_mib


# --------------------------------------------------------------------------
# 2 & 3. Daemon and model presence
# --------------------------------------------------------------------------
def check_daemon() -> None:
    section("2. Ollama daemon")
    try:
        tags = _api("/api/tags")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fail(
            f"cannot reach Ollama at {OLLAMA_HOST} ({exc}). Start it with "
            "`ollama serve`, or fix OLLAMA_HOST in .env",
            fatal=True,
        )
        return
    models = tags.get("models", []) or []
    installed = [m.get("name", "?") for m in models]
    ok(f"reachable at {OLLAMA_HOST} — {len(installed)} model(s) installed")

    section("3. Configured model")
    # Ollama reports "qwen3:8b"; accept a bare "qwen3" as matching "qwen3:latest".
    wanted = MODEL if ":" in MODEL else f"{MODEL}:latest"
    if wanted in installed:
        entry = next((m for m in models if m.get("name") == wanted), {})
        size = entry.get("size") or None
        extra = f" ({size / GIB:.1f} GiB on disk)" if size else ""
        ok(f"{wanted} is pulled{extra}")
        return size
    fail(
        f"{wanted} is not pulled. Run `make model` (or "
        f"`ollama pull {MODEL}`). Installed: {', '.join(installed) or 'none'}",
        fatal=True,
    )
    return None


# --------------------------------------------------------------------------
# 4 & 5. Context window sizing
# --------------------------------------------------------------------------
def check_context(total_vram_mib: int | None, size_bytes: int | None) -> None:
    section("4. Context window")
    try:
        info = _api("/api/show", {"model": MODEL})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        warn(f"could not read model metadata: {exc}")
        return

    details = info.get("model_info", {}) or {}
    trained = next(
        (v for k, v in details.items() if k.endswith(".context_length")), None
    )
    if isinstance(trained, int):
        if NUM_CTX <= trained:
            ok(f"num_ctx={NUM_CTX} fits the model's trained window of {trained}")
        else:
            fail(
                f"num_ctx={NUM_CTX} exceeds the model's trained window of "
                f"{trained}. Lower DEEP_AGENT_NUM_CTX — going past this "
                "degrades quality even though Ollama accepts it."
            )
    else:
        warn("model metadata does not report a context length; skipping check")

    section("5. VRAM headroom")
    if not (total_vram_mib and size_bytes):
        warn(
            "skipping the estimate — "
            + ("no GPU detected" if not total_vram_mib else "model size unknown")
            + ". Check 6 measures the real figure once the model is loaded."
        )
        return

    weights_gib = size_bytes / GIB
    total_gib = total_vram_mib / 1024
    # Rough KV-cache estimate. Real usage depends on layers, heads and cache
    # dtype; this is deliberately pessimistic so the advice errs toward safety.
    kv_gib = NUM_CTX / 1024 * 0.12
    need = weights_gib + kv_gib
    # On a laptop your desktop is also using this card. Leave room for it.
    budget = total_gib - 1.0
    print(
        f"        weights ~{weights_gib:.1f} GiB + KV cache ~{kv_gib:.1f} GiB "
        f"= ~{need:.1f} GiB, and ~{budget:.1f} GiB is usable "
        f"(of {total_gib:.1f} GiB, minus ~1 GiB for your display)"
    )
    if need < budget:
        ok(f"fits, with ~{budget - need:.1f} GiB spare")
        return

    # Largest num_ctx whose KV cache still fits, rounded down to 4k and never
    # suggested above the model's own trained window.
    suggestion = int((budget - weights_gib) * 1024 / 0.12) // 4096 * 4096
    if isinstance(trained, int):
        suggestion = min(suggestion, trained // 4096 * 4096)
    if suggestion >= 8192:
        advice = f"Try DEEP_AGENT_NUM_CTX={suggestion}"
    else:
        advice = "This model is too big for this card — try a smaller one"
    warn(
        f"this will probably spill into system RAM and get very slow. {advice}, "
        "or set OLLAMA_KV_CACHE_TYPE=q8_0 to roughly halve the KV cache."
    )


# --------------------------------------------------------------------------
# 6. Tool calling — the real gate
# --------------------------------------------------------------------------
def check_tool_calling() -> None:
    section("6. Tool calling")

    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"sunny in {city}"

    try:
        model = build_local_model(validate=False, num_predict=256)
        bound = model.bind_tools([get_weather])
        reply = bound.invoke("What is the weather in Copenhagen? Use the tool.")
    except Exception as exc:  # noqa: BLE001 - surface any failure verbatim
        fail(f"tool-call probe raised {type(exc).__name__}: {exc}")
        return

    calls = getattr(reply, "tool_calls", None) or []
    if calls:
        ok(f"model emitted a tool call: {calls[0].get('name')}({calls[0].get('args')})")
    else:
        fail(
            "model answered in prose instead of calling the tool. A deep agent "
            f"cannot work with this. Pick a tool-calling model — {DEFAULT_MODEL} "
            "is the tested default. Text was: "
            f"{(getattr(reply, 'text', None) or str(reply.content))[:160]!r}"
        )


def check_speed_and_thinking() -> None:
    """How fast does it generate, and is it silently spending tokens thinking?

    This is the check for the run that is healthy on every other measure and
    still takes ten minutes. Two things cause that on a GPU-resident model:

    * The model is thinking. qwen3 is a hybrid reasoning model, and a long
      `<think>` block costs hundreds of tokens per turn that you never see in
      the answer. `reasoning=False` asks Ollama to turn that off, but the
      `think` parameter needs a recent Ollama and a model that honours it — so
      measure rather than assume.
    * Generation is simply slow, which points back at the GPU.

    Both show up as tokens: how many, and how fast.
    """
    section("6c. Generation speed, and whether it is thinking")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      "In one short sentence, what is a KV cache?"}],
        "stream": False,
        "think": False,
        "keep_alive": os.environ.get("DEEP_AGENT_KEEP_ALIVE", "30m"),
        "options": {"num_ctx": NUM_CTX, "temperature": 0},
    }
    try:
        body = _api("/api/chat", payload=payload, timeout=180)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        warn(f"could not run the speed probe: {exc}")
        return

    msg = body.get("message") or {}
    thinking = (msg.get("thinking") or "").strip()
    eval_count = body.get("eval_count") or 0
    eval_ns = body.get("eval_duration") or 0
    prompt_count = body.get("prompt_eval_count") or 0
    prompt_ns = body.get("prompt_eval_duration") or 0

    if eval_count and eval_ns:
        tps = eval_count / (eval_ns / 1e9)
        print(f"        generated {eval_count} tokens at {tps:.0f} tok/s")
        if tps < 15:
            warn(
                f"{tps:.0f} tok/s is slow for a local 8B on a GPU. A deep-agent "
                "run makes dozens of calls, so this is minutes per step. Check "
                "6b above, and close anything else using the card."
            )
        else:
            ok(f"{tps:.0f} tok/s — fine for a deep agent")
    if prompt_count and prompt_ns:
        pps = prompt_count / (prompt_ns / 1e9)
        print(f"        read the prompt at {pps:.0f} tok/s")

    if thinking:
        fail(
            f"the model returned a {len(thinking)} character reasoning block "
            "even though thinking was switched off. Every turn pays for tokens "
            "you never see, which is the usual reason a healthy GPU still takes "
            "minutes per step. Update Ollama (`think` needs a recent version), "
            "or use a model without a thinking mode."
        )
    else:
        ok("thinking is off — no hidden reasoning tokens per turn")


def check_residency(has_gpu: bool = True) -> None:
    """Where did the model actually land? Estimates lie; /api/ps measures.

    Run after the tool-call probe, so the weights are loaded. This is the check
    that catches the failure people report as a freeze: a model that spilled to
    system RAM still answers, just seconds per token.

    Args:
        has_gpu: Whether check 1 found a GPU at all. Running entirely on the
            CPU is a *failure* on a machine with a card — something spilled and
            you can fix it. On a machine with no card it is simply the
            supported CPU path, so it warns instead, and the advice changes:
            there is no VRAM to free.
    """
    section("6b. Where the model actually landed")
    try:
        ps = _api("/api/ps")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        warn(f"could not read /api/ps: {exc}")
        return

    running = ps.get("models", []) or []
    if not running:
        warn("no model is resident — it may have been unloaded already")
        return

    wanted = MODEL if ":" in MODEL else f"{MODEL}:latest"
    entry = next((m for m in running if m.get("name") == wanted), running[0])
    total = entry.get("size") or 0
    on_gpu = entry.get("size_vram") or 0
    if not total:
        warn("Ollama did not report a size for the resident model")
        return

    pct = on_gpu / total * 100
    print(
        f"        {entry.get('name', '?')} — {total / GIB:.1f} GiB resident, "
        f"{on_gpu / GIB:.1f} GiB of it on the GPU ({pct:.0f}%)"
    )
    if pct >= 99:
        ok("fully on the GPU — this will run at full speed")
    elif pct <= 1 and not has_gpu:
        warn(
            "entirely on the CPU, as expected on a machine with no NVIDIA card. "
            "Every lab still completes, but budget minutes per step rather than "
            "seconds — and expect the first call to look like a hang. "
            "DEEP_AGENT_MODEL=qwen3:4b makes it noticeably less painful."
        )
    elif pct <= 1:
        fail(
            "entirely on the CPU even though you have a GPU. It will still "
            "answer, but at seconds per token, which is the thing people report "
            "as a hang. Lower DEEP_AGENT_NUM_CTX, pick a smaller model, or free "
            "up VRAM."
        )
    else:
        warn(
            f"only {pct:.0f}% is on the GPU, so the rest is running on the CPU "
            "and the whole run goes at that speed. Lower DEEP_AGENT_NUM_CTX or "
            "set OLLAMA_KV_CACHE_TYPE=q8_0."
        )


# --------------------------------------------------------------------------
# 7. deepagents assembly
# --------------------------------------------------------------------------
def check_deep_agent() -> None:
    section("7. deepagents harness")
    try:
        from deepagents import create_deep_agent
        from deepagents.middleware.summarization import compute_summarization_defaults
        from langchain.agents.middleware import TodoListMiddleware

        model = build_local_model(validate=False)
        defaults = compute_summarization_defaults(model)
        trigger = defaults["trigger"]
        if trigger[0] == "fraction":
            ok(
                f"auto-compaction is context-relative ({trigger[1]:.0%} of "
                f"{model.num_ctx} tokens)"
            )
        else:
            fail(
                f"auto-compaction trigger is {trigger}, not a fraction of the "
                "local window. The model profile is not being declared — see "
                "local_model.build_local_model()."
            )

        agent = create_deep_agent(
            model=model,
            tools=[],
            system_prompt="probe",
            middleware=[TodoListMiddleware()],
        )
        tools = sorted(agent.nodes["tools"].bound.tools_by_name)
        missing = {"write_todos", "write_file", "read_file", "task"} - set(tools)
        if missing:
            fail(f"expected tools missing from the graph: {sorted(missing)}")
        else:
            ok(f"graph assembled with {len(tools)} tools: {', '.join(tools)}")
    except Exception as exc:  # noqa: BLE001
        fail(f"harness assembly raised {type(exc).__name__}: {exc}")


def summary() -> None:
    print("\n" + "=" * 68)
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) failed:\033[0m")
        for f in _failures:
            print(f"  - {f}")
    if _warnings:
        print(f"\033[33m{len(_warnings)} warning(s).\033[0m")
    if not _failures:
        print("\033[32mAll hard checks passed — you are ready to run the agents.\033[0m")
    print("=" * 68)


if __name__ == "__main__":
    print(f"deep-agents preflight — model={MODEL} num_ctx={NUM_CTX}")
    vram = check_gpu()
    size_bytes = check_daemon()
    check_context(vram, size_bytes)
    check_tool_calling()
    check_residency(has_gpu=vram is not None)
    check_speed_and_thinking()
    check_deep_agent()
    summary()
    sys.exit(1 if _failures else 0)
