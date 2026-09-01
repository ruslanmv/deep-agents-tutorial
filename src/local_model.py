"""The one place that knows how to point `deepagents` at a local Ollama model.

Every numbered script imports `build_local_model()` from here, because getting
a deep agent to behave on a 16 GB laptop GPU comes down to four settings that
are easy to get wrong and painful to debug:

1. `num_ctx` — Ollama does not raise when you exceed the context window, it
   silently drops the oldest tokens. A deep agent whose plan scrolls out of
   the window turns into a shallow agent that has forgotten the task.
2. `profile={"max_input_tokens": num_ctx}` — this is the important one.
   `deepagents` sizes its auto-compaction from `model.profile`. `ChatOllama`
   does not publish a profile, so `deepagents` falls back to a conservative
   *cloud* default and only compacts at 170,000 tokens. With a 32,768-token
   local window, compaction would never fire before Ollama started truncating.
   Declaring the profile switches `deepagents` to fraction-based thresholds
   (compact at 85% of the real window, keep the last 10%).
3. `keep_alive` — a deep agent makes many short calls. Without this Ollama
   unloads the weights between them and every step pays a reload.
4. `reasoning=False` — visible `<think>` blocks spend the context budget you
   just fought for, and some tool-call parsers choke on them.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen3:8b"
# 24k is chosen to be safe on the smallest card this tutorial targets, a 12 GB
# RTX 4080 Laptop that is also driving your desktop. On 16 GB or more you can
# raise this — see the table in the README.
DEFAULT_NUM_CTX = 24_576
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_PORT = 11434


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_base_url(raw: str) -> str:
    """Turn whatever someone put in `OLLAMA_HOST` into a URL the client can use.

    `OLLAMA_HOST` is Ollama's *own* environment variable, and its documented
    form is `host:port` with no scheme — so people quite reasonably set
    `localhost` or `127.0.0.1:11434`. Those happen to work, because the client
    fills in the missing pieces.

    The one that does not is a scheme with no port: `http://localhost` resolves
    to port 80, gets a connection refused, and reads like the daemon is down.
    This adds the default port in that case, and leaves an explicit port or an
    https endpoint (which is behind a proxy on 443) alone.
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return "http://localhost:11434"
    if "://" not in url:
        url = "http://" + url
    scheme, _, rest = url.partition("://")
    host = rest.split("/", 1)[0]
    # An explicit port, an IPv6 literal, or TLS (assume 443 behind a proxy).
    if ":" in host.rsplit("]", 1)[-1] or scheme == "https":
        return url
    return f"{scheme}://{rest}:{DEFAULT_PORT}" if "/" not in rest else url


def gpu_residency(base_url: str | None = None, timeout: float = 3.0):
    """How much of the loaded model is actually on the GPU, per Ollama.

    Returns `{"name", "total", "on_gpu", "pct"}` for the resident model, or
    `None` when nothing is loaded or the daemon cannot be reached. Estimates
    lie about this; `/api/ps` measures it.

    This is the difference between a run that takes 49 seconds and one that
    takes ten minutes, and nothing else in a run tells you which you are in.
    """
    import json  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    url = normalize_base_url(base_url or os.environ.get("OLLAMA_HOST", ""))
    # Ollama is on this machine, so go straight at it. A corporate `http_proxy`
    # that does not list localhost in `no_proxy` would otherwise swallow the
    # probe and quietly cost you the warning.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/api/ps", timeout=timeout) as resp:
            running = (json.loads(resp.read()) or {}).get("models") or []
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    if not running:
        return None

    wanted = os.environ.get("DEEP_AGENT_MODEL", DEFAULT_MODEL)
    wanted = wanted if ":" in wanted else f"{wanted}:latest"
    entry = next((m for m in running if m.get("name") == wanted), running[0])
    total = entry.get("size") or 0
    if not total:
        return None
    on_gpu = entry.get("size_vram") or 0
    return {
        "name": entry.get("name", "?"),
        "total": total,
        "on_gpu": on_gpu,
        "pct": on_gpu / total * 100,
    }


def ollabridge_api_key() -> str:
    """The key to send to an OllaBridge gateway — usually a formality.

    OllaBridge's default auth mode is `local-trust`: a request arriving over
    loopback is trusted and the key is never checked. So the OllaBridge lab
    needs no `.env` entry at all, and this returns a stand-in.

    `OPENAI_API_KEY` is deliberately not consulted: this endpoint is your own
    gateway, and quietly forwarding a real OpenAI key to it would be a
    surprise. Set `OLLABRIDGE_API_KEY` when you actually need one — a remote
    gateway, or one started with `AUTH_MODE=required`.
    """
    key = os.environ.get("OLLABRIDGE_API_KEY", "").strip()
    # `.env.example` ships the literal placeholder; treat it as absent.
    if not key or key.startswith("sk-ollabridge-..."):
        return "local-trust"
    return key


def build_local_model(*, validate: bool = True, **overrides):
    """Build the chat model, for whichever backend `DEEP_AGENT_BACKEND` names.

    `ollama` (default) talks straight to the Ollama daemon. `ollabridge` goes
    through an OllaBridge gateway instead, which is OpenAI-compatible — see
    `build_ollabridge_model` for what that costs you.
    """
    backend = os.environ.get("DEEP_AGENT_BACKEND", "ollama").strip().lower()
    if backend == "ollabridge":
        return build_ollabridge_model(**overrides)
    if backend not in {"", "ollama"}:
        raise ValueError(
            f"DEEP_AGENT_BACKEND={backend!r} is not recognised. "
            "Use 'ollama' (default) or 'ollabridge'."
        )
    return build_ollama_model(validate=validate, **overrides)


def build_ollabridge_model(**overrides):
    """Build a `ChatOpenAI` pointed at an OllaBridge gateway.

    OllaBridge fronts your local Ollama (and remote GPUs, and BYOK provider
    accounts) behind one OpenAI-compatible URL, so the client is `ChatOpenAI`
    rather than `ChatOllama`.

    No key needed. OllaBridge starts in `local-trust` mode, which accepts any
    request arriving over loopback without checking the key — so nothing here
    has to be configured. Set `OLLABRIDGE_API_KEY` only if you moved the
    gateway to another machine or switched its auth mode to `required`.

    Two things do not survive the trip, and both matter here:

    * `num_ctx` has no place in the OpenAI chat API, so you cannot set the
      context window per request. Bake it into the model instead — see
      `Modelfile.example` in the repo root — or the gateway serves whatever
      default the model was built with.
    * Tool calling has to survive the gateway, and not every
      OpenAI-compatible proxy forwards it. Current OllaBridge does. Run
      `make ollabridge` to confirm the gateway you actually have before
      pointing the agent labs at it.
    """
    from langchain_openai import ChatOpenAI  # noqa: PLC0415 — optional path

    num_ctx = int(os.environ.get("DEEP_AGENT_NUM_CTX", DEFAULT_NUM_CTX))
    num_ctx = int(overrides.pop("num_ctx", num_ctx))

    settings = {
        "model": os.environ.get("DEEP_AGENT_MODEL", DEFAULT_MODEL),
        "base_url": os.environ.get(
            "OLLABRIDGE_URL", "http://localhost:11435/v1"
        ).rstrip("/"),
        "api_key": ollabridge_api_key(),
        "temperature": 0.0,
        # Same trick as the Ollama path: tell deepagents the real window size so
        # it compacts against that instead of a cloud-sized default.
        "profile": {"max_input_tokens": num_ctx},
    }
    settings.update(overrides)
    return ChatOpenAI(**settings)


def build_ollama_model(*, validate: bool = True, **overrides) -> ChatOllama:
    """Build a `ChatOllama` configured for long-horizon agent work.

    Args:
        validate: Ask Ollama at construction time whether the model is actually
            pulled. Costs one HTTP round-trip and turns a confusing mid-run
            failure into an immediate, readable error. Pass `False` when you
            only want to assemble a graph without a running daemon.
        **overrides: Any `ChatOllama` field, e.g. `model=...`, `num_ctx=...`.

    Returns:
        A model instance that reports its own context window to `deepagents`.
    """
    num_ctx = int(os.environ.get("DEEP_AGENT_NUM_CTX", DEFAULT_NUM_CTX))
    num_ctx = int(overrides.pop("num_ctx", num_ctx))

    settings = {
        "model": os.environ.get("DEEP_AGENT_MODEL", DEFAULT_MODEL),
        "base_url": normalize_base_url(os.environ.get("OLLAMA_HOST", "")),
        "num_ctx": num_ctx,
        "keep_alive": os.environ.get("DEEP_AGENT_KEEP_ALIVE", DEFAULT_KEEP_ALIVE),
        "reasoning": _env_flag("DEEP_AGENT_REASONING", default=False),
        # Long tool-calling runs want determinism, not creativity.
        "temperature": 0.0,
        "validate_model_on_init": validate,
        # Tell deepagents how big the window really is, so its compaction
        # thresholds are computed against the local model instead of a
        # cloud-sized default. See the module docstring.
        "profile": {"max_input_tokens": num_ctx},
    }
    settings.update(overrides)
    return ChatOllama(**settings)


def describe(model) -> str:
    """One-line summary for script banners, for either backend."""
    name = getattr(model, "model", None) or getattr(model, "model_name", "?")
    window = (model.profile or {}).get("max_input_tokens", "?")
    if isinstance(model, ChatOllama):
        return f"{name} @ {model.num_ctx} ctx via ollama {model.base_url}"
    # ChatOpenAI keeps the endpoint on openai_api_base.
    endpoint = getattr(model, "openai_api_base", None) or "?"
    return f"{name} @ {window} ctx (declared) via ollabridge {endpoint}"
