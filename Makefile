# Deep agents on your own GPU.
#
# Everything runs through uv, which keeps the environment in ./.venv and
# pins exact versions in uv.lock. No target touches your system Python, and
# no target downloads model weights unless you ask for it with `make model`.

SHELL  := /bin/bash
SRC    := src
UV     := uv

# Kept in sync with DEEP_AGENT_MODEL in .env.example.
MODEL  ?= qwen3:8b

.DEFAULT_GOAL := help
.PHONY: help setup env model doctor check shallow deep observed toolbox ollabridge advanced clean distclean

help: ## Show this help
	@echo "Deep agents on your own GPU"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Start here:  make setup && make env && make model && make doctor"

setup: ## Install dependencies into ./.venv (needs uv: https://docs.astral.sh/uv)
	@command -v $(UV) >/dev/null 2>&1 || { \
	  echo "!! uv not found. Install it:"; \
	  echo "     curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	$(UV) sync
	@echo ">> ready. next: make env && make model && make doctor"

env: ## Create .env from .env.example if you don't have one yet
	@if [ -f .env ]; then \
	  echo ">> .env already exists, leaving it alone"; \
	else \
	  cp .env.example .env && echo ">> wrote .env — now add your TAVILY_API_KEY"; \
	fi

model: ## Pull the Ollama model (override: make model MODEL=qwen3:14b)
	@command -v ollama >/dev/null 2>&1 || { \
	  echo "!! ollama not found. Install it from https://ollama.com/download"; exit 1; }
	ollama pull $(MODEL)
	@echo
	@ollama list

doctor: ## Check your machine can actually do this (run before anything else)
	$(UV) run python $(SRC)/00_doctor.py

check: ## Byte-compile the sources — no GPU or Ollama needed
	$(UV) run python -m compileall -q $(SRC) && echo ">> all sources compile"

shallow: ## Lab 1: the plain tool loop
	$(UV) run python $(SRC)/01_shallow_agent.py

deep: ## Lab 2: the deep agent (plan + files + sub-agent)
	$(UV) run python $(SRC)/02_deep_agent.py

toolbox: ## Lab 3: the built-in tools on a tiny project (no API key needed)
	$(UV) run python $(SRC)/04_toolbox_agent.py

observed: ## Lab 4: the deep agent, traced with Langfuse
	$(UV) run python $(SRC)/03_observed_agent.py

ollabridge: ## Lab 5: the same model through an OllaBridge gateway (nothing to configure)
	DEEP_AGENT_BACKEND=ollabridge $(UV) run python $(SRC)/05_ollabridge_demo.py

advanced: ## Lab 6: custom middleware and harness settings (no API key needed)
	$(UV) run python $(SRC)/06_advanced_agent.py

clean: ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

distclean: clean ## Also remove the virtual environment
	rm -rf .venv
