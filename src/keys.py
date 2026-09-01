"""Fail fast, and fail readably, on a missing hosted-service key.

`make env` copies `.env.example` to `.env`, and that file ships placeholders —
`tvly-...`, `pk-lf-...`. An empty check alone therefore passes for anyone who
ran `make env` and has not edited the file yet, which is precisely the person
the check exists to help. So "missing" here means empty *or* still the
placeholder.

The other half is timing. A key that is only read the first time a tool fires
turns a one-line configuration mistake into a traceback from inside the agent
loop, several model calls deep — a minute or more on a real GPU. Labs call
`require()` at startup so the message arrives before any work begins.
"""

from __future__ import annotations

import os
import sys


def is_placeholder(value: str) -> bool:
    """True when this looks like a `.env.example` stand-in rather than a key.

    Every placeholder in `.env.example` ends in "..." — that is the whole
    convention, and it is cheaper to rely on than a list of prefixes that has
    to be kept in step with the file.
    """
    return value.endswith("...")


def missing(name: str) -> bool:
    """True when `name` is unset, blank, or still holding its placeholder."""
    value = os.environ.get(name, "").strip()
    return not value or is_placeholder(value)


def require(*names: str, service: str, url: str) -> None:
    """Exit with a readable message unless every named key is really set.

    Args:
        *names: Environment variable names that must all be present.
        service: Human name of the service, for the message.
        url: Where to get a key.
    """
    absent = [n for n in names if missing(n)]
    if not absent:
        return
    verb = "is" if len(absent) == 1 else "are"
    sys.exit(
        f"\n{', '.join(absent)} {verb} not set.\n\n"
        f"  This lab needs a {service} key. Get a free one at {url},\n"
        f"  then put it in .env (run `make env` first if you have no .env yet).\n\n"
        f"  The model itself is local — this is the one hosted piece.\n"
    )
