"""Entry point. `python main.py` — that's it.

Runtime dependencies are expected from the active Python environment
(usually `.venv`). Configuration is in `config.py` (gitignored).
"""

from __future__ import annotations

import asyncio
import sys


# Force UTF-8 for Windows consoles so artist names / track titles render.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

import config  # noqa: E402
from bot.app import run  # noqa: E402
from core.logging_setup import setup as setup_logging  # noqa: E402


def main() -> int:
    setup_logging(level="INFO")
    if not config.BOT_TOKEN or config.BOT_TOKEN.startswith("123456:REPLACE"):
        print("error: set BOT_TOKEN in config.py", file=sys.stderr)
        return 2
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        print("bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
