"""Where the machine-local cache lives."""

import os
from pathlib import Path


def cache_dir() -> Path:
    return Path(os.environ.get("FOOTNOTE_CACHE") or "~/.cache/footnote").expanduser()
