"""The baselines: importing this package registers them."""

from typing import Any

from footnote.baselines.long_context import SPEC as LONG_CONTEXT
from footnote.baselines.no_docs import SPEC as NO_DOCS


def baseline_settings() -> dict[str, Any]:
    """Everything that changes baseline behavior, as published in data/baseline_settings.json."""
    return {s.family: s.config for s in (NO_DOCS, LONG_CONTEXT)}
