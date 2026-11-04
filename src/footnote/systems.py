"""The registry of runnable systems (the agent and the baselines register themselves here)."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from footnote.benchmark import AgentQuestion
from footnote.llm.client import LLMClient
from footnote.prediction import Prediction

# A runner should catch its own failures and return an error Prediction carrying the usage it
# accumulated. If it raises instead, the eval run records an error and charges the system's
# max_cost_per_question (actual spend is unknown): a conservative fallback, not the normal path.
SystemRunner = Callable[[AgentQuestion, LLMClient], Prediction]


@dataclass(frozen=True)
class SystemSpec:
    family: str
    runner: SystemRunner
    max_cost_per_question: Decimal
    config: dict[str, Any]
    config_hash: str

    @property
    def version(self) -> str:
        return str(self.config.get("version", ""))


def hash_config(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


_REGISTRY: dict[str, SystemSpec] = {}


def register_system(
    family: str,
    runner: SystemRunner,
    max_cost_per_question: Decimal,
    config: dict[str, Any],
) -> SystemSpec:
    """config must hold the version and everything that changes behavior (limits, prompt text)."""
    spec = SystemSpec(family, runner, max_cost_per_question, config, hash_config(config))
    _REGISTRY[family] = spec
    return spec


def get_system(family: str) -> SystemSpec:
    try:
        return _REGISTRY[family]
    except KeyError:
        raise KeyError(f"unknown system {family!r}; registered: {sorted(_REGISTRY)}") from None
