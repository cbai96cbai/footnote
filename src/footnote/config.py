"""Settings read from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

from footnote.paths import cache_dir


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str = ""
    azure_api_key: str = ""
    agent_deployment: str = "footnote-agent"
    grader_deployment: str = "footnote-grader"
    cache: Path = Path("~/.cache/footnote")

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            azure_endpoint=env.get("AZURE_OPENAI_ENDPOINT", ""),
            azure_api_key=env.get("AZURE_OPENAI_API_KEY", ""),
            agent_deployment=env.get("FOOTNOTE_AGENT_DEPLOYMENT", "footnote-agent"),
            grader_deployment=env.get("FOOTNOTE_GRADER_DEPLOYMENT", "footnote-grader"),
            cache=cache_dir(),
        )
