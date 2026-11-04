from pathlib import Path

import pytest

from footnote.config import Settings
from footnote.paths import cache_dir

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = ("FOOTNOTE_AGENT_DEPLOYMENT", "FOOTNOTE_GRADER_DEPLOYMENT", "FOOTNOTE_CACHE")


def test_empty_optional_settings_mean_their_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OPTIONAL:
        monkeypatch.setenv(name, "")
    s = Settings.from_env()
    assert s.agent_deployment == "footnote-agent"
    assert s.grader_deployment == "footnote-grader"
    assert cache_dir() == Path("~/.cache/footnote").expanduser()


def test_env_example_sets_no_optional_variable_to_empty() -> None:
    active = [
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    names = {n.strip() for n, _ in active}
    assert {"AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"} <= names
    assert not names & set(OPTIONAL), "optional settings stay commented out"
