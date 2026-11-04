from pathlib import Path
from typing import Any

import pytest
from support import DOCS, GOLD, LINKS, QUESTIONS, write_run


@pytest.fixture
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1", "q2", "q3"])
    write_run(root, "nodocs-dev-1", "no-docs", "dev", ["q1", "q2", "q3"])
    return root


@pytest.fixture
def inject() -> dict[str, Any]:
    return {
        "questions": QUESTIONS,
        "gold": GOLD,
        "docs": DOCS,
        "link_status": LINKS,
        "featured": ["q1", "q3"],
        "failures": [{"id": "q2", "system": "agent", "note": "Picked the wrong year."}],
    }
