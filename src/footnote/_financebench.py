"""Raw FinanceBench rows. Private: only `benchmark` (agent-safe fields) and `eval.gold` read it."""

import json
from typing import Any

from huggingface_hub import hf_hub_download

from footnote.paths import cache_dir

REPO = "PatronusAI/financebench"
FILENAME = "financebench_merged.jsonl"
REVISION = "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"


def read_rows() -> list[dict[str, Any]]:
    path = hf_hub_download(
        REPO,
        FILENAME,
        repo_type="dataset",
        revision=REVISION,
        cache_dir=cache_dir() / "hf",
    )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
