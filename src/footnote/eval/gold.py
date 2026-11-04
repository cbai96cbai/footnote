"""Gold answers and evidence. The only module that exposes them."""

from dataclasses import dataclass
from typing import Any

from footnote._financebench import read_rows


@dataclass(frozen=True)
class Evidence:
    doc_name: str
    page_index: int
    text: str
    full_page: str


@dataclass(frozen=True)
class Gold:
    id: str
    answer: str
    justification: str
    evidence: list[Evidence]


def _gold(row: dict[str, Any]) -> Gold:
    return Gold(
        id=row["financebench_id"],
        answer=row["answer"],
        justification=row["justification"],
        evidence=[
            Evidence(
                doc_name=e["doc_name"],
                page_index=int(e["evidence_page_num"]),
                text=e["evidence_text"],
                full_page=e["evidence_text_full_page"],
            )
            for e in row["evidence"]
        ],
    )


def load_gold() -> dict[str, Gold]:
    return {g.id: g for g in map(_gold, read_rows())}
