"""FinanceBench questions as the agent side may see them: no answers, justifications or evidence."""

from dataclasses import dataclass
from typing import Any

from footnote import _financebench as _fb


@dataclass(frozen=True)
class AgentQuestion:
    id: str
    company: str
    doc_name: str
    question_type: str
    question: str


@dataclass(frozen=True)
class DocInfo:
    doc_name: str
    company: str
    gics_sector: str
    doc_type: str
    doc_period: int
    doc_link: str


def _question(row: dict[str, Any]) -> AgentQuestion:
    return AgentQuestion(
        id=row["financebench_id"],
        company=row["company"],
        doc_name=row["doc_name"],
        question_type=row["question_type"],
        question=row["question"],
    )


def load_questions() -> list[AgentQuestion]:
    return [_question(r) for r in _fb.read_rows()]


def doc_info() -> dict[str, DocInfo]:
    out: dict[str, DocInfo] = {}
    for r in _fb.read_rows():
        out.setdefault(
            r["doc_name"],
            DocInfo(
                doc_name=r["doc_name"],
                company=r["company"],
                gics_sector=r["gics_sector"],
                doc_type=r["doc_type"],
                doc_period=int(r["doc_period"]),
                doc_link=r["doc_link"],
            ),
        )
    return out
