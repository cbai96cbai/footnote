from pathlib import Path
from typing import Any

from support import publish, write_run

from footnote.site.build import build_site


def _build(root: Path, tmp_path: Path, inject: dict[str, Any]) -> Path:
    out = tmp_path / "out"
    build_site(root, out, **inject)
    return out


def test_pages_come_from_the_union_of_test_and_development_runs(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1"])
    write_run(root, "agent-dev-1", "agent", "dev", ["q2"], version="agent-1")
    publish(root, ["agent-test-1", "agent-dev-1"])
    out = _build(root, tmp_path, inject)
    assert (out / "q" / "q1.html").exists() and (out / "q" / "q2.html").exists()
    questions = (out / "questions.html").read_text()
    assert 'href="q/q1.html"' in questions and 'href="q/q2.html"' in questions


def test_an_overlapping_question_shows_every_runs_prediction_labelled(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1"])
    write_run(root, "agent-dev-1", "agent", "dev", ["q1", "q2"], version="agent-1")
    write_run(root, "agent-dev-2", "agent", "dev", ["q1"], version="agent-2")
    write_run(root, "nodocs-test-1", "no-docs", "test", ["q1"])
    publish(root, ["agent-test-1", "agent-dev-1", "agent-dev-2", "nodocs-test-1"])
    out = _build(root, tmp_path, inject)
    q1 = (out / "q" / "q1.html").read_text()
    for label in (
        "Footnote agent · test run",
        "Footnote agent · development run (agent-1)",
        "Footnote agent · development run (agent-2)",
        "Without documents · test run",
    ):
        assert f"{label} <span" in q1, label
    q2 = (out / "q" / "q2.html").read_text()
    assert "development run (agent-1)" in q2 and "agent-2" not in q2 and "test run" not in q2


def test_health_care_questions_are_scored_but_never_shown(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = _build(results, tmp_path, inject)
    assert not (out / "q" / "q3.html").exists()
    questions = (out / "questions.html").read_text()
    assert "q/q3.html" not in questions and "Pfizer" not in questions
    assert "1 question about Health Care companies is included in the scores but not shown." in (
        " ".join(questions.split())
    )
    assert "Pfizer" not in (out / "failures.html").read_text()


def test_the_health_care_count_is_plural_and_absent_when_zero(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"])
    publish(root, ["agent-dev-1"])
    assert "Health Care" not in (_build(root, tmp_path, inject) / "questions.html").read_text()
    inject["questions"] = inject["questions"] + [
        type(inject["questions"][2])("q4", "Pfizer", "PFE_2022_10K", "x", "Another Pfizer one?")
    ]
    inject["gold"] = {**inject["gold"], "q4": inject["gold"]["q3"]}
    write_run(root, "agent-dev-1", "agent", "dev", [inject["questions"][2], inject["questions"][3]])
    html = " ".join(
        (_build(root, tmp_path / "again", inject) / "questions.html").read_text().split()
    )
    assert (
        "2 questions about Health Care companies are included in the scores but not shown." in html
    )
