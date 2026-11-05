"""Behaviour added after the first review: honest labels, hidden rows, stacked cards, copy."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from support import QUESTIONS, U, prediction, publish, write_run

from footnote.answers import AgentAnswer, Citation, FinalValue
from footnote.eval.grade import Verdict
from footnote.eval.records import append_prediction, append_verdict
from footnote.prediction import Prediction, Step
from footnote.site.build import build_site
from footnote.site.filters import step_arguments
from footnote.site.wording import final_value_line, problem_notes
from footnote.verify import verify_answer

QUOTE = "Net sales were 1,234 million dollars"
ROOT = Path(__file__).resolve().parents[2] / "src/footnote/site"


def _site(root: Path, tmp_path: Path, inject: dict[str, Any]) -> Path:
    out = tmp_path / "out"
    build_site(root, out, **inject)
    return out


def _swap(
    root: Path, run_id: str, pred: Prediction | None = None, verdict: Verdict | None = None
) -> None:
    run = root / run_id
    if pred is not None:
        (run / "predictions.jsonl").unlink()
        append_prediction(run, pred)
    if verdict is not None:
        (run / "verdicts.jsonl").unlink()
        append_verdict(run, verdict)


# H4: failure reasons and hidden rows
def test_a_system_without_pages_is_never_said_to_have_read_the_right_ones(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = results
    write_run(
        root,
        "nodocs-dev-2",
        "no-docs",
        "dev",
        ["q1", "q2"],
        version="nodocs-2",
        labels=["incorrect"] * 2,
    )
    inject["gold"] = inject["gold"]
    publish(root, ["agent-dev-1", "nodocs-dev-1", "nodocs-dev-2"])
    for run_id in ("nodocs-dev-2",):
        verdicts = root / run_id / "verdicts.jsonl"
        verdicts.unlink()
        for q in ("q1", "q2"):
            append_verdict(root / run_id, Verdict(q, "incorrect", "llm", "r", U))
    html = (_site(root, tmp_path, inject) / "failures.html").read_text()
    assert "the answer was graded incorrect" in html
    assert "right pages were read" not in html


def test_hidden_health_care_failures_are_counted_and_never_read_as_a_clean_run(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1", "q3"], labels=["correct", "incorrect"])
    publish(root, ["agent-dev-1"])
    html = " ".join((_site(root, tmp_path, inject) / "failures.html").read_text().split())
    assert "No failed questions shown." in html
    assert "1 failed question about Health Care companies is not shown." in html
    assert "No failed questions in this run." not in html


# H5
def test_unchecked_answer_with_no_final_number_does_not_claim_the_quotes_were_checked(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "nodocs-dev-1", "no-docs", "dev", ["q1"], quote=QUOTE, verify_ok=None)
    publish(root, ["nodocs-dev-1"])
    html = (_site(root, tmp_path, inject) / "q" / "q1.html").read_text()
    assert "No final number was given; this system's answers are not checked." in html
    assert "only the quotes were checked" not in html


# H6: producer-built
def test_a_negative_cited_page_reads_in_plain_words() -> None:
    ans = AgentAnswer(
        answer="x", final_value=None, declined=False, decline_reason=None,
        citations=[Citation(page_index=-1, quote=QUOTE)],
    )  # fmt: skip
    problems = verify_answer(ans, {0: QUOTE}, "q", [])[1]
    assert problem_notes(problems) == ["Cited page 0 is not in the filing; this text is on page 1"]


def test_end_to_end_the_check_to_the_page_wording() -> None:
    ans = AgentAnswer(
        answer="x", final_value=None, declined=False, decline_reason=None,
        citations=[Citation(page_index=3, quote=QUOTE)],
    )  # fmt: skip
    problems = verify_answer(ans, {3: "none", 7: QUOTE}, "q", [])[1]
    assert problem_notes(problems) == [
        "Cited page does not match: this text is on page 8, not page 4"
    ]


# H8
def test_questions_become_stacked_cards_on_phones() -> None:
    css = (ROOT / "style.css").read_text()
    media = css.split("@media (max-width: 37.5rem)")[1]
    assert "table.questions tr" in media and "display: block" in media
    assert "table.questions thead" in media and "td.company" in media


# H10 / J5
def test_trace_arguments_are_a_short_readable_phrase() -> None:
    assert step_arguments({"page_index": 4, "x": 1}) == "page 5, x 1"
    assert step_arguments({"query": "net sales"}) == '"net sales"'
    assert step_arguments({"expression": "1+2"}) == "1+2"
    assert step_arguments({"last_page": [1, 2]}) == "last page (2 items)"


def test_the_unit_other_has_no_label() -> None:
    assert final_value_line(FinalValue(value=Decimal(0), unit="other")) == "Final value: 0"
    assert final_value_line(FinalValue(value=Decimal("1.5"), unit="ratio")) == (
        "Final value: 1.5 (ratio)"
    )


def test_a_long_reference_answer_is_shown_in_full(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    from html import unescape

    gold = inject["gold"]["q1"].answer
    assert len(gold) > 400
    page = unescape((_site(results, tmp_path, inject) / "q" / "q1.html").read_text())
    assert " ".join(gold.split()) in " ".join(page.split())
    assert "…" not in page.split("Reference answer")[1].split("</p>")[0]


# H1: truncation on the long-context question section
def test_a_cut_filing_is_stated_on_the_question_page(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "longctx-dev-1", "long-context", "dev", ["q1"])
    base = prediction(QUESTIONS[0], "long-context", quote=QUOTE)
    pack = Step("pack", {"pages_in_filing": 382, "pages_included": 220, "truncated": True}, "x", U)
    _swap(root, "longctx-dev-1", replace(base, trace=[pack]))
    publish(root, ["longctx-dev-1"])
    html = (_site(root, tmp_path, inject) / "q" / "q1.html").read_text()
    assert "it was given 220 of 382 pages" in html
    assert "whole filing" not in html.lower()


# H11, H12
def test_about_does_not_imply_a_test_run_when_none_is_published(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    about = " ".join((_site(results, tmp_path, inject) / "about.html").read_text().split())
    assert "No test-split run has been published yet" in about
    assert "ran once" not in about
    root = tmp_path / "t" / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1"])
    publish(root, ["agent-test-1"])
    about = " ".join((_site(root, tmp_path / "t", inject) / "about.html").read_text().split())
    assert "rest on the test split, which each system family ran once" in about


def test_the_link_after_the_headline_note_is_separated_by_a_space(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    index = (_site(results, tmp_path, inject) / "index.html").read_text()
    assert "published yet. <a" in " ".join(index.split()) and "yet.<a" not in index


def test_question_pages_show_a_verdict_strip_under_the_heading(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = (_site(results, tmp_path, inject) / "q" / "q1.html").read_text()
    strip = html.split('<p class="strip">')[1].split("</p>")[0]
    assert strip.count('class="badge') == 2 and html.index("strip") < html.index("Reference answer")


def test_the_note_says_the_agent_rate_is_by_design(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    flat = " ".join((_site(results, tmp_path, inject) / "index.html").read_text().split())
    assert "The agent answers only once its answer passes, so its rate is 100% by design" in flat
    assert "the other systems are checked after the fact" in flat


def test_the_front_page_link_states_the_hidden_health_care_count(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    index = " ".join((_site(results, tmp_path, inject) / "index.html").read_text().split())
    assert (
        "Browse the questions (1 about Health Care companies is scored but not shown)</a>" in index
    )
    root = tmp_path / "r2" / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"])
    publish(root, ["agent-dev-1"])
    other = " ".join((_site(root, tmp_path / "r2", inject) / "index.html").read_text().split())
    assert "Browse the questions</a>" in other


def test_code_grading_reads_answer_key_not_gold() -> None:
    from footnote.site.wording import grader_note

    assert grader_note("gold '1.73' vs final value 1.73 ratio") == (
        "answer key '1.73' vs final value 1.73 ratio"
    )
    assert grader_note("The answer matches the reference.") == "The answer matches the reference."
