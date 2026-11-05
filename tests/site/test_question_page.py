from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from support import QUESTIONS, U, prediction, publish, write_run

from footnote.answers import AgentAnswer, FinalValue
from footnote.eval.grade import Verdict
from footnote.eval.records import append_prediction, append_verdict
from footnote.prediction import Prediction, Step
from footnote.site.build import build_site

QUOTE = "Net sales were 1,234 million dollars"
LEGEND = (
    "Graded by code: the number is compared with the answer key. "
    "Graded by a model: a separate model compares the meaning."
)


def _page(
    tmp_path: Path,
    inject: dict[str, Any],
    system: str = "agent",
    *,
    pred: Prediction | None = None,
    verdict: Verdict | None = None,
    **kw: Any,
) -> str:
    root = tmp_path / "results"
    write_run(root, "run-dev-1", system, "dev", ["q1"], quote=QUOTE, **kw)
    run = root / "run-dev-1"
    if pred is not None:
        (run / "predictions.jsonl").unlink()
        append_prediction(run, pred)
    if verdict is not None:
        (run / "verdicts.jsonl").unlink()
        append_verdict(run, verdict)
    publish(root, ["run-dev-1"])
    out = tmp_path / "out"
    build_site(root, out, **inject)
    return (out / "q" / "q1.html").read_text()


def _declined(
    system: str, trace: list[Step], answer: str = "", reason: str | None = "why"
) -> Prediction:
    ans = AgentAnswer(
        answer=answer, final_value=None, declined=True, decline_reason=reason, citations=[]
    )
    return Prediction("q1", system, "v1", ans, "declined", None, [], trace, U, 1.0)


def test_final_value_is_its_own_line_before_the_grader_line(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _page(tmp_path, inject, final_value=FinalValue(value=Decimal(1577), unit="usd"))
    assert "Final value: 1,577 (usd, in the scale the question asks for)" in html
    assert html.index("Final value: 1,577") < html.index("Graded by code: r")


def test_no_final_number_line_and_the_disclosure_stay(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _page(tmp_path, inject)
    assert "No final number</p>" in html
    assert "No final number was given; only the quotes were checked." in html


def test_the_legend_and_plain_grader_labels(tmp_path: Path, inject: dict[str, Any]) -> None:
    html = _page(tmp_path, inject)
    assert LEGEND in " ".join(html.split())
    assert "Grader (" not in html and "Graded by code: r" in html
    llm = _page(tmp_path / "b", inject, verdict=Verdict("q1", "correct", "llm", "same meaning", U))
    assert "Graded by a model: same meaning" in llm
    none = _page(
        tmp_path / "c",
        inject,
        verdict=Verdict("q1", "error", "none", "the run ended in an error", U),
    )
    assert "Not graded: the run ended in an error" in none


def test_trace_shows_only_tool_steps(tmp_path: Path, inject: dict[str, Any]) -> None:
    base = prediction(QUESTIONS[0], "agent", quote=QUOTE)
    steps = [
        Step("chat", {"attempt": 1}, "finish_reason=stop", U),
        Step("pack", {"last_page_included": 3}, "packed", U),
        Step("search_filing", {"query": "revenue"}, "found 3 pages", U),
        Step("read_page", {"page_index": 4}, "text", U),
        Step("submit_answer", {}, "verified", U),
    ]
    html = _page(tmp_path, inject, pred=replace(base, trace=steps))
    trace = html.split('<ol class="trace">')[1].split("</ol>")[0]
    assert "finish_reason" not in trace and "last_page_included" not in trace
    assert ">search<" in trace and ">read page<" in trace and ">submit<" in trace
    assert "Trace (3 steps)" in html


def test_no_trace_block_when_no_tool_step_exists(tmp_path: Path, inject: dict[str, Any]) -> None:
    base = prediction(QUESTIONS[0], "long-context", quote=QUOTE)
    html = _page(tmp_path, inject, pred=replace(base, trace=[Step("chat", {}, "stop", U)]))
    assert "<details>" not in html


REJECT = "verification failed: quote not found on page 4: 'x'"


def test_own_decline_names_the_system_and_gives_the_reason(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    pred = _declined("plain-retrieval", [Step("search", {}, "found", U)], reason="no basis")
    html = _page(tmp_path, inject, "plain-retrieval", pred=pred, labels=["declined"])
    assert "Plain retrieval did not give an answer" in html and "no basis" in html
    assert "Partial answer" not in html


def test_partial_answer_text_is_labelled_not_scored(tmp_path: Path, inject: dict[str, Any]) -> None:
    pred = _declined("plain-retrieval", [], answer="At least two material items", reason="thin")
    html = _page(tmp_path, inject, "plain-retrieval", pred=pred, labels=["declined"])
    assert "Partial answer (not scored as an answer)" in html
    assert "At least two material items" in html


def test_a_withheld_answer_says_the_code_check_failed_and_why(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    trace = [Step("submit_answer", {}, REJECT, U), Step("submit_answer", {}, REJECT, U)]
    pred = replace(
        _declined("agent", trace, reason=REJECT),
        verify_ok=False,
        verify_problems=["quote not found on page 4: 'x'; this exact text is on page_index 9"],
    )
    html = _page(tmp_path, inject, pred=pred, labels=["declined"])
    assert "The answer failed the code check twice, so it was withheld" in html
    assert "Cited page does not match: this text is on page 10, not page 5" in html
    assert "page_index" not in html and "did not give an answer" not in html


def test_one_rejection_then_a_stop_is_stated_as_that(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    trace = [Step("submit_answer", {}, REJECT, U), Step("decline", {}, "call limit", U)]
    pred = replace(_declined("agent", trace), verify_ok=False, verify_problems=["no citations"])
    html = _page(tmp_path, inject, pred=pred, labels=["declined"])
    assert "The answer failed the code check and the run stopped before a retry" in html


def test_title_description_and_hyphenated_form(tmp_path: Path, inject: dict[str, Any]) -> None:
    html = _page(tmp_path, inject)
    assert "<title>Acme 10-K 2022: What was Acme revenue? | Footnote</title>" in html
    assert '<meta name="description" content="What was Acme revenue?">' in html
    assert "Acme, 10-K 2022." in html and "Acme, 10K" not in html


def test_question_heading_is_smaller_on_phones() -> None:
    css = (Path(__file__).resolve().parents[2] / "src/footnote/site/style.css").read_text()
    assert "@media (max-width" in css and ".question h1" in css
