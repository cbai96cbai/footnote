from dataclasses import replace
from typing import Any

from support import QUESTIONS, prediction

from footnote.budget.limits import LIMITS
from footnote.prediction import Step
from footnote.site.headline import headline
from footnote.site.runs import Run

CHECK = "gives an answer only when its quotes and final number pass the code check"


def run(
    system: str,
    accuracy: float | None,
    verified: float | None,
    cost: float = 0.01,
    split: str = "test",
    version: str = "v1",
    n: int = 93,
) -> Run:
    summary: dict[str, Any] = {
        "accuracy": accuracy,
        "verified_rate": verified,
        "cost_per_question": cost,
        "n": n,
    }
    return Run(f"{system}-{split}-{version}", system, split, version, summary, {}, {})


def test_agent_highest_names_the_split_and_count() -> None:
    runs = [
        run("agent", 0.87, 1.0, 0.009),
        run("plain-retrieval", 0.45, 0.37, 0.002),
        run("long-context", 0.72, 0.20, 0.027),
    ]
    assert headline(runs) == (
        f"On the 93 test questions, the agent scores highest (87%) and {CHECK}; "
        "20% of Long context answers pass the same check. "
        "The agent costs 33% as much per question as Long context."
    )


def test_a_baseline_highest() -> None:
    runs = [
        run("agent", 0.72, 1.0, 0.0077, split="dev"),
        run("plain-retrieval", 0.45, 0.37, 0.002, split="dev"),
        run("long-context", 0.87, 0.20, 0.0271, split="dev"),
        run("no-docs", 0.33, None, 0.001, split="dev"),
    ]
    assert headline(runs) == (
        "On the 93 development questions, Long context scores highest (87%). "
        f"The agent scores 72% and {CHECK}; 20% of Long context answers pass the same check. "
        "The agent costs 28% as much per question as Long context."
    )


def test_ties_name_every_system_that_shares_the_top() -> None:
    runs = [run("agent", 0.8, 1.0, 0.01), run("long-context", 0.8, 0.2, 0.03)]
    text = headline(runs) or ""
    assert text.startswith(
        "On the 93 test questions, the agent and Long context tie for the highest score (80%)."
    )
    assert f"The agent {CHECK}; 20% of Long context answers pass the same check." in text
    assert "The agent scores" not in text
    three = [
        run("agent", 0.8, 1.0),
        run("long-context", 0.8, 0.2),
        run("plain-retrieval", 0.8, 0.4),
    ]
    assert (headline(three) or "").startswith(
        "On the 93 test questions, the agent, Plain retrieval and Long context tie for the "
        "highest score (80%)."
    )


def test_missing_verified_rates_drop_only_their_clauses() -> None:
    runs = [run("agent", 0.6, None, 0.01), run("long-context", 0.9, 0.2, 0.03)]
    assert headline(runs) == (
        "On the 93 test questions, Long context scores highest (90%). "
        "The agent scores 60%. The agent costs 33% as much per question as Long context."
    )
    runs = [run("agent", 0.6, 1.0, 0.01), run("long-context", 0.9, None, 0.03)]
    text = headline(runs) or ""
    assert f"The agent scores 60% and {CHECK}." in text and "pass the same check" not in text


def test_no_agent_or_no_accuracy() -> None:
    assert headline([run("long-context", 0.9, 0.2)]) == (
        "On the 93 test questions, Long context scores highest (90%)."
    )
    assert headline([]) is None
    assert headline([run("agent", None, 1.0)]) is None


def test_cost_is_a_whole_percent_of_the_best_other_system() -> None:
    def text(agent_cost: float) -> str:
        return (
            headline([run("agent", 0.5, None, agent_cost), run("long-context", 0.9, None, 0.04)])
            or ""
        )

    assert "The agent costs 50% as much per question as Long context." in text(0.02)
    assert "The agent costs 200% as much" in text(0.08)
    real = headline([run("agent", 0.5, None, 0.0077), run("long-context", 0.9, None, 0.0271)])
    assert "The agent costs 28% as much" in (real or "")


def test_published_test_runs_rank_ahead_of_development_runs() -> None:
    runs = [
        run("agent", 0.9, 1.0, split="dev"),
        run("agent", 0.5, 1.0),
        run("long-context", 0.6, 0.2),
    ]
    text = headline(runs) or ""
    assert "93 test questions, Long context scores highest (60%)" in text and "90%" not in text


def test_several_versions_rank_the_latest_of_each_system_and_say_so() -> None:
    runs = [
        run("agent", 0.95, 1.0, split="dev", version="v1"),
        run("agent", 0.60, 1.0, split="dev", version="v2"),
        run("long-context", 0.80, 0.2, split="dev"),
    ]
    text = headline(runs) or ""
    assert text.startswith("On the 93 development questions, Long context scores highest (80%).")
    assert "The agent scores 60%" in text and "95%" not in text


def _truncated(r: Run, k: int, total: int) -> Run:
    preds = {}
    for i in range(total):
        p = prediction(QUESTIONS[0], "long-context")
        pack = Step(
            "pack",
            {"pages_in_filing": 382, "pages_included": 220, "truncated": i < k},
            "x",
            p.usage,
        )
        preds[f"q{i}"] = replace(p, question_id=f"q{i}", trace=[pack])
    return replace(r, predictions=preds)


def test_truncated_long_context_filings_are_disclosed_from_the_record() -> None:
    lc = _truncated(run("long-context", 0.87, 0.2), 12, 93)
    text = headline([run("agent", 0.72, 1.0), lc]) or ""
    window = f"{LIMITS['long_context'].max_input_tokens:,}"
    assert (
        "Long context could read only part of 12 of 93 filings, which were longer than its "
        f"{window}-token window." in text
    )
    assert "whole filing" not in text
    untruncated = _truncated(run("long-context", 0.87, 0.2), 0, 93)
    assert "could read only part" not in (headline([untruncated]) or "")
