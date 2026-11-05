from typing import Any

from footnote.site.headline import headline
from footnote.site.runs import Run


def run(system: str, accuracy: float | None, verified: float | None, cost: float = 0.01) -> Run:
    summary: dict[str, Any] = {
        "accuracy": accuracy,
        "verified_rate": verified,
        "cost_per_question": cost,
    }
    return Run(f"{system}-1", system, "test", "v1", summary, {}, {})


def test_agent_highest() -> None:
    runs = [
        run("agent", 0.87, 1.0, 0.009),
        run("plain-retrieval", 0.45, 0.37, 0.002),
        run("long-context", 0.72, 0.20, 0.027),
    ]
    assert headline(runs) == (
        "The agent scores highest (87%), and 100% of its answers passed the code check "
        "(whole filing: 20%), at about a third of the cost per question of reading the whole "
        "filing."
    )


def test_a_baseline_highest_reads_as_the_example() -> None:
    runs = [
        run("agent", 0.72, 1.0, 0.009),
        run("plain-retrieval", 0.45, 0.37, 0.002),
        run("long-context", 0.87, 0.20, 0.027),
        run("no-docs", 0.33, None, 0.001),
    ]
    assert headline(runs) == (
        "Reading the whole filing scores highest (87%). The agent scores 72%, and 100% of its "
        "answers passed the code check (whole filing: 20%), at about a third of the cost per "
        "question."
    )


def test_ties_name_every_system_that_shares_the_top() -> None:
    runs = [run("agent", 0.8, 1.0, 0.01), run("long-context", 0.8, 0.2, 0.03)]
    text = headline(runs)
    assert text is not None
    assert text.startswith(
        "The agent and reading the whole filing tie for the highest score (80%)."
    )
    assert "The agent scores" not in text
    assert "100% of the agent's answers passed the code check (whole filing: 20%)" in text
    three = [
        run("agent", 0.8, 1.0),
        run("long-context", 0.8, 0.2),
        run("plain-retrieval", 0.8, 0.4),
    ]
    assert (headline(three) or "").startswith(
        "The agent, plain retrieval and reading the whole filing tie for the highest score (80%)."
    )


def test_missing_verified_rate_drops_that_clause_only() -> None:
    runs = [run("agent", 0.6, None, 0.01), run("long-context", 0.9, 0.2, 0.03)]
    assert headline(runs) == (
        "Reading the whole filing scores highest (90%). The agent scores 60%, "
        "at about a third of the cost per question."
    )
    runs = [run("agent", 0.6, 1.0, 0.01), run("long-context", 0.9, None, 0.03)]
    text = headline(runs) or ""
    assert (
        "100% of its answers passed the code check, at about" in text
        and "whole filing:" not in text
    )


def test_no_agent_or_no_accuracy() -> None:
    assert (
        headline([run("long-context", 0.9, 0.2)])
        == "Reading the whole filing scores highest (90%)."
    )
    assert headline([]) is None
    assert headline([run("agent", None, 1.0)]) is None


def test_cost_is_compared_with_the_best_other_system_and_rounded_honestly() -> None:
    runs = [run("agent", 0.5, None, 0.0077), run("long-context", 0.9, None, 0.0271)]
    assert "at about a quarter of the cost per question." in (headline(runs) or "")


def test_cost_wording_follows_the_ratio() -> None:
    def text(agent_cost: float) -> str:
        return (
            headline([run("agent", 0.5, None, agent_cost), run("long-context", 0.9, None, 0.04)])
            or ""
        )

    assert "at about half the cost per question" in text(0.02)
    assert "at about the same cost per question" in text(0.04)
    assert "at about 2 times the cost per question" in text(0.08)
    assert "at about a quarter of the cost per question" in text(0.01)
    assert "at about 60% of the cost per question" in text(0.024)


def test_the_latest_version_of_a_system_is_used() -> None:
    old = run("agent", 0.1, 0.1)
    new = Run("agent-2", "agent", "dev", "v2", {"accuracy": 0.7, "verified_rate": 1.0}, {}, {})
    text = headline([old, new, run("long-context", 0.5, 0.2)]) or ""
    assert "(70%)" in text and "10%" not in text
