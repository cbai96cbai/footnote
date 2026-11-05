"""The one-sentence result above the chart, written from the published runs' own numbers."""

from collections.abc import Callable

from footnote.budget.limits import LIMITS
from footnote.site.runs import LABELS, ORDER, Run

CHECK = "gives an answer only when its quotes and final number pass the code check"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _name(system: str) -> str:
    return "the agent" if system == "agent" else LABELS.get(system, system)


def _basis(runs: list[Run]) -> tuple[list[Run], str]:
    """The runs the sentence ranks: the test split when published, else the development split
    (never a gate run), the latest published version of each system. Runs without an accuracy
    are left out."""
    graded = [r for r in runs if r.summary.get("accuracy") is not None]
    test = [r for r in graded if r.split == "test"]
    dev = [r for r in graded if r.split == "dev"]
    chosen, split = (test, "test") if test else (dev, "development")
    latest = {r.system: r for r in chosen}  # published order: a later run replaces an earlier one
    ordered = sorted(
        latest.values(), key=lambda r: ORDER.index(r.system) if r.system in ORDER else 99
    )
    return ordered, split


def _joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _agent_part(agent: Run, others: list[Run], lead: str | None) -> str | None:
    """What the agent's answers must pass, set beside how the best-checked baseline does."""
    if agent.summary.get("verified_rate") is None:
        return lead
    part = f"{lead} and {CHECK}" if lead else f"The agent {CHECK}"
    ref = next(
        (
            r
            for s in ("long-context", "plain-retrieval")
            for r in others
            if r.system == s and r.summary.get("verified_rate") is not None
        ),
        None,
    )
    if ref is not None:
        rate = _pct(ref.summary["verified_rate"])
        part += f"; {rate} of {_name(ref.system)} answers pass the same check"
    return part


def _cost(agent: Run, others: list[Run]) -> str | None:
    ref = max(others, key=lambda r: r.summary["accuracy"], default=None)
    a = agent.summary.get("cost_per_question")
    b = ref.summary.get("cost_per_question") if ref else None
    if ref is None or not a or not b:
        return None
    return f"The agent costs {round(a / b * 100)}% as much per question as {_name(ref.system)}."


def _cut_short(scored: list[Run], filing_of: Callable[[str], str]) -> str | None:
    lc = next((r for r in scored if r.system == "long-context" and r.truncated_questions), None)
    if lc is None:
        return None
    questions = lc.truncated_questions
    filings = len({filing_of(q) for q in questions})
    window = LIMITS["long_context"].max_input_tokens
    return (
        f"For {len(questions)} of the {len(lc.predictions)} questions "
        f"({filings} {'filing' if filings == 1 else 'filings'}), the filing was longer than "
        f"{LABELS['long-context']}'s {window:,}-token window, so it read only part of it."
    )


def headline(runs: list[Run], filing_of: Callable[[str], str] = str) -> str | None:
    """Which system scores highest and how the agent compares, for the runs it names."""
    scored, split = _basis(runs)
    if not scored:
        return None
    count = max(r.summary.get("n") or 0 for r in scored)
    what = f"{count} {split} questions" if count else f"{split} split"
    if split == "development":
        what += ", comparing the latest version of each system"
    opener = f"On the {what}, "
    best = max(r.summary["accuracy"] for r in scored)
    top = [r for r in scored if r.summary["accuracy"] == best]
    agent = next((r for r in scored if r.system == "agent"), None)
    others = [r for r in scored if r is not agent]
    names = _joined([_name(r.system) for r in top])
    verb = "scores highest" if len(top) == 1 else "tie for the highest score"
    sentences: list[str] = []
    lead: str | None
    if agent is not None and top == [agent]:
        lead = f"{opener}the agent scores highest ({_pct(best)})"
        sentences.append(f"{_agent_part(agent, others, lead)}.")
    else:
        sentences.append(f"{opener}{names} {verb} ({_pct(best)}).")
        if agent is not None:
            lead = None if agent in top else f"The agent scores {_pct(agent.summary['accuracy'])}"
            if part := _agent_part(agent, others, lead):
                sentences.append(f"{part}.")
    if agent is not None and (cost := _cost(agent, others)):
        sentences.append(cost)
    if cut := _cut_short(scored, filing_of):
        sentences.append(cut)
    return " ".join(sentences)
