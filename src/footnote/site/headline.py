"""The one-sentence result above the chart, written from the published runs' own numbers."""

from footnote.site.runs import LABELS, ORDER, Run

SUBJECT = {
    "agent": "the agent",
    "plain-retrieval": "plain retrieval",
    "long-context": "reading the whole filing",
    "no-docs": "answering without documents",
}
SHORT = {"long-context": "whole filing", "plain-retrieval": "plain retrieval"}
FRACTIONS = (
    (0.5, "half the"),
    (1 / 3, "a third of the"),
    (0.25, "a quarter of the"),
    (0.2, "a fifth of the"),
)
CLOSE = 0.1  # costs within this share of each other are "about the same"
NEAR = 0.15  # a ratio this close to a simple fraction is called by that fraction


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _subject(system: str) -> str:
    return SUBJECT.get(system, LABELS.get(system, system).lower())


def _latest(runs: list[Run]) -> list[Run]:
    """One run per system with an accuracy: the last published, in display order."""
    last = {r.system: r for r in runs if r.summary.get("accuracy") is not None}
    return sorted(last.values(), key=lambda r: ORDER.index(r.system) if r.system in ORDER else 99)


def _names(systems: list[str]) -> str:
    names = [_subject(s) for s in systems]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _cost_words(agent: float, other: float) -> str | None:
    if agent <= 0 or other <= 0:
        return None
    r = agent / other
    if abs(r - 1) <= CLOSE:
        return "at about the same cost per question"
    if r > 1:
        times = f"{r:.1f}".removesuffix(".0")
        return f"at about {times} times the cost per question"
    value, word = min(FRACTIONS, key=lambda f: abs(f[0] - r))
    if abs(value - r) / value <= NEAR:
        return f"at about {word} cost per question"
    return f"at about {_pct(r)} of the cost per question"


def _agent_sentence(
    agent: Run, top: list[Run], others: list[Run], lead: str | None, in_top: bool
) -> str | None:
    rate = agent.summary.get("verified_rate")
    verified = None
    if rate is not None:
        subject = "its" if lead else "the agent's"
        verified = f"{_pct(rate)} of {subject} answers passed the code check"
        for system in ("long-context", "plain-retrieval"):
            ref = next((r for r in others if r.system == system), None)
            if ref is not None and ref.summary.get("verified_rate") is not None:
                verified += f" ({SHORT[system]}: {_pct(ref.summary['verified_rate'])})"
                break
    cost = None
    ref_cost = max(others, key=lambda r: r.summary["accuracy"], default=None)
    if ref_cost is not None and agent.summary.get("cost_per_question") is not None:
        cost = _cost_words(
            agent.summary["cost_per_question"], ref_cost.summary.get("cost_per_question") or 0
        )
        if cost and in_top:
            joiner = " as " if cost.startswith("at about the same") else " of "
            cost += f"{joiner}{_subject(ref_cost.system)}"
    if lead is None and verified is None:
        return None
    parts = [lead] if lead else []
    if verified:
        parts.append(f"and {verified}" if lead else verified)
    sentence = ", ".join(parts)
    return f"{sentence}, {cost}." if cost else f"{sentence}."


def headline(runs: list[Run]) -> str | None:
    """Which system scores highest, and how the agent compares; None without a graded run."""
    scored = _latest(runs)
    if not scored:
        return None
    best = max(r.summary["accuracy"] for r in scored)
    top = [r for r in scored if r.summary["accuracy"] == best]
    agent = next((r for r in scored if r.system == "agent"), None)
    others = [r for r in scored if r is not agent]
    score = _pct(best)
    if agent is None or agent not in top:
        first = f"{_names([r.system for r in top])} "
        first += (
            f"scores highest ({score})."
            if len(top) == 1
            else f"tie for the highest score ({score})."
        )
        first = first[0].upper() + first[1:]
        if agent is None:
            return first
        lead = f"The agent scores {_pct(agent.summary['accuracy'])}"
        return " ".join([first, _agent_sentence(agent, top, others, lead, False) or f"{lead}."])
    if len(top) == 1:
        return (
            _agent_sentence(agent, top, others, f"The agent scores highest ({score})", True) or ""
        )
    tie = _names([r.system for r in top])
    first = f"{tie[0].upper()}{tie[1:]} tie for the highest score ({score})."
    rest = _agent_sentence(agent, top, others, None, True)
    return f"{first} {rest}" if rest else first
