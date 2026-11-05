"""Plain words for what a page shows: final values, filings, check problems, declines, steps."""

import re

from footnote.answers import FinalValue
from footnote.benchmark import DocInfo
from footnote.page_problems import parse_page_problem
from footnote.prediction import Prediction, Step
from footnote.site.filters import EXCERPT_LIMIT, excerpt
from footnote.site.links import form_name
from footnote.site.runs import pack_truncation

NOT_TRACED = "Final number not traced to the quotes"
NOT_CONFIRMED = "Quote not confirmed on the cited page"
# the tool steps a reader follows; chat and packing steps are plumbing
STEP_NAMES = {
    "search": "search",
    "search_filing": "search",
    "read_page": "read page",
    "calculate": "calculate",
    "submit_answer": "submit",
}
_REJECTED = "verification failed"
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
SCALED = frozenset({"usd", "count", "other"})  # answered in the scale the question asks for
MAX_LISTED = 3  # the quote check records at most this many other pages


def form(doc_type: str) -> str:
    return form_name(doc_type) or doc_type


def filing(doc: DocInfo) -> str:
    return f"{doc.company} {form(doc.doc_type)} {doc.doc_period}"


def final_value_line(fv: FinalValue | None) -> str:
    if fv is None:
        return "No final number"
    unit = f"{fv.unit}, in the scale the question asks for" if fv.unit in SCALED else fv.unit
    return f"Final value: {fv.value:,f} ({unit})"


def _pages(indexes: tuple[int, ...]) -> str:
    """Zero-based indexes as the site prints pages (index + 1). The check records at most three
    other pages, so a full list of three may be only part of the matches."""
    nums = [str(n + 1) for n in indexes]
    if len(nums) == 1:
        return f"page {nums[0]}"
    text = f"pages {', '.join(nums[:-1])} and {nums[-1]}"
    return f"at least {text}" if len(nums) == MAX_LISTED else text


def _page_note(cited: int, elsewhere: tuple[int, ...], page_exists: bool) -> str:
    if not page_exists:
        note = f"Cited page {cited + 1} is not in the filing"
        return f"{note}; this text is on {_pages(elsewhere)}" if elsewhere else note
    if elsewhere:
        return (
            f"Cited page does not match: this text is on {_pages(elsewhere)}, not page {cited + 1}"
        )
    return NOT_CONFIRMED


def problem_notes(problems: list[str]) -> list[str]:
    """The check's problems in plain words, each distinct note once, in order."""
    notes: list[str] = []
    for message in problems:
        page = parse_page_problem(message)
        if page is not None:
            note = _page_note(page.cited, page.elsewhere, page.page_exists)
        elif message.startswith("final value"):
            note = NOT_TRACED
        else:
            note = message
        if note not in notes:
            notes.append(note)
    return notes


def shown_steps(trace: list[Step]) -> list[tuple[str, Step]]:
    return [(STEP_NAMES[s.tool], s) for s in trace if s.tool in STEP_NAMES]


def decline_headline(p: Prediction, system_name: str) -> str:
    """What a decline says, from the trace only: its own decline, or rejected submissions."""
    submits = [s for s in p.trace if s.tool == "submit_answer"]
    rejected = [s for s in submits if s.summary.startswith(_REJECTED)]
    if (submits and submits[-1].summary == "declined") or not rejected:
        return f"{system_name} did not give an answer"
    if len(rejected) >= 2:
        return "The answer failed the code check twice, so it was withheld"
    after = p.trace[p.trace.index(rejected[0]) + 1 :]
    if any(s.tool in STEP_NAMES for s in after):
        return (
            "The answer failed the code check; a retry reply was made but it did not "
            "include a new submission"
        )
    return "The answer failed the code check and the run stopped before a retry"


def stop_reason(p: Prediction) -> str | None:
    """The reason the run recorded when it stopped (a call or context limit), if it did."""
    return next((s.summary for s in reversed(p.trace) if s.tool == "decline"), None)


def reference_excerpt(text: object, limit: int = EXCERPT_LIMIT) -> str:
    """A reference answer in full, or cut at the end of a sentence (never mid-sentence)."""
    s = " ".join(str(text).split())
    if len(s) <= limit:
        return s
    ends = [m.end() for m in _SENTENCE_END.finditer(s[: limit - 1])]
    return s[: ends[-1]].rstrip() + "…" if ends else excerpt(s, limit)


def pack_note(p: Prediction) -> str | None:
    cut = pack_truncation(p.trace)
    if cut is None:
        return None
    return (
        f"This filing was longer than the system's window: it was given {cut[0]} of {cut[1]} pages."
    )
