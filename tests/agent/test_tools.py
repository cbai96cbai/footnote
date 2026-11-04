from decimal import Decimal

import pytest
from fakes import FakeIndex

from footnote.agent.tools import READ_PAGE_MAX_TOKENS, Toolbox, ToolError
from footnote.llm.tokens import count_text_tokens

PAGES = {0: "Revenue was $1,234 million.", 1: "word " * 5000}


def box() -> Toolbox:
    return Toolbox(PAGES, FakeIndex([(0, "Revenue was $1,234 million."), (1, "other")]))


def test_search_shows_page_index_and_reports_pages() -> None:
    out = box().run("search_filing", {"query": "revenue", "k": 2})
    assert "[page_index 0]" in out.content and out.pages_seen == [0, 1]
    assert len(out.summary) <= 200


def test_read_page_cuts_long_pages() -> None:
    out = box().run("read_page", {"page_index": 1})
    assert count_text_tokens(out.content) <= READ_PAGE_MAX_TOKENS + 60 and "cut" in out.summary


def test_read_page_out_of_range_is_tool_error() -> None:
    for bad in (5, -1, "0", True, 1.5):
        with pytest.raises(ToolError):
            box().run("read_page", {"page_index": bad})


def test_calculate_logs_calc_and_rejects_non_arithmetic() -> None:
    b = box()
    out = b.run("calculate", {"expression": "(1,234 - 1,000) / 2"})
    assert out.content.endswith("= 117") and b.calcs[0].result == Decimal(117)
    for bad in ("__import__('os')", "abs(3)", "1 +", "2 ** 99", "x", ""):
        with pytest.raises(ToolError):
            b.run("calculate", {"expression": bad})
    assert len(b.calcs) == 1


def test_unknown_tool_and_bad_search_arguments() -> None:
    for name, args in (
        ("nope", {}),
        ("search_filing", {}),
        ("search_filing", {"query": "a", "k": 99}),
    ):
        with pytest.raises(ToolError):
            box().run(name, args)


def test_trace_args_cuts_only_long_quote_strings() -> None:
    from footnote.agent.tools import trace_args

    long = "q" * 500
    args = {"query": long, "citations": [{"page_index": 1, "quote": long}, {"quote": "short"}]}
    out = trace_args(args)
    assert out["query"] == long and out["citations"][1]["quote"] == "short"
    assert len(out["citations"][0]["quote"]) == 400 and out["citations"][0]["quote"][-1] == "…"
    assert args["citations"][0]["quote"] == long  # the input is not changed
