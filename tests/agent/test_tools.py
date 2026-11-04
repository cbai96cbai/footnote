from decimal import Decimal

import pytest
from fakes import FakeIndex

from footnote.agent.tools import READ_PAGE_MAX_TOKENS, Toolbox, ToolError
from footnote.llm.tokens import count_text_tokens

PAGES = {0: "Revenue was $1,234 million.", 1: "word " * 5000}


def box() -> Toolbox:
    return Toolbox(PAGES, FakeIndex([(0, "Revenue was $1,234 million."), (1, "other")]))


def test_search_shows_page_index_and_reports_pages() -> None:
    out = box().run("search_filing", {"query": "revenue"})
    assert "[page_index 0]" in out.content and out.pages_seen == [0, 1]
    assert len(out.summary) <= 200


def test_search_schema_has_no_k_and_says_plain_words() -> None:
    from footnote.agent.prompts import TOOLS

    fn = next(t["function"] for t in TOOLS if t["function"]["name"] == "search_filing")
    assert list(fn["parameters"]["properties"]) == ["query"]
    assert "plain words" in fn["description"] and "up to 8 passages" in fn["description"]


def test_search_always_asks_for_eight_and_ignores_a_k_from_the_model() -> None:
    index = FakeIndex([(i, f"passage {i}") for i in range(12)])
    b = Toolbox({i: "x" for i in range(12)}, index)
    out = b.run("search_filing", {"query": "revenue"})
    assert out.pages_seen == list(range(8)) and index.ks == [8]
    for k in (2, 99, "many", None):
        out = b.run("search_filing", {"query": "revenue", "k": k})
        assert len(out.pages_seen) == 8
    assert index.ks == [8] * 5
    few = Toolbox({0: "x"}, FakeIndex([(0, "only one")]))
    assert few.run("search_filing", {"query": "q"}).pages_seen == [0]


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


def test_tool_results_show_hyphen_marks_as_hyphens_and_a_displayed_quote_verifies() -> None:
    from footnote.verify import quote_in_page

    raw = "Registration under Regulation S\x02T and forward\x02looking statements."
    b = Toolbox({0: raw}, FakeIndex([(0, raw)]))
    for out in (b.run("read_page", {"page_index": 0}), b.run("search_filing", {"query": "x"})):
        assert "\x02" not in out.content and "Regulation S-T and forward-looking" in out.content
        shown = out.content.split("\n", 1)[1]
        assert quote_in_page("Regulation S-T and forward-looking statements", shown)
        assert quote_in_page("Regulation S-T and forward-looking statements", raw)
