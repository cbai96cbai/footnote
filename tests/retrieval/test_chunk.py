from footnote.corpus.parse import Page
from footnote.llm.tokens import count_text_tokens
from footnote.retrieval.chunk import chunk_pages


def _page(i: int, text: str) -> Page:
    return Page("DOC", i, text)


def _long(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunks_never_cross_pages() -> None:
    pages = [_page(0, _long("a", 900)), _page(1, _long("b", 900)), _page(2, "")]
    passages = chunk_pages(pages)
    assert {p.page_index for p in passages} == {0, 1}
    assert all(p.text.split()[0][0] == ("a" if p.page_index == 0 else "b") for p in passages)
    assert all(
        all(w[0] == ("a" if p.page_index == 0 else "b") for w in p.text.split()) for p in passages
    )
    assert passages[0].id == "DOC:p0:0" and passages[1].id == "DOC:p0:1"


def test_chunk_token_limit() -> None:
    passages = chunk_pages([_page(0, _long("word", 2000))], max_tokens=350, overlap_tokens=50)
    assert len(passages) > 1
    assert all(count_text_tokens(p.text) <= 350 for p in passages)
    # overlap: consecutive passages share words, and every word is covered, none split
    first, second = passages[0].text.split(), passages[1].text.split()
    assert set(first) & set(second)
    covered = {w for p in passages for w in p.text.split()}
    assert covered == {f"word{i}" for i in range(2000)}


def test_short_and_empty_pages() -> None:
    passages = chunk_pages([_page(3, "short page text"), _page(4, "  \n ")])
    assert [(p.id, p.text) for p in passages] == [("DOC:p3:0", "short page text")]


def test_unbreakable_run_is_still_bounded() -> None:
    passages = chunk_pages([_page(0, "x" * 5000)], max_tokens=100, overlap_tokens=10)
    assert all(count_text_tokens(p.text) <= 100 for p in passages)
