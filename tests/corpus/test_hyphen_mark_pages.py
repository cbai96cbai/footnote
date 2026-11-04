import json
import random
import re

import pytest

from footnote.paths import cache_dir
from footnote.verify import quote_in_page

MARK = re.compile(r"[^\W\d_]+(?:\x02[^\W\d_]+)+")  # a hyphenated word written with U+0002


@pytest.mark.corpus
def test_quote_typed_with_a_plain_hyphen_matches_real_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOOTNOTE_CACHE")  # use the real corpus cache
    files = sorted((cache_dir() / "pages").glob("*.jsonl"))
    assert files, "no cached pages; fetch the corpus first"
    pages = [
        json.loads(line)["text"]
        for f in files
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    affected = [t for t in pages if "\x02" in t]
    assert len(affected) > 1000  # the marker is on roughly one page in seven
    sample = random.Random(0).sample(affected, 300)
    tested = 0
    for text in sample:
        m = MARK.search(text)
        if not m:
            continue
        # whole words around the hyphenated one: back to a space before, on to a space after
        lo = text.rfind(" ", 0, max(0, m.start() - 30)) + 1
        hi = text.find(" ", m.end() + 30)
        span = text[lo : hi if hi != -1 else len(text)].strip()
        quote = span.replace("\x02", "-")
        assert quote_in_page(quote, text), quote
        tested += 1
    assert tested >= 200
