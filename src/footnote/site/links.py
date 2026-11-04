"""Where a reader can open a filing: the benchmark's link when it works, else an EDGAR search.

`python -m footnote.site.links` checks each document's link once and writes
data/link_status.json ({doc_name: {"ok": bool, "status": int | null}}).
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from footnote.benchmark import DocInfo, doc_info

LINK_STATUS = Path(__file__).resolve().parents[3] / "data" / "link_status.json"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
FORMS = {"10K": "10-K", "10Q": "10-Q", "8K": "8-K"}
EDGAR = "https://www.sec.gov/cgi-bin/browse-edgar"


def edgar_search_url(company: str, doc_type: str) -> str:
    params = {
        "company": company,
        "type": FORMS.get(doc_type, ""),
        "dateb": "",
        "owner": "include",
        "count": "40",
        "action": "getcompany",
    }
    return f"{EDGAR}?{urllib.parse.urlencode(params)}"


def edgar_fallback(doc: DocInfo, status: dict[str, Any]) -> str | None:
    """An EDGAR search link, offered next to the original link when it is not known to work."""
    entry = status.get(doc.doc_name)
    if entry and entry.get("ok"):
        return None
    return edgar_search_url(doc.company, doc.doc_type)


def check_link(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """One GET with a browser-like User-Agent; only the first bytes are read."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https links
            resp.read(1024)
            return {"ok": 200 <= resp.status < 400, "status": resp.status}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"ok": False, "status": None}


def check_all(
    docs: dict[str, DocInfo], check: Callable[[str], dict[str, Any]] = check_link
) -> dict[str, dict[str, Any]]:
    names = sorted(docs)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: check(docs[n].doc_link), names))
    return dict(zip(names, results, strict=True))


def load_link_status(path: Path = LINK_STATUS) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    status = check_all(doc_info())
    LINK_STATUS.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    live = sum(v["ok"] for v in status.values())
    print(f"{live} of {len(status)} links live -> {LINK_STATUS}")


if __name__ == "__main__":
    main()
