import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "footnote"
EVAL_EXEMPT = {"eval", "site"}
# the only places allowed to touch raw benchmark rows (which hold the gold answers)
RAW_EXEMPT = {"eval", "site", "benchmark.py", "_financebench.py"}


def _imports(tree: ast.AST, name: str) -> bool:
    """True if the tree imports footnote.<name> (or a submodule), in any import form."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[:2] == ["footnote", name] for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            takes_name = any(a.name == name for a in node.names)
            if node.level == 0:
                if mod.split(".")[:2] == ["footnote", name] or (mod == "footnote" and takes_name):
                    return True
            elif mod.split(".")[0] == name or (not mod and takes_name):
                return True
    return False


def _offenders(name: str, exempt: set[str]) -> list[str]:
    return [
        str(p.relative_to(PKG))
        for p in PKG.rglob("*.py")
        if p.relative_to(PKG).parts[0] not in exempt and _imports(ast.parse(p.read_text()), name)
    ]


def test_agent_side_never_imports_eval() -> None:
    assert _offenders("eval", EVAL_EXEMPT) == []


def test_only_benchmark_and_eval_import_raw_rows() -> None:
    assert _offenders("_financebench", RAW_EXEMPT) == []


def test_scan_detects_eval_imports() -> None:
    assert _imports(ast.parse("from footnote.eval.gold import load_gold"), "eval")
    assert _imports(ast.parse("import footnote.eval"), "eval")
    assert _imports(ast.parse("from footnote import eval"), "eval")
    assert _imports(ast.parse("from .eval import gold"), "eval")
    assert not _imports(ast.parse("from footnote.benchmark import load_questions"), "eval")


def test_scan_detects_raw_row_imports() -> None:
    for src in (
        "from footnote._financebench import read_rows",
        "import footnote._financebench",
        "from footnote import _financebench",
        "from . import _financebench",
        "from .._financebench import read_rows",
    ):
        assert _imports(ast.parse(src), "_financebench"), src
    assert not _imports(ast.parse("from footnote.benchmark import load_questions"), "_financebench")
