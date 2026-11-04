import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "footnote"
EXEMPT = {"eval", "site"}


def _imports_eval(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[:2] == ["footnote", "eval"] for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and (
                mod.split(".")[:2] == ["footnote", "eval"]
                or (mod == "footnote" and any(a.name == "eval" for a in node.names))
            ):
                return True
            if node.level > 0 and (
                mod.split(".")[0] == "eval"
                or (not mod and any(a.name == "eval" for a in node.names))
            ):
                return True
    return False


def test_agent_side_never_imports_eval() -> None:
    offenders = []
    for path in PKG.rglob("*.py"):
        rel = path.relative_to(PKG)
        if rel.parts[0] in EXEMPT:
            continue
        if _imports_eval(ast.parse(path.read_text())):
            offenders.append(str(rel))
    assert offenders == []


def test_scan_detects_eval_imports() -> None:
    assert _imports_eval(ast.parse("from footnote.eval.gold import load_gold"))
    assert _imports_eval(ast.parse("import footnote.eval"))
    assert _imports_eval(ast.parse("from footnote import eval"))
    assert _imports_eval(ast.parse("from .eval import gold"))
    assert not _imports_eval(ast.parse("from footnote.benchmark import load_questions"))
