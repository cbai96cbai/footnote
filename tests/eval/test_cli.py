import json
from pathlib import Path

from typer.testing import CliRunner

from footnote.cli import app


def test_compare_prints_paired_difference(tmp_path: Path) -> None:
    from footnote.benchmark import load_questions

    ids = [q.id for q in load_questions()][:6]
    for name, labels in (("a", ["correct"] * 6), ("b", ["incorrect"] * 6)):
        d = tmp_path / name
        d.mkdir()
        u = {"input_tokens": 0, "output_tokens": 0, "cost": "0", "charged_max": False}
        (d / "verdicts.jsonl").write_text(
            "".join(
                json.dumps(
                    {"question_id": i, "label": lb, "method": "llm", "rationale": "", "usage": u}
                )
                + "\n"
                for i, lb in zip(ids, labels, strict=True)
            )
        )
    out = CliRunner().invoke(app, ["compare", "a", "b", "--results-dir", str(tmp_path)])
    assert out.exit_code == 0 and "a minus b: +1.000" in out.output
