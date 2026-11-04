"""footnote eval | compare (failures comes later)."""

from pathlib import Path
from typing import Annotated, cast, get_args

import typer

from footnote.benchmark import load_questions
from footnote.budget.tally import Tally
from footnote.config import Settings
from footnote.eval.records import read_verdicts
from footnote.eval.registry import TestRunRefused
from footnote.eval.run import DEFAULT_RESULTS, register_known_systems, run_eval
from footnote.eval.split import SplitName
from footnote.eval.stats import cluster_bootstrap, paired_cluster_bootstrap
from footnote.eval.summary import outcomes_by_company
from footnote.llm.client import LLMClient

app = typer.Typer(no_args_is_help=True, add_completion=False)
ResultsDir = Annotated[Path, typer.Option(help="Where run folders live.")]


@app.command("eval")
def eval_cmd(
    system: Annotated[str, typer.Option(help="System family.")],
    split: Annotated[str, typer.Option(help="gate, dev or test.")],
    run_id: Annotated[str, typer.Option(help="Run folder name; reuse it to resume.")],
    rerun_test: Annotated[bool, typer.Option("--rerun-test")] = False,
    reason: Annotated[str | None, typer.Option(help="Why a test run is repeated.")] = None,
    results_dir: ResultsDir = DEFAULT_RESULTS,
) -> None:
    if split not in get_args(SplitName):
        raise typer.BadParameter("split must be gate, dev or test")
    register_known_systems()
    settings = Settings.from_env()
    try:
        s = run_eval(
            system,
            cast(SplitName, split),
            run_id,
            client=LLMClient(settings),
            tally=Tally(),
            settings=settings,
            results_root=results_dir,
            rerun_test=rerun_test,
            reason=reason,
        )
    except TestRunRefused as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    lo, hi = s["ci95"]
    typer.echo(
        f"{run_id}: {s['correct']}/{s['n']} correct = {s['accuracy']:.3f} "
        f"(95% CI {lo:.3f} to {hi:.3f}), {s['errors']} errors, ${s['cost_usd']:.4f}"
    )


@app.command("compare")
def compare_cmd(run_a: str, run_b: str, results_dir: ResultsDir = DEFAULT_RESULTS) -> None:
    by_id = {q.id: q for q in load_questions()}
    va, vb = read_verdicts(results_dir / run_a), read_verdicts(results_dir / run_b)
    shared = [by_id[i] for i in sorted(set(va) & set(vb))]
    if not shared:
        typer.echo("the two runs share no questions", err=True)
        raise typer.Exit(1)
    a, b = outcomes_by_company(shared, va), outcomes_by_company(shared, vb)
    for name, o in ((run_a, a), (run_b, b)):
        acc, lo, hi = cluster_bootstrap(o)
        typer.echo(f"{name}: {acc:.3f} (95% CI {lo:.3f} to {hi:.3f}) on {len(shared)} questions")
    d, lo, hi = paired_cluster_bootstrap(a, b)
    typer.echo(f"{run_a} minus {run_b}: {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
