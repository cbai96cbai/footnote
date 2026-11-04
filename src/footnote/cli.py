"""footnote eval | compare | failures | site."""

from pathlib import Path
from typing import Annotated, cast, get_args

import typer

from footnote.benchmark import load_questions
from footnote.budget.tally import BudgetExceeded, Tally
from footnote.config import Settings
from footnote.eval.failures import FAILURE_TYPES, classify_run, failures_markdown
from footnote.eval.gold import load_gold
from footnote.eval.identity import RunIdentityMismatch, RunLocked
from footnote.eval.records import read_predictions, read_verdicts
from footnote.eval.registry import TestRunRefused
from footnote.eval.run import DEFAULT_RESULTS, register_known_systems, run_eval
from footnote.eval.split import SplitName
from footnote.eval.stats import cluster_bootstrap, paired_cluster_bootstrap
from footnote.eval.summary import is_grader_error, outcomes_by_company
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
    except (TestRunRefused, RunLocked, RunIdentityMismatch, BudgetExceeded) as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    if s["ci95"] is None:
        typer.echo(
            f"{run_id}: nothing graded ({s['grader_errors']} grader errors); rerun to resume"
        )
        raise typer.Exit(1)
    lo, hi = s["ci95"]
    typer.echo(
        f"{run_id}: {s['correct']}/{s['n_graded']} correct = {s['accuracy']:.3f} "
        f"(95% CI {lo:.3f} to {hi:.3f}), {s['errors']} errors, ${s['cost_usd']:.4f}"
    )
    if not s["complete"]:
        typer.echo(
            f"incomplete: {s['retriable_errors']} questions rate limited, "
            f"{s['grader_errors']} grader errors; rerun with the same run id"
        )
        raise typer.Exit(1)


@app.command("compare")
def compare_cmd(run_a: str, run_b: str, results_dir: ResultsDir = DEFAULT_RESULTS) -> None:
    by_id = {q.id: q for q in load_questions()}
    va, vb = read_verdicts(results_dir / run_a), read_verdicts(results_dir / run_b)
    shared = [
        by_id[i]
        for i in sorted(set(va) & set(vb))
        if not (is_grader_error(va[i]) or is_grader_error(vb[i]))
    ]
    if not shared:
        typer.echo("the two runs share no questions", err=True)
        raise typer.Exit(1)
    a, b = outcomes_by_company(shared, va), outcomes_by_company(shared, vb)
    for name, o in ((run_a, a), (run_b, b)):
        acc, lo, hi = cluster_bootstrap(o)
        typer.echo(f"{name}: {acc:.3f} (95% CI {lo:.3f} to {hi:.3f}) on {len(shared)} questions")
    d, lo, hi = paired_cluster_bootstrap(a, b)
    typer.echo(f"{run_a} minus {run_b}: {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")


@app.command("failures")
def failures_cmd(run_id: str, results_dir: ResultsDir = DEFAULT_RESULTS) -> None:
    run_dir = results_dir / run_id
    verdicts = read_verdicts(run_dir)
    if not verdicts:
        typer.echo(f"{run_id}: no verdicts found", err=True)
        raise typer.Exit(1)
    types = classify_run(read_predictions(run_dir), verdicts, load_gold())
    (run_dir / "failures.md").write_text(failures_markdown(run_id, types), encoding="utf-8")
    for t in FAILURE_TYPES:
        typer.echo(f"{t:15} {sum(x == t for x in types.values())}")
    typer.echo(f"{'total':15} {len(types)} of {len(verdicts)} questions; wrote failures.md")


@app.command("site")
def site_cmd(
    results_dir: Annotated[
        Path, typer.Option("--results", help="Where run folders live.")
    ] = DEFAULT_RESULTS,
    out: Annotated[Path, typer.Option(help="Where to write the static site.")] = Path("_site"),
) -> None:
    from footnote.site.build import build_site

    build_site(results_dir, out)
    typer.echo(f"site written to {out}")
