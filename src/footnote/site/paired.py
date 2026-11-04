"""The agent's accuracy minus each other system's, on the test split, with a paired interval."""

from footnote.benchmark import AgentQuestion
from footnote.eval.stats import paired_cluster_bootstrap
from footnote.eval.summary import is_grader_error, outcomes_by_company
from footnote.site.runs import Run

Difference = tuple[float, float, float]


def paired_differences(
    test_runs: list[Run], questions: dict[str, AgentQuestion]
) -> dict[str, Difference | None] | None:
    """Per run id: (agent minus run, lo, hi) over the questions both runs graded, resampling
    whole companies; None for a run it cannot be computed for. None overall without an agent run.
    """
    agent = next((r for r in test_runs if r.system == "agent"), None)
    if agent is None:
        return None
    out: dict[str, Difference | None] = {}
    for r in test_runs:
        if r is agent:
            continue
        shared = [
            questions[i]
            for i in sorted(set(agent.verdicts) & set(r.verdicts))
            if i in questions
            and not (is_grader_error(agent.verdicts[i]) or is_grader_error(r.verdicts[i]))
        ]
        out[r.run_id] = (
            paired_cluster_bootstrap(
                outcomes_by_company(shared, agent.verdicts),
                outcomes_by_company(shared, r.verdicts),
            )
            if shared
            else None
        )
    return out
