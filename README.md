# footnote

An agent that answers questions about SEC filings and shows exactly where every number came from.

Recording: (added when the site is published)
Live site: (link added when the site is published)

Footnote answers questions about SEC filings from the FinanceBench open sample,
then measures itself against simpler systems that use the same model, on a test split that is run
once.

- **Answers with quotes.** Each answer cites the filing page and the exact quote behind it. A code
  check confirms the quotes exist and that the final number follows from them.
- **Compared on equal terms.** The agent is scored against three baselines (no documents, plain
  retrieval, long context) with the same model, the same filing access and the same grading.
- **Honest uncertainty.** Accuracy comes with a 95% interval that resamples whole companies, plus
  answer coverage, cost per question and latency. A static site shows every answer, citation,
  trace and grader verdict, including the failures.

## Architecture

Dependencies run one way. The gold answers are read only by `eval` (to score) and `site` (to show the reference answer next to each result); nothing upstream can read them.

```mermaid
flowchart LR
    B[benchmark<br/>questions, no gold] --> C[corpus<br/>fetch, parse PDFs]
    C --> R[retrieval<br/>BM25 + embeddings]
    R --> S[agent and baselines]
    B --> S
    S --> E[eval<br/>split, grading, stats<br/>scoring, reads gold]
    E --> W[site<br/>static HTML]
```

`tests/test_import_boundaries.py` fails if any upstream package imports `eval` or touches the raw benchmark rows.

## Results

Not yet available. The table is filled in after the single frozen run of each system on the test
split (57 questions, 11 companies). No number below is a measurement.

| System | Accuracy (95% interval, company-clustered) | Answer coverage | Cost per question | Mean latency |
| :- | :-: | :-: | :-: | :-: |
| Without documents | - | - | - | - |
| Plain retrieval | - | - | - | - |
| Long context | - | - | - | - |
| Agent | - | - | - | - |

Historical context only, from the FinanceBench paper (Islam et al., arXiv:2311.11944), GPT-4-Turbo,
November 2023, graded by humans: 19% with one shared vector store, 50% with one store per filing,
79% with long context, 85% when given the evidence pages. Different models, grading and setup, so
these are not a baseline for the table above.

## Failure examples

Added from the real test run, sorted by type: declined, unsupported, retrieval miss, wrong number,
reasoning, error (`footnote failures` produces the per-run list). Failures are published, not
selected away.

## Design decisions

- **Explicit tool loop, no framework.** The agent is a plain loop over three tools (search the
  filing, read a page, calculate) with a call cap per question. Every step is a record the site can
  show, and spend limits are easy to see in the code.
- **Equal document access.** Every system is told which filing a question is about. The comparison
  then isolates how a system uses the document, not whether it can find it. Finding the filing
  among all 84 is a harder, separate setting that is not tested here.
- **Numbers are graded by code.** Numeric questions are scored on a structured final value (number
  and unit) against the reference with a stated tolerance, so they need no model grader. The
  remaining questions go to a grader that is a different model from the agent, never told which
  system produced the answer, using a fixed rubric.
- **The split is frozen first.** Questions are split by company (no company in both halves) before
  any tuning, and the test split is run once per system family; a repeat is refused unless a reason
  is recorded. Development runs are labeled exploratory on the site.
- **Spend limits by construction.** Each call has input and output token caps, each question a call
  cap, and each run is admitted only if its worst-case cost fits under a $25 cap tracked in an
  append-only ledger. A crashed run stays charged its maximum.
- **Declining counts as a miss.** Every benchmark question has an answer, so a refusal is scored
  wrong, and answer coverage is reported beside accuracy.

## What the grounding check proves

After the agent submits, code checks two things.

1. Every quote it cites appears verbatim on the cited page of the filing.
2. The final numeric answer is a correct rounding of a number in a cited quote, or the result of a
   logged calculation over quoted numbers (plus small constants such as /2, x100, x365).

It does not check numbers in the explanation text. It does not prove a number means what the answer
says: a quoted figure can be in the wrong units or for the wrong period and still pass. The grader
measures correctness; the check measures traceability.

## Data and license

- Questions and reference answers: the FinanceBench open sample (150 questions, 84 filings),
  Patronus AI, [arXiv:2311.11944](https://arxiv.org/abs/2311.11944), licensed CC BY-NC 4.0 on
  Hugging Face. Our reading is that non-commercial portfolio use fits that license; it is a reading,
  not a verified grant.
- The filings are the companies' public documents. The PDFs are downloaded at run time from the
  FinanceBench repository, pinned by SHA-256 in `data/corpus_manifest.json`, and never committed or
  redistributed. The site shows short attributed quotes and links to the source.
- Code: MIT.

Corpus quality, measured against the benchmark's own evidence passages (`results/corpus_quality.json`):
for 93.1% (176 of 189) the passage is found on the right page by word coverage of at least 90%.
Only 33.3% match word for word, because PDF extraction orders table cells differently from the
benchmark's text; the page is right, the character sequence is not. The grounding check therefore
runs against our extracted text, not the benchmark's.

## Run locally

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync
cp .env.example .env   # fill in the five variables, then export them
```

The variables are `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `FOOTNOTE_AGENT_DEPLOYMENT`,
`FOOTNOTE_GRADER_DEPLOYMENT` and `FOOTNOTE_CACHE` (where PDFs and indexes are cached). The agent
runs on gpt-5-mini and the grader on gpt-4.1-mini, both Azure OpenAI deployments.

```
uv run footnote eval --system agent --split gate --run-id agent-gate-1
uv run footnote compare agent-gate-1 plain-gate-1      # paired difference with an interval
uv run footnote failures agent-gate-1                  # failures by type
uv run footnote site                                   # writes _site/
```

Systems are `no-docs`, `plain-retrieval`, `long-context` and `agent`. Splits are `gate` (30
development questions for pipeline checks), `dev` (93) and `test` (57). Checks: `uv run ruff check .`,
`uv run mypy src`, `uv run pytest`.

## Limitations

- 57 test questions from 11 companies. Intervals are wide, and a difference inside them is not
  claimed.
- One benchmark, one model family for the agent, and a grader that is itself a model and can be
  wrong in both directions. Grader agreement checked against AI-written reference labels is an
  AI-reference estimate, not a human agreement rate.
- Equal document access hides the harder problem of finding the right filing.
- The no-documents result measures performance without documents, not memorization. The benchmark
  has been public since 2023 and may be in training data.
- Page references follow the benchmark PDF's zero-based page index plus one, which can differ from
  the number printed on the page.
- Some source links have gone dead; the site falls back to an SEC EDGAR search.
