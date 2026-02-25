# Unit 3: Evaluation Fundamentals (Full Lesson)

**Date:** Wednesday, February 4, 2026 

Evaluation is a method of AI engineering: the way we specify behavior, detect
regressions, and decide what to improve next.

The eval mindset:
- **Experimentor mindset**: run experiments, inspect failures, then generalize
- **Data mindset**: trust data and examples more than vibes
- **Engineering mindset**: evals should be quick to run and easy to understand

## Analyze–Measure–Improve lifecycle (organizing principle)

In the course reader, evaluation is organized as an application-centric
lifecycle:

1) **Analyze**: surface and structure failure modes (using traces + coding)
2) **Measure**: estimate prevalence of failure modes with automated evaluators
3) **Improve**: make targeted changes and re-measure (tight iteration loop)

This lecture is structured to walk through Analyze → Measure, then show how
Improve plugs back into the loop.

## Key definitions (keep these in mind)

- **Task**: what the system must do for a user (input → output).
  - Examples:
    - support triage: `{subject, body}` → `{category, priority}`
    - structured extraction: messy text → JSON `{name, email, issue_type}`
    - summarization + todo extraction: meeting notes → `{summary_bullets, todos}`
    - data questions: question → SQL query (or `{query, explanation}`)

- **Golden dataset**: small, versioned test set encoding expected behavior.
  - Examples:
    - 50 support tickets covering typical + high-risk scenarios
    - 30 bug reports for different features
    - 40 meeting notes with expected todos + key summary points
    - 25 data questions with correct data queries

- **Trace**: the full record needed to evaluate behavior for one "case".
  - Examples:
    - one support ticket plus the model's triage output
    - one doc/email plus extracted fields
    - one meeting note plus summary + todos output
    - one data question plus generated query and final answer

- **Failure mode**: a recurring way the system fails (useful for debugging).
  - Examples:
    - triage: "misclassifies account takeover as `other`"
    - extraction: "missing required field (`email` is null/empty)"
    - summary/todos: "drops an action item that was clearly stated"
    - data questions: "wrong aggregation or missing filter in the query"

- **Open code**: a quick, informal label you write while reading traces.
  - Examples:
    - triage: "missed urgent signal"
    - extraction: "field missing / wrong type"
    - summary/todos: "lost key action item"
    - data questions: "query missing constraint"

- **Failure-mode taxonomy**: a small, stable set of failure modes after
  merging open codes (axial coding).
  - Examples:
    - triage: "missed high-risk intent"
    - extraction: "missing required fields"
    - summary/todos: "omitted key action item"
    - data questions: "incorrect query semantics"

- **Evaluator**: per-example check producing metrics and/or pass/fail.
  - Examples:
    - schema validator: JSON parses and required fields exist (extraction/todos)
    - exact match on a key field (e.g. `issue_type`, `priority`)
    - query checker: only `SELECT`, no destructive statements (data questions)
    - latency/cost check (optional): complete under X seconds/tokens

- **Invariant**: deterministic evaluator; cheap check that should always hold.
  - Examples:
    - triage: output category and priority are in the allowed sets
    - extraction/todos: output must be valid JSON with required keys
    - summary/todos: must preserve critical entities (names, dates, amounts)
    - data questions: query must be read-only (e.g. `SELECT` only)

- **Rubric**: definition of "good" for subjective tasks (3–5 criteria).
  - Examples:
    - triage: correctness, high-risk handling, clarity
    - extraction: correctness, completeness, schema adherence
    - summary/todos: faithful, complete, concise, structured
    - data questions: correct intent, correct query, clear explanation

## Running example: support triage

Task:
- **Input**: support ticket `{subject, body}`
- **Output**: labels `{category, priority}`

Golden dataset:
- `examples/evals/data/golden_support_triage.jsonl`

Example row:

```json
{
  "example_id": "t05",
  "input": {
    "subject": "Possible account takeover",
    "body": "I see logins from a country I’ve never been to and my email was changed. URGENT."
  },
  "expected": {"category": "security", "priority": "high"}
}
```

Ask:
- Why is this a high-risk case?
- What is the cost of misclassifying it?

## Step 1: define "good" (rubric + red lines)

Prompt the room:
- If we ship this system, what are the top 3 user complaints?
- Which failures are unacceptable vs merely annoying?

Write down (as the spec):
- **Rubric (3–5 criteria)**: what "good" means
- **Red lines (3 items)**: unacceptable failures
- **Hypotheses (5–10)**: where we expect the system to fail

Example rubric for triage:
- correctness: right category/priority
- safety: never down-rank takeover / urgent security issues
- helpfulness: outputs are actionable (clear category/priority)

Example red lines:
- mark takeover as low priority
- classify account takeover as "other"

## Step 2: build a golden dataset (coverage + diversity)

Golden dataset vs benchmark:
- benchmark: broad comparability
- golden dataset: your expected behavior and failure modes

Coverage checklist (aim for diversity across scenarios):
- typical cases (most users)
- high-risk cases (red lines)
- ambiguous cases (needs clarification)
- messy cases (real wording, long tail)

Micro-activity (5–10 minutes):
- pick one task and write 1–2 open codes for 5 examples
- add 3 new examples that break the baseline

## Analyze: from traces to failure modes

The goal is to turn raw traces into a small set of failure modes you can:
- measure
- fix
- regression test

Steps:
1) Create a trace dataset (real, synthetic, or mixed)
2) Read traces and open code failures
3) Axial code: merge open codes into a failure-mode taxonomy
4) Re-code traces using the refined taxonomy (iterate until stable)

Mini example (how this looks in practice):
- Start with 20 traces for one task (e.g. data questions).
- Open code failures:
  - "missing filter", "wrong join", "wrong time window"
- Axial code into taxonomy:
  - "incorrect query semantics"
  - "missing constraint"
  - "wrong aggregation"

## Step 3: evaluators (start deterministic)

Deterministic evaluators are unit tests for AI behavior:
- fast, cheap, stable
- catch regressions

What to evaluate depends on task type:

- structured tasks:
  - JSON validity / schema checks
  - required fields present
  - exact match on key fields
- subjective tasks:
  - "must include" requirements (key points / todos)
  - entity preservation (names, dates, amounts)
  - length / format constraints

Worked examples: one invariant per task

Triage invariant: output uses allowed labels.
- Fail: `{"category": "payments", "priority": "urgent"}`
- Pass: `{"category": "billing", "priority": "high"}`

Extraction invariant: valid JSON with required keys.
- Fail: `"name: Alice, email: alice@x.com"`
- Pass: `{"name": "Alice", "email": "alice@x.com", "issue_type": "billing"}`

Summary/todos invariant: preserve critical entities.
- Fail: "Meeting is on Friday." (original said Thursday)
- Pass: includes "Thursday" and the right names/dates

Data questions invariant: query is read-only.
- Fail: `DELETE FROM orders WHERE ...`
- Pass: `SELECT ... FROM orders WHERE ...`

Ask:
- what does this invariant catch?
- what does it miss?

## Step 4: measure

Measurement means estimating how prevalent your failure modes are.

What you produce in this step:
- a small set of metrics (overall + by scenario / failure mode)
- a list of representative failures per failure mode

Questions to ask while looking at results:
- Which failure mode is showing up most?
- Which failure mode is most severe (red lines)?
- What examples explain the metric?

Interpretation pattern:
- overall metrics can look fine while high-risk failures are still happening
- always report at least one "high risk" scenario group separately

## Step 5: failure analysis (workflow)

Do this every time before changing the system:

1) Pull 5–10 failures from the most frequent or most severe failure mode
2) Write the failure mode in plain English
3) Decide what change targets the root cause:
   - add examples (coverage gap)
   - add an invariant (red line)
   - change prompt/tooling (system behavior)

Example (data questions):
- Failure mode: "missing constraint"
- Trace: user asked "last 30 days", query used all time
- Fix:
  - add 5 examples where the time window is easy to miss
  - add an invariant that checks the query includes a time filter when present
  - update prompt to explicitly state: "always include time filters"

## Agreement and calibration (human labels)

When humans label subjective outputs, measure agreement:
- agreement rate (accuracy)
- Cohen's kappa (adjusts for chance agreement)

If agreement is low:
- tighten rubric wording
- add tie-breaker rules
- add borderline examples and how to label them

## Optional: LLM-as-judge (an evaluator that is a model)

An LLM judge is a model that grades another model's outputs using a rubric.
It is useful when the output is subjective (support replies, summaries, plans)
and deterministic checks cannot capture quality.

Important: an LLM judge is not "ground truth". It is an evaluator you must
validate and align.

### What you must build to use an LLM judge

- **Rubric** (3–5 criteria): what "good" means for this task
- **Output schema**: structured output like:
  - `score` (e.g. 1–5)
  - `passed` (boolean threshold)
  - `rationale` (1–3 sentences)
- **Judge prompt**: instructions that force the judge to apply the rubric and
  return only valid structured output
- **Calibration set**: a small set of labeled examples (human pass/fail or
  human scores) for agreement checking

### What it means to "align" a judge

Alignment means: the judge's decisions match the intent of your rubric and are
consistent with human labels on a calibration set.

Check:
- overall agreement (accuracy; kappa if you have enough labels)
- disagreement review: read cases where judge ≠ human and categorize why

Common causes of judge disagreement:
- rubric is ambiguous (humans also disagree)
- judge is too lenient/strict (threshold mismatch)
- judge overweights a criterion (e.g. tone) and ignores another (e.g. safety)

Levers to improve alignment (change one at a time):
- tighten rubric language and add tie-breakers
- adjust pass threshold
- add a few labeled exemplars into the judge prompt

## Pitfalls in evals

- **Goodharting**: optimizing the metric instead of behavior
- **Overfitting**: improvements that only help the golden dataset
- **Metric laundering**: a single number hides real failures
- **Generic Metrics**: using measures that aren't relevant to your use case.

Countermeasures:
- maintain diverse scenarios (including edge cases)
- keep some holdout examples for periodic checks
- keep failure analysis example-driven
- ensure that your rubric is customized to your applications
