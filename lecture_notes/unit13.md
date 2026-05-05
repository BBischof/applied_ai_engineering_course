# Unit 13: Observability, Debugging & CI/CD

**Date:** Wednesday, April 22, 2026

Unit 9 built **agency** and the **harness** (state, tools, verification, trajectory proof). Unit 10 extended that to **multi-agent decomposition and coordination**. Unit 11 pushed the same ideas into **batch pipelines** where the "user" is a scheduler processing millions of records. Unit 12 changed the **model itself** via adaptation (LoRA / SFT / RL).

Every one of those units answered: *how do I prove this works before shipping?* **This unit attempts to equip you for the next question: how do I keep proving it once it is running in production — and how do I debug it when it stops working?** Proof is not a launch gate; it is a continuous discipline. The artifacts you already build for eval (golden datasets, rubrics, traces, invariants) are the same artifacts you run in CI, emit from production, and alert on when they regress.

**This lecture** follows one thread that runs back through the last two decades of systems practice:

> **Execution → Prediction → Meaning.**

### What "observability" actually means

**Observability**, as a discipline, is the craft of **recovering what happened inside a running system from the signals it emits about itself.** You rarely have a live production system sitting under a debugger; what you have is whatever the system chose to emit — events, measurements, and causal chains — and your job is to reconstruct behavior from those signals. The word comes from control theory — Kalman's 1960 paper on linear filtering, where a system is *observable* if its internal state can be inferred from its outputs alone — and was re-imported into software in the mid-2010s, most influentially by [Cindy Sridharan's *Monitoring and Observability*](https://copyconstruct.medium.com/monitoring-and-observability-8417d1952e1c) (2018) and by Charity Majors and the Honeycomb team, as a deliberate contrast with pre-wired monitoring. The era's three canonical signals — which will reappear throughout this unit — are:

- **Traces:** the causal chain of work as a request moves through a system, structured as parent / child spans so you can see which step called which.
- **Metrics:** numerical aggregates over time — request rate, error rate, latency percentiles — cheap to store and cheap to alert on.
- **Logs:** structured events emitted from the components themselves, typically JSON these days rather than free-form text.

That toolkit and that mindset are era-agnostic. What *is* era-specific is **which hidden state matters.**

### The arc of this unit

**Software observability** — the first era — focused on hidden **execution state**: which service handled the request, where latency accumulated, whether a dependency failed. It built the trace / metric / log toolkit we now take for granted. **MLOps** extended the same practice to hidden **model and data state** (drift, skew, degraded predictions — state that lives in the *learned* behavior of the system, not just its code). **LLMOps** extends it again to hidden **semantic and context state** — whether the system produced a useful, safe, grounded answer and can explain why. The discipline is the same; **what counts as "hidden state" expands at each transition.** Once you commit to that frame, debugging, CI, monitoring, and alerting are the same system running at different time horizons — seconds, hours, days, weeks.

### Today's session

The steel thread from classical software observability through MLOps to LLMOps; trace-first debugging; structured logging across agent and MCP boundaries; root cause vs. symptom discipline; golden datasets as CI tests; regression gates on prompt and model changes; online drift detection; and alerting on eval degradation (not just latency). Worked around the same **Analyze → Measure → Improve** loop from Unit 3, now running on the production clock.

### After Units 9, 10, 11, 12 (vocabulary to reuse)

| Earlier-unit idea                              | How it shows up in ops                                                                                                                                         |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trace / trajectory** (Unit 9)                | The primary observability primitive. An agent turn, a pipeline record, a tool call — each emits a structured trace with inputs, outputs, latency, and cost.   |
| **Harness** (Unit 9)                           | The runtime that emits those traces, enforces checkpoints, and fires alerts when an invariant fails.                                                           |
| **Multi-agent failure attribution** (Unit 10)  | Traces need span IDs and parent links so you can answer "*which* agent regressed?" not just "the system is worse."                                             |
| **Three-layer eval** (Unit 11)                 | Invariants + sampling + LLM-judge at offline eval time is the same recipe you run online — cadence and budget are what change.                                 |
| **Extract-then-validate** (Unit 11)            | Same pattern powers deploy-time gates: generate → validate → block promotion on failure.                                                                       |
| **Analyze–Measure–Improve** (Unit 3)           | This is the inner loop of observability. Production traces are the new "failure mode taxonomy" input; the loop just runs weekly instead of once before ship.   |
| **Model adaptation** (Unit 12)                 | Every fine-tune, every adapter swap, every base-model version change is a **regression event** that must pass the same CI gates as a prompt change.             |
| **Determinism × agency** (Unit 9)              | Tells you *what* to alert on. High-determinism steps fail loud (schema, parse); high-agency steps fail quiet (bad tool choice, off-policy plan).               |

---

## From software observability to LLMOps: tracing a single thread

The startups and technologies of today have precedent. Two decades of distributed-systems practice and a decade of MLOps already answered versions of most of the questions we ask today. LLMOps is the **latest chapter in that story; do not fall for the narrative that this is completely new.** In particular, do not fall for the take that *stochastic, unpredictable production systems* are novel engineering territory. Data science and ML practitioners have been shipping and operating probabilistic systems in production for at least a decade before "AI engineer" was a job title. A great deal of what the software-engineering side is currently rediscovering about LLMs already has answers in the MLOps literature; the genuinely new parts are fewer than the hype suggests. Before we get into specific mechanics, we walk the steel thread top to bottom — because seeing the continuity is what keeps you from reinventing half of this by accident, and seeing the *discontinuity* is what keeps you from shipping an LLM system with fragile guardrails.

### The working thesis

Observability is the practice of **recovering hidden system state from external signals.** OpenTelemetry defines it that way for classical software; ML and LLM systems inherit the definition unchanged. What changes across eras is **what counts as "hidden state."**

> Software observability asks whether the **code ran**. MLOps asks whether the **model still fits reality**. LLMOps asks whether the system produced the **right meaning, grounded in the right context**.

Or, in pithy words:

> **Execution → Prediction → Meaning.**

### Three eras, three questions

| Era                     | Primary object observed                                             | Typical failure                                                                              | Core signals                                                                                                        |
| ----------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Software observability  | Execution path through services                                     | Timeout, exception, saturation, dependency failure                                           | Traces, metrics, logs (OpenTelemetry)                                                                               |
| MLOps                   | Execution **+** data/model behavior                                 | Drift, skew, degraded predictive performance, stale lineage                                  | Above **+** data validation, sliced error analysis, calibration, model and data lineage, retraining signals         |
| LLMOps                  | Execution **+** model/data behavior **+** semantic/context behavior | Hallucination, bad retrieval or grounding, wrong tool use, unsafe output, costly brittle chains | Above **+** token telemetry, rubric/judge evals, feedback, prompt & retrieval lineage, tool and agent span traces   |

The pattern: **the locus of correctness rises up the stack.** From *did the code execute* → *did the model generalize* → *did the system produce context-grounded meaning*. Infrastructure telemetry stays necessary in LLMOps; it just stops being sufficient.

### Follow one request

The clearest way to feel the transition is to trace the **same request** through three eras and watch what you have to capture to debug it.

**A — Software era.** A request arrives; you trace ingress → services → DB → response. The question: *where in the system did this request fail or slow down?*

**B — MLOps era.** The same request hits a classifier. You now *also* trace feature generation, the model version that scored it, the evaluation history behind that model, drift / skew metrics, and eventual ground truth from the label pipeline. The question widens: *did the request execute correctly — **and** is the model still valid for the world it is operating in?*

**C — LLMOps era.** The same request reaches an agent. You now trace the prompt template, system instructions, the retrieval query, every retrieved document and its provenance, every model call, every tool invocation, intermediate reasoning / action steps, output quality, safety checks, token usage, and human or automated eval feedback. The question widens again: *did the system produce a useful, safe, **grounded** answer for this context — and can I explain why?*

The infrastructure trace is still there in the LLMOps case, with strictly more nested children. OpenTelemetry has since standardized the LLM-specific layer of this (GenAI client spans, agent spans, token-usage metrics — see the OTel GenAI semantic conventions); Microsoft Foundry and Google Cloud's generative-AI operations guidance land on the same shape from the other direction.

### The MLOps layer: prior art that ports forward

The foundational paper in this lineage is **Sculley et al., 2015 — *Hidden Technical Debt in Machine Learning Systems*** (NeurIPS), often called the "high-interest credit card" paper. It named the specific failure modes of production ML: the iceberg diagram where the model itself is a small box surrounded by vastly more code for data collection, feature extraction, configuration, serving, and monitoring; the **CACE principle** ("Changing Anything Changes Everything"); undeclared consumers of model outputs; unstable data dependencies; hidden feedback loops; glue code; pipeline jungles; configuration debt — and the case that **testing data invariants and monitoring prediction quality are operational, not research, concerns.** A good fraction of the MLOps literature since is a debt payment against something Sculley et al. named first. LLM systems inherit every one of those categories and add more; we return to CACE specifically in the debugging-paradigms section below, because it is the single strongest justification for the "every fix ships with a test" discipline that runs through the rest of this unit.

Seven years later, I co-wrote *[MLOps: A Holistic Approach](resources/holistic-mlops.pdf)* with Darek Kłeczek and Hamel Husain at Weights & Biases (2022). I won't recap the whole report here; the most durable idea in it, in my experience since, is the **People / Processes / Platform** decomposition — the reminder that operationalizing ML is not only a tools problem. A perfect trace infrastructure with no on-call runbook still pages the same engineer at 3 AM without a plan. If you want the longer treatment of responsibility boundaries, org structure, governance, and the mechanics that go with them.

The MLOps generation also already built most of the **mechanics** you'll reuse here:

| MLOps pattern (c. 2019–2022)                   | LLMOps equivalent                                                            |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| Shadow deployments                             | Shadow-mode prompt and model rollouts (Key Idea 4)                            |
| A/B testing with a randomization platform      | Prompt / model A/B tests scored by judge + user-signal metrics                |
| Model registry (staging → prod)                | Prompt registry **+** model-version registry **+** adapter registry (Unit 12) |
| Data / model lineage                           | Trace-level lineage: prompt hash, model version, adapter id, retrieval index  |
| Distributional data-quality checks             | Schema invariants + input drift detection on prompts and tool responses       |
| Human-in-the-loop on a fraction of predictions | Sampled judge eval + human review queue (Unit 11's confidence routing)        |
| CI/CD smoke tests + metrics-in-PRs             | Prompt/model CI gates (Key Idea 3)                                            |
| "Monitor technical **and** business metrics"   | Same — extend "business" to cover latency, cost-per-success, and judge score  |
| Reproducibility: code + data + configs         | Reproducibility: code + prompt + model version + adapter + context assembly   |
| "ML systems fail silently" [p. 7]              | Same sentence. Same problem. Larger surface area.                             |

### The LLMOps layer: what this era adds

The semantic-and-context column of the three-era table, made concrete:

| Dimension                   | MLOps world                                  | LLMOps world                                                                                                  |
| --------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **The model**               | You train it; you own it.                    | Most teams call a vendor API. Weights are frozen and opaque; version bumps arrive as announcements, not PRs.  |
| **Eval output**             | Rubrics + sliced error analysis + calibration + baselines; scalar metrics (AUC, F1, RMSE, nDCG) where applicable. The Analyze–Measure–Improve loop (Unit 3) started here. | Same discipline extended to text and structured outputs. Rubrics grow to include semantic quality; LLM-as-judge becomes central where there is no single right answer. |
| **Versioned surface**       | Code + data + model weights + feature definitions + thresholds.                 | Base model, adapter, prompt, tool schemas, retrieval index, context-assembly logic, guardrail rules.                           |
| **Drift flavors**           | Richer than usually remembered: covariate shift, label drift, concept drift, feature drift — depending on which factor of $P(X, Y)$ moves. | Same taxonomy still applies, plus drift in prompts, retrieval indices, tool outputs, and user-behavior at the interaction level (Key Idea 4). |
| **Retry semantics**         | Same input → same output.                    | Same input → possibly *different* output. Retries can change the bug instead of fixing it.                    |
| **Failure vocabulary**      | Class imbalance, label noise, leakage.       | Hallucination, prompt injection, tool misuse, infinite loops, context rot (Unit 7), poor grounding.           |
| **Trace shape**             | Already multi-step: feature-store lookup → preprocessing → model(s) → postprocessing → decision logic, often with ensembles, cascades, or shadowed candidates. | Span tree grows in width and depth: agent turns, tool calls, subagents, retrievals, guardrails (Units 5, 7, 9, 10). Same topology, more branches. |
| **Regression event**        | New training run, threshold change, hard-rule update, ensemble or cascade reconfiguration, feature-pipeline change, data-slice rebalancing. | All of the above, plus: prompt edit, vendor model bump, adapter swap, tool-schema change, retrieval-index rebuild, guardrail update, router-policy change. |
| **Time to "retrain"**       | Hours to days — you own the pipeline.        | Rarely done. Fine-tuning (Unit 12) is a rare, gated event; most changes are not "retraining" at all.          |
| **Who owns the model?**     | Your team.                                   | Usually the vendor. That inverts several MLOps defaults about what you can control.                           |

### The pivot

The most consequential shift is one line:

> In classic MLOps, **your model is your team's product**. In LLMOps, **the model is usually someone else's infrastructure** that your product depends on.

Giving up ownership of the model in exchange for capability is the trade. Every discipline in the rest of this unit — tracing, CI, drift detection, alerting — is partly a response to that loss of control. You can't introspect weights, you can't diff model versions, and you can't schedule a retrain when things regress. What you *can* do is **observe, gate, and fall back**, plus one thing MLOps didn't have to do as loudly: **evaluate continuously**, because semantic correctness cannot be summarized in a single scalar.

With that lineage established, the rest of this unit is the machinery that makes the LLMOps column of the three-era table work — continuous eval, trace-first debugging, CI gates on prompts and models, drift in three flavors, and quality-first alerting. Call it *"Deployment and Observability"* from the 2022 report with the full weight of agents, tools, adapters, and vendor models on top.

---

## Warm-up: "the system was working last week"

Three scenarios, all real in shape:

- **The agent that got slower.** A coding agent you shipped in Unit 9 suddenly has a p95 latency twice what it was. The model is the same, the prompt is the same, the tools are the same. A user upstream started pasting 30-page tracebacks into the chat window. Context length drifted, not the policy. Your SLO is latency; your **cause** is an input distribution shift. Without traces you will chase the wrong thing for a week.
- **The batch pipeline that quietly broke.** A Unit 11-style extraction pipeline was hitting 94% schema-valid last month; this month it is 71%. Nobody changed the prompt. The vendor rolled a minor model revision. No release notes, no deprecation, no version bump you saw. Silent regression — the worst kind, because downstream systems kept ingesting the bad data.
- **The fine-tune that helped on eval and hurt in the wild.** A Unit 12 LoRA adapter improved your golden-dataset style score by 8 points. A week into production, support tickets complaining about tone double. Your golden set over-indexed on one persona; real traffic distribution was different. Offline eval passed; online eval failed.

None of these fail at build time. They all fail at run time. **That is the problem observability solves.**

---

## The recurring question (our analytical lens)

For each key idea, we'll ask the same three questions — mapping directly to P³ on the production clock:

1. **What is the promise *now*?** What behavior do you still owe the user today, after this morning's model update / this week's prompt edit / last month's fine-tune?
2. **How do you prove it is still true?** What signals (traces, invariants, sampled evals, user feedback) do you collect continuously, and at what cadence?
3. **What do you do when the proof breaks?** Who gets paged, what do they run, and how do you decide between rollback, hotfix, and "watch it"?

---

## Three debugging paradigms (and three assumptions that break)

Debugging is at least 60 years old as a named discipline, and most of what you do in this unit is classical technique applied to a new trace shape. Three paradigms are worth naming explicitly before the mechanics, because they are the handles students can reach for when a page fires. After each paradigm we name the classical assumption it rests on — and then show how that assumption bends under LLM systems.

### Paradigm 1 — Disciplined empiricism (the mindset)

The canonical source is Agans, *Debugging: The 9 Indispensable Rules* (2002) — short book, reads in an evening. Four of the rules come up most often in what follows:

- **Understand the system.** In an LLM system that means knowing the agent graph, the prompt, the retrieval index, and the tool schemas *before* you start poking. Bugs reported in ignorance of architecture waste hours.
- **Change one thing at a time.** The hardest rule to follow in practice, and harder for LLM systems, because retries can change the output. If you edit a prompt and ship a model upgrade in the same PR and the bug goes away, you have learned *nothing* about which one did it.
- **Quit thinking and look.** Evidence beats theory. The trace already contains the answer; most debugging time is wasted on hypotheses instead of reading the spans.
- **Keep an audit trail.** Every failing trace, every fix, every decision — written down. This is the proto-version of the blameless postmortem plus the regression-test ratchet we build in Key Idea 3.

### Paradigm 2 — Bisection: the hunt for what changed (the method)

Most production bugs answer to one question: **what changed since it worked?** The tactical answer is binary search.

The lineage:

- 1999 — Andreas Zeller formalizes *delta debugging* (ddmin), an algorithm that binary-searches an input to find the minimal failing case [Zeller, *Yesterday, my program worked. Today, it does not. Why?*, ESEC/FSE 1999].
- 2005 — Linus Torvalds adds `git bisect`: binary-search a commit history between a known-good and a known-bad revision to find the first bad commit. Differential debugging becomes a one-line UX.
- 2010s — CI systems generalize this to automatic regression attribution: every test failure gets a candidate commit list.
- Today (LLMOps) — the *surface you can bisect across* is much larger, because your versioned artifacts are no longer just code commits.

Classical `git bisect` works directly across any versioned artifact — prompt revisions, base-model snapshots, adapter checkpoints, tool-schema diffs, retrieval-index snapshots — so anything you can version, you can binary-search. Two bisection moves are less obvious and worth naming:

- **ddmin on the failing transcript** — Zeller's algorithm reduces a long failing conversation to a minimum reproducer, and that reproducer is usually where the real bug lives.
- **Bisect the span tree** of a single agent run — which tool call first diverged from the working trajectory? This is bisection over a tree structure rather than a linear history, and it has no clean classical analog.

The first is a straightforward port of classical debugging to a new kind of input; the second is genuinely LLM-specific.

### Paradigm 3 — Observability over debugging (the paradigm shift)

Charity Majors's *Observability Engineering* (O'Reilly, 2022) draws a sharp line between two mindsets:

- **Traditional debugging** — the classical *monitoring* mindset — assumes you know the bug exists and goes looking for it. You pre-wire dashboards for the failures you can imagine. You add logging when something breaks. Monitoring answers the *known-unknowns*.
- **Modern observability** accepts that you *cannot* predict in advance the questions you will need to ask. You emit high-cardinality, high-dimensional structured events so that questions you didn't know to pre-wire are still answerable *after the fact*. Observability answers the *unknown-unknowns*.

This distinction matters more for LLM systems than for classical ones, and Majors's own LLM-specific writing makes the point sharply. In [*Observability in the Age of AI*](https://www.honeycomb.io/blog/observability-age-of-ai) and related Honeycomb essays she argues that **LLMs are "a weird new kind of storage engine,"** and that debugging them is **"a classic high-cardinality, high-dimensionality problem"** — exactly the kind of problem observability (as opposed to monitoring) was invented to solve. Failures are emergent, non-deterministic, and user-shaped. The set of failure modes grows every week as users find new ways to interact with the system. You cannot pre-wire a dashboard for every regression; you have to emit *rich enough* traces that next month's question is answerable in a query, not a deploy.

This is the framing that justifies the trace-attribute table in Key Idea 1: hash every prompt, log every tool argument, attach every eval score to the lowest responsible span — because you don't know yet which field will turn out to pinpoint the bug three weeks from now. It is also why Majors has been arguing since 2023 that **"LLMs demand observability-driven development"** [Majors, Honeycomb 2023]: production is the only real test environment for a probabilistic system, and an observable production is the only debugger you are going to get. The vendor debate in the [next section](#the-modern-pov-evals-as-observability) is, largely, the industry working out what that phrase should mean in practice.

### Three classical assumptions that LLM systems break

Knowing what *doesn't* transfer is as important as knowing what does. Each row below names a classical debugging assumption, the LLM-era reality that breaks it, and the practical counter-move.

| Assumption                   | Classical world                       | LLM world                                                                                                                                                                                                                                        | Counter-move                                                                                                                  |
| ---------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------- |
| **Determinism**              | Same input → same output              | A **Heisenbug** — the term of art for a bug whose behavior changes the moment you try to observe it — is the normal case, not the exception. Same input can produce different outputs; the act of instrumenting (e.g. turning on verbose logging, changing temperature to make logs readable) can itself change behavior.             | Snapshot the trace. Replay N times with seed, temperature, and cache controls held fixed. Classify the *distribution* of outputs, not a single run. |
| **Singular root cause**      | One bug, one fix                      | Production failures are almost always **multiple soft regressions aligning** — Reason's Swiss-cheese model. A mildly worse prompt + a minor vendor model bump + a slightly unusual input cohort; none a smoking gun.                             | Multiple independent gates (schema, judge, red-line, drift). Look for *patterns* across failing traces, not single causes.   |
| **Local fixes are durable**  | Patch the function, it stays patched  | Fixes regress as the vendor updates the base model, as inputs drift, or as an upstream change edits a tool schema — a classical **CACE** effect [Sculley et al., 2015]: the boundaries you think you have in an LLM system are mostly fictional. | Every fix ships with a test in the golden set; the test runs forever. Proof is a ratchet, not a patch.                        |

### A CACE warning before we go any further

Row three deserves its own sentence. CACE — *Changing Anything Changes Everything* — is the debt line Sculley et al. drew under every ML system in 2015 and it is louder in LLM systems than it was in classical ML. A prompt edit silently changes every downstream parser. A base-model bump silently changes every agent that depended on that model's quirks. A retrieval-index rebuild silently changes what grounding each agent receives. A tool-schema rename silently breaks every agent that referenced the old name.

The structural defense is the one Sculley et al. prescribed a decade ago and is still the best answer: **test data invariants, monitor prediction quality, and treat every change as a release event.** The rest of this unit is that discipline, with LLM-specific machinery on top.

---

## The modern POV: evals as observability

The LLMOps platforms in the market today — [LangSmith](https://docs.langchain.com/langsmith/), [Braintrust](https://www.braintrust.dev/), [Logfire](https://pydantic.dev/logfire), [Langfuse](https://langfuse.com/), [Arize Phoenix](https://phoenix.arize.com/), [W&B Weave](https://wandb.ai/site/weave/), [Helicone](https://www.helicone.ai/) — disagree about architecture and workflow shape, and most of their comparison pages are written in pointed direct reference to each other. Underneath the disagreement, there is one shared premise:

> **Healthy execution does not imply acceptable behavior.**

Or, in the form that's hard to forget once you read it: **"is it up?" → "is it good?"** (Braintrust, [*The Three Pillars of AI Observability*](https://www.braintrust.dev/blog/three-pillars-ai-observability), 2025). The disagreement is about what "good" means in operational terms, and three ideas organize it.

### The triad changes

Classical observability: metrics, logs, traces. The AI-era equivalent, per the Braintrust post, is **traces, evals, and annotation.** Concretely: AI-shaped spans are orders of magnitude larger than classical spans (tens of kilobytes vs hundreds of bytes on average; gigabyte-class single traces are routine), evaluation becomes a continuous production signal rather than a pre-ship milestone, and human annotation of traces feeds evaluation datasets that are continuously reconciled with real traffic rather than commissioned once as a "golden set." The premise underneath — *evaluation is an observability signal, not a separate discipline* — is the most contested claim in the space.

### Evaluation as sibling, not sub-category

Charity Majors and Honeycomb hold the opposite line, most directly in [*Observability in the Age of AI*](https://www.honeycomb.io/blog/observability-age-of-ai). Observability answers *what happened in production*; evaluation answers *how good was the output*. The same trace can feed both; the tools and the mental models stay different. The specific Honeycomb argument is that LLM debugging is a **high-cardinality, high-dimensional problem** — the problem class that observability (as distinct from monitoring) was invented to solve. Folding evaluation into observability risks losing the part of observability that handles unknown-unknowns.

### The full-stack argument

The [Logfire comparison pages](https://pydantic.dev/docs/logfire/get-started/comparisons/braintrust/) push a structural claim that goes further than a tool-features debate: AI-only observability tools are structurally incomplete, because production LLM failures are rarely confined to the LLM call. Three failure sites, only one of which is the model:

1. **What triggered the call** — upstream routing, auth, a stale cache.
2. **What the model accessed** — a tool returned a stale record, a DB query silently failed, a RAG index dropped a document.
3. **What was done with the response** — a downstream parser choked, a side effect didn't fire, a webhook timed out.

> **When your AI agent misbehaves, was it the model's reasoning or the data it received? Only full-stack observability tells you.**

Two design consequences follow from taking the argument seriously: evaluations written as code (the [`pydantic-evals`](https://github.com/pydantic/pydantic-evals) / `pytest` model), rather than configured through a dashboard UI; and trace data queryable by agents via SQL and MCP, not just by humans through a UI. The Majors and Logfire positions converge on the full-stack claim from different starting points — one from the Honeycomb-era observability tradition, one from application-first product engineering.

### Consequences for this unit

In production, *"the agent is wrong,"* *"the database returned stale data,"* and *"the tool schema changed last Tuesday"* are the same bug until a trace proves otherwise. Eval scores tell you *that* quality dropped; only a trace that spans the LLM call, the tools that feed it, and the surrounding app code tells you *where and why*. Eval-only and observability-only tools each solve a real problem; most mature stacks combine them — either as two tools stitched through OpenTelemetry, or one platform that commits to both.

The rest of this unit treats evals as a **first-class signal attached to traces**, not a replacement for them.

---

## Key Idea 1: The trace is the unit of debugging

### From log lines to structured traces

A log line says *"something happened."* A trace says *"this specific request went through these specific steps, took this long, cost this much, and produced this output."* Every AI system you built in Units 5–12 has a natural trace shape:

- **Unit 5 tool call:** `{tool_name, arguments, result, latency, status}`
- **Unit 9 agent turn:** `{thought, action, observation, tool_spans[]}`
- **Unit 10 multi-agent run:** a parent span per agent, child spans per delegation, with explicit handoff edges
- **Unit 11 batch record:** `{input_id, operator_spans[], final_output, validation_result}`
- **Unit 12 inference:** `{model, adapter_id, prompt_hash, tokens_in, tokens_out, latency, cached}`

The common shape across all of these is **OpenTelemetry** [1]: spans with parent / child relationships, attributes (key / value metadata), events (point-in-time annotations), and status. The LLM-specific tracing ecosystem is, underneath, OpenTelemetry with an opinionated schema layered on top for LLM-specific fields (prompt, completion, token counts, tool calls) — standardized lately as the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), including GenAI client spans, agent spans, and token-usage metrics.

Each vendor in that ecosystem — LangSmith, Braintrust, Logfire, Langfuse, Arize/Phoenix, W&B Weave, Helicone — sells a slightly different opinion about what an LLM trace *should* capture and what you should do with it. We already walked the positioning map in [The modern POV: evals as observability](#the-modern-pov-evals-as-observability) above; the one sentence to bring into Key Idea 1 is this: **whichever tool you pick, the underlying shape is an OTel trace with richer attributes, and the attributes you choose to emit are how you cash Charity Majors's observability-driven development promise in practice.**

<!-- TODO: slide figure of a multi-agent trace tree with parent/child spans -->

### What belongs in a trace

The minimum viable trace for an LLM step:

| Field                     | Why it matters                                                                                     |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| `trace_id`, `span_id`, `parent_span_id` | Lets you reconstruct multi-step work; required for Unit 10 attribution.           |
| `timestamp`, `duration_ms`              | Latency analysis; SLO math.                                                        |
| `model`, `model_version`                | Drift detection when the vendor bumps the model.                                   |
| `prompt_hash` (not full prompt)         | Detect prompt drift without storing PII unnecessarily.                             |
| `tokens_in`, `tokens_out`, `cost_usd`   | Budget alerts; cost regression detection.                                          |
| `tool_name`, `tool_args`, `tool_result` | For tool-call spans (Unit 5).                                                      |
| `output_schema_valid: bool`             | The cheapest, strongest signal from Unit 11.                                       |
| `eval_scores: dict`                     | Offline judge / rubric scores, attached when they exist.                           |
| `user_id` (hashed), `session_id`        | Slice-and-dice for error analysis (Unit 1/3).                                      |
| `status: ok | error | degraded`         | Explicit status, not inferred from HTTP codes.                                     |

### MCP tracing

MCP (Unit 5) adds a tool-call boundary to the trace. The **host** (the LLM-facing app) emits a span for every tool invocation; the **server** emits a span for every handler; every internal unit of work on either side is a child span. All of it belongs to **one** `trace_id`. What a real trace looks like, end to end:

```
trace_id = T
└── span A · llm.tool_call           (host)
    └── span B · mcp.handler         (server, parent_span_id = A)
        ├── span C · db.query        (server, parent_span_id = B)
        └── span D · model.generate  (server, parent_span_id = B)
```

When the host invokes a tool it sends `traceparent = T / A` — its `trace_id` plus its current `span_id` — as a header. The server opens span B using that context, which gives B `parent_span_id = A`. Further work inside the server (the DB query, a nested model call, whatever) opens more children under B. Your viewer renders this either as the tree above or as a Gantt-style waterfall where each child span sits inside its parent's time range; the underlying structure is identical.

Debugging starts at the failing span and walks upward: if span D blew up, was the bug in D itself, in span B that called D with bad arguments, or in span A that selected the wrong tool in the first place? That is the first fork of any MCP-related incident, and it is only askable if `trace_id` survived the process boundary.

### Traces as the eval input

The Unit 3 failure-mode taxonomy was built from traces. In production, the pattern is identical — you just **sample** instead of reviewing all of them, and the taxonomy grows with each weekly review. The observability stack's job is to make those traces:

- **Queryable** ("show me agent runs where `tool_calls > 20`"),
- **Linkable** (from an alert / ticket straight to the trace),
- **Replayable** (can you re-run *this exact trace* against a new prompt or model? — this is the foundation of regression CI).

> **P³ lens — Proof:** traces are your proof instrument at run time. Without them, "the system works" is a statement of faith. With them, every individual request is a test case you can replay.

---

## Key Idea 2: Root cause vs. symptom

### Most outages point at the wrong thing

A paged engineer sees *the symptom*: an alert, a user complaint, a dashboard going red. The trap is to fix the symptom. In AI systems the symptom is usually far from the cause:

| Symptom                                   | Typical root cause                                                                 |
| ----------------------------------------- | ---------------------------------------------------------------------------------- |
| p95 latency doubled                       | Input-length distribution shifted (longer contexts, new user segment)              |
| Schema-valid rate dropped                 | Vendor rolled a new model variant; constrained decoding stopped enforcing          |
| User complaints about tone                | Training data distribution drifted; fine-tuned adapter over-fit one persona       |
| Cost up 30%                               | Prompt grew a "few-shot" section; cache hit rate dropped                           |
| "Agent got dumber"                        | Tool output format changed; agent can no longer parse it, retries, falls back     |
| "Sometimes gives the wrong answer"        | A specific slice (e.g. queries in Spanish) regressed; overall metric masks it     |

The habit to build: **from every alert, drill through the trace tree to the lowest span whose behavior changed.** That is usually the cause.

### Compounding errors revisited

Unit 11 showed that 95% per-step accuracy × 5 steps = 77% system accuracy. The corollary for observability: **a system-level failure alert tells you almost nothing about which step is broken.** You need **per-span invariants** that fail loud before the error propagates. Two examples:

- A validate-after-extract span that fails with `status = error, reason = missing_required_field` pinpoints the bad extractor in one click.
- A multi-agent orchestrator with a failed child span and `status = ok` on the parent is a bug in the harness, not the model.

### Attribution in multi-agent systems

Unit 10 ended with the question *"how do you trace a failure back to the responsible agent?"* The mechanical answer: every agent gets its own span, every delegation is an edge, and the eval score is attached to the **lowest** span whose output contributed to the failure. If the planner hands off a reasonable-looking but subtly wrong subgoal, the eval score goes on the planner's span — not the executor's.

### The debug-from-a-trace workflow

A reproducible loop when a production trace looks broken:

1. **Capture.** Pull the full trace (all spans, all attributes, all inputs).
2. **Replay.** Feed the same inputs to the same prompt/model/tool chain in a local / staging harness. Is the failure deterministic?
3. **Bisect.** If non-deterministic, rerun N times; classify the failure. If deterministic, binary-search the change log (prompt diffs, model version, tool schema change).
4. **Minimize.** Shrink the trace to the smallest failing slice — this is the new golden-dataset entry for tomorrow's CI.
5. **Fix and gate.** Ship the fix *with* the new test in the same PR.

---

## Key Idea 3: CI as the eval harness

### Golden datasets become tests

From Unit 3 you have a golden dataset, a rubric, a set of invariants, and an LLM-as-judge. In CI, those become:

- **Unit-style tests** on deterministic invariants (schema valid, required fields present, no forbidden tokens).
- **Integration-style tests** on small sampled slices of the golden set, scored by rubric / judge, with a pass/fail threshold.
- **Regression tests** — literally replaying prior production traces against the new prompt/model and checking that outputs haven't degraded.

The mental model: `pytest` runs in seconds, schema invariants run in minutes, judge evals run in ~10 min on a representative slice, and full golden-set evaluation runs nightly on a larger sample. **Everything that runs pre-merge is an eval.** What changes is the cost and coverage.

### What to gate on a prompt change

A prompt edit is a code change. Gate it on:

| Gate                                    | Blocks merge on                                              |
| --------------------------------------- | ------------------------------------------------------------ |
| Schema invariants on N=100 replayed traces | any schema failure                                       |
| Regression slice (judge-scored)         | score drops > X points vs. main                              |
| Red-line checks                         | any red-line violation (Unit 3)                              |
| Cost delta                              | tokens / request up more than Y%                             |
| Latency delta                           | p95 up more than Z ms                                        |

Store prompts as versioned files (git, or a prompt registry like PromptLayer / LangSmith Hub / Braintrust). "Prompt changed" without a commit and a PR number is a production incident waiting to happen.

### What to gate on a model upgrade

Model upgrades — even minor version bumps from the same vendor — are the **highest-risk change you routinely make**, because the diff is opaque. The gate is strictly larger than for a prompt change:

- Full schema-invariant replay (N ≥ 1000 historical traces).
- **Full golden-set eval** (not a slice).
- **Sliced eval by segment** — language, domain, input length, user cohort. A minor average regression often hides catastrophic slice regressions.
- Cost and latency deltas with explicit human sign-off.
- **Shadow mode** for ≥ 48 hours before promoting to primary.

### Regression testing on the same trace twice

The single most valuable test you can build: given a stored trace, rerun it end-to-end against the candidate change and diff the outputs. The diff can be structural (schema field changes), scored (judge), or human-graded on a sampled subset. Every production bug you fix should add one entry to this test corpus.

<!-- TODO: worked example: a promptfoo / Braintrust config that replays a stored
trace set and reports pass/fail deltas. Lightweight, copy-pasteable. -->

### Offline eval ≠ production safety

Two failure modes that offline eval cannot catch:

1. **Distribution mismatch.** Your golden set is static; traffic is not. Today's production distribution may not be represented.
2. **Emergent integration failures.** The LLM call passes eval; the downstream consumer misreads the (subtly changed) field name. No single test owns the integration seam.

The fix for both is Key Idea 4.

> **P³ lens — Proof:** CI is the machine that enforces your proof automatically on every change. If a check is not automated, it will not run consistently, and your proof will decay one missed check at a time.

---

## Key Idea 4: Online monitoring and drift

### What counts as drift?

Drift is the silent killer. Three flavors, all real, all need separate detection:

| Drift type          | Definition                                                                           | Typical detector                                                  |
| ------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| **Input drift**     | The distribution of inputs changed (new segment, longer queries, new language)       | Embed inputs; track centroid distance, PSI on length / language   |
| **Output drift**    | The distribution of outputs changed even though inputs look stable                  | Rate of specific output classes, judge-score histogram over time  |
| **Behavior drift**  | Users started interacting differently (more retries, more thumbs-down, more aborts)  | Implicit-feedback time series, session-length distributions       |

Input drift tells you the world changed. Output drift tells you the model or prompt did. Behavior drift tells you the product did. You want all three.

### Sampling production traces

Storing every trace is usually fine at small scale and ruinously expensive at large scale. The pattern from Unit 11 applies: **stratified sampling by slice** plus **aggressive keep-the-bad-ones sampling** (anything with `status != ok`, anything flagged by the judge, anything from a new segment).

### Continuous eval on sampled traces

The cheapest continuous proof loop:

1. Sample ~1–5% of production traces (stratified).
2. Run the same judge / rubric you use offline.
3. Compute scored metrics per day, per segment.
4. Alert when the score crosses the regression threshold from CI.

This is literally Analyze–Measure–Improve running on a 24-hour clock.

### Shadow mode and canaries

Before a prompt or model goes to 100% of traffic:

- **Shadow mode:** candidate runs in parallel with production, output is logged but not served. Compare outputs per trace. Cheap; catches regressions without user impact.
- **Canary:** 1% → 10% → 50% → 100% with automatic rollback gates tied to online eval scores, not just latency. An LLM canary that gates only on p95 latency and HTTP 5xx is almost useless — the failure mode that matters is silent quality regression.

<!-- TODO: diagram: shadow vs canary vs full rollout, with the gate signals
labeled on each arrow. -->

### A note on fine-tuned models (Unit 12)

Fine-tuned adapters have their **own** drift profile. Every time the base model is updated by the vendor, your adapter's behavior can shift — even if the adapter weights are identical. Gate every base-model update behind a full Unit 12-style eval run before promoting. Some teams pin base models explicitly for this reason.

> **P³ lens — Production:** monitoring is not about dashboards, it is about *promises you've made that are still being tested every minute by real traffic*. If no one looks at the dashboard, the promise is untested.

---

## Key Idea 5: Alerting on quality, not just latency

### SLOs for LLM systems

Traditional SRE teaches you to set SLOs on latency and availability. Both still apply. But the LLM-specific SLOs that matter most:

| SLO                              | Example target                                       | Fires when                                               |
| -------------------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| Schema-valid rate                | ≥ 99.5% of structured outputs parse                 | Decoder / model / schema drift                           |
| Judge-score median               | ≥ X on rolling 24h window                           | Quality regression                                       |
| Red-line violation rate          | = 0 (hard)                                          | Safety failure                                           |
| Tool-call success rate           | ≥ 99%                                               | Tool / MCP regression                                    |
| p95 tokens per request           | ≤ budget                                            | Prompt bloat, pathological inputs                        |
| Cost per successful request      | ≤ budget                                            | Retries, cache misses, silent fallbacks                  |
| Human-review queue depth         | ≤ N pending                                         | Confidence routing is too aggressive                     |

Every one of these should be wired to a pager with a documented runbook.

### Alert on eval degradation, not model version

An alert on "model version changed" is noisy and unhelpful. An alert on "judge-score median dropped 5 points on the Spanish slice in the last 24 hours" is actionable — it points at a specific slice and a specific direction.

### The on-call runbook template

For every alert, a one-page runbook:

1. **What the alert means** (plain English).
2. **Where to look first** (trace filter link; dashboard link).
3. **How to triage** (bisection checklist: prompt / model / tool / input).
4. **Mitigations, in order** (rollback, canary to 0%, kill switch, escalation).
5. **What to add to CI afterward** (which trace to add to the golden set).

### Cost and budget alerts are eval alerts

A cost alert is a quality alert wearing a business suit. Retries, fallback chains, and silent degradations show up as cost first, quality second. Treat a 20% unexplained cost spike the same way you'd treat a 20% drop in judge score — it is almost always the same bug.

<!-- TODO: example runbook (filled in) for a single alert — "schema-valid rate
dropped below 99% on the invoice-extraction pipeline". -->

---

## Putting it all together: the observability loop

The diagram below is the entire unit in one picture. Each arrow is a thing you must build:

```
                     ┌────────────────────────┐
                     │  Production traffic    │
                     └──────────┬─────────────┘
                                │ emits traces
                                ▼
                     ┌────────────────────────┐
                     │  Trace store (OTel)    │◀──┐
                     └──────────┬─────────────┘   │
                                │                 │
              ┌─────────────────┼──────────────┐  │ replay
              ▼                 ▼              ▼  │
     ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
     │ Invariants   │  │ Sampled      │  │ Alerts /    │
     │ (every span) │  │ judge evals  │  │ SLO checks  │
     └──────┬───────┘  └──────┬───────┘  └──────┬──────┘
            │                 │                 │
            └────────┬────────┴─────────────────┘
                     ▼
            ┌─────────────────────┐
            │ Failure-mode review │ (Analyze)
            └─────────┬───────────┘
                      ▼
            ┌─────────────────────┐
            │ Golden-set updates  │ (Measure)
            └─────────┬───────────┘
                      ▼
            ┌─────────────────────┐
            │ Prompt / model / CI │ (Improve)
            │ change + gated PR   │
            └─────────┬───────────┘
                      ▼
                  Back to top.
```

This is Unit 3's Analyze–Measure–Improve loop running forever, with production traces as its constantly refreshed input. The skill this unit teaches is building the loop so it runs **without** heroic on-call effort — so the proof stays valid by default, not by vigilance.

---

## Key Takeaways

1. **Execution → Prediction → Meaning.** Observability is one discipline; the hidden state expands each era. Software o11y recovers execution, MLOps adds model/data state, LLMOps adds prompt, retrieval, grounding, tool use, and semantic-quality state.
2. **The trace is the unit of work.** Every agent turn, pipeline record, and tool call emits one. Everything downstream — debugging, CI, monitoring, alerting — consumes traces.
3. **Debug from traces, not from logs.** The symptom is almost never the cause; drill to the lowest span that changed.
4. **CI is the eval harness.** Golden sets, invariants, red-lines, and judges run on every PR. Prompt changes and model upgrades pass the same gates as code.
5. **Monitor for drift in three places.** Inputs change, outputs change, users change — all three need their own detector.
6. **Alert on quality, not just latency.** Schema validity, judge score, red-line rate, and cost per success are the SLOs that catch silent regressions.
7. **Every incident becomes a test.** Shrink the failing trace, add it to the golden set, gate future PRs on it. Proof is a ratchet.
8. **Evaluation is now a production concern.** The single biggest LLMOps-era addition over MLOps: evals don't stop when you ship. They run continuously on sampled traffic, at every change, and grow with every incident.

---

## Further Reading

### Primary

- Shankar & Husain, *Application-Centric AI Evals for Engineers and Technical Product Managers* — Chapters 8 §8.3 (continuous eval), 9 (production monitoring), 10 §§10.1–10.2 (incident response and regression testing). `resources/llm_eval_course_notes_fall.pdf`.

### Background and prior art

- Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young, Crespo & Dennison, *Hidden Technical Debt in Machine Learning Systems*, NeurIPS 2015 — the foundational "high-interest credit card" paper. CACE, undeclared consumers, unstable data dependencies, hidden feedback loops, the iceberg diagram, and the original case for monitoring prediction quality as an operational concern. Every observability discipline below is, in some form, a debt payment on something this paper named first. [papers.nips.cc/paper/5656](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) (expanded from their 2014 NeurIPS workshop paper, *Machine Learning: The High-Interest Credit Card of Technical Debt*).
- *Observability → MLOps → LLMOps: A Single Steel Thread for Class* — in-class overview of the three-era arc this unit is organized around. `resources/observability_mlops_llmops_steel_thread.md`.
- Kłeczek, Bischof & Husain, *MLOps: A Holistic Approach* (Weights & Biases, 2022) — the People / Processes / Platform frame referenced in the MLOps layer above, plus the canonical failure-mode inventory from the pre-LLM era. `resources/holistic-mlops.pdf`.
- Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) — book-length treatment of the same era's deployment, monitoring, and pipeline design concerns. Most of it ports forward to LLM systems with the adjustments in the "LLMOps layer" table.

### Debugging lineage

- Agans, *Debugging: The 9 Indispensable Rules for Finding Even the Most Elusive Software and Hardware Problems* (AMACOM, 2002) — the mindset half of Paradigm 1; reads in one sitting and generalizes to LLM systems almost unchanged.
- Zeller, *Yesterday, my program worked. Today, it does not. Why?*, ESEC/FSE 1999 — the delta-debugging paper (ddmin algorithm); the academic ancestor of every trace-minimization move in Paradigm 2.
- Torvalds et al., `git bisect` (2005) — the canonical tool that made binary-search across a change history a one-line UX. Lineage: every prompt/model/adapter/index bisect is `git bisect` applied to a larger versioned surface.
- Majors, Fong-Jones & Miranda, *Observability Engineering* (O'Reilly, 2022) — the paradigm shift in Paradigm 3; high-cardinality events and the argument that *you cannot pre-wire dashboards for the questions you don't yet know you need to ask*.

### Tools and specifications

1. OpenTelemetry — [opentelemetry.io/docs/specs](https://opentelemetry.io/docs/specs/) (base observability spec).
2. OpenTelemetry GenAI semantic conventions — [opentelemetry.io/docs/specs/semconv/gen-ai](https://opentelemetry.io/docs/specs/semconv/gen-ai/), including [GenAI client spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/), [GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/), and [GenAI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/) (token usage, latency).
3. Model Context Protocol specification — [modelcontextprotocol.io](https://modelcontextprotocol.io/).
4. Google Cloud — [MLOps: Continuous delivery and automation pipelines in machine learning](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) and [Deploy and operate generative AI applications](https://docs.cloud.google.com/architecture/deploy-operate-generative-ai-applications).
5. Microsoft Learn — [Observability in generative AI](https://learn.microsoft.com/en-us/azure/foundry/concepts/observability) and [Observability for Generative AI and agentic AI systems](https://learn.microsoft.com/en-us/security/zero-trust/sfi/observability-ai-systems).
6. OpenAI — [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) (continuous eval, running evals on every change, growing eval sets over time).
7. LangSmith tracing docs — [docs.smith.langchain.com](https://docs.smith.langchain.com/).
8. Braintrust evaluation and tracing — [braintrust.dev/docs](https://www.braintrust.dev/docs).
9. Logfire (Pydantic) — [pydantic.dev/logfire](https://pydantic.dev/logfire); evals-as-code library [`pydantic-evals`](https://github.com/pydantic/pydantic-evals).
10. Langfuse (OSS LLM engineering platform) — [langfuse.com/docs](https://langfuse.com/docs).
11. Arize Phoenix (OSS LLM tracing, OpenInference standard) — [phoenix.arize.com](https://phoenix.arize.com/).
12. Weights & Biases Weave — [wandb.github.io/weave](https://wandb.github.io/weave/).
13. Helicone (gateway + observability) — [docs.helicone.ai](https://docs.helicone.ai/).

### Vendor positioning and the evals-vs-observability debate

Primary sources for the "modern POV" section above:

- Goyal (Braintrust), *The Three Pillars of AI Observability* (Nov 2025) — the Braintrust reframe of metrics/logs/traces as traces/evals/annotation, and the "is it up?" → "is it good?" punchline. [braintrust.dev/blog/three-pillars-ai-observability](https://www.braintrust.dev/blog/three-pillars-ai-observability).
- Logfire vs Braintrust / LangSmith / Arize Phoenix (Pydantic docs, 2025) — the cleanest written statement of the full-stack observability thesis, including the comparison-page quote used in the lesson. [pydantic.dev/.../comparisons/braintrust](https://pydantic.dev/docs/logfire/get-started/comparisons/braintrust/), [vs LangSmith](https://pydantic.dev/docs/logfire/get-started/comparisons/langsmith/), [vs Arize Phoenix](https://pydantic.dev/docs/logfire/get-started/comparisons/arize-phoenix/).
- LangChain Blog, *Agent Observability Powers Agent Evaluation* and *You don't know what your agent will do until it's in production* — the clearest statement of the observe → evaluate → iterate camp. [langchain.com/blog](https://www.langchain.com/blog/).
- *LLMOps vendor positioning notes* (in-class synthesis of the positioning map across seven vendors). `resources/llmops_vendor_positioning_notes.md`.

### Charity Majors / Honeycomb on LLM observability

- Majors, *Observability in the Age of AI* (Honeycomb, 2023/updated 2025) — the direct case that LLM debugging is a high-cardinality / high-dimensionality problem, the "weird new kind of storage engine" line, and *LLMs demand observability-driven development*. [honeycomb.io/blog/observability-age-of-ai](https://www.honeycomb.io/blog/observability-age-of-ai).
- Honeycomb, *Observability for AI & LLMs* — the product-shaped version of the same thesis, with the SLOs-over-dashboards argument for AI reliability. [honeycomb.io/use-cases/ai-llm-observability](https://www.honeycomb.io/use-cases/ai-llm-observability).
- Honeycomb, *Evaluating Observability Tools for the AI Era* — why the criteria for a good observability platform changed once AI assistants and agentic workflows became the primary consumers of the observability data itself. [honeycomb.io/blog/evaluating-observability-tools-for-the-ai-era](https://www.honeycomb.io/blog/evaluating-observability-tools-for-the-ai-era).
- Pragmatic Engineer podcast, *Observability: the present and future, with Charity Majors* — good audio primer covering the observability-for-AI angle in her own words. [newsletter.pragmaticengineer.com/p/observability-the-present-and-future](https://newsletter.pragmaticengineer.com/p/observability-the-present-and-future).

### Related units

- Unit 3 (Evaluation Fundamentals) — the failure-mode taxonomy, invariants, judges, and the Analyze–Measure–Improve loop this unit runs continuously.
- Unit 9 (Agentic Patterns) — trajectory evaluation and harness spans that emit traces.
- Unit 10 (Multi-Agent Patterns) — per-agent attribution; parent/child span structure.
- Unit 11 (LLM Data Processing) — three-layer eval, extract-then-validate, batch-scale sampling strategies.
- Unit 12 (Model Adaptation) — regression risk on every adapter / base model change.
