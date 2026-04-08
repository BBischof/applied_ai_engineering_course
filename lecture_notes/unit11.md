# Unit 11: LLM-Powered Data Processing

**Date:** Wednesday, April 8, 2026

Units 9 and 10 already changed who “owns” the work: Unit 9 framed **agency** and the **harness** (policy, state, verification, subagents, trajectory proof). Unit 10 framed **decomposition and coordination** — shared context vs isolation, fan-out / fan-in, verifier-style loops, planner–executor handoffs — for long-horizon work.

This unit is a third setting: the **client** of the LLM is not a person in a chat thread — it is a **pipeline orchestrator** (batch jobs, SLAs, millions of records). Agent-style loops still appear **inside** those jobs; what changes is *who owns the loop* (scheduler, queue, checkpoint store) rather than a live chat. Examples in this lecture: **Harvey** (staged subagents per deal), **Hebbia** (subagent per analytical step), **Walmart** (worker–checker per catalog row), **Reducto / Elysian** (multi-pass agentic OCR with gates), and **gleaning** with **DocETL** (generate–validate–re-prompt until outputs pass checks).

**This lecture** follows one thread: **LLMs went from answering questions to doing work — and that required solving increasingly hard engineering problems.** Each section introduces a key idea, shows how real companies solved it, and connects it back to P³.

### Today's session

LLMs as **data operators** at production scale — primitives and MapReduce-style pitfalls, case studies on trust, economics, evaluation, and provenance, declarative pipelines (MCP, skills), and **P³** end to end.

### DocETL and DocWrangler

Two systems from the UC Berkeley EPIC Lab that you will see in depth today.

**DocETL** [1] is a declarative pipeline framework with an **optimizer**: you write a sequence of LLM operators (map, reduce, resolve, etc.) and the system can automatically rewrite that pipeline for better accuracy — inserting decomposition steps when a document is too long, adding **gleaning loops** (generate → validate → re-prompt) when single-pass quality is too low, and detecting when a resolve should precede a reduce.

**DocWrangler** [18] adds an interactive layer on top: inspect outputs at each step, annotate errors, and have the system **refine prompts** based on your annotations. Think of it as the Analyze–Measure–Improve loop applied to pipeline development rather than prompt engineering.

Both show up again in the **Declarative pipelines** section later in these notes.

### After Units 9 and 10 (vocabulary to reuse)

| Unit 9–10 idea | How it shows up in batch / data processing |
|----------------|--------------------------------------------|
| **Harness** (gates, checkpoints, traces) | Schema validation, confidence routing, human review queues, immutable provenance — the runtime that wraps each operator. |
| **Subagents / scoped context** | Harvey-style stages (clauses → cross-refs → risk): same pattern as Unit 9, with handoffs between steps instead of one giant prompt. |
| **Verifier loop** | Extract-then-validate (e.g., Walmart): worker + checker; same structural idea as “model proposes / model or rule verifies,” tuned for parseable outputs. |
| **Fan-out / fan-in** | Parallel **map** over chunks or records; **gather / reduce** is where Unit 10’s “merge with context limits and ordering” reappears (stuff vs tree reduce). |
| **DAG vs loops** | Declarative configs often *look* linear; production needs gleaning, branching, and pause/resume — the “why not a DAG?” point aligns with Unit 10’s focus on **control flow and synchronization**, not cartoon multi-agent chatter. |
| **Determinism × agency (Unit 9)** | Many catalog pipelines are **low agency per record** (fixed operator graph) but push **high determinism** (schemas, validators, audits). Adaptive optimizers (DocETL) raise effective capability by rewriting that graph — closer to Unit 10’s “adaptive decomposition” than to a single chatty agent. |

---

## Warm-up: when chat isn't the product

Consider what these companies actually need:

- Walmart needs to extract color, size, and material from product descriptions — across **850 million** catalog entries [9]
- Elysian (commercial insurance automation) needs to review insurance claims averaging **5,400 pages** of scanned faxes and handwritten annotations [8]
- Abridge (clinical AI documentation) needs to generate clinical notes from doctor-patient conversations — where a single hallucinated detail could harm a patient [4]. This is hallucination (Unit 2) at pipeline scale — not a chatbot giving a wrong answer a user can question, but wrong data silently entering production systems.

None of these are chat problems. They're data-processing problems that happen to require language understanding. The model isn't the product — it's an **operator** inside a larger system.

---

## The recurring question (our analytical lens)

For each key idea, we'll ask the same three questions — mapping directly to P³:

1. **What's the promise?** What does the system need to deliver, and what are the acceptance criteria?
2. **How do you prove it?** How did the team measure quality before scaling — and how do they measure it continuously?
3. **What makes it production-grade?** What engineering decisions keep the proof valid at scale?

---

## Key Idea 1: LLMs as data operators

### The conceptual shift

Before LLMs, data processing meant rules and trained classifiers: regex for extraction, supervised models for classification, templates for generation. Each task needed its own labeled dataset, its own model, its own deployment. Adding a new category or handling a new document format meant retraining.

LLMs collapse this. A single model can classify, extract, summarize, resolve entities, and enrich records — guided by a prompt instead of a training set.

### The fundamental primitives

| Operator | What it does | SQL analogy |
|----------|-------------|-------------|
| **Map** | Transform each record independently | `SELECT f(col)` |
| **Filter** | Keep or discard records | `WHERE` |
| **Classify** | Assign a label from a fixed set | `CASE WHEN` |
| **Reduce** | Aggregate multiple records into one | `GROUP BY + AGG` |
| **Resolve** | Deduplicate / merge records referring to the same entity | Self-join + merge |
| **Enrich** | Add new fields derived from existing content | Computed column |

Every operator produces **structured output** — a Pydantic model or JSON schema that the next step can parse and validate. This is the interface contract between pipeline steps. At batch scale, a 1% parse-failure rate on 50K records = 500 broken records, so schema enforcement (constrained decoding from Unit 5) matters even more here than in chat. Here's what a single classify operator looks like:

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

class TicketClassification(BaseModel):
    category: Literal[
        "billing", "technical", "account", "security", "other"
    ]
    priority: Literal["low", "medium", "high"]
    confidence: float

client = OpenAI()

def classify_ticket(ticket: dict) -> TicketClassification:
    return client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=TicketClassification,
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": (
                f"Subject: {ticket['subject']}\n"
                f"Body: {ticket['body']}"
            )},
        ],
    ).choices[0].message.parsed
```

Under the hood, this is the function-calling mechanism from Unit 5: structured input → model call with schema → typed output. The operator abstraction is syntactic sugar over that loop.

### The MapReduce analogy

If you've seen MapReduce in a distributed systems or data engineering context, the parallel is direct — and the differences are instructive.

In classic MapReduce:
- **Map** applies a deterministic function to each record independently. Same input → same output, every time.
- **Reduce** aggregates mapped outputs using an associative, commutative combiner. `sum(a, b) = sum(b, a)` — order doesn't matter.

In LLM data processing:
- **Map** applies a *prompt* to each record. Same input → *similar* output most of the time, but nondeterministic. The function can hallucinate, drift, or change behavior when the provider updates the model.
- **Reduce** asks the LLM to summarize, merge, or synthesize across records. Order *does* matter (lost-in-the-middle applies), context limits constrain how many records you can see at once, and the output is lossy in ways that `SUM` or `COUNT` never are.

| | Classic MapReduce | LLM MapReduce |
|---|---|---|
| **Map function** | Deterministic | Nondeterministic (prompt-based) |
| **Reduce function** | Associative, commutative | Order-sensitive, context-limited |
| **Failure mode** | Crash (retry works) | Silent wrong answer (retry may give *different* wrong answer) |
| **Scaling** | Add machines | Add API concurrency (rate-limited) |
| **Cost driver** | Compute time | Tokens × price per token |

This analogy is useful because it tells you what carries over and what doesn't. Parallelism carries over — map is still embarrassingly parallel. Fault tolerance doesn't carry over cleanly — retrying a failed LLM call may produce a different (not necessarily better) result. And reduce needs careful architectural attention that classic MapReduce's combiner pattern doesn't prepare you for.

### Fan-out / fan-in: runtime parallelism

Classic MapReduce has a fixed number of mappers decided at deploy time. In LLM pipelines, the number of parallel tasks often depends on the input data itself — a document with 3 sections needs 3 parallel extractors, but another might need 50.

This is the **fan-out / fan-in** (scatter-gather) pattern:

```
Input → Split into N chunks (N depends on document)
         ├→ LLM call (chunk 1)
         ├→ LLM call (chunk 2)
         ├→ …
         └→ LLM call (chunk N)
              └→ Gather / reduce results
```

The implementation detail that matters: each parallel branch can carry its own state (different chunk text, different schema fields, different model choice) while sharing a common output format for the gather step. This is how you process a 200-page contract: split into sections, fan out extraction across all sections in parallel, fan in the structured results.

### Checkpointing: human-in-the-loop at the record level

In batch processing, some records need human review. The simplest version is binary: every record either **passes** or gets **flagged**.

1. Process each record through the pipeline
2. If a validation step fails (schema error, missing field, disallowed label) or the output is marked as potentially requiring human feedback, **flag** the record and save its state
3. Continue processing the rest of the batch
4. A human reviews the flagged records and provides corrections
5. Corrected records re-enter the pipeline from where they stopped

Once this works, you graduate to **thresholds** — confidence scores, agreement between redundant calls, domain-specific rules — that route records into pass / review / reject buckets instead of a binary split. But start binary: it’s easier to debug and already handles the worst failures.

Either way, the batch doesn’t block on human review — records that clear validation finish while flagged records wait in a review queue. Each record maintains its own checkpoint, so resuming one doesn’t affect others.

### Map: the workhorse

Most pipeline steps are **map** operations — apply the same prompt to each record. Parallelism is straightforward:

```python
async def llm_map(
    records: list[dict],
    prompt_template: str,
    output_schema: type[BaseModel],
    model: str = "gpt-4o-mini",
    concurrency: int = 20,
) -> list[BaseModel]:
    """Apply an LLM prompt to each record in parallel."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(record: dict) -> BaseModel:
        async with semaphore:
            prompt = prompt_template.format(**record)
            return await call_llm(prompt, output_schema, model)

    return await asyncio.gather(
        *[process_one(r) for r in records]
    )
```

### Reduce: where the analogy breaks

LLM reduce is hard because:
- **Context limits:** 100 records × 500 tokens each = 50K tokens. You can't always fit everything in one call.
- **Order effects:** which records the LLM sees first influences the output (recall lost-in-the-middle from Unit 2).
- **No commutativity:** `reduce(A, B)` ≠ `reduce(B, A)` in practice.

Three strategies, each with tradeoffs:

| Strategy | How it works | Accuracy | Cost |
|----------|-------------|----------|------|
| **Stuff** | Fit all records into one call | Best (if it fits) | One call |
| **Incremental (fold)** | Process records one at a time, updating a running summary | Recency-biased | N calls |
| **Tree reduce** | Reduce pairs → reduce results → … until one remains | Balanced | log(N) calls |

The right choice depends on how many records you're reducing and whether the full set fits in context. DocETL's optimizer can detect when stuff-reduce fails and automatically switch to tree reduce. This three-strategy decision is the exact same problem RAPTOR solves in retrieval (Unit 7): how do you build a hierarchy of summaries so you can answer both broad and narrow questions? The answer in both cases is tree-based summarization — chunk, summarize clusters, summarize summaries. The difference is that RAPTOR does it at index-build time for retrieval, while your reduce pipeline does it at processing time for batch data.

#### Reduce in practice: three worked examples

**Example 1: Summarize a patient's visit.** A patient sees three specialists in one hospital visit. Each produces a clinical note (~800 tokens). The discharge summary needs to integrate all three into a coherent narrative with no contradictions.

```
Cardiology note  ─┐
Neurology note   ─┼─→ LLM reduce → Integrated discharge summary
Radiology report ─┘
```

This is a **stuff** reduce — three notes fit in context. But the task is harder than it sounds: the LLM must reconcile overlapping medication lists, resolve conflicting observations ("stable" vs. "concerning"), and preserve every clinician's key findings without hallucinating connections that weren't stated. A traditional `GROUP BY` would never face this problem.

**Example 2: Synthesize themes across 200 customer interviews.** A product team has 200 interview transcripts (~2,000 tokens each = 400K total). They want 5–7 themes with supporting quotes.

```
Interviews (200)
  ├─ Map: extract key quotes + preliminary themes per interview
  ├─ Tree reduce (depth 3): merge 200 → 25 → 4 → 1
  └─ Final summary: 5–7 themes with representative quotes
```

**Stuff** is impossible (400K tokens). **Incremental fold** would be recency-biased — later interviews overwrite earlier themes. **Tree reduce** works: merge clusters of ~8 interviews into theme summaries, then merge those summaries, then merge again. Each merge level is a "summarize and integrate" call, and information is lost at every level — the art is losing the right information.

**Example 3: Aggregate financial data across 50 quarterly reports.** An analyst wants a trend summary of revenue, margins, and risk factors across 50 10-Q filings for one company.

```
10-Q filings (50)
  ├─ Map: extract {revenue, margin, risk_factors} per quarter
  ├─ Sort by date (deterministic, not LLM)
  └─ Reduce: "Given these 50 quarterly snapshots, identify
             trends, inflection points, and emerging risks"
```

Here the **map** step produces structured data, and the **reduce** operates on structured intermediate results — not raw text. This is a common pattern: use map to compress unstructured documents into structured summaries, then reduce over the structured output. The reduce input is much smaller and more predictable.

The lesson across all three: LLM reduce isn't just "call the model on a bigger input." It requires choosing a strategy (stuff, fold, or tree), deciding what information to preserve at each level, and often pre-processing with map to make the reduce tractable.

### Why this matters: the cold-start advantage

**DoorDash** uses LLMs for attribute extraction across their merchant catalog [13]. New merchants onboard constantly, each with different menu formats and zero labeled training data. Traditional NLP needs labeled examples per category — retraining for every new merchant. LLMs generalize from instructions, solving the cold-start problem. Compare against a simple baseline (Unit 1) before celebrating — a regex extractor may handle 60% of records at near-zero cost, and you need the baseline to quantify the LLM's marginal value.

**Instacart's PARSE** system uses multimodal LLMs to extract product attributes (flavor, size, nutritional content) from millions of SKUs [11]. The key: it handles ambiguities that rule-based and traditional ML systems couldn't — like distinguishing a product's primary flavor from variant flavors mentioned in the same description. This is **representation engineering** (Unit 7) applied to data pipelines: Instacart is creating multiple structured representations of each SKU — not to make it *findable* in a search index, but to make it *processable* by downstream systems. The same principle applies in both domains: you can only work with what you've represented.

> **P³ lens — Promise:** the promise is "extract structured attributes
> from any merchant's data without task-specific training." The
> acceptance criteria: coverage (what % of SKUs have correct
> attributes) and accuracy (what % of extracted values are right).

---

## Key Idea 2: The accuracy problem

### Errors compound

If each step in your pipeline has 95% accuracy, a 5-step pipeline has:

$$0.95^5 = 0.774$$

Nearly 1 in 4 records has at least one error. In a chat, the user re-asks. In a pipeline, bad data propagates silently into downstream systems. This is the central engineering challenge. Slice these rates by record type and difficulty (Unit 1) — a 95% average can hide 70% accuracy on your hardest records. This is the data-processing equivalent of context rot (Unit 7): more steps, like more context, doesn't mean better — it means more opportunities for errors to compound.

### Pattern: extract-then-validate

The most widespread solution is a two-agent pattern: one LLM does the work, a second LLM (or deterministic check) validates it. Framed with Units 9–10: this is a **specialized verifier loop** and **separation of worker vs checker context** — not “multi-agent banter,” but the same reliability pattern you saw for long-horizon agents, here applied at **record throughput** with structured outputs.

**Walmart** uses exactly this — one LLM extracts product attributes, a separate LLM quality-checks the extraction [9]. The result: improved over 850 million product data entries, with superior recall vs. supervised models on diverse descriptions.

**Harvey AI** takes it further with agentic multi-step extraction for legal documents [5]. Each sub-agent handles a specific reasoning task: clause identification, cross-reference resolution, risk classification. The result: 98.47% accuracy on SPA deal-point extraction, vs. 66% for raw GPT-4o on the same documents [5]. The model is the same — the pipeline architecture makes the difference. Harvey's sub-agents are the subagent pattern from Unit 5 — each with its own context, tools, and system prompt — and the staged pipeline is the Unit 9 **harness** plus Unit 10 **decomposition** idea in document form: narrow handoffs instead of one model holding the whole brief. A sketch of what this looks like:

```python
async def analyze_spa(document: str) -> SPAAnalysis:
    clauses = await run_subagent(
        system_prompt=CLAUSE_ID_PROMPT,
        tools=[search_document, list_sections],
        model="gpt-4o-mini",
        input=document,
        output_schema=list[Clause],
    )
    cross_refs = await run_subagent(
        system_prompt=CROSS_REF_PROMPT,
        tools=[lookup_defined_term, resolve_reference],
        model="gpt-4o",  # harder task, better model
        input=clauses,
        output_schema=list[ResolvedClause],
    )
    risk = await run_subagent(
        system_prompt=RISK_CLASSIFY_PROMPT,
        tools=[],  # pure classification, no tools
        model="gpt-4o-mini",
        input=cross_refs,
        output_schema=list[RiskAssessment],
    )
    return SPAAnalysis(
        clauses=clauses,
        cross_references=cross_refs,
        risk_assessments=risk,
    )
```

The clause-ID agent gets document search tools; the cross-reference agent gets defined-term lookup and a more expensive model (because the task is harder); the risk classifier needs no tools at all. Context isolation means the cross-reference agent doesn't see 50 pages of irrelevant clause text — just the clauses from step 1.

> **P³ lens — Proof:** Harvey built BigLaw Bench, a custom benchmark
> developed with law firms to measure real legal task performance [6].
> That's the proof instrument. They didn't ship and hope — they
> defined the bar first, measured against it, and published the
> results.

### Pattern: decomposition

When documents are too long or tasks too complex for a single call:

```
Split → Map → Gather
```

1. **Split** the document into chunks (by section, page, paragraph)
2. **Map** the LLM operation over each chunk independently
3. **Gather** context from surrounding chunks so the LLM can see local context

**Elysian + Reducto** faces this at the extreme — 5,400-page insurance claims with scanned faxes and handwritten annotations [8]. Reducto's pipeline runs three stages: (1) computer vision layout parsing to segment pages into regions, (2) vision-language models to interpret each region, (3) agentic OCR with multi-pass correction until confidence gates pass. Result: 16x faster claim review.

The lesson: before you can run LLMs on documents, you need reliable **ingestion**. The parsing pipeline is often harder than the analysis pipeline. In Unit 7's Collector → Ranker → Server framework, this is the Collector's job — and it sets the ceiling for everything downstream.

### Validation between steps

Every step can fail silently. In practice, a pipeline interleaves LLM operators with deterministic tool calls — the LLM handles judgment, the tools handle precision:

```python
async def enrich_product(product: dict) -> EnrichedProduct:
    barcode_data = await barcode_api.lookup(product["upc"])
    pricing = await pricing_db.get(product["sku"])

    attributes = await call_llm(
        prompt=EXTRACT_ATTRS_PROMPT.format(
            description=product["description"],
            image_url=product.get("image_url"),
        ),
        output_schema=ProductAttributes,
        model="gpt-4o-mini",
    )
    summary = await call_llm(
        prompt=SUMMARIZE_PROMPT.format(
            name=product["name"],
            attributes=attributes.model_dump_json(),
            price=pricing.current_price,
        ),
        output_schema=ProductSummary,
        model="gpt-4o-mini",
    )

    # Deterministic validation — no LLM needed
    taxonomy_check = taxonomy_service.validate(
        category=attributes.category,
        subcategory=attributes.subcategory,
    )
    return EnrichedProduct(
        **product,
        barcode=barcode_data,
        pricing=pricing,
        attributes=attributes,
        summary=summary,
        taxonomy_valid=taxonomy_check.is_valid,
    )
```

The barcode lookup, pricing fetch, and taxonomy validation are traditional tool calls — fast, deterministic, cheap. The attribute extraction and summary generation are LLM calls — slow, nondeterministic, require structured output schemas. The pipeline interleaves both.

Insert validation between LLM steps:

- **Schema validation:** does the output parse? Required fields present?
- **Invariant checks:** labels from the allowed set? Dates in range?
- **Confidence filtering:** route uncertain outputs to a more expensive model or human review
- **Cross-record consistency:** do extracted entities match across related documents?

This validation pattern is the data-pipeline equivalent of pre-filtering in retrieval (Unit 7): narrow the candidate set *before* the expensive step runs. In retrieval, you pre-filter on metadata so the ANN search only sees eligible candidates. In data pipelines, you validate outputs before passing them downstream so the next operator only sees clean inputs. Same principle — fail fast and cheap, not slow and expensive.

---

## Key Idea 3: The economics problem

### Cost at volume

At scale, token costs dominate:

| Scenario | Model | Cost for 50K records |
|----------|-------|---------------------|
| gpt-4o-mini (800 in / 200 out tokens per record) | Cheap | ~$12 |
| gpt-4o (same tokens) | Expensive | ~$600 |
| gpt-4o with 5-step pipeline | Expensive × 5 | ~$3,000 |

Model choice matters enormously at volume.

### Pattern: model routing

Not every record needs the same model. Route cheap-model-first, escalate on uncertainty:

```
Record → Cheap Model → Confidence check
                         │
              High ───→ Accept
              Low  ───→ Expensive Model → Accept
```

**Amazon's catalog team** built a self-learning version of this [10]. Multiple smaller models process routine products; when they disagree, a larger supervisor investigates. The key innovation: disagreements generate reusable learnings stored in a dynamic knowledge base. Over time, the cheap models handle increasingly complex cases without needing the supervisor — the system teaches itself.

**Hebbia** routes across o3-mini, o1, and GPT-4o depending on step complexity [7]. 92% accuracy on complex financial tasks, vs. 68% for traditional RAG. The routing decision is part of the architecture, not an afterthought. Under the hood, each analytical step is a subagent with its own model and tool set — the multi-agent pattern from Unit 5, applied to document analysis rather than chat.

**Unit 10 link:** Amazon’s disagreement-then-supervisor flow and Hebbia’s per-step model choice are both **routing** — deciding which capability runs next, on what evidence or confidence signal. That is the same design space Unit 10 used for long-horizon agents (routers, escalation, planners); here it is specialized for **batch throughput and unit economics** on schema-bound outputs.

### Pattern: fine-tuning for economics

**Ramp** processes every corporate receipt through an LLM: OCR → fine-tuned model → structured JSON [14]. They started with a general-purpose model to prove the task works, then fine-tuned a smaller model for production. Result: 97% categorization accuracy, 34% fewer receipts needing manual review, **79% cost savings** vs. the general-purpose API.

> **P³ lens — Production:** Ramp's story is a textbook P³ arc.
> **Promise:** categorize receipts accurately at scale.
> **Proof:** general-purpose model demonstrates the task is feasible;
> golden dataset measures 97% accuracy. **Production:** fine-tune for
> economics, batch-process on Modal with 256 parallel workers, monitor
> the 34% manual-review rate as an ongoing metric.

### Batch APIs and throughput

Provider batch endpoints change the math:
- OpenAI Batch API: 50% off, 24-hour turnaround
- Anthropic Message Batches: 50% off, up to 100K requests per batch

**Instacart's Maple** service goes further — an internal batch platform that runs LLM workloads across teams (Catalog, Fulfillment, Search), reducing costs 50% vs. real-time API calls [12]. Uses Temporal for fault tolerance and Parquet for data handling. This is what the **Production** phase of P³ looks like for LLM data processing at scale.

---

## Key Idea 4: Eval as the scaling lever

You can't human-label 50,000 outputs. So how do you know if your pipeline works?

### The three-layer eval strategy

| Layer | What it covers | Cost | Coverage |
|-------|---------------|------|----------|
| **Invariant checks** | Schema, label sets, ranges, non-null, cross-field consistency | Free | 100% of outputs |
| **Sampled human eval** | Stratified sample (200–500 records), human-labeled, per-step metrics | Expensive | 1–5% of outputs |
| **LLM-as-judge** | Subjective quality checks on a sample | Cheap | 5–20% of outputs |

This is the same per-stage measurement principle from Unit 7's RAG evaluation: measure at each pipeline step, not just end-to-end. In RAG, you measure Collector recall, Ranker precision, and Server faithfulness separately. In data pipelines, you measure each operator's accuracy independently. A single end-to-end metric hides which component is failing — the "metric laundering" pitfall from Unit 3.

Goodhart's Law (Unit 1) applies doubly at pipeline scale — if invariant pass-rate becomes the optimization target, teams will simplify schemas rather than improving extraction quality.

### Spotify's eval story

**Spotify** replaced an entire stack of specialized ML models (sentiment, topic modeling, speech classification, ad detection) with a single LLM for podcast previews [2]. 4.6% engagement lift, 5x processing efficiency.

But the eval story is the real contribution. They built a "profile-aware LLM-as-judge" [3]: summarize a listener's preferences from ~90 days of activity, then have the LLM score how well candidate episodes match. 75% alignment with listener judgments — good enough to filter candidates without running an A/B test on everything.

> **P³ lens — Proof:** Spotify's judge is a proof instrument
> calibrated against real user behavior data. It sits between slow-
> but-accurate human labeling and fast-but-noisy offline metrics.
> The feedback loop: judge scores → identify underperforming
> recommendations → improve the pipeline → re-score.

### Progressive processing

A practical pattern: don't process 50K records and then evaluate. Instead:

1. Process a **sample** (200–500 records)
2. Run full eval (human + invariants + judge)
3. Fix issues in the prompt/pipeline
4. Scale to full corpus
5. Run invariants on 100% and judge on a sample
6. Feed new failures back into the eval set

---

## Key Idea 5: The trust problem

### Provenance: tracing every output to its source

In high-stakes domains, "the model said so" isn't enough. You need to show **where** in the source data each output came from.

**Abridge** generates clinical notes from doctor-patient conversations [15]. Every sentence in the note links back to the source audio via provenance tracking. Their published research on "confabulation elimination" [4] describes purpose-built guardrails backed by 1,000+ hours of human validation — 6x more likely to catch and fix hallucinations vs. off-the-shelf models.

They've gone further with specialty-specific models: tuned for neurology (precise symptom chronology), surgery (risk-benefit documentation), etc. — deployed without requiring clinicians to change their workflow [15].

**Harvey** similarly traces every extracted deal point back to the specific clause and cross-reference in the source SPA [5]. This isn't just nice-to-have — in legal and medical contexts, an output you can't trace is an output you can't use.

> **P³ lens — all three:** Abridge's P³ story is rigorous.
> **Promise:** "generate clinical notes without fabricating
> information" — the acceptance criteria include zero confabulation on
> critical details. **Proof:** 1,000+ hours of human validation,
> confabulation detection research, specialty-specific evaluation
> against clinician feedback [4, 16]. **Production:** provenance
> tracking as an architectural choice, specialty models deployed
> behind the same interface, continuous monitoring through structured
> evaluation checklists.

---

## The P³ matrix across case studies

| Company | Promise | Proof | Production |
|---------|---------|-------|------------|
| **Harvey** | Extract deal points with lawyer-grade accuracy | BigLaw Bench (custom eval with law firms) [5, 6] | Agentic pipeline with sub-agents; Vault processes thousands of docs per matter |
| **Hebbia** | Structured insights from 10K+ docs per query | 92% accuracy vs. 68% RAG baseline [7] | Multi-model routing (o3-mini/o1/GPT-4o); distributed orchestration engine |
| **Walmart** | Correct attributes across 850M catalog entries | Two-agent extract + validate loop [9] | LLMs replaced work that would have taken 100x manual labor |
| **Amazon** | Self-improving catalog extraction | Consensus-based routing; disagreements become learnings [10] | Dynamic knowledge base; smaller models improve over time without retraining |
| **Instacart** | Flavor, size, nutrition from millions of SKUs | PARSE multimodal extraction [11] | Maple batch service across teams; 50% cost reduction [12] |
| **DoorDash** | Attribute extraction for any new merchant | LLMs generalize from instructions; no per-merchant training [13] | Knowledge graph enrichment; continuous onboarding |
| **Ramp** | 97% receipt categorization at scale | General model → measure → fine-tune → re-measure [14] | Fine-tuned model on Modal; 256 parallel workers; 79% cost savings |
| **Spotify** | Better podcast previews than specialized ML stack | Profile-aware LLM-as-judge (75% agreement with listeners) [2, 3] | Replaced 5 specialized models; 5x processing efficiency |
| **Abridge** | Clinical notes without fabrication | 1,000+ hours human validation; confabulation research [4] | Provenance tracking; specialty models; structured eval checklists [15, 16] |
| **Elysian** | 16x faster insurance claim review | Audit-grade accuracy on 5,400-page claims [8] | Reducto 3-stage ingestion; agentic OCR with confidence gates [19] |

---

## Declarative pipelines

The patterns above — operators, validation, routing, decomposition — converge in **declarative LLM pipelines**: systems where you specify *what* to do (in config) and the platform handles *how* (parallelism, batching, retries, optimization).

```yaml
pipeline:
  - name: extract_fields
    type: map
    prompt: prompts/extract_contract_fields.txt
    output_schema: ContractFields
    model: gpt-4o-mini

  - name: validate_extraction
    type: filter
    condition: "output.confidence > 0.8"

  - name: classify_risk
    type: classify
    prompt: prompts/classify_risk_level.txt
    labels: [low, medium, high, critical]
    model: gpt-4o

  - name: summarize_by_vendor
    type: reduce
    group_by: vendor_name
    prompt: prompts/summarize_vendor_contracts.txt
    model: gpt-4o
```

Notice that every step declares an `output_schema` or `labels` — structured output is the contract between pipeline steps. The platform can validate outputs against the schema automatically and route parse failures to retry logic.

### Why not a DAG?

Early pipeline systems modeled workflows as directed acyclic graphs (DAGs) — step A feeds step B feeds step C, no cycles. This works for simple ETL but breaks for LLM pipelines because:

- **Gleaning needs loops:** generate → validate → re-prompt → validate again
- **Confidence routing needs branching:** high-confidence records skip expensive steps; low-confidence records get extra processing
- **Human review needs pause/resume:** the pipeline suspends on uncertain records and resumes when a human provides input

Real LLM data pipelines need **event-driven** or **state-machine** orchestration, not rigid DAGs. The pipeline config above looks linear, but the runtime may loop, branch, and checkpoint underneath.

**Unit 10 link:** This matches what Unit 10 stressed about long-horizon work: **control flow and synchronization** (when to repeat, branch, or wait) dominate; a static DAG diagram often **under-describes** the real system.

### Reusable operators via MCP

When an operator is valuable across multiple pipelines or clients, package it as an MCP server (from Unit 5). Here's a contract extraction operator exposed via MCP:

```python
from mcp.server import Server
from mcp.types import TextContent

app = Server("contract-extraction")

@app.tool()
async def extract_contract_fields(
    document_text: str,
) -> list[TextContent]:
    """Extract vendor, dates, value, and risk from a contract."""
    result = await call_llm(
        prompt=EXTRACT_PROMPT.format(text=document_text),
        output_schema=ContractFields,
        model="gpt-4o-mini",
    )
    return [TextContent(
        type="text",
        text=result.model_dump_json(),
    )]
```

Now any client — a batch pipeline, a Slack bot, a Cursor workflow, an analyst's notebook — can call `extract_contract_fields` without reimplementing the logic. The batch pipeline calls it 50,000 times; the Slack bot calls it once. Same server, same prompt, same schema.

### Pipelines as skills

Recall **skills** from Unit 5 — reusable, file-based workflow expertise that an agent loads on demand. A declarative pipeline config is essentially a machine-readable skill: it encodes the procedure ("extract, then validate, then classify, then summarize"), the tools to use (which model, which schema, which prompt), and the quality checks. Here's what a `SKILL.md` might look like for insurance claim review:

```markdown
---
name: insurance-claim-review
description: Process commercial insurance claims end-to-end
triggers:
  - "review insurance claim"
  - "process claim documents"
---

## Procedure

1. **Ingest** all claim attachments using Reducto parse API
2. **Classify** each document (policy, incident report, medical record, correspondence, photo, estimate)
3. **Extract** structured fields per document type
4. **Resolve** entities across documents
5. **Validate** extracted fields against policy terms
6. **Summarize** the full claim with citations to source documents
7. **Flag** any inconsistencies for human adjuster review

## Quality checks
- Every extraction must pass schema validation
- Cross-document entity resolution must link ≥90% of providers
- Summary must cite at least one source document per claim
```

When Harvey builds a workflow for SPA deal-point extraction, or Walmart builds one for catalog enrichment, they're codifying expertise that was previously in a human analyst's head. The pipeline *is* the skill — portable, versionable, and executable.

---

## Key Takeaways

1. **LLMs are data operators, not just chatbots.** The primitives
   — map, filter, classify, reduce, resolve, enrich — are the building blocks. DoorDash and Instacart use them because LLMs solve the cold-start problem that traditional NLP can't.
2. **Errors compound; validate between steps.** The extract-then-validate pattern (Walmart, Harvey) is universal. Two agents — one to work, one to check — is the simplest reliable architecture.
3. **Route models by difficulty.** Amazon's consensus routing and Hebbia's multi-model orchestration show that not every record needs the same model. Prove with expensive, run with cheap.
4. **Fine-tuning is a cost optimization, not a capability unlock.** Ramp's trajectory: general model → measure → fine-tune → 79% savings. Proof first, then optimize for production economics.
5. **Eval at scale = invariants + sampling + LLM-judge.** Spotify's profile-aware judge is the gold standard for scaling subjective eval without A/B testing everything.
6. **In high-stakes domains, provenance is mandatory.** Abridge and Harvey both trace every output to its source. If you can't show where it came from, you can't use it.
7. **Ingestion is often the hardest step.** Reducto's three-stage pipeline for 5,400-page insurance claims shows that parsing precedes understanding.

---

## Further Reading

### Research papers

1. Shankar et al., *DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing* (2024) —
   [arxiv.org/abs/2410.12189](https://arxiv.org/abs/2410.12189)
2. Zhu et al., *Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems*, ACL 2025 Industry Track —
   [aclanthology.org/2025.acl-industry.26](https://aclanthology.org/2025.acl-industry.26/)
3. Fabbri et al., *Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge*, RecSys 2025 —
   [arxiv.org/abs/2508.08777](https://arxiv.org/abs/2508.08777)
4. Liang et al., *The Science of Confabulation Elimination: Toward Hallucination-Free AI-Generated Clinical Notes*, Abridge (2025) —
   [abridge.com/ai/science-confabulation-hallucination-elimination](https://www.abridge.com/ai/science-confabulation-hallucination-elimination)

### Engineering blogs and case studies

5. Harvey AI, *BigLaw Bench Workflows: SPA Deal Points* (Sep 2024) —
   [harvey.ai/blog/biglaw-bench-workflows-spa-deal-points](https://www.harvey.ai/blog/biglaw-bench-workflows-spa-deal-points)
6. Harvey AI, *Introducing BigLaw Bench* (Sep 2024) —
   [harvey.ai/blog/introducing-biglaw-bench](https://www.harvey.ai/blog/introducing-biglaw-bench)
7. OpenAI, *Hebbia's Deep Research Automates 90% of Finance and Legal Work* (2025) —
   [openai.com/index/hebbia](https://openai.com/index/hebbia)
8. Reducto, *How Elysian Uses Reducto to Review Insurance Claims 16x Faster* (Sep 2025) —
   [reducto.ai/blog/reducto-elysian-case-study](https://reducto.ai/blog/reducto-elysian-case-study)
9. Walmart Global Tech, *How Walmart Uses LLMs to Manage Its Massive Product Catalogs* (May 2025) —
   [tech.walmart.com/.../using-llms-to-manage-product-catalogs](https://tech.walmart.com/content/walmart-global-tech/en_us/blog/post/using-llms-to-manage-product-catalogs.html)
10. AWS ML Blog, *How the Amazon.com Catalog Team Built Self-Learning
    Generative AI at Scale with Amazon Bedrock* (2025) —
    [aws.amazon.com/blogs/machine-learning/...catalog-team...bedrock](https://aws.amazon.com/blogs/machine-learning/how-the-amazon-com-catalog-team-built-self-learning-generative-ai-at-scale-with-amazon-bedrock/)
11. Instacart Tech, *Scaling Catalog Attribute Extraction with
    Multi-modal LLMs* (Aug 2025) —
    [instacart.com/.../scaling-catalog-attribute-extraction...](https://www.instacart.com/company/tech-innovation/scaling-catalog-attribute-extraction-with-multi-modal-llms)
12. Instacart Tech, *Simplifying Large-Scale LLM Processing Across
    Instacart with Maple* (Aug 2025) —
    [instacart.com/.../simplifying-large-scale-llm-processing...maple](https://www.instacart.com/company/how-its-made/simplifying-large-scale-llm-processing-across-instacart-with-maple/)
13. DoorDash Engineering, *Building DoorDash's Product Knowledge
    Graph with Large Language Models* (2024) —
    [careersatdoordash.com/blog/...product-knowledge-graph...](https://careersatdoordash.com/blog/building-doordashs-product-knowledge-graph-with-large-language-models/)
14. Modal, *How Ramp Automated Receipt Processing with Fine-Tuned
    LLMs* (2025) —
    [modal.com/blog/ramp-case-study](https://modal.com/blog/ramp-case-study)
15. Abridge, *Innovation at the Speed of Trust: Continuous
    Improvement in Clinical AI* (Dec 2025) —
    [abridge.com/blog/clinical-ai-specialty-models](https://www.abridge.com/blog/clinical-ai-specialty-models)
16. Abridge, *Pioneering the Science of AI Evaluation* —
    [abridge.com/ai/science-ai-evaluation](https://www.abridge.com/ai/science-ai-evaluation)

### Tools and platforms referenced

17. DocETL documentation — [docetl.org](https://www.docetl.org/)
18. DocWrangler blog post — [Interactive LLM-Powered Data Processing
    with DocWrangler](https://data-people-group.github.io/blogs/2025/01/13/docwrangler/)
19. Reducto technical overview — [Hybrid Architecture: Agentic OCR
    Deep Dive](https://llms.reducto.ai/hybrid-architecture-agentic-ocr-deep-dive)
