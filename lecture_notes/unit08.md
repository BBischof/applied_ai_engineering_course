# Unit 8: Context Engineering

**Date:** Wednesday, March 11, 2026

Unit 7 ended with the server side of RAG: given a set of retrieved documents,
memories, user messages, and tool results, what actually goes into the prompt?
Unit 8 makes that question explicit. Context engineering is the discipline of
choosing, ordering, compressing, isolating, and inspecting the information that a
model sees while it works.

The simplest bad mental model is that the context window is a buffer. It is not.
It is a scarce working-memory budget. Every token you append competes with every
other token for attention, cost, latency, and output room.

> **Context engineering is the practice of giving the model the right
> information, in the right form, at the right time, while keeping the wrong
> information out.**

This session is organized around a live demo: six progressively better ways to
answer questions over the same support-ticket corpus, all measured with the same
eval.

---

## Today's Session

| Part | Core Question |
|------|---------------|
| **Context Window as Working Memory** | Why is more context often worse? |
| **Seeing What Your Agent Sees** | How do we inspect and measure context? |
| **Practitioner's Toolkit** | Which context-management patterns work? |
| **Live Demo: Demos 01-06** | How do pruning, summaries, quarantine, and RLM compare? |
| **Recursive Language Models** | What changes when the model manages context through code? |
| **Wrap-Up** | What should move into the harness? |

---

## The Context Window as Working Memory

The guest lecture gave the conceptual landscape: prompt engineering, context
engineering, and harness engineering form a hierarchy.

- **Prompt engineering:** what words condition one call? Examples include the
  system prompt, few-shot examples, and output format.
- **Context engineering:** what information should the agent carry over time?
  Examples include retrieved docs, tool results, memory, and summaries.
- **Harness engineering:** what scaffold makes the loop reliable? Examples
  include tools, checkpoints, traces, budgets, and evals.

Context engineering sits between the prompt and the harness. It is not just
writing better instructions. It is deciding what evidence and state the model is
allowed to see.

### The working memory analogy

Human working memory is limited. You can remember a childhood phone number for
decades, but you cannot juggle twenty new phone numbers at once. LLMs have much
larger raw capacity, but they show a similar pattern: effective attention
degrades as context grows, irrelevant information interferes with useful
information, and recent or salient tokens can dominate.

A long context window is useful, but it does not remove the need for selection.
A focused 5,000-token prompt can beat a noisy 100,000-token prompt.

### Context as a budget

Every token appended is a spending decision. The context window must pay for:

- the system prompt,
- conversation history,
- retrieved documents,
- tool inputs and outputs,
- memory records,
- intermediate reasoning artifacts,
- and the final answer itself.

A useful practitioner question is:

> **Does this piece of content earn its spot in the context?**

If the answer is no, shorten it, move it to external storage, summarize it, or
keep it out entirely.

---

## Seeing What Your Agent Sees

Many agent systems fail because nobody inspects the actual prompt assembled at
runtime. That is like optimizing a database without looking at query plans.
Before optimizing context, measure it.

A minimal inspection utility can already change how you think:

```python
def inspect_context(messages: list[dict[str, str]]) -> None:
    enc = tiktoken.encoding_for_model("gpt-4o")
    total = 0

    for message in messages:
        content = message.get("content") or ""
        token_count = len(enc.encode(content))
        total += token_count
        role = message["role"].ljust(12)
        bar = "=" * (token_count // 50)
        print(f"{role} {token_count:>5} tok |{bar}")

    print(f"{'TOTAL':>12} {total:>5} tok")
```

This reveals which message types dominate, how fast the context grows per agent
step, whether tool results are proportional to their usefulness, and when the
system is approaching the limit.

The growth curve matters. A naive agent often grows superlinearly because each
step adds tool output, reasoning, and conversation history. Better harnesses keep
the root context bounded through summaries, quarantine, or external memory.

---

## Three Patterns That Actually Work

### Pattern 1: Rolling summarization

After each step, compress accumulated state into a compact running summary. This
is useful for long sequential tasks where each step builds on previous findings.
The cost is an extra model call and the risk that the summary drops details that
later become important.

Summarization is best when approximate state is acceptable. It is dangerous when
the downstream task needs exact counts, exact quotes, or exact provenance.

### Pattern 2: Context quarantining

Isolate each subtask in its own context thread. The subtask can read detailed
raw evidence, but the coordinator only receives distilled output.

This works well for parallel, independent subtasks. It reduces contamination and
keeps the coordinator context small. The cost is more calls and a harder handoff:
the subagent can miss information if the task boundary is wrong.

### Pattern 3: Memory offloading

Write intermediate findings to an external store and retrieve only what is
needed later. This is useful when total state exceeds any single context window
or must persist across sessions.

The tradeoff is retrieval quality. Memory offloading turns the context problem
back into a retrieval problem, which means Units 6 and 7 apply again.

### Choosing the pattern

| Question | Likely Pattern |
|----------|----------------|
| Are subtasks independent? | Quarantine |
| Does total state exceed the context limit? | Memory offloading |
| Is this a sequential task with approximate state? | Rolling summary |
| Do we need exact answers from raw records? | Quarantine or direct retrieval, not lossy summary |

Most production systems combine all three: subagents for independent work,
running summaries for sequential depth, and external memory for persistence.

---

## Live Demo: Six Progressive Approaches

The demo uses one corpus and one eval throughout.

**Corpus:** 400 synthetic Axiom Software support tickets, each in a
pipe-delimited format like:

```text
id=T001 || plan=Enterprise || date=2024-11-01 || msg=...
```

**Task:** answer 15 classification questions about the corpus. The questions
require semantic reasoning; keyword matching is not enough.

**Metric:** score from 0-15 plus peak context tokens. A better design should
usually improve both quality and context size, but the demo shows where that
intuition breaks.

### Demo 01: broken baseline

The baseline sends every ticket verbatim for every question.

```python
for question in QUESTIONS:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Tickets:\n\n{CORPUS}\n\n"
                f"Question: {question['question']}"
            ),
        },
    ]
```

What is wrong:

- the system prompt is padded and repetitive,
- all tickets are dumped into every call,
- each question sees the same irrelevant noise,
- the model pays the full context cost 15 times.

**Result:** score `5.94/15`, peak about `18,644` tokens.

The failure modes are context distraction and lost-in-the-middle behavior. The
model has enough raw information, but the signal is buried.

### Demo 02: system prompt compression

The first optimization rewrites the system prompt as a single imperative line:

```python
SYSTEM = "Count matching tickets. Reply with a single integer."
```

This removes about 97% of the system-prompt tokens. It is useful, but the full
corpus still dominates.

**Result:** score improves to `6.92/15`, peak remains about `18,300` tokens.

The lesson: prompt trimming helps, especially when repeated many times, but it
cannot fix a bad evidence strategy.

### Demo 03: ticket field pruning

The next optimization removes fields that cannot help answer the current
question. For many questions, `id` is useless; `plan` and `msg` matter; `date`
matters only for date-range questions.

```python
def _prune_line(line: str) -> str:
    plan = re.search(r"plan=(\w+)", line)
    msg = re.search(r"msg=(.+)$", line)

    if plan is None or msg is None:
        raise ValueError(f"Could not parse ticket line: {line}")

    return f"plan={plan.group(1)} || msg={msg.group(1)}"
```

The key question for each field is: could the model answer this question without
it? If yes, drop it.

**Result:** corpus drops about 30%, and score improves to `8.78/15`.

Less noise improves reasoning because the model sees more relevant density per
token.

### Demo 04: summarize then answer

A tempting approach is to summarize the corpus once, then answer each question
from the summary.

```python
summary, _ = build_summary(client)

for question in QUESTIONS:
    raw = answer_from_summary(client, summary, question["question"])
```

The answer calls become tiny, around 500 tokens each. But the summary is lossy.
A 300-word summary can say that roughly 20% of tickets are auth-related; it
cannot reliably preserve exact counts across 15 questions.

**Result:** peak answer context falls sharply, but score drops to `4.38/15`.

The lesson: do not summarize data that you need to count exactly.

### Demo 05: question quarantine

Each question is answered in its own isolated sub-context using small batches of
tickets. The coordinator never sees ticket data; it only receives integers.

```python
for question in QUESTIONS:
    raw = quarantined_answer(client, question)
    main_messages.append({"content": f"[{question['id']}] = {raw}"})
```

The coordinator context is bounded by the number of answers, not the size of the
corpus. Sub-contexts spike briefly and are discarded.

**Result:** coordinator context about `108` tokens, sub-context peak around
`837` tokens, score `8.80/15`.

This is the first approach that improves quality and reduces root context at the
same time.

### Demo 06: Recursive Language Model

The final demo gives the model a Python REPL and lets it manage context through
code. The root model sees only the query and REPL input/output. The corpus is a
Python variable that can be inspected programmatically. Sub-LLM calls process
chunks in isolated contexts.

The model can discover strategies such as:

1. **Peek:** inspect the data shape before committing.
2. **Grep:** filter to relevant rows.
3. **Partition and map:** send chunks to sub-LLMs.
4. **Aggregate in Python:** count, sum, and filter deterministically.
5. **Iterate:** refine if uncertain.

**Result:** average root context about `1,733` tokens, score `9.15/15`, roughly
91% context savings versus the naive single-call approach.

The frontier idea is that context management can become model behavior rather
than only harness behavior. The harness supplies the REPL and sub-LLM primitive;
the model decides how to use them.

---

## Comparing the Approaches

| Demo | Technique | Peak Context | Score / 15 | Lesson |
|------|-----------|--------------|------------|--------|
| 01 | Baseline | 18,644 tokens | 5.94 | Raw context is not enough |
| 02 | Prompt compression | 18,300 tokens | 6.92 | Trim repeated prompt fat |
| 03 | Field pruning | 13,100 tokens | 8.78 | Remove fields that do not earn their spot |
| 04 | Summarize then answer | 555 answer tokens | 4.38 | Lossy compression breaks exact tasks |
| 05 | Quarantine | 108 coordinator tokens | 8.80 | Isolate raw evidence from coordination |
| 06 | RLM | 1,733 average root tokens | 9.15 | Let code and sub-contexts manage long data |

The major pattern is not just "shorter is better." Demo 04 is shorter and worse.
The goal is not minimum context; the goal is the right information at the right
boundary.

---

## Recursive Language Models

Recursive Language Models, following Zhang and Khattab's 2025 framing, treat the
root LLM as a controller over a computation environment. It can write code,
inspect data, dispatch narrower model calls, and aggregate results.

The root context stays small because it does not ingest the full corpus. It sees
only the query, the code it wrote, and compact REPL outputs. Large raw data lives
outside the prompt.

This matters because it separates three concerns:

- **storage:** data can live in files, variables, databases, or retrieval stores,
- **computation:** deterministic code can count, filter, and aggregate,
- **judgment:** LLM calls can handle semantic classification where code is weak.

The strongest version of this pattern uses the model for semantic decisions and
ordinary software for bookkeeping. That is often the right split.

---

## Takeaways

1. **Measure first.** You cannot manage context you do not inspect.
2. **Every append is a budget decision.** Raw tool output rarely deserves to be
   copied in full.
3. **Compression must match the task.** Summaries are useful for themes, risky
   for exact counts.
4. **Quarantine is a powerful default.** Independent subwork should often happen
   in isolated contexts.
5. **Context management belongs in the harness.** Mature systems make pruning,
   summarization, memory, and sub-contexts runtime primitives.
6. **The frontier is self-managed context.** REPL-style and RLM-style systems let
   models use tools to keep their own working memory small.
