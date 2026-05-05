# Unit 10: Multi-Agent Patterns

**Date:** Wednesday, April 1, 2026

Unit 9 gave one agent autonomy. Unit 10 asks what happens when the work becomes
long, stateful, and too broad for one clean thread. The tempting question is
"how do I build a team of agents?" That is usually the wrong starting point.

The more useful question is:

> **How do I decompose work while preserving the right context, right control
> flow, and right synchronization boundaries?**

Multi-agent design is a systems-design problem for long-horizon language-model
work. The hard part is rarely "more agents." It is deciding what should stay in
one thread, what should be delegated, what state should be shared, and how the
pieces should be evaluated.

---

## Today's Session

| Part | Core Question |
|------|---------------|
| **Reframing the Problem** | Why is "agent team" the wrong first abstraction? |
| **Patterns That Mostly Work** | Which decomposition patterns are reliable today? |
| **Patterns That Help but Add Inertia** | When does planning or recursion help? |
| **Synchronization Debate** | Shared context or isolated workers? |
| **Frontier Bets** | What are systems like Slate, Manus, Gastown, and Beads betting on? |
| **Decision Checklist** | When should we decompose at all? |

---

## Reframing the Problem

The superficial question is:

> How do I make a team of agents?

That phrasing pushes you toward agent identities, roles, chat protocols, and
fictional org charts. It can make the design more complicated before you know
what bottleneck you are solving.

The stronger framing is decomposition and coordination. Ask which part of the
work needs isolation, specialization, parallelism, memory, verification, or
human approval.

### What the current evidence supports

| More Confidence | Less Settled |
|-----------------|--------------|
| Long contexts degrade usefulness, not just cost | Many agents vs. one threaded agent |
| Reactive loops often beat rigid plans | Message-passing subagents as a universal primitive |
| Adaptive decomposition helps on complex tasks | Planner-implementer-reviewer as a durable default |
| Context management is central | Orchestration-heavy agent factories |
| Bounded externalized state helps | Sweeping claims about inevitable agent teams |

The left column has repeated empirical and practitioner support. The right
column is still mostly architecture opinion.

### Four core tensions

Most architectures are points in this tradeoff space:

| Tension | One Side | Other Side |
|---------|----------|------------|
| **Context** | shared context | isolated local contexts |
| **Control flow** | reactive step-by-step loops | explicit plans and schedules |
| **Memory** | keep everything in-thread | compress, externalize, retrieve |
| **Organization** | one generalist loop | many specialists or workers |

The bottlenecks in long-horizon systems are usually context management,
synchronization, control, and evaluation. The engineering task is to choose
boundaries that reduce those bottlenecks instead of multiplying them.

---

## Pattern 1: Router / Delegate

The simplest useful decomposition is often a router:

1. inspect the request,
2. classify the kind of work,
3. delegate to the right specialist, tool cluster, or workflow.

Routers are strong because they solve a narrow problem. They give specialization,
policy separation, permission separation, and simpler prompts downstream.

But routers are not magic. They are classifiers with consequences.

```python
route = classify(request)

if route.confidence < THRESHOLD:
    return fallback_agent(request)

agent = agents[route.label]
return agent.run(request)
```

Evaluate routing like any classifier: edge cases, abstention, a confusion matrix,
and regression tests whenever the router prompt changes.

Common failure modes:

- bad or overconfident routing,
- hidden behavior changes when router instructions change,
- no abstain or fallback path,
- routes that are labels rather than actual capability boundaries.

---

## Pattern 2: Fan-Out / Fan-In

Fan-out/fan-in is the next strong pattern:

1. split the task into mostly independent subproblems,
2. run them in parallel,
3. merge the results.

It is compelling when the decomposition is obvious: inspect many files,
summarize many documents, classify many records, or investigate several
hypotheses.

The split is often easy. The merge is harder.

Aggregators must reconcile duplicate findings, conflicting evidence, uneven
quality, and uncertainty that sounds fluent. Better merge behavior usually comes
from structured outputs, provenance, explicit uncertainty, and a distinction
between reducing evidence and judging conclusions.

Fan-out works best when the subproblems are independent and the merge operation
is simple or verifiable.

---

## Pattern 3: Verifier Loops

A very strong pattern is not "many agents talking." It is:

> **generate -> check against grounded constraints -> revise if needed**

Verifier loops work because the verifier can often be more reliable than the
generator. Good verifiers include:

- tests,
- schemas,
- type checks,
- execution traces,
- retrieval-backed evidence,
- deterministic policy checks.

A critic and a verifier are different. A critic is usually another LLM offering
comments. It can be useful, but it often shares the generator's blind spots. A
verifier is grounded in rules, execution, or evidence. It is narrower, easier to
trust, and easier to evaluate.

In practice, verifier loops deserve more confidence than free-form debate loops.

---

## Patterns That Help but Add Inertia

### Planner-executor

Planner-executor systems keep reappearing for good reasons. They make long tasks
legible, force explicit decomposition, reduce early stopping, and create
artifacts that humans can review.

They are strongest when task order matters, dependencies are visible, completion
criteria are explicit, and a plan must be reviewed by a human or another system.
Examples include research workflows, complex coding tasks, and operational
workflows with approvals.

The failure mode is rigidity. Upfront decomposition can be incomplete, the world
can change during execution, compressed status can lose state, and the system can
follow a stale plan for too long.

Planning helps, but too much explicit orchestration can destroy flexibility.

### Adaptive decomposition

ADaPT-style decomposition is a useful middle ground: do not decompose the whole
task up front. Try the current step first; if it is too hard, decompose that
piece; recurse only when needed.

Adaptive decomposition preserves the reactivity that makes ReAct-like systems
strong.

### Recursive decomposition and RLM

RLM-style systems push the idea further:

- keep the root context small,
- externalize large state,
- recurse into narrower subproblems,
- return compact results upward.

This combines isolation, search, and compression. It also introduces the risk of
over-decomposition. Recursive systems need maximum depth, maximum branching,
explicit stopping criteria, typed intermediate results, and a way to kill
unproductive branches.

Think of recursion as a search mechanism with context isolation, not a free
lunch.

---

## The Central Debate: Synchronization

The strongest disagreement in current agent discourse is not really "more agents
or fewer agents." It is:

> **How much context should be shared, and what information is safe to summarize,
> compress, or hand off across boundaries?**

### The shared-context view

Cognition's "Don't Build Multi-Agents" argument is an important corrective.
Actions carry implicit decisions. Isolated agents can act on different
assumptions. Parallelism is not automatically progress if it fragments the
decision boundary.

The lesson is not "never decompose." The lesson is that shared assumptions must
remain shared.

### The externalize-and-handoff view

The Manus and Amp-style view emphasizes that context windows cannot hold
everything. Files, artifacts, memory stores, and durable task records act as
external memory. Handoffs can beat naive compaction when they preserve exactly
what the next thread needs.

The alternative to "keep everything in context" is often externalized state, not
more agents.

### Compaction, handoff, and episode boundaries

| Mechanism | Strength | Main Risk |
|-----------|----------|-----------|
| **Compaction** | keeps one thread alive | unpredictable information loss |
| **Handoff** | starts a focused new thread | depends on transfer quality |
| **Episodes / returns** | gives clean completion boundaries | still needs useful summarization |

The open problem is whether the compression boundary matches the task structure.

---

## The Ralph Wiggum Loop

Geoffrey Huntley's "Ralph Wiggum" loop is the opposite of fancy agent-society
design: keep one monolithic agent loop running, give it one concrete task at a
time, and use repeated prompting plus backpressure to keep it on the rails.

```bash
while :; do
  cat PROMPT.md | claude-code
done
```

The useful ideas are simple:

- one item per loop,
- stable prompt scaffolding,
- externalized plan and spec files,
- strong backpressure from tests, builds, and type checks.

Ralph is interesting because it highlights an uncomfortable possibility: a dumb
but disciplined monolithic loop can beat a clever multi-agent system if the
clever system fragments context too aggressively.

---

## Frontier Bets

### Manager-worker systems

Manager-worker systems are more structured than fan-out/fan-in and more dynamic
than planner-executor. They support branching, retries, and scheduling, but they
also add manager bottlenecks, synchronization overhead, and stale summaries.

Promising, but not a default answer.

### Hierarchies and agent teams

Full hierarchies promise scale, parallelism, specialization, and durable division
of labor. They also intensify synchronization, stale state, conflicting
assumptions, latency, cost, and evaluation difficulty.

Treat them as speculative until the evaluation story is clear.

### Gastown and Beads

Steve Yegge's Gastown and Beads vision treats the developer as the foreman of an
AI dockyard. Tasks become durable work objects; agent identities persist;
orchestration becomes first-class.

The throughput promise is real if you are willing to run a small AI shipping
port. The question is whether that overhead outlasts the current generation of
agent limitations.

### What the frontier systems are betting on

| System / View | Main Bet |
|---------------|----------|
| **Cognition** | shared context and single-threaded reliability beat naive fragmentation |
| **Manus** | context engineering and external memory drive long-horizon success |
| **Amp** | focused handoff beats meandering compaction |
| **Slate** | synchronization boundaries and episodic returns matter more than agent count |
| **Gastown / Beads** | durable orchestration can industrialize coding workflows |

---

## Worked Example: REST to GraphQL Migration

Suppose the task is: rewrite 40 endpoint files, update tests, and keep CI
passing.

| Pattern | Verdict | Why |
|---------|---------|-----|
| Single loop | viable | files are independent; tests give backpressure |
| Router | not helpful | one kind of work, no specialization needed |
| Fan-out/fan-in | strong fit | 40 files are embarrassingly parallel |
| Verifier loop | strong fit | CI and type checks give grounded verification |
| Planner-executor | likely overkill | dependencies are simple |
| Manager-worker | maybe | useful if schema changes are cross-cutting |

A likely design is fan-out per file, verifier loop per worker, then a single
merge and integration pass. Add a manager only if cross-file schema coordination
becomes real.

---

## Practical Decision Checklist

Before adding decomposition or orchestration, ask:

1. What exact bottleneck am I solving: quality, latency, context, or control?
2. Can one threaded loop solve it with better context engineering first?
3. Are the subproblems truly independent?
4. What assumptions must stay shared?
5. What information can be safely compressed?
6. How will I evaluate each boundary separately?

### When not to decompose

Keep a single agent loop when:

- subtasks share mutable state,
- the split is not obvious or testable,
- pieces cannot be evaluated independently,
- you have not tried better context engineering yet,
- coordination cost exceeds the parallelism gain.

If you cannot write a clear interface between the parts, they are probably one
part.

---

## Takeaways

1. Multi-agent design is really decomposition and coordination design.
2. Routers, fan-out/fan-in, and verifier loops are the strongest practical
   patterns today.
3. Reactive loops are still hard to beat; adaptive decomposition is usually
   safer than rigid decomposition.
4. Synchronization quality matters more than agent count.
5. Externalized state is often better than free-form agent conversation.
6. Be cautious about claims that one architecture has already won.

---

## References and Further Reading

- [Random Labs, *Slate*](https://randomlabs.ai/blog/slate)
- [Walden Yan, *Don't Build Multi-Agents*](https://cognition.ai/blog/dont-build-multi-agents)
- [Manus, *Context Engineering for AI Agents*](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [Amp, *Handoff*](https://ampcode.com/news/handoff)
- [Alex Zhang, *Recursive Language Models*](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prasad et al., *ADaPT*](https://arxiv.org/abs/2311.05772)
- [Yao et al., *ReAct*](https://arxiv.org/pdf/2210.03629)
- [Hong, Troynikov, Huber, *Context Rot*](https://research.trychroma.com/context-rot)
- [Steve Yegge, *Welcome to Gas Town*](http://steve-yegge.medium.com/welcome-to-gas-town-4f25ee16dd04)
- [Steve Yegge, *Beads*](https://github.com/steveyegge/beads)
- [Geoffrey Huntley, *The Ralph Loop*](https://ghuntley.com/ralph/)
- [Anthropic, *Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI, *A Practical Guide to Building Agents*](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
