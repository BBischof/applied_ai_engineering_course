# Unit 9: Agentic Patterns

**Date:** Wednesday, March 25, 2026

Unit 5 gave the model tools, but the user was always in control of the loop.
This lecture adds one step: **agency**. The model now decides when to reason,
when to act, and
when to stop. That shift turns tool use into a runtime systems problem.

This lecture uses three compatible definitions of an agent:

1. **Classical AI perspective (Russell/Norvig):** an agent perceives an
   environment and acts on it.
2. **RL/control perspective:** an agent makes sequential decisions and updates
   behavior from feedback over time.
3. **Modern engineering perspective:** an agent executes a workflow on the
   user's behalf with bounded independence, tools, and guardrails.

For this course, we combine them:

> **An agent is an LLM-driven control policy that repeatedly observes state,**
> **chooses actions, and updates behavior from outcomes, under harness**
> **constraints, until a verifiable exit condition is met.**

The model is the policy engine. The harness defines authority and boundaries:
state assembly, tool permissions, verification steps, checkpointing, and
traceability.

### Three perspectives, concrete examples

| Perspective | Core idea | Concrete example |
|---|---|---|
| **Classical AI (perceive -> act)** | Sense environment, then take action | A robot vacuum reads bump/dirt sensors, then chooses `move`, `turn`, or `clean` based on what it perceives. |
| **RL/control (sequential decisions + feedback)** | Actions are judged by downstream outcomes | A game-playing agent chooses a move, observes score/reward shift, and updates future move preferences to improve long-run return. |
| **Modern agent engineering (workflow execution)** | Complete multi-step work with tools under constraints | A support agent reads a ticket, queries CRM and billing tools, drafts a resolution, asks for approval on refund > `$500`, then updates the case. |

These are the same loop at different abstraction levels:

1. observe current state,
2. choose an action,
3. observe outcome,
4. decide whether to continue or stop.

Unit 9 focuses on what changes when this loop is LLM-driven and tool-augmented:
the hardest problems move from "what should I say?" to "what should I do next,
with what evidence, and under what constraints?"

| Part | Core Question |
|------|---------------|
| **Recap and Transition** | What changed from Units 5, 7, and 8? |
| **From Tools to Agent Loops** | What makes ReAct an engineering pattern? |
| **Agent as State Machine** | How do we reason about autonomous control flow? |
| **Harness Engineering** | What runtime capabilities make agents work in practice? |
| **Subagents, Skills, and Search** | How do we compose capability without context collapse? |
| **Frontier Directions** | Where is agent design moving right now? |
| **Trajectory Evaluation** | How do we prove an agent is correct beyond final answers? |

---

## Recap: The Ladder of Agency (Units 5, 7, and 8)

### Unit 5: from text to executable action

Unit 5 is best read as the first half of an agency ladder:

1. **Text output:** model can describe what should happen.
2. **Structured output/function calling:** model can emit executable intent.
3. **Tool-use loop (ReAct pattern):** model can iterate through action and
   observation.
4. **MCP:** model can discover and invoke interoperable tools/resources.
5. **Subagents and skills:** model can delegate and reuse capability.

The key contribution from Unit 5 was a systems framing of tool use:

- analyze every approach via the same loop-check questions (how the call is
  expressed, who executes it, and what guarantees exist),
- climb the determinism ladder from free-form text to schema-constrained
  execution,
- and introduce scale primitives (MCP standardization, subagent delegation,
  skills for reusable workflows).

Execution accountability was part of that framing, but not the whole story.

Unit 5's meta-principle still applies here: start simple (prompts), add
structure when reliability demands it, add execution when the world demands
it, add standardization when scale demands it, add delegation when complexity
demands it, and add reusable expertise when consistency demands it.

### Determinism ladder x agency ladder

These ladders are best treated as two axes in a 2x2:

- **Determinism:** how constrained, testable, and auditable each step is.
- **Agency:** how much workflow control the system owns across steps.

|  | **Low determinism** | **High determinism** |
|---|---|---|
| **Low agency** | **Q1: Advisory explorers** | **Q2: Constrained copilots** |
| **High agency** | **Q3: Autonomous explorers** | **Q4: Operational agents** |

Every quadrant has legitimate value. The right choice depends on task risk,
cost of mistakes, and how formal your acceptance criteria are.
Placement is deployment-dependent: the same product can sit in different
quadrants depending on permissions, approvals, and runtime controls.

#### Q1 — Advisory explorers (low agency, low determinism)

System role: generate options, hypotheses, and draft analyses; a human drives
execution and final decisions.

Representative product examples:

- [**Notion AI**](https://www.notion.so/help/guides/everything-you-can-do-with-notion-ai) drafts and summarizes workspace content; teams edit and decide final output.
- [**Jasper**](https://www.jasper.ai/) generates marketing copy and campaign drafts for human review.
- [**Canva Magic Write / Magic Design**](https://www.canva.com/magic-design/) produces draft text and design concepts for creative iteration.
- [**Adobe Firefly**](https://www.adobe.com/products/firefly.html) generates visual variants used as starting concepts.
- [**ChatGPT (standard chat use)**](https://chatgpt.com/) helps with exploratory analysis and option generation without executing side effects.
- [**Perplexity (standard search mode)**](https://www.perplexity.ai/) returns source-backed exploratory answers over web content.

Best-fit profile:

- ambiguous or creative tasks,
- low execution risk,
- value comes from breadth, speed, and perspective diversity.

Example prompts:

- "Give me five positioning angles for this product launch, with pros/cons for each."
- "Draft three campaign concepts for IT buyers in regulated industries."
- "Scan this topic and propose the top unresolved questions we should investigate."
- "Generate three narrative structures for this quarterly business review."

#### Q2 — Constrained copilots (low agency, high determinism)

System role: produce structured, policy-constrained outputs that a human
reviews and executes.

Representative product examples:

- [**GitHub Copilot (editor/chat flows)**](https://github.com/features/copilot) suggests code and tests; developers choose what to apply.
- [**Claude Code**](https://docs.anthropic.com/en/docs/claude-code) / [**Cursor**](https://cursor.com/) in approval-first mode propose edits/commands; developer approves execution.
- [**Hex Notebook Agent**](https://hex.tech/blog/introducing-notebook-agent/) generates SQL/Python notebook steps; analysts validate and run.
- [**Dropzone AI (analyst-in-the-loop mode)**](https://www.dropzone.ai/product) drafts security investigations; analysts decide containment actions.
- [**Nuance DAX Copilot**](https://www.microsoft.com/en-us/healthcare/solutions/nuance-dax-copilot) drafts clinical notes; clinicians review before sign-off.
- [**Zendesk AI Copilot**](https://www.zendesk.com/ai/copilot/) suggests support replies and actions; agents approve final responses.
- [**Rossum Copilot**](https://rossum.ai/blog/rossum-aurora-1-5-and-copilot/) extracts document fields; operators verify low-confidence values.
- [**UiPath Document Understanding + Action Center**](https://docs.uipath.com/autopilot/other/latest/user-guide/designing-autopilot-automations) classifies/extracts documents and routes approval tasks.

Best-fit user stories:

- "I want AI speed, but a licensed expert must sign off before action."
- "I need structured output in our schema so downstream systems do not break."
- "I need an audit trail showing what the system suggested and what humans approved."

#### Q3 — Autonomous explorers (high agency, low determinism)

System role: run multi-step exploration loops independently, then return
ranked findings for review before any sensitive action.

Representative product examples:

- [**ChatGPT Deep Research**](https://openai.com/index/introducing-deep-research) runs multi-step web research and returns cited reports.
- [**Perplexity Deep Research**](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) autonomously browses sources and compiles synthesis outputs.
- [**OpenClaw**](https://open-claw.org/) orchestrates multi-step assistant tasks; exact placement depends on granted permissions and controls.
- [**You.com ARI (Advanced Research and Insights)**](https://you.com/ari) performs iterative research passes and returns compiled findings.
- [**Elicit**](https://elicit.com/) finds and summarizes research literature with extracted evidence fields.
- **Internal "research scout" agents** built on enterprise search stacks
  (Glean/Elastic/custom) to surface evidence and open questions.

Best-fit user stories:

- "I need a first-pass landscape report before my team does deep validation."
- "I need broad evidence gathering and synthesis, then human judgment on decisions."
- "I can tolerate uncertainty in intermediate reasoning, but I still need source
  trails and review checkpoints."

Boundary note:

If OpenClaw (or similar orchestration systems) is granted write capabilities
across high-impact tools, its risk profile changes quickly. In that deployment,
it should be governed with Q4-style controls.

#### Q4 — Operational agents (high agency, high determinism)

System role: execute end-to-end workflows with explicit controls, invariants,
and verifiable completion criteria.

Workflow specification in this quadrant is explicit and machine-checkable:
runbooks, state-machine definitions, policy rules, approval thresholds, and
clear exit criteria.

Representative product examples:

- [**Intercom Fin**](https://www.intercom.com/fin/capabilities) resolves support conversations directly and escalates when rules/thresholds require.
- [**Sierra AI agents**](https://sierra.ai/product/meet-your-ai-agent) execute customer-service actions in connected systems under defined guardrails.
- [**ServiceNow AI Agents**](https://www.servicenow.com/products/ai-agents.html) run enterprise workflows with orchestration, approvals, and governance controls.
- [**UiPath Agent Builder**](https://www.uipath.com/product/agent-builder) coordinates agents, RPA, and humans for operational workflows with approval routing.
- [**Moveworks agentic automation**](https://www.moveworks.com/platform/ai-agent) executes IT/HR service operations through integrated enterprise tools.
- **Runbook-based incident automation stacks** ([PagerDuty + Rundeck](https://www.pagerduty.com/integrations/rundeck-runbook-automation/)) execute bounded remediation steps from predefined runbooks.
- [**Dropzone AI (autonomous triage mode)**](https://www.dropzone.ai/) investigates and classifies security alerts at scale under SOC policy controls.
- [**Claude Code**](https://docs.anthropic.com/en/docs/claude-code) / full-stack coding agents in autonomous mode plan, edit, test, and iterate to explicit exit criteria.

Best-fit user stories:

- "I run a high-volume workflow and need lower time-to-resolution without
  sacrificing controls."
- "I can define success criteria, allowed actions, and escalation boundaries
  ahead of time."
- "I need reliability and resumability when failures happen mid-workflow."

Summary rule:

Every quadrant is useful. The right product depends on your tolerance for
autonomy, your need for guarantees, and your governance context. Start where
your current controls and data quality are strong, then increase agency only as
fast as your evaluation and safety discipline can keep up.

### Unit 7: retrieval as an agent subsystem

Unit 7 maps directly onto the agency × determinism matrix.

Retrieval can operate in every quadrant:

- **Q1 (advisory explorers):** one-shot search and quick synthesis to widen
  options and frame a problem.
- **Q2 (constrained copilots):** retrieval with fixed schemas, citation rules,
  and approval requirements before downstream action.
- **Q3 (autonomous explorers):** iterative query rewriting, branching, and
  multi-hop discovery with human review at milestones.
- **Q4 (operational agents):** retrieval as a governed production subsystem that
  drives decisions in workflows with strict policy and audit constraints.

Unit 7's Collector -> Ranker -> Server architecture is what lets retrieval move
across these quadrants safely:

- **Collector** determines what is findable and what metadata can be filtered.
- **Ranker** determines which evidence is surfaced and in what order.
- **Server** determines what context enters the loop and what constraints are
  enforced before action.

This is why multi-hop retrieval, query rewriting, and context poisoning are not
only retrieval concerns. In higher-agency systems, retrieval quality directly
controls action quality.

### Unit 8: context engineering is control, not prompt polish

Unit 8 supplied the control logic that makes higher-agency systems viable.

The key hierarchy is:

1. prompt engineering (wording of one call),
2. context engineering (what enters the window across a run),
3. harness engineering (the runtime that enforces both).

In the agency × determinism matrix, context engineering is one of the main
levers for moving right (toward higher determinism) without reducing agency.
It does this by making state selection explicit instead of implicit.

Core Unit 8 mechanisms that connect directly to Unit 9:

- **budgeting:** allocate tokens deliberately across instructions, evidence,
  memory, tool outputs, and response space;
- **ordering:** control sequence effects so critical evidence is visible when
  the model decides actions;
- **compaction:** summarize or offload stale detail while preserving decision-
  relevant state;
- **boundary control:** quarantine noisy or untrusted context before it can
  steer action policy;
- **assembly discipline:** build context from validated components, not raw
  transcript accumulation.

This is also where Unit 8's failure modes become operational constraints in
Unit 9: poisoning, distraction, confusion, and fighting-the-weights are not
just model-quality issues; they are context-construction failures in the
harness.

Context-engineering priorities by quadrant:

| Quadrant | Typical context challenge | What to focus on |
|---|---|---|
| **Q1 Advisory explorers** | Over-broad context leads to generic, shallow output | Optimize for breadth with light structure: diverse sources, fast summarization, and explicit "unknowns" sections rather than forced certainty |
| **Q2 Constrained copilots** | Correct output format but wrong evidence grounding | Optimize for precision and traceability: strict schemas, citation requirements, scoped retrieval filters, and reviewer-visible evidence links |
| **Q3 Autonomous explorers** | Context drift across long multi-step exploration loops | Optimize for iterative state hygiene: milestone summaries, context resets between branches, tool-output compaction, and checkpointed research logs |
| **Q4 Operational agents** | Small context errors become real side-effect errors | Optimize for safety-critical context assembly: policy-first context blocks, immutable constraints, pre-action validation context, and explicit rollback/recovery state |

A practical sequence when moving up/right in the matrix:

1. define required context components for each step (not one global prompt),
2. enforce admission rules for what is allowed into decision context,
3. add per-step evidence checks before action,
4. persist minimal durable state for resumability,
5. measure failures by context failure mode, then tighten assembly rules.

### What Unit 9 adds to the ladder

Unit 9 is where the system levels up in four concrete ways:

1. **From single calls to runs:** one request can require many tool decisions.
2. **From outputs to trajectories:** we evaluate the full action sequence, not
   only the final answer text.
3. **From prompts to harnesses:** reliability comes from runtime structure,
   not only instruction wording.
4. **From stateless turns to durable state:** plans, checkpoints, and traces
   survive failures and handoffs.

With the full Unit 5-8 context in place, this level-up is broader:

1. **From capability demos to workflow ownership:** the system is now
   responsible for progressing work across multiple steps under constraints.
2. **From one-dimensional quality to system quality:** evaluation covers answer
   quality, trajectory validity, environment outcomes, and policy compliance.
3. **From ad hoc context to engineered context:** budgeting, ordering,
   compaction, and boundary control are explicit runtime responsibilities.
4. **From isolated tool calls to operating models:** products and internal
   systems can be intentionally placed in the agency × determinism matrix, then
   governed according to their risk profile.

The shift is from "the model can use tools" to "the organization can deploy
the right level of agency for each job, with evidence and controls."

In P cubed terms for this unit:

- **Promise:** agents complete real jobs end-to-end, not just
  answer questions. They should plan, use tools, recover from errors, and
  finish the task correctly.
- **Proof:** we can demonstrate correct behavior with trajectory evidence:
  right tool chosen, right arguments, right order, valid environment outcome,
  and policy compliance.
- **Production:** we ship harness controls that keep the proof true in the
  wild: checkpointing, rate limits, human approval gates, retries, and
  resumable execution.

---

## Part 1: From Tools to Agent Loops

### The transition

Tool use is "call this function now." Agentic behavior is "decide which
function to call, whether more calls are needed, and when to terminate."

Matrix view of this transition:

- **Q1:** model suggests options; human drives execution.
- **Q2:** model produces constrained recommendations; human still owns control.
- **Q3:** model runs multi-step exploration; humans review outcomes.
- **Q4:** model executes bounded operations with explicit controls and proofs.

The canonical pattern is ReAct:

```
Thought -> Action -> Observation -> Thought -> ... -> Final answer
```

This is no longer just prompting style. It is runtime control flow.

### ReAct-like does not automatically mean "agent"

A lot of production systems use ReAct-style loops but are still copilots, not
agents. The loop structure can look similar while authority is very different.

Examples that often use ReAct-like patterns but are usually **not agents**:

1. **Customer support copilot in a ticketing UI**
   - The model searches docs/CRM, drafts a response, and suggests next steps.
   - A human agent decides what to send and what actions to execute.
   - Why not agentic: workflow control and final execution authority stay with
     the human.

2. **Security triage assistant (SOC copilot)**
   - The model gathers logs, correlates indicators, and proposes containment.
   - A human analyst decides whether to isolate hosts or block traffic.
   - Why not agentic: model supports diagnosis but does not own action policy.

3. **IDE coding assistant with tool use**
   - The model reads files, proposes patches, and suggests test commands.
   - The developer decides what to run, apply, and commit.
   - Why not agentic: the model is advisory; it does not run the full loop to
     completion on its own.

Examples that are typically **agents**:

1. **Refund resolution agent**
   - Reads ticket + order history, checks policy, decides next tool calls,
     drafts user message, updates case state, and escalates only when needed.
   - Agent trait: it owns multi-step workflow execution under policy gates.

2. **Cloud incident remediation agent**
   - Ingests alerts, runs diagnostics, executes approved runbook steps,
     verifies service recovery, then posts incident summary and next actions.
   - Agent trait: it closes the observe -> act -> verify loop over environment
     state, not only over text.

3. **Long-running coding agent**
   - Plans milestones, edits code, runs tests, fixes failures, checkpoints
     progress, and resumes across sessions until exit criteria are met.
   - Agent trait: durable autonomy with explicit stop conditions and evidence.

### Copilot vs agent: diagnostic table

| System | Who decides next action? | Who executes side effects? | Who decides "done"? | Agent? |
|---|---|---|---|---|
| **Support copilot** | Human support rep | Human rep / backend systems | Human rep | **No** (copilot) |
| **SOC triage copilot** | Human analyst | Human analyst / security tooling | Human analyst | **No** (copilot) |
| **IDE coding assistant** | Developer | Developer + local tooling | Developer | **No** (copilot) |
| **Refund resolution agent** | Agent policy + guardrails | Agent via tools (with approval gates) | Agent exit criteria + policy checks | **Yes** |
| **Incident remediation agent** | Agent policy + runbook constraints | Agent runbooks + infra APIs | Agent verifies recovery + closes task | **Yes** |
| **Long-running coding agent** | Agent planner/executor loop | Agent tools (edit/test/shell) | Agent milestones + test pass criteria | **Yes** |

Diagnostic heuristic: if humans still own next action and completion criteria
at each step, the system is a copilot, not an agent.

### Why ReAct is still the baseline

ReAct remains the practical starting point because it gives a durable loop:

1. reason about current state,
2. choose an action,
3. observe real outcome,
4. update plan,
5. repeat until stop condition.

The innovation in 2025-2026 is not replacing this loop. It is surrounding
this loop with better harnesses.

This is the key level-up idea: ReAct gives the loop skeleton; harness design
determines where that loop can safely sit in the matrix.

### A compact mental model

```
tool-using model != agent
agent = model + loop + state + environment + harness controls
```

In practice, the dividing line is not "does it call tools?" It is "who owns
the workflow policy and completion criteria under constraints?"

### How systems level up in agency (practical path)

Two common upgrade paths:

1. **Support path (Q1 -> Q2 -> Q4)**
   - Q1: draft replies and summarize tickets.
   - Q2: enforce policy templates/citation requirements before agent send.
   - Q4: allow autonomous case resolution for bounded intents with escalation.

2. **Coding path (Q2 -> Q4)**
   - Q2: suggest edits/tests, developer approves each execution step.
   - Q4: agent plans milestones, edits, runs tests, and loops to exit criteria.

A reliable pattern is to move one axis at a time:

- first move right (more determinism),
- then move up (more agency),
- then widen scope only after evaluation and guardrails hold.

---

## Part 2: Agent as a State Machine

### Why this framing matters

Without a state-machine view, agent behavior can feel opaque. With this view,
it becomes inspectable, testable, and debuggable.

Matrix implication:

- Q1/Q2 systems can remain mostly request-response.
- Q3/Q4 systems need explicit state transitions to prevent drift and loops.

An agent execution can be represented as:

- **State:** what the agent currently knows (goal, plan, artifacts, memory).
- **Action policy:** what action is legal and useful in this state.
- **Transition:** how state updates after tool result or environment change.
- **Termination:** success, failure, timeout, budget limit, or human stop.

### Practical state nodes

Most production loops repeatedly enter these nodes:

1. **Plan** (decompose task)
2. **Act** (tool call or subagent delegation)
3. **Observe** (read tool output/environment delta)
4. **Verify** (check invariant, test, rubric)
5. **Commit checkpoint** (write progress/state artifact)
6. **Continue or stop**

As agency increases, these nodes should be explicit artifacts, not implied
prompt behavior.

### Checkpointing as a core requirement

Long-running agents fail in realistic ways: process crash, stale context,
provider outage, tool timeout, transient permission issues. Checkpoints let
you resume without replaying the entire trajectory.

Useful checkpoint artifacts:

- current plan and completed milestones,
- latest verified outputs,
- open risks and next actions,
- tool call history pointer,
- pending approval requests.

Level-up note:

- Q2 often needs lightweight checkpoints (review queues, draft versions).
- Q4 needs durable resumability across failures, sessions, and handoffs.

---

## Part 3: Harness Engineering (Model + Harness)

### Definition

A **harness** is the non-parametric runtime layer around the model:
instructions, tools, skill loading, memory policies, orchestration, safety
gates, and trace infrastructure.

### Three-layer model

```
Model        -> reasoning and generation
Harness      -> control, state, safety, orchestration
Environment  -> files, APIs, browser, databases, OS
```

### Core harness responsibilities

| Responsibility | What it does | Typical failure without it |
|---|---|---|
| **Tool/runtime policy** | Restricts and validates actions | Hallucinated tools, unsafe calls |
| **State management** | Stores recoverable artifacts | Progress loss on restart |
| **Context management** | Compacts and loads only needed context | Context bloat and drift |
| **Delegation** | Uses subagents with scoped permissions | Overloaded single-agent context |
| **Verification** | Runs tests/invariants before advancing | "Looks done" but is wrong |
| **Observability** | Captures traces and outcomes | No root-cause debugging path |

Matrix upgrade map for harness design:

- moving right (determinism): strengthen policy, schemas, verification, and
  evidence requirements;
- moving up (agency): strengthen state, orchestration, recovery, and stop logic.

### The practical shift since early agents

The frontier has moved from "better prompting for tool calls" to
"better harness design for reliable execution over longer horizons."

---

## Part 4: Harness Primitives That Matter in Practice

### 1) Durable workspace and externalized state

Long-horizon agents stay coherent by writing state outside the prompt:
plans, progress logs, intermediate outputs, tests, and environment snapshots.
This is the bridge from short chat loops to multi-hour execution.

Practical level-up example:

- Q2 coding copilot: local draft context is enough.
- Q4 coding agent: persistent plans, checkpoints, and test artifacts are required.

### 2) Code execution as a universal action space

Schema-bound function calls are excellent for narrow interfaces. But many
tasks require composition. Code execution gives the agent a general substrate:
compose tools, inspect outputs, write utilities, and verify behavior.

Matrix note:

- code execution increases potential agency;
- policy and sandbox controls must increase in parallel.

### 3) Context engineering inside the loop

Context is finite. Harnesses need explicit policies for:

- offloading large tool outputs,
- compaction and summarization boundaries,
- progressive loading of skills/instructions,
- preserving only what is needed for the next decision.

Practical level-up example:

- Q1/Q2: context mostly supports drafting quality.
- Q3/Q4: context directly changes action selection and side-effect risk.

### 4) Verification loops

Modern agents should not trust self-reported completion. They should run
verification steps:

- execute tests,
- check invariants,
- inspect external effects,
- retry or escalate when checks fail.

Verification is the mechanism that turns autonomous behavior from Q3-style
exploration into Q4-style operational reliability.

### 5) Human-in-the-loop gates

Sensitive actions (writes, spending, production changes, legal/privacy
boundaries) should pause behind approval gates. The harness should make these
boundaries explicit and auditable.

Approval gates are not anti-agent; they are often the bridge from Q2 to Q4.

### 6) Rate limits, retries, and stop conditions

An autonomous loop without budget controls is a runaway process. Production
harnesses need explicit cost/time caps, retry policies, and loop breakers.

This is where "more agentic" becomes "more governable," not just "more
autonomous."

---

## Part 5: Subagents, Skills, and Agentic Search

### Tool vs skill vs subagent

This distinction is now foundational:

- **Tool:** low-level action with narrow contract (e.g., run query, read file).
- **Skill:** reusable workflow package with specialized instructions.
- **Subagent:** delegated autonomous worker with scoped context/tools.

Matrix interpretation:

- tools mostly increase capability breadth,
- skills increase repeatability/determinism,
- subagents increase agency while preserving context boundaries.

### Why subagents matter beyond parallelism

Subagents solve two hard problems at once:

1. **Specialization:** different instructions and toolsets for distinct tasks.
2. **Context isolation:** parent receives compressed outputs, not every trace.

This directly mitigates context poisoning and uncontrolled context growth.

### Unit 7 carryover: multi-hop retrieval inside agent systems

A practical pattern:

1. Orchestrator defines retrieval objective.
2. Retrieval subagent executes multi-hop search loop.
3. Synthesis subagent composes evidence and confidence.
4. Orchestrator verifies citations and unresolved gaps.

This turns "advanced retrieval techniques" into callable capability inside the
agent runtime.

In matrix terms, this pattern is a common path from Q2 retrieval assistants to
Q4 retrieval-backed operators: add iterative planning, evidence checks, and
explicit escalation when evidence is weak.

---

## Part 6: Frontier Directions (2025-2026)

The current research frontier can be organized into three directions.

### A) Code-native runtimes

The move from flat tool schemas to code-native execution environments
increases expressiveness. Instead of selecting one tool call at a time, the
agent can program against runtime state and compose actions fluidly.

Core idea: sometimes the right abstraction is not "pick a function," but
"write and execute a small program."

Matrix effect: raises potential agency quickly; requires stronger controls to
stay in high-determinism regimes.

### B) Recursive context access (RLM-style ideas)

The problem is often not missing tools, but poor context selection.
Recursive scaffolds treat prompt/context as navigable environment:
inspect slices, recurse on subproblems, and compose results symbolically.

Core idea: avoid indiscriminate context stuffing; navigate context
programmatically.

Matrix effect: improves determinism at higher agency by reducing context drift
and evidence omission.

### C) Automatic harness synthesis

Instead of hand-writing every verifier and legality checker, synthesis
approaches search over harness programs using environment feedback.

Core idea: harness quality is an optimization target, not static scaffolding.

Matrix effect: can move systems rightward by learning better constraints,
validators, and orchestration policies.

### Comparison summary

| Direction | Main change | Core line |
|---|---|---|
| **Code-native runtime** | Action substrate | "Program the runtime, not just call tools." |
| **Recursive context access** | Context control | "Navigate context, don't dump context." |
| **Harness synthesis** | Harness construction | "Learn the wrapper, not only the policy." |

---

## Part 7: Proof = Trajectory Evaluation

### Why final-answer grading is insufficient

A final answer can look plausible while the trajectory is invalid:
wrong tool choice, wrong argument schema, policy violation, stale state use,
or unverified side effects.

For agents, proof requires trajectory-aware evidence.

### What to evaluate

At minimum, evaluate four layers:

1. **Answer correctness** (did it complete the task?)
2. **Trajectory correctness** (right tools, args, order, stop condition)
3. **Environment correctness** (did real world state match claims?)
4. **Policy compliance** (did it follow constraints while succeeding?)

Proof depth should scale with quadrant:

- **Q1:** output usefulness and source quality.
- **Q2:** schema compliance, reviewer agreement, policy adherence.
- **Q3:** trajectory quality and research/evidence stability across loops.
- **Q4:** all of the above plus environment-state correctness, rollback safety,
  and operational reliability under failure.

### Common failure modes

- hallucinated tool names or illegal arguments,
- infinite/self-reinforcing loops,
- premature completion claims,
- stale memory and wrong resume state,
- context exhaustion and dropped critical evidence,
- policy bypass via unsafe delegation.

### A practical eval harness shape

Each task in the eval set should include:

- task prompt and acceptance criteria,
- allowed/disallowed tools and policies,
- expected trajectory invariants,
- expected environment outcomes,
- grader for both trace and final state.

This is how Unit 3's Analyze -> Measure -> Improve loop becomes operational
for autonomous systems.

Level-up release rule:

promote a system to a higher-agency quadrant only after it meets the current
quadrant's proof criteria consistently.

---

## Exercise: Diagnose the Harness

Use one task and three execution traces:

1. **Naive loop:** plain ReAct, no checkpointing, no verifier.
2. **Harnessed loop:** adds subagent delegation and context isolation.
3. **Verifier-augmented loop:** adds action legality checks and retry policy.

For each trace, identify:

- model failure vs harness failure,
- missing invariant,
- where to place a checkpoint or approval gate,
- what trace signal would have caught the issue earlier.

---

## Suggested Board Flow (90 Minutes)

1. Unit 5 -> Unit 9 transition (autonomy)
2. ReAct as runtime loop
3. Agent as state machine
4. Model + harness definition
5. Practical harness primitives
6. Tool vs skill vs subagent
7. Agentic search and multi-hop retrieval inside agents
8. Frontier directions: runtime, context recursion, synthesis
9. Trajectory evaluation and production controls
10. Exercise debrief

---

## Key Takeaways

1. **Agents are runtime systems, not prompt templates.**
2. **ReAct is still the core loop, but harness design determines matrix
   placement and reliability.**
3. **Move right before moving up: strengthen determinism before increasing
   agency.**
4. **Context engineering, state design, and delegation are core control
   surfaces.**
5. **Trajectory evaluation is the proof surface for autonomous behavior, and
   proof depth must scale with agency.**
6. **Frontier work is harness-centric: better runtimes, better context control,
   and better harness construction.**

---

## Further Reading

### Required

- Shankar & Husain, Chapter 8 (Sections 8.1-8.2)
- Yao et al. (2022), ReAct

### Strongly recommended

- OpenAI, Building Agents
- Anthropic, Effective Harnesses for Long-Running Agents
- Anthropic, Effective Context Engineering for AI Agents
- OpenAI Codex docs: AGENTS.md, Skills, MCP
- LangChain, The Anatomy of an Agent Harness

### Frontier

- Wang et al. (2024), CodeAct
- Zhang et al. (2025/2026), Recursive Language Models
- Lou et al. (2026), AutoHarness
- Symbolica, Agentica overview and docs

### Evaluation and benchmarks

- WebArena
- OSWorld
- BrowseComp-Plus
- TRAJECT-Bench
- AgentDojo
- ST-WebAgentBench
