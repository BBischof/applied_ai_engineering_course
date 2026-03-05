# Unit 5: Tools and the Road to MCP

**Date:** Wednesday, February 18, 2026

Last week we treated the model's text input as the primary lever—prompt
engineering. This week we ask: what happens when text isn't enough?

The arc of this lecture follows a single thread: **language models want to
act in the world, and we've been inventing better ways to let them.** Each
milestone solves a real limitation of the previous one, and each introduces
patterns you'll use as an AI engineer. 

---

## Warm-up: where prompt engineering hits a wall

In Unit 4 you learned to write prompts that follow patterns, produce predictable output,
reason step-by-step, and self-evaluate. That's powerful—but consider three
requests a user might make:

1. "What's the current price of AAPL stock?"
2. "Book me a meeting with Priya at 2pm tomorrow."
3. "Run my test suite and fix any failures."

No amount of prompt crafting helps here. The model doesn't have live data,
can't take action which have effects, and can't execute code. It can only *produce
text*. So the natural next question is: **what if the model could ask
something else to do things?**

Or:
> Is there a way to make generated text actionable in systems.

That question—and the increasingly sophisticated answers to it—is the
subject of today's lecture.

---

## The tool-use loop (the key mental model)

Every milestone we cover is a variation on this loop:

```
1. User sends a request
2. Model decides: "I need to call tool X with arguments Y"
3. Something executes that call (your code, the provider, an MCP server...)
4. The result comes back to the model
5. Model continues (possibly with more tool calls)
```

At every milestone, we'll answer the same three questions—and each answer
becomes a row in the comparison matrix we're building together:

1. **How does the model express the call?** (step 2)
2. **Who or what executes it?** (step 3)
3. **What structural guarantees do you get?** (is output format
   deterministic, enforced, or aspirational?)

---

## Milestone 1: Research roots — "language in, action out" (pre-2021)

### Why it matters

Long before LLMs, NLP researchers were building systems that translate
natural language into executable commands: SQL queries, API calls, robot
instructions. This is the intellectual pre-cursor of everything we cover.

### What changed

- **Semantic parsing** framed the problem: map a user's sentence into a
  structured program (SQL, API call, script) that a runtime can execute.

  Example — a natural-language interface to a weather API:

  ```
  User: "What's the forecast for Chicago tomorrow?"
        ↓ parser
  GET /forecast?location=chicago&date=2026-02-19
        ↓ runtime executes HTTP request
  {"high": 28, "low": 15, "conditions": "snow"}
        ↓ response generator
  "Tomorrow in Chicago: high of 28°F, low of 15°F, snow expected."
  ```

  **What's inside the parser?** You'd write grammar rules or pattern
  templates by hand. A simplified version:

  ```python
  import re
  from datetime import date, timedelta

  PATTERNS = [
      # "forecast for {city} tomorrow"
      (r"forecast for (\w+) tomorrow",
       lambda m: {
           "endpoint": "/forecast",
           "params": {
               "location": m.group(1).lower(),
               "date": (date.today() + timedelta(days=1)).isoformat()
           }
       }),
      # "weather in {city}"
      (r"weather in (\w+)",
       lambda m: {
           "endpoint": "/current",
           "params": {"location": m.group(1).lower()}
       }),
  ]

  def parse(utterance: str) -> dict | None:
      for pattern, builder in PATTERNS:
          m = re.search(pattern, utterance, re.IGNORECASE)
          if m:
              return builder(m)
      return None  # no matching rule → system can't help

  # parse("What's the forecast for Chicago tomorrow?")
  # → {"endpoint": "/forecast",
  #    "params": {"location": "chicago", "date": "2026-02-19"}}
  ```

  This works perfectly for the patterns you wrote and fails silently
  on everything else. Want to handle "weather next Tuesday in
  Chicago"? Write another rule. Want to handle open-ended questions?
  You can't—this approach doesn't scale.

- **Task-oriented dialogue** systems (think Siri-era) ran a multi-turn
  pipeline: track dialogue state, query a database/API, generate a
  response. Modern tool use compresses this, but the architecture is
  the same.

  Example — a flight-booking dialogue system:

  ```
  User: "I need a flight to Denver next Friday."
        ↓ dialogue state tracker
  {intent: book_flight, dest: "DEN", date: "2026-02-27", origin: null}
        ↓ system realizes origin is missing
  Agent: "Where are you flying from?"
  User: "Newark."
        ↓ state updated: {origin: "EWR", ...}
        ↓ database query
  SELECT * FROM flights
    WHERE origin='EWR' AND dest='DEN' AND date='2026-02-27';
        ↓ results formatted into response
  Agent: "I found a United flight departing 8:15am for $247..."
  ```

  **What's inside the state tracker?** A set of slots to fill, rules
  for what's required, and logic for what to ask next:

  ```python
  from dataclasses import dataclass

  SLOT_SCHEMA: dict[str, dict] = {
      "origin": {"type": "airport_code", "required": True},
      "dest":   {"type": "airport_code", "required": True},
      "date":   {"type": "date",         "required": True},
  }

  PROMPTS: dict[str, str] = {
      "origin": "Where are you flying from?",
      "dest":   "Where are you flying to?",
      "date":   "What date do you want to fly?",
  }

  @dataclass
  class DialogueState:
      intent: str | None = None
      slots: dict[str, str | None] = None

      def __post_init__(self) -> None:
          if self.slots is None:
              self.slots = {k: None for k in SLOT_SCHEMA}

      def next_action(self) -> str:
          """Return a clarifying question or 'ready'."""
          for slot, meta in SLOT_SCHEMA.items():
              if meta["required"] and self.slots.get(slot) is None:
                  return PROMPTS[slot]
          return "ready"

  # After two turns:
  #   state.slots = {"origin": "EWR", "dest": "DEN",
  #                  "date": "2026-02-27"}
  #   state.next_action() → "ready" → system runs the SQL query
  ```

  Each component—intent classifier, slot filler, state tracker, query
  builder, response generator—is a separate module, often built and
  maintained by a different team. The "LLM tool use" story is about
  collapsing most of these boxes into a single model that handles
  language understanding, state tracking, and action selection—but the
  API call in the middle never goes away.

### Key insight

> A language model is a conditional translator: natural language in,
> structured command out. Execution happens elsewhere.

This idea never goes away—it's the through-line to MCP.

**Loop check — semantic parsing:**
1. *How does the model express the call?* A specialized parser maps NL to
   a structured command (SQL, API call). No LLM in the loop yet.
2. *Who executes?* An external runtime (database engine, API server).
3. *What guarantees?* Deterministic within the parser's grammar—if it
   parses at all, the output is well-formed. But coverage is narrow.

### Comparison matrix (start)

| Approach | Who decides? | Who executes? | Output format |
|----------|-------------|---------------|---------------|
| Semantic parsing | Specialized parser | External runtime | SQL / API call |

---

## Milestone 2: LLMs learn to reach for tools (2021–2022)

Once large pretrained models became strong instruction-followers, people
realized they could *prompt* them to emit tool calls. Several landmark
papers crystallized different patterns—and each one expanded the
**capability envelope** of what LLMs could do. Before this era, LLMs
were chatbots. After it, they were starting to become labor.

### 2a. WebGPT — "browse as a tool" (2021)

OpenAI trained a model to interact with a browser environment (search,
click, quote) and optimized it with human feedback.

**New capabilities unlocked:** factual question-answering with citations
over live web content. The model could now answer questions about things
that happened after its training cutoff.

**What the model outputs:** structured browser commands in a custom
text format—`Search`, `Click`, `Quote`, `Scroll`—interpreted by a
browser environment that returns rendered page text.

**How the model learned this:** this was *not* free from prompting.
OpenAI collected human demonstrations of browsing behavior, trained
via behavior cloning (imitation learning), then refined with RLHF
using a reward model trained on human preferences over answer quality.
The browsing behavior was taught, not emerged.

> **Read:** Nakano et al., *WebGPT: Browser-Assisted Question-Answering
> with Human Feedback*, 2021.
> [arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332)

### 2b. MRKL — the modular blueprint (2022)

MRKL (Modular Reasoning, Knowledge and Language) argued explicitly: combine
a general language model with specialized modules—calculator, database,
search engine—each accessible as a "tool."

**New capabilities unlocked:** accurate arithmetic, database queries,
real-time lookups—anything a specialized module can do. The LLM handles
language; the modules handle precision.

**What the model outputs:** a routing decision ("this question needs
the calculator module") plus a text input for that module. The MRKL
system's "neural router" decides which expert to call.

**How the model learned this:** MRKL proposed training a dedicated
router—a classifier that maps user inputs to the right expert module.
The individual modules (calculator, DB query engine) are pre-built;
the learning happens in the routing layer. This is *not* something you
get from few-shot prompting a base LM—you needed a trained dispatcher.

> **Read:** Karpas et al., *MRKL Systems*, 2022.
> [arxiv.org/abs/2205.00445](https://arxiv.org/abs/2205.00445)

### 2c. PAL — "write code, execute it" (2022)

Program-Aided Language Models showed that instead of forcing the model to
do arithmetic in text, you have it write Python, execute the code, and
use the output. This is the ancestor of every code-interpreter tool.

**New capabilities unlocked:** reliable math, symbolic reasoning, data
transformations—anything you can express in code. The model decomposes
the problem; the interpreter computes the answer.

**What the model outputs:** executable Python (not just pseudocode).
The key structural difference from ReAct is that the output *is* the
program—there's no `Action:` / `Observation:` wrapper. The entire
response is code:

```python
def solve():
    # Problem: tax on $47.50 at 8%
    price = 47.50
    tax_rate = 0.08
    total = price * (1 + tax_rate)
    return total
# Runtime executes solve() -> returns 51.30 -> model uses the result
```

**How the model learned this:** PAL is one of the rare wins that *was*
largely free from prompting. It uses few-shot prompting on code-trained
LMs (like Codex). The key insight: if the model was already trained on
lots of code, you can get it to generate executable solutions just by
showing a few input/output examples in code form. No fine-tuning needed.
This worked because the base model's code-training already encoded
"how to translate word problems into programs."

> **Read:** Gao et al., *PAL: Program-Aided Language Models*, 2022.
> [arxiv.org/abs/2211.10435](https://arxiv.org/abs/2211.10435)

### 2d. SayCan — selecting from available skills (2022)

SayCan combined a language model's semantic scoring with a robot's
feasibility constraints: the LM says what *should* happen, the
affordance model says what *can* happen. The intersection is what the
robot does.

**New capabilities unlocked:** grounded physical action. For the first
time, a language model's "understanding" of tasks translates into a
real robot doing things in the world—picking up objects, navigating
rooms, completing multi-step instructions.

**What the model outputs:** SayCan doesn't have the model emit action
text directly. Instead, the LM scores candidate actions from a
pre-defined skill menu (e.g. "pick up the sponge", "go to the counter")
and the system selects the highest-scoring *feasible* action. The output
is a probability distribution over a closed action set, not free text.

**How the model learned this:** the LM's scoring ability came from
pre-training (it "knows" that wiping a counter involves a sponge).
But the affordance functions—which say whether a skill is physically
possible right now—were trained separately via reinforcement learning
on the robot. Neither component alone could do the task; the
architecture is the contribution.

> **Read:** Ahn et al., *Do As I Can, Not As I Say: Grounding Language
> in Robotic Affordances*, 2022.
> [arxiv.org/abs/2204.01691](https://arxiv.org/abs/2204.01691)

### 2e. ReAct — reasoning + acting, interleaved (2022)

ReAct formalized the prompting pattern that dominates agent frameworks
today: interleave *reasoning traces* with *actions* and *observations*.

**New capabilities unlocked:** multi-step research tasks requiring
planning. The model can chain together searches, lookups, and
comparisons to answer complex questions like "Did the director of
Jaws also direct a movie about dinosaurs?"

**What the model outputs:** strictly alternating `Thought:` / `Action:`
/ `Observation:` blocks. The thoughts are free-text reasoning; the
actions are structured tool calls; the observations come from the
environment. This explicit structure is what makes the trace debuggable:

```
Thought: I need to find who directed Jaws.
Action: search("Jaws 1975 director")
Observation: Jaws was directed by Steven Spielberg.
Thought: Now I need to check if Spielberg directed a dinosaur movie.
Action: search("Steven Spielberg dinosaur movie")
Observation: Spielberg directed Jurassic Park (1993).
Thought: Yes — Spielberg directed both.
Action: finish("Yes, Steven Spielberg directed both Jaws and Jurassic Park.")
```

**How the model learned this:** ReAct started as a prompting technique—
few-shot examples on existing LLMs, no fine-tuning. But "works via
prompting" doesn't mean "works well out of the box." The paper
carefully designed the prompt format and showed that without the
interleaved reasoning traces, the model makes significantly more errors.
The format itself is the engineering contribution.

What happened next is just as important: **providers started training
on these traces.** Once you have thousands of examples of
Thought/Action/Observation chains that lead to correct answers, you
can fine-tune on them. And it turns out this helps a lot—models
trained on reasoning traces follow the format more reliably, decompose
problems more effectively, and recover from errors mid-chain better
than models that only see the pattern at inference time via few-shot.
This is why modern tool-using models (GPT-4, Claude, etc.) are so
much better at tool use than the base models the ReAct paper tested
on: they've been trained on exactly this kind of data.

> **Read:** Yao et al., *ReAct: Synergizing Reasoning and Acting in
> Language Models*, ICLR 2023.
> [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

### Aside: reasoning models and test-time compute

The ReAct trace pattern—think, act, observe, think again—has a deeper
descendant worth knowing about: **reasoning models** (o1, o3, Claude
with extended thinking, DeepSeek-R1, etc.).

The core idea is the same: let the model produce intermediate reasoning
steps before committing to a final answer. But reasoning models take
this further in two ways:

1. **Training on chains of thought.** Where ReAct showed the model a
   few examples and hoped it would generalize, reasoning models are
   trained on massive datasets of step-by-step reasoning traces—often
   generated by the model itself and filtered for correctness (a
   technique sometimes called STaR: Self-Taught Reasoner). The traces
   become part of the training data, not just the prompt.

2. **Test-time compute scaling.** Instead of doing one forward pass
   and hoping the first answer is right, these models spend more
   inference-time compute by generating longer chains, exploring
   multiple paths, or re-checking their work. The key insight is that
   you can trade compute at inference time for accuracy—more thinking
   tokens = better answers, up to a point.

**Why this matters for tool use:** reasoning models don't just think
harder in text—they also make better tool-use decisions. A model
that can reason through "I should search for X, but if that fails
I should try Y" before emitting the first `Action:` will make fewer
wasted tool calls. This is why the latest agent systems (Claude with
extended thinking, o3-based agents) show step-change improvements in
agentic benchmarks like SWE-bench. The reasoning *about* tools
improves alongside reasoning in general.

**Connection to the training theme:** notice the pattern. ReAct
introduced a format via prompting. Then providers trained on that
format and it got dramatically better. Then they scaled test-time
compute on top of it and it got better again. This is the recurring
story we'll see at every milestone: a prompting innovation shows
what's possible, then training turns it from fragile to robust.

### The capability envelope so far

It's worth pausing to see how much ground was covered in just two years:

| Paper | What LLMs couldn't do before | What they can do now |
|-------|------------------------------|---------------------|
| WebGPT | Answer questions about current events | Browse the web, cite sources |
| MRKL | Do precise math, query databases | Route to specialized modules |
| PAL | Solve multi-step math reliably | Write + execute code |
| SayCan | Act in the physical world | Select feasible robot actions |
| ReAct | Plan multi-step research tasks | Chain reasoning with tool calls |

LLMs went from "impressive chatbot" to "system that can browse, compute,
query, and act." But notice the training story: **most of these were not
free.** WebGPT needed RLHF on browsing demonstrations. MRKL needed a
trained router. SayCan needed RL-trained affordance functions. Only PAL
and ReAct worked primarily through prompting—and both of those leaned
heavily on capabilities the base model got from code pre-training and
instruction tuning. The lesson: when you see a new capability, ask
"what training made this possible?" The answer matters when you're
deciding whether you can replicate it with your own models.

**Loop check — prompted tool use (ReAct, MRKL):**
1. *How does the model express the call?* Free-form text following a
   convention you defined in the prompt (e.g. `Action: search(...)`).
2. *Who executes?* Your glue code—regex parsing, dispatching to APIs.
3. *What guarantees?* None. The model *might* follow your format, or it
   might drift, hallucinate tool names, or produce unparseable arguments.
   You're parsing on hope.

**Loop check — code generation (PAL):**
1. *How does the model express the call?* It writes executable code.
2. *Who executes?* A Python interpreter (or similar runtime).
3. *What guarantees?* Partial. The interpreter enforces syntax (won't run
   invalid Python), but semantics are unconstrained—the code can still
   be wrong, unsafe, or call nonexistent APIs.

### Comparison matrix update

| Approach | Who decides? | Who executes? | Output format |
|----------|-------------|---------------|---------------|
| Semantic parsing | Specialized parser | External runtime | SQL / API call |
| **Prompted tool use (ReAct, MRKL)** | **LLM via prompt** | **Your glue code** | **Free-form text / XML** |
| **Code generation (PAL)** | **LLM writes code** | **Python interpreter** | **Executable code** |

---

## Milestone 3: Training models to use tools (2023)

Milestone 2 showed that LLMs can use tools — but the training story were varied. 
WebGPT needed RLHF on browsing demos. MRKL needed a
trained router. SayCan needed RL-trained affordance functions. PAL and
ReAct got away with prompting, leaning on code pre-training and
instruction tuning already in the base model. What none of them did was
teach a model *general-purpose* tool use — when to call which tool, with
what arguments, across an open-ended set of APIs. That's this milestone.

### 3a. Toolformer — self-supervised tool learning

Toolformer showed a model can teach itself when to call tools with a
clever self-supervised procedure:

1. Start with plain training text (e.g. "The population of Toronto is
   2,794,356").
2. For each position in the text, ask: "would inserting a tool call here
   have helped the model predict the next tokens better?"
3. Try candidate calls — e.g. inserting `[Calculator(2794356)]` or
   `[QA("population of Toronto")]` — and keep the ones that actually
   reduce perplexity (recall: perplexity measures how "surprised" the
   model is by the next token — lower perplexity means the tool call
   made the continuation easier to predict).
4. Fine-tune the model on the augmented text (original text with
   successful tool calls spliced in).

**What the tool definitions look like:** each tool is defined by a
simple text-based API signature the model learns to emit inline. The
training data contains tokens like:

```
The population of Toronto is [QA("population of Toronto")] → 2,794,356
The square root of 256 is [Calculator(sqrt(256))] → 16.0
```

The model learns to emit these `[ToolName(args)]` tokens as part of
normal next-token prediction. The special tokens `[`, `]`, and `→`
delimit where calls start, end, and where results are injected.

**New capabilities unlocked:** the model decides *on its own* when a
tool call would help — no few-shot prompt needed. Only a handful of
demonstrations per API are required to bootstrap the procedure.

**How this differs from prompting:** with ReAct, the model follows a
format you showed it in the prompt. With Toolformer, the format is in
the weights — the model has seen thousands of examples of when a
calculator call helps and when a search call helps, and it generates
them naturally as part of its token stream.

**Important limitation:** the tool set is closed and pre-defined by the
researchers (5 tools: calculator, Q&A, search, translator, calendar).
The model learns *when and how* to call these specific APIs, but it
can't invent new tools or generalize to APIs it hasn't seen. Each new
tool requires re-running the self-supervised data generation and
fine-tuning pipeline. This is exactly the gap Gorilla targets next.

> **Read:** Schick et al., *Toolformer: Language Models Can Teach
> Themselves to Use Tools*, NeurIPS 2023.
> [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)

### 3b. Gorilla — training for massive API surfaces

Toolformer worked with a handful of simple tools (calculator, QA,
search, translator, calendar). Gorilla targeted the real-world scaling
problem: there are thousands of APIs, their signatures are complex,
and their docs change.

**What the tool definitions look like:** Gorilla trains on
(instruction, API call) pairs scraped from real API documentation.
The training data is structured as natural-language task descriptions
paired with fully-specified API invocations:

```
Instruction: "I want to classify images using a pre-trained model
              that works well on ImageNet"
API call:    hub.load('pytorch/vision', 'resnet152', pretrained=True)
```

The model is fine-tuned on thousands of these pairs across HuggingFace,
TorchHub, and TensorHub APIs. Critically, Gorilla also uses
**Retriever-Aware Training (RAT)**: during both training and inference,
relevant API documentation is retrieved and included in context. This
means the model learns to *ground its API calls in the retrieved docs*
rather than relying on memorized (and potentially stale) knowledge.

**New capabilities unlocked:** correct API calls across massive,
changing API surfaces. Gorilla also introduced **APIBench** for
evaluating whether a model's generated API call is functionally correct.

**Why this matters for what comes next:** Gorilla's approach —
fine-tuning on (task, API call) pairs + retrieval over API docs at
inference time — is directly ancestral to how modern providers handle
tool use. When you give GPT-4 or Claude a list of tool definitions
and they generate correct calls, the capability comes from training on
exactly this kind of data: tool schemas paired with correct invocations.

> **Read:** Patil et al., *Gorilla: Large Language Model Connected with
> Massive APIs*, NeurIPS 2024.
> [arxiv.org/abs/2305.15334](https://arxiv.org/abs/2305.15334)

**Loop check — trained tool use (Toolformer, Gorilla):**
1. *How does the model express the call?* A learned call format baked into
   the weights—the model was trained on successful tool-call examples.
2. *Who executes?* Still your glue code, but the calls are better-formed.
3. *What guarantees?* Statistical, not structural. Training makes correct
   format *likely* but nothing enforces it at decode time. You still
   need validation.

### Comparison matrix update

| Approach | Who decides? | Who executes? | Output format |
|----------|-------------|---------------|---------------|
| Semantic parsing | Specialized parser | External runtime | SQL / API call |
| Prompted tool use (ReAct) | LLM via prompt | Your glue code | Free-form text / XML |
| Code generation (PAL) | LLM writes code | Python interpreter | Executable code |
| **Trained tool use (Toolformer, Gorilla)** | **LLM (fine-tuned)** | **Your glue code** | **Learned call format** |

---

## Milestone 4: Tool use becomes a platform feature (2023–2024)

This is where tool use leaves the research lab and becomes something you
configure in an API call. Each step solves a reliability problem from the
one before it.

But before we dive in, a principle worth stating bluntly:

> **If you only need structured data, don't jump to tools.** A prompt
> that asks for JSON output + a Pydantic validator is simpler, cheaper,
> and faster than wiring up tool calling. Tools are for when the model
> needs to *do* something — fetch data, trigger a side effect, run code.
> If all you need is a structured extraction from text the model already
> has in context, structured outputs are the right answer. Don't over-
> engineer.

### 4a. The XML-tag era — "prompt-invented protocols"

Before providers shipped official tool-calling APIs, developers invented
conventions in prompts. You'd tell the model: "if you want to use a tool,
emit `[tool: name][input: ...]` and I'll execute it." Then you'd parse
the tags and feed back an observation.

**Example prompt convention (early Claude + LangChain):**

```
If you need a tool, respond EXACTLY in this format:
  Tool: weather_lookup
  Input: {"city": "New York"}
Then STOP and wait for the result.
```

Here's what a full round-trip looked like. The model would respond:

```
I need to check the weather.
Tool: weather_lookup
Input: {"city": "New York"}
```

And your runtime would parse and dispatch it:

```python
import re
import json
from typing import Any

TOOL_PATTERN = re.compile(
    r"Tool:\s*(\w+)\s*\nInput:\s*(\{.*\})",
    re.DOTALL,
)

def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    if name == "weather_lookup":
        return get_weather(**args)
    if name == "stock_price":
        return get_stock_price(**args)
    raise ValueError(f"Unknown tool: {name}")

def maybe_handle_tool_call(model_response: str) -> str | None:
    """Parse a tool call from model text. Returns result or None."""
    match = TOOL_PATTERN.search(model_response)
    if not match:
        return None
    tool_name = match.group(1)
    tool_args = json.loads(match.group(2))
    return dispatch_tool(tool_name, tool_args)

# Then you'd inject the result back into the conversation:
#   Observation: {"temp": 34, "conditions": "cloudy"}
# ...and let the model continue.
```

Notice everything that can go wrong: the model might not include the
`Tool:` prefix, might emit malformed JSON, might hallucinate a tool
name that doesn't exist in your `dispatch_tool` function, or might
put extra text between `Tool:` and `Input:` that breaks your regex.
Every failure mode is a string-parsing bug you have to handle.

**Pedagogical point:** this is not "true" tool use by itself—it's a
protocol *you* invented in prompts. It becomes real tool use only when
your runtime actually executes the call and returns results. And the
fragility of this parsing code is exactly why providers eventually
built structured tool-calling into their APIs.

### Aside: the road toward agents

Notice what the XML-tag pattern is *almost* doing: the model observes a
situation, decides to act, receives a result, and decides what to do
next. That's the skeleton of an **agent** — a system that runs a loop
of perceive → decide → act → observe → repeat.

But in this era, none of the pieces are truly agentic yet:

- **No loop:** the developer's code handles the control flow. The model
  doesn't decide "should I call another tool or am I done?" — your
  `while` loop does.
- **No planning:** the model responds to one turn at a time. There's
  no multi-step plan it's executing against.
- **No autonomy:** every tool call requires your parsing code to work;
  every result requires you to inject it back manually.

What's happening here is that the *interface* for agents is being
invented — the idea that models can emit structured actions and receive
observations — even though the *systems* around them aren't autonomous
yet. We'll cover full agentic patterns in Unit 9, but keep this in
mind: tools are a prerequisite for agents, and the XML-tag era is where
practitioners first wired up the perceive/act/observe cycle by hand.

### 4b. OpenAI Plugins — tools described via OpenAPI (Mar 2023)

OpenAI's ChatGPT Plugins were a major public step: tools defined by a
web API, described with a manifest (`ai-plugin.json`) and an OpenAPI
schema, which the model could discover and call.

**How it actually worked under the hood:** the implementation was simpler
than it sounded. OpenAI took each plugin's `ai-plugin.json` manifest and
OpenAPI spec and **injected them into the system prompt** as structured
text. The model received a list of available plugins, their descriptions,
and their API schemas as part of its context — essentially a very fancy
version of the XML-tag prompting we just saw, but with the provider
managing the prompt injection and response parsing instead of you.

The model was fine-tuned to understand these injected schemas and emit
correct API calls when appropriate. No public details on the training
data were released, but the architecture was essentially: stuff the
OpenAPI spec into the system message, fine-tune the model to read specs
and emit calls, and handle dispatch server-side. Community reverse-
engineering of leaked system prompts confirmed this — the plugin
instructions appeared as plain text in the system message.

This also introduced the first major **security concerns** with tool
use: researchers showed that malicious plugins could inject instructions
via their API responses (prompt injection through tool output), and
that the model could be tricked into chaining plugins to exfiltrate
user data. These issues foreshadowed the security considerations that
MCP's spec would later address explicitly.

This normalized the idea that:
- tools are **externally described** (schema, not prompt hacking),
- models can **select and call them**,
- tool descriptions matter for both **safety and correctness**,
- and that tool-use capability comes from fine-tuning on schema/call
  pairs — not just prompting.

> **Read:** [OpenAI Plugins blog post](https://openai.com/index/chatgpt-plugins)
> (plugins have since been deprecated in favor of GPT Actions/function calling)

### 4c. Function Calling — "model outputs arguments I can execute" (Jun 2023)

This is arguably the single biggest inflection point in practical tool
use. Everything before this — XML tags, regex parsing, prompt-invented
protocols — required you to *hope* the model would emit the right format
and then write brittle glue code to parse it. Function calling changed
the contract entirely: **the API itself becomes the structured channel.**

Instead of the model producing free text that you parse, you declare
your tools as JSON Schema definitions in the API request, and the model
returns structured arguments in a dedicated field — not embedded in
prose, not wrapped in XML, but as a first-class part of the API
response. Your code receives clean, typed arguments it can dispatch
directly. No regex. No hope.

This mattered so much because it turned tool use from a research
technique into a **production-grade engineering primitive.** Before
function calling, every tool-using application was one malformed model
response away from a crash. After function calling, tool invocation
became as predictable as calling a REST endpoint. It's the moment tool
use became something you could put in front of real users.

**Pattern — declaring a function and getting structured output:**

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=tools,
)
# response.choices[0].message.tool_calls contains:
#   function.name = "get_weather"
#   function.arguments = '{"city": "New York", "units": "fahrenheit"}'
```

Your code then calls `get_weather(...)` for real, sends the result back
as a `tool` message, and the model completes with the final answer.

> **Read:** [OpenAI Function Calling guide](https://platform.openai.com/docs/guides/function-calling)

### 4d. JSON Mode — "valid JSON" without schema guarantees (Nov 2023)

JSON mode was announced at OpenAI's DevDay as a reliability layer: set
`response_format: {"type": "json_object"}` and the model is constrained
to produce valid JSON. This sounds like it solves everything, but in
practice it was a source of enormous frustration.

**What it actually guaranteed:** the output would be valid JSON — i.e.,
`json.loads()` wouldn't crash. That's it. It said nothing about which
fields would appear, what types values would have, or whether the
structure matched your schema at all.

**What went wrong in practice:** developers immediately discovered that
JSON mode would:
- omit required fields unpredictably,
- invent field names not in the requested schema,
- return objects when you asked for arrays (and vice versa),
- occasionally produce incomplete JSON (missing closing brackets) when
  the output hit the token limit.

This led to a cottage industry of retry-and-validate wrappers — the
canonical pattern was: call the API, try `json.loads()`, check with
Pydantic, retry on failure. Libraries like Instructor and Marvin were
built essentially to paper over this gap.

**The implementation mystery:** OpenAI never disclosed exactly how JSON
mode worked internally. The community debated whether it was constrained
decoding (masking invalid tokens at generation time), post-hoc rejection
sampling (generate, check, retry server-side), or just strong
fine-tuning that made valid JSON very likely. The truth probably involved
some combination — the output was *almost always* valid JSON, but not
100%, which argued against pure constrained decoding. Whatever the
mechanism, it clearly wasn't enforcing schema at the grammar level.

**When this helps:** you just need parseable JSON and can validate the
structure yourself (or your schema is simple enough that the model
rarely deviates).

**When it's not enough:** you need guaranteed field names, types, or
enums. JSON mode won't enforce those, and your Pydantic validators
will be doing real work.

### 4e. Structured Outputs — "exactly match this JSON Schema" (Aug 2024)

OpenAI introduced Structured Outputs to ensure model responses conform
*exactly* to a developer-supplied JSON Schema. This uses constrained
decoding—the model literally cannot produce tokens that would violate
the schema.

### Aside: how constrained decoding works

Recall from Unit 2 that an LLM generates text one token at a time. At
each step, the model produces a probability distribution over its entire
vocabulary — every possible next token gets a score. Normally, you
sample from this distribution (or take the argmax) and move on.

Constrained decoding intervenes at this step. Before sampling, the
system computes a **token mask**: which tokens are valid given the
current position in the schema? Every invalid token gets its probability
set to zero. The model can only choose from tokens that keep the output
schema-conformant.

Concretely, if your schema says the next field must be `"city"` and
the model has just emitted `{"ci`, the mask allows `ty` and disallows
everything else (no `rcle`, no `nema`, no `5.0`). If the schema says
a field is an enum of `["celsius", "fahrenheit"]`, the mask at that
position only allows tokens that continue one of those two strings.

```
Normal decoding:     P(token) → sample → next token
Constrained:         P(token) → mask invalid tokens → sample → next token
                                  ↑
                     Computed from JSON Schema + position in output
```

OpenAI's implementation (called LLGuidance) converts JSON Schemas into
context-free grammars, then uses an optimized lexer/parser to compute
masks in ~50 microseconds per token — fast enough that it doesn't
meaningfully slow down generation.

**Why this matters:** constrained decoding makes schema conformance a
property of the *generation process*, not a post-hoc validation check.
The output can't be wrong in the way JSON mode outputs could be wrong.
It's the difference between "the model usually gets it right" and "the
model structurally cannot get it wrong."

**The determinism ladder:**

```
Free text          → no guarantees, you parse and hope
XML-tag prompting  → no guarantees, you regex and hope harder
JSON mode          → valid JSON guaranteed, schema not enforced
Structured outputs → schema-conformant, enforced at decode time
```

> **Read:** [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
> and [guide](https://platform.openai.com/docs/guides/structured-outputs)

### 4f. Anthropic tool use — `tool_use` / `tool_result` blocks

Anthropic's tool use follows the same conceptual loop as OpenAI's
function calling — declare tools, model emits structured calls, you
execute and return results — but the wire format is different in a way
worth understanding.

In OpenAI's API, tool calls live in a dedicated field on the response
message (`message.tool_calls`), separate from text content. In
Anthropic's Messages API, tool calls are **content blocks** — they
appear inline in the same `content` array as text blocks. A single
assistant turn can interleave text and tool calls:

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Let me check the weather for you."},
    {"type": "tool_use", "id": "call_01", "name": "get_weather",
     "input": {"city": "New York"}}
  ]
}
```

You execute the tool and return the result as a `tool_result` block in
the next user turn:

```json
{
  "role": "user",
  "content": [
    {"type": "tool_result", "tool_use_id": "call_01",
     "content": "72°F, partly cloudy"}
  ]
}
```

**Why the difference matters in practice:** Anthropic's content-block
model makes it natural for the model to "narrate" its tool use — it can
say "Let me look that up" and emit the tool call in the same response.
OpenAI's separation is cleaner for parsing but means text and tool calls
are in different fields. Neither is strictly better; you just need to
know which wire format you're working with because your dispatch code
will look different.

**What's the same:** both providers give you JSON Schema-based tool
definitions, structured arguments, and a typed response channel. The
core engineering win from function calling — no more regex parsing —
applies to both.

> **Read:** [Anthropic Tool Use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)

**Loop check — function calling / structured outputs:**
1. *How does the model express the call?* Via a provider API—the model
   emits structured JSON arguments guided by a schema you supply.
2. *Who executes?* Your code. You receive the structured arguments and
   dispatch the actual function call.
3. *What guarantees?* This is where determinism actually arrives.
   Function calling gives you valid JSON with schema guidance. Structured
   outputs (strict mode) use constrained decoding—the model *cannot*
   produce tokens that violate the schema. Format conformance goes from
   "usually" to "always."

### Comparison matrix update

| Approach | Who decides? | Who executes? | Output format | Format guarantees |
|----------|-------------|---------------|---------------|------------------|
| Semantic parsing | Specialized parser | External runtime | SQL / API call | Deterministic (within grammar) |
| Prompted tool use (ReAct) | LLM via prompt | Your glue code | Free-form text | None (hope-based) |
| Code generation (PAL) | LLM writes code | Interpreter | Executable code | Syntax only |
| Trained tool use | LLM (fine-tuned) | Your glue code | Learned format | Statistical |
| **Function calling** | **LLM via API** | **Your code** | **JSON (schema-guided)** | **Valid JSON** |
| **Structured outputs** | **LLM via API** | **Your code** | **JSON (schema-enforced)** | **Schema-deterministic** |

---

## Milestone 5: MCP — a protocol for tool interoperability (Nov 2024)

### The problem MCP solves

Function calling solved: *"How do I get the model to emit a well-structured
call in my app?"*

But it created a new scaling problem: if you have 5 tools and 1 model,
function calling is fine. If you have 50 tools across GitHub, Slack,
Postgres, Stripe, Jira—and you want them available to Claude Desktop,
Cursor, your custom agent, and a teammate's notebook—you're writing N x M
bespoke connectors.

MCP solves: *"How do we stop rewriting connectors for every model/app/tool
combination?"*

### What MCP is

Anthropic announced and open-sourced the **Model Context Protocol** on
November 25, 2024. It's inspired by the Language Server Protocol (LSP)
that made every editor speak to every language toolchain.

**Architecture:**

```
Host (Claude Desktop, Cursor, your app)
  └── Client (speaks MCP)
        └── Server (exposes tools, resources, prompts via JSON-RPC 2.0)
              └── Your system (DB, API, filesystem, etc.)
```

An MCP server exposes three primitives:
- **Tools:** executable functions the model can call
- **Resources:** structured data the model can read
- **Prompts:** templated instructions for common workflows

Clients discover what's available via `tools/list`, `resources/list`, etc.

### When to use MCP vs. plain function calling

**Prefer function calling when:**
- You have a small, stable set of internal functions
- You want maximal control inside one backend
- Tools don't need to be reused across multiple hosts/apps

**Prefer MCP when:**
- Tools live across many systems (GitHub + Slack + DB + payments + ...)
- You want a standardized server that multiple clients can connect to
- You need tool discovery (`tools/list`) and centralized policy

Two properties of MCP are especially underappreciated:

**Distribution.** An MCP server is a standalone process. It can run on a
different machine, in a container, behind a firewall — anywhere. This
means the team that owns a system (say, the payments team) can build and
ship an MCP server for their service, and every agent host in the org
can connect to it. Nobody writes bespoke integration code. The tool
author distributes the server; consumers just point their MCP client at
it. This is the same win that APIs gave web services, but for
agent-to-tool connectivity: you publish once, everyone connects.

**Data modeling via resources.** Function calling only gives you tools —
callable functions. MCP adds **resources**: structured, read-only data
that the model can browse and pull into context. A database MCP server
doesn't just expose a `query` tool; it can expose tables as resources
that clients can list, preview, and selectively load. A docs MCP server
can expose articles as resources with URIs, so the agent reads the
specific doc it needs instead of searching blindly.

This matters because it separates two concerns that function calling
lumps together:
- **Tools** answer "what can I do?" (actions with side effects)
- **Resources** answer "what can I see?" (structured data to reason over)

An agent that can browse resources before deciding which tool to call
makes better decisions — it has context. This is why MCP servers for
databases, documentation systems, and knowledge bases are among the
most popular: they give models structured access to data, not just
action endpoints.

### What MCP looks like in practice

A minimal MCP server in Python (using the `mcp` SDK):

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("weather-server")

@app.tool()
async def get_weather(city: str) -> list[TextContent]:
    """Get current weather for a city."""
    # call your real weather API here
    data = await fetch_weather(city)
    return [TextContent(
        type="text",
        text=f"{city}: {data['temp']}F, {data['conditions']}"
    )]
```

Any MCP-compatible host (Claude Desktop, Cursor, a custom agent) can now
discover and call `get_weather` without knowing anything about its
implementation.

### MCP vs. other approaches (common confusions)

**MCP vs. Plugins/Apps:** OpenAI's plugins are a distribution + UX concept
tied to one platform. MCP is a vendor-agnostic protocol.

**MCP vs. RAG:** RAG retrieves text. MCP provides retrieval (resources)
*plus* actions (tools) with structured I/O and authorization.

### Practical tips for MCP (hard-earned)

- **Treat MCP servers like real APIs.** Auth, audit logs, rate limits,
  versioning. An MCP server is a network service; give it the same
  operational rigor you'd give a REST API.
- **Avoid tool explosion.** Don't dump 50 tools into every conversation.
  Models degrade when the tool list gets long — they pick wrong tools,
  hallucinate arguments, or get confused by similar-sounding options.
  Curate the tool set per task, or use discovery (`tools/list`) to let
  the agent search for what it needs.
- **Use approval policies for dangerous calls.** MCP supports the concept
  of human-in-the-loop approval for sensitive operations (payments,
  deletions, production writes). Build this in from the start.

**Governance note:** Anthropic donated MCP to the Linux Foundation's
Agentic AI Foundation in December 2025, signaling that MCP is intended
as a vendor-neutral standard, not an Anthropic product.

**Loop check — MCP:**
1. *How does the model express the call?* Same as function calling (the
   host translates MCP tool schemas into the model's native tool format).
2. *Who executes?* An MCP server—a separate process that owns the tool
   implementation. Your host just proxies.
3. *What guarantees?* Inherits function-calling determinism for the call
   format. MCP adds a new guarantee: *discoverability*—clients enumerate
   available tools and their schemas at runtime via `tools/list`.

> **Read:**
> - [Anthropic MCP announcement](https://www.anthropic.com/news/model-context-protocol)
> - [MCP specification (latest)](https://modelcontextprotocol.io/specification/latest)
> - [OpenAI MCP tool guide](https://platform.openai.com/docs/guides/tools)

### Comparison matrix update

| Approach | Who decides? | Who executes? | Output format | Format guarantees | Reusability |
|----------|-------------|---------------|---------------|------------------|-------------|
| Semantic parsing | Specialized parser | External runtime | SQL / API call | Deterministic (within grammar) | Low |
| Prompted tool use | LLM via prompt | Your glue code | Free-form text | None | Low |
| Code generation | LLM writes code | Interpreter | Executable code | Syntax only | Low |
| Trained tool use | LLM (fine-tuned) | Your glue code | Learned format | Statistical | Low |
| Function calling | LLM via API | Your code | JSON (schema) | Valid JSON | Medium |
| Structured outputs | LLM via API | Your code | JSON (enforced) | Schema-deterministic | Medium |
| **MCP** | **LLM via API** | **MCP server** | **JSON-RPC** | **Schema-deterministic + discoverable** | **High** |

---

## Milestone 6: Agent-to-Agent communication — A2A (2025)

### Why agents talking to agents matters

MCP connects an agent to *tools and data*. But there are tasks where
a tool isn't the right abstraction — where what you actually need is
*another agent.*

**Why tools aren't always enough.** A tool is a function: deterministic
input, deterministic output, no judgment required. `get_weather("NYC")`
always means the same thing. But consider:

- "Negotiate a contract with this vendor" — that requires back-and-forth,
  judgment, domain expertise, and a model of the counterparty.
- "Review this PR for security issues" — that requires reading code,
  reasoning about attack surfaces, and explaining findings. It's not a
  function call; it's a *task* you'd hand to a specialist.
- "Process this insurance claim" — that might touch underwriting rules,
  fraud detection, customer communication, and payment — each owned by
  a different team with their own agent.

In all these cases, the capability you're reaching for doesn't fit the
tool interface (call with args, get result). You need something that
can receive an open-ended task, work on it autonomously, and report
back — possibly over minutes or hours. You need another agent.

**Why a protocol matters.** Without a standard, every agent-to-agent
integration is bespoke. Your travel agent talks to your payments agent
via a custom internal API. If you want to integrate a third-party
compliance agent, you write another custom integration. This is the
same N×M problem MCP solved for tools — but at the agent level.

### A2A: the protocol

Google announced the Agent2Agent Protocol (A2A) in April 2025 as an open
protocol for agent interoperability — explicitly complementing MCP:

- **MCP:** "Let my agent use external tools/data sources."
- **A2A:** "Let my agent delegate to / collaborate with other agents."

The key difference from a tool call: A2A tasks are **opaque and
long-lived.** The requesting agent doesn't know (or care) how the
remote agent accomplishes the task. It might involve multiple tool calls,
human approvals, sub-delegations — that's the remote agent's problem.
The protocol handles capability discovery, task lifecycle management,
and secure communication. Built on HTTP, SSE, and JSON-RPC.

**Loop check — A2A:**
1. *How does the model express the call?* The agent discovers a peer's
   capabilities and sends a structured task request via the protocol.
2. *Who executes?* A remote agent—potentially on a different team, org,
   or vendor's infrastructure.
3. *What guarantees?* The protocol layer is deterministic (structured
   messages, capability negotiation). But the remote agent's *behavior*
   is opaque—you get protocol guarantees, not outcome guarantees.
   Trust boundaries matter here.

> **Read:**
> - [A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability)
> - [A2A specification](https://google.github.io/A2A/specification/)

---

## Milestone 7: Subagents — delegation inside a system (2025)

An important distinction up front: **subagents are an architecture
pattern, not a protocol.** MCP and A2A are wire protocols — specs you
implement to interoperate across systems. Subagents are a design
decision *within* your system: how you decompose work internally.
There's no subagent spec to implement; it's about how you structure
your agent's runtime.

### The problem subagents solve

Long-running agent sessions hit three chronic issues:
1. **Context overflow:** exploration fills the window with noise.
2. **Permission sprawl:** a research subtask shouldn't have write access
   to production databases.
3. **Specialization:** different subtasks benefit from different system
   prompts, tools, and even models.

### How subagents work

A subagent is a child agent with:
- its own context window (isolated from the parent),
- a custom system prompt,
- a restricted or expanded tool set,
- optionally a different (cheaper/faster) model.

The parent launches a subagent like calling a tool—it sends a task and
gets back a result.

**Example: Cursor's subagent pattern:**

```
Parent agent (main task: "refactor the auth module")
  └── Subagent 1 (explore: "find all files importing auth.py")
        → returns file list
  └── Subagent 2 (shell: "run the test suite")
        → returns test results
  └── Parent continues with both results in context
```

### Subagents in practice

- **Claude Code:** documents subagents with their own context windows,
  tool restrictions, permission modes, and model choices.
- **OpenAI Agents SDK:** frames delegation as "handoffs" — tools like
  `transfer_to_refund_agent` that route tasks to specialized agents.

### How the model learns to delegate

This is a great place to apply our recurring question: is delegation
something the model learns through training, or is it "free" from
the architecture?

The answer: **delegation is presented to the model as tool use.** The
framework turns subagents into tools — just like function calling.

In OpenAI's Agents SDK, a handoff is literally a tool definition:

```python
billing_agent = Agent(name="Billing agent", ...)
refund_agent = Agent(name="Refund agent", ...)

triage_agent = Agent(
    name="Triage agent",
    handoffs=[billing_agent, refund_agent],
)
# The model sees tools named "transfer_to_billing_agent"
# and "transfer_to_refund_agent" with descriptions
# explaining when each should be used.
```

In Claude Code, the Task tool serves the same role — the model sees
a tool it can call with a task description, and the runtime spawns
a subagent to handle it.

So the model doesn't need special "delegation training." It's reusing
the same tool-calling capability it already has from the function-calling
era (Milestone 4). The decision of *when* to delegate is guided by the
tool descriptions — each subagent has a description explaining what it's
good at, and the model matches tasks to descriptions the same way it
matches queries to any other tool.

What *is* trained (or at least heavily prompted) is the **orchestration
behavior**: knowing when a task is complex enough to decompose, how to
split work into parallelizable subtasks, and how to synthesize results
from multiple subagents. In Claude Code, the orchestrator (typically a
stronger model like Opus) is prompted with explicit decomposition
patterns. In the Agents SDK, you can customize handoff descriptions and
input schemas to steer routing decisions. But the underlying mechanism
is always the same: subagents are tools, and delegation is a tool call.

**Loop check — subagents:**
1. *How does the model express the call?* The parent agent launches a
   subagent the same way it'd call a tool — task description in, result
   out. The framework converts subagents into tool definitions.
2. *Who executes?* A child agent runtime with its own context, tools,
   and (optionally) a different model.
3. *What guarantees?* Isolation is the new guarantee: the subagent's
   context, permissions, and failures are scoped. The parent gets a
   result without its own context being polluted by exploration noise.

> **Read:**
> - [Claude Code subagents](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
> - [OpenAI Agents SDK handoffs](https://openai.github.io/openai-agents-python/handoffs/)

---

## Milestone 8: Skills — reusable expertise bundles (Oct 2025)

### Why we need another mechanism

At this point you might reasonably ask: "Don't we already have enough
layers? Why can't I solve this with what I already have?" Let's be
explicit about what each existing mechanism *can't* do.

**"Just write a better prompt."** You could put your workflow
instructions in the system prompt — "When writing a QBR, always
include an executive summary, then quarterly metrics, then..." But
system prompts are per-conversation. If you have 15 different
workflows your org needs done consistently, you'd need to stuff all of
them into every conversation's system prompt (wasting context), or
manually pick the right one each time (error-prone), or build a
routing system to select prompts (reinventing skills from scratch).

**"Use a tool."** A tool is a function call — `generate_qbr(data)`.
But a QBR isn't one function call. It's a multi-step workflow:
gather data from three sources, compute metrics, draft an executive
summary, format as a slide deck, check against brand guidelines. A
tool does one thing; a workflow is a *recipe* that coordinates many
things. You could build a monolithic tool that does all of this
internally, but then the model can't inspect, modify, or learn from
the process — it's a black box.

**"Use an MCP server."** Same problem. MCP gives you tools and
resources — building blocks. It doesn't give you the *instructions
for how to combine them.* An MCP server can expose `query_metrics`,
`fetch_template`, `export_slides` — but who says to call them in
that order, what to check between steps, and what the output should
look like? That's the skill's job.

**"Use a subagent."** Closer — you could create a subagent with a
system prompt that says "you are a QBR specialist." But a subagent's
instructions live in code (the system prompt you pass when spawning
it). They're not discoverable, not versioned as files, not shareable
across teams, and not loadable on demand. If you have 15 specialized
workflows, you'd need 15 hardcoded subagent configurations. Skills
externalize those instructions into files that the agent discovers
and loads dynamically.

### What skills actually are

A skill fills the gap none of the above mechanisms cover: **reusable,
discoverable, file-based workflow expertise that the agent loads on
demand.**

| | Tool | Skill |
|---|------|-------|
| What it is | A callable function | A package of instructions + resources |
| Analogy | A screwdriver | A carpentry manual |
| Example | `get_weather(city)` | "How to write a quarterly business review" |
| Invocation | Model calls it | Model *loads and follows* it |
| Shareable | Via MCP / API | Via filesystem — copy the directory |
| Discoverable | `tools/list` | Agent scans skill directories automatically |

A skill says: "here is the procedure, the checklist, the templates,
and the scripts. Follow these steps and use whatever tools you need
along the way." It's the difference between giving someone a hammer
and giving them a building plan that says when to use the hammer.

### How skills work (Anthropic's model)

A skill is a directory with a required `SKILL.md` file plus optional
scripts, templates, and resources. Claude discovers skills dynamically
and loads them when relevant—using a three-tier system:

1. **Metadata** (always loaded): YAML frontmatter (~100 tokens)
2. **Instructions** (loaded when triggered): the core procedure
3. **Resources** (loaded as needed): scripts, templates, examples

**Why this matters:** skills reduce repeated prompt-stuffing and encode
organizational best practices. They're portable across Claude.ai, Claude
Code, and the API.

**Loop check — skills:**
1. *How does the model express the call?* It doesn't "call" a skill—it
   *loads and follows* one. The agent discovers relevant skills and
   incorporates their instructions into its own behavior.
2. *Who executes?* The agent itself, guided by the skill's instructions
   (which may include using tools, MCP servers, or subagents).
3. *What guarantees?* Consistency, not determinism. The same skill
   produces the same *process* every time (same steps, same checks),
   even though individual LLM outputs within the workflow still vary.
   Think of it as deterministic procedure, nondeterministic execution.

### The modern stack

The current best practice for complex agent systems often looks like:

```
Skills       → reusable workflow expertise
  +
MCP servers  → standardized access to external systems
  +
Subagents    → role specialization and context isolation
  +
A2A          → cross-organization agent collaboration
```

> **Read:**
> - [Introducing Agent Skills](https://www.anthropic.com/news/skills)
> - [Skills explained (vs prompts, MCP, subagents)](https://www.claude.com/blog/skills-explained)
> - [Agent Skills specification](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/overview)

---

## Important clarification: real execution vs. pretend execution

This is one of the most important conceptual points for students. Three
things look similar in a chat log but are fundamentally different:

### A) Real tool execution (verifiable)

The system actually runs code/tools and returns real results. You can see
artifacts (files, logs, error messages). Code Interpreter, MCP servers,
and shell tools all work this way.

### B) Tool use by code generation (real only if executed)

The model writes code that *would* call an API. If your runtime executes
it, great. If not, it's just a plan. This is the PAL lineage.

**Common misconception:** "The agent called the API" vs. "the agent wrote
code that would call the API."

### C) Pure simulation (never executed)

The model emits text that *looks like* a tool call or terminal output,
but nothing actually ran. Especially common in the XML-tag era if the
runtime didn't execute anything.

**In your assignments:** always log (1) tool request from model, (2)
execution trace, (3) result returned, (4) final answer. This makes the
difference between A, B, and C unmistakable.

---

## The full comparison matrix

| Approach | Who decides? | Who executes? | Format | Format guarantees | Reusability | Best for |
|----------|-------------|---------------|--------|------------------|-------------|----------|
| Prompt + format request | LLM via prompt | Nobody (text only) | Free text / JSON | None (hope-based) | None | Simple structured output |
| Code generation (PAL) | LLM writes code | Interpreter | Code | Syntax only | Low | Computation, data transforms |
| ReAct-style prompting | LLM via prompt | Your glue code | Text conventions | None (hope-based) | Low | Prototyping, research |
| Function calling | LLM via API | Your code | JSON Schema | Valid JSON | Medium | App-specific tools (1–10) |
| Structured outputs | LLM via API | Your code | Strict JSON Schema | Schema-deterministic | Medium | When schema compliance is critical |
| MCP tool | LLM via API | MCP server | JSON-RPC | Schema-deterministic + discoverable | High | Cross-app tools, ecosystems |
| Subagent | Parent agent | Child agent runtime | Task delegation | Isolation (context, permissions) | High | Complex multi-step tasks |
| Skill | Agent loads instructions | Agent follows workflow | Markdown + resources | Consistent process, not output | Very high | Repeatable workflows, standards |
| A2A | Agent discovers peers | Remote agent | Protocol messages | Protocol-deterministic, behavior opaque | Very high | Cross-org agent collaboration |

---

## Decision flowchart: "what should I use?"

Use this when deciding how to give your LLM system new capabilities:

```
START: "I want the model to do X"
  │
  ├─ Can the model do X with just a better prompt?
  │   YES → Write a better prompt (Unit 4). Done.
  │   NO  ↓
  │
  ├─ Does X require external data or side effects?
  │   NO  → Use structured outputs to get reliable format. Done.
  │   YES ↓
  │
  ├─ Is X a single function in your own codebase?
  │   YES → Use function calling (+ structured outputs). Done.
  │   NO  ↓
  │
  ├─ Do multiple apps/hosts need to call X?
  │   NO  → Function calling is still fine. Done.
  │   YES ↓
  │
  ├─ Build an MCP server for X.
  │   │
  │   └─ Does the task require multi-step planning or exploration?
  │       NO  → Single tool call via MCP. Done.
  │       YES ↓
  │
  ├─ Does the subtask need isolation (context, permissions, model)?
  │   YES → Use a subagent with restricted tools. Done.
  │   NO  → Agent loop with MCP tools. Done.
  │
  └─ Is this a repeatable workflow you want standardized?
      YES → Package it as a skill. Done.
      NO  → You're probably fine with the above.
```

**The meta-principle:** start simple (prompts), add structure when
reliability demands it (structured outputs), add execution when the world
demands it (tools), add standardization when scale demands it (MCP),
add delegation when complexity demands it (subagents), and add expertise
when consistency demands it (skills).

---

## Timeline summary (for your notes)

| Year | Milestone | Key idea |
|------|-----------|----------|
| pre-2021 | Semantic parsing, task-oriented dialogue | NL to structured commands |
| 2021 | WebGPT | Train models to use browser as a tool |
| 2022 | MRKL, PAL, SayCan, ReAct | Prompting patterns for tool use |
| 2023 | Toolformer, Gorilla | Training tool use into model weights |
| Mar 2023 | OpenAI Plugins | Tools described via OpenAPI spec |
| Jun 2023 | Function calling | Structured API channel for tool calls |
| Nov 2023 | JSON mode | Valid JSON output (no schema guarantee) |
| Aug 2024 | Structured Outputs | Schema-enforced output via constrained decoding |
| Nov 2024 | MCP announced | Vendor-agnostic tool/resource protocol |
| Apr 2025 | A2A announced | Agent-to-agent interoperability protocol |
| Oct 2025 | Agent Skills | Reusable expertise bundles |
| 2025-26 | MCP in production | OpenAI, Anthropic, Cursor operationalize MCP |

---

## Security note: tools create new attack surfaces

As systems gain tool access (web, files, databases, MCP servers), they
inherit new risks. MCP's specification explicitly calls out:

- **User consent and control:** humans must approve tool access
- **Data privacy:** tools should not leak data across contexts
- **Tool safety:** destructive actions require explicit authorization
- **Prompt injection:** adversarial inputs can manipulate tool calls

This will come up again in Unit 13 (Observability & Debugging), but keep
it in mind as you build: every tool you add is an attack surface.

---

## Homework reading

### Foundational research papers

PDFs for all of these are in `unit05_readings.zip`.

1. Su et al., *Building Natural Language Interfaces to Web APIs* (2017) —
   [ysu1989.github.io/papers/cikm17_nl2api.pdf](https://ysu1989.github.io/papers/cikm17_nl2api.pdf)
2. Gupta et al., *Semantic Parsing for Task-Oriented Dialog* (2018) —
   [aclanthology.org/D18-1300](https://aclanthology.org/D18-1300/)
3. Nakano et al., *WebGPT* (2021) —
   [arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332)
4. Karpas et al., *MRKL Systems* (2022) —
   [arxiv.org/abs/2205.00445](https://arxiv.org/abs/2205.00445)
5. Gao et al., *PAL: Program-Aided Language Models* (2022) —
   [arxiv.org/abs/2211.10435](https://arxiv.org/abs/2211.10435)
6. Ahn et al., *SayCan* (2022) —
   [arxiv.org/abs/2204.01691](https://arxiv.org/abs/2204.01691)
7. Yao et al., *ReAct* (2022) —
   [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
8. Schick et al., *Toolformer* (2023) —
   [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)
9. Patil et al., *Gorilla* (2023) —
   [arxiv.org/abs/2305.15334](https://arxiv.org/abs/2305.15334)
10. Li et al., *API-Bank* (2023) —
    [arxiv.org/abs/2304.08244](https://arxiv.org/abs/2304.08244)
11. Qin et al., *ToolLLM / ToolBench* (2023) —
    [arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789)
12. Yang et al., *SWE-agent* (2024) —
    [arxiv.org/abs/2405.15793](https://arxiv.org/abs/2405.15793)

### Platform docs and protocol specs

13. [OpenAI Function Calling guide](https://platform.openai.com/docs/guides/function-calling)
14. [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
15. [Anthropic Tool Use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
16. [MCP announcement](https://www.anthropic.com/news/model-context-protocol)
17. [MCP specification](https://modelcontextprotocol.io/specification/latest)
18. [A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability)
19. [A2A specification](https://google.github.io/A2A/specification/)
20. [Anthropic Agent Skills overview](https://www.anthropic.com/news/skills)
21. [Skills explained (comparison)](https://www.claude.com/blog/skills-explained)
22. [The Complete Guide to Building Skills for Claude (PDF)](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf)

### Context and background

23. [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
    (Anthropic, Sep 2025) — bridges prompt engineering to the agentic
    context management that tool-using systems require
