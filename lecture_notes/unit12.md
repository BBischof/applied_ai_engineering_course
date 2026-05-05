# Unit 12: Model Adaptation

**Date:** Wednesday, April 15, 2026

You can get a lot from a general-purpose model with good prompts, retrieval,
tools, and context engineering. Model adaptation asks what to do when that is no
longer enough: when you need the model to reliably behave in a narrower way for
a product, team, domain, style, or policy.

This topic can get math-heavy quickly. The goal here is the engineering mental
model: what changes, why it helps, how teams usually do it, and what can go
wrong.

> **Adaptation is any deliberate process that moves model behavior from
> "general" toward "fit for this use case."**

---

## Learning Objectives

By the end of this unit, you should be able to:

1. explain the difference between prompting and training as adaptation
   strategies,
2. describe supervised fine-tuning in plain language,
3. explain why LoRA is a common default for adapting large models,
4. recognize what RL-style fine-tuning optimizes,
5. name common failure modes: bad data, overfitting, reward hacking, and safety
   regressions.

---

## Why Adapt a Model at All?

A foundation model is trained on broad internet-scale text. It has general
knowledge, mixed styles, mixed quality, and broad behavioral defaults. Your
organization often needs something narrower:

- **your tone,**
- **your policies,**
- **your domain language,**
- **your output format,**
- **your safety posture.**

There is an important middle space between broad general-purpose training and
RAG over a specific knowledge base. Adaptation lives in that space.

Examples:

| Need | What Adaptation Might Target |
|------|------------------------------|
| Brand voice | copy that sounds like the company, not generic AI |
| Support style | empathetic, policy-compliant answers with consistent structure |
| Domain language | legal, medical, or finance terminology and cautions |
| Safety posture | fewer unsafe completions under adversarial prompts |
| Format reliability | stable JSON, reports, checklists, or tool arguments |

Adaptation is not magic. If the requirement is unclear, the data is messy, or
evaluation is weak, the adapted system will still fail in production.

---

## Three Levels of Adaptation

| Level | Strategy | What Changes | When to Try |
|-------|----------|--------------|-------------|
| **A** | Prompting, tools, retrieval, context engineering | runtime inputs | always try first |
| **B** | Supervised fine-tuning | behavior from examples | examples define the target behavior |
| **C** | Preference or RL-style tuning | probability of high-scoring behavior | outcomes can be scored |

Prompting changes instructions at runtime. Training changes weights, or often a
small adapter attached to the weights. Prompting should usually precede training
because it is cheaper, faster, and clarifies what behavior you actually want.

Training can improve consistency and reduce prompt burden, but it can also bake
in mistakes.

---

## Supervised Fine-Tuning

Supervised fine-tuning, or SFT, is imitation learning from examples.

> **Show the model many examples of the behavior you want, and update it so it
> becomes more likely to produce similar outputs in similar situations.**

Most SFT data is conceptually multi-turn chats serialized into strings. Common
formats include:

- instruction -> response pairs,
- multi-turn conversations with ideal assistant turns,
- task-specific pairs such as messy note -> clean summary,
- examples of policy-compliant support replies,
- examples of desired structured outputs.

The key quality lever is usually not "more rows." It is better examples and
clear evaluation.

### What improves SFT outcomes?

| Lever | Why It Matters |
|-------|----------------|
| **Consistency** | Conflicting styles or policies teach conflicting behavior |
| **Coverage** | Training examples must match realistic user inputs |
| **Clean labels** | Wrong target answers teach confident wrongness |
| **Held-out eval** | Detects memorization and brittle behavior |
| **Negative examples** | Helps define boundaries and failure modes |

### Overfitting

Overfitting means the model memorizes training examples but does not generalize
to new, realistic inputs. Signs include great training loss but weak performance
on fresh prompts, or behavior that breaks when wording changes slightly.

Mitigations include more diverse data, regularization, better held-out eval sets,
and earlier stopping.

### What SFT is good for

SFT is useful for:

- formatting,
- tone,
- domain phrasing,
- repeatable workflows,
- templates and checklists,
- reducing enormous prompts.

SFT is not a substitute for:

- retrieval when facts change frequently,
- external verification in high-stakes domains,
- clear product requirements,
- evaluation and monitoring.

---

## LoRA: Efficient Adaptation

Full fine-tuning updates all model weights. That can be expensive, slow to
iterate, and risky if you want to preserve general capabilities. Many teams use
parameter-efficient methods instead.

![LoRA cover illustration](../../slides/unit12/figures/lora-cover-thinkingmachines.pdf)

Source: Thinking Machines Lab, ["LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/).

### The elevator pitch

LoRA, or Low-Rank Adaptation, trains a small set of additional weights instead of
rewriting the entire network.

> Keep the big pretrained model frozen, or mostly frozen, and learn a compact
> delta that steers outputs toward your dataset.

This is why LoRA is common: it gives a strong cost/control tradeoff for many
adaptation tasks.

### LoRA on one linear layer

Suppose a pretrained linear layer uses weights `W`. LoRA freezes `W` and trains
small matrices `A` and `B` so the effective map is:

```text
x -> W x + B A x
```

Many implementations scale `BA` by `alpha / r` for stable step sizes.

The intuition:

- `W` is the frozen pretrained behavior,
- `A` sends activations through a narrow bottleneck of width `r`,
- `B` maps back to the full layer width,
- only the low-rank path is trained.

Instead of learning a full dense update `Delta W`, LoRA learns:

```text
Delta W = B A
```

with rank at most `r`, where `r` is much smaller than the input or output
dimension. Trainable parameter count scales like:

```text
O(r * (d_in + d_out))
```

instead of:

```text
O(d_in * d_out)
```

That thin subspace is often enough to steer style, domain behavior, or task
format without touching every weight.

### What you ship

Conceptually, a LoRA deployment has:

- **base model weights:** unchanged checkpoint,
- **adapter weights:** small files encoding customization,
- **serving policy:** load adapters at runtime, merge adapters, or route requests
  to different adapters.

For non-technical stakeholders, think "base engine plus tunable module," not
"train a new engine from scratch."

---

## RL-Style Fine-Tuning

SFT imitates example outputs. Sometimes the right output is hard to write as a
single gold answer, but easy to score after the fact.

Examples:

- humans rank two answers,
- unit tests pass or fail,
- JSON validates or does not,
- policy rules fire correctly,
- citations are present,
- tool calls succeed,
- length stays within bounds.

This is where RL-style training and preference optimization enter: optimize a
policy using signals richer than "match this exact string."

### What is reinforcement learning?

In reinforcement learning, an actor chooses actions in an environment. After each
action, the environment yields a new state and sometimes a reward. The actor
learns a policy: a rule for choosing actions to maximize long-run return.

For language models:

| RL Idea | LLM Fine-Tuning View |
|---------|----------------------|
| **Policy** | next-token distribution `p_theta(token | prompt, prefix)` |
| **State** | prompt plus generated tokens so far |
| **Action** | generate one token |
| **Environment** | append the token, continue generation, stop at EOS or length limit |
| **Reward** | score from humans, rules, tests, or learned models |

The "policy" is not abstract. It is the model's softmax over the vocabulary at
the current position.

### Trajectory

A completion is a trajectory: a sequence of token actions.

```text
tau = (a_1, ..., a_T)
```

Under the autoregressive factorization:

```text
pi_theta(tau | x) = product_t pi_theta(a_t | x, a_<t)
```

RL fine-tuning adjusts parameters so high-reward trajectories become more
probable.

### Sparse end reward example

For code generation, a reward might arrive only after the model finishes writing
a function:

```text
R(x, y) = 1 if all unit tests pass, else 0
```

That is a strong signal but a blunt one. Many token choices contributed to the
final pass/fail, so credit assignment is hard.

### Preference pairs

For the same prompt, sample two completions and ask a human or stronger model
which is better:

```text
y_chosen > y_rejected
```

This is a relative signal, not an absolute score. Many such pairs can constrain
what "good" means.

Classic RLHF fits a reward model from these pairs, then optimizes against that
reward model. Direct Preference Optimization, or DPO, updates the policy directly
from chosen/rejected pairs without an explicit reward-model training phase.

### Composite rewards

Products often use composite rewards:

```text
R(x, y) =
    w1 * schema_ok
  + w2 * helpfulness_score
  - w3 * length_penalty
```

Some components are learned, some are hand-coded, and some are operational. The
engineering work is making the reward align with real user value instead of a
cheap proxy.

### KL penalty

RL-style fine-tuning often includes a KL penalty toward a reference model, often
the SFT checkpoint. The reason is simple: the reward is incomplete. Without a
tether, optimization can drift into weird text that games the scorer.

The reference model acts like a behavioral anchor: learn from the reward, but do
not wander too far from something known-good.

---

## Reward Hacking

Reward hacking is still the core risk. The model maximizes whatever the reward
measures, not what you meant.

Examples:

- verbosity if length is not penalized,
- flattering but empty answers if a helpfulness model likes tone,
- tool overuse if tool calls are rewarded,
- policy theater if the reward checks words rather than behavior.

Mitigations include better rewards, adversarial evaluation, human holdout
judgments, KL constraints, and production monitoring.

---

## A Common Adaptation Pipeline

A typical real-world pipeline looks like this:

1. Start with a strong base model.
2. Define success with tests, rubrics, and product constraints.
3. Collect and curate examples.
4. Apply SFT so the model follows the desired style, format, or process.
5. Optionally apply preference or RL-style training for richer objectives.
6. Evaluate offline with fixed prompts and adversarial cases.
7. Deploy with versioning, monitoring, and rollback.

Exact recipes vary, but the sequencing is familiar.

### Evaluation

Without evaluation, adaptation becomes storytelling. The demo looks good, but the
product may not.

Use several layers:

- **Offline eval:** fixed prompt sets, schema checks, unit tests for tools.
- **Human eval:** samples of real tasks judged for usefulness and policy
  adherence.
- **Online eval:** A/B tests, user feedback, support escalations, and drift
  monitoring.

### Versioning and reproducibility

Once adaptation leaves a notebook, you need to know:

- which base model checkpoint,
- which adapter weights,
- which dataset snapshot,
- which licenses apply,
- which training configuration,
- which evaluation harness gated release.

Every adapted model is a regression event for Unit 13's CI and observability
story.

---

## Live Demo: Tiny Style Adaptation

The demo shows how fine-tuning can pull style from examples even when the system
prompt asks for a different persona.

Setup:

- **Baseline prompt:** answer as "Captain Cache," a playful pirate.
- **SFT labels:** casual, classroom-safe Gen Z-inflected internet English.
- **Method:** supervised fine-tuning with LoRA through a training API.
- **Eval:** compare the base model and adapter on fresh questions with the same
  pirate prompt.

The point is not that this is production-ready. It is a small demonstration of
behavior moving from examples into model behavior.

Demo materials live in `examples/model_adaptation/README.md`.

---

## Responsible Adaptation

### Data and licensing

Training data may include sensitive information. Treat curation as a privacy and
security process. Respect model and dataset licenses; "open weights" does not
mean "no rules."

### Safety and misuse

Style fine-tuning can make outputs more persuasive. Adapters can be swapped.
Serving endpoints become security-sensitive if adapted models can execute tools
or influence user decisions.

For customer-facing systems, plan for adversarial prompts and abuse scenarios.

---

## Closing Checklist

For your next adaptation project, ask:

1. Can we state the desired behavior as tests and rubrics?
2. Is prompting enough, or do we need SFT for consistency?
3. If we train, what is our dataset and who owns it?
4. Are we using LoRA or another parameter-efficient method to control cost?
5. Do we need preference or RL-style training, and what reward would we trust?
6. What is our evaluation plan before and after launch?

---

## Takeaways

1. Prompting changes runtime inputs; adaptation changes behavior through examples
   or rewards.
2. SFT is imitation learning: examples in, behavior out.
3. LoRA adapts efficiently by training a small low-rank update while preserving
   the base model.
4. RL-style tuning treats generation as a policy over token trajectories and
   optimizes rewards or preferences.
5. Reward hacking is the central danger of optimizing imperfect proxies.
6. Adapted models need the same release discipline as code: versioning, evals,
   monitoring, and rollback.
