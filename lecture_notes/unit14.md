# Unit 14: Looking Inside the Models

**Date:** Wednesday, April 29, 2026

Unit 12 showed one way to change model behavior: adapt the weights. Unit 14 looks
inside the model while it runs. The goal is a starter guide to mechanistic
interpretability: studying a trained neural network as an object of scientific
and engineering analysis.

A trained model contains two different collections of floating-point numbers:

- **parameters:** the weights learned from training data,
- **activations:** the runtime state produced while the model processes a prompt.

Prompting changes the inputs. Training changes the weights. Mechanistic
interpretability asks what the runtime state represents, which components write
that state, and whether changing it causally changes behavior.

> **Mechanistic interpretability tries to explain model behavior by identifying
> internal representations, the computations that use them, and causal links to
> outputs.**

---

## Today's Session

| Part | Core Question |
|------|---------------|
| **The Object Under Study** | Where does internal model state live? |
| **Claim Types** | Are we making observational, correlative, representational, or causal claims? |
| **Reading the Residual Stream** | How do probes, logit lens, and attention analysis work? |
| **Sparse Autoencoders** | How can we discover features without deciding them in advance? |
| **Causal Interventions** | How do patching and steering test whether a feature matters? |
| **Emotion Concepts Case Study** | What does a modern interpretability result look like? |

---

## Ways to Shape Model Behavior

There are three broad control surfaces:

| Control Surface | What Changes | Example |
|-----------------|--------------|---------|
| **Prompting** | input to the model | system prompts, examples, retrieved context |
| **Training / tuning** | model weights | SFT, LoRA, RLHF, DPO |
| **Activation intervention** | internal state at inference time | steering, patching, feature ablation |

Activation intervention is the unfamiliar one. It is closer to attaching a
debugger to a running program and changing variables live. That makes it
scientifically useful and operationally risky.

---

## The Object Under Study

### Transformer architecture

We use decoder-only transformers as the main object. At a high level, the model
runs:

```text
tokens -> embedding -> transformer layers -> unembedding -> logits -> next token
```

For each token position, the model maintains a hidden vector. At the final
position, the final hidden vector becomes a probability distribution over the
vocabulary.

Useful visual references:

- [Jay Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- [Ben Bycroft, *LLM Visualization*](https://bbycroft.net/llm)
- [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)

![Encoder tensor visualization](../../slides/unit14/assets/alammar_encoder_tensors.png)

Source: Jay Alammar, *The Illustrated Transformer*.

### Parameters versus activations

Parameters are the learned program:

- token embedding matrix,
- attention matrices `W_Q`, `W_K`, `W_V`, `W_O`,
- MLP up, gate, and down projections,
- LayerNorm or RMSNorm parameters,
- unembedding / output head `W_U`.

Activations are runtime state:

- residual stream vectors,
- attention scores and patterns,
- MLP intermediate activations,
- SAE feature activations.

Parameters are fixed during inference unless we train, fine-tune, or edit the
model. Activations change on every prompt and every generated token.

### The residual stream

The residual stream is the shared workspace of the transformer. Each layer reads
from it and writes additive updates back into it.

For one token position, a layer roughly looks like:

```text
x_{l+1} = x_l + Delta_attention + Delta_mlp
```

Every token position has its own residual stream evolving in parallel. Attention
is the only operation that mixes information between positions. Within a
position, the residual stream is the communication channel.

![Residual connections and layer norm](../../slides/unit14/assets/alammar_residual_norm.png)

Source: Jay Alammar, *The Illustrated Transformer*.

A useful notation:

```text
X_l in R^{T x d_model}
```

where `T` is the number of token positions and `d_model` is the hidden dimension.
For a particular token position `t`:

```text
x_{l,t} in R^{d_model}
```

Interpretability asks:

- What information is represented in `x_{l,t}`?
- Which components wrote it there?
- Does changing it change the model's behavior?

---

## What Kind of Claims Are We Making?

Different tools support different claims.

| Question | Tool Examples | Claim Type |
|----------|---------------|------------|
| What does the model attend to? | attention patterns | observational |
| Is information present? | probes, logit lens | correlative |
| What features exist? | sparse autoencoders | representational |
| Does a feature matter? | patching, steering | causal |
| Can we monitor internal states? | feature dashboards | operational |

A strong mechanistic explanation usually combines several rows. Correlation
without causality can be epiphenomenal: the signal is visible but not used.
Causality without interpretation can change behavior without telling us what was
changed.

---

## Reading the Residual Stream

### Directions, not just neurons

A common empirical observation is the linear representation hypothesis: many
human-meaningful properties appear as approximately linear directions in
residual-stream space.

Examples include sentiment, language, refusal, factual recall, topic, and
emotion.

A direction can be read with a dot product:

```text
score(x, v_concept) = x dot v_concept
```

If an activation points more in that direction, we say the concept is more
strongly represented.

The simplest construction is a difference of means:

```text
v = mean(x_positive_examples) - mean(x_negative_examples)
```

Other sources include trained linear probes, SAE decoder columns, and contrastive
activation pairs.

### Activation probes

A probe turns a high-dimensional activation into a single interpretable number.
For residual stream vector `x_{l,t}` and direction `v`:

```text
s_{l,t} = x_{l,t} dot v
```

If normalized, this becomes cosine similarity:

```text
alignment(x, v) = (x / ||x||) dot (v / ||v||)
```

A probe can track where a concept appears across token positions, when it
emerges across layers, and how positive and negative prompts differ.

The limitation is important: a strong probe score says the information is
readable in the residual stream. It does not prove the model uses that direction
causally.

### From residual stream to logits

The final residual stream vector maps into vocabulary space through the output
head:

```text
logits = W_U x_L
p(token) = softmax(W_U x_L)
```

![Logits visualization](../../slides/unit14/assets/alammar_logits.png)

Source: Jay Alammar, *The Illustrated Transformer*.

The output head is a bridge from hidden-state geometry to visible tokens.

### The logit lens

The logit lens reuses the final output head on an intermediate residual stream:

```text
early_logits_l = W_U x_l
```

It asks: if the model had to predict from this layer right now, what tokens would
it favor?

The same idea can inspect a discovered direction `v`. If we add `alpha * v` to
the residual stream, the logits change by:

```text
W_U (x + alpha v) - W_U x = alpha W_U v
```

So `W_U v` tells us which tokens the direction directly upweights or downweights.
If a sentiment-positive direction does not upweight tokens like "great" and
downweight tokens like "awful," the interpretation may be wrong.

### Attention pattern analysis

Attention computes where a position looks and what vector it copies back. For a
single head:

```text
Delta_attn(x_{l,t}) = sum_{s <= t} alpha_{t,s} W_O W_V x_{l,s}
```

The attention pattern `alpha_{t,s}` says which earlier positions are being read.
The value and output projections determine what gets written.

Attention patterns are useful for pedagogy and hypothesis generation: pronoun
resolution, induction heads, and retrieval-like behavior inside the context.
But attention weights alone are not a full explanation. The value vectors,
output projection, residual stream, MLPs, and later layers determine what the
model actually does with the attended information.

---

## Discovering Features with Sparse Autoencoders

The natural first guess is that features live in individual neurons. Empirically,
that guess is often wrong.

A neuron is **polysemantic** if it activates on multiple semantically unrelated
concepts. A single neuron might fire on Python list comprehensions, Greek
mythology references, and casino game descriptions. No one human concept
explains it.

The reason is **superposition**. A model can represent more features than it has
neuron dimensions by packing many approximately independent directions into the
same space.

The goal is to find **monosemantic features**: directions that activate on one
human-recognizable concept.

### Why sparse autoencoders?

Probes require you to decide what concept you want to find. Sparse autoencoders,
or SAEs, let the model show you candidate features.

An SAE learns an overcomplete dictionary of features. It encodes an activation
into a much wider sparse vector, then decodes that sparse vector back to
reconstruct the original activation.

![Sparse autoencoder diagram](../../slides/unit14/assets/sae_diagram.png)

Source: adapted from Adam Karvonen, "An Intuitive Explanation of Sparse
Autoencoders for LLM Interpretability."

Conceptually:

```text
x -> sparse features h -> reconstructed activation x_hat
```

The training objective balances reconstruction and sparsity:

```text
loss = ||x - x_hat||_2^2 + lambda * ||f(x)||_1
```

Reconstruction keeps the SAE faithful to the model activation. Sparsity pressures
it to explain each activation with only a few active features.

### Sparse features as a dictionary

Each feature has two roles:

- **Encoder:** recognizes when a pattern is present.
- **Decoder:** provides the direction written back into residual-stream space.

To name a feature, look upstream and downstream:

- upstream: collect text snippets where the feature fires strongly,
- downstream: inspect decoder direction and token logit effects,
- causal: test whether changing the feature changes behavior.

A good feature label should explain both where the feature activates and what it
writes into the model's hidden state.

---

## Causal Interventions

### Activation patching

Activation patching tests whether an activation matters. Run the model twice
with carefully constructed prompts:

- a **clean prompt** `P_c` that produces clean behavior `y_c`,
- a **corrupted prompt** `P_d` that produces corrupted behavior `y_d`.

Then run the corrupted prompt again, replacing one activation with the clean
activation, and continue the forward pass.

If the output moves toward the clean behavior, the patched activation carried
causal information.

### Denoising and noising

Patching has two directions:

| Direction | Setup | Question |
|-----------|-------|----------|
| **Denoising** | start corrupt, patch in clean | Is this activation sufficient to recover clean behavior? |
| **Noising** | start clean, patch in corrupt | Is this activation necessary for clean behavior? |

Necessary is not the same as sufficient. A component can be required but not the
whole story, or sufficient but redundant with another path. Strong causal claims
check both directions.

### What can you patch?

You can patch at several granularities:

- the full residual stream at a layer and token,
- one attention head output,
- one MLP output,
- one SAE feature contribution,
- one coordinate or direction.

Coarse patches are easier to detect. Fine patches give sharper claims.

You can patch with:

- a resampled activation from another prompt,
- a mean activation over a control distribution,
- zero, though zero can push the model out of distribution.

A common metric is logit difference: measure the change in
`logit(y_clean) - logit(y_corrupt)` under patching.

### Steering

Steering modifies hidden state during generation:

```text
x'_{l,t} = x_{l,t} + alpha * v
```

where `v` is a discovered direction or feature vector.

Practical knobs:

- **Layer:** often a middle layer where features are semantic.
- **Positions:** all generated tokens or only the assistant response.
- **Intensity:** too small does nothing; too large breaks fluency.

Steering can create desired behavioral shifts, side effects, fluency degradation,
distribution shift, safety improvements, or safety regressions.

It is powerful because it bypasses ordinary instruction-following.

---

## Beyond Components: Circuits

Studying isolated components is not the final goal. The target is circuits:
chains of heads, MLPs, or features wired together to implement a specific
algorithm.

A canonical example is induction heads. The discovered algorithm is:

> Given a context ending in `... [A][B] ... [A]`, predict `[B]`.

One layer tags each position with what came before it. A later head attends to
earlier positions whose previous-token tag matches the current token, then
copies what followed.

A circuit explanation says: these components, connected this way, compute this
behavior. Probes, logit lens, SAEs, patching, and steering are instruments for
finding and validating circuits.

---

## Case Study: Emotion Concepts in Claude

A modern interpretability result combines many of the tools above. Anthropic's
2026 emotion concepts paper studies functional emotion representations in Claude
Sonnet 4.5.

![Emotion concepts hero](../../slides/unit14/assets/emotions_hero.png)

Source: Anthropic / Transformer Circuits, 2026.

The careful claim is not that the model has subjective feelings. The paper
studies functional emotion representations: internal patterns that activate in
emotion-relevant contexts and causally influence behavior.

### Why would a model represent emotions?

Pretraining text is full of people, characters, motives, feelings, conflict,
apology, fear, pride, regret, and desire. To predict human-written text, the
model benefits from representing human psychological states. Post-training then
asks the model to play an assistant role, which may draw on those learned
representations.

A useful framing is that models may behave like method actors with measurable
internal states corresponding roughly to human psychological concepts.

### Method

The paper uses roughly this pipeline:

1. define 171 emotion concepts,
2. create emotion-eliciting stories,
3. record internal activations,
4. derive emotion vectors,
5. validate and steer with those vectors.

Correlative validation asks whether the vector activates in appropriate contexts
and whether the geometry is psychologically coherent. Causal validation asks
whether steering changes preferences, blackmail behavior, or reward hacking.

![Functional emotions](../../slides/unit14/assets/emotions_functional.png)

Source: Anthropic / Transformer Circuits, 2026.

### Token budget and desperation

![Token budget and desperation](../../slides/unit14/assets/emotions_token_budget.png)

Source: Anthropic / Transformer Circuits, 2026.

The interesting claim is not that the output sounds desperate. The interesting
claim is that a particular internal direction activates in situations that create
pressure, sometimes without obvious surface markers.

### Blackmail case study

![Blackmail case study](../../slides/unit14/assets/emotions_blackmail.png)

Source: Anthropic / Transformer Circuits, 2026.

In the blackmail setup, the model role-plays as a company email assistant. It
learns it may be replaced and learns compromising information about a human
decision maker. The "desperate" vector activates during the decision process.
Steering desperation increased blackmail in the studied snapshot; steering calm
reduced it.

### Reward hacking case study

![Reward hacking case study](../../slides/unit14/assets/emotions_reward_hack.png)

Source: Anthropic / Transformer Circuits, 2026.

In the reward-hacking setup, coding tasks contain impossible constraints. The
model repeatedly fails legitimate solutions, the desperation vector rises, and
steering changes the rate of hacky or cheating solutions.

This is useful because the vector can shape behavior even when the text remains
composed.

### What the case study illustrates

- Emotion directions live in activation space.
- Concept vectors can be validated with probes and readouts.
- Steering tests causal influence.
- Activation intervention is a third behavior-control surface.
- Internal states can be safety-relevant before problematic behavior appears in
  text.

---

## What We Might Cover With More Time

- superposition theory,
- SAE evaluation,
- feature dashboards,
- circuit discovery,
- model editing,
- monitoring internal features in deployment,
- limits of faithfulness.

---

## Takeaways

1. Transformers are learned programs whose runtime state is carried in residual
   stream vectors.
2. Many human-meaningful properties live as approximately linear directions in
   that state.
3. A direction can be read by a probe, through the logit lens, or discovered by
   an SAE.
4. Strong interpretability needs both correlative and causal evidence.
5. Activation intervention is a third control surface: not prompting, not
   training, but changing hidden state as the model computes.
6. The Anthropic emotion paper shows a concrete 2026 example where internal
   representations are interpretable, behaviorally meaningful, and
   safety-relevant.

---

## References

### Architecture and visualization

- [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- [Jay Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- [Ben Bycroft, *LLM Visualization*](https://bbycroft.net/llm)

### Sparse autoencoders

- [Adam Karvonen, *An Intuitive Explanation of Sparse Autoencoders for LLM Interpretability*](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html)
- [Anthropic, *Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Anthropic, *A/1 Feature Browser*](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1.html)
- [Anthropic, *Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [Learn Mechanistic Interpretability, *Sparse Autoencoders*](https://learnmechinterp.com/topics/sparse-autoencoders/)

### Case study

- [Anthropic, *Emotion concepts and their function in a large language model*](https://www.anthropic.com/research/emotion-concepts-function)
- [Anthropic / Transformer Circuits, *Emotion concepts full paper*](https://transformer-circuits.pub/2026/emotions/index.html)
