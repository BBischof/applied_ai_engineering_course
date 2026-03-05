# Unit 2: Foundations of Large Language Models

**Date:** Wednesday, January 28, 2026

This lecture traces the path from the oldest problem in NLP—language modeling—to the modern LLM-based assistants you use every day. By the end you should be able to explain what happens inside the API call, why certain failure modes exist, and where a standalone LLM stops being enough.

| Part | Core Question |
|------|---------------|
| **The Language Modeling Problem** | What does a language model do, and how did we get here? |
| **The LLM "API"** | What knobs does the model expose, and what do they control? |
| **From Language Model to Assistant** | How does a raw text predictor become a helpful chatbot? |
| **Practical Considerations** | Where do LLMs shine, and where do they break? |
| **Beyond the LLM** | Why do real systems need more than a model? |

---

## Part 1: The Language Modeling Problem

**Language modeling** is one of the oldest problems in NLP.

> Given some text, what word should come next?

**Applications:**
- **Autocomplete:** smartphone keyboards, search engines
- **Spelling correction:** compare input against most probable word sequences
- **Machine translation:** predict the most likely sequence in a target language
- **Text generation:** chatbots, writing assistants

A good language model captures what makes text "sound right"—grammar, meaning, style, and world knowledge.

### Language Modeling = Next-Word Prediction

A **language model** assigns probabilities to sequences of words.

Given a sequence of words $w_1, w_2, \ldots, w_{n-1}$ (the **context**), we want to estimate:

$$P(w_n \mid w_1, w_2, \ldots, w_{n-1})$$

That is: a probability distribution over the entire vocabulary, given all of the previous words.

**Example:** "The capital of France is" → ?

| Token | Probability |
|-------|-------------|
| "Paris" | 0.92 |
| "the" | 0.03 |
| "a" | 0.01 |
| ... | ... |

### Why Is This Hard?

The space of possible sentences is essentially infinite.

- With a vocabulary of 50,000 words:
  - 10-word sentences: $50{,}000^{10} \approx 10^{47}$ possibilities
  - More than atoms in the observable universe
- We can never enumerate all valid sentences
- Yet humans effortlessly judge "The cat sat on the mat" vs. "Mat the on sat cat the"

**The challenge:** how do we estimate next-word probabilities for sequences whose prefixes we've never seen?

**The history:** N-gram models (1980s) → Neural LMs (2003) → RNNs (2010s) → Transformers (2017) → LLMs (2020s)

### N-gram Models: The Foundation

**The Markov Assumption.** Approximate the full history by only the last $n-1$ words:

$$P(w_i \mid w_1, \ldots, w_{i-1}) \approx P(w_i \mid w_{i-n+1}, \ldots, w_{i-1})$$

**Types:**
- **Unigram ($n=1$):** $P(w_i)$ — ignores all context
- **Bigram ($n=2$):** $P(w_i \mid w_{i-1})$ — one word of context
- **Trigram ($n=3$):** $P(w_i \mid w_{i-2}, w_{i-1})$ — two words of context

**Key insight:** we can estimate these probabilities by *counting* occurrences in a corpus.

#### N-gram example: training

**Training corpus** (3 sentences):
1. "the cat sat on the mat"
2. "the cat ate the fish"
3. "the dog sat on the rug"

**Bigram counts** (how often does word B follow word A?):

|  | the | cat | dog | sat | ate | on | mat | fish | rug |
|---|-----|-----|-----|-----|-----|-----|-----|------|-----|
| the | 0 | 2 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| cat | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| sat | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| on | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Bigram probability:**

$$P(\text{cat} \mid \text{the}) = \frac{\text{count}(\text{the cat})}{\text{count}(\text{the } \cdot)} = \frac{2}{6} = 0.33$$

#### N-gram example: inference

**Query:** What word usually follows "the"?

| Word | Count | Probability |
|------|-------|-------------|
| "cat" | 2 | 2/6 = 0.33 |
| "dog" | 1 | 1/6 = 0.17 |
| "mat" | 1 | 1/6 = 0.17 |
| "fish" | 1 | 1/6 = 0.17 |
| "rug" | 1 | 1/6 = 0.17 |

**Query:** What word usually follows "cat"?

| Word | Count | Probability |
|------|-------|-------------|
| "sat" | 1 | 1/2 = 0.50 |
| "ate" | 1 | 1/2 = 0.50 |

> This is language modeling: given context, output a probability
> distribution over next words.

### Limitations of N-gram Models

What happens with "the elephant"? We never saw "elephant" in training, so: $P(\text{elephant} \mid \text{the}) = 0$.

**Fundamental problems:**
1. **Data sparsity:** many valid n-grams never appear in training
2. **Limited context:** bigrams can't capture "The cat that I saw yesterday sat…"
3. **No generalization:** "cat" and "kitten" are completely unrelated
4. **Storage explosion:** $50{,}000^3 = 10^{14}$ possible trigrams

> **The solution: neural networks.** Instead of counting, learn
> *continuous representations* of words. Similar words get similar
> vectors, enabling generalization to unseen combinations.

### From N-grams to Neural Language Models

| | N-gram approach | Neural approach |
|---|---|---|
| Representation | Discrete: each word is independent | Continuous: words are vectors |
| Similarity | "cat" and "kitten" unrelated | "cat" and "kitten" are similar |
| Unseen inputs | Zero probability | Smooth probabilities for everything |

> Modern LLMs are neural language models trained at massive scale. The
> core task is still the same: predict the next token. But neural
> networks can generalize far beyond what they've seen.

### Word2Vec: Words as Vectors

**Key insight (Mikolov et al., 2013):** words can be represented as dense vectors in a continuous space, learned from co-occurrence patterns.

- Each word becomes a vector of ~100–300 numbers
- Similar words have similar vectors (close in space)
- Trained by predicting words from their context (or vice versa)

**Two training approaches:**
- **Skip-gram:** given a word, predict surrounding words
- **CBOW:** given surrounding words, predict the center word

This was the breakthrough that made neural language models practical— "cat" and "kitten" become close neighbors, enabling generalization beyond exact matches.

### The Geometry of Meaning

Semantic relationships become geometric relationships:

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

Other relationships:
- $\vec{\text{Paris}} - \vec{\text{France}} + \vec{\text{Italy}} \approx \vec{\text{Rome}}$
- $\vec{\text{walking}} - \vec{\text{walk}} + \vec{\text{swim}} \approx \vec{\text{swimming}}$

> Words live in a high-dimensional "latent space" where directions
> encode meaning. This is the foundation for all modern
> embeddings—including LLM representations.

### From Word2Vec to Modern Embeddings

**Word2Vec's limitation:** one vector per word, regardless of context. "bank" (financial) and "bank" (river) get the same vector.

**Evolution of word representations:**
1. **Word2Vec / GloVe (2013–2014):** static embeddings
2. **ELMo (2018):** context-dependent embeddings from LSTMs
3. **BERT / Transformers (2018+):** contextualized embeddings
4. **Modern LLMs:** rich, layered representations that evolve through the network

Modern LLMs still use embeddings at their core—they just compute context-dependent embeddings that change based on surrounding text.

### What Does a Word Vector Actually Look Like?

Each word becomes a list of numbers. Here's a peek at the first 8 dimensions of a 300-dimensional embedding for "king":

```
king → [0.127, -0.834, 0.291, 0.045, -0.512, 0.678, -0.103, 0.419, ...]
        └────────── first 8 of 300 dimensions ──────────┘
```

- Each dimension captures some aspect of meaning (not human-interpretable)
- Similar words have similar patterns of numbers
- The values are learned from co-occurrence statistics in text

**Typical embedding dimensions:**

| Model | Dimensions |
|-------|-----------|
| Word2Vec | 100–300 |
| BERT | 768 |
| GPT-3 | 12,288 |
| OpenAI text-embedding-3-large | 3,072 |

> **Why this matters for retrieval:** embeddings power **semantic
> search**—convert documents and queries to vectors, then find the
> closest matches. This is the foundation of RAG (Retrieval-Augmented
> Generation), which we'll cover later.

### Tokens and Vocabulary

Working with whole words seems natural but is problematic:

- **Vocabulary explosion:** English has 170,000+ words, plus names, technical terms, misspellings…
- **Unknown words:** what about "ChatGPT" or "COVID-19"? Any new word is unrepresentable.
- **Morphology blindness:** "run", "running", "runs" treated as completely unrelated
- **Multilingual inefficiency:** each language needs its own vocabulary

> **Solution: subword tokenization.** Break text into **tokens**—pieces
> that are often smaller than words. This gives a compact, flexible
> vocabulary that can represent any text.

A **token** is the atomic unit of text that the model processes. Tokens are typically subword units—pieces of words.

**Examples of tokenization:**
- "Hello" → ["Hello"]
- "unhappiness" → ["un", "happiness"] or ["un", "happ", "iness"]
- "ChatGPT" → ["Chat", "G", "PT"]

**Rule of thumb:** 1 token ≈ 0.75 words (or ~4 characters in English)

**Benefits:**
- Fixed, manageable vocabulary size (32K–200K tokens)
- Can represent *any* text, including novel words
- Shares subword knowledge ("un-" tends to mean negation across words)

#### Tokenization in action

```
"The quick brown fox jumps"
 → [The] [_quick] [_brown] [_fox] [_jumps]     (5 tokens)
     464    4062    14198   39935   35308        (token IDs)

"ChatGPT is extraordinary!"
 → [Chat] [G] [PT] [_is] [_extra] [ordinary] [!]  (7 tokens)
    16047  38  2898  374    5066     74468      0
```

Key observations:
- Leading space ("\_") is often part of the token
- Novel words like "ChatGPT" are split into pieces
- Common words stay whole; rare words are formed from pieces

#### Vocabulary

The **vocabulary** is the complete set of tokens the model can produce.

| Model | Vocabulary size |
|-------|----------------|
| GPT-2 | 50,257 tokens |
| GPT-3/4 | 100,277 tokens |
| GPT-4o | 199,997 tokens |
| LLaMA | 32,000 tokens |

**Special tokens:**
- `<|endoftext|>` — end of generation
- `<|im_start|>` — message start
- `<|im_end|>` — message end

Common words become single tokens; rare words are split into pieces. The vocabulary emerges from the statistics of real text via algorithms like Byte Pair Encoding (BPE).

### Fundamentals Summary

- **Language modeling** = predicting the next token given previous tokens
- **Tokens** are subword units; vocabulary sizes range from 32K–200K
- Base LLMs are trained to predict plausible continuations from a large corpus

> An LLM is a function: `text_in` → `probability_distribution_over_next_token`
>
> Everything else—chat, code generation, reasoning—emerges from
> iterating this simple operation at scale.

### The Transformer Architecture

In 2017, *Attention Is All You Need* introduced the **Transformer**. This is the architecture behind modern LLMs: GPT, Claude, LLaMA, Gemini, etc.

**High-level structure:**
1. **Embedding layer:** convert input tokens to vectors
2. **Transformer blocks** (repeated $N$ times):
   - Self-attention: tokens "look at" each other
   - Feedforward network: process each position
3. **Output layer:** predict next token probabilities

### Self-Attention: The Core Mechanism

**The question attention answers:** "What other tokens should I pay attention to?"

Remember "bank" (building) vs "bank" (river)? Self-attention uses the surrounding tokens to disambiguate.

- Each token computes a *relevance score* with every other token
- Scores determine how much information to gather from each position
- This happens in parallel across all positions—a good fit for accelerators like GPUs

**Example:** "The cat sat on the mat because **it** was tired"
- What does "it" refer to? Attention learns to look back at "cat"
- This long-range dependency is something N-grams couldn't capture

### The Cost of Attention

**The catch:** every token attends to every other token.

**Computational complexity:** $O(n^2)$ where $n$ = sequence length

| Context Length | Attention Pairs |
|----------------|----------------|
| 1,000 tokens | 1 million |
| 10,000 tokens | 100 million |
| 100,000 tokens | 10 billion |

Memory requirements for self-attention scale quadratically in context length. This is why context windows have hard limits.

---

## Part 2: The LLM "API"

Think of a language model as a function with a simple API:

```
next_token_probs = model(input_tokens)
```

**Input:** a sequence of tokens (your prompt)

**Output:** a probability distribution over all possible next tokens

Every interaction with an LLM—whether asking a question, generating code, or having a conversation—is built on this simple operation. The model has no memory between calls (except what's in the context). Each request starts fresh.

### The Generation Loop

LLMs generate text **one token at a time** in an autoregressive loop:

1. Start with input prompt tokens
2. Model outputs probability distribution over vocabulary
3. Sample (or select) the next token from the distribution
4. Append the selected token to the context
5. Repeat until a stop condition is met

> Generation is **fundamentally sequential**. Each token depends only on
> previous tokens. The model cannot "look ahead" or revise earlier
> tokens. This is why LLMs sometimes paint themselves into corners—they
> can't go back and fix earlier mistakes.

### Why the First Token Is Slow

Generation happens in two phases:

```
──────────────────────────── time ──────────────────────────────►

 ┌──────────────────────────┐  ┌────┐┌────┐┌────┐┌────┐
 │  Process entire prompt   │  │ t1 ││ t2 ││ t3 ││... │
 │  (all tokens at once)    │  └────┘└────┘└────┘└────┘
 └──────────────────────────┘
  Prompt Processing              Token Generation
  Slow (scales with prompt       Fast (constant per token, length)                        reuses cached work)
```

| Phase | What happens | Speed |
|-------|-------------|-------|
| **Prompt processing** | "Read" all input tokens, compute internal state | Time ∝ prompt length |
| **Token generation** | One new token at a time, reusing prior work | Time ≈ constant per token |

A 10,000-token prompt might take 2 seconds before you see the first word—but then tokens stream quickly. This is called **time-to-first-token** (TTFT).

### Prompt Caching

If multiple requests share a common prefix, we can skip re-processing it.

```
┌──────────────────────────────────────────┐
│  System prompt + few-shot examples       │  ← processed once
└──────────┬──────────────┬────────────────┘
           │              │
     ┌─────▼─────┐ ┌─────▼─────┐ ┌─────────────┐
     │ Query 1   │ │ Query 2   │ │ Query 3     │
     └───────────┘ └───────────┘ └─────────────┘
      only new part  only new part  only new part
```

**Benefits:**
- **Latency:** skip prompt processing for shared prefix → faster TTFT
- **Cost:** many APIs offer 50%+ discount on cached input tokens
- **Throughput:** more efficient use of compute

**When to use:** long system prompts, few-shot examples, RAG with common instructions—anywhere you repeat the same prefix across requests.

### Choosing the Next Token

Given a probability distribution, how do we pick which token to output?

**Greedy decoding:** always pick the highest probability token
- Deterministic (mostly)
- Can be repetitive and boring

**Sampling:** randomly sample according to the probabilities
- More varied and creative
- Can be incoherent if not controlled

In practice, we use **controlled sampling** with parameters like temperature, top-k, and top-p to balance coherence and creativity.

### Temperature

**Temperature** ($T$) controls the "creativity" of sampling:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| Setting | Behavior |
|---------|----------|
| **$T \to 0$ (greedy)** | Always pick highest probability; deterministic |
| **$T = 1$ (neutral)** | Sample from the true distribution; balanced |
| **$T > 1$ (creative)** | Flatter distribution; more randomness |

**For most applications:** use $T = 0$ or low temperature for consistency; higher for creative writing.

### Top-k and Top-p Sampling

Additional controls to filter the distribution before sampling:

**Top-k sampling:**
- Only consider the $k$ most probable tokens
- Set probability of all others to 0, renormalize
- Typical: $k = 40$–$100$

**Top-p (nucleus) sampling:**
- Include tokens until cumulative probability $\geq p$
- Adapts to the distribution shape
- Typical: $p = 0.9$–$0.95$

**Combining strategies:** apply temperature first, then top-k or top-p filtering, then sample.

### Stop Conditions

Generation continues until:

1. **End-of-sequence token:** model outputs `<|endoftext|>` or similar
2. **Max tokens reached:** hit the configured output limit
3. **Stop sequences:** model outputs a specified string
4. **Context window full:** no more space for tokens

> Always set a reasonable `max_tokens` limit. Unbounded generation can
> be expensive and produce rambling output.

### Multi-Turn Conversations

Modern chat APIs structure conversations as a **list of messages**, each with a role:

| Role | Purpose | Written by |
|------|---------|-----------|
| **system** | Instructions for the model's behavior | Developer |
| **user** | Messages from the human | End user |
| **assistant** | Messages from the model | Model |

**Example conversation:**

```
sys:  "You are a helpful tutor."
usr:  "What is 2+2?"
ast:  "2+2 equals 4."
usr:  "And times 3?"
ast:  "4 times 3 equals 12."
```

> This is just a convention. The entire conversation is concatenated
> into a single token sequence using special delimiter tokens, then
> fed to the model.

#### What the model actually sees

**Your API call:**

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "What's 2+2?"}
]
```

**What the model receives** (approximately):

```
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
Hello!<|im_end|>
<|im_start|>user
What's 2+2?<|im_end|>
```

The model then generates tokens until it produces `<|im_end|>` or hits a limit.

#### Conversation format: implications

Understanding the format explains common behaviors:

- **Context grows each turn:** long conversations get expensive and slow
- **Earlier messages can be "forgotten":** lost-in-the-middle applies
- **System prompt is just a convention:** it's the first message, not magic
- **You can "edit" history:** modify past messages before sending
- **You can inject assistant messages:** pre-fill responses for formatting

For long conversations, consider summarizing earlier messages or using a sliding window to manage context length.

### Structured Output Generation

LLMs can generate text that follows specific formats:

**Prompting for structure:**
- "Respond in valid JSON format…"
- "Output as a markdown table…"
- Provide examples of desired format

**Constrained decoding:**
- API-level enforcement (OpenAI's JSON mode, structured outputs)
- Grammar-guided generation
- Only allow tokens that keep output valid

Structured outputs enable LLMs to be integrated into software systems—parse JSON, call functions, fill templates.

### Function Calling / Tool Use

Via structured outputs, modern LLM APIs support **function calling**:

1. Define available functions with schemas
2. Model decides when to call a function
3. Model generates structured arguments
4. Your code executes the function
5. Return result to model for next step

**Example functions:**
- `search_database(query: str)`
- `get_weather(city: str, date: str)`
- `send_email(to: str, subject: str, body: str)`

This enables LLMs to take actions and access real-time information. We'll cover this in depth in Unit 5.

### Context Window

LLMs can only "see" a limited number of tokens at once.

> **Context limit:** the maximum number of tokens the model can process.
> (Remember: attention is $O(n^2)$!)

**Context limits by model:**

| Model | Context window |
|-------|---------------|
| GPT-3 | 4,096 tokens |
| GPT-4 | 8,192 or 128,000 tokens |
| Claude 3 | 200,000 tokens |
| Gemini 1.5 | 1,000,000+ tokens |

Context window = prompt tokens + generated tokens. If you exceed it, the model cannot see the beginning of the conversation.

#### Hard limits vs. soft limits

- **Hard limit:** exceeding causes errors or truncation
- **Soft limit:** model performance degrades as context fills up

#### The "Lost in the Middle" problem

Models attend most strongly to the **beginning** and **end** of context. Information in the middle of long contexts may be "forgotten."

```
Recall
  │
  │  ●                           ●   ← High
  │   \                         /
  │    \                       /
  │     \_____________________/
  │              ↑
  │          Low (middle)
  └─────────────────────────────── Position in context
       Start      Middle       End
```

**Practical implications:**
- Place critical info at start or end of prompts
- In RAG: order retrieved docs strategically
- Shorter contexts often outperform longer ones

> Liu et al. (2023), "Lost in the Middle: How Language Models Use Long
> Contexts"

Just because you *can* use 128K tokens doesn't mean you *should*. Shorter prompts are faster and often more accurate.

### LLM "API" Summary

- LLMs are functions: `tokens_in` → `next_token_distribution` →
  `sampled_next_token`
- Generation is autoregressive: one token at a time, no backtracking
- **Temperature** controls randomness (0 = deterministic, >1 = creative)
- **Top-k / top-p** filter unlikely tokens before sampling
- **Structured outputs** (JSON, function calls) enable integration with software
- **Context windows** limit how much text the model can process

Understanding the generation process helps explain LLM behavior—why they're sometimes repetitive, why they can't "unsay" things, and why the first token takes longest.

---

## Part 3: From Language Model to Assistant

A raw language model just predicts text. How do we get a helpful assistant?

```
Pre-training  →  Instruction Tuning  →  RLHF
```

1. **Pre-training:** learn language from massive internet text
2. **Instruction Tuning:** learn to follow instructions
3. **RLHF:** align with human preferences (helpful, harmless, honest)

> Pre-training gives the model *knowledge*. Post-training gives it
> *behavior*.

### Pre-training: Learning Language

**Goal:** learn general language understanding from massive unlabeled data.

**The process:**
- Collect enormous amounts of text (web, books, code, etc.)
- Train the model to predict the next token
- That's it—no labels, no human annotation

**Scale:**

| Model | Training tokens |
|-------|----------------|
| GPT-3 | 300 billion |
| LLaMA 2 | 2 trillion |
| Modern models | 10+ trillion |

**Cost:** millions of dollars in compute. This is why few organizations train from scratch.

#### Pre-training data: what goes in

LLMs learn from a diverse mix of internet-scale text:

| Source | Share (LLaMA example) |
|--------|----------------------|
| Common Crawl | 67% |
| C4 (cleaned web) | 15% |
| GitHub | 4.5% |
| Wikipedia | 4.5% |
| Books | 4.5% |
| ArXiv | 2.5% |
| StackExchange | 2% |

> **Data quality matters.** Significant effort goes into filtering:
> removing duplicates, toxic content, low-quality pages, and personal
> information. "Garbage in, garbage out" applies at scale.

#### What pre-training produces

A pre-trained model is a **text completion engine**, not an assistant.

If you prompt a base model with:
> "What is the capital of France?"

You might get:
> "What is the capital of Germany? What is the capital of Spain? What
> is the capital of…"

The model learned that questions are often followed by more questions (from web data like FAQ pages). Pre-training creates a powerful model, but it doesn't know it should *answer* questions rather than continue them.

### Instruction Tuning: Learning to Follow Instructions

**Goal:** teach the model to respond helpfully to instructions.

**Method:** fine-tune on (instruction, response) pairs.

**Example training data:**
- **Instruction:** "Summarize this article in 3 bullet points"
- **Input:** [article text]
- **Output:** [3 bullet points]

**Key datasets:**
- FLAN (Google): 1,800+ tasks
- Natural Instructions: 1,600+ tasks
- Self-Instruct: synthetic instruction generation

#### The effect of instruction tuning

**Before (base model):**
> User: What is the capital of France?
> Model: What is the capital of Germany? What is the capital of…

**After (instruction-tuned):**
> User: What is the capital of France?
> Model: The capital of France is Paris.

> Instruction tuning doesn't add new knowledge—the model already "knew"
> that Paris is the capital of France from pre-training. It just learns
> the *format* of helpful responses.

### RLHF: Aligning with Human Preferences

**Goal:** make the model useful.

**The problem with instruction tuning alone:**
- Model follows instructions, but may be verbose, evasive, or unhelpful
- Hard to capture "good response" with examples alone
- Humans can more easily *compare* responses than *write* perfect ones

**RLHF (Reinforcement Learning from Human Feedback):**
1. Generate multiple responses to a prompt
2. Humans rank which responses they prefer
3. Train a "reward model" to predict human preferences
4. Fine-tune the LLM to maximize the reward model's score

#### The RLHF process

Three stages:

| Stage | What happens | Result |
|-------|-------------|--------|
| **SFT** | Train on human-written responses | Follows instructions |
| **Reward model** | Humans compare pairs of responses; train a model to predict preferences | Captures "what humans prefer" |
| **RL fine-tuning** | Generate responses, score with reward model, update LLM to produce higher-scoring responses | Helpful and aligned |

#### The effect of RLHF

**Before RLHF:**
- Technically correct but verbose or unhelpful
- May generate harmful content if asked
- Doesn't clearly refuse inappropriate requests

**After RLHF:**
- More concise and directly helpful
- Refuses harmful requests politely
- Acknowledges uncertainty rather than hallucinating confidently

> GPT-3 existed for years before ChatGPT. The breakthrough was RLHF
> making it actually *pleasant to use*.

### In-Context Learning

LLMs can learn from examples in the prompt—no gradient updates required.

**Zero-shot:**
> Translate to French: "Hello, how are you?"

**Few-shot:**
> English: Hello → French: Bonjour
> English: Thank you → French: Merci
> English: Goodbye → French: ?

This is one of the most surprising emergent capabilities of large language models.

### Chain-of-Thought Prompting

**Idea:** ask the model to show its reasoning step by step.

**Standard prompt:**
> Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many?
> A: 11

**Chain-of-thought prompt:**
> Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many?
> A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls.
> 5 + 6 = 11.

**Simple trick:** add "Let's think step by step" to your prompts. Dramatically improves performance on reasoning tasks.

### Training Pipeline Summary

| Stage | What It Does | Result |
|-------|-------------|--------|
| Pre-training | Next-token prediction on web text | Knows language and facts |
| Instruction Tuning | Fine-tune on (instruction, response) pairs | Follows instructions |
| RLHF | Optimize for human preferences | Helpful and aligned |

> An LLM assistant is built in layers:
> - Pre-training: *what* to say (knowledge)
> - Post-training: *how* to say it (behavior)

### Optional: Task-Specific Fine-Tuning

After post-training, you can **further fine-tune** for specific use cases:

**Why fine-tune?**
- Adapt to a specific domain (legal, medical, finance)
- Learn a particular style or format
- Improve performance on narrow tasks
- Reduce prompting costs (bake instructions into weights)

**Common approaches:**

| Approach | Description |
|----------|------------|
| **Full fine-tuning** | Update all model weights (expensive) |
| **LoRA / QLoRA** | Update small adapter layers (efficient) |
| **Prompt tuning** | Learn soft prompts (very efficient) |

Fine-tuning requires curated data and compute, but can significantly improve quality and reduce inference costs for specific applications. We'll explore this more in a later lecture.

---

## Part 4: Practical Considerations

### What LLMs Are Good At

LLMs excel at "text problems":

| Text and Content | Code and Analysis |
|-----------------|-------------------|
| Summarization | Code explanation |
| Translation | Code generation |
| Style transfer | Bug identification |
| Content generation | Data extraction |
| Grammar correction | Classification |

**Key insight:** if the task can be framed as "produce plausible text given context," LLMs are often effective.

### What LLMs Are Bad At

LLMs struggle with "precision problems":

- **Arithmetic:** cannot reliably compute $347 \times 892$
- **Counting:** how many r's in "strawberry"?
- **Exact retrieval:** what was said in paragraph 47?
- **Logical consistency:** complex multi-step deduction
- **Determinism:** same prompt can give different outputs

> LLMs predict *plausible* tokens, not *correct* answers. They
> pattern-match rather than compute.

### Hallucination

**Definition:** generating plausible-sounding but factually incorrect information.

**Types:**
- **Factual hallucination:** made-up facts, dates, citations
- **Fabrication:** inventing entities that don't exist
- **Conflation:** mixing up similar but distinct things

**Why it happens:**
- Models optimize for plausibility, not truth
- Training data contains errors
- Models cannot reliably say "I don't know"

**Mitigation:** RAG (retrieval), citations, verification, human review

### Other Common Limitations

**Knowledge limitations:**
- Training data cutoff—no knowledge of recent events
- Cannot access external data unless provided in prompt
- User-specific context unknown unless in prompt

**Instruction following:**
- May ignore parts of complex instructions
- Struggles with negation ("don't mention X")
- Can be verbose when brevity requested

**Reasoning failures:**
- Correct-looking but wrong reasoning chains
- Struggles with novel problem structures
- Overconfident in incorrect answers

### When NOT to Use LLMs

In P³ terms, this is a **Promise** question: if your acceptance criteria include determinism, exact math, or sub-millisecond latency, an LLM fails the promise before you ever reach Proof.

**Avoid LLMs when the problem:**

1. **Requires deterministic execution** — same input must always produce same output
2. **Is latency-sensitive** — LLM calls take 100ms–10s; too slow for real-time
3. **Requires mathematical precision** — use a calculator, not an LLM
4. **Needs exact data integrity** — don't trust LLMs for financial calculations
5. **Involves sensitive data across API boundaries** — PII, credentials, proprietary information

**Also avoid when:**
- Simple text transformation would suffice (regex, templating)
- Basic data validation (type checking, schema validation)
- An existing API does it better
- Cost-inefficient for the task (high volume + simple task)

> **Rule of thumb:** if a non-ML solution exists and works reliably, use
> it. LLMs add complexity, cost, and non-determinism.

### Cost Structure

LLM APIs charge per token:

| Component | Notes |
|-----------|-------|
| **Input tokens** | Usually cheaper (you send once) |
| **Output tokens** | Usually more expensive (model computes) |
| **Cached prompts** | Often discounted (reused prefixes) |

**Example pricing (GPT-4o, 2024):**
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens
- Cached input: $1.25 / 1M tokens

**Implication:** minimize both prompt length *and* output length for cost efficiency.

### Latency Sources

| Source | Description |
|--------|------------|
| **Network latency** | Round-trip to API servers |
| **Queue time** | Waiting for available GPU capacity |
| **Prompt processing** | Linear in input token count |
| **Token generation** | Linear in output token count |

**Typical latencies:**
- Time-to-first-token: 200ms–2s
- Token generation: 20–100 tokens/second
- Total: 500ms–30s depending on task

---

## Part 5: Beyond the LLM — Compound AI Systems

Real-world AI applications rarely use an LLM alone. Modern systems combine LLMs with other components:

- **Retrieval:** fetch relevant documents from a knowledge base
- **Tools:** execute code, call APIs, search the web
- **Memory:** persist information across conversations
- **Guardrails:** filter inputs/outputs for safety
- **Orchestration:** route between multiple models or agents

> **Compound AI systems** combine multiple AI components (including LLMs)
> with traditional software to solve complex tasks. This is where the
> field is heading.

### Why Compound Systems?

Through the P³ lens: each component below exists because a bare LLM can't fulfill the full **Promise**. Retrieval closes the proof gap on hallucination, tools close the proof gap on "can't act," and guardrails are a **Production** requirement for safety durability.

| LLM-only problem | Compound system solution |
|-----------------|------------------------|
| Knowledge cutoff | Retrieval (RAG) |
| Hallucination | Grounding + citations |
| No persistent memory | External memory stores |
| Can't take actions | Tool use / function calling |
| Expensive for repeated work | Caching + smaller models |

The LLM acts as the "brain"—reasoning over retrieved context, deciding which tools to call, and maintaining coherent conversations.

### Key Patterns We'll Explore

1. **Retrieval-Augmented Generation (RAG):** ground LLM responses in your own data; reduce hallucination with retrieved evidence
2. **Tool Use and Agents:** LLMs that can take actions in the world; multi-step reasoning with external feedback
3. **Evaluation and Observability:** how do you know if your system works? Monitoring, logging, and continuous improvement
4. **Production Deployment:** scaling, caching, cost optimization; safety, security, and reliability

> Building effective AI systems is now a **systems engineering**
> problem, not just a modeling problem. Understanding LLM foundations
> (today) enables building these systems (rest of course).

---

## Key Takeaways

1. **Language modeling = next-token prediction.** Everything else emerges from this simple operation at scale.
2. **The LLM "API":** text in → probability distribution → sample next token. Generation is autoregressive, one token at a time.
3. **Pre-training gives knowledge; post-training gives behavior.** Instruction tuning + RLHF turn a text predictor into an assistant.
4. **LLMs predict plausible text, not correct answers.** Great for text problems; hallucinate on precision problems.
5. **Know when NOT to use LLMs.** Simpler solutions are often more reliable and cost-effective.

---

## In-Class Exercise

Go to `chat.openai.com`, `claude.ai`, or `gemini.google.com` and try:

1. **Counting:** "How many r's are in the word strawberry?"
2. **Recent events:** ask about something from the past week
3. **Math:** "What is 347 × 892?" (check with a calculator)
4. **Generation:** "Write a limerick about machine learning"
5. **Reasoning:** "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles?"

What does the model do well? Where does it struggle? These observations illustrate the strengths and limitations we discussed.

---

## Further Reading

**Foundational:**
- Radford et al. (2018, 2019) — GPT, GPT-2
- Brown et al. (2020) — GPT-3 ("Language Models are Few-Shot Learners")

**Training and Alignment:**
- Ouyang et al. (2022) — InstructGPT / RLHF
- Wei et al. (2022) — Chain-of-Thought Prompting

**Practical:**
- OpenAI Cookbook — [cookbook.openai.com](https://cookbook.openai.com)
- Anthropic's Claude Documentation —
  [docs.anthropic.com](https://docs.anthropic.com)
