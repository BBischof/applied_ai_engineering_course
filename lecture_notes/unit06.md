# Unit 6: Retrieval Fundamentals

**Date:** Wednesday, February 25, 2026

LLMs are powerful reasoners, but they only know what was in their training data. This lecture covers *why* retrieval matters for AI systems and *how* search actually works — from classical lexical matching to semantic embeddings to hybrid strategies to agents that write their own queries.

| Part | Core Question |
|------|---------------|
| **Why Retrieval?** | What are the knowledge gaps in LLMs, and how does RAG fill them? |
| **What Is Search?** | What are the core abstractions of a search system? |
| **Lexical Search** | How does text-matching search work (TF-IDF, BM25)? |
| **Semantic Search** | How does meaning-based search work (embeddings, ANN)? |
| **Hybrid Search** | How do we combine lexical and semantic results? |
| **Agentic Search** | What happens when an LLM writes its own search queries? |
| **Non-Query Signals** | What ranking signals exist beyond the query itself? |

---

## Part 1: Why Retrieval?

### What Does an LLM Actually "Know"?

Everything an LLM can do comes from its **weights** — parameters learned during training on a fixed snapshot of data. This creates two fundamental gaps:

1. **Temporal gap:** the training data has a **cutoff date**. Anything that happened after that date is invisible to the model.
2. **Access gap:** private, proprietary, or domain-specific data was never in the training set to begin with.

```
  Training Data (internet, books, code, …)
       │
       ▼
  Model Weights (frozen knowledge)
       ▲
       ╎ inaccessible
  Yesterday's news? Your company data?
```

### The Knowledge Cutoff Problem

```
──────────────────────────────────────────────────────► Time

 ██████████████████████████│
  Training data window     │Cutoff
                           │
  Historical  Known        │  Recent     Today
  events      facts        │  events     (unknown)
```

**The model can answer:**
- "What were the key features of GPT-4?"
- "Explain the PageRank algorithm."

**The model cannot answer:**
- "What did the Fed announce yesterday?"
- "What's in our Q4 revenue report?"

Worse: the model doesn't know what it doesn't know. It may **hallucinate** a plausible-sounding but wrong answer.

### The Private Data Problem

Even within the training window, vast amounts of information were **never available** to the model.

**Examples of private data:**
- Internal company documents
- Customer records and CRM data
- Proprietary codebases
- Internal wikis and runbooks
- Private Slack / email threads
- Data behind authentication walls

**Why this matters:**
- Most high-value enterprise use cases depend on *private* data
- You can't (and shouldn't) retrain a foundation model on your internal data every week
- Fine-tuning helps with *style* and *behavior*, but is a poor mechanism for injecting *facts*

> If we want LLMs to be useful with real-world, up-to-date, and
> private information, we need a way to get that information **into
> the prompt at inference time**.

### The Idea: Retrieval-Augmented Generation (RAG)

> Don't change the model. Change what you **show** it.

**RAG** is a simple but powerful pattern:

1. User asks a question
2. **Retrieve** relevant documents from an external knowledge source
3. **Augment** the LLM's prompt with those documents
4. **Generate** a response grounded in the retrieved context

```
User       Retrieval        LLM          Grounded
Question → System     →    (question     Answer
           ↑               + docs) →
        Knowledge
          Base
```

### Why RAG Instead of Other Approaches?

| Alternative | Pros | Cons | Verdict |
|------------|------|------|---------|
| **Retrain the model** | Freshest knowledge | Costs millions, takes weeks, stale again immediately | Impractical for most orgs |
| **Fine-tune** | Good for teaching style and behavior | Poor at reliably injecting specific facts; needs periodic re-runs | Complementary to RAG, not a replacement |
| **Stuff everything in the prompt** | Simple | Context windows are finite; cost scales linearly; needle-in-haystack problems | Works for small KBs; doesn't scale |
| **RAG** | No model changes; knowledge updated in real time; scales to millions of docs; only relevant context enters the prompt | Requires building a retrieval system | The practical default for knowledge-grounded LLM apps |

This table is a P³ **Production** decision: you choose the approach that lets you prove the promise most efficiently.

### What RAG Buys You

1. **Grounding and factual accuracy** — the model answers from retrieved evidence, not from memory; dramatically reduces hallucination on factual queries
2. **Up-to-date information** — the knowledge base can be updated continuously; no retraining or fine-tuning required
3. **Access to private data** — connect LLMs to internal documents, databases, and APIs; data never leaves your infrastructure
4. **Auditability** — you can inspect *which* documents were retrieved; enables citation and source attribution; makes debugging and trust-building possible

> Think of RAG as giving the LLM an **open-book exam** instead of a
> closed-book one. The model is still doing the reasoning — but now
> it has the right reference material in front of it.

### A Concrete Example

**Scenario:** a support engineer asks an internal AI assistant about an outage.

**Without RAG:**
> **Q:** "What caused the payment service outage on Feb 20?"
>
> **A:** "Payment service outages are commonly caused by database
> connection pool exhaustion, network timeouts, or deployment
> failures. I'd recommend checking your monitoring dashboards…"
>
> *Generic, not grounded in actual incident data.*

**With RAG:**
> **Q:** "What caused the payment service outage on Feb 20?"
>
> *Retrieved: incident report INC-4821, PagerDuty alert #7392*
>
> **A:** "Per INC-4821, the Feb 20 outage was caused by a
> misconfigured rate limiter deployed at 14:32 UTC. The fix was
> rolled back at 15:10 UTC by the payments-oncall. Full RCA is
> linked in the incident."
>
> *Specific, grounded, citable.*

The model is the same in both cases. The difference is what information it had access to.

### The Retrieval Bottleneck

RAG shifts the hard problem from *model training* to **information retrieval**. The quality of a RAG system is **bottlenecked by retrieval quality**. If you retrieve the wrong documents, the LLM will confidently synthesize the wrong answer. In P³ terms: your **Promise** is grounded answers over private data, but that promise can only be **Proven** if retrieval works — precision@k, recall@k, and relevance judgments are the proof instruments here. These metrics need baselines (Unit 1) — a BM25-only baseline, a random-document baseline — to quantify how much value your retrieval pipeline actually adds.

```
Ingest &       Retrieve          Generate
Index Data  →  Relevant Docs  →  Response
└─────────────────────┘
  This is the search & retrieval problem
```

---

## Part 2: What Is Search?

> "To organize the world's information and make it universally
> accessible and useful." — Google's 2001 mission statement

A search system processes a **query** and returns an ordered (possibly empty) collection of **documents**.

- **Documents:** the objects we are interested in finding. Information is assumed to be present in documents.
- **Queries:** expressions of intent that the search system must interpret.

### Documents: The Core Abstraction

Most search systems assume objects are represented by a **Document** type — typically groups of named key–value **fields**. Documents have **types**, and we store similar-typed documents in **collections**.

```python
@dataclass
class Document:
    id: UUID
    author_id: UUID
    name: str                  # indexed for lexical search
    description: str | None    # indexed for lexical search
    created_at: datetime       # useful for filtering
    endorsed: bool             # useful for filtering
```

### Queries: From Unstructured to Structured

**Typically**, to indicate their intent, a user issues an "unstructured query" — a string literal propagated into the search system. But queries don't *have* to be single strings.

**Simple (unstructured) query:**

```
"find me all blubs"
```

**Structured query:**

```json
{
  "query": {
    "match": {
      "object_type": "blub"
    }
  }
}
```

The order of returned documents is determined by a **ranking function** — the heart of any search system.

#### A few kinds of structured queries

| Query type | Description | Index used |
|-----------|-------------|-----------|
| **Basic Match** | Searches for a term within a specific field | Lexical index |
| **Exact Phrase** | Searches for an exact phrase, maintaining term order | Lexical index |
| **Range Query** | Filters on numeric or date fields | Numeric / BTree index |

---

## Part 3: Lexical Search

> Lexical search uses the **text content** of queries to identify
> documents with similar **text**.

### The Full Pipeline

```
Offline:
  Raw Documents → Pre-process (lowercase, stem, …) → Tokenize → Build Inverted Index
                                                                       │
  ──────────────────────────────────────────────────────────────────────┤
                                                                       │
Online:                                                                ▼
  User Query → Pre-process + Tokenize → Index Lookup → Rank (BM25) → Ranked Results
```

### Document Pre-Processing

| Step | What it does | When to skip |
|------|-------------|-------------|
| **Lowercasing** | Transform fields to all lowercase | If capitalization disambiguates documents |
| **Punctuation removal** | Replace punctuation with whitespace | If punctuation is meaningful (e.g. code) |
| **Stemming** | Convert words to root form ("running" → "run") | If exact morphology matters |
| **Stop word removal** | Remove filler words: "a", "the", "of", … | If filler words carry meaning in your domain |

> All of these transformations are **optional**. The right choices
> depend on your corpus. Poor preprocessing is a top cause of missing
> search results.

### Document Tokenization

| Level | How it splits | Example |
|-------|-------------|---------|
| **Word** | Split on whitespace | "hello world" → ["hello", "world"] |
| **Character** | Split at character boundaries | "cat" → ["c", "a", "t"] |
| **Subword** | Split inside words | "unhappiness" → ["un", "happiness"] |
| **N-gram** | Sequences of $n$ characters | "unhappiness" (n=4) → ["unha", "nhap", "happ", …, "ness"] |

Note: search tokenization is related to but *not the same as* LLM tokenization.

### Worked Example: Pre-Processing in Action

**Document:**

```json
{"id": "ff1f...",
 "name": "Barry's Forecast of Revenue",
 "description": "Sometimes you really want to know how much money
  you have. Sometimes you need to know how much money you WILL have."}
```

**"name" field config:** lowercase, remove punctuation, tokenize with 5-gram (character)

```
Preprocessed: barrys forecast of revenue
Tokens: ["barry","arrys","rrys ","rys f", …, "venue"]
```

**"description" field config:** lowercase, remove punctuation, remove stopwords, tokenize at word boundaries

```
Preprocessed: sometimes really want know much money sometimes need
              know much money
Tokens: ["sometimes","really","want","know","much","money",…]
```

### From Tokens to Sparse Vectors

After processing and tokenizing each field, we collect all tokens seen for that field into a **universal vocabulary**. Each document field becomes a **sparse vector** where each vocabulary element is a dimension, and the value is the token's count in that field.

**Vocabulary:** ["once", "sometimes", "money", "know", "since", "really", "need", "much", "want", "forever", "cold", …]

```
doc1 description: [0, 2, 2, 2, 0, 1, 1, 2, 1, 0, 0, …]
doc2 description: [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, …]
```

### Indexing: The Inverted Index

The sparse vectors above are *row-oriented*: one vector per document. An **inverted index** (also called a **posting list**) is the *column-oriented* view — a mapping from each token to the set of documents that contain it.

```
once       → [doc2]
sometimes  → [doc1]
money      → [doc1, doc2]
know       → [doc1, doc2]
really     → [doc1]
```

**Why an inverted index?**
- Given a query token, instantly look up which documents contain it
- No need to scan every document at query time
- This is the data structure behind most text search engines (Lucene, Elasticsearch, etc.)

### Querying with Unstructured Queries

At query time, we apply the **same pre-processing steps** for each field we are searching over.

**Raw query:** "barry cold money forecast"

```
Query for "name" field (5-gram):  ["barry ","arry ","rry c",…,"ecast"]
Query for "desc" field (word):    ["barry","cold","money","forecast"]
```

Query terms that aren't in the index contribute zero to the ranking score — the document can't match on a term it never contained.

### Ranking: TF-IDF

TF-IDF is a simple ranking function: the product of **term frequency** and **inverse document frequency**.

**Definitions:**
- $\mathrm{TF}(t, d)$ = number of times term $t$ appears in document
  $d$
- $\mathrm{DF}(t)$ = number of documents containing $t$
- $\mathrm{IDF}(t) = \log\!\bigl(\frac{N}{1 + \mathrm{DF}(t)}\bigr)$
  where $N$ = total documents

**Scoring:** for each query term $t$ and document $d$:

$$\mathrm{TFIDF}(t, d) = \mathrm{TF}(t, d) \times \mathrm{IDF}(t)$$

Sum over all query terms to get the document's total score.

> Terms that appear frequently *in a document* but rarely *across the
> corpus* are the most useful for distinguishing relevant documents.
> IDF can be **pre-computed**.

#### TF-IDF worked example

Query tokens: ["barry", "cold", "money", "forecast"]  ($N = 2$ documents)

**doc1 scores:**
- barry: TF=0 ⇒ 0
- cold: TF=0 ⇒ 0
- money: TF=2, IDF=log(2/3)=−0.176 ⇒ −0.352
- forecast: TF=0 ⇒ 0
- **Total: −0.352**

**doc2 scores:**
- barry: TF=0 ⇒ 0
- cold: TF=1, IDF=log(2/2)=0 ⇒ 0
- money: TF=1, IDF=−0.176 ⇒ −0.176
- forecast: TF=0 ⇒ 0
- **Total: −0.176** ← better match

### Problems with TF-IDF

1. **Term frequency saturation:** TF-IDF grows unboundedly with term frequency — a term appearing 100 times has disproportionate weight vs. 10 times, even though relevance plateaus in practice
2. **Document length bias:** doesn't properly account for document length; penalizes longer documents that naturally repeat terms
3. **No tunable parameters:** rigid formula with no way to adjust saturation or length normalization
4. **Poor handling of common terms:** IDF doesn't sufficiently dampen very common terms
5. **Linear TF scaling:** doesn't match actual relevance patterns

### Ranking: BM25

**BM25** is the modern successor to TF-IDF, addressing its key weaknesses:

$$\mathrm{BM25}(t, d) = \mathrm{IDF}(t) \times \frac{\mathrm{TF}(t,d) \cdot (k_1 + 1)}{\mathrm{TF}(t,d) + k_1 \cdot \bigl(1 - b + b \cdot \frac{|d|}{\mathrm{avgdl}}\bigr)}$$

**Key improvements over TF-IDF:**
- **Saturation curve:** diminishing returns after ~5–10 occurrences
- **Length normalization:** adjustable penalty for long documents
- **Tunable parameters**

**Parameters:**
- $k_1$: saturation parameter (typically 1.2–2.0)
- $b$: length normalization (typically 0.75)
- $\mathrm{avgdl}$: average document length in the corpus

```
Score
  │
  │          TF-IDF (linear, unbounded)
  │        /
  │      /      BM25 (saturates at k₁+1)
  │    /    ────────────────────────────
  │  / ──/
  │/──
  └──────────────────────────────── Term Frequency
```

- **TF-IDF:** score grows linearly — the 100th occurrence of a term is weighted 10× more than the 10th
- **BM25:** score saturates — after ~5–10 occurrences, additional matches barely increase the score

> BM25 is the default ranking function in Elasticsearch, Solr, and
> most modern search engines. If you're doing lexical search, you're
> almost certainly using BM25.

### The "API" of Lexical Search

We can think of lexical search at three levels of abstraction:

```python
# 1. Single field
def lexical_search(
    query: str, field: str, doc_type: DocType
) -> list[tuple[DocId, float]]: ...

# 2. All text fields (combine via "boost" weights)
def lexical_search(
    query: str, doc_type: DocType
) -> list[tuple[DocId, float]]: ...

# 3. All document types (hard — BM25 scores aren't comparable
#    across corpora; use z-score or min-max normalization)
def lexical_search(
    query: str
) -> list[tuple[DocId, float]]: ...
```

---

## Part 4: Semantic Search

> Semantic search uses the **meaning** of queries to search against
> the **meaning** of documents.

**Example:**
- **Query:** "I'm growing a forest and only want tall trees, find me those"
- **doc1:** {name: "Opera Singers throughout History", content: "many opera singers existed…"}
- **doc2:** {name: "Oaks and Pines", content: "these fine plants are very large"}
- **Intuition:** doc2 should match and rank higher — even though "tall trees" appears in neither document. Lexical search would struggle here; semantic search understands meaning.

### The Key Idea: Embedding Space

Imagine a very high-dimensional vector space of "ideas."

1. Turn documents into **vectors** in this space using an **encoder**
2. At query time, encode the query into the same space
3. Find documents whose vectors are **nearest neighbors** to the query vector

In practice, the encoder is a **transformer language model** that accepts strings as input and produces normalized vectors as output. Typical dimensionality: ~1024.

> Document vectors can be **pre-computed** offline. At query time,
> only one embedding call is needed — for the query string.

### Indexing for Semantic Search

**Storage:** $D$ documents × $\mathrm{DIM}$ dimensions × 4 bytes (f32) = total memory.

For 1M documents at 1024 dims: $1{,}000{,}000 \times 1024 \times 4 =$ **4 GB**.

**The core operation:** compute **nearest document neighbors** for a query vector.

**Distance metrics:** Euclidean, cosine similarity, inner product.

**Naïve approach:** compare the query to *every* document vector, then sort. This is expensive at scale.

> **Solution: Approximate Nearest Neighbors (ANN).** Build an index
> structure on top of the raw vectors that quickly identifies
> "potentially good" neighbors. Trades a small amount of recall for
> large speedups.

### Vector ANN Indices

An **ANN index** creates structure on top of raw vectors for fast approximate neighbor lookup.

**Common index structures:**
- **HNSW** (Hierarchical Navigable Small World) — graph-based, very popular
- **IVF** (Inverted File Index) — partition-based
- **Product Quantization** — compression-based

**Trade-off:** there is a systematic chance of missing the true best neighbors (controlled by a recall parameter), but this allows massive query acceleration.

**Popular vector databases:**
- LanceDB, Pinecone, Weaviate, Qdrant, Milvus, pgvector (Postgres)

### HNSW: Intuition

**Key idea:** a multi-layer graph where each layer is a "navigable small world."

```
Layer 2 (sparse):  ●─────────●─────────●
                              │
Layer 1 (medium):  ●───●───●───●───●
                                   │
Layer 0 (dense):   ●─●─●─●─●─●─●─●─●─●
                                       ▲
                                     query
```

1. **Top layers** are sparse with long-range edges for coarse navigation
2. **Bottom layers** are dense with short-range edges for fine search
3. Search enters at the top and greedily descends, narrowing in on the nearest neighbor

Like a skip list, but for high-dimensional vector space.

### Turning Documents into Strings for Embedding

A key detail: how do we turn rich, structured objects into **single strings** for the embedding model?

This is document-specific. Different document types need different strategies:
- **Simple documents:** concatenate key text fields (name + description)
- **Tabular data:** serialize column names and sample values
- **Complex documents (e.g. notebooks):** may need special treatment

#### HyDE (Hypothetical Document Embedding)

Instead of embedding raw content, feed it to an LLM and ask: "What questions could this document answer?" Then embed *those questions* instead of the raw content.

This aligns the document's embedding with the kinds of queries users will actually issue.

**Example HyDE prompt:**

```
You are a data science manager with a deep understanding of
Python and SQL. Your task: given a project's metadata and
contents, generate questions that the project COULD ANSWER.

Respond as a JSON array of strings.

Example questions for a "Customer Growth" project:
1. How many customers do we have this year?
2. What was our customer growth over the last year?
3. How many new customers did we add per month?
4. What is the average customer acquisition cost per campaign?
5. What are the key factors influencing customer retention?

Here is the project status: {STATUS}
Here are the project categories: {CATEGORIES}
Here are the project contents: {CELL_CONTENTS}

Generate five questions. Each question: concise, max two sentences.
```

The generated questions are then embedded by the embedding model — *not* the raw cell contents.

---

## Part 5: Hybrid Search

> Hybrid search combines **lexical** and **semantic** search results,
> using the strengths of both.

**The setup:** run both search strategies independently:

```
lexical_results:  list[tuple[DocId, float]]
semantic_results: list[tuple[DocId, float]]
```

**The problem:** scores are **incomparable** — BM25 scores have no bounded range; cosine similarities are in $[0, 1]$. We need a **score-agnostic** merge strategy.

### Option 1: Set Union (When Order Doesn't Matter)

The simplest approach — just merge the sets of document IDs:

```python
def merge_results(
    v1: list[tuple[DocId, float]],
    v2: list[tuple[DocId, float]],
) -> set[DocId]:
    return {doc_id for doc_id, _ in v1} | {doc_id for doc_id, _ in v2}
```

Result: at most `limit_lexical + limit_semantic` documents (with deduplication).

> In the agentic world, it's actually unclear whether presentation
> order of search results matters at all — the agent processes all of
> them regardless.

### Option 2: Interleaving (Reciprocal Rank Fusion)

If order matters, you need to **interleave** the ranked lists — merge them in a way that respects rank position without comparing raw scores. The most common interleaving formula today is **Reciprocal Rank Fusion (RRF)**.

**Key idea:** documents near the top of *both* result lists are likely the best overall.

For each document $d$ appearing at rank $r$ in a result list:

$$\mathrm{RRF}(d) = \sum_{\text{list } \ell} \frac{1}{K + r_\ell(d)}$$

where $K = 60$ is a constant that works well empirically.

| DocId | Lexical Rank | Semantic Rank | RRF Score |
|-------|-------------|--------------|-----------|
| 10 | 1 | 2 | 1/61 + 1/62 = 0.0326 |
| 19 | 4 | 1 | 1/64 + 1/61 = 0.0320 |
| 17 | 5 | 4 | 1/65 + 1/64 = 0.0310 |

**Advantage:** not sensitive to the score distribution of either search method. You never have to compare a BM25 score to a cosine similarity — you only use rank positions.

### Option 3: Threshold-Based Interleaving (Score Fusion)

An alternative to rank-based interleaving: combine the actual *scores* from each search method using a weighted linear combination.

$$\mathrm{score}(d) = w_1 \cdot s_{\text{vector}}(d) + w_2 \cdot s_{\text{lexical}}(d)$$

The catch: the scores from different search methods live on **different scales**. Cosine similarity returns values in $[-1, 1]$ (or $[0, 1]$ for normalized embeddings). BM25 returns unbounded positive values that depend on corpus statistics — a BM25 score of 12.7 means nothing without knowing the corpus. You can't just add them.

**To make this work, normalize the scores first.** Common approaches:
- **Min-max normalization:** scale each method's scores to $[0, 1]$ using the min and max from the current result set
- **Z-score normalization:** subtract the mean and divide by standard deviation of each method's scores

After normalization, a weighted combination becomes meaningful:

```python
def threshold_interleave(
    vector_results: list[tuple[DocId, float]],
    lexical_results: list[tuple[DocId, float]],
    vector_weight: float = 0.7,
) -> list[tuple[DocId, float]]:
    # Normalize each to [0, 1]
    v_norm = min_max_normalize(vector_results)
    l_norm = min_max_normalize(lexical_results)

    # Weighted combination
    combined = {}
    for doc_id, score in v_norm:
        combined[doc_id] = vector_weight * score
    for doc_id, score in l_norm:
        combined[doc_id] = combined.get(doc_id, 0) + (1 - vector_weight) * score

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

**When to use threshold-based vs. rank-based:**
- **Threshold-based** is useful when the raw scores carry meaningful information about confidence — a document with cosine similarity 0.95 is probably more relevant than one at 0.72, and you want to preserve that distinction rather than collapsing both to "rank 1 vs. rank 2."
- **Rank-based (RRF)** is safer when you don't trust the score distributions — when one method returns many high-confidence results and the other returns a flat distribution, score fusion can be dominated by the overconfident method. RRF treats both methods equally regardless of their score shapes.

In practice, threshold-based interleaving gives you a tunable knob (the weight) that lets you bias toward one method — but the weight is corpus-specific. You need to tune it on your eval set (Unit 3's golden dataset), not guess.

### Option 4: Model-Based Reranking

For the highest quality ordering, use a **model-based reranker**. Cross-encoders are the most common type.

```
Lexical Search  ─┐
                  ├→ Merge candidates → Cross-Encoder Reranker → Top-k
Semantic Search ─┘
```

**How it works:**
- The reranker is a transformer that is **query-aware** — it performs full attention over query *and* document jointly
- API: `rerank(query: str, documents: list[str]) -> list[int]`
- Use lexical + semantic search for first-stage candidate retrieval, then pass all candidates through the reranker

> **Trade-off:** computationally expensive — can't run over the full
> corpus. Use it as a second stage on a small candidate set.

---

## Part 6: Agentic Search

> Equip an LLM agent with search tools and let it **call them in a
> loop** with different arguments.

```
User Query → LLM Agent ──rewritten queries──→ Search Tools
                  ▲                                │
                  └────────── results ─────────────┘
```

### Search in a Loop

This is fundamentally different from classical search.

- The agent takes an initial user query as input
- It can issue searches — hybrid, semantic, or lexical — **in parallel**
- Critically, the tool calls take **agent-provided search queries** as inputs

> The agent doesn't pass the original user query into the search
> tools — it gets to **rewrite its own queries** based on:
> - The original user query
> - Whatever it has learned during the course of its execution
> - Results from previous search iterations

This is how modern search agents work: the actual user query is not used directly. The agent decides what to search for.

### Agentic Search: Worked Example

**User:** "What were our top-performing product lines last quarter?"

**Turn 1:**
- Searches: `semantic("product line revenue Q3 2025")` |
  `lexical("quarterly performance")`
- Finds: Q3 revenue dashboard, product catalog

**Turn 2:**
- Searches: `semantic("Q3 2025 revenue breakdown by product category")`
- Finds: detailed breakdown report with per-line revenue

**Turn 3:**
- Agent synthesizes all retrieved documents and responds to the user.

> None of the search queries match the user's original phrasing. The
> agent **decomposed the intent** and issued targeted,
> domain-specific queries — refining based on what it found.

---

## Part 7: Non-Query Signals

> Even the best query isn't the whole picture.

The documents in a user's workspace carry **intrinsic signals** — metadata, structure, and usage patterns — that are valuable for ranking *independent* of any query.

| Signal | Description |
|--------|------------|
| **Recency** | Entities opened, executed, or modified more recently are more likely to be relevant |
| **Authority** | Entities with endorsed statuses, published versions, or stars are likely more relevant |
| **PageRank** | Entity importance as a graph eigenvalue property |
| **Personalization** | If a user consistently uses an entity, it may be conditionally more relevant to *them* |
| **Search history** | Repeated searches for the same things indicate importance |
| **Relational context** | Leverage entity relationships in a knowledge graph — "people also search for" |

---

## Discussion Questions

1. **From "find information" to "complete a task":** traditional search optimizes for returning relevant documents quickly. How should we rethink success metrics and product design when the goal shifts to task completion? What gets lost and gained?

2. **Value vs. compounding errors in agentic search:** agentic search can chain together multiple queries and synthesize across sources. When does this create value versus when does it introduce compounding errors? How do we design for transparency and control in multi-step flows? Apply error analysis (Unit 1) to each step: was the query decomposition wrong, did search return irrelevant results, or did the agent misinterpret findings? A failure-mode taxonomy (Unit 3) tells you where to invest.

3. **Shallow vs. deep:** agents could exhaustively research a topic, but users often just need a quick answer. How should we calibrate depth? Should it be user-controlled, context-dependent, or learned? What are the risks of getting this wrong?

---

## Key Takeaways

1. **LLMs have knowledge gaps** — training cutoffs and missing private data mean the model doesn't know everything you need it to
2. **RAG bridges these gaps at inference time** — retrieve relevant documents and include them in the prompt, no retraining required
3. **RAG beats the alternatives for factual grounding** — retraining is too expensive, fine-tuning is for style not facts, and stuffing everything in the context doesn't scale. Build a golden dataset (Unit 1) of query–document–answer triples — typical, long-tail, and adversarial queries — to measure whether retrieval improvements actually help.
4. **Search = documents + queries + ranking** — the core abstractions are simple; the details are where quality lives
5. **Lexical search matches on text; semantic search matches on meaning** — TF-IDF → BM25 for lexical; embedding models + ANN indices for semantic
6. **Hybrid search combines both for better coverage** — merge strategies: set union, interleaving (RRF), cross-encoder reranking
7. **Agentic search lets LLMs rewrite and iterate on queries** — the agent decides what to search for, not the user directly
8. **Non-query signals matter** — recency, authority, personalization, and graph structure all improve relevance

---

## Further Reading

**Foundational:**
- Robertson & Zaragoza (2009) — "The Probabilistic Relevance Framework: BM25 and Beyond"
- Karpukhin et al. (2020) — "Dense Passage Retrieval for Open-Domain Question Answering"
- Malkov & Yashunin (2020) — "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs" (HNSW)

**Hybrid and Reranking:**
- Cormack et al. (2009) — "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"
- Nogueira & Cho (2019) — "Passage Re-ranking with BERT"
- Gao et al. (2022) — "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)
