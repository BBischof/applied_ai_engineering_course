# Unit 7: Advanced Retrieval & RAG

**Date:** Wednesday, March 4, 2026

---

## Recap: Unit 6 — Retrieval Fundamentals

Before we go further, let's do a quick recap of last week's foundations. Everything today builds on those ideas.

### Why retrieval?

LLMs have two knowledge gaps: a **temporal gap** (training cutoff — nothing after that date exists) and an **access gap** (private, proprietary, or domain-specific data was never in training). RAG bridges both at inference time: retrieve relevant documents, put them in the prompt, generate a grounded answer. No retraining required.

We compared the alternatives — retraining (too expensive), fine- tuning (teaches style not facts), stuffing everything in the context (doesn't scale). RAG is the practical default.

### The search abstractions

A search system has three core abstractions:

- **Documents:** objects with named fields stored in collections. We represented them as dataclasses with typed fields.
- **Queries:** expressions of user intent — unstructured (a string) or structured (field-specific matches, range filters).
- **Ranking functions:** the heart of search — how you score and order results.

### Two search paradigms

**Lexical search (BM25):** match on text. Preprocess documents (lowercase, stem, remove stopwords), tokenize, build an inverted index (token → document list). At query time, apply the same preprocessing to the query, look up matching docs, score with BM25 (term frequency with saturation + length normalization). BM25 is 30 years old and still a strong baseline because it does no information compression — exact keyword matching.

**Semantic search:** match on meaning. Encode documents and queries as dense vectors via an embedding model, then find nearest neighbors in the vector space. Requires an Approximate Nearest Neighbor (ANN) index — Hierarchical Navigable Small World graphs (HNSW) is the most common. Handles synonyms and paraphrases that lexical search misses, but can miss exact-match requirements.

### Combining them

**Hybrid search** runs both in parallel and merges results. Four merge strategies:

1. **Set union** — just combine document IDs (simplest; fine when order doesn't matter to an agent, you just want to cover relevant documents)
2. **Rank-based interleaving** — merge ranked lists in a score-agnostic way, using only rank position rather than raw scores. The most common formula is Reciprocal Rank Fusion (RRF): $\mathrm{RRF}(d) = \sum_\ell \frac{1}{K + r_\ell(d)}$. Avoids the score-normalization problem entirely.
3. **Threshold-based interleaving** — combine results using a weighted linear combination of the normalized scores from each source (e.g. 0.7 × vector score + 0.3 × FTS score). Simple, fast, and tunable — but the weights are corpus-specific, so you need to tune them on your own eval set.
4. **Model-based reranking** — a second-stage model rescores a candidate set. Cross-encoders (process query + document jointly) are the most common, but other approaches include context-aware rerankers that consider the full result set, LLM-based rerankers (like Rank1, covered below), and lightweight learned scoring functions. More expensive than first-stage retrieval, so applied to a small candidate set.

### Beyond the query

So far, everything above assumes the user types a query and the system matches it against documents. But some of the most powerful retrieval techniques operate *around* the query — rewriting it, enriching the documents before the query ever arrives, or using signals that have nothing to do with query text at all.

**Agentic search:** the LLM rewrites and iterates on queries rather than passing the user's raw text to the search engine. The agent decomposes intent and issues targeted, domain-specific queries.

**HyDE (Hypothetical Document Embedding):** instead of embedding raw document content, generate "what questions could this document answer?" and embed those questions. Aligns document representations with likely user queries.

**Non-query signals:** recency, authority, personalization, search history, relational context — ranking signals that exist independent of the query text.

### The P³ connection

The **retrieval bottleneck** — RAG quality is limited by retrieval quality. If you retrieve the wrong documents or simply miss important ones, the LLM will confidently synthesize the wrong answer. _The easiest way to ensure a model will hallucinate is to ask it questions about resources it doesn't have access to_. In P³ terms: the Promise is grounded answers; the Proof requires measuring retrieval quality (precision@k, recall@k); Production requires baselines (BM25-only, random documents) to quantify how much value the retrieval pipeline adds.

> **Check yourself:** if any of the above feels unfamiliar, review
> `notes/unit-06-retrieval-fundamentals/lesson.md` before
> continuing. Everything below assumes this vocabulary.

---

## How this lecture connects to last week

Unit 6 was about **the parts.** You learned what lexical search is, what semantic search is, how to combine them, and what agentic query rewriting looks like. Each technique was presented on its own terms — here's how BM25 works, here's how embeddings work, here's how interleaving merges results.

Unit 7 is about **the system.** Now that you have the parts, the questions change:

- How do you **decide which parts to use** for a given problem?
- How do you **compose them** into a production architecture?
- How do you **measure** whether the composition is working — not just whether individual components are working?
- What happens when the obvious approach (**one embedding, cosine similarity**) fails — and what are the deeper reasons *why* it fails?
- What does retrieval look like at **real scale** — not 100 documents, but millions?

To organize this, we're going to borrow a framework from **recommendation systems** — a field that has been solving exactly this "compose parts into a system" problem for decades. If you've used Netflix, Spotify, or Amazon, you've been on the receiving end of this framework. Today you'll learn to build with it.

---

## This week's key idea

**RAG is a recommendation system.**

That claim needs unpacking — especially if you haven't built a recommender before. So let's start with a 60-second primer.

---

## What is a recommendation system?

A recommendation system solves a deceptively simple problem:

> Given a collection of things that may be recommended, choose an
> ordered few for the current context and user that best match
> according to some objective.

Netflix recommending movies. Spotify surfacing playlists. Amazon ranking products. Tiktok assembling a for-you page. At the heart of 4 out of 5 FAANG companies lies one or many recommendation systems.

The core data structure is the **user-item matrix** (*BPRS* Ch. 2) — a (usually huge, usually sparse) table where rows are users, columns are items, and entries are signals of preference: explicit ratings ("4 stars"), implicit signals (watched, clicked, purchased), or nothing at all. The fundamental question recommendation systems answer:

> Predict how much a user will like an item they've never seen.

Imagine five friends rating four cheeses on a 1–5 scale. Some cheeses don't get tried:

|  | Gouda | Chèvre | Emmentaler | Brie |
|---|---|---|---|---|
| **A** | 5 | 4 | 4 | 1 |
| **B** | 2 | 3 | 3 | 4.5 |
| **C** | 3 | 2 | 3 | 4 |
| **D** | 4 | 4 | 5 | ? |
| **E** | 3 | ? | ? | ? |

Most of the matrix is filled in, but D never tried Brie and E only tried Gouda. The core question of recommendation systems is: **can we predict the missing entries?** Would D like Brie? What should E try next?

A few natural questions emerge from this matrix:
- Which rows (users) are most similar to each other? A and D both rate Gouda and Chèvre highly — they might have similar taste.
- Which columns (items) are most similar? Chèvre and Emmentaler get nearly identical ratings from everyone — they might appeal to the same people.

These observations lead to two strategies:

- **User-user collaborative filtering:** similar *rows*. Users with similar taste will continue to have similar taste. D looks like A, and A hated Brie — so maybe D won't like Brie either.
- **Item-item collaborative filtering:** similar *columns*. Items with similar fans will continue to attract similar fans. E liked Gouda; Emmentaler gets similar ratings to Gouda — so recommend Emmentaler to E.

These two strategies — model the user (rows), or model the item (columns) — are the dual engines of every recommender (*BPRS* Ch. 2–3). They'll reappear throughout this lecture mapped onto RAG.

### The Collector → Ranker → Server architecture

Every recommendation system decomposes into three components (*BPRS* Ch. 0,1):

**Collector.** Knows what's in the catalog and the relevant features of each item. Manages availability and eligibility. Think of a waiter who checks what's still on the menu and knows the characteristics of each dish. Operates mostly *offline* (batch processing) with an *online* component (real-time updates).

**Ranker.** Takes the collection and orders items by relevance to the current user and context. This is where the model lives. The waiter ranks the desserts: mentions the most popular (banana creme pie), a contextual match for your tastes (pomegranate ice cream if you like pomegranate), and the safe bet (donut a la mode — most popular overall). Operates *online* at request time.

**Server.** Takes the ranked output, applies business logic (filtering, deduplication, diversity constraints, permissions), satisfies the response schema, and returns the final recommendations. The waiter presents a coherent set of 3 options with explanations — not a raw sorted list of 50 desserts.

Even the absolute simplest recommender — return a random available item — fits this framework. And the most sophisticated (e.g. Netflix's personalized homepage) is the same three components, scaled up.

### Why this matters for RAG

Now map it:

| Component | Recommendation system | RAG system |
|-----------|----------------------|-----------|
| **Collector** | Know the item catalog; precompute features; manage availability | Know the document corpus; precompute embeddings and indices; manage ingestion and freshness |
| **Ranker** | Score items for the current user via a model | Score documents for the current query via retrieval + reranking |
| **Server** | Apply business logic; satisfy the API schema; return K items | Apply permissions; format context for the LLM; respect token budgets; return grounded answer |

The user-item matrix becomes a **query-document relevance matrix** — equally sparse, equally the core question: *predict how relevant a document is to a query the system has never seen before.* Two CF strategies map directly:

- **User-user (query-side):** similar queries need similar documents. Agentic search (Unit 6) exploits this.
- **Item-item (document-side):** similar documents answer similar queries. HyDE exploits this — rewriting documents toward likely queries.

Once you see this mapping, the landscape of "advanced RAG techniques" will feel _almost_ obvious. Every technique we cover today is an improvement to the Collector, the Ranker, or the Server.

---

## A brief history: RAG is 6 years old — but really decades old

The term "RAG" was coined in 2020. The idea — augmenting a language model with retrieved external knowledge — is much older. The history matters because each milestone solved a specific problem you'll still encounter today, and each one maps to a C→R→S component.

### Era 1: The infrastructure (2009–2016)

**BM25** (Robertson & Zaragoza, 2009). The probabilistic relevance framework that still underpins Elasticsearch, Solr, and most production search. BM25 solved two problems TF-IDF couldn't: term-frequency saturation (the 100th mention of a word shouldn't count 100× more than the 1st) and document-length normalization. Thirty years later, it remains a strong baseline — and we now understand *why* (no information compression; see the pooling discussion below). **C→R→S: Ranker.**

**HNSW** (Malkov & Yashunin, 2016). Approximate nearest-neighbor search via hierarchical navigable small-world graphs. Made it practical to search millions of vectors in milliseconds. Now the dominant index algorithm in production vector databases (FAISS, Pinecone, Weaviate, etc.). Without HNSW, dense retrieval at scale wouldn't be feasible. **C→R→S: Collector infrastructure.**

### Era 2: Neural memory and retrieve-then-read (2014–2018)

**Memory Networks** (Weston et al., 2014). The breakthrough idea: separate *knowledge storage* from *computation*. A neural network with a discrete external memory that it can read from and write to. Applied to question-answering, it showed that retrieving and reasoning over supporting facts from memory improves answer accuracy. This is the intellectual ancestor of every RAG system — the realization that a model's parameters shouldn't have to store all knowledge. **C→R→S: Collector concept.**

**DrQA** (Chen et al., 2017). The first *retrieve-and-read* pipeline: TF-IDF retriever over all of Wikipedia, plus a neural document reader to extract answer spans. DrQA treated Wikipedia as a giant non-parametric memory for QA — demonstrating that combining IR with neural comprehension outperforms standalone models. This is the template every modern RAG system follows: retrieve documents, then process them with a model. **C→R→S: Collector + Ranker.**

**Wizard of Wikipedia** (Dinan et al., 2018). Extended the idea to dialogue: an agent retrieves a Wikipedia paragraph and *conditions on it* to generate a knowledgeable response. This demonstrated retrieval-grounded conversation — the agent doesn't just find an answer; it uses retrieved text to inform the style and content of its reply. The Server component appears: context construction matters, not just retrieval. **C→R→S: Server.**

### Era 3: Learned retrieval (2019–2020)

**ORQA** (Lee et al., 2019). Before ORQA, retrievers were fixed (BM25 or TF-IDF). ORQA introduced the first *end-to-end learned* retriever for open-domain QA — jointly training the retriever and reader from question-answer pairs alone, without ever seeing labeled evidence passages. It treated retrieval as a latent variable problem and showed large gains over fixed IR. This opened the door to retrieval models that improve as they see more data. **C→R→S: Ranker (learned).**

**DPR** (Karpukhin et al., 2020). Dense Passage Retrieval made learned retrieval simple and practical. Two BERT encoders (one for queries, one for passages) trained on QA pairs. DPR outperformed BM25 by 9–19% on passage recall and became the standard dense retriever — the one most 2023-era "naive RAG" systems used. Its release with pre-trained models and a Wikipedia index gave the community a turnkey dense retriever. **C→R→S: Collector + Ranker.**

**ColBERT** (Khattab & Zaharia, 2020). The late-interaction idea: instead of compressing each document into a single vector (and losing information), keep all token-level representations and compute relevance via a MaxSim operator. This balanced the rich interactions of cross-encoders with the efficiency of bi-encoders. Note that chunking documents into smaller pieces and embedding each chunk separately is the other common architectural response to the single-vector problem — rather than keeping all token representations within the model (ColBERT's approach), you decompose the document externally into sub-units that each get their own vector. Both strategies are trying to solve the same underlying information-loss problem; they just do it at different levels. We'll cover both in the Collector section. **C→R→S: Collector + Ranker.**

**REALM** (Guu et al., 2020). Went further than ORQA by integrating retrieval into *pre-training itself*. During masked-language-model training, REALM retrieves text from Wikipedia and backpropagates the training loss through the retrieval step — across millions of documents. This made the retriever improve as part of the model's own learning, not as an external bolt-on. **C→R→S: Collector + Ranker (joint).**

### Era 4: RAG gets a name (2020–2021)

**RAG** (Lewis et al., 2020). The paper that coined the term. Combined a pre-trained BART generator with a DPR-based Wikipedia index in an end-to-end trainable framework. Two architectures: RAG-Sequence (condition the whole generation on fixed top-k passages) and RAG-Token (can retrieve new passages at each decoding step). Achieved state-of-the-art on multiple QA benchmarks and showed that a generative model augmented with retrieval produces more specific, factual output than a parametric model alone. This formalized the retrieve → augment → generate pattern. **C→R→S: Full system.**

**FiD — Fusion-in-Decoder** (Izacard & Grave, 2021). Solved a Server problem: how do you use *many* retrieved passages (not just 5) without the context exploding? FiD encodes each passage independently, then concatenates the encoded representations for the decoder's cross-attention. This let the model synthesize answers from 50+ passages simultaneously — a huge advance for complex questions requiring evidence from multiple sources. **C→R→S: Server (context fusion).**

### Era 5: RAG meets LLMs (2022)

**Atlas** (Meta, 2022). Demonstrated that a small model (11B) with a retriever can match or beat much larger parametric models (540B+) on few-shot tasks. The "smaller but grounded" principle: you don't need infinite parameters if you have good retrieval. Atlas also showed efficient knowledge updates — just re-index documents, no model retraining required. **C→R→S: Full system.**

**WebGPT** (OpenAI, 2022). The first browser-augmented LLM: the model searches the web, follows links, and quotes sources. Proved that letting an LM interact with live information dramatically improves factual accuracy. This was the prototype for Bing Chat and ChatGPT browsing — and it established the expectation that models should cite their sources. Covered in Unit 5. **C→R→S: Ranker (agentic).**

**ReAct** (Yao et al., 2022). Formalized the pattern of interleaving reasoning with tool use: Thought → Action → Observation → repeat. Became the template for agentic systems and most RAG agent frameworks. Covered in detail in Unit 5. **C→R→S: Ranker (agentic).**

### Era 6: Reasoning and multi-representation (2024–2025)

**Promptriever / Rank1** (Weller et al., 2024). Made retrieval models instruction-following (Promptriever) and reasoning-capable (Rank1). Covered in depth later in this lecture. These represent a new frontier: retrieval models that understand natural-language instructions and generate auditable reasoning chains. **C→R→S: Ranker (reasoning).**

**GraphRAG** (Microsoft, 2024). LLM extracts entities and relationships into a knowledge graph for structured retrieval. Influential but often over-prescribed. Covered in the Server section. **C→R→S: Collector (structured).**

### The timeline at a glance

| Year | Milestone | Breakthrough idea | C→R→S |
|------|-----------|------------------|-------|
| 2009 | BM25 | Probabilistic term matching with saturation | Ranker |
| 2014 | Memory Networks | Separate knowledge storage from computation | Collector |
| 2016 | HNSW | Fast ANN search at scale | Collector |
| 2017 | DrQA | Retrieve-and-read over Wikipedia | Collector + Ranker |
| 2018 | Wizard of Wikipedia | Retrieval-grounded dialogue | Server |
| 2019 | ORQA | End-to-end learned retriever | Ranker |
| 2020 | DPR | Simple, effective dense retrieval | Collector + Ranker |
| 2020 | ColBERT | Late interaction (no pooling) | Collector + Ranker |
| 2020 | REALM | Retrieval integrated into pre-training | Joint |
| 2020 | RAG | Coined the term; end-to-end framework | Full system |
| 2021 | FiD | Fusion-in-decoder for many passages | Server |
| 2022 | Atlas | Small model + retriever beats large model | Full system |
| 2022 | WebGPT / ReAct | Browser-augmented and agentic retrieval | Ranker |
| 2024 | Promptriever / Rank1 | Instruction-following and reasoning retrieval | Ranker |
| 2024 | GraphRAG | Entity/relationship extraction for structured retrieval | Collector |

### Lessons from the history

The Collector got better (BM25 → dense → late interaction → multiple representations). The Ranker got smarter (fixed → learned → agentic → reasoning). The Server got more sophisticated (simple concatenation → fusion-in-decoder → context engineering).

1. **The ideas predate the hype.** Memory Networks (2014) and DrQA (2017) were doing RAG before the acronym existed. Understanding the lineage helps you evaluate whether a "new" technique is genuinely novel or a rediscovery.
2. **The durable pattern is retrieve → augment → generate.** Every milestone above is a variation on this loop. The specifics change (what you retrieve, how you augment, what generates) but the pattern remains consistent. Furthermore, the Collector → Ranker → Server framework gives you a principled way to decompose it further, so you can reason about each stage independently, identify where failures come from, and evaluate them separately.
3. **Build a working model quickly and iterate** — the lesson from the Netflix Prize (recsys book). The winning solution took so long that business circumstances changed and it was no longer useful. Ship hybrid BM25 + dense, measure, improve.

---

## The Collector: generating representations

### Feature engineering is not dead

There's a tempting narrative in the LLM era: embeddings handle everything, feature engineering is a relic. This is wrong. **Feature engineering for retrieval — more precisely, representation engineering — is more important than ever.** It just looks different. Instead of hand-crafting numerical features for a classifier, you're crafting *representations* of your documents that determine what's findable. **Representation engineering** is the discipline of deliberately shaping how your items and queries are encoded so that semantically relevant things end up close together in the spaces you actually search.

Why this matters:

1. **Models expect something different from your raw data.** An embedding model trained on web search data expects natural prose, not your internal Jira ticket format or your 200-page regulatory filing. Put things in the language you want the model to be using them — reshape your data to match what the model was trained on.
2. **Representation shapes what knowledge can be inferred.** How you segment and describe your information determines what any retrieval system can find. A financial report embedded as a single blob loses the distinction between revenue data, risk factors, and management commentary. Segmented into purposeful representations, each becomes independently findable.
3. **Scale problems hit faster than you think — in two different dimensions.** First, document stores grow: thousands of documents becomes millions, and a vector search that worked fine at 10K records becomes slow and imprecise at 10M. Second, documents themselves are often larger than the context window: a 200-page regulatory filing can't be embedded as a whole, and even if you could, it shouldn't all go into the LLM's context. Both problems compound: you need to find the right 3 paragraphs from the right 5 documents out of 10 million. A single similarity search can't do that reliably on its own — you need pre-filtering to narrow the search space before the vector search runs.
4. **Precision matters because context poisoning exists.** The longer the context, the higher the odds of a token that sends the model down the wrong path (context rot, covered below). Good features = shorter, more precise context = fewer wrong paths.
5. **Personalization requires features.** "Good answer" is subjective. Without sufficient detail captured in your representations — user role, expertise level, task context — you can't accommodate different users' notions of relevance (connects to the memory discussion below).

> **The central insight of the Collector:**
> You can only find what you've made findable.
> A movie described only by genre is matchable only by genre. A movie described by genre + mood + era + director style is matchable on all of those dimensions. A document embedded as a single vector is matchable only on whatever the model decided to prioritize. Every representation decision is a decision about what questions can be answered.

The same applies to documents. A single embedding is one "map" of the document. I call this idea **the map is not the territory.**: _You can create many different maps of the same territory, each optimized for different queries._

### The user-item matrix, revisited

The fundamental data structure in recommendation systems is the **user-item matrix** — a sparse matrix where rows are users, columns are items, and entries are ratings (explicit or implicit). The core question: *predict how much a user will like an item they've never seen.*

In RAG, the equivalent is a **query-document relevance matrix** — equally sparse, equally the core question: *predict how relevant a document is to a query the system has never seen before.* Every retrieval technique is an attempt to fill in this matrix better.

Two strategies from collaborative filtering apply directly:

- **User-user (query-side):** similar queries need similar documents. Agentic search (Unit 6) exploits this — the agent rewrites the query to be more like queries that have worked before.
- **Item-item (document-side):** similar documents answer similar queries. HyDE exploits this — rewriting documents into the language users search with aligns the document representation toward likely queries.

### Deconstructing RAG buzzwords

Every "advanced RAG" technique is just an answer to one of the three collector/ranker responsibilities, dressed up with marketing:

| Buzzword | What it actually is | C→R→S component |
|----------|-------------------|----------------|
| **Naive RAG** | Simple single-vector search | Basic Collector + Ranker |
| **Advanced RAG** | Any RAG system that improves on naive; often means multi-stage or hybrid | Improvements across all three components |
| **Modular RAG** | Treating the pipeline as composable, swappable modules | Formalizes the C→R→S decomposition |
| **Agentic RAG** | LLM rewrites queries and iterates (query enrichment) | Ranker improvement |
| **Hybrid RAG** | Multiple search signals combined | Ranker using multiple Collector outputs |
| **Graph RAG** | LLM extracts entities and relationships into a knowledge graph | Collector creates a structured representation |
| **Multi-Modal RAG** | Search across text + images (or multiple index modes) | Collector with multiple representation types |
| **Self-RAG** | Model decides when to retrieve and critiques its own retrieved results using special reflection tokens | Ranker (adaptive retrieval) + Server (self-critique) |
| **Corrective RAG (CRAG)** | Validates retrieved documents; falls back to web search if retrieved docs are low quality | Server (quality gate between Ranker and generation) |
| **Speculative RAG** | Small specialist model drafts multiple answers from different document subsets in parallel; larger model verifies | Server (draft-then-verify for latency/accuracy) |
| **Adaptive RAG** | Routes between no-retrieval, single-step retrieval, and multi-step retrieval based on query complexity | Ranker (dynamic routing) |

All of these could have been derived from first principles by asking "which of the three responsibilities (C→R→S) is underperforming, and how can I improve it?" The buzzwords obscure this clarity but capture a surprising number of retweets.

### Multiple representations: many maps of the same territory

Instead of one embedding per document, create several. Let's make this concrete with **Semantic Dot Art** ([semantic.art](https://semantic.art)) — an art search engine that demonstrates every Collector concept in one system. Try it after the lecture: search for a mood, a poem, or upload an image and see which paintings it surfaces.

The project indexes hundreds of thousands of artworks. Each artwork is a single "territory" — a painting hanging in a museum. But users search for art in wildly different ways:

- "an oil painting of the side profile of a woman surrounded by
  flowers" (literal description)
- "A path winds on, beneath a vibrant sky. Sun-warmed grasses
  whisper secrets low." (poetic mood)
- "nature, solitude, dream, deep" (mood keywords)
- or just an image of something they want more of

A single embedding can't serve all of these. So the Collector creates **multiple representations** of each artwork:

| Representation | What it captures | Example for Van Gogh's *Path Through a Field with Willows* |
|---------------|-----------------|-----------------------------------------------------------|
| **Poetic caption** | Mood and emotion as prose | "A lonely journey, bathed in golden light. Nature breathes, both calm and wild." |
| **Natural language caption** | Literal + evocative description | "A path meanders under a bright sky, where sun-warmed grasses softly rustle…" |
| **Mood keywords** | Emotional and thematic tags | `nature, solitude, dream, deep, intensity, path` |
| **Image embedding** | Visual similarity | The image itself, embedded via a vision model |

Each representation gets its own index — vector indices for the captions, a keyword index for mood terms, a multimodal index for images. The same artwork appears in all of them, but each "map" captures a different facet of what makes it matchable.

The general pattern applies to any domain:

| Domain | Representations you might create |
|--------|--------------------------------|
| **Artworks** | Poetic caption, NL caption, mood keywords, image embedding |
| **Financial docs** | Summary, data tables, entity lists, form types |
| **Code repos** | Function signatures, docstrings, README, dependency graph |
| **Products** | Title, description, attributes, review sentiment, images |

This is the recsys equivalent of Netflix representing each movie with collaborative features (who watched it), content features (genre, director, mood), visual features (thumbnail similarity), and text features (plot description). Different users match on different dimensions. The canonical example of this thinking in recsys is **PinSage** (Ying et al., 2018) — Pinterest's graph-based recommender that combined visual, textual, and engagement signals into rich per-pin representations. It was one of the earliest production systems to make "multi-modal representation" standard practice, and the intuition carries directly into how we think about document representations in RAG.

### Document enrichment: creating the maps

HyDE (from Unit 6) is actually a **document enrichment** pipeline — it rewrites documents into the language users search with. More broadly, document enrichment is any offline process that creates new representations.

> **The semantic-space mismatch issue.** A user opens Semantic Dot Art and types "a quiet optimism painted in the sky." That query gets embedded into a point in semantic space. Now consider what the raw image embedding of Van Gogh's *Wheat Field with Cypresses* looks like: it's a dense vector derived from pixel patterns, color distributions, and visual structure — trained on images, not poetry. There is no reason to expect those two vectors to be close. The user's query lives in language space; the raw image lives in vision space. They're not even playing the same game.
>
> This is the fundamental mismatch problem. Your items were created in one modality, one format, one register — and your users will express their intent in an entirely different one. A legal filing was written in legalese; a user asks "what's our liability if a contractor gets hurt on site?" A financial report was structured for auditors; an analyst asks "is this company burning cash?" A medical record was written in clinical shorthand; a patient asks "why did my doctor order that test?"
>
> Representation engineering is how you bridge this gap. Document enrichment is one of its primary tools. In Semantic Dot Art, we ran every artwork through a multimodal LLM to generate a poetic caption, a natural-language description, and mood keywords — in the *language users actually use to describe what they're looking for*. Now when a user searches "a quiet optimism painted in the sky," their query can find the artwork's poetic caption ("colors sing a song of solitude, hope lingers where the track ascends") rather than trying to match compressed pixel patterns. The Collector built the bridge offline so the Ranker can cross it at query time.

In Semantic Dot Art, the enrichment pipeline uses multimodal LLMs (Google Gemini) to generate the poetic and natural-language captions from each artwork image, and to extract mood keywords. This is the Collector doing its job: preparing the catalog so the Ranker has rich features to work with.

A critical design choice: all representations live in **the same row** of a single LanceDB table — vectors, captions, mood keywords, and the original JPEG bytes together. This means adding a new representation (say, a color-palette embedding or a brushstroke fingerprint) is adding a column, not rebuilding the index. The corpus keeps growing, but the storage architecture doesn't need to be refactored.

The scale challenge: doing this for hundreds of thousands of artworks requires distributed computation. This pattern generalizes: any document enrichment pipeline at scale is a batch LLM data-processing job (we will revisit in Unit 11). The same patterns apply — fan-out, structured output, validation between steps.

### Chunking: what it looks like when you're an expert

Most RAG tutorials show fixed-size chunking: split every document into 500-token blocks and embed each one. This is usually dumb.

The problem: chunk too aggressively and you break the macro understanding. A chunk about "Q3 revenue growth" gets retrieved for a revenue question, but the chunk has no idea that the document's *conclusion* says growth is decelerating. You've atomized the document into fragments that are individually relevant but collectively incoherent.

The opposite problem: embed the whole document as one vector and you lose the ability to find specific sections. One vector can't represent both the revenue data and the risk factors.

What expert chunking looks like:

- **Inconsistently sized chunks.** Sections that are semantically coherent stay together, even if one is 200 tokens and another is 2,000. Don't impose a grid on content that has natural structure.
- **Prepend the situation.** Every chunk should carry context about *where it came from*: "From the Risk Factors section of Acme Corp's 2025 10-K filing: [chunk text]". Without this, the chunk is an orphan — relevant text with no frame of reference.
- **Chunk along document structure, not character counts.** Use headings, sections, paragraphs, and semantic boundaries. A chapter break is a chunking signal; the 512th token is not.
- **Overlap when structure isn't available.** If you must use fixed-size chunks (plain text with no structure), overlap them by 10–20% so boundary-spanning ideas aren't split.

The deeper insight: chunking *is* feature engineering. Every chunking decision is a decision about what knowledge can be inferred from any atomic unit. Make those decisions with domain expertise, not with `text[:512]`.

### Long-document strategies: RAPTOR and hierarchical indexing

Chunking alone breaks the macro understanding. But sometimes you need *both* local detail (what did paragraph 14 say?) *and* global context (what is this document's overall argument?). Chunking gives you the former; a single document embedding gives you the latter. You often need both simultaneously.

**Hierarchical indexing** maintains two index levels: a document-level index (one embedding per document, or per major section) and a chunk-level index (fine-grained pieces). At query time, first retrieve at the document level to identify relevant documents, then retrieve at the chunk level within those documents. This is the document-level vs. chunk-level indices pattern — two C→R→S pipelines in series.

**RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) takes this further [Sarthi et al., ICLR 2024 — arxiv.org/abs/2401.18059]. It builds a tree of summaries at multiple granularities:

1. Chunk the document into leaf nodes
2. Cluster similar chunks and summarize each cluster → level-1 nodes
3. Cluster level-1 summaries and summarize again → level-2 nodes
4. Repeat until a single root summary covers the whole document

At query time, embed the query and search *all levels simultaneously* — the query might match a high-level summary (broad topic query) or a specific leaf chunk (narrow factual query), and RAPTOR surfaces both. This lets a single index support "what is this document about?" and "what did it say about X on page 34?" without making two separate calls.

The recsys equivalent: a multi-scale item representation where you can match on a movie's genre (coarse) or its specific plot point (fine), using the same index. The granularity that wins depends on what the user is really asking.

### Multimodal representations: RAG over non-text content

Real enterprise document corpora are rarely pure text. PDFs contain tables, charts, and figures. Technical documentation has code blocks and architecture diagrams. Product catalogs have images. Financial filings mix structured data and prose.

Two broad approaches to multimodal RAG:

**Extract-then-embed:** convert non-text content into text descriptions (using an LLM or vision model), then embed the text. A chart becomes "Bar chart showing quarterly revenue: Q1 $2.1B, Q2 $2.4B, Q3 $2.2B, Q4 $2.8B — overall upward trend with Q3 dip." This text can then be searched with standard vector or lexical search. The quality depends entirely on the extraction step.

**Native multimodal embeddings:** models like ColPali [Faysse et al., 2024 — arxiv.org/abs/2407.01449] (the ColBERT late-interaction architecture applied to document images) embed the *visual representation* of a page directly — no OCR, no extraction. The page image becomes a set of token-level visual embeddings. MaxSim then matches query tokens against visual tokens. This handles charts, figures, and mixed layouts that OCR would mangle, at the cost of higher storage and a narrower ecosystem.

**Structured data (tables, SQL):** tabular data doesn't embed well as raw text. Approaches include:
- **Serialize rows as sentences:** "In Q1 2025, revenue was $2.1B, margin was 18%, and headcount was 12,400." Each row becomes a retrieval unit.
- **Text-to-SQL (NL2SQL):** treat the database as a tool (Unit 5) — the LLM generates a SQL query and executes it, returning exact structured results rather than approximate semantic matches. This is often better than trying to embed tabular data: SQL is precise, semantic search is approximate.
- **Hybrid:** embed a natural-language summary of the table for semantic retrieval, then use SQL for the precise computation once you've identified the right table.

The common thread: **match the representation to the query type**. If users ask approximate conceptual questions ("what was the trend in revenue?"), embedding a prose description works. If they ask exact factual questions ("what was Q3 2025 revenue to the nearest million?"), SQL is more reliable than semantic search.

### Embedding model selection

Not all embedding models are created equal, and "which model is best?" depends on your domain, query type, and constraints.

**Starting points:**
- MTEB/BEIR leaderboards — useful for narrowing the field but contaminated; don't trust top-10 rankings as ground truth on your data
- Domain match: a model trained on legal documents will outperform a general model on legal retrieval, even with a smaller parameter count
- Context length: check whether the model degrades at your typical document length. Many claim 8K tokens but perform poorly beyond 4K.
- Latency/cost: larger models are slower and more expensive to embed at scale; for the Collector's offline embedding job this matters less, but for online query embedding it matters a lot

**Matryoshka embeddings (MRL):** some models are trained with Matryoshka Representation Learning [Kusupati et al., NeurIPS 2022 — arxiv.org/abs/2205.13147] — the embedding is structured so that the first N dimensions alone form a good representation, not just all D dimensions together. This means you can *truncate* embeddings to save storage and speed up ANN search (at a small accuracy cost) without retraining. If you're operating at scale and storage is a constraint, prefer an MRL-trained model.

**Quantization:** storing embeddings as float32 uses 4 bytes per dimension. Quantizing to int8 (1 byte) cuts storage by 4× with minimal recall drop on most benchmarks. Binary quantization (1 bit per dimension) cuts storage 32× at larger — but often acceptable — recall cost. Useful when you have billions of embeddings.

**When to fine-tune:**
- If your domain vocabulary, writing style, or query patterns differ substantially from the training data, fine-tuning pays off
- You need ~1K–10K positive (query, relevant passage) pairs to see gains; you can mine these from user logs, LLM-generated synthetics, or from labeled evaluation data
- The test: fine-tune on a sample, evaluate on your golden dataset, compare recall@K before and after. If the lift is less than 5%, the general model is probably good enough.

### Pre-filtering: narrow before you search

At scale, you don't want to run a vector search across 10 billion chunks. Pre-filtering uses structured metadata to narrow the search space *before* the embedding-based similarity search runs, so the ANN index only needs to consider a relevant subset.

Pre-filters are applied as SQL-style conditions on document metadata fields:

```python
results = (
    table.search(query_embedding)
         .where("department = 'finance' AND year = 2025", prefilter=True)
         .limit(20)
)
```

This only searches chunks where `department` is finance and `year` is 2025 — a potentially huge reduction in the candidate space before any vector math runs. And this is where multiple representations compound the benefit: each new representation you create for a document is a potential new filtering dimension. Mood keywords (Semantic Dot Art), entity lists, document type, author, recency, category — every enrichment step adds structured metadata you can pre-filter with, so the ANN search focuses its budget on exactly the right region of the index.

The key distinction between pre-filtering and post-filtering:
- **Pre-filter (before search):** reduce the index to eligible candidates, then run ANN. Faster and more precise — the similarity search focuses its budget on the right region of the embedding space.
- **Post-filter (after search):** run ANN over everything, then discard results that don't match. Wasteful and can leave you with fewer results than requested — or none at all.

> **The empty-context issue.** A user asks "what did our legal team say about GDPR compliance last month?" Your system retrieves the top-20 most semantically similar chunks from the entire corpus, then filters to `department = 'legal' AND date >= '2025-02-01'`. The top-20 ANN results all happen to come from the finance team discussing a related data privacy topic — semantically close, but wrong department. Every result is discarded. The LLM receives an empty context and confidently says "I don't have information about that" — or hallucinates an answer — even though the answer exists in the index. The document was retrievable; you just didn't narrow the search space first. This is one of the most common silent failures in production RAG systems.

Pre-filtering is what makes the Semantic Dot Art mood-keyword step work: the index is narrowed to artworks sharing at least one mood keyword *before* the vector similarity search, dramatically reducing the search space and avoiding cross-category noise.

**What metadata to expose for filtering** is a Collector decision: when you design your document schema, think about what structured dimensions users will want to narrow by — date ranges, document type, author, department, status, category. These fields should be indexed as structured columns alongside the embedding vectors.

### Putting it together: a chunking and retrieval experiment

To make these Collector ideas concrete, here's a retrieval experiment that tested **9 chunking strategies × 4 retrieval methods = 36 combinations** on a corpus of expert interview transcripts that I ran, scored against 69 curated benchmark queries.

**Chunking strategies tested:**
- **QA-pair chunking:** LLM extracts question-answer pairs from each transcript — each pair is a chunk (a form of document enrichment)
- **Key-question segment chunking:** LLM identifies key questions, then chunks around them with surrounding context (expert chunking with situational prepending)
- **Sliding window:** overlapping fixed-size windows (the "dumb but safe" baseline)
- **Fixed-token:** non-overlapping fixed-size blocks (the "even dumber" baseline)
- **Key-question only:** just the extracted questions, no surrounding context

Each strategy was tested at 256-token and 512-token sizes where applicable, yielding 9 total configurations.

**Retrieval methods tested:**
- Pure vector search (OpenAI embeddings)
- Pure full-text search (BM25)
- Threshold-based interleaving within a single index (weighted score combination of vector + FTS)
- Multi-index interleaving (two different chunking strategies searched in parallel, merged via RRF)

**Evaluation:** LLM-as-judge (Gemini 2.5 Flash) alongside standard IR metrics — nDCG, MRR, Precision@k, Recall@k, HitRate@k. The benchmark queries were curated with an explicit audit trail — golden dataset construction (Unit 1) applied to retrieval.

**What we observed** (on this corpus — your results will differ):

1. **Multi-index interleaving gave the best results.** Running two different chunking strategies in parallel and merging with RRF outperformed any single index. The key insight is general even if the specifics aren't: you can interleave across *chunking strategies*, not just across search methods. Each chunking strategy captures different aspects of the same documents — one preserves local QA context, the other preserves broader conversational flow. Interleaving lets you get the best of both without choosing.
2. **Threshold-based interleaving outperformed pure vector or pure FTS** within a single index. But the weights are not transferable — what worked for this conversational corpus wouldn't necessarily work for a legal corpus or a codebase. This is where evals become essential (see below).
3. **Larger chunks outperformed smaller chunks** within the same strategy family. On this conversational corpus, smaller chunks lost too much surrounding context. But on a corpus of short, self-contained records (support tickets, product descriptions), the opposite might hold. The point: chunk size is a hyperparameter, not a constant. Measure it.
4. **Full-text search added little over pure vector** for this particular corpus (conversational transcripts with few domain- specific keywords). A legal or medical corpus with precise terminology would almost certainly show the opposite. This is why you run your own eval rather than trusting someone else's findings — including these.
5. **LLM-as-judge caught things IR metrics missed.** A chunk could score well on nDCG (the right document, ranked high) but score poorly with the judge (the chunk lacked enough surrounding context to actually answer the question). Using both metric types together gave a more complete picture than either alone.

The experiment demonstrates the full Collector eval workflow: design multiple representations (chunking strategies), build a golden dataset of queries, measure each combination with both automated metrics and LLM-as-judge, and pick the winner based on evidence — not intuition. This is P³ applied to Collector design.

### Multi-index interleaving and learning weights

The experiment above tested two kinds of merging: rank-based (RRF across indices) and threshold-based (score fusion within a single index). In practice, you often want both — and the question becomes: **how do you find the right weights?**

Consider a system with three indices: a semantic index over document summaries, a lexical index over raw text, and a late-interaction index over entity-rich sections. You have two layers of interleaving decisions:

1. **Within each index:** if the index supports hybrid search (vector + FTS), what threshold weights do you use for that index's internal score fusion?
2. **Across indices:** how do you interleave the results from all three indices into a single ranked list?

For across-index merging, RRF is often the safest default because you don't need to compare scores across fundamentally different systems. But threshold-based interleaving can outperform RRF when you have good weights — because it preserves the *confidence* information that RRF discards (a document ranked #1 with score 0.98 is probably better than one ranked #1 with score 0.51, but RRF treats them identically).

**Using evals to learn the weights:**

The weights are hyperparameters. Treat them like any other hyperparameter — tune them on your eval set:

1. Build your golden dataset of queries with labeled relevant documents (Unit 1, Unit 3)
2. Define your target metric (precision@k, nDCG, or a composite)
3. Run a grid search or random search over the weight space — for two sources this is a single parameter (the blend ratio), for three or more it's a simplex
4. Evaluate each weight combination on the golden dataset
5. Pick the weights that maximize your target metric
6. **Re-evaluate periodically** — as your corpus changes (new document types, new query patterns), the optimal weights shift. Build this into your monitoring.

In the experiment, this is exactly what I did: test each retrieval method (including different weight configurations for threshold-based interleaving) against the same 69 benchmark queries, score with both IR metrics and LLM-as-judge, and select the configuration with the best performance. The winning approach — multi-index RRF — won not because RRF is inherently better, but because on *this* corpus the score distributions across chunking strategies were different enough that rank-based merging was more robust than score-based merging. On a different corpus, threshold-based might win. **You find out by measuring.**

This becomes even more essential as you add representations. With two indices, you have one weight to tune. With five — semantic embeddings, lexical index, entity list, HyDE questions, and a late-interaction index — you have a five-dimensional weight space. Without a golden eval set, you're guessing in a space too large to reason about intuitively. The more representations you create (and you should create many), the more dependent you become on systematic measurement to know which ones are actually contributing and how much to trust each.

### Online vs. offline: the dual-path pattern

In classic recsys, every component has both an offline and an online mode, and this distinction is very important for understanding the parameters of the system you'll need to build. For the collector:

- **Offline collector:** batch-process the corpus, compute embeddings, build indices. Runs on a schedule or triggered by data changes.
- **Online collector:** handle real-time updates — new documents, edits, deletions. Must update indices without full recomputation.

**Notion — the Collector at consumer scale.** Notion is a collaborative workspace where millions of users store notes, projects, wikis, and databases. When they launched Notion AI Q&A, the product needed to answer natural-language questions like "what did we decide in last week's product review?" by searching across a user's entire workspace — pages they wrote, meeting notes from teammates, documents buried in nested databases. Keyword search fails here because users don't know the exact words in the target document. Vector search lets Notion match intent to content semantically across heterogeneous, constantly-changing workspaces.

The engineering challenge: each workspace is isolated (you can only search your own content), workspaces range from a single student's notes to an enterprise with thousands of employees, and pages are edited continuously in real time. Notion documented how they scaled this in a [detailed engineering post](https://www.notion.com/blog/two-years-of-vector-search-at-notion) (Feb 2026) — worth reading in full. Here we pull out the architecture lessons.

**The dual-path Collector:**
- **Offline path:** batch jobs chunk existing documents, generate embeddings, and bulk-load vectors into the index. Used for initial workspace onboarding and for model migration.
- **Online path:** a streaming consumer processes individual page edits in real time at sub-minute latency — new pages, deleted blocks, permission changes.

This dual-path architecture is the canonical Collector design. The offline path handles volume; the online path handles freshness. Without both, you're either slow to onboard or stale on updates.

**Time as a first-class dimension:**
Notion tracks when each chunk was last modified. This makes recency a pre-filter and a non-query ranking signal — pages edited this week rank higher for a query about "recent decisions" than a page from two years ago, even if both are semantically similar. This connects directly to the broader principle: your Collector should index structured metadata alongside every embedding, and time is almost always one of the most valuable fields to expose for filtering and ranking. Without it, you can't distinguish "what does our policy say?" (any version is fine) from "what is the current policy?" (recency matters critically).

**Smart change detection — the 70% reduction:**
The trickiest Collector problem: when a document changes, how much do you re-index? Naive answer: re-embed and reload the entire document on every edit. At Notion's scale, this was financially unsustainable.

Their solution: hash both the *text* and *metadata* of each chunk separately, and store those hashes. On every edit, diff the new hashes against the stored ones:

- **Only text changed:** re-embed and reload just the changed chunks, not the whole document.
- **Only metadata changed** (e.g. permissions): skip re-embedding entirely — the vector is still valid. Issue a metadata update to the stored record.
- **Neither changed:** do nothing.

Result: 70% reduction in data volume (and associated embedding API costs and index write costs). The key insight is that text content and metadata are orthogonal concerns — treating them separately lets you avoid the expensive work (embedding generation) when only the cheap work (metadata update) is needed.

This is an important reminder why "representations must evolve". The engineering challenge is detecting *exactly what changed* and doing only the necessary work.

The lesson for Collector design: the index isn't a static artifact you build once. It's a living system that must handle creates, updates, and deletes efficiently.

A static embedding goes stale. If a financial report is embedded today and the company announces a major acquisition tomorrow, the embedding doesn't reflect that context. The recsys equivalent: a movie's features change when a new season drops or when it wins an award. The Notion case shows what this looks like in practice: text changes require re-embedding; metadata changes don't; and the system needs to tell them apart cheaply. Without that distinction, the cost of keeping representations fresh becomes a ceiling on how aggressively you can enrich your documents.

### The pooling problem: why single-vector search fails

Recall from Unit 6: semantic search encodes documents as single vectors via an encoder + pooling step.

Pooling compresses all token-level information into one vector. The compression is *selective* — the model prioritizes what its training data considered important. Trained on movie-review data where queries ask about actors? It encodes actor names and discards plot details. On out-of-domain data (legal contracts, cooking recipes), the learned notion of "important" is wrong.

In the recsys analogy: encoding movies by genre alone loses director, mood, and era. Users asking "films like Blade Runner" (genre + mood + era) get bad recommendations because the model only sees "sci-fi."

### Late interaction: keep all the tokens

**ColBERT** and late interaction models solve the pooling problem by skipping the compression step [reader Ch. 4]. They keep *all* token- level representations. The **MaxSim** operator computes relevance by finding the maximum similarity between each query token and all document tokens, then summing:

```
Query tokens:    [q₁]  [q₂]  [q₃]
                   ↓     ↓     ↓
                  max   max   max   ← max similarity against
                   ↓     ↓     ↓      all doc tokens
Document tokens: [d₁] [d₂] [d₃] [d₄] [d₅] ...

Score = max_sim(q₁) + max_sim(q₂) + max_sim(q₃)
```

Results from the course reader:
- Late interaction (19.61) vs. dense (12.31) on BRIGHT — **same backbone and training data** — architecture alone accounts for 60% improvement
- **150M-parameter late interaction model outperforms all 7B-parameter dense models** on reasoning tasks (45x smaller)
- **Interpretability bonus:** token-level matching shows exactly which parts of a document contributed — useful for debugging and for providing precise context to the LLM

A further finding: late interaction models trained on documents of only 300 tokens generalize to 8K+ token contexts — the architecture handles long documents without long-document training. And the approach extends to other modalities: **ColPali** applies the ColBERT architecture to images for OCR-free document retrieval.

> **P³ lens:** late interaction and multiple representations are
> **Collector** architecture decisions. You choose which maps to
> maintain based on which failure modes your **Proof** reveals. If
> eval shows poor recall on entity queries, add an entity index. If
> it shows poor performance on long documents, add a summary
> representation. The eval tells you whether the investment paid off.

### Measuring the Collector

The Collector's job is to make documents *findable*. You measure this with **retrieval recall** — of all relevant documents in the corpus, what fraction did the Collector surface as candidates?

| What to measure | Metric | What it tells you |
|----------------|--------|------------------|
| Can the system find relevant docs at all? | Recall@50, Recall@100 | Upper bound on system quality — if the Collector misses a doc, the Ranker can't fix it |
| Are the right representations being used? | Recall per index type (lexical, semantic, entity) | Which maps contribute and which are dead weight |
| Does enrichment help? | Recall before/after adding HyDE, summaries, etc. | Whether the enrichment pipeline investment pays off |
| How fresh are the indices? | Staleness audit (% of docs with embeddings older than X days) | Whether the online Collector is keeping up |

The critical insight: the Collector sets the **ceiling** for the whole system. If a relevant document never makes it into the candidate set, no amount of ranking or context engineering will recover it. Measure recall at a generous K (50–100) as your Collector metric — this is separate from the final precision@K the user sees.

---

## The Ranker: matching intent to documents

The ranker takes what the collector prepared and scores it for the current query. In recsys terms: user modeling → candidate generation → scoring → reranking. In RAG: query understanding → retrieval → reranking.

### Query understanding (user modeling for RAG)

In a recommendation system, user modeling is everything — you need to know what someone wants before you can find it. In retrieval, the equivalent is **query understanding**: transforming a raw query into something that retrieval can act on.

Unit 6 introduced two approaches:
- **Agentic search:** the LLM rewrites queries in a loop
- **Structured queries:** field-specific, typed queries

Both are user-side collaborative filtering — the system infers what the user really needs, which may differ from what they typed.

**Semantic Dot Art shows the full query-understanding pipeline.** When a user searches for art, the Ranker:

1. **Classifies the query** as "natural" or "poetic" via an LLM agent — this determines which representation index to search (query understanding → routing)
2. **Generates an image caption** if the user provided an image input, in the appropriate style
3. **Rewrites the query** combining text and image caption into the target style (query rewriting)
4. **Extracts mood keywords** from the rewritten query for prefiltering (structured enrichment)
5. **Prefilters** the index to only artworks sharing at least one mood keyword (fast candidate reduction)
6. **Searches** the appropriate representation index — full-text, vector, or hybrid depending on query length
7. **Reranks** results using a weighted combination of mood-keyword precision and recall between the query and each artwork

The actual search call, after all that routing and rewriting, is compact:

```python
results = (
    table.search(
        query_type="hybrid",
        vector_column_name="poetic_vector",
    )
    .vector(query_embedding)
    .text(query_text)
    .where(prefilter_clause, prefilter=True)
    .limit(10)
    .rerank(CustomKeywordRanker(keywords))
)
```

Hybrid search, prefiltering on mood keywords, and custom reranking in one chained call. The complexity lives in the *routing* (which column to search, which query to use), not in the retrieval API itself.

This is all three classic search techniques — query understanding, query rewriting, multi-index search — orchestrated by an LLM agent. The agent is the translation layer between raw human expression ("Where may I find a piece full of grief and woe?") and the structured representations the Collector prepared.

### Instruction-following retrieval: Promptriever

A limitation of query rewriting: it happens *outside* the retriever. The retriever itself is a dumb function — takes a string, returns nearest neighbors. It can't understand instructions like "find documents that use metaphors."

**Promptriever** (Weller et al.) changes this [reader Ch. 3]. A bi-encoder trained on *instruction negatives* — documents relevant to the query but irrelevant to the instruction. The model encodes instructions directly into the query embedding.

The result: the first bi-encoder with positive scores on instruction-following benchmarks. And a remarkable property — **zero-shot hyperparameter optimization via prompting.** Tell the model "have really high recall" in natural language and it adjusts its internal strategy.

In recsys terms: Promptriever is a recommendation system where the user can say "show me adventurous picks, not safe ones" and the ranking function itself changes behavior.

### Reasoning at reranking time: Rank1

**Rank1** (Weller et al.) applies test-time compute to reranking [reader Ch. 3]. A cross-encoder that generates explicit reasoning chains to assess relevance — identifying key phrases, analyzing relationships, questioning its own interpretations.

Results:
- Nearly **doubles** baseline on reasoning benchmarks (BRIGHT) and negation tasks (NevIR)
- **10-point gain** just from training with thinking traces vs. without — learned via distillation from DeepSeek-R1, not RL
- Discovers **novel relevant documents** that previous systems missed entirely — when researchers re-judged previously unjudged docs from DL19/DL20, Rank1 became the top-performing model

In recsys terms: Rank1 is a recommendation system that *explains its ranking decisions*. The reasoning chain is auditable.

### Multi-stage ranking (same as recsys)

Production recommendation systems almost always use multi-stage ranking (*BPRS* Ch. 12): a fast, cheap candidate generator followed by a slow, expensive ranker. Retrieval works the same way:

```
Full corpus (millions)
  → Stage 1: fast retrieval (BM25 + dense, top 100-500)
    → Stage 2: cross-encoder reranker (top 10-50)
      → Stage 3 (optional): reasoning reranker (Rank1)
```

Each stage trades throughput for accuracy. The merge strategies from Unit 6 (interleaving/RRF, set union) handle Stage 1. Stages 2–3 can now include reasoning — impossible before Promptriever and Rank1.

**Instacart — the Ranker with LLM-powered query understanding.** Instacart's search serves millions of product queries, many of which are long-tail and ambiguous: "bread no gluten", "x large zip lock", "healthy snacks for kids lunch". Their old system used separate traditional ML models for each query-understanding task (spell-check, synonym expansion, intent classification), creating system complexity and poor performance on rare queries due to data sparsity.

Their new **Intent Engine** architecture:
- **Query understanding (Ranker stage 0):** a single LLM call replaces multiple specialized models, handling spell correction, synonym expansion, intent classification, and query rewriting in one step. The LLM generalizes to long-tail queries that individual models couldn't handle.
- **Multi-stage retrieval (Ranker stages 1–2):** the rewritten query feeds into lexical + semantic search, then reranking. The LLM doesn't replace retrieval — it *improves what the retrieval system sees*.
- **Multimodal enrichment (Collector):** their PARSE system (Unit 11) extracts product attributes from text *and* images, so the Collector has rich features for the Ranker to match against.

The lesson for Ranker design: the biggest retrieval gains often come from better query understanding, not better embeddings. If the Ranker receives "bread no gluten" and passes it verbatim to BM25, no amount of embedding quality will fix the mismatch. Rewriting it to "gluten-free bread" is a Ranker-level improvement.

### BM25 still works (and now we know why)

BM25 — a 30-year-old algorithm — outperforms many dense models on long-context and reasoning benchmarks [reader Ch. 1, 4]. The explanation maps to the pooling problem: BM25 does *no compression*. It matches exact keywords without losing information.

BM25 fails when there's no lexical overlap (synonyms, different languages). Hybrid search — combining BM25 with semantic methods — consistently outperforms either alone. Complementary failure modes.

In the recsys analogy: BM25 is "filter by exact attribute" — ignores subtlety but never misses a direct match.

### Cold-start: the retrieval version

In recsys, cold-start is when a new user arrives with no history, or a new item has no interactions. Encoder architectures and feature-based warm starting address this (*BPRS* Ch. 7).

In retrieval, cold-start appears as:
- **New document types** the embedding model has never seen (a codebase when it was trained on web text)
- **New query patterns** from a new user population
- **Domain shift** when the system is applied to a new industry

Late interaction models handle cold-start better than dense models because they don't compress — there's less opportunity for the training distribution bias to discard the "wrong" information. BM25 handles cold-start perfectly because it requires no training at all.

### Ranking for the LLM, not for a user

Before we talk about measuring the Ranker, there's a subtlety that changes what "good ranking" means when the consumer is an LLM.

In classical search, the user reads results top-to-bottom. Rank #1 gets the most attention, rank #20 gets almost none. All standard IR metrics (MRR, NDCG, Precision@K) are built around this assumption — higher rank is always better.

For RAG, the consumer is an LLM, and LLMs don't behave like linear readers. Unit 2 introduced **lost in the middle**: models attend most strongly to the *beginning* and *end* of their context. Documents placed in the middle of a long context window are systematically underused, regardless of their rank.

This has a non-obvious consequence: **for RAG, ordinal ranking is not always what you want.** If you rank 20 documents and pass them all to the LLM in rank order, the 10 documents in the middle may be largely ignored — even if they contain the evidence needed for a correct answer. Context poisoning (covered in the Server section below) makes this worse: a mediocre but plausible document in the middle can actually hurt more than help.

Practical implications for Ranker design:
- **Be selective about K.** Passing 20 docs when 5 would do exposes the LLM to more noise and more middle-context degradation.
- **Consider position, not just rank.** The best documents should go at the beginning *or* end of the context, not in the middle — this is a Server-level concern, but the Ranker needs to identify which documents are most important so the Server can place them correctly.
- **Diversity matters more than pure relevance ordering.** If your top-3 documents all say the same thing and docs 4–6 cover different critical facets, the LLM may miss those facets entirely if they fall in the middle. A Ranker optimized for NDCG (which rewards putting the single most relevant doc first) can paradoxically hurt RAG quality if it stacks similar documents together.

This is one place where the recsys metrics translation breaks down slightly: recsys ranks for humans who scan lists; RAG ranks for a model that has a context window with position-dependent attention. Keep both in mind.

**Highlighted documents — making importance explicit**

One practical response to lost-in-the-middle is to not just *position* the best documents, but to tell the LLM explicitly that they're highly relevant. Rather than presenting a flat list of retrieved passages, the Server can annotate the top results:

```
[HIGHLY RELEVANT - confidence: 0.94]
"The GDPR compliance policy was updated on March 1, 2025. All vendors 
must complete the new DPA before April 30..."

[SUPPORTING CONTEXT]
"Our data processing team uses the following template..."
```

This is representation engineering applied to the context window itself. The LLM's attention is shaped not only by position but by the signal in the tokens — an explicit "highly relevant" tag is a strong prior that this passage deserves attention even if it falls in the middle.

You can derive the annotation from the Ranker's confidence score, from reranker output, or from a combination (e.g. top-2 results above a score threshold get the highlight; everything else is supporting context). The threshold is a hyperparameter you tune on your eval set — the same way you tune retrieval weights.

### Context rot and context poisoning

The case for selective, well-presented context isn't just intuition — there's empirical research behind it. Kelly Hong et al. tested 18 state-of-the-art models on how performance degrades as context grows in their [Context Rot technical report](https://research.trychroma.com/context-rot) (July 2025):

- **Semantic matching degrades faster than lexical matching** as context grows
- **Distractors are devastating** — this is **context poisoning:** the longer the context, the higher the odds that a single misleading token sends the model down the wrong path. Semantically similar but incorrect information causes confident hallucination (GPT models worst; Claude more likely to abstain)
- **More distractors + longer context = compounding degradation**
- **Shuffled context slightly *helps*** — models do better when the haystack is randomly shuffled than when it's a coherent essay. LLMs don't process context linearly.
- **The U-shaped attention pattern may not hold anymore** — Chroma found no consistent advantage for placing information at the start or end of context, qualifying the "lost in the middle" finding from Unit 2.
- **No single model resists context rot across all tasks** — performance is highly task-dependent.

Three takeaways for Ranker design:
1. Performance is not uniform across context lengths — your Ranker should be calibrated to the context lengths you actually send
2. Having the right information isn't enough — presentation matters, which means how the Ranker orders and the Server positions documents directly affects answer quality
3. **Retrieve less but better.** A Ranker that returns 5 high-precision results consistently outperforms one that returns 20 and hopes the LLM figures it out.

In recsys terms: showing a user 1,000 items is worse than showing 10 good ones. **Retrieval quality > retrieval quantity.**

### Measuring the Ranker

The Ranker's job is to put the best documents at the top. You measure this with **ranking metrics** — how good is the ordering?

| What to measure | Metric | What it tells you |
|----------------|--------|------------------|
| Is the best doc in the top positions? | MRR (Mean Reciprocal Rank) | How quickly the user (or LLM) sees the first relevant result |
| Are the top-K results good? | Precision@K, NDCG@K | Quality of the ranked list the Server will use |
| Did query rewriting help? | Precision/recall before vs. after rewriting | Whether the LLM query-understanding step adds value |
| Does reranking help? | NDCG with/without the reranking stage | Whether the expensive Stage 2/3 justifies its latency and cost |
| Per-stage contribution | Recall@K at each stage boundary | Where in the pipeline quality is gained or lost |

The critical practice: **measure at each stage boundary, not just end-to-end.** If your Stage 1 (BM25 + dense) has 60% recall@100 and your final answer accuracy is 55%, the bottleneck is the Collector (not enough candidates), not the Ranker. If Stage 1 has 95% recall@100 but final accuracy is still 55%, the Ranker is failing to sort the good candidates to the top.

> **P³ lens:** the ranker is the **Proof** target. Your retrieval
> eval (precision@k, recall@k, coverage, diversity) measures whether
> the ranker does its job. Always compare against baselines (Unit 1):
> BM25-only, random documents, and "stuff the whole corpus." The
> baseline quantifies how much value your ranking pipeline adds.

---

## The Server: from ranked results to grounded answers

In a recommendation system, the server isn't just "return the top K." It applies business logic (inventory constraints, hard avoids, diversity rules), satisfies the response schema, and ensures the final set of recommendations is coherent.

In RAG, the server component is equally important but often neglected. It's the bridge between "here are the top-K retrieved documents" and "here's a grounded answer for the user."

The practical pattern for the Server when context rot is a concern: **orchestrator + subagents.** Each subagent gets focused context (just the relevant documents for its subtask), returns only essential findings. This prevents context overload and keeps each subagent's context clean of distractors.

### Measuring the Server

The Server's job is to turn ranked results into a *good answer*. This is where you measure the full RAG system end-to-end — not just retrieval quality, but generation quality conditioned on retrieved context.

| What to measure | Metric | What it tells you |
|----------------|--------|------------------|
| Does the answer use the retrieved evidence? | **Faithfulness** (is every claim grounded in a retrieved doc?) | Whether the LLM is generating from context or hallucinating |
| Is the answer complete? | **Coverage** (what % of required facts appear in the answer?) | Whether the Server provided enough evidence — FreshStack's nugget coverage |
| Are the retrieved docs diverse enough? | **Diversity** (alpha-nDCG) | Whether the Server is deduplicating or passing 10 copies of the same fact |
| Does context length hurt? | Answer quality at short vs. long context | Whether context rot is degrading your specific system |
| Does the answer cite sources? | **Citation accuracy** (do citations point to the right passages?) | Whether provenance is working |

The diagnostic pattern: when the end-to-end answer is wrong, trace *where* it broke:

```
Collector missed the doc?  → Recall problem (add representations)
Ranker had it but ranked low? → Ranking problem (improve reranker)
Doc was in context but LLM ignored it? → Context rot (reduce context, reorder)
LLM used the doc but generated wrong answer? → Generation problem (prompt engineering, Unit 4)
```

This decomposition — Collector vs. Ranker vs. Server failure — is what makes C→R→S an evaluation framework, not just an architecture framework. When something goes wrong, you know *where to look*.

> **P³ lens:** the full RAG system's **Promise** is "grounded,
> accurate answers over private data." The **Proof** requires
> measuring at every stage: Collector recall, Ranker precision/NDCG,
> and Server faithfulness/coverage. A single end-to-end metric hides
> which component is failing — exactly the "metric laundering" pitfall
> from Unit 3. In **Production**, these per-stage metrics become
> monitoring signals: if Collector recall drops (new doc types not
> being indexed), you catch it before end-to-end quality degrades.

### Memory vs. retrieval: the user model

In a recommendation system, the **user model** is what makes recommendations personal — it tracks preferences, history, and how both evolve over time. Without it, every user gets the same results. The user model is not the item catalog; it's the system's understanding of *who is asking*.

Unfortunately, many people cant distinguish the difference between Retrieval and Memory, so let's try to be very clear.

**Stateless retrieval** answers: "given this query, which documents are relevant?" It doesn't care who's asking. The same query returns the same results for every user. This is what our Collector and Ranker do — index documents, match queries.

**Stateful memory** answers: "given this user's history and current state, what additional context should influence the response?" This is the user model — preferences, past interactions, facts that have changed over time.

The failure mode when you use retrieval for memory:

```
Week 1:  "We're using Postgres for everything, it's working great."
Week 4:  "Postgres is getting slow, we're migrating to a data warehouse."
Week 6:  "Migration done — Snowflake is now our source of truth."
Week 8:  User asks "What database should we query for the revenue report?"

Semantic search: returns "Postgres is working great" (highest similarity to "database")
Memory system:   returns "Snowflake is now our source of truth" (current state)
```

Semantic similarity finds the closest text to "database." But the *current truth* has changed — "Postgres is working great" is **temporally invalid**, superseded by the migration. A memory system tracks when facts became true or were superseded.

The dimensions that separate memory from retrieval:

| | Stateless retrieval | Stateful memory |
|---|---|---|
| **Scoping** | Same results for all users | User/entity-specific |
| **Temporal** | No concept of "when" | Tracks validity windows |
| **Invalidation** | Documents are current or deleted | Facts can be superseded without deletion |
| **Relationships** | Document-to-query similarity | Entity relationships (user→preference→brand) |

> **A complete RAG system retrieves from memories too.** A production system typically has at least two retrieval targets: the document corpus (what does the knowledge base say?) and the memory store (what do we know about this user's current state?). Both are retrieved, both are ranked, and both are assembled by the Server into the final context. Memory isn't a separate system bolted on the side — it's another collection in your Collector, another signal in your Ranker, and another source of context in your Server.

In the C→R→S framework, memory spans all three components:
- **Collector:** stores user-scoped observations with timestamps and metadata (not just document embeddings)
- **Ranker:** uses recency, preference history, and temporal validity as ranking signals (this is the "non-query signals" concept from Unit 6 — personalization and recency)
- **Server:** assembles both retrieved documents *and* user memories into the context, ensuring the LLM sees the user's current state alongside the relevant knowledge

The practical question: do you need a separate memory system, or can you handle it with user-scoped retrieval + metadata filtering
+ temporal weighting? For many applications, adding a `user_id`
scope, a `created_at` timestamp, and a recency boost to your existing retrieval infrastructure gets you 80% of the way there. For applications where entity relationships and fact invalidation are central (personal assistants, long-running agents), a dedicated memory layer is worth the investment.

We'll return to memory management in depth in Unit 8 (Context Engineering), where the question becomes: *how do you decide what goes into the context window when you have both retrieved documents and accumulated memories?*


### Agents as routers

With multiple representations in the collector, the server must decide which one to search for each query. This is the **routing** problem.

For small route sets (5–10 index types), an LLM classifies via tool call — the agentic search pattern from Unit 6. For larger route sets, a fine-tuned classifier is more appropriate.

**Glean — the Server with permission-aware routing.** Glean provides enterprise search across 100+ SaaS tools (Slack, Google Drive, Jira, Confluence, Salesforce, etc.). The Server component is where their architecture gets interesting:

- **Routing across 100+ sources:** each SaaS connector has its own index with different document schemas, update frequencies, and access patterns. The Server decides which sources to query for each request — a routing problem at scale.
- **Personal graph:** models individual employee work patterns — which tools they use, which documents they access, who they collaborate with. This is non-query signal personalization (from Unit 6) implemented as a graph.
- **Enterprise graph:** models company entities and their relationships — org structure, project ownership, team membership. Used to augment relevance with organizational context.
- **Permission enforcement at retrieval time:** this is the Server's most critical job. A document may be highly relevant but the user doesn't have access. Glean enforces permissions *during* retrieval, not after — otherwise the system would leak information through relevance signals even without showing content.

The lesson for Server design: in enterprise RAG, the Server isn't just "format results for the LLM." It's the enforcement layer for access control, organizational context, and cross-source routing. The recsys equivalent is the business logic layer (*BPRS* Ch. 14) — inventory constraints, regional availability, age restrictions — that determines what a specific user is *allowed* to see, not just what they'd *like* to see.

### RAG vs. tool use: where's the line?

Students who are still attending to tokens from Unit 5 may be asking: "isn't retrieval just another tool? Why is RAG a separate concept from MCP/function calling?" The distinction matters because it affects how you architect, evaluate, and debug the system.

**Retrieval (RAG)** puts documents *into the context window* for the LLM to reason over. The model sees the evidence and synthesizes an answer. You control what it sees; it controls what it says. The Collector → Ranker → Server pipeline we've been building is about curating that context.

**Tool use (MCP/function calling)** lets the model *take actions* — query a database, call an API, run code — and get back a result. The model decides *when* to call and *what arguments* to pass; the tool handles execution. The result comes back as a message, not as context documents.

The real-world overlap:

| | RAG | Tool use |
|---|---|---|
| **What the model receives** | Retrieved documents in context | Tool result as a message |
| **Who decides what to fetch** | The retrieval pipeline (Collector + Ranker) | The model (via function calling) |
| **When it happens** | Before generation (context is pre-assembled) | During generation (model decides mid-response) |
| **Best for** | "Answer this question using these documents" | "Look up this specific fact" or "take this action" |
| **Eval focus** | Retrieval quality (recall, coverage, diversity) | Tool selection accuracy, argument correctness |

In practice, the two compose. An agentic RAG system uses **tool use to trigger retrieval**: the model calls a `search` tool (Unit 5's function calling), the tool runs the Collector → Ranker pipeline (this lecture), and the results come back as context for the next generation step. Glean's 100+ source routing is exactly this — the LLM uses a tool call to select which sources to query, and the retrieval pipeline does the actual searching.

**Subagents** (Unit 5) add another layer. In complex multi-step tasks, a parent agent can spawn subagents each with their own focused retrieval context — one subagent retrieves from the legal knowledge base, another from the financial data, another from recent meeting notes. Each subagent runs its own C→R→S pipeline against the relevant collection, then returns only its findings to the parent. This is the orchestrator + subagents pattern applied to retrieval: it solves context poisoning (each subagent gets a clean, distractor-free context) and permissioning (each subagent can be scoped to only the collections it's authorized to search).

The P³ implication: RAG and tool use have **different proof requirements**. For RAG, you measure retrieval quality (did we find the right documents?) and faithfulness (did the model use them correctly?). For tool use, you measure tool selection (did it call the right tool?) and argument correctness (did it pass the right parameters?). A system that combines both needs eval for both — don't let one hide failures in the other.

### Diversity and coverage: the recsys lesson

In recsys, recommending 10 variations of the same item is a known failure mode. Diversity constraints (*BPRS* Ch. 15) ensure the final set covers multiple aspects of the user's interests.

In RAG, the equivalent failure is retrieving 10 documents that all say the same thing. The LLM doesn't get the evidence breadth it needs. FreshStack (Chapter 2) formalizes this with **three evaluation dimensions**:

| Dimension | Metric | What it measures |
|-----------|--------|-----------------|
| **Relevance** | Recall@50 | Are the docs on-topic? |
| **Grounding (Coverage)** | Coverage@20 | What % of unique atomic facts ("nuggets") are supported? |
| **Diversity** | alpha-nDCG@10 | Non-redundancy — penalize retrieving docs that repeat the same fact |

Coverage is especially important: even if you retrieve 10 relevant documents, if they all support the same nugget and miss others, the LLM generates an incomplete answer.

A key FreshStack finding: **no single model performs best across all topics** [reader Ch. 2]. The gap between current performance and the theoretical maximum is large, indicating significant room for improvement. This reinforces why you need domain-specific eval rather than trusting a leaderboard.

### Graph structures in retrieval

Graph-based reasoning shows up in both recsys and RAG, and the graph structure can be highly useful.

A graph is a collection of **nodes** (entities) connected by **edges** (relationships). Nodes can be any type of object — documents, users, concepts, products. Edges encode a relationship between two nodes: "cites," "was-accessed-together," "shares-an-author," "co-purchased." Edges can be directed (A→B means A cites B, not necessarily the reverse) or undirected (A—B means A and B co-occur symmetrically).

Graphs become especially powerful in retrieval when the corpus is **heterogeneous** — where different types of nodes exist and different types of edges connect them. A document graph might have nodes for documents, authors, topics, and organizations, with edges encoding authorship, topic membership, and citation relationships. Querying across these different node types simultaneously is where graphs outshine flat vector search.

When a graph is **homogeneous** (only one type of node, one type of edge), it's often more naturally represented as a similarity matrix or an ANN index — which is why HNSW, the dominant vector index algorithm, is itself a navigable graph over embedding vectors.

**In recommendation systems:** graphs encode relationships that aren't obvious from item content alone. [PinSage](https://arxiv.org/abs/1806.01973) (Ying et al., 2018 — Pinterest) used a graph where pins are nodes and edges encode co-saves (if many users saved both pin A and pin B, they're connected). This lets the system recommend pin A not because it looks like what you searched for, but because users who engaged with your kind of content also engaged with it — the relational signal that no single-item embedding captures.

The same logic applies to documents. A document might not be semantically close to your query, but it co-occurs with relevant documents in citations, in link structure, or in access patterns. Graph structure surfaces these connections.

**Link retrieval — two patterns:**

The first pattern is **neighbor retrieval** (the cluster hypothesis): documents similar to a relevant document are also likely relevant. Once your initial retrieval identifies the best candidates, run a second retrieval pass for documents similar to those candidates.

```
Query → [Retrieve top-k] → [Rerank to top-3] → [Retrieve neighbors of top-3] → final candidate set
```

This is graph traversal using your existing vector index. You're walking one hop through the similarity graph. Useful when the corpus has dense clusters of related content and the initial query might only land in the neighborhood, not on the exact target.

The second pattern is **bridging retrieval**: use one representation to find a document, then use a different relationship to find what the LLM actually needs. For example — a user asks a vague question about quarterly performance. Your semantic search finds an analyst memo that's closely related to the query in embedding space. That memo cites a specific financial report. You follow that citation link to retrieve the actual report. The memo was the bridge; the report is what the LLM needs for a grounded answer.

In recsys terms: the memo is the "gateway item" — it's similar to the user's expressed intent (the query), but the thing that's actually useful for completing the task (the report) is one relational hop away.

**Multi-hop retrieval** is the natural extension of bridging retrieval: instead of one hop (memo → report), you chain multiple hops to build up the evidence needed for a complex question.

> **Issue:** a user asks "given our current vendor contracts and last quarter's spend data, which vendors should we renegotiate with this cycle?" Answering this requires: (1) retrieve vendor contracts to understand current terms and expiry dates, (2) retrieve spend data to identify which vendors have the highest costs, (3) join those two results to find vendors with unfavorable terms *and* high spend, (4) retrieve any recent performance reviews or incident reports for those vendors. No single retrieval pass can get all of this.

Multi-hop retrieval addresses this by treating retrieval as a series of dependent steps — each retrieval's result informs the next query. The agent from Unit 5's agentic search pattern handles this naturally: the LLM issues a search, reads the results, decides what additional information is needed, and issues another search with a more specific query. Each hop refines the target.

The key engineering consideration: **each hop compounds context.** After three retrieval hops, you have three sets of documents in context, and context poisoning risk rises with each one. Subagents (Unit 5) are the structural answer: each hop runs in its own clean context, and only the distilled finding gets passed to the next hop or back to the orchestrator.

**Do you need a graph database for this?** Rarely. Both neighbor retrieval and bridging retrieval can be implemented with a vector store plus structured metadata (citation links, access-pattern co-occurrence, document relationships stored as JSON or in Postgres). Purpose-built graph databases (Neo4j etc.) are worth adding only when you have complex multi-hop traversal at scale where SQL or in-memory graphs become too slow. For most RAG systems, the cluster hypothesis can be exploited with your existing vector index, and explicit link structures can live in a standard relational store alongside your embeddings. When you only have 300 documents, just put them all in context.

---

> **Why do people say "RAG is dead"?** They're reacting to a specific thing: the 2023-era approach of embedding every document as a single vector and doing cosine similarity popularized by the DevRels and content creators. That approach has real problems — information loss from pooling, training distribution mismatch, no diversity or coverage. Everything in this lecture is a response to one or more of those problems. "RAG is dead" means the naive single-vector version is insufficient; it doesn't mean retrieval is dead.

---

## Retrieval evaluation: your own journey

The leaderboards (BEIR, MTEB) are useful for orientation, but they're contaminated — models are trained on these benchmarks now, and leaderboard rank doesn't predict performance on your corpus. The only eval that matters is the one you build on your data.

This is Unit 3's lesson applied to retrieval. You already know the pattern:

1. **Build a golden dataset.** Collect 50–100 real or realistic queries from your actual use case. Curate them carefully — cover typical queries, long-tail queries, and the hard cases where you expect failure. This is the most important investment; everything else depends on it.
2. **Define your failure modes.** Before you run anything, write down what "bad retrieval" looks like for your system. Missing a critical document? Returning 10 copies of the same fact? Retrieving the right document but the wrong chunk? Each failure mode suggests a different metric.
3. **Establish baselines.** BM25 alone. Dense alone. Hybrid. Run all three on your golden dataset before adding any complexity. You'll be surprised how often BM25 is competitive — and you need to know the baseline before you can claim a win.
4. **Iterate with evidence.** Add a new chunking strategy? Measure it. Switch embedding models? Measure it. Add a late-interaction reranker? Measure it. The experiment I described earlier tested 9 chunking strategies × 4 retrieval methods on 69 benchmark queries — that level of systematic measurement is what lets you trust your conclusions.
5. **Use multiple metrics.** Precision@K tells you about the top results; recall@K tells you whether the Collector found the relevant docs at all; an LLM judge catches quality issues that IR metrics miss (a chunk can be "relevant" by nDCG but lack enough context to actually answer the question). Use all three.
6. **Feed failures back.** Every time the system returns a bad result, add the query to your eval set. The golden dataset should grow with your system, not sit frozen while production diverges from it.

The goal isn't to maximize a benchmark — it's to build enough measurement infrastructure that you can confidently say "this change made the system better for our users." That's the P³ Proof: the eval is the instrument; the C→R→S improvements are the experiments; and the golden dataset is what keeps you honest.

---

## Key Takeaways

1. **RAG is a recommendation system.** Collector → Ranker → Server. The recsys playbook — user modeling, item modeling, multi-stage ranking, diversity, cold-start — applies directly. The three responsibilities (generate representations, predict intent, match intent to representation) organize every advanced technique.
2. **You can only find what you've made findable.** Representation engineering is the discipline of deliberately shaping how items and queries are encoded so that semantically relevant things end up close together. This is the Collector's core job — and it's more important than your choice of vector index.
3. **Chunking is feature engineering.** Every chunking decision is a decision about what knowledge can be inferred from any atomic unit. Expert chunking: prepend situation context, follow document structure, use inconsistent chunk sizes based on semantic coherence. The "right" chunk size is a hyperparameter — measure it on your corpus.
4. **Pre-filter before you search, not after.** Post-filtering can yield zero results even when the answer exists. Pre-filtering narrows the search space so the ANN search focuses its budget on the right region. Every enriched representation is a potential new filtering dimension.
5. **Pooling is the fundamental flaw of single-vector search.** Late interaction (ColBERT / MaxSim) keeps token-level detail and beats dense models with the same backbone — often by 60%+ on reasoning tasks.
6. **Rank for the LLM, not for a human.** Lost-in-the-middle means ordinal rank isn't what you want — the best documents should be at the beginning or end of context, not in the middle. Be selective about K. Annotate highly relevant documents explicitly. Diversity matters more than stacking the same top-ranked documents.
7. **Context rot and context poisoning are real.** More context makes models worse, especially under distractors. A single misleading token can send the model down the wrong path. Retrieve less but better.
8. **Memory is another collection in your RAG system.** A complete RAG system retrieves from both the document corpus and the user's memory store. Memory isn't a separate paradigm — it's stateful, user-scoped retrieval with temporal validity.
9. **Graphs are already in your retrieval stack.** HNSW is a graph. Neighbor retrieval (cluster hypothesis) and bridging retrieval (gateway documents) give you graph-like patterns with your existing vector index. You rarely need a separate graph database.
10. **Interleaving weights are hyperparameters — tune them on your eval set.** With two indices, you have one weight to tune. With five, you have a five-dimensional space. The more representations you create, the more dependent you become on systematic measurement.
11. **Build your own eval and iterate.** Golden dataset → baselines → systematic experiments → LLM judge + IR metrics → feed failures back. The leaderboards don't know your corpus. Ship hybrid BM25 + dense, measure, improve.

---

## Further Reading

### Course texts
- Bischof & Yee, *Building Production Recommendation Systems in Python and JAX* (O'Reilly, 2024) — Ch. 1–4 (Collector → Ranker → Server), Ch. 2–3 (user-item matrix, collaborative filtering), Ch. 7 (cold-start), Ch. 12 (multi-stage ranking), Ch. 14–15 (business logic, diversity)
- Husain, Clavié, Thakur, Weller, Chaffin, Bischof, et al., *Beyond Naive RAG: Practical Advanced Methods* (2025) — available in `resources/beyond-naive-rag.pdf`

### Research papers

**Retrieval foundations (history section)**
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) — the BM25 paper
- Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW* (2016) — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (DPR, 2020) — [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) — [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
- Izacard & Grave, *Leveraging Passage Retrieval with Generative Models for Open Domain QA* (FiD, 2021) — [arxiv.org/abs/2007.01282](https://arxiv.org/abs/2007.01282)

**Late interaction and reasoning retrieval**
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (2020) — [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)
- Weller et al., *Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models* (2024) — [arxiv.org/abs/2409.11136](https://arxiv.org/abs/2409.11136)
- Weller et al., *Rank1: Test-Time Compute for Reranking in Information Retrieval* (2025) — [arxiv.org/abs/2502.18418](https://arxiv.org/abs/2502.18418)

**Multiple representations, long-document strategies, and multimodal**
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE, 2022) — [arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496)
- Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (ICLR 2024) — [arxiv.org/abs/2401.18059](https://arxiv.org/abs/2401.18059)
- Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models* (2024) — [arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449)
- Kusupati et al., *Matryoshka Representation Learning* (NeurIPS 2022) — [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147)
- Ying et al., *Graph Convolutional Neural Networks for Web-Scale Recommender Systems* (PinSage, 2018) — [arxiv.org/abs/1806.01973](https://arxiv.org/abs/1806.01973)

**Evaluation**
- Thakur et al., *FreshStack: Building Realistic Benchmarks for RAG Evaluation* (2025)

### Production case studies and reports
- Bischof et al., *Semantic Dot Art* — [semantic.art](https://semantic.art) · [LanceDB technical blog](https://lancedb.com/blog/semanticdotart/)
- Notion, *Two Years of Vector Search: 10x Scale, 1/10th Cost* (Feb 2026) — [notion.com/blog/two-years-of-vector-search-at-notion](https://www.notion.com/blog/two-years-of-vector-search-at-notion)
- Instacart, *Building the Intent Engine: Revamping Query Understanding with LLMs* — [instacart.com/.../building-the-intent-engine](https://www.instacart.com/company/tech-innovation/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms)
- Glean, *Unified Index for Enterprise AI* — [glean.com/product/system-of-context](https://www.glean.com/product/system-of-context)
- Hong et al., *Context Rot: How Increasing Input Tokens Impacts LLM Performance* (July 2025) — [research.trychroma.com/context-rot](https://research.trychroma.com/context-rot)

### Evaluation resources
- FreshStack leaderboard — [github.com/FreshStack](https://github.com/FreshStack)
- MTEB leaderboard (use with caution) — [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
