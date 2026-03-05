# Lecture 01: ML Foundations Crash Course

**Unit:** 1 (Data & ML Foundations)  
**Date:** Wednesday, January 21, 2026  
**Length:** 1 lecture (crash course)

This lecture is a fast reset on core ML ideas, organized around three anchors:

| Anchor | Core Question |
|--------|---------------|
| **Problem Framing** | What decision are we trying to improve? |
| **Data Framing** | What evidence do we have (or can we get)? |
| **Objective Framing** | How will we measure and optimize "good"? |

Everything else—models, training, metrics—is downstream of these three choices.

---

## Why ML, in One Sentence

Machine learning is the practice of building systems that **infer** (predict, rank, classify, estimate) from data, and proving they work by **evaluating** those inferences against an objective on data that was not used to build the system.

That "prove they work" part is the essential engineering link to the rest of this course: **evaluation as a test harness**.

---

## Part 1: Problem Framing

Problem framing is the translation from a vague desire ("make it smarter") into a well-scoped engineering goal.

### A Good Problem Framing Answers

| Question | What It Clarifies |
|----------|-------------------|
| **Decision** | What decision will the system make or support? |
| **Actionability** | What changes if the prediction is different? |
| **User + Workflow** | Who consumes the output, when, and how? |
| **Constraints** | Latency, cost, privacy, interpretability, safety, compliance |
| **Failure Costs** | What is worse: false positives or false negatives? |
| **Success Criteria** | What does "better" look like in the real world? |

### Example: Spam Filtering (Classification)

- **Decision**: Route message to inbox vs. spam
- **Failure costs**: False positives are expensive (lost real mail)
- **Constraints**: Low latency; robust to adversarial text
- **Success**: Fewer spam messages at the same (or lower) false positive rate

### Example: Search (Ranking)

- **Decision**: Order documents so the user finds the answer quickly
- **Failure costs**: Wrong top-1 is worse than wrong rank-50
- **Success**: Better top-k relevance and time-to-answer, not just accuracy

### The ML Mindset vs. Traditional Programming

In traditional programming, you write explicit rules:

```python
def classify_email(email: str) -> str:
    spam_words = ["lottery", "winner", "click here", "urgent"]
    if any(word in email.lower() for word in spam_words):
        return "spam"
    return "not_spam"
```

This works until:
- Spammers change their vocabulary
- You encounter edge cases you didn't anticipate
- The rules become too numerous to maintain

In ML, you provide examples and let the algorithm discover the rules:

```python
# You provide labeled examples
training_data = [
    ("Congratulations! You've won...", "spam"),
    ("Meeting tomorrow at 3pm", "not_spam"),
    # ... thousands more
]

# The algorithm learns a function
model = train(training_data)

# The model makes predictions on new data
prediction = model.predict("Your package is delayed")
```

### When NOT to Use ML

Use ML when:
- (a) Rules are brittle
- (b) Patterns are complex or high-dimensional
- (c) You can define an evaluation method

**If you can't evaluate it, you can't engineer it.**

### Types of ML Problems

| Type | Output | Examples |
|------|--------|----------|
| **Classification** | Discrete categories | Spam detection, sentiment, image recognition |
| **Regression** | Continuous values | Price prediction, temperature forecasting |
| **Ranking** | Ordered list | Search results, recommendations |
| **Generation** | New content | Text, images, code |

---

## Part 2: Data Framing

Data framing is about turning the problem into a dataset with a clear *unit of analysis* and a trustworthy measurement process.

### Key Choices

| Choice | Question |
|--------|----------|
| **Unit of analysis** | What is one row? A user? A session? A query? A document? |
| **Inputs (features)** | What signals are available at decision time? |
| **Target (label)** | What are we trying to predict (or approximate)? |
| **Time** | What timestamps matter? What is known at prediction time? |
| **Sampling** | Where does the data come from, and what gets left out? |

### Features and Representations

A **feature** is a measurable property of the data that might be useful for prediction.

**Raw data** → **Feature extraction** → **Feature vector** → **Model**

#### Example: House Price Prediction

Raw data:
- Address: "123 Main St, Austin, TX"
- Description: "Charming 3BR/2BA with updated kitchen..."
- Photos: [img1.jpg, img2.jpg, ...]

Extracted features:
```python
features = {
    "square_feet": 1850,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1985,
    "distance_to_downtown_miles": 5.2,
    "school_rating": 8,
}
```

### Feature Engineering

Feature engineering is the art of creating useful features from raw data. It's often the difference between a mediocre model and a great one.

| Technique | Example |
|-----------|---------|
| **Binning** | Age → age_group (18-25, 26-35, ...) |
| **Interaction** | bedrooms × bathrooms |
| **Aggregation** | Average transaction amount over last 30 days |
| **Time features** | Day of week, hour of day, is_weekend |
| **Text features** | Word count, sentiment score, TF-IDF |
| **Encoding categories** | One-hot encoding, target encoding |

### Data Is a Measurement Instrument

Treat your dataset like a sensor. Sensors have bias, noise, missingness, and failure modes.

| Issue | Description |
|-------|-------------|
| **Label noise** | Human error, ambiguous cases, shifting definitions |
| **Proxy labels** | Clicks as proxy for satisfaction; purchases as proxy for "good recommendations"—proxies can be useful and dangerous |
| **Feedback loops** | Your model changes user behavior, which changes data |
| **Missing data** | Systematic vs. random missingness |
| **Outliers** | Errors vs. real but rare phenomena |
| **Class imbalance** | 99% not-fraud, 1% fraud |

### The Most Common Data Failure: Leakage

**Data leakage** is when training data accidentally contains information that would not be available at inference time. This inflates evaluation metrics and leads to models that fail in production.

| Pattern | Example | Problem |
|---------|---------|---------|
| **Future information** | Using tomorrow's stock price to predict today's | Won't have this at prediction time |
| **Target leakage** | Using "account_closed" to predict churn | Effect used to predict cause |
| **Train-test contamination** | Normalizing before splitting | Test statistics influence training |
| **Duplicates across splits** | Same user in train and test | Not testing generalization |
| **Aggregates including label** | Average outcome per group | Label information in features |

### Split Discipline

```
┌─────────────────────────────────────────────────────────────┐
│                      ALL DATA                                │
├───────────────────┬─────────────────┬───────────────────────┤
│    TRAIN SET      │  VALIDATION SET │       TEST SET        │
│     (60-70%)      │    (15-20%)     │       (15-20%)        │
│                   │                 │                       │
│  Fit parameters   │  Choose hyper-  │  Final estimate       │
│                   │  parameters &   │  Keep it boring       │
│                   │  design choices │  and sacred           │
└───────────────────┴─────────────────┴───────────────────────┘
```

**Critical rules**:
- **Train**: Used to fit parameters
- **Validation**: Used to choose hyperparameters and make design decisions
- **Test**: Used once for a final estimate; keep it "boring and sacred"

**K-Fold Cross-Validation** (when data is limited):

```
Fold 1: [Val][Train][Train][Train][Train]
Fold 2: [Train][Val][Train][Train][Train]
Fold 3: [Train][Train][Val][Train][Train]
Fold 4: [Train][Train][Train][Val][Train]
Fold 5: [Train][Train][Train][Train][Val]
```

### Temporal Considerations

If the world is time-dependent, prefer **time-based splits** over random splits:

```python
# WRONG for time-series
X_train, X_test = train_test_split(data, random_state=42)

# RIGHT: Time-based split
train = data[data["date"] < "2024-01-01"]
test = data[data["date"] >= "2024-01-01"]
```

### Dataset Shift (Why Production Is Hard)

| Type | What Changes | Example |
|------|--------------|---------|
| **Covariate shift** | P(X) changes | New users, new topics |
| **Label shift** | P(Y) changes | Fraud rate changes |
| **Concept drift** | P(Y\|X) changes | Meaning of "spam" changes |

Your evaluation dataset must resemble the distribution you will face.

---

## Part 3: Objective Framing

Objective framing is how you decide what "good inference" means and how you will measure it. It turns engineering goals into metrics and tests.

**This is the key ML idea to carry forward:**

1. **Objective framing** defines a test for inference quality (metric + protocol)
2. **Evaluation data** is the input to that test
3. You iterate on the system to improve the measured result

### Loss Functions: What Models Optimize

A **loss function** measures how wrong the model's predictions are. Training is the process of finding parameters that minimize this loss.

#### Regression Losses

| Loss | Formula | Behavior |
|------|---------|----------|
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | Heavily penalizes large errors; sensitive to outliers |
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Linear penalty; more robust to outliers |
| **Huber** | MSE for small errors, MAE for large | Best of both worlds |

#### Classification Losses

| Loss | Use Case |
|------|----------|
| **Cross-entropy (log loss)** | Probabilistic classification; foundation of logistic regression |
| **Hinge loss** | SVMs; encourages confident predictions with margin |

### Training Objective vs. Evaluation Metric

Often, models optimize a convenient loss but you evaluate with a different metric:

- Train logistic regression with log loss; evaluate with F1 or PR-AUC
- Train a ranker with pairwise loss; evaluate with NDCG@k

**If your loss and metric disagree, you can get "better training" but worse user outcomes.** That is an objective framing problem, not a model problem.

### The Key Insight: Loss Functions Define Behavior

**What you optimize is what you get.**

Example—medical diagnosis where:
- False negative (missing cancer) is catastrophic
- False positive (unnecessary biopsy) is inconvenient

Standard loss treats both errors equally. You might want asymmetric costs:

```python
def weighted_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fn_cost = 100  # Missing cancer is very bad
    fp_cost = 1    # Unnecessary test is inconvenient
    
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    
    return fn_cost * fn + fp_cost * fp
```

### Multi-Objective Reality

Most real systems are multi-objective:

- **Maximize utility**: accuracy, relevance, helpfulness
- **Minimize harm**: unsafe outputs, unfair outcomes
- **Minimize cost**: latency, tokens, compute, human review time

Objective framing turns this into something you can evaluate:

| Component | Purpose |
|-----------|---------|
| **Primary metric** | What you optimize for most |
| **Guardrails** | Metrics that must not regress (safety, latency, cost) |
| **Slices** | Metrics by subgroup, traffic type, or scenario |

**If you only track a single metric, you will accidentally optimize the wrong thing (Goodhart's Law).**

---

## Part 4: Optimization (Why Training Works)

Most training is: choose parameters θ that minimize a loss on training examples:

$$\hat{\theta} = \arg\min_{\theta} \frac{1}{n}\sum_{i=1}^{n}\ell(f_{\theta}(x_i), y_i)$$

Where:
- $f_{\theta}$: your model
- $\ell$: your training loss (objective function)

### Gradient Descent

The workhorse of ML optimization:

1. Compute the gradient (direction of steepest increase) of the loss
2. Step *against* the gradient to reduce the loss
3. Repeat until improvements stop

```
Loss
  │
  │    ●  (start)
  │     \
  │      \
  │       \
  │        \_________●_____
  │                (minimum)
  └────────────────────────── Parameters
```

**Learning rate** is crucial:
- Too large → Overshoot minimum, diverge
- Too small → Very slow convergence
- Just right → Steady progress toward minimum

### Stochastic Gradient Descent (SGD)

Computing gradients over all training data is expensive. SGD uses mini-batches:

```python
for epoch in range(num_epochs):
    for batch in get_batches(training_data, batch_size=32):
        gradients = compute_gradients(model, batch)
        model.parameters -= learning_rate * gradients
```

- Noisier gradients but much faster iterations
- Noise can help escape local minima
- Foundation of modern deep learning

### Overfitting Is Optimization Doing Its Job Too Well

A model can fit the training data extremely well and still be wrong on new data:

```
Training accuracy: 99.9%
Test accuracy: 62.3%    ← Overfit!
```

That is why evaluation uses **held-out** data.

### Regularization: Preventing Overfitting

| Technique | How It Works |
|-----------|--------------|
| **L2 (Ridge)** | Penalize large weights: $L + \lambda\sum\theta_i^2$ |
| **L1 (Lasso)** | Encourage sparse weights: $L + \lambda\sum\|\theta_i\|$ |
| **Dropout** | Randomly zero out neurons during training |
| **Early stopping** | Stop training when validation loss starts increasing |

```
Loss
  │
  │  Training loss keeps decreasing ────
  │  \                                  \
  │   \    Validation loss             ●─●─●  (overfitting)
  │    \   starts increasing →       /
  │     \─────────────●─────────────●
  │                   ↑
  │            Stop here!
  └───────────────────────────────────────── Epochs
```

---

## Part 5: Model Evaluation (The Centerpiece)

Evaluation answers: **how good is our inference on new data, under our objective?**

### The Evaluation Paradigm

```
┌─────────────────────────────────────────────────────────────┐
│                    THE ML EVALUATION LOOP                    │
│                                                              │
│   1. Define evaluation dataset (held-out, representative)    │
│   2. Pick metrics aligned with the objective                 │
│   3. Run evaluation consistently (same data, code, slicing)  │
│   4. Use error analysis to guide iteration                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Classification Metrics

#### The Confusion Matrix

```
                      Predicted
                   Neg      Pos
              ┌─────────┬─────────┐
    Actual    │   TN    │   FP    │
      Neg     │         │ Type I  │
              ├─────────┼─────────┤
      Pos     │   FN    │   TP    │
              │ Type II │         │
              └─────────┴─────────┘
```

#### Derived Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | $\frac{TP + TN}{Total}$ | Balanced classes, symmetric costs |
| **Precision** | $\frac{TP}{TP + FP}$ | False positives are costly (spam filter) |
| **Recall** | $\frac{TP}{TP + FN}$ | False negatives are costly (disease screening) |
| **F1** | $2 \cdot \frac{P \cdot R}{P + R}$ | Need single number, care about both |
| **PR-AUC** | Area under precision-recall curve | Imbalanced classes, rare positives |

**Warning**: Accuracy is misleading for imbalanced classes! A model that always predicts "not fraud" achieves 99% accuracy when fraud is 1%.

#### Thresholds and Operating Points

Most classifiers output a probability. Your **objective framing** picks how to convert that into action:

```python
prediction = "positive" if model.predict_proba(x) > threshold else "negative"
```

Adjusting the threshold trades off precision and recall:

```
Precision
    │
1.0 ●
    │ \
    │  \
    │   \___
    │       \____
    │            \___●
0.0 └──────────────────── Recall
   0.0                  1.0
```

**You do not "find the best model" without also choosing an operating point.**

### Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | $\frac{1}{n}\sum\|y-\hat{y}\|$ | Robust to outliers; interpretable units |
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ | Punishes large errors more |
| **RMSE** | $\sqrt{MSE}$ | Same units as target |
| **R²** | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | Variance explained (0-1) |

### Ranking / Retrieval Metrics

| Metric | What It Measures |
|--------|------------------|
| **Precision@k** | Fraction of top-k that are relevant |
| **Recall@k** | Fraction of relevant items in top-k |
| **MRR** | How soon does the first relevant item appear? |
| **NDCG@k** | Graded relevance with top-heavy weighting |

### Baseline Models

Always compare against baselines:

| Type | Example | Purpose |
|------|---------|---------|
| **Random** | Random predictions | Sanity check |
| **Majority class** | Always predict most common | Bar for imbalanced data |
| **Simple heuristic** | Rule-based approach | Domain knowledge baseline |
| **Simple model** | Logistic regression | Complexity baseline |

If your complex model doesn't beat simple baselines, something is wrong.

### Error Analysis: The Fastest Path to Improvement

After you compute metrics, look at failures:

- Bucket errors by type (missing data, ambiguous labels, rare topics)
- Add slices that correspond to real product risk
- Use this to improve data framing (labels, coverage) and objective framing (metrics, rubrics)

---

## Part 6: A Tour of Models

You can do a lot with a small set of model families. Pick the simplest model that meets your constraints and objectives.

### Linear Models (Logistic / Linear Regression)

$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

| Aspect | Details |
|--------|---------|
| **Strengths** | Fast, stable, interpretable, hard to overfit with regularization |
| **Failure mode** | Underfits complex patterns unless features are engineered well |
| **Use when** | You need interpretability, have limited data, or as a baseline |

### Trees and Ensembles

**Decision Trees**: Sequence of if-then-else rules

```
              Is income > 50K?
              /            \
           Yes              No
            |                |
     Has debt > 10K?     Approve
        /        \
      Yes         No
       |           |
    Deny       Approve
```

**Random Forests**: Ensemble of trees, each on random subset of data/features

**Gradient Boosting** (XGBoost, LightGBM, CatBoost): Trees trained sequentially, each correcting prior errors

| Aspect | Details |
|--------|---------|
| **Strengths** | Strong tabular baseline, handles non-linear interactions |
| **Failure mode** | Can overfit with leakage; harder to calibrate |
| **Use when** | Tabular data; often state-of-the-art for structured data |

### Neural Networks

**Basic architecture**:
```
Input → [Linear → Activation] → [Linear → Activation] → Output
         Hidden Layer 1          Hidden Layer 2
```

| Aspect | Details |
|--------|---------|
| **Strengths** | Representation learning; great for text, vision, audio; scalable |
| **Failure mode** | Data hungry, sensitive to training setup, harder to debug |
| **Use when** | Unstructured data (text, images), large datasets |

### Transformers: The Foundation of Modern LLMs

The transformer architecture (Vaswani et al., 2017) is the basis for GPT, BERT, Claude, and all modern LLMs.

**Key innovation: Self-Attention**

Instead of processing sequences one step at a time, each position can "attend to" all other positions:

```
         ┌───────────────────────────────────────┐
Input:   │ The  │ cat  │ sat  │ on   │ the  │ mat │
         └───────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────────────┐
         │         Self-Attention                 │
         │   (each word attends to all others)   │
         └───────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────────────┐
Output:  │ Contextualized representations        │
         └───────────────────────────────────────┘
```

We'll dive deeper in Unit 2. For now, understand that everything we've covered (loss functions, optimization, evaluation) applies to transformers too.

---

## Part 7: A Tour of ML Paradigms

You don't need all the details today; you need to recognize which setting you're in because it determines what "data" and "evaluation" mean.

### Supervised Learning

You have labeled pairs $(x, y)$. Train to predict $y$ from $x$.

- **Examples**: Spam detection, sentiment, credit risk, QA classification
- **Data need**: Labeled examples
- **Evaluation**: Compare predictions to held-out labels

### Unsupervised Learning

No labels; learn structure in $x$.

- **Examples**: Clustering users, anomaly detection, dimensionality reduction
- **Data need**: Unlabeled examples
- **Evaluation**: Domain-specific metrics (cluster quality, reconstruction error)

### Self-Supervised Learning (How Modern LLMs Get Built)

Create "labels" from the data itself.

- **Example**: Next-token prediction on text
- **Data need**: Large corpus of text (no manual labels)
- **Evaluation**: Perplexity on held-out text, downstream task performance

### Semi-Supervised Learning

Mix small labeled data with large unlabeled data.

- **Example**: Pseudo-labeling, consistency regularization
- **Use when**: Labels are expensive but unlabeled data is plentiful

### Reinforcement Learning (Including Preference Optimization)

Learn a policy by feedback on actions.

- **Example**: Optimize a chatbot to match preferences (RLHF / DPO)
- **Data need**: Reward signal or preference comparisons
- **Evaluation**: Reward achieved, preference win rate

---

## Part 8: Statistics (Enough to Avoid Common Mistakes)

Statistics matters in ML because evaluation produces numbers from samples. Your evaluation metric has uncertainty.

### Sampling and Uncertainty

If you evaluate on $n$ independent examples, your metric is an estimate. As $n$ grows, the estimate becomes more stable.

**Practical guidance**:
- Report **both** the metric and the dataset size
- Prefer **confidence intervals** (or bootstraps) for key comparisons
- Watch out for cherry-picked slices; predefine slices when possible

### Correlation Is Not Causation

Your dataset is an observation of a process you did not control. Confounding and selection bias can make models look good offline and fail online.

In AI systems, this shows up as:
- **Clicks as feedback**: Biased toward what you showed
- **Human ratings**: Inconsistent rubrics, drift over time
- **Self-fulfilling outputs**: Model changes what gets measured

---

## Part 9: Connecting to AI Engineering

LLM outputs are often subjective, multi-dimensional, and non-deterministic. The same framing still applies:

| Classical ML | AI/LLM Systems |
|--------------|----------------|
| Training set | Few-shot examples |
| Test set | Golden dataset (eval set) |
| Accuracy/F1 | LLM-as-judge scores, rubric scores |
| Cross-validation | Multiple eval scenarios |
| Overfitting | Prompt overfitting |
| Data leakage | Eval contamination |

### The Practical Translation

- **Problem framing**: What decision or behavior are we specifying?
- **Data framing**: What are the test cases (prompts, contexts, tools, traces)?
- **Objective framing**: What rubric defines "good"? What are guardrails?

The implementation:
1. Build a **golden dataset** of representative scenarios
2. Define a **rubric** (possibly multi-metric)
3. Evaluate with **humans**, **LLM-as-judge**, or both
4. Track regressions in CI like unit tests

### What Changes with LLMs

| Aspect | Classical ML | LLM-Based Systems |
|--------|--------------|-------------------|
| **Training** | You train the model | Model is pre-trained; you prompt/fine-tune |
| **Features** | You engineer features | Model handles raw text |
| **Evaluation** | Objective metrics | Often subjective (is this summary good?) |
| **Iteration** | Retrain with new data | Adjust prompts, context, retrieval |

### What Stays the Same

1. **You need evaluation data** — Can't improve what you can't measure
2. **You need clear success criteria** — "Make it better" isn't actionable
3. **You need to prevent contamination** — Test data must be truly held out
4. **You need baselines** — How good is "good"?
5. **You need to iterate systematically** — Change one thing at a time

---

## Key Takeaway

**Objective framing creates the test, and the evaluation dataset is how you run it.**

This is the central idea that connects classical ML to everything we'll do with LLMs, retrieval, and agents in this course.

---

## Checklists (Keep These Handy)

### Problem Framing Checklist

- [ ] Can I explain the decision in one sentence?
- [ ] Do I know who consumes the output and what they do with it?
- [ ] Do I know which error type is more costly?
- [ ] Have I identified constraints (latency, cost, safety)?

### Data Framing Checklist

- [ ] Is my unit of analysis clear?
- [ ] Is my label definition stable and measurable?
- [ ] Does every feature exist at inference time?
- [ ] Do I have a split strategy that matches deployment?
- [ ] Have I checked for leakage?

### Objective Framing Checklist

- [ ] Does the metric match the decision and costs?
- [ ] Do I have guardrails and slices?
- [ ] Is my evaluation dataset representative?
- [ ] Have I defined baselines?

---

## In-Class Exercise: Framing Worksheet (15-20 minutes)

Pick one of these example systems:

1. A course assistant that answers questions using course notes
2. A tool that ranks documents for a user query (RAG retrieval stage)
3. A classifier that flags unsafe or low-quality outputs

Fill in:

**Problem Framing**
- Decision:
- User/workflow:
- Failure costs:
- Constraints:

**Data Framing**
- Unit of analysis:
- Inputs available at decision time:
- Label definition (or proxy):
- Leakage risks:
- Proposed split strategy:

**Objective Framing**
- Primary metric:
- Guardrails:
- Key slices:
- Evaluation dataset plan:

*If you can fill this out cleanly, implementation is usually straightforward.*

---

## Looking Ahead

- **Unit 2** will treat LLMs as a specific ML system with special constraints (tokens, sampling, cost, non-determinism)
- **Unit 3** will go deep on evaluation: golden datasets, rubrics, judges, and CI

The evaluation mindset we established today will guide how we think about LLM-based systems throughout the course.

---

## Lab Preview

In today's lab, you'll:
1. Build a complete ML pipeline with proper train/test splits
2. Implement multiple evaluation metrics
3. Compare models and understand the precision-recall tradeoff
4. Experience how data leakage inflates metrics

This hands-on experience will make the concepts concrete and prepare you for building evaluation pipelines for LLM systems.
