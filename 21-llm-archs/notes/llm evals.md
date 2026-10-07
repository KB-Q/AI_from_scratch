# LLM Evaluation

## 1. First principles: what an eval is

An evaluation is three components:
1. A **task distribution** $D$ — the prompts/inputs you care about.
2. A **scoring function** $s$ — maps a model output to a number (correct/incorrect, quality, preference).
3. An **aggregation** — usually a mean over the sample, with an uncertainty estimate.

$$\text{Score} = \frac{1}{N}\sum_{i=1}^{N} s\big(M(x_i),\, x_i,\, y_i^*\big)$$

For classification, $s$ is trivial (does the label match). For LLMs the output is **open-ended text**, so the entire difficulty is building $s$ — how to turn a free-form generation into a reliable number. Everything below is a way to construct $s$.

Every eval implicitly claims three kinds of validity:
- **Construct validity** — the metric measures the capability you care about, not a proxy.
- **Internal validity** — the score reflects the model, not artifacts (prompt format, contamination, judge bias).
- **External validity** — performance on $D$ predicts performance in deployment.

---

## 2. The scoring-function taxonomy

Ordered by how tightly the task constrains the output:
1. **Closed-form / exact** — one canonical answer (MCQ letter, a number, a class). $s$ = exact match after normalization.
2. **Reference-overlap** — free text with a gold reference; score textual overlap (BLEU, ROUGE, token-F1).
3. **Embedding similarity** — semantic overlap via embeddings (BERTScore).
4. **Probability-based** — score the probability the model assigns to the correct continuation (perplexity, logprob ranking of choices).
5. **Programmatic / verifiable** — an executable oracle checks correctness (unit tests for code, a math checker, JSON/tool-call schema validation).
6. **Model-graded (LLM-as-judge)** — another LLM scores or compares outputs.
7. **Human** — ratings or preferences; the ground truth the rest approximate.

**Governing principle: use the most constrained scorer the task allows.** Reliability and cost-per-eval both improve as you move up the list: verifiable > reference-overlap > model-graded > human-only. Reserve judges and humans for constructs cheaper scorers genuinely can't capture (open-ended helpfulness, safety, style).

---

## 3. Metrics & formulas

### Closed-form
- **Accuracy** = fraction exactly correct. For multiple choice, two ways to read the model:
	- **String match** — parse the letter/text the model emits (simple, but fails on format noise).
	- **Logprob ranking** — pick the option with the highest average token log-probability under the model (no parsing, needs logit access). Sensitive to option length, so normalize by token count. **(WHY normalize? raw sequence logprob favors shorter options — fewer negative log-terms to sum.)**
- **Exact Match (EM)** and **token-F1** for extractive QA (SQuAD-style): normalize (lowercase, strip articles/punctuation), then EM = exact equality; F1 = harmonic mean of token precision/recall between prediction and reference.

### Reference-overlap (free-form generation)
- **BLEU** (translation, precision-oriented):
$$\text{BLEU} = \text{BP}\cdot\exp\!\Big(\sum_{n=1}^{N} w_n \log p_n\Big), \qquad \text{BP} = \min\!\big(1,\, e^{1 - r/c}\big)$$
$p_n$ = modified $n$-gram precision, $r$ = reference length, $c$ = candidate length, BP = brevity penalty. **(WHY "modified" precision? clip each n-gram's match count to its max count in the reference, so repeating one correct word can't inflate the score.)**
- **ROUGE** (summarization, recall-oriented): ROUGE-N = $n$-gram recall vs reference; ROUGE-L = based on longest common subsequence.
- Limit: both reward surface overlap, not meaning — a correct paraphrase scores low. Hence embedding metrics.

### Embedding-based
- **BERTScore**: embed candidate and reference tokens with a pretrained encoder, greedily match by cosine similarity, report precision/recall/F1 over matches. Captures paraphrase; inherits the encoder's blind spots.

### Probability-based
- **Perplexity** = exponentiated average negative log-likelihood on held-out text:
$$\text{PPL} = \exp\!\Big(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i \mid x_{<i})\Big)$$
Measures language-modeling quality, not task performance, and is only comparable across models sharing a tokenizer. **(WHY tokenizer-dependent? PPL is per-token, and different tokenizers split the same text into different numbers of tokens.)**

### Verifiable — code: pass@k
An output is correct iff it passes the unit tests. "Sample $k$, does at least one pass" is **pass@k**. Estimating it by generating exactly $k$ is high-variance, so generate $n \ge k$, count $c$ that pass, and use the **unbiased estimator**:
$$\text{pass@}k = \mathbb{E}\Big[\,1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\,\Big]$$
**(WHY this form? $\binom{n-c}{k}/\binom{n}{k}$ is the probability a random size-$k$ subset contains *no* passing sample; one minus that is "at least one passes." Compute in log-space to avoid overflow.)**

### Pairwise / arena — Elo & Bradley-Terry
When quality is only judged *relatively* (A vs B), give each model a latent skill $\beta_i$ and fit it from win/loss records. **Bradley-Terry**:
$$P(i \text{ beats } j) = \sigma(\beta_i - \beta_j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}$$
Fit $\{\beta_i\}$ by maximum likelihood over all recorded pairwise outcomes (a logistic regression). **Elo** is the online version with a fixed per-game update:
$$R_i \leftarrow R_i + K\,(S - E), \qquad E = \frac{1}{1 + 10^{(R_j - R_i)/400}}$$
$S \in \{1, 0.5, 0\}$ = actual outcome, $E$ = expected. Chatbot Arena aggregates human pairwise votes this way. **(WHY relative? open-ended chat has no absolute gold answer, and humans compare two responses far more reliably than they rate one in isolation.)**

### Judge validation — agreement
Before trusting any automatic judge, measure agreement with human labels *beyond chance*. **Cohen's kappa**:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
$p_o$ = observed agreement, $p_e$ = agreement expected by chance. $\kappa = 1$ perfect, $0$ = chance-level.

---

## 4. LLM-as-judge

Use an LLM to score outputs where no cheap oracle exists. Three modes:
- **Pointwise** — judge scores one output on a rubric (e.g. 1–10). Cheap, but scores drift and aren't calibrated across items.
- **Pairwise** — judge picks the better of two outputs. More reliable (a relative call), and pairs with Elo/BT.
- **Reference-guided** — give the judge a gold answer or rubric to compare against; reduces the judge's own-knowledge errors.

Known **biases** (internal-validity threats):
- **Position bias** — favoring a fixed slot (often the first).
- **Verbosity / length bias** — longer answers rated higher regardless of quality.
- **Self-preference** — a judge prefers text from its own model family.
- **Format / sycophancy** — markdown, confident tone, or agreeing with the prompt inflate scores.

**Function PairwiseJudge(prompt $x$, answers $A$, $B$, judge $J$):**  → winner
1. Score both orderings to cancel position bias:
	1. $r_1 = J(x, A, B)$  (A shown first)
	2. $r_2 = J(x, B, A)$  (B shown first)
2. If $r_1$ and $r_2$ pick the same answer → that answer wins
3. Else → declare a **tie** **(WHY? consistency across the swap is the evidence the preference is about content, not slot order.)**
4. (Optional) require the judge to give reasoning *before* its verdict — chain-of-thought raises agreement with humans

Then **validate**: sample a subset, collect human labels, report judge↔human $\kappa$ (or correlation); only scale the judge once agreement is acceptable.

---

## 5. Decoding effects — the model is not one fixed function

The same weights give different scores depending on how you decode:
- **Greedy / low temperature** — deterministic; best for pass@1 and reproducibility.
- **Sampling ($t > 0$)** — needed for pass@k and diversity.
- **Self-consistency** — for reasoning, sample $m$ chains and majority-vote the final answer; trades compute for accuracy.

**Function SelfConsistency(model $M$, prompt $x$, samples $m$):**
1. For $j = 1..m$: sample a reasoning chain, extract its final answer $a_j$
2. Return $\text{mode}(\{a_1, \dots, a_m\})$  (most frequent answer)

Always report the decoding config with a score — "accuracy" is undefined without it.

---

## 6. Statistical rigor

- **Uncertainty.** A benchmark score is a sample mean, so report a confidence interval. For a proportion (accuracy) the standard error is $\sqrt{\hat p(1-\hat p)/N}$; for compound metrics use the **bootstrap**.
- **Comparisons.** To claim A > B, test the *difference* — paired, since both models see the same items — not just point estimates. Across many benchmarks, correct for multiple comparisons.
- **Prompt sensitivity.** Scores move with phrasing, few-shot examples, and option order; report the prompt and, ideally, average over variants.

**Function BootstrapCI(per-item scores $s_1..s_N$, resamples $B$):**
1. For $b = 1..B$: draw $N$ items with replacement; record the resample mean $\bar s_b$
2. Return the 2.5th and 97.5th percentiles of $\{\bar s_b\}$ (a 95% CI)

---

## 7. Threats to validity (first-principles checklist)

- **Contamination / leakage** — test items (or near-duplicates) in pretraining data inflate scores. Detect via canary strings, n-gram overlap against training data, or fresh/held-out sets.
- **Goodhart's law** — once a benchmark is a target it stops measuring the capability (models over-fit to it). Rotate, refresh, or keep sets private/dynamic.
- **Construct mismatch** — high BLEU ≠ good translation; high MMLU ≠ good assistant. Match the metric to the deployment construct.
- **Judge circularity** — an LLM judge can bake in its own model's biases and errors; always validate against humans.
- **Distribution shift** — a static academic set may not resemble deployment inputs.

---

## 8. The end-to-end pipeline

**Function Evaluate(model $M$, dataset $D$, scorer $s$):**
1. For each item $(x_i, y_i^*)$ in $D$:
	1. Render the prompt from a fixed template (instructions + few-shot examples)
	2. Generate $\hat y_i = M(x_i)$ with a fixed, **reported** decoding config
	3. Parse $\hat y_i$ into the answer field (extract letter / code block / JSON)
	4. Score $s_i = s(\hat y_i, y_i^*)$
2. Aggregate: report $\bar s = \frac1N\sum_i s_i$ with a bootstrap CI
3. Log the prompt template, decoding config, and scorer version next to the number **(WHY? parsing and prompt choices routinely move scores by several points, so a bare number is neither reproducible nor comparable.)**

---

## 9. Benchmark landscape (quick reference)

| Benchmark | Measures | Scorer type |
|---|---|---|
| MMLU | broad knowledge, 57 subjects | MCQ (exact / logprob) |
| GSM8K | grade-school math word problems | answer match (often self-consistency) |
| HumanEval / MBPP | code generation | pass@k (unit tests) |
| GPQA | hard graduate science, contamination-resistant | MCQ |
| HELM / BIG-bench | broad multi-task, multi-metric suites | mixed |
| MT-Bench | multi-turn chat quality | LLM-as-judge |
| Chatbot Arena | human head-to-head preference | pairwise → Elo / Bradley-Terry |
| TruthfulQA | truthfulness / hallucination resistance | MCQ + judge |
| Safety / red-team sets | refusal, policy compliance | judge + human |
