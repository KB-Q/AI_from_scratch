
# NLP Basics

Notation used throughout:
- Corpus $C$ = collection of documents. Document $d$ = sequence of tokens $w_1, w_2, ..., w_n$.
- Vocabulary $V$ = set of unique tokens. $|V|$ = vocab size.
- $count(x)$ = number of times $x$ occurs in the corpus.

---

## 1. Text Preprocessing

Raw text is messy; every representation below assumes a cleaned token stream. Typical pipeline:

1. **Tokenization**: split text into tokens (words / subwords / characters). "don't" → `["do", "n't"]` or `["don't"]` depending on tokenizer.
2. **Normalization**: lowercasing, unicode normalization, stripping punctuation/accents.
3. **Stop-word removal**: drop high-frequency low-information words ("the", "is", "of"). Optional — hurts tasks where function words matter.
4. **Stemming**: chop suffixes with crude rules (Porter stemmer). "running" → "run", "studies" → "studi". Fast, not linguistically valid.
5. **Lemmatization**: map to dictionary base form using morphology + POS. "better" → "good", "studies" → "study". Slower, correct.
6. **Subword tokenization** (modern): BPE / WordPiece / SentencePiece split rare words into frequent sub-units so the vocab stays bounded and there are no out-of-vocabulary (OOV) tokens.

Stemming vs lemmatization: stemming is a heuristic string chop; lemmatization is a dictionary lookup that respects part of speech.

---

## 2. Bag of Words (BoW)

Represent a document as a vector of token counts over $V$, discarding word order.

- Vector length = $|V|$. Entry $i$ = count (or presence 0/1) of vocab word $i$ in the document.
- Pros: simple, works with any linear model. Cons: ignores order and meaning, very sparse, large vocab.

Example over $V = \{cat, dog, sat\}$: "cat sat sat" → $[1, 0, 2]$.

---

## 3. N-grams

An **n-gram** is a contiguous sequence of $n$ tokens. It re-injects a little local word order into BoW-style models.

- $n=1$ unigram: `["the", "cat", "sat"]`
- $n=2$ bigram: `["the cat", "cat sat"]`
- $n=3$ trigram: `["the cat sat"]`

Use n-grams as features (instead of / alongside unigrams) to capture short phrases like "not good". Trade-off: feature space grows toward $|V|^n$, so higher $n$ means sparser, higher-variance features.

---

## 4. TF-IDF

Raw counts over-weight common words. **TF-IDF** down-weights terms that appear in many documents (they discriminate poorly) and up-weights terms frequent in a document but rare across the corpus.

- Term Frequency: $tf(t, d) = \dfrac{count(t, d)}{\sum_{t'} count(t', d)}$  (term count in doc, normalized by doc length)
- Inverse Document Frequency: $$idf(t) = \log \frac{N}{1 + df(t)}$$ where $N$ = number of documents, $df(t)$ = number of documents containing $t$. **(WHY? the $+1$ smoothing and the log — log dampens the effect so a term in 1 doc vs 10 docs isn't weighted linearly)**
- $tfidf(t, d) = tf(t, d) \cdot idf(t)$

**Function ComputeTFIDF(corpus C):**
1. Build vocabulary $V$ from all tokens in $C$
2. For each term $t$ in $V$:
	1. $df(t) = $ number of documents in $C$ containing $t$
	2. $idf(t) = \log ( N / (1 + df(t)) )$
	3. Repeat for each term $t$
3. For each document $d$ in $C$:
	1. $len(d) = \sum_{t'} count(t', d)$
	2. For each term $t$ in $d$:
		1. $tf(t, d) = count(t, d) / len(d)$
		2. $tfidf(t, d) = tf(t, d) \cdot idf(t)$
		3. Repeat for each term $t$
	3. Repeat for each document $d$
4. Return matrix of $tfidf$ vectors (one row per document)

Similarity between two TF-IDF (or any) vectors is usually **cosine similarity**: $\cos(a, b) = \dfrac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$ — angle, not magnitude, so document length cancels out.

---

## 5. N-gram Language Models

A **language model (LM)** assigns a probability to a sequence, or predicts the next token given history. An n-gram LM makes the **Markov assumption**: the next token depends only on the previous $n-1$ tokens.

$$P(w_1, ..., w_m) = \prod_{i=1}^{m} P(w_i \mid w_{i-(n-1)}, ..., w_{i-1})$$

Maximum-likelihood estimate of each conditional (bigram case, $n=2$):
$$P(w_i \mid w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}$$

Problem: any n-gram unseen in training gets probability 0, killing the whole product. Fix with **smoothing**.

- **Add-k / Laplace smoothing**: $P(w_i \mid w_{i-1}) = \dfrac{count(w_{i-1}, w_i) + k}{count(w_{i-1}) + k|V|}$  (k=1 is Laplace)
- **Backoff / interpolation**: if the trigram is unseen, fall back to (or blend with) the bigram, then unigram estimate.

**Evaluation — Perplexity**: exponentiated average negative log-likelihood per token on held-out text. Lower is better; a perplexity of $p$ means the model is as confused as if choosing uniformly among $p$ words.
$$PP(W) = P(w_1, ..., w_m)^{-1/m} = \exp\left(-\frac{1}{m}\sum_{i=1}^{m} \log P(w_i \mid history)\right)$$

**Function TrainNgramLM(corpus C, n, k):**
1. Initialize count tables: $ctx\_count[\cdot]$, $ngram\_count[\cdot]$
2. For each document $d$ in $C$:
	1. Pad $d$ with $n-1$ start tokens `<s>` and one end token `</s>`
	2. For $i$ from $n$ to $len(d)$:
		1. $context = (w_{i-(n-1)}, ..., w_{i-1})$
		2. $ngram\_count[context, w_i] \mathrel{+}= 1$
		3. $ctx\_count[context] \mathrel{+}= 1$
		4. Repeat for each position $i$
	3. Repeat for each document $d$
3. Store probability rule with smoothing: $$P(w \mid context) = \frac{ngram\_count[context, w] + k}{ctx\_count[context] + k|V|}$$
4. Return the count tables + probability rule

**Function Generate(LM, n, max_len):**
1. $context = (n-1)$ start tokens `<s>`
2. For $i$ from 1 to max_len:
	1. Sample $w \sim P(\cdot \mid context)$ from the LM
	2. If $w$ is `</s>`: break
	3. Emit $w$; slide context = last $n-1$ tokens including $w$
	4. Repeat for each step $i$
3. Return generated sequence

---

## 6. Word Embeddings (distributed representations)

BoW/TF-IDF vectors are sparse and treat every word as unrelated ("cat" and "dog" are orthogonal). **Embeddings** map each word to a dense low-dimensional vector where semantically similar words are close. Based on the **distributional hypothesis**: words in similar contexts have similar meanings.

### Word2Vec

Two shallow-network training objectives:
- **CBOW** (continuous bag of words): predict the center word from the average of its context words. Faster, better on frequent words.
- **Skip-gram**: predict each context word from the center word. Better on rare words and small corpora.

Training the full softmax over $|V|$ is expensive, so use **negative sampling**: for each true (center, context) pair, treat it as positive and draw $K$ random "negative" words as negatives, turning it into $K+1$ binary logistic-regression problems. Two vectors per word are learned: $v_w$ (as center) and $u_w$ (as context).

For a pair, sigmoid score $\sigma(x) = 1/(1 + e^{-x})$. Objective per positive pair maximizes $\log \sigma(u_o \cdot v_c) + \sum_{k} \log \sigma(-u_{neg_k} \cdot v_c)$.

**Function TrainSkipGramNegSampling(corpus C, window m, dim D, negatives K, lr $\eta$):**
1. Initialize $v_w, u_w \in \mathbb{R}^D$ randomly for every $w$ in $V$
2. For each epoch:
	1. For each document $d$ in $C$:
		1. For each center position $i$ in $d$ (center word $c = w_i$):
			1. For each context offset $j$ in $[-m, m], j \neq 0$ (context word $o = w_{i+j}$):
				1. **Positive update**: $err = \sigma(u_o \cdot v_c) - 1$
				2. Accumulate gradient on center: $grad\_c \mathrel{+}= err \cdot u_o$
				3. $u_o \mathrel{-}= \eta \cdot err \cdot v_c$
				4. For $k$ from 1 to K:
					1. Sample negative word $neg \sim P_{noise}$ (unigram$^{3/4}$ distribution) **(WHY? the 3/4 power flattens the frequency distribution so rare words get sampled more)**
					2. $err_k = \sigma(u_{neg} \cdot v_c) - 0$
					3. $grad\_c \mathrel{+}= err_k \cdot u_{neg}$
					4. $u_{neg} \mathrel{-}= \eta \cdot err_k \cdot v_c$
					5. Repeat for each negative $k$
				5. $v_c \mathrel{-}= \eta \cdot grad\_c$
				6. Repeat for each context offset $j$
			2. Repeat for each center position $i$
		2. Repeat for each document $d$
	2. Repeat for each epoch
3. Return embeddings $\{v_w\}$ (typically the center vectors are used as the word embeddings)

### GloVe (brief)

Instead of local sliding windows, GloVe factorizes the global word-word co-occurrence count matrix. It fits embeddings so that $v_i \cdot v_j + b_i + b_j \approx \log(X_{ij})$, where $X_{ij}$ = number of times word $j$ appears in the context of word $i$, weighting rare/frequent co-occurrences with a capping function. Result is similar to Word2Vec but trained on aggregate statistics.

Embeddings famously support analogies via vector arithmetic: $v_{king} - v_{man} + v_{woman} \approx v_{queen}$.

---

## 7. Topic Modeling — LDA

**Topic modeling** discovers latent themes ("topics") in a corpus unsupervised. A **topic** is a probability distribution over the vocabulary; a **document** is a mixture of topics.

**Latent Semantic Analysis (LSA)** — the precursor — just runs truncated SVD on the TF-IDF matrix to get a low-rank approximation; topics are linear-algebra components, not probabilistic.

**Latent Dirichlet Allocation (LDA)** is a generative probabilistic model. Its **generative story** — how it imagines each document was written:

1. For each topic $k$ in $1..K$: draw a word distribution $\phi_k \sim \text{Dirichlet}(\beta)$
2. For each document $d$:
	1. Draw a topic mixture $\theta_d \sim \text{Dirichlet}(\alpha)$
	2. For each word position in $d$:
		1. Draw a topic $z \sim \text{Multinomial}(\theta_d)$
		2. Draw the word $w \sim \text{Multinomial}(\phi_z)$

$\alpha, \beta$ are Dirichlet priors: small $\alpha$ → each doc concentrated on few topics; small $\beta$ → each topic concentrated on few words. Training inverts this story to infer the hidden $z$, $\theta$, $\phi$ from observed words. A common inference method is **collapsed Gibbs sampling**, which integrates out $\theta$ and $\phi$ and just resamples the topic assignment of each word.

**Function LDACollapsedGibbs(corpus C, K, $\alpha$, $\beta$, iterations):**
1. Initialize: assign each word token a random topic $z \in 1..K$
2. Build count tables:
	1. $n_{d,k}$ = number of words in doc $d$ assigned to topic $k$
	2. $n_{k,w}$ = number of times word $w$ is assigned to topic $k$
	3. $n_k$ = total words assigned to topic $k$
3. For each iteration:
	1. For each document $d$:
		1. For each word token $w$ at position with current topic $z$:
			1. Remove current assignment: decrement $n_{d,z}, n_{z,w}, n_z$ by 1
			2. For each topic $k$ in $1..K$: compute unnormalized probability **(WHY? this is the collapsed conditional $P(z=k \mid \text{rest})$, product of a doc-topic term and a topic-word term)** $$p(k) \propto \frac{n_{d,k} + \alpha}{\sum_{k'}(n_{d,k'} + \alpha)} \cdot \frac{n_{k,w} + \beta}{n_k + |V|\beta}$$
			3. Sample new topic $z' \sim p(\cdot)$ after normalizing
			4. Add new assignment: increment $n_{d,z'}, n_{z',w}, n_{z'}$ by 1
			5. Repeat for each word token
		2. Repeat for each document $d$
	2. Repeat for each iteration
4. Estimate outputs:
	1. Topic-word: $\phi_{k,w} = (n_{k,w} + \beta) / (n_k + |V|\beta)$
	2. Doc-topic: $\theta_{d,k} = (n_{d,k} + \alpha) / (\sum_{k'} n_{d,k'} + K\alpha)$
5. Return $\phi$ (topics as word distributions), $\theta$ (docs as topic mixtures)

---

## 8. Other Fundamentals (quick reference)

- **POS tagging**: label each token with its part of speech (noun, verb, ...). Classic model: HMM or CRF.
- **Named Entity Recognition (NER)**: tag spans as PERSON / ORG / LOCATION, usually with BIO tagging (`B-`egin, `I-`nside, `O`utside).
- **One-hot encoding**: a word as a $|V|$-length vector, 1 at its index, 0 elsewhere. The sparse baseline embeddings improve on.
- **Cosine similarity**: the standard vector-similarity measure in NLP (see §4).
- **OOV (out-of-vocabulary)**: tokens unseen in training; handled by an `<UNK>` token or subword tokenization.
- **Edit (Levenshtein) distance**: min insert/delete/substitute operations to turn one string into another; used for spelling correction.
- **BLEU / ROUGE**: n-gram overlap metrics for machine translation / summarization quality.
- **Zipf's law**: word frequency is inversely proportional to rank — a few words are extremely common, a long tail is rare. Explains why smoothing and subwords matter.
