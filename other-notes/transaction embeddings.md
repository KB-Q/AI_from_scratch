# Transaction Embedding Methods

Methods for turning transaction / event data into fixed-length vector representations for downstream tasks (fraud, churn, credit, anomaly detection). They differ mainly in how they *view* the data — as a per-entity event sequence, a graph, sequential tabular records, or a multivariate time series — and in the self-supervision signal used to train without labels.

| Method | Data view | Backbone encoder | What it outputs |
|---|---|---|---|
| CoLES | one event sequence per entity (e.g. a card's transactions) | event MLP + GRU (RNN) | one vector per sequence/entity |
| GraphSAGE | graph (nodes = entities, edges = interactions) | sample + aggregate (GNN) | one vector per node |
| FATA-Trans | sequential tabular records (static + dynamic fields) | two field transformers + BERT | one vector per record (and sequence) |
| TS2Vec | multivariate time series | input projection + dilated CNN | per-timestamp vectors (pool → any granularity) |

---

## 1. CoLES — Contrastive Learning for Event Sequences

Self-supervised (no labels). Encodes a user's whole event sequence into one vector by **metric/contrastive learning**: sub-sequences drawn from the *same* user are treated as positive pairs (should embed close), sub-sequences from *different* users as negatives (should embed far). The encoder is an event-level network feeding a GRU whose last hidden state is the sequence embedding.

**Function EventEncoder(event $x_t$):**  → event vector $z_t$
1. For each categorical attribute of the event: look up its embedding vector
2. For each numerical attribute: apply batch normalization
3. Concatenate all attribute outputs → event embedding $z_t$

**Function SequenceEncoder($z_{1:T}$):**  → sequence vector $c_T$
1. $c_0$ = learned initial state
2. For $t$ from 1 to $T$: $c_t = \text{GRU}(z_t, c_{t-1})$
3. Return the last state $c_T$ (embedding of the whole sequence)

Inputs: a set of event sequences (one per entity); encoder $M = \text{SequenceEncoder} \circ \text{EventEncoder}$; sub-sequences per entity $K$, entities per batch $N$

Output: trained encoder $M$ producing $c_T \in \mathbb{R}^d$ per sequence

Steps:
1. Build a batch:
	1. Randomly pick $N$ sequences
	2. For each sequence, sample $K$ sub-sequences as contiguous random **slices** (event order preserved) **(WHY? sub-sequences of the same entity are used as positive views; this augmentation exploits the periodicity / repeatability of a user's behavior, which is what makes label-free self-supervision valid here)**
	3. Pair labels: sub-sequences from the same sequence = positive, from different sequences = negative → batch has $N \times K$ samples
2. Encode every sub-sequence: per event $z_t = \text{EventEncoder}(x_t)$, then $c_T = \text{SequenceEncoder}(z_{1:T})$
3. Contrastive / metric loss: pull positive pairs' $c_T$ together, push negatives apart; update $M$
4. Repeat over batches
5. Inference: for any sequence, $c_T = M(\text{sequence})$ is its embedding (the RNN also lets you *update* an existing embedding with new events instead of recomputing from scratch)

---

## 2. GraphSAGE — Sample and Aggregate

**Inductive** node embedding: instead of learning one fixed vector per node (transductive), it learns *aggregator functions* + weight matrices that generate a node's embedding from its **features and local neighborhood** — so it produces embeddings for nodes (or whole graphs) unseen at training time. For transactions: nodes = cards / accounts / merchants, edges = transactions between them; features = node attributes.

**Function GraphSAGE-Embed(graph $G$, node features $\{x_v\}$, depth $K$):**
1. Initialize base representations: $h_v^{0} = x_v$ for all nodes $v$
2. For $k$ from 1 to $K$:  (each step reaches one hop further out)
	1. For each node $v$:
		1. Sample a **fixed-size** neighbor set $N(v)$ (uniform draw) **(WHY? sampling a constant number of neighbors per depth bounds per-batch compute and memory regardless of node degree — this is what lets it scale to huge graphs and stay inductive)**
		2. Aggregate neighbors: $h_{N(v)}^{k} = \text{AGGREGATE}_k\big(\{\, h_u^{k-1} : u \in N(v) \,\}\big)$
		3. Concatenate own + neighborhood, then transform: $h_v^{k} = \sigma\big(W^{k} \cdot [\, h_v^{k-1} \,\|\, h_{N(v)}^{k} \,]\big)$
		4. Normalize: $h_v^{k} = h_v^{k} / \lVert h_v^{k} \rVert_2$
		5. Repeat for each node $v$
	2. Repeat for each depth $k$
3. Return node embeddings $z_v = h_v^{K}$ for all $v$

Aggregator choices for step 2.1.2:
- **Mean** (convolutional variant): $h_v^{k} = \sigma\big(W \cdot \text{MEAN}(\{h_v^{k-1}\} \cup \{h_u^{k-1} : u \in N(v)\})\big)$ — folds own + neighbors into one mean, skipping the explicit concat
- **Pooling**: $\max\big(\{\, \sigma(W_{pool}\, h_u^{k-1} + b) : u \in N(v) \,\}\big)$
- **LSTM**: apply an LSTM to neighbors in random order

Training note (brief): $W^{k}$ and the aggregators are learned by SGD. Unsupervised objective pulls nodes that co-occur on short random walks together and pushes negatively-sampled nodes apart; it can be swapped for a supervised task loss.

---

## 3. FATA-Trans — Field- And Time-Aware Transformer

A transformer for **sequential tabular** data (a user's records over time), where each record (row) has **static fields** (constant across the sequence, e.g. account attributes) and **dynamic fields** (change per record, e.g. amount, merchant). Two-level architecture: field transformers build a per-record embedding, then a BERT-style encoder builds sequence embeddings, injecting field-type and real-time information.

Notation: sequence $X = [x_0, ..., x_{l-1}]$; record $x_i$ = static fields + dynamic fields; $t_i$ = time interval between record $x_i$ and $x_0$ ($t_0 = 0$).

**Level 1 — field transformers (build a record embedding):**
1. **Static Field Transformer**: encode the static fields once (they're constant over the sequence) → static record embedding $TE_S$
2. **Dynamic Field Transformer**: for each record $i$, encode its dynamic fields → dynamic record embedding $TE_{D,i}$

   **(WHY? static and dynamic fields go through separate transformers instead of copying the static values into every record — this removes the compute overhead and the artificially-easy masked-field pre-training that replication-based models like TabBERT suffer from)**

**Level 2 — FATA-BERT (build sequence embeddings across records):**
3. For each record, form its input embedding as the element-wise sum of three parts:
	1. record embedding: $TE_S$ (static) or $TE_{D,i}$ (dynamic)
	2. **field-type embedding**: $FE_S$ for static, $FE_D$ for dynamic — tells the encoder which type each is
	3. **time-aware position embedding** $P(i)$: a length-$d$ vector from a trainable function $TPos(i)$ that merges the order index $i$ **and** the time interval $t_i$ (params $w_p, w_t, b$); static fields use $P(0)$ **(WHY? encodes both sequence order and the actual elapsed time between transactions, so irregular time gaps shape the representation)**
	- so $IE_S = TE_S + FE_S + P(0)$  and  $IE_{D,i} = TE_{D,i} + FE_D + P(i)$
4. Feed $[\,IE_S, IE_{D,0}, IE_{D,1}, ..., IE_{D,l-1}\,]$ into **FATA-BERT** (BERT-style transformer encoder)
5. Output sequence embeddings $[\,SE_S, SE_{D,0}, ..., SE_{D,l-1}\,]$; use for downstream tasks (pre-trained via masked-field prediction, fine-tuned per task)

---

## 4. TS2Vec — Hierarchical Contrastive Time-Series Representation

Universal, self-supervised representation for time series. Produces a vector **per timestamp**; a representation for any sub-series (or the whole series) is obtained by max-pooling over the relevant timestamps. Trained by **contextual consistency**: the same timestamp encoded under two different augmented "contexts" should match, contrasted at multiple time scales.

**Function Encoder(series $x$):**  → per-timestamp representations $r$
1. **Input projection** (fully connected): $z_{t} = W x_{t} + b$ — map each observation to a latent vector
2. **Timestamp masking**: mask $z$ at randomly chosen timestamps (Bernoulli) to produce an augmented context view **(WHY? masking is applied to the latent vectors, not raw values — time-series values are unbounded, so there is no valid "mask token" for the raw input)**
3. **Dilated CNN** (10 residual blocks, block $l$ uses dilation $2^{l}$ → large receptive field): produce contextual representation $r_{t}$ per timestamp
4. Return $r$

Inputs: time series; the Encoder above

Output: trained encoder producing per-timestamp representations

Steps:
1. For each input series, sample two **overlapping** crops $[a_1, b_1]$ and $[a_2, b_2]$ with $a_1 \le a_2 \le b_1 \le b_2$; the overlap $[a_2, b_1]$ is shared by both views **(WHY? the two views of the overlap are the positive pair; random cropping forces position-agnostic representations and prevents representation collapse)**
2. Encode both crops → representations $r$ and $r'$ on the overlapping segment
3. Compute the hierarchical contrastive loss (below) over $r, r'$; update the encoder
4. Repeat over batches

**Function HierLoss($r$, $r'$):**
1. $L = L_{dual}(r, r')$, where $L_{dual}$ = temporal contrastive loss + instance-wise contrastive loss:
	1. **temporal**: same timestamp across the two views = positive; different timestamps of the same series = negative
	2. **instance-wise**: at a given timestamp, the two views of the same series = positive; other series in the batch = negative
2. While the time length of $r$ > 1:
	1. $r = \text{maxpool1d}(r, \text{kernel}=2)$;  $r' = \text{maxpool1d}(r', \text{kernel}=2)$
	2. $L = L + L_{dual}(r, r')$
	3. Repeat — each pooling step is a coarser time scale **(WHY? pooling then re-contrasting at every scale makes the representation capture structure from fine timestamps up to the whole instance)**
3. Return $L$ averaged over the levels

Inference: encode → per-timestamp representations; max-pool over a chosen window (or the full series) to get coarser-granularity embeddings.
