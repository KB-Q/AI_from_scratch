
# Transformers — Full Flow & Attention Variants

Covers (a) the complete text → token flow of a decoder-only (GPT-style) transformer, and (b) the formulae for the common attention / connection variants and where each slots into that flow.

### Notation & convention

Token vectors are **rows**: $X \in \mathbb{R}^{s \times d}$ ($s$ = sequence length, $d = d_{model}$). This is the convention the variant papers below use, so attention reads $A = \text{softmax}(Q K^\top / \sqrt{d_k})$ and the residual update reads $X' = X + A V$.

Some treatments use the transposed **column** convention (token vectors as columns, $E \in \mathbb{R}^{d \times s}$), where the same operations become $A = \text{softmax}(K^\top Q / \sqrt{d_k})$ and $E' = E + V A$. The two are equivalent up to transposition:

| Quantity | Column convention | Row convention (used here) |
|---|---|---|
| Sequence | $E \in (d, s)$ | $X \in (s, d)$ |
| Attention weights | $A = \text{softmax}(K^\top Q / \sqrt{d_k})$ | $A = \text{softmax}(Q K^\top / \sqrt{d_k})$ |
| Update | $E' = E + V A$ | $X' = X + A V$ |

---

## 1. The full flow (decoder-only / GPT-style)

Three pieces are easy to leave out but are what actually make the architecture work: **positional information**, and — around *both* the attention and the FFN — a **residual (skip) connection** and a **normalization**. The unit that repeats $L$ times is `MHA + FFN`, each sub-layer wrapped in norm + residual. End to end:

Inputs:
- Prompt text
- Token embedding matrix $W_E \in \mathbb{R}^{|V| \times d}$ ($|V|$ = vocab size)
- Positional scheme (absolute table $P$, or RoPE applied inside attention — see §5)
- $L$ transformer blocks; each has attention params ($W_Q, W_K, W_V, W_O$) and FFN params ($W_1, b_1, W_2, b_2$) and norm params
- Unembedding matrix $W_U \in \mathbb{R}^{d \times |V|}$ (often **weight-tied**: $W_U = W_E^\top$)

Output: probability distribution over the next token (loop to generate a sequence)

Steps:
1. **Tokenize**: text → token IDs $[t_1, ..., t_s]$ (subword tokenizer: BPE / WordPiece)
2. **Embed**: look up each ID → $X = [\,W_E[t_1]; \; ...; \; W_E[t_s]\,] \in \mathbb{R}^{s \times d}$ (one row per token) — this is the **residual stream**
3. **Add positional info** (absolute case): $X = X + P$ (skip if using RoPE, which is applied to $Q, K$ inside attention instead)
4. For $\ell$ from 1 to $L$:  (one transformer block)
	1. **Attention sub-layer** (pre-norm form):
		1. $\tilde{X} = \text{Norm}(X)$
		2. $\text{attn} = \textbf{MHA}(\tilde{X})$  ← §2–3
		3. $X = X + \text{attn}$  (residual add)
	2. **FFN sub-layer** (pre-norm form):
		1. $\tilde{X} = \text{Norm}(X)$
		2. $\text{ff} = \textbf{FFN}(\tilde{X})$  ← §4
		3. $X = X + \text{ff}$  (residual add)
	3. Repeat for each block $\ell$
5. **Final norm**: $X = \text{Norm}(X)$
6. **Unembed (LM head)**: $\text{logits} = X \cdot W_U \in \mathbb{R}^{s \times |V|}$
7. **Next-token distribution**: $p = \text{softmax}(\text{logits}[\,\text{last row}\,])$ (only the last position matters for generating the next token)
8. **Select** token (argmax, or sample with temperature / top-k / top-p); append it and repeat from step 1 to generate autoregressively

Why residual + norm matter: the residual stream lets gradients flow straight through $L$ deep blocks (no vanishing), and each sub-layer only writes a *delta* onto it. Normalization keeps activation scale stable. **Pre-norm** (Norm *inside* the branch, as above) is standard in modern LLMs because it trains stably at depth; the original 2017 paper used **post-norm** ($X = \text{Norm}(X + \text{sublayer}(X))$).

---

## 2. Scaled dot-product attention (one head)

Each query attends to all keys; the softmax turns compatibility scores into a weighted average of values. For per-head matrices $Q, K \in \mathbb{R}^{s \times d_k}$ and $V \in \mathbb{R}^{s \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

- $Q K^\top \in \mathbb{R}^{s \times s}$: entry $(j,k)$ = how much token $j$ (query) attends to token $k$ (key).
- $/\sqrt{d_k}$: **(WHY? the dot product of two $d_k$-dim vectors has variance $\propto d_k$; dividing keeps logits ~unit-scale so softmax doesn't saturate into vanishing gradients)**
- $M$ = **causal mask**: $M_{jk} = -\infty$ for $k > j$, else $0$. After softmax those become $0$, so a token can't attend to future tokens (required for autoregressive generation).
- softmax is row-wise (each query's weights over keys sum to 1).

---

## 3. Multi-head attention (MHA)

Run $h$ attention heads in parallel on different learned projections of the input, so different heads specialize (syntax, coreference, position, ...), then recombine. With $d_k = d_v = d / h$:

**Function MHA(X):**  ($X \in \mathbb{R}^{s \times d}$)
1. Project to queries, keys, values: $Q = X W_Q,\; K = X W_K,\; V = X W_V$  (each $\in \mathbb{R}^{s \times d}$)
2. Split each along the feature dim into $h$ heads: $Q \to (Q^1, ..., Q^h)$, likewise $K, V$; each $Q^i \in \mathbb{R}^{s \times d_k}$
3. For each head $i$ in $1..h$:
	1. scores: $S^i = Q^i (K^i)^\top / \sqrt{d_k}$  ($\in \mathbb{R}^{s \times s}$)
	2. apply causal mask $M$
	3. weights: $A^i = \text{softmax}(S^i + M)$  (row-wise)
	4. head output: $O^i = A^i V^i$  ($\in \mathbb{R}^{s \times d_k}$)
	5. Repeat for each head $i$
4. Concatenate heads: $O = [\,O^1 \,|\, O^2 \,|\, ... \,|\, O^h\,] \in \mathbb{R}^{s \times d}$
5. **Combine via output matrix**: return $O \, W_O$  ($W_O \in \mathbb{R}^{d \times d}$)

The whole MHA operation as a one-liner. Note $W_O$ multiplies the **concatenation of all $h$ head outputs**, once — it is not a per-head projection. Here $W_Q^i, W_K^i \in \mathbb{R}^{d \times d_k}$ and $W_V^i \in \mathbb{R}^{d \times d_v}$ are the per-head slices of $W_Q, W_K, W_V$, and $[\,\cdot \mid \cdot\,]$ is column-wise concatenation over heads:

$$\text{MHA}(X) = \Big[\, O^1 \;\big|\; \cdots \;\big|\; O^h \,\Big]\, W_O, \qquad O^i = \text{softmax}\!\left(\frac{(X W_Q^i)(X W_K^i)^\top}{\sqrt{d_k}} + M\right) X W_V^i$$

Equivalently, since $W_O \in \mathbb{R}^{d \times d}$ splits into $h$ row-blocks $W_O^i \in \mathbb{R}^{d_v \times d}$, "concat then project" equals a sum of independent per-head contributions:

$$\text{MHA}(X) = \sum_{i=1}^{h} O^i\, W_O^i$$

In a block this sits inside the residual and norm: $X' = X + \text{MHA}(\text{Norm}(X))$.

**KV cache** (inference): during generation, $K$ and $V$ for past tokens are recomputed every step unless cached. Caching them makes decoding $O(1)$ new columns per step instead of $O(s)$ — but the cache costs $2 \cdot n_h \cdot d_h \cdot l$ elements per token. Shrinking this cache is what motivates MQA / GQA / MLA (§6).

---

## 4. FFN, normalization (the per-token glue)

**Feed-forward network** — applied to each token position independently; this is where most parameters and most "stored knowledge" live:
$$\text{FFN}(x) = \sigma(x W_1 + b_1)\,W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d \times d_{ff}},\; W_2 \in \mathbb{R}^{d_{ff} \times d}$$
Usually $d_{ff} = 4d$. $\sigma$ = ReLU (original), **GELU** (BERT/GPT-2), or **SwiGLU** (modern Llama/PaLM): $\text{SwiGLU}(x) = (\text{Swish}(x W_1) \odot x W_3) W_2$ — a gated variant with an extra matrix.

**Normalization**:
- **LayerNorm**: $\text{LN}(x) = \gamma \odot \dfrac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ (per-token mean $\mu$ / variance $\sigma^2$ over features).
- **RMSNorm** (modern default — cheaper, no mean subtraction): $\text{RMS}(x) = \gamma \odot \dfrac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$.

---

## 5. Positional information

Self-attention is permutation-invariant, so order must be injected.
- **Absolute** (original): add a fixed sinusoidal or learned table $P$ to $X$ once at the input (flow step 3).
- **RoPE** (Rotary Position Embedding — modern default): rotate each pair of dimensions of $q$ and $k$ by an angle proportional to the token's position, *inside* attention before the dot product. Because the score depends on the rotation *difference*, it encodes **relative** position and extrapolates better to long contexts. Fits at MHA step 1 (between projecting $Q,K$ and computing scores). RoPE's incompatibility with KV-compression is exactly what forces MLA's "decoupled RoPE" (§7).

---

## 6. KV-cache attention variants — MQA, GQA, MLA

All three attack the same cost: the KV cache. They change **how keys/values are produced/shared** — they slot into MHA steps 1–2, leaving the rest of the flow unchanged.

### MQA / GQA

- **MHA**: $n_h$ query heads, $n_h$ key heads, $n_h$ value heads.
- **MQA** (Multi-Query Attention): $n_h$ query heads share **one** K head and **one** V head.
- **GQA** (Grouped-Query Attention): query heads are split into $g$ groups; each group shares one K/V head. It interpolates: $g = 1$ is MQA, $g = n_h$ is MHA.

Only the number of distinct K/V heads changes; per head the computation is still §2. KV cache per token (elements, over $l$ layers):

| Mechanism | K/V heads | KV cache / token |
|---|---|---|
| MHA | $n_h$ | $2\, n_h\, d_h\, l$ |
| GQA | $g$ groups | $2\, g\, d_h\, l$ |
| MQA | $1$ | $2\, d_h\, l$ |
| MLA | latent | $(d_c + d_h^R)\, l$ |

### MLA (Multi-head Latent Attention, DeepSeek-V2)

Instead of sharing K/V heads, MLA **compresses** keys and values jointly into a small latent vector $c^{KV}$ that is cached; per-head K and V are reconstructed on the fly by up-projection. Because RoPE can't be absorbed into the up-projections, position is carried on a separate small **decoupled** key/query pair. Per token $h_t$:

- KV down-projection (this latent is what gets cached): $$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll n_h d_h$$
- K/V up-projections (content part, "nope" = no positional encoding): $$k_t^{C} = W^{UK} c_t^{KV}, \qquad v_t^{C} = W^{UV} c_t^{KV}$$
- Query compression (training-memory saving): $c_t^{Q} = W^{DQ} h_t$, then $q_t^{C} = W^{UQ} c_t^{Q}$
- Decoupled RoPE — a shared key $k_t^R$ and per-head queries $q_t^R$ that carry position: $$q_t^{R} = \text{RoPE}(W^{QR} c_t^{Q}), \qquad k_t^{R} = \text{RoPE}(W^{KR} h_t)$$
- Concatenate content + positional parts per head: $q_{t,i} = [\,q_{t,i}^{C}; q_{t,i}^{R}\,]$, $\;k_{t,i} = [\,k_{t,i}^{C}; k_{t}^{R}\,]$
- Attention & output: $$o_{t,i} = \sum_{j \le t} \text{softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^{R}}}\right) v_{j,i}^{C}, \qquad u_t = W^{O}[\,o_{t,1}; ...; o_{t,n_h}\,]$$

**(WHY? only $c_t^{KV}$ and the shared $k_t^R$ are cached, not full per-head K/V — cache shrinks to $(d_c + d_h^R)$ per token per layer, ~like GQA with ~2.25 groups, while quality matches or beats MHA)**. Trick: at inference $W^{UK}$ can be absorbed into $W^{Q}$ and $W^{UV}$ into $W^{O}$, so full K/V need never be materialized.

---

## 7. Linear-attention / recurrent variant — KDA

Softmax attention is $O(s^2)$ in sequence length and needs a growing KV cache. **Linear attention** replaces the softmax with a **recurrent matrix-valued state** $S$ that accumulates key→value associations, giving $O(s)$ time and a **fixed-size** state (no growing cache). KDA is the current refinement of this line. It **replaces the whole attention operation** (MHA steps 1–5) with a recurrence; in Kimi Linear it is interleaved with full-attention (MLA) layers at a 3:1 ratio, since pure linear attention is weaker at exact long-range retrieval.

The lineage (each row adds one degree of control over the state update $S_t$):

| Method | State update $S_t$ |
|---|---|
| Linear attention | $S_{t-1} + k_t v_t^\top$ |
| RetNet | $\alpha\, S_{t-1} + \beta_t k_t v_t^\top$ (fixed scalar decay) |
| Mamba2 | $\alpha_t\, S_{t-1} + \beta_t k_t v_t^\top$ (data-dependent scalar decay) |
| GLA | $\text{Diag}(\alpha_t)\, S_{t-1} + k_t v_t^\top$ (per-channel decay) |
| DeltaNet | $(I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ (delta-rule erase+write) |
| Gated DeltaNet | $\alpha_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ (scalar decay + delta) |
| **KDA** | $(I - \beta_t k_t k_t^\top)\,\text{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top$ (per-channel decay + delta) |

**Kimi Delta Attention (KDA)** — per-channel forget gate $\alpha_t \in [0,1]^{d_k}$ (each feature decays at its own rate) combined with the delta rule (erase the old value at $k_t$, write the new one, strength $\beta_t \in (0,1)$):
$$S_t = \left(I - \beta_t k_t k_t^\top\right)\text{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top, \qquad o_t = S_t^\top q_t$$
Inputs and output gate around the recurrence:
$$q_t, k_t = \text{L2Norm}(\text{Swish}(\text{ShortConv}(W_{q/k}\, x_t))), \quad v_t = \text{Swish}(\text{ShortConv}(W_v\, x_t))$$
$$\alpha_t = f(W_\alpha^{\uparrow} W_\alpha^{\downarrow} x_t) \in [0,1]^{d_k}, \quad \beta_t = \text{Sigmoid}(W_\beta x_t) \in (0,1)$$
$$o_t = W_o\Big[\text{Sigmoid}(W_g^{\uparrow} W_g^{\downarrow} x_t) \odot \text{RMSNorm}\big(\text{KDA}(q_t, k_t, v_t, \alpha_t, \beta_t)\big)\Big]$$
**(WHY? the update is a constrained Diagonal-Plus-Low-Rank transition $S_t = (D_t - a_t b_t^\top)S_{t-1} + k_t v_t^\top$ with $D_t = \text{Diag}(\alpha_t),\, a_t = \beta_t k_t,\, b_t = k_t \odot \alpha_t$; tying $a, b$ to $k$ is what makes the chunkwise kernel ~2× faster than general DPLR)**

---

## 8. Connection / residual variants

These change **flow step 4.i.3 / 4.ii.3** — the residual "Add" — or add extra skip paths. The plain residual $X = X + \text{sublayer}(\tilde X)$ is a special case of all of them.

### Attention residuals

Extra skip connections involving attention internals across depth, to fight over-smoothing (deep tokens collapsing toward each other) and dilution of early-layer information. Two forms in use:

- **Value residual (ResFormer / SVFormer)** — before attention, blend the current layer's value with the **first** layer's value $V_1$ (they share the current attention matrix): $$V_n = \lambda_{n,1} V_1 + \lambda_{n,2}\, H_{n-1} W_n^{V}$$ with $\lambda$ scalars (constant or learnable). SVFormer takes it to the extreme — all layers reuse $V_1$ — halving the KV cache.
- **AttnRes (Kimi K3, block variant)** — replace the fixed residual *sum across depth* with a **softmax attention over preceding block outputs**, so each block selectively retrieves earlier representations with learned, input-dependent weights (KDA = selective forgetting across *time*; AttnRes = selective retrieval across *depth*).

### Hyper-connections (HC / DHC, also referred to as mHC)

Generalizes the residual connection into a learnable little network of connections. Expand the residual stream into $n$ parallel copies (a **hyper-hidden matrix** $H \in \mathbb{R}^{n \times d}$) and learn how they feed the layer and each other. Connection weights form an $(n{+}1)\times(n{+}1)$ matrix of $B$ (weights on the layer output), $A_m$ (mix the $n$ copies into the layer input), $A_r$ (width-mixing among copies):

$$h_0^\top = A_m^\top H \quad (\text{layer input}), \qquad \hat{H} = B^\top\, \mathcal{T}(h_0)^\top + A_r^\top H$$

where $\mathcal{T}$ is the block (attention or FFN). This decomposes into **depth-connections** (generalized, weighted residuals — can up/down-weight or reorder layers) and **width-connections** (lateral exchange between the $n$ streams). **Static HC** = learnable scalars; **Dynamic HC (DHC)** = weights predicted from the input. Plain Pre-Norm / Post-Norm residuals are the non-trainable $n=1$ special cases. Fixes the "seesaw" between gradient vanishing and representation collapse at depth.

---

## 9. FFN variant — Mixture of Experts (MoE)

Replaces the single dense FFN (flow step 4.ii) with $N$ expert FFNs plus a router that sends each token to only its top-$k$ experts — so parameter count grows without proportional compute per token:
$$\text{MoE}(x) = \sum_{i \in \text{TopK}(g(x))} g_i(x)\, \text{FFN}_i(x), \qquad g(x) = \text{softmax}(x W_{gate})$$
$g_i$ = router gate weight for expert $i$. DeepSeekMoE adds fine-grained + shared "always-on" experts. Only $k$ of $N$ experts run per token (e.g. 6 of 160), so a 100B+ param model activates only a few B params per token.

---

## Where each variant fits — quick map

| Variant | Modifies which flow step | One-line role |
|---|---|---|
| MQA / GQA | MHA step 1–2 (how many K/V heads) | shrink KV cache by sharing K/V across query heads |
| MLA | MHA step 1–2 (K/V production) | cache one low-rank latent instead of full K/V; decoupled RoPE for position |
| KDA | replaces MHA steps 1–5 | linear recurrent state, fixed size, no growing cache; per-channel gated delta rule |
| RoPE | MHA step 1 (rotate Q,K) | relative position, long-context extrapolation |
| Attention residuals | residual "Add" (4.i.3) / value path | extra skip from early-layer values or attention-over-depth; fights over-smoothing |
| Hyper-connections (mHC) | residual "Add" (4.i.3 / 4.ii.3) | learnable multi-stream generalization of the residual |
| MoE | FFN sub-layer (4.ii) | routed sparse experts — more params, same per-token compute |
