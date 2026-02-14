# LoRA (Low-Rank Adaptation) and Common Variants — with simple formulas

This note explains **LoRA** and several widely-used **variants** in a way that matches how they’re implemented in practice for Transformers.

If you want to see concrete numbers, run: `python transformers/lora_examples.py`.

---

## Base setup (a single Linear layer)

Let a linear layer be:

$$
y = xW
$$

where:
- $x \in \mathbb{R}^{d_\text{in}}$
- $W \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$ (or transposed depending on convention)

In full fine-tuning you learn $W$. In **parameter-efficient fine-tuning (PEFT)**, you keep the pretrained weight $W_0$ frozen and learn a small modification.

---

## LoRA (the core algorithm)

### Idea
Instead of learning a full update $\Delta W$, learn a **low-rank** update:

$$
\Delta W = BA
$$

with:
- $B \in \mathbb{R}^{d_\text{out} \times r}$
- $A \in \mathbb{R}^{r \times d_\text{in}}$
- $r \ll \min(d_\text{in}, d_\text{out})$

and the effective weight becomes:

$$
W_\text{eff} = W_0 + s \cdot \Delta W
$$

Most implementations use a scale:

$$
s = \frac{\alpha}{r}
$$

so the forward pass is:

$$
y = x\left(W_0 + \frac{\alpha}{r}BA\right)
$$

### Why it saves parameters
Full fine-tuning learns $d_\text{out} \cdot d_\text{in}$ parameters.
LoRA learns $r(d_\text{in} + d_\text{out})$, which is much smaller when $r$ is small.

### Common practical details
- **Freeze $W_0$**; only train $A,B$ (and sometimes biases / layer norms depending on recipe).
- Often initialize one factor (commonly $B$) to zeros so the adapter starts as $\Delta W \approx 0$.
- Often apply **LoRA dropout** on the adapter branch only (not shown in formulas above).
- At inference you can **merge**: store $W_0 + sBA$ as a single matrix.

---

## rsLoRA (Rank-stabilized / alternative scaling)

LoRA’s common scale is $s = \alpha/r$. Some variants change the scaling to reduce sensitivity to rank:

$$
s = \frac{\alpha}{\sqrt{r}}
$$

Everything else is the same; only the scaling differs:

$$
W_\text{eff} = W_0 + \frac{\alpha}{\sqrt{r}}BA
$$

Intuition: if you increase $r$, the update norm can grow differently; $\sqrt{r}$ scaling can make behavior more stable across ranks in some settings.

---

## QLoRA (Quantized LoRA)

### Idea
QLoRA keeps the **base model weights quantized** (e.g., 4-bit) to save memory, while still training LoRA adapters in higher precision.

Conceptually you compute:

$$
y = x\Big(\hat{W}_0 + sBA\Big)
$$

where $\hat{W}_0$ is a dequantized view of a quantized $W_0$.

### Simple quantization formula (toy)
One simple symmetric quantization is:

$$
W_q = \text{clip}\left(\text{round}\left(\frac{W_0}{\delta}\right), -Q, Q\right), \quad
\hat{W}_0 = \delta W_q
$$

where $Q = 2^{b-1}-1$ for $b$ bits (e.g., $b=4 \Rightarrow Q=7$), and $\delta$ is a scale (often derived from max/percentile statistics).

### Quantization 101 (what’s happening conceptually?)

**Quantization** replaces real-valued weights with values from a **small discrete set** so they take fewer bits to store. You can think of it as “rounding onto a grid”.

- **Quantize**: map a float weight $w$ to an integer code $q$.
- **Dequantize**: map that integer code back to an approximate float $\hat{w}$ used in computation.
- **Quantization error**: $e = \hat{w} - w$ (this is the approximation you accept to save memory/bandwidth).

Two common parameterizations:

- **Symmetric (zero-point = 0)**:

$$
q = \text{clip}\left(\text{round}\left(\frac{w}{\delta}\right), -Q, Q\right), \quad
\hat{w} = \delta q
$$

- **Affine / asymmetric (with zero-point $z$)**:

$$
q = \text{clip}\left(\text{round}\left(\frac{w}{\delta}\right) + z, 0, 2^b - 1\right), \quad
\hat{w} = \delta (q - z)
$$

**Per-tensor vs per-channel** scaling:
- **Per-tensor**: one $\delta$ for the whole matrix $W$ (simpler, often worse accuracy).
- **Per-channel**: one $\delta_i$ per output channel/row (usually better accuracy).

Tiny numeric example (symmetric 4-bit, $Q=7$):
- Suppose $\max|W| = 1.4 \Rightarrow \delta = 1.4/7 = 0.2$.
- If $w = 0.33$: $q=\text{round}(0.33/0.2)=2$, so $\hat{w}=0.2\cdot 2=0.4$.

Real QLoRA uses more careful schemes (e.g., NF4, double quantization, paged optimizers), but the high-level computation is still “**quantized frozen base + LoRA update**”.

---

## AdaLoRA (Adaptive rank allocation)

### Idea
Instead of using a fixed rank $r$ everywhere, **AdaLoRA allocates rank where it matters** under a total parameter budget.

You can think of having per-layer ranks $r_\ell$ with a budget:

$$
\sum_\ell r_\ell \le R_\text{budget}
$$

and each layer has:

$$
\Delta W_\ell = B_\ell A_\ell, \quad \text{rank}(\Delta W_\ell) \le r_\ell
$$

The algorithm uses an **importance score** per layer / per singular direction to decide where rank is most useful, then increases/decreases $r_\ell$ during training.

### Simplified “importance” intuition
A crude proxy (useful for intuition and tiny demos) is:

$$
\text{importance}_\ell \propto \lVert \Delta W_\ell \rVert_F
$$

and then allocate more rank to layers with higher importance. Real methods use finer-grained signals (e.g., singular-value-like importance per direction) and schedules to gradually prune.

---

## DoRA (Weight-Decomposed LoRA)

### Idea
DoRA separates **direction** and **magnitude** of weights, because full fine-tuning often changes both.

One common view (row-wise, i.e., per output channel) is:

- Start with pretrained weights $W_0$
- Define magnitude $m$ and direction $V$:

$$
m_i = \lVert (W_0)_i \rVert_2, \quad
V_i = \frac{(W_0)_i}{\lVert (W_0)_i \rVert_2}
$$

- Learn a low-rank **direction update** $\Delta V = BA$
- Form the adapted weight:

$$
W_i = m_i \cdot \frac{V_i + (\Delta V)_i}{\lVert V_i + (\Delta V)_i \rVert_2}
$$

So DoRA is “LoRA on direction” plus an explicit magnitude parameterization.

---

## IA³ (often used alongside LoRA; not low-rank but PEFT)

IA³ uses **learned multiplicative vectors** instead of a low-rank matrix update.

Two simple forms you’ll see:

- **Input scaling**:

$$
y = (x \odot s_\text{in})W_0
$$

- **Output scaling**:

$$
y = (xW_0) \odot s_\text{out}
$$

where $s$ is a learned vector and $\odot$ is elementwise multiplication. In Transformers, IA³ is often applied to attention $K/V$ projections and FFN intermediate activations.

---

## How to use the examples

Run:
- `python transformers/lora_examples.py`

It prints:
- $W_0$, $\Delta W$, $W_\text{eff}$
- outputs \(y\) for LoRA / QLoRA / DoRA / IA³
- a tiny “budget allocation” demo for AdaLoRA-style thinking


