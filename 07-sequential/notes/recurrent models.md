
# Recurrent Models — RNN, LSTM, GRU

## 0. Why recurrent models

Feed-forward nets take a fixed-size input and have no memory. Sequences (text, time series, audio) have variable length and order matters. A **recurrent** net processes one time step at a time and carries a **hidden state** $h_t$ that summarizes everything seen so far, so the same weights are reused at every step (weight sharing across time).

Notation:
- Input sequence $x_1, x_2, ..., x_T$ (each $x_t$ is a vector).
- Hidden state $h_t$; output $\hat{y}_t$; target $y_t$.
- Weights (shared across all $t$): $W_{xh}$ (input→hidden), $W_{hh}$ (hidden→hidden), $W_{hy}$ (hidden→output); biases $b_h, b_y$.
- $L_t$ = loss at step $t$; total loss $L = \sum_{t=1}^{T} L_t$.

---

## 1. Vanilla RNN — forward pass

At each time step, combine the current input with the previous hidden state:

- $a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$
- $h_t = \tanh(a_t)$
- $\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)$

$h_0$ is initialized to zeros. The recurrence $h_t = f(h_{t-1}, x_t)$ is what gives the network memory.

---

## 2. Backpropagation Through Time (BPTT) — RNN only

BPTT = ordinary backprop applied to the RNN **unrolled** across time into a $T$-step feed-forward graph. The subtlety: because $W_{xh}, W_{hh}, W_{hy}$ are **shared** at every step, each weight's gradient is the **sum** of its contributions from all time steps. And because $h_t$ feeds both the output at $t$ and the hidden state at $t+1$, the gradient flowing into $h_t$ has **two sources**.

Inputs:
- Sequence $x_1, ..., x_T$ and targets $y_1, ..., y_T$
- Weights $W_{xh}, W_{hh}, W_{hy}, b_h, b_y$; learning rate $\eta$
- Cached forward values $a_t, h_t, \hat{y}_t$ for all $t$

Output: updated weights

Steps:
1. Run the forward pass (§1) for $t = 1..T$, caching $a_t, h_t, \hat{y}_t$
2. Initialize all weight gradients to zero: $dW_{xh}, dW_{hh}, dW_{hy}, db_h, db_y = 0$
3. Initialize downstream hidden gradient: $dh_{next} = 0$  (this carries $\partial L / \partial h_t$ coming from step $t+1$)

4. For $t$ from $T$ down to 1:  (backward in time)
	1. Output-layer gradient (softmax + cross-entropy): $dy_t = \hat{y}_t - y_t$ **(WHY? softmax followed by cross-entropy collapses to this clean difference)**
	2. Accumulate output weights: $dW_{hy} \mathrel{+}= dy_t \cdot h_t^{\top}$ and $db_y \mathrel{+}= dy_t$
	3. Gradient into $h_t$ from both paths — the output at $t$ **and** the next step $t+1$: $dh_t = W_{hy}^{\top} dy_t + dh_{next}$
	4. Backprop through the $\tanh$ nonlinearity: $da_t = dh_t \odot (1 - h_t^2)$ **(WHY? $\tanh'(a) = 1 - \tanh^2(a) = 1 - h_t^2$; $\odot$ is elementwise)**
	5. Accumulate the shared hidden/input weights (contribution of this step):
		1. $dW_{xh} \mathrel{+}= da_t \cdot x_t^{\top}$
		2. $dW_{hh} \mathrel{+}= da_t \cdot h_{t-1}^{\top}$
		3. $db_h \mathrel{+}= da_t$
	6. Pass gradient back one step in time to $h_{t-1}$: $dh_{next} = W_{hh}^{\top} da_t$
	7. Repeat for each time step $t$

5. (Optional) **Gradient clipping**: if $\lVert grad \rVert > \tau$, rescale $grad \leftarrow \tau \cdot grad / \lVert grad \rVert$ — guards against exploding gradients
6. Update every weight: $W \mathrel{-}= \eta \cdot dW$ (and biases likewise)
7. Return updated weights

**Truncated BPTT**: for long sequences, unroll and backprop only over a window of $k$ steps at a time instead of the full $T$, to bound memory and compute.

---

## 3. Why plain RNNs struggle — vanishing / exploding gradients

In step 4.6 the gradient is multiplied by $W_{hh}^{\top}$ (and a $\tanh'$ factor $\le 1$) once per step back in time. Over many steps this is effectively $W_{hh}$ raised to a power:

- If the relevant eigenvalues are $< 1$: the signal shrinks geometrically → **vanishing gradient** → the network can't learn long-range dependencies (early inputs get no gradient).
- If they are $> 1$: the signal blows up → **exploding gradient** → unstable training (mitigated by clipping in step 5).

LSTM and GRU fix the *vanishing* case with gated additive state paths that let gradients flow across many steps without repeated multiplicative shrinkage.

---

## 4. LSTM — forward pass

The **LSTM** adds a separate **cell state** $c_t$ (a "memory highway") alongside $h_t$, and three **gates** (each a sigmoid $\sigma$ producing values in $[0,1]$) that regulate information flow. Let $[h_{t-1}, x_t]$ denote concatenation.

- **Forget gate** — what to erase from memory: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
- **Input gate** — how much new info to write: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
- **Candidate** — the new info proposed: $\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$
- **Cell state update** — forget old, add new: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- **Output gate** — how much cell state to expose: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
- **Hidden state**: $h_t = o_t \odot \tanh(c_t)$

Key point: $c_t = f_t \odot c_{t-1} + ...$ is an **additive** update. When $f_t \approx 1$, the cell state passes through nearly unchanged and its gradient neither vanishes nor explodes — this is the constant error carousel that enables long-range memory.

---

## 5. GRU — forward pass

The **GRU** is a lighter LSTM: it merges the cell and hidden state and uses two gates instead of three (fewer parameters, often comparable accuracy).

- **Update gate** — blends old state vs new (plays forget+input in one): $z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$
- **Reset gate** — how much past state to use when forming the candidate: $r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$
- **Candidate**: $\tilde{h}_t = \tanh(W_h [\, r_t \odot h_{t-1},\; x_t \,] + b_h)$
- **Hidden state update** — convex mix of previous and candidate: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

The additive convex-combination update plays the same gradient-preserving role as the LSTM cell state.

---

## 6. Extending BPTT to LSTM and GRU

The training algorithm is **the same BPTT** as §2 — unroll across time, walk backward from $t=T$ to $1$, and **sum** each shared weight's gradient over all steps. What changes is only the per-step backward computation inside the loop (step 4.3–4.6), because there are more internal quantities and gates to differentiate:

- Instead of a single $dh_{next}$, you carry **two** recurrent gradients back one step: $dh_{next}$ **and** $dc_{next}$ (LSTM). GRU carries only $dh_{next}$.
- At each step you distribute $dh_t$ (and $dc_t$) through the gate equations — each gate ($f, i, o, z, r$) and the candidate gets its own local gradient via its $\sigma'$ or $\tanh'$ factor, and each contributes to its own weight matrix's accumulated gradient.
- The additive cell update means the dominant gradient path is $dc_{t-1} \mathrel{+}= f_t \odot dc_t$ — no repeated multiplication by a weight matrix, which is exactly why the gradient survives across many steps.

So: same outer structure (unroll → backward loop → accumulate shared-weight gradients → clip → update), just more gate-wise terms per step. The full per-gate gradient expressions are not written out here.

---

## 7. Variants (quick reference)

- **Bidirectional RNN**: run one RNN left→right and another right→left, concatenate their hidden states. Uses both past and future context (only for non-streaming tasks where the full sequence is available).
- **Stacked / deep RNN**: feed the hidden states of one recurrent layer as inputs to another; more representational depth.
- **Seq2seq (encoder–decoder)**: one RNN encodes the input sequence into a context vector, another decodes it into an output sequence (translation, summarization). Attention was added to fix the context-vector bottleneck and eventually led to Transformers, which dropped recurrence entirely.
