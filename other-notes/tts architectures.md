# Neural TTS Architectures — Codec-Token AR vs Diffusion-AR (end to end)

Two ways modern neural TTS turns text into a waveform, written to the fundamental operations. Both are autoregressive over time and share the transformer and decoder building blocks in Part 0. They differ in **what the AR model emits per step**:

- **Codec-token AR** (Qwen3-TTS style) — emit *discrete* neural-codec tokens; a convolutional codec decoder renders the waveform.
- **Diffusion-AR** (VoxCPM style) — emit *continuous* VAE latents; each step runs a small flow-matching sampler, and a VAE decoder renders the waveform.

Notation: $S$ = sequence length, $d$ = model width, $d_h$ = head dim, $\odot$ = elementwise product, $\sigma$ = sigmoid, $\leftarrow$ = assignment, `[a ; b]` = concatenation. All $W_\bullet$ (weights), $C^k$ (codebooks), $g_\bullet$ (gains) are learned. Sub-components below follow the standard realizations these models are built on (Qwen3 LM layer, Mimi-style RVQ codec, depth-wise multi-codebook prediction); VoxCPM's wiring is from its paper.

---

## Building blocks (used throughout)

| Function | Role |
|----------|------|
| **RMSNorm** | normalize a vector to unit RMS, rescale by gain |
| **RoPE** | rotate Q/K by position (relative position) |
| **Attention** | grouped-query self-attention (+ QK-norm, RoPE) |
| **SwiGLU_FFN** | gated feed-forward |
| **TransformerStack** | $L$ layers of Attention + SwiGLU_FFN — the "backbone" |
| **SampleToken** | logits → one discrete token (repetition-aware) |
| **FlowMatchingSample** | integrate the noise→data ODE |
| **TransposedConvDecoder** | latent frames → waveform |
| **PredictResidualCodebooks** | depth model for RVQ codebooks 1..K−1 ("MTP") |
| **CodecDecode** | RVQ dequantize → TransposedConvDecoder |
| **FSQ** | per-dimension scalar quantize (+ straight-through) |
| **LocEnc** | compress past latent patches → history |
| **LocDiT** | diffusion-transformer velocity network |

Entry points: **SynthesizeCodecAR** (Part 1) and **SynthesizeDiffusionAR** (Part 2).

---

## Part 0 — Primitive operations

### RMSNorm($x$, $g$)

$$\text{RMSNorm}(x, g) \;=\; g \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_j x_j^2 \,+\, \epsilon}}$$

Normalize to unit root-mean-square, then rescale each dimension by learned gain $g$.

### RoPE($v$, pos)

$$\text{for each pair }(2k, 2k{+}1)\text{ of }v\in\mathbb{R}^{d_h},\ \theta_k = \text{pos}\cdot b^{-2k/d_h}:\quad \begin{pmatrix} v_{2k} \\[2pt] v_{2k+1}\end{pmatrix} \leftarrow \begin{pmatrix}\cos\theta_k & -\sin\theta_k \\[2pt] \sin\theta_k & \cos\theta_k\end{pmatrix}\begin{pmatrix} v_{2k} \\[2pt] v_{2k+1}\end{pmatrix}$$

A position-dependent rotation of Q/K, so their later dot product depends on *relative* position (base $b\approx 10^4$).

### Attention($X$, causal)

**Input** $X\in\mathbb{R}^{S\times d}$; $W_Q\!:\!d\to n_q d_h$, $W_K,W_V\!:\!d\to n_{kv}d_h$, $W_O\!:\!n_q d_h\to d$; QK-norm gains $g_q,g_k$.

1. $Q \leftarrow X W_Q,\quad K \leftarrow X W_K,\quad V \leftarrow X W_V$
2. reshape $Q$ into $n_q$ heads; $K,V$ into $n_{kv}$ heads
3. **for** each query head $h$, with kv-head $g=\lfloor h\,n_{kv}/n_q\rfloor$:
    1. **for** each position $s$: $\ \tilde Q_h[s] \leftarrow \text{RoPE}(\text{RMSNorm}(Q_h[s], g_q), s);\ \ \tilde K_g[s] \leftarrow \text{RoPE}(\text{RMSNorm}(K_g[s], g_k), s)$
    2. $\mathbf{Sc} \leftarrow \tilde Q_h\, \tilde K_g^{\top} / \sqrt{d_h}$;  **if** causal: $\mathbf{Sc}[i,j] \leftarrow -\infty$ for $j>i$
    3. $O_h \leftarrow \text{softmax}(\mathbf{Sc})\, V_g$
4. **return** $[\,O_1 \mid \cdots \mid O_{n_q}\,]\, W_O$

Grouped-query attention: each K/V head is shared by $n_q/n_{kv}$ query heads (GQA); QK-norm normalizes every Q/K row before the rotation and scaled dot-product.

### SwiGLU_FFN($x$)

$$\text{SwiGLU\_FFN}(x) = W_{\text{down}}\Big[\,\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\,\Big], \qquad \text{SiLU}(u) = u\,\sigma(u)$$

Gated MLP; $W_{\text{gate}}, W_{\text{up}}\!:\!d\to d_{ff}$, $W_{\text{down}}\!:\!d_{ff}\to d$.

### TransformerStack($X$, causal)

**Input** embeddings $X\in\mathbb{R}^{S\times d}$; $L$ pre-norm layers.

1. **for** $\ell = 1$ to $L$:
    1. $X \leftarrow X + \text{Attention}(\text{RMSNorm}(X, g^1_\ell),\ \text{causal})$
    2. $X \leftarrow X + \text{SwiGLU\_FFN}(\text{RMSNorm}(X, g^2_\ell))$
2. **return** $\text{RMSNorm}(X, g^{\text{final}})$

The "backbone": stacked pre-norm attention + feed-forward blocks with residual connections.

### SampleToken($h$, $W_{\text{head}}$, history)

**Input** hidden state $h$; output head $W_{\text{head}}$; recent-token window `history`.

1. $z \leftarrow h\, W_{\text{head}}^{\top}$
2. $t \leftarrow \text{NucleusSample}(z)$
3. **if** $t$ occurs in `history` window: $\ z[t] \leftarrow -\infty;\ \ t \leftarrow \text{Sample}(z)$  (resample over full vocab)
4. **return** $t$

Repetition-aware sampling (RAS): nucleus-sample by default; if the pick is already repeating, mask it and resample so the token LM can escape loops.

### FlowMatchingSample($v_{\text{net}}$, cond, $N$)

**Input** velocity network $v_{\text{net}}$; conditioning `cond`; step count $N$.

1. $x \sim \mathcal{N}(0, I);\quad \Delta t \leftarrow 1/N$
2. **for** $n = 0$ to $N-1$, with $t \leftarrow n\,\Delta t$:
    1. $u \leftarrow v_{\text{net}}(x, t, \text{cond})$  (optional CFG: $u \leftarrow (1{+}w)\,v_{\text{net}}(x,t,\text{cond}) - w\,v_{\text{net}}(x,t,\varnothing)$)
    2. $x \leftarrow x + \Delta t \cdot u$
3. **return** $x$

Euler integration of the noise→data ODE. Training uses straight noise→data paths, so the field is ~constant along a trajectory and few steps suffice.

### TransposedConvDecoder($Z$)

**Input** latent frames $Z\in\mathbb{R}^{T\times d}$; upsample strides $[s_1,\dots,s_K]$; $\text{Act}$ = Snake/ELU.

1. $h \leftarrow \text{Conv1d}(Z)$
2. **for** each upsample stage $k$ (stride $s_k$):
    1. $h \leftarrow \text{ConvTranspose1d}(\text{Act}(h),\ \text{stride}{=}s_k)$
    2. **for** each residual unit (dilations $1,3,9,\dots$): $\ h \leftarrow h + \text{Conv1d}_{1\times1}(\text{Act}(\text{Conv1d}_{\text{dil}}(\text{Act}(h))))$
3. **return** $\text{Conv1d}(\text{Act}(h))$  (1 waveform channel)

SEANet/VAE-style decoder; $\prod_k s_k$ = audio samples produced per latent frame.

---

## Part 1 — Codec-token AR (Qwen3-TTS style)

*Setup:* a neural codec quantizes each 12.5 Hz frame into $K$ codebook ids (cb0 = semantic, distilled from a speech SSL model; cb1..$K{-}1$ = acoustic residual); $C^k[\text{id}]\in\mathbb{R}^d$ is codebook $k$'s embedding table.

### SynthesizeCodecAR(text, [reference_audio])

**Input** `text`, optional `reference_audio`.  **Output** waveform.

1. $\text{ids} \leftarrow \text{BPE}(\text{normalize}(\text{text}));\quad X \leftarrow \text{EmbedText}(\text{ids}) \in \mathbb{R}^{N\times d}$
2. **if** `reference_audio` given: prepend summed-codebook embeddings of $\text{CodecEncode}(\text{reference\_audio})$ to $X$
3. $F \leftarrow [\,];\quad \text{seq} \leftarrow X$
4. **for** $t = 1, 2, \dots$:
    1. **if** $t>1$: $\ e_{t-1} \leftarrow \sum_{k=0}^{K-1} C^k[F[t{-}1][k]];\ \ \text{seq} \leftarrow [\,\text{seq}\,;\, e_{t-1}\,]$
    2. $h_t \leftarrow \text{TransformerStack}(\text{seq},\ \text{causal}{=}\text{true})[\text{last}]$
    3. $c_0 \leftarrow \text{SampleToken}(h_t,\ W^0_{\text{head}},\ \text{hist}_0)$
    4. **if** $c_0 = \text{EOS}$: **break**
    5. $[c_1,\dots,c_{K-1}] \leftarrow \text{PredictResidualCodebooks}(h_t,\ c_0)$
    6. $F.\text{append}([c_0, c_1, \dots, c_{K-1}])$
5. **return** $\text{CodecDecode}(F)$

The backbone is KV-cached, so step 4.2 only computes the new position. Codebook-0 is sampled with RAS; the acoustic codebooks are filled by the depth model below.

### PredictResidualCodebooks($h_t$, $c_0$)

**Input** backbone hidden $h_t$; chosen cb-0 token $c_0$. `DepthTransformer` is a small causal TransformerStack over the codebook-depth axis.

1. $D \leftarrow [\, h_t + C^0[c_0]\,]$
2. **for** $k = 1$ to $K-1$:
    1. $g_k \leftarrow \text{DepthTransformer}(D,\ \text{causal}{=}\text{true})[\text{last}]$
    2. $c_k \leftarrow \arg\max\big(g_k\,(W^k_{\text{head}})^{\top}\big)$
    3. $D.\text{append}(g_k + C^k[c_k])$
3. **return** $[c_1,\dots,c_{K-1}]$

The depth model (Qwen3-TTS "MTP") predicts a frame's residual codebooks along the depth axis; residual code $k$ depends on the ones chosen before it. Parallel variant: $K{-}1$ independent heads on $h_t$, predicted at once.

### CodecDecode($F$)

**Input** frames $F$ (each = $K$ codebook ids).  **Output** waveform.

1. **for** each frame $t$: $\ z_q[t] \leftarrow \sum_{k=0}^{K-1} C^k[F[t][k]]$
2. **return** $\text{TransposedConvDecoder}(z_q)$

Step 1 is RVQ reconstruction — the continuous latent is the sum of every codebook's contribution. (25 Hz single-codebook variant: map codes → mel with a flow-matching DiT, i.e. FlowMatchingSample + LocDiT, then a BigVGAN vocoder — a TransposedConvDecoder — renders the waveform.)

---

## Part 2 — Diffusion-AR (VoxCPM style)

*Setup:* an AudioVAE maps a 16 kHz waveform ↔ 25 Hz continuous latents (encoder downsamples by $[2,5,8,8]=640$; decoder mirrors it). The AR model works on **patches** $z_i\in\mathbb{R}^{P\times D}$ of $P$ latent frames ($P=2$ → AR runs at 12.5 Hz; $D$ = latent dim, e.g. 64). `TSLM` and `RALM` are causal TransformerStacks.

### SynthesizeDiffusionAR(text)

**Input** `text`.  **Output** waveform.

1. $\text{ids} \leftarrow \text{tokenize}(\text{normalize}(\text{text}));\quad X \leftarrow \text{EmbedText}(\text{ids})$
2. $Z \leftarrow [\,]$
3. **for** $i = 1, 2, \dots$:
    1. $E \leftarrow \text{LocEnc}(Z)$
    2. $h^{\text{tslm}} \leftarrow \text{TSLM}([\,X ; E\,],\ \text{causal}{=}\text{true})[\text{last}]$
    3. **if** $\sigma(\text{MLP}_{\text{stop}}(h^{\text{tslm}})) > 0.5$ and $i>1$: **break**
    4. $\text{sk} \leftarrow \text{FSQ}(h^{\text{tslm}})$
    5. $\text{res} \leftarrow \text{RALM}([\,X ; E ; \text{sk}\,],\ \text{causal}{=}\text{true})[\text{last}]$
    6. $h^{\text{final}} \leftarrow \text{sk} + \text{res}$
    7. $z_i \leftarrow \text{FlowMatchingSample}(\text{LocDiT},\ (h^{\text{final}},\, z_{i-1}),\ N)$
    8. $Z.\text{append}(z_i)$
4. **return** $\text{TransposedConvDecoder}(\text{flatten}(Z))$  (AudioVAE decoder, strides $[8,8,5,2]$)

Each patch's conditioning is a **stable skeleton** $\text{FSQ}(\text{TSLM}(\cdot))$ plus a **residual detail** $\text{RALM}(\cdot)$; that guides LocDiT's flow-matching sampler to draw the continuous latent patch, conditioned on the previous patch $z_{i-1}$ so patches join smoothly.

### FSQ($h$)

$$\text{FSQ}(h)_j = \Delta \cdot \text{clip}\big(\text{round}(h_j / \Delta),\ -L,\ L\big)$$

Per-dimension scalar quantization (≈36 dims, 9 levels) onto a lattice. Forward = round; backward = straight-through, $\partial/\partial h := 1$, so gradients pass through.

### LocEnc($Z$)

**Input** past patches $Z$.  **Output** acoustic-history embeddings $E$.

1. **if** $Z = \varnothing$: **return** $\varnothing$
2. flatten $Z$ to latent frames;  $E \leftarrow$ (a few strided $\text{Conv1d} + \text{Act}$ layers, compressing to per-patch embeddings)
3. **return** $E$

Gives TSLM/RALM the acoustic context of what has already been spoken.

### LocDiT($z_t$, $t$, cond)

**Input** noisy patch $z_t\in\mathbb{R}^{P\times D}$; diffusion time $t$; cond $=(h^{\text{final}},\, z_{i-1})$.  ($P$ frames = tokens; attention non-causal.)

1. $\text{tok} \leftarrow \text{Linear}(z_t) + \text{PosEmb}(1{:}P);\quad c \leftarrow h^{\text{final}} + \text{TimeEmbed}(t) + \text{Linear}(z_{i-1})$
2. **for** $\ell = 1$ to $L_{\text{DiT}}$:
    1. $(\gamma_1,\beta_1,\gamma_2,\beta_2) \leftarrow \text{Linear}(c)$  (adaLN)
    2. $\text{tok} \leftarrow \text{tok} + \gamma_1 \odot \text{Attention}(\gamma_1\!\cdot\!\text{LN}(\text{tok})+\beta_1,\ \text{causal}{=}\text{false})$
    3. $\text{tok} \leftarrow \text{tok} + \gamma_2 \odot \text{SwiGLU\_FFN}(\gamma_2\!\cdot\!\text{LN}(\text{tok})+\beta_2)$
3. **return** $\text{Linear}(\text{tok})$  (velocity, shape $P\times D$)

The bidirectional DiT velocity network that FlowMatchingSample integrates; the condition (skeleton+residual, diffusion time, previous patch) drives per-layer scale/shift via adaLN. RoPE inside Attention is left as identity here since patch positions are absolute (added as PosEmb).

---

## The two side by side

| | Codec-token AR | Diffusion-AR |
|---|---|---|
| AR step emits | $K$ discrete codec tokens per frame | one continuous $P\times D$ latent patch |
| Per-step generator | SampleToken (head + sampling) + PredictResidualCodebooks | FlowMatchingSample over LocDiT |
| Text→units bottleneck | codec quantization floor (lossy) | none — latents are continuous |
| Units → waveform | CodecDecode (RVQ dequantize + TransposedConvDecoder) | TransposedConvDecoder (VAE decoder) |
| Main trait | discrete, streaming, LM-friendly; token LM can loop (needs SampleToken's RAS) | no quantization loss, richer prosody; runs a sampler every patch |
