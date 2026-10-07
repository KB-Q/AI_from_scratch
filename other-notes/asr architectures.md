# Neural ASR Architectures — Transducer, Speech-to-Intent, LLM-AR, Diffusion-NAR (end to end)

Ways modern systems turn a waveform into text (or directly into meaning), written to the fundamental operations. Every family shares a **front end** — log-mel features + an acoustic encoder that turns audio into a frame sequence — and differs in the **decoder / output head** and, crucially, in **how output tokens depend on each other**:

- **Neural ASR** (classic voice-assistant stack) — a task-specific head on the encoder: **CTC** (per-frame, tokens independent), **RNN-T / transducer** (streaming, a label LM fused with the encoder), or **AED** (attention encoder-decoder). RNN-T is the streaming workhorse.
- **Speech-to-Intent (S2I) / end-to-end SLU** — skip the transcript entirely; map audio → intent + slots.
- **General transformer AR** (Qwen3-ASR / Whisper style) — an audio encoder feeds a full LLM decoder that autoregressively emits transcript tokens; the LLM's language prior does the heavy lifting.
- **Diffusion-NAR** (Whisfusion / dLLM-ASR style) — replace left-to-right decoding with **parallel iterative denoising** of a masked token sequence. (There is no true "diffusion-AR" ASR — see Part 4.)
- **Audio-native reasoning** — *not a separate architecture*; it is the Part 3 front end wired to a general reasoning LLM. See Part 5.

Notation: $X$ = input frames, $T$ = #frames after subsampling, $U$ = #output tokens, $d$ = model width, $\varnothing$ = CTC/transducer **blank** symbol, $\mathcal{V}$ = token vocabulary, $\odot$ = elementwise product, $\sigma$ = sigmoid, $\leftarrow$ = assignment, `[a ; b]` = concatenation. All $W_\bullet$ are learned. Qwen3-ASR wiring is from its technical report + reference implementation; transducer/CTC/AED are the standard formulations.

---

## Building blocks (used throughout)

| Function | Role |
|----------|------|
| **LogMel** | waveform → log-mel filterbank frames |
| **ConvSubsample** | conv stack that downsamples the frame rate (4×–8×) |
| **Attention / SwiGLU_FFN / RMSNorm / RoPE** | standard transformer primitives (Part 0) |
| **TransformerStack** | $L$ layers of Attention + FFN — the encoder/decoder backbone |
| **ConformerBlock** | attention + convolution encoder layer (the modern ASR encoder) |
| **CTCHead / CTCCollapse** | per-frame token posteriors; collapse a frame path to a label string |
| **PredictionNetwork / JointNetwork** | transducer label-LM and encoder-label fusion |
| **TransducerDecode** | streaming greedy decode over the $T\times U$ lattice |
| **AEDDecodeStep** | one cross-attention autoregressive decoder step |
| **AudioEncoderLLM** | conv-stem + transformer + projector front end for an LLM decoder |
| **TranscribeAR** | LLM autoregressive transcript decode (Qwen3-ASR) |
| **MaskedDiffusionDecode** | parallel iterative denoising of a masked transcript |

Entry points: **NeuralASR** (Part 1), **SpeechToIntent** (Part 2), **TranscribeAR** (Part 3), **MaskedDiffusionDecode** (Part 4).

---

## Part 0 — Front end + shared primitives

### LogMel(wav)

**Input** waveform at 16 kHz; window 25 ms, hop 10 ms; $M$ mel bins (80 typical; 128 for Qwen3-ASR).  

**Output** frames $X\in\mathbb{R}^{T_0\times M}$ at 100 Hz.

1. **for** each frame (25 ms window, stepped by 10 ms): $\ s \leftarrow \text{STFT}(\text{window}\odot\text{wav}_{\text{frame}})$
2. $P \leftarrow |s|^2$  (power spectrum)
3. $X \leftarrow \log\big(\max(\text{MelFilterbank}\cdot P,\ \varepsilon)\big)$  (mel projection, then log-compress)
4. **return** normalized $X$

The universal ASR front end: a log-mel spectrogram at 100 frames/s. Everything downstream consumes these frames; the choice of decoder is what separates the paradigms.

### ConvSubsample($X$)

**Input** mel frames $X\in\mathbb{R}^{T_0\times M}$; $J$ conv layers, each stride 2 in time.  

**Output** $E\in\mathbb{R}^{T\times d}$, $T=T_0/2^{J}$.

1. $c \leftarrow X$ reshaped to $[1, 1, T_0, M]$
2. **for** $j = 1$ to $J$: $\ c \leftarrow \text{Act}(\text{Conv2d}(c,\ \text{stride}{=}2))$
3. $E \leftarrow \text{Linear}(\text{flatten channels/freq of } c)$
4. **return** $E$ + positional encoding

Downsamples 100 Hz to 25–12.5 Hz so the encoder runs on a short sequence and each frame spans enough acoustic context to align to a token. $J{=}2$ gives 4× (25 Hz, classic ASR); Qwen3-ASR uses $J{=}3$ (8×, 12.5 Hz).

### Transformer primitives (standard)

- **RMSNorm**$(x,g) = g\odot x/\sqrt{\tfrac1d\sum_j x_j^2+\epsilon}$ — normalize to unit RMS, rescale by learned gain (LayerNorm, used in conformer/Whisper encoders, additionally subtracts the mean).

- **RoPE**$(v,\text{pos})$ — rotate each coordinate pair of a Q/K vector by angle $\text{pos}\cdot b^{-2k/d_h}$, making the later dot product depend on *relative* position.

- **Attention**$(X,\text{mask})$ — $Q,K,V \leftarrow XW_Q,XW_K,XW_V$; $O \leftarrow \text{softmax}(QK^\top/\sqrt{d_h}+\text{mask})V$; return $OW_O$. Encoders use a full (bidirectional) mask; decoders use a causal mask; grouped-query (GQA) shares each K/V head across several Q heads.

- **SwiGLU_FFN**$(x) = W_{\text{down}}\big[\text{SiLU}(xW_{\text{gate}})\odot(xW_{\text{up}})\big]$ — gated feed-forward.

### TransformerStack($X$, mask)

**Input** $X\in\mathbb{R}^{S\times d}$; $L$ pre-norm layers.  

**Output** contextualized $X$.

1. **for** $\ell = 1$ to $L$:
    1. $X \leftarrow X + \text{Attention}(\text{RMSNorm}(X),\ \text{mask})$
    2. $X \leftarrow X + \text{SwiGLU\_FFN}(\text{RMSNorm}(X))$
2. **return** $\text{RMSNorm}(X)$

The plain backbone. An ASR **encoder** runs it with a full mask (each frame sees the whole utterance — or a bounded window for streaming); an LLM **decoder** runs it with a causal mask.

### ConformerBlock($x$)

**Input** frame sequence $x\in\mathbb{R}^{T\times d}$; depthwise kernel size $\kappa$ (15–31).  

**Output** $x$.

1. $x \leftarrow x + \tfrac12\,\text{SwiGLU\_FFN}(\text{LN}(x))$  (Macaron half-step FFN)
2. $x \leftarrow x + \text{Attention}(\text{LN}(x),\ \text{mask})$  (relative-position self-attention)
3. **ConvModule** — $x \leftarrow x + \big[\text{PWConv}\circ\text{Swish}\circ\text{BN}\circ\text{DWConv}_{\kappa}\circ\text{GLU}\circ\text{PWConv}\big](\text{LN}(x))$
4. $x \leftarrow x + \tfrac12\,\text{SwiGLU\_FFN}(\text{LN}(x))$
5. **return** $\text{LN}(x)$

The standard modern acoustic encoder layer: self-attention captures long-range context, the depthwise-convolution module captures local phonetic detail, and the two half-FFNs sandwich them. Streaming variants mask the attention (and pad the conv causally) so a frame sees only bounded right-context.

---

## Part 1 — Neural ASR: CTC, RNN-T, AED

*Setup:* $H = \text{TransformerStack/Conformer}(\text{ConvSubsample}(\text{LogMel}(\text{wav})))$ is the shared **encoder** output $H=[h_1,\dots,h_T]$. The three heads below differ only in how they turn $H$ into tokens.

### CTCHead + CTCCollapse

$$p(\pi_t \mid X) = \text{softmax}(W\,h_t) \ \text{ over } \mathcal{V}\cup\{\varnothing\}, \qquad p(Y\mid X) = \!\!\sum_{\pi\,:\,\mathcal{B}(\pi)=Y}\ \prod_{t=1}^{T} p(\pi_t\mid X)$$

Each frame emits a token *independently* (or $\varnothing$). The **collapse** $\mathcal{B}(\pi)$ maps a length-$T$ frame path to a label string by (1) merging consecutive duplicates, then (2) deleting every $\varnothing$. Training maximizes $p(Y\mid X)$ summed over all paths that collapse to $Y$, computed by a forward-backward DP; greedy decode is `CTCCollapse(argmax_t)`. **(WHY blanks + collapse?** frames outnumber tokens and the alignment is unknown; blank = "emit nothing here", duplicate-merge lets one token span several frames.**)** Limitation: the per-frame independence assumption means CTC has no built-in language model — it needs an external LM to be competitive.

### PredictionNetwork($Y_{<u}$) and JointNetwork($h_t$, $g_u$)

$$g_u = \text{PredictionNetwork}(Y_{<u}), \qquad z(t,u) = W_o\,\tanh(W_e\,h_t + W_p\,g_u)\ \text{ over } \mathcal{V}\cup\{\varnothing\}$$

The **prediction network** is an audio-blind label LM (LSTM or small causal transformer) over previously emitted non-blank tokens; the **joint network** fuses one encoder frame $h_t$ with one predictor state $g_u$ into a distribution over the vocabulary plus blank. Because $g_u$ conditions on prior tokens, the transducer models output dependencies that CTC cannot.

### TransducerDecode($H$)  — RNN-T, streaming greedy

**Input** encoder frames $H=[h_1,\dots,h_T]$; `max_sym` per frame.  **Output** token string $Y$.

1. $Y \leftarrow [\,];\quad u \leftarrow 0;\quad g_0 \leftarrow \text{PredictionNetwork}(\langle\text{sos}\rangle)$
2. **for** $t = 1$ to $T$:
    1. $\text{emitted} \leftarrow 0$
    2. **loop**:
        1. $k \leftarrow \arg\max\,\text{softmax}\big(\text{JointNetwork}(h_t,\ g_u)\big)$
        2. **if** $k = \varnothing$ **or** $\text{emitted} \ge \text{max\_sym}$: **break**  (advance to next frame)
        3. $Y.\text{append}(k);\quad u \leftarrow u+1;\quad \text{emitted} \leftarrow \text{emitted}+1$
        4. $g_u \leftarrow \text{PredictionNetwork}(Y)$  (KV-cached: only the new token is processed)
3. **return** $Y$

The decode walks a monotonic path on the $T\times U$ lattice: **blank advances time** (next frame), a **label advances the token index** and updates the predictor. The encoder consumes one frame per outer step regardless of how many tokens come out, so it streams in constant memory — this is why RNN-T (with a conformer or LSTM encoder) is the classic production voice-assistant recognizer. Training marginalizes $p(Y\mid X)$ over all such lattice paths with a forward-backward DP, the transducer loss. (Beam search keeps several $(Y,u)$ hypotheses instead of the argmax.)

### AEDDecodeStep($y_{<u}$, $H$)  — attention encoder-decoder (LAS / Whisper-style)

**Input** tokens so far $y_{<u}$; encoder memory $H$.  **Output** next-token distribution.

1. $s \leftarrow \text{EmbedTokens}(y_{<u}) + \text{PosEmb}$
2. **for** $\ell = 1$ to $L$:
    1. $s \leftarrow s + \text{Attention}(\text{RMSNorm}(s),\ \text{causal})$  (self-attention over emitted tokens)
    2. $s \leftarrow s + \text{CrossAttention}(\text{RMSNorm}(s),\ \text{keys/values}{=}H)$  (attend to audio)
    3. $s \leftarrow s + \text{SwiGLU\_FFN}(\text{RMSNorm}(s))$
3. **return** $\text{softmax}(s[\text{last}]\,W_{\text{head}})$

A standard sequence-to-sequence decoder: it emits tokens left-to-right, at each step **cross-attending** to the full encoder output to decide what to say next (LAS used a pyramidal BiLSTM encoder + LSTM decoder; the modern form is an all-transformer AED — the architecture Whisper uses). No blank symbol; the soft attention learns the alignment. Powerful and simple to train, but the cross-attention needs the whole utterance, so vanilla AED is **not streaming** (streaming needs monotonic/chunked attention variants).

---

## Part 2 — Speech-to-Intent (S2I) / end-to-end SLU

*Goal:* go straight from audio to **meaning** — an intent label and its slot values — without producing (or depending on) a transcript.

### SpeechToIntent(wav)

**Input** waveform.  **Output** `(intent, slots)`.

1. $H \leftarrow \text{Encoder}(\text{ConvSubsample}(\text{LogMel}(\text{wav})))$  (encoder often pretrained on ASR, then fine-tuned)
2. **Classification head** (fixed intent set):
    1. $p \leftarrow \text{Pool}(H)$  (mean or attention pooling → one utterance vector)
    2. $\text{intent} \leftarrow \arg\max(W_{\text{int}}\,p)$
    3. **for** each frame $t$: $\ \text{slot}_t \leftarrow \arg\max(W_{\text{slot}}\,h_t)$  (BIO tagging over frames)
3. **or Seq2seq head** (open schema): an AED decoder cross-attending to $H$ emits a **serialized semantic frame** — `[intent] slot₁=value₁ slot₂=value₂ …` — as a token string
4. **return** `(intent, slots)`

The transcript is removed from the target: the model is trained directly on `(audio → intent/slots)` pairs. **(WHY skip the transcript?** a cascade (ASR → NLU) lets recognition errors corrupt the intent — a mistranscribed entity becomes a wrong slot — and it discards prosody that signals intent; end-to-end S2I optimizes the objective you actually care about and is lower-latency.**)** The cost is data: intent-labeled speech is scarce, so the encoder is usually **ASR-pretrained** (or trained multi-task with an auxiliary transcript head) to bootstrap acoustic-linguistic features before the small intent head is fitted. A fully **ASR-free** variant trains the whole thing from audio to intent with no text at all — best when the intent set is small and fixed (on-device wake-word/command systems).

---

## Part 3 — General transformer AR (Qwen3-ASR / Whisper style)

*Setup:* an audio encoder produces embeddings that are fed into a full pretrained **LLM decoder**; the LLM autoregressively writes the transcript, using its language prior for spelling, punctuation, rare words, and context. This is the current state-of-the-art paradigm (LALM / speech-LLM).

### AudioEncoderLLM(wav)  — Qwen3-ASR AuT encoder

**Input** waveform.  **Output** audio embeddings $A\in\mathbb{R}^{N\times d_{\text{dec}}}$ at 12.5 Hz.

1. $\text{mel} \leftarrow \text{LogMel}(\text{wav})$  (128 bins, 100 Hz)
2. $c \leftarrow \text{mel}$ reshaped to $[1,1,128,T_0]$
3. **for** $k = 1$ to $3$: $\ c \leftarrow \text{GELU}(\text{Conv2d}(c,\ \text{stride}{=}2))$  (8× downsample in time and frequency → 12.5 Hz)
4. $E \leftarrow \text{Linear}(\text{flatten}(c)) + \text{SinusoidalPosEmb}$
5. **for** $\ell = 1$ to $L_{\text{enc}}$: $\ E \leftarrow \text{TransformerStack layer}(E,\ \text{mask}{=}\text{windowed-bidirectional})$  (attend within an ~8 s window)
6. $A \leftarrow \text{Linear}\big(\text{GELU}(\text{Linear}(\text{LN}(E)))\big)$  (projector → decoder width)
7. **return** $A$

Convolutions cut the frame rate and mix local spectro-temporal detail; a bidirectional transformer (windowed, so long audio splits into independent chunks) contextualizes; a two-layer **projector** maps the encoder space into the decoder's embedding space. The encoder is pretrained on ASR, then bolted onto the LLM — the projector is what reconciles the two separately-trained representation spaces.

### TranscribeAR(wav)

**Input** waveform.  **Output** transcript text.

1. $A \leftarrow \text{AudioEncoderLLM}(\text{wav});\quad N \leftarrow |A|$
2. $\text{ids} \leftarrow \text{PREFIX} \;\Vert\; [\,\texttt{audio\_pad}\,]\times N \;\Vert\; \text{SUFFIX}$  (chat template: system / user + `<audio_start>…<audio_end>` / assistant + `<asr_text>`)
3. $\text{emb} \leftarrow \text{EmbedTokens}(\text{ids})$
4. **for** each position $p$ with $\text{ids}[p] = \texttt{audio\_pad}$: $\ \text{emb}[p] \leftarrow A[\,\cdot\,]$  (**replace** the placeholder embedding with the audio embedding)
5. $h \leftarrow \text{TransformerStack}(\text{emb},\ \text{causal})$  (prefill; builds the KV cache)
6. $y \leftarrow [\,];\quad \text{tok} \leftarrow \arg\max(h[\text{last}]\,W_{\text{lmhead}})$
7. **while** $\text{tok} \notin \{\text{EOS}\}$:
    1. $y.\text{append}(\text{tok})$
    2. $h \leftarrow \text{TransformerStack step}(\text{EmbedTokens}(\text{tok}),\ \text{causal})$  (KV-cached, one new position)
    3. $\text{tok} \leftarrow \arg\max(h\,W_{\text{lmhead}})$  (greedy; ASR runs low/zero temperature)
8. **return** ParseTranscript$(y)$  (split on `<asr_text>`; the model first emits a language tag)

The decoder is a standard Qwen3 causal stack (GQA, per-head Q/K-RMSNorm, RoPE, SwiGLU). The key wiring is step 4: audio embeddings **replace** the `audio_pad` token embeddings in-place, so the LLM reads audio and text in one homogeneous sequence and just keeps predicting the next token — transcription becomes ordinary conditional language modeling. **(WHY greedy / low temperature?** ASR is a near-deterministic mapping; sampling would only invent errors.**)** Whisper is the earlier encoder-decoder form of the same idea: a transformer decoder that **cross-attends** to the audio encoder (as in AEDDecodeStep) rather than reading audio embeddings inline, with special tokens selecting language/task/timestamps.

---

## Part 4 — Diffusion-NAR ASR (Whisfusion / dLLM-ASR style)

*Does diffusion ASR exist?* Yes, but **not as "diffusion-AR."** In TTS, a diffusion sampler sits *inside* the autoregressive loop, drawing one continuous latent per step. ASR output is discrete text, so diffusion there is used the opposite way: a **discrete / masked diffusion** model that **replaces** the autoregressive loop with parallel iterative refinement of the whole transcript. It is non-autoregressive (NAR). This is research-stage (2025–2026), motivated by latency — decode cost is roughly independent of transcript length.

### MaskedDiffusionDecode(wav, $L$, $K$)

**Input** waveform; target length $L$; refinement steps $K$.  **Output** transcript.

1. $A \leftarrow \text{AudioEncoderLLM}(\text{wav})$
2. $y \leftarrow [\,\texttt{MASK}\,]\times L$  (or initialize from a cheap ASR prior, then adapt $L$ — dLLM-ASR)
3. **for** step $s = 1$ to $K$:
    1. $\text{logits} \leftarrow \text{DiffusionDecoder}(y,\ A)$  (**bidirectional** self-attention over $y$ + cross-attention/conditioning on $A$; predicts **all** positions at once)
    2. **for** each position $j$: $\ \hat y_j \leftarrow \arg\max(\text{logits}_j);\ \ \text{conf}_j \leftarrow \max\text{softmax}(\text{logits}_j)$
    3. $n_{\text{keep}} \leftarrow \lfloor L\cdot s/K \rfloor$  (unmasking schedule — reveal more each step)
    4. $\text{keep} \leftarrow$ the $n_{\text{keep}}$ positions with highest $\text{conf}$
    5. **for** each position $j$: $\ y[j] \leftarrow \hat y_j$ **if** $j\in\text{keep}$ **else** $\texttt{MASK}$
4. **return** Detokenize$(y)$

Start from an all-`MASK` transcript and denoise it in $K$ passes: each pass re-predicts every token in parallel conditioned on the audio and the currently-revealed tokens, commits the **most confident** ones, and re-masks the rest for the next pass (coarse-to-fine). Because a pass sees the whole sequence bidirectionally, it captures output dependencies CTC cannot, without left-to-right latency. **(WHY confidence-ordered unmasking?** committing certain tokens first gives the uncertain positions more context, and lets easy tokens exit early — dLLM-ASR prunes converged tokens to cut steps.**)** Open issues: the length $L$ must be set or predicted (over-allocation wastes compute, under-allocation truncates), and quality still trails strong AR LLM-ASR — the speedup (≈2–4×) is the draw.

---

## Part 5 — Audio-native reasoning: a separate paradigm?

**No — it is Part 3's architecture, repurposed.** "Audio-native" means the model reasons on **audio embeddings directly** instead of on an ASR transcript. But feeding audio embeddings into a transformer decoder is exactly what `AudioEncoderLLM → TranscribeAR` already does. The distinction is not the wiring, it is **what the decoder LLM is and what it was trained to do**:

- **ASR speech-LLM** (Qwen3-ASR) — the decoder is fine-tuned to *only transcribe*; its output space is collapsed to "write what was said."
- **Audio-native reasoner** (Qwen3-Omni, Qwen2-Audio, GPT-4o-audio) — the *same* encoder→projector→LLM stack, but the decoder is a general instruction-tuned LLM, so the audio embeddings become first-class input tokens it can **reason over, answer, or act on** — jointly with interleaved text — rather than transcribe.

So general AR *architectures* already ingest audio directly; "audio-native reasoning" is a **capability/training distinction layered on the same architecture**, not a new one. Its whole value is avoiding the cascade: an ASR transcript is a lossy, silently-lossy bottleneck (a mistranscribed entity breaks a downstream tool call; prosody and intent are deleted), so routing the continuous audio embeddings straight into the reasoner preserves what a transcript would throw away. The engineering cost is that an ASR-specialized final training stage can overwrite the base model's reasoning, so alignment must recover reasoning while preserving transcription (see the companion SFT/DPO note).

---

## The paradigms side by side

| | Neural ASR (CTC / RNN-T / AED) | Speech-to-Intent | General transformer AR | Diffusion-NAR |
|---|---|---|---|---|
| Output | transcript tokens | intent + slots | transcript tokens | transcript tokens |
| Token dependency | none (CTC) / label-LM (RNN-T) / cross-attn (AED) | pooled / seq2seq | full LLM prior, left-to-right | bidirectional, parallel |
| Decode order | streaming per-frame (RNN-T) | one shot | autoregressive $\mathcal{O}(U)$ steps | iterative $\mathcal{O}(K)$ parallel passes |
| Language modeling | weak (CTC) → built-in (RNN-T/AED) | intent-level only | strongest (pretrained LLM) | moderate |
| Streaming | native (CTC, RNN-T) | native | chunked / re-encode | not naturally |
| Main trait | compact, low-latency, production-proven | robust to ASR errors, direct meaning | best accuracy, rare words, punctuation; heavier | length-independent latency; research-stage |
