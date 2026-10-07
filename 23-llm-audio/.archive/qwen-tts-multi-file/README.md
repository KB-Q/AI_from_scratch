# Qwen3-TTS (from scratch, toy scale)

A small, readable PyTorch reimplementation of the **Qwen3-TTS** architecture
([technical report, arXiv:2601.15621](https://arxiv.org/abs/2601.15621)). The
goal is to understand the *ideas*, not to reproduce the 0.6B/1.7B models or
their audio quality. Everything is sized to train fast on a laptop (~12-14M
params per variant), and the two model families from the paper are both here.

## The core idea: TTS as a language model

Qwen3-TTS treats speech synthesis as autoregressive token prediction on a
**dual-track** sequence. A short conditioning prefix is built and the model then
predicts speech tokens one step at a time:

```
[speaker embed] [text tokens] [BOS] -> speech token, speech token, ... , [EOS]
└──────────── conditioning prefix ───────────┘ └──── autoregressive speech ────┘
```

Text and speech share **one** transformer stack and **one** position line — two
token vocabularies flowing through a single sequence, hence "dual-track". A
discrete **neural codec** provides the speech-token vocabulary (audio ->
tokens -> audio). The two variants differ in how tokens become audio.

## Two variants (both implemented)

| | **Qwen3TTS12Hz** (RVQ + MTP) | **Qwen3TTS25Hz** (Flow-Matching) |
|---|---|---|
| Codec | Multi-codebook **RVQ** (NQ codebooks/frame) | Single **semantic** codebook |
| Backbone predicts | Codebook 0 (coarse/semantic) | The one semantic token |
| Fills detail via | **MTP** head → residual codebooks 1..NQ-1 | **Flow-Matching DiT** → mel-spectrogram |
| Waveform via | Causal-conv codec decoder (direct) | **Vocoder** (mel → wav, BigVGAN-style) |
| Idea it teaches | RVQ + single-frame multi-token prediction | conditional generative decoding of audio |

Both share `DualTrackBackbone` (the Qwen3-style trunk) and `SpeakerEncoder`.

## File map (idea → file)

| File | What lives here |
|---|---|
| `layers.py` | Qwen3 primitives: **RMSNorm**, **RoPE**, **grouped-query attention with QK-norm**, **SwiGLU**, pre-norm block; plus cross-attention used by the DiT |
| `codec.py` | Causal conv encoder/decoder, **VectorQuantizer**, **ResidualVQ**, `MultiCodebookCodec` (12Hz), `SingleCodebookCodec` (25Hz), log-mel helper |
| `backbone.py` | `SpeakerEncoder` (reference audio → voice vector) and `DualTrackBackbone` (the shared AR trunk + prompt assembly) |
| `mtp.py` | `MultiTokenPredictor` — a tiny transformer over the **codebook axis** that predicts all RVQ codebooks of a frame in one backbone step |
| `flow_dit.py` | `FlowMatchingDiT` (adaLN timestep conditioning + cross-attention to token features) and `MelVocoder` |
| `qwen_tts.py` | Top-level `Qwen3TTS12Hz` / `Qwen3TTS25Hz`: prompt → loss → `generate`, plus `save`/`load` and a runnable smoke test |

## Qwen3-style backbone vs. the older `gpt_torch.py`

The trunk uses the modern internals that make a Qwen3 transformer, which is why
it does **not** reuse the blocks in `../torch/gpt_torch.py`:

- **RMSNorm** instead of LayerNorm (no mean subtraction, no bias)
- **RoPE** relative position instead of additive sinusoidal encodings
- **Grouped-query attention** (fewer KV heads than Q heads) + **QK-norm**
  (RMSNorm on per-head q/k), a Qwen3 stability trick
- **SwiGLU** gated FFN instead of a plain GELU MLP
- **Pre-norm** residual blocks: `x + attn(norm(x))`, `x + ffn(norm(x))`

## Multi-Token Prediction (MTP), concretely

The RVQ codec emits `NQ` codebooks per frame. The backbone predicts only
codebook 0. MTP then produces codebooks `1..NQ-1` for that same frame so a full
frame is emitted per backbone step. It runs a small **causal transformer along
the codebook axis** (length `NQ`): position 0 is the backbone hidden state,
position `k` is the embedding of codebook `k-1`'s chosen index, and the output
at position `k` gives the logits for codebook `k`. This mirrors how RVQ
residuals depend on the earlier codebooks.

## Flow Matching, concretely

To turn semantic tokens into a mel, the 25Hz path trains a DiT with flow
matching. Define a straight path from noise `x0 ~ N(0, I)` to the data mel `x1`:

```
x_t = (1 - t) * x0 + t * x1,   t ∈ [0, 1]      (constant velocity  x1 - x0)
```

The DiT `v_θ(x_t, t, c)` regresses that velocity (MSE), conditioned on the token
features `c` via cross-attention and on `t` via adaLN. Sampling starts from
noise and Euler-integrates `dx = v_θ · dt` up to `t = 1` in a handful of steps —
simpler and faster than score-based diffusion (the F5-TTS / CosyVoice recipe).

## Voice cloning & control

`SpeakerEncoder` turns a reference clip into one embedding vector, projected and
prepended as the first "token" of the prefix — this is the voice-cloning hook.
Instruction/style control from the paper (prepended text instructions, the
"thinking" toggle) is **not** implemented; it slots in naturally as extra text
tokens in the prefix.

## What is simplified vs. the paper

- **Scale**: ~12-14M params vs. 0.6B/1.7B; frame rates are toy ("12Hz"/"25Hz"
  name the *design family*, not the literal rate here).
- **Codec**: a compact VQ/RVQ trained by reconstruction, not the two-stage
  Qwen2-Audio-based semantic tokenizer (WavLM teacher, GAN losses) in the paper.
- **Vocoder**: a small transposed-conv net standing in for BigVGAN.
- **Streaming**: modules are built **causally** (causal convs, causal attention)
  so streaming is possible in principle, but block-wise/sliding-window streaming
  inference and first-packet-latency tricks are not implemented.
- **DiT**: full (non-windowed) attention over mel frames rather than the paper's
  block-wise sliding window.

## Training note (why some params show "no gradient")

The codec (and the 25Hz vocoder) are **not** trained by the LM/flow loss. They
have their own reconstruction objectives:

- `MultiCodebookCodec.forward(wav) -> (recon, codes, vq_loss)` — reconstruction
  + VQ/commitment loss.
- `SingleCodebookCodec.encode_with_loss(wav)` — VQ loss for its encoder.
- The `MelVocoder` trains on a mel → waveform reconstruction loss.

So a full setup trains in two logical stages (codec/vocoder first, then the
LM + MTP/DiT on the frozen codec's tokens), or jointly with the losses summed.
`compute_loss` here covers the **LM side** (token prediction + flow matching);
`generate` uses the codec/vocoder to decode. When you call only `compute_loss`,
the codec/vocoder params correctly receive no gradient — that is expected, not a
bug. Data loading and the training loop are intentionally left for the next step.

## Run the smoke test

```bash
cd 23-llm-audio/.archive/qwen-tts-multi-file
python3 qwen_tts.py
```

This builds both variants on random data and exercises `codec.encode`,
`compute_loss` (forward), and `generate` (autoregressive tokens → waveform),
printing shapes and parameter counts.
