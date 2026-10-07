# Qwen3-TTS (from scratch, minimal)

A tiny, single-file PyTorch implementation of the **core idea** of **Qwen3-TTS**
([technical report, arXiv:2601.15621](https://arxiv.org/abs/2601.15621)). This
is a learning implementation — the goal is to *understand the idea*, not to
reproduce the real models or their audio quality. ~1.25M params, trains in
seconds on a laptop.

## The one idea

> **Text-to-speech is an autoregressive language model over a single sequence
> `[speaker][text tokens][BOS] → discrete speech tokens`, where the speech
> tokens come from a small audio codec.**

That's it. A neural **codec** turns audio into a short sequence of discrete
tokens (and back), and a transformer predicts those tokens from text — exactly
how a text LM predicts words. Text and speech share one transformer and one
position line, which is what "dual-track" means. To keep it Qwen3-flavored, each
step emits a whole audio *frame* via **Multi-Token Prediction (MTP)**.

## Everything lives in one file: `qwen3_tts.py`

Read it top to bottom; it's organized in six numbered sections:

| Section | What it is |
|---|---|
| 1. Qwen3 primitives | `RMSNorm`, RoPE (`RotaryEmbedding`/`apply_rope`), grouped-query attention (`GQA`) with QK-norm, `SwiGLU`, and the pre-norm `Block` |
| 2. `TinyCodec` | A small **RVQ VQ-VAE**: conv encoder → residual vector quantizer → conv decoder. Provides the speech-token vocabulary (audio ↔ tokens) |
| 3. `Backbone` + `SpeakerEncoder` | The dual-track autoregressive transformer, and the reference-audio → voice-vector encoder (the voice-cloning hook) |
| 4. `MTPHead` | Predicts the **residual** codebooks (1..NQ-1) of a frame per step via a tiny transformer over the codebook axis; codebook 0 comes from the backbone |
| 5. `Qwen3TTS` | Assembly: `compute_loss`, `codec_loss`, `generate`, `save`/`load` |
| 6. smoke test | `python3 qwen3_tts.py` |

## How it flows

```
reference audio ─► SpeakerEncoder ─┐
                                   ▼
text ─► text tokens ─► [speaker][text][BOS] ─► Backbone ─► code0 (coarse token)
                                                             │
                                                             ▼   per frame
                                                          MTPHead ─► codebooks 1..NQ-1
                                                             │
                                     all NQ codebooks per frame
                                                             ▼
                                              TinyCodec.decode ─► waveform
```

Training has two independent objectives (see the smoke test):

- `codec_loss(wav)` — reconstruction + VQ loss, trains the **codec** to turn
  audio into good tokens.
- `compute_loss(text_ids, codes, speaker_wav)` — the **language-model** loss:
  next-token cross-entropy on codebook 0, plus the MTP loss on the residual
  codebooks (1..NQ-1).
  (This is why the codec's parameters get no gradient from `compute_loss` — they
  learn from `codec_loss` instead. Normal setup: train the codec first, then the
  LM on the frozen codec's tokens.)

## Why "Qwen3"-style and not a plain GPT

The backbone uses the modern internals that distinguish a Qwen3 transformer from
the vanilla GPT in [08-transformers/torch/gpt_torch.py](../../08-transformers/torch/gpt_torch.py):

- **RMSNorm** instead of LayerNorm
- **RoPE** rotary positions instead of additive sinusoidal encodings
- **Grouped-query attention** (fewer KV heads than Q heads) + **QK-norm**
- **SwiGLU** gated FFN instead of a plain GELU MLP

## What's left out (on purpose)

This keeps only the essence. The following ideas from the paper are **not** here:
a second single-codebook path, a **flow-matching Diffusion Transformer**, a
**BigVGAN-style vocoder**, a semantic tokenizer with a WavLM teacher, streaming
inference, and instruction/style control. A faithful, fuller build of all of
that lives in **[`.archive/qwen-tts-multi-file/`](../.archive/qwen-tts-multi-file/)** (6 files, ~12-14M params) if you want
to see the complete architecture — start with its `README.md`.

## Run it

```bash
cd 23-llm-audio/torch
python3 qwen3_tts.py
```

Builds the model on random data and exercises codec reconstruction,
`compute_loss` (+ backward), and `generate` (text → tokens → waveform), printing
shapes and the ~1.25M parameter count.

## Next step

No dataset or training loop yet (intentionally deferred). The hooks are ready:
`codec_loss` for codec pretraining and `compute_loss` for the LM. A small real
audio subset + a `train.py` is the natural follow-up.
