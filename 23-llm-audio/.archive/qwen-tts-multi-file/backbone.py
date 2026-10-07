"""
Dual-track autoregressive backbone for Qwen3-TTS.

This is the shared heart of both model variants. The idea ("TTS as an LM"):

  1. A short prompt of conditioning tokens is built:
        [speaker embedding] [text tokens] [BOS]
     - the speaker embedding is a single vector from a reference clip, projected
       into the model width and prepended as one "token" (voice cloning);
     - text tokens describe *what* to say;
     - a BOS marker begins the speech track.
  2. From there the model autoregressively predicts speech tokens. Text and
     speech share one transformer stack and one position line - hence
     "dual-track": two token vocabularies flowing through a single sequence.

The backbone itself is Qwen3-style (RMSNorm + RoPE + grouped-query attention +
SwiGLU) and is codebook-agnostic. What sits on top of it differs by variant:

  * 12Hz: a head predicting codebook 0, then an MTP module (mtp.py) for the
    residual codebooks.
  * 25Hz: a single head predicting the semantic token, then a Flow-Matching
    DiT (flow_dit.py) for waveform.

so this file exposes the trunk (`DualTrackBackbone`) and the shared embedding /
conditioning logic; the variant assembly lives in `qwen_tts.py`.
"""
import torch
import torch.nn as nn

from layers import Qwen3Block, RMSNorm, RotaryEmbedding, causal_mask, count_parameters


class SpeakerEncoder(nn.Module):
    """Turns a reference waveform into a single speaker embedding vector.

    A small stack of strided 1D convs over the raw waveform followed by mean
    pooling over time. Real systems use x-vectors / ECAPA-TDNN; this keeps the
    same interface (audio in, fixed-size voice vector out) at toy scale.
    """

    def __init__(self, embed_dim=128, channels=(16, 32, 64)):
        super().__init__()
        convs = []
        in_ch = 1
        for out_ch in channels:
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=4, padding=3),
                nn.ELU(),
            ]
            in_ch = out_ch
        self.convs = nn.Sequential(*convs)
        self.proj = nn.Linear(in_ch, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, wav):
        """wav: (batch, samples) or (batch, 1, samples) -> (batch, embed_dim)."""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        h = self.convs(wav)            # (batch, channels, time)
        h = h.mean(dim=-1)             # mean-pool over time
        return self.proj(h)


class DualTrackBackbone(nn.Module):
    """Qwen3-style transformer trunk over the [speaker][text][speech] sequence.

    It builds the input embeddings from three sources, runs them through the
    shared blocks with a causal mask, and returns per-position hidden states.
    Heads are attached by the variant models.
    """

    def __init__(
        self,
        text_vocab_size,
        speech_vocab_size,
        d_model=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        d_ff=768,
        max_seq_len=1024,
        speaker_dim=128,
    ):
        super().__init__()
        self.d_model = d_model
        self.text_vocab_size = text_vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.max_seq_len = max_seq_len

        # Separate embedding tables per track keep the two vocabularies clean.
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.speech_embedding = nn.Embedding(speech_vocab_size, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))  # start-of-speech marker
        self.speaker_proj = nn.Linear(speaker_dim, d_model)
        nn.init.normal_(self.text_embedding.weight, std=0.02)
        nn.init.normal_(self.speech_embedding.weight, std=0.02)
        nn.init.normal_(self.bos, std=0.02)

        head_dim = d_model // num_heads
        self.rotary = RotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            [Qwen3Block(d_model, num_heads, num_kv_heads, d_ff) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(d_model)

    def build_prompt(self, text_ids, speaker_embed):
        """Assemble the conditioning prefix: [speaker][text][BOS].

        Returns (prefix_embeds, prefix_len). text_ids: (batch, text_len);
        speaker_embed: (batch, speaker_dim) or None.
        """
        batch = text_ids.shape[0]
        parts = []
        if speaker_embed is not None:
            parts.append(self.speaker_proj(speaker_embed).unsqueeze(1))  # (b,1,d)
        parts.append(self.text_embedding(text_ids))                      # (b,T,d)
        parts.append(self.bos.expand(batch, -1, -1))                     # (b,1,d)
        prefix = torch.cat(parts, dim=1)
        return prefix, prefix.shape[1]

    def run_blocks(self, x):
        """x: (batch, seq_len, d_model) -> hidden states (same shape)."""
        seq_len = x.shape[1]
        cos, sin = self.rotary(seq_len, x.device)
        mask = causal_mask(seq_len, x.device)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.final_norm(x)

    def forward(self, text_ids, speech_embeds, speaker_embed=None):
        """Teacher-forced forward over prompt + given speech embeddings.

        text_ids: (batch, text_len)
        speech_embeds: (batch, speech_len, d_model) - embeddings of the speech
            tokens fed as input (shifted by the caller for next-token training).
        Returns (hidden, prefix_len): hidden is (batch, total_len, d_model);
        the speech predictions are hidden[:, prefix_len-1 : -1] aligned to
        speech targets. See qwen_tts.py for how heads consume this.
        """
        prefix, prefix_len = self.build_prompt(text_ids, speaker_embed)
        x = torch.cat([prefix, speech_embeds], dim=1)
        hidden = self.run_blocks(x)
        return hidden, prefix_len

    def num_parameters(self):
        return count_parameters(self)
