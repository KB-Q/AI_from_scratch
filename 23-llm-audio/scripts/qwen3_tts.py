"""
Qwen3-TTS, minimal from-scratch version (learning edition).

ONE core idea, in one file:

    Text-to-speech = an autoregressive language model over a single sequence
    "[speaker][text tokens][BOS] -> discrete speech tokens", where the speech
    tokens come from a small audio codec, and each step emits a whole audio
    *frame* via Multi-Token Prediction (MTP).

That is the essence of Qwen3-TTS's 12.5Hz path. This file keeps just enough to
see it work end to end and train in seconds on a laptop (~1-2M params). The
full, paper-faithful build (two paths, flow-matching DiT, vocoder, richer
codec) lives in .archive/qwen-tts-multi-file/ (next to this file) and is referenced from notes/tts-qwen3.md.

Read the file top to bottom:
    1. Qwen3 primitives   - RMSNorm, RoPE, grouped-query attention, SwiGLU
    2. TinyCodec          - waveform <-> RVQ tokens (the speech vocabulary)
    3. Backbone           - the dual-track autoregressive transformer
    4. MTP head           - fills a frame's residual codebooks (1..NQ-1) per step
    5. Qwen3TTS           - assembly: compute_loss / generate / save / load
    6. smoke test         - run `python3 qwen3_tts.py`

Scope & simplifications (vs. arXiv:2601.15621):
    - Only the 12.5Hz variant (multi-codebook RVQ + MTP + conv decoder). The
      paper's other path (25Hz single semantic codebook -> block-wise DiT ->
      vocoder) is omitted; that decoder is the complex, non-LM part.
    - No streaming: the codec convs are non-causal and the LM reads the whole
      text prefix before emitting speech, so the paper's real-time, time-aligned
      dual-track (97ms first-packet) is not reproduced.
    - The codec is a plain reconstruction-trained RVQ, not the paper's semantic
      tokenizer (Qwen-Audio / WavLM-teacher).
    - Description-based style control is not implemented (it slots in as extra
      text tokens in the prefix).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Qwen3 primitives
#    These are what make the backbone "Qwen3-style" rather than a vanilla GPT
#    (uses LayerNorm + sinusoidal positions + GELU, see ../../08-transformers/scripts/torch/gpt_torch.py).
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square norm: rescale by RMS, no mean subtraction, no bias."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Builds the (cos, sin) tables for rotary position embedding (RoPE).

    RoPE encodes position by *rotating* the query/key vectors (see apply_rope),
    so unlike sinusoidal PE nothing is added to the residual stream - this
    module just precomputes the rotation angles' cos/sin for a given length.
    Wrapping it as a module keeps the frequency math (arange/outer/cos/sin) as
    one labelled block instead of loose ops in the graph.
    """

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def forward(self, x):
        # x is only used for its sequence length and device (RoPE tables don't
        # depend on x's values). Taking a tensor lets it appear as one block.
        seq_len, device = x.shape[1], x.device
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)          # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim)
        return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Rotate q/k so attention depends on *relative* position. x: (b,h,T,d)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, None] + rotated * sin[None, None]


class GQA(nn.Module):
    """Grouped-query self-attention with QK-norm and RoPE.

    Grouped-query: fewer key/value heads than query heads (they are shared
    across a group of query heads) - a standard Qwen3 efficiency choice.
    QK-norm: an RMSNorm on each head's query and key, for training stability.
    """

    def __init__(self, d_model, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.nh, self.nkv = num_heads, num_kv_heads
        self.hd = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.q = nn.Linear(d_model, num_heads * self.hd, bias=False)
        self.k = nn.Linear(d_model, num_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(d_model, num_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(num_heads * self.hd, d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.nh, self.hd).transpose(1, 2)   # (b, nh, t, hd)
        k = self.k(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.nkv, self.hd).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)                        # QK-norm
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)      # RoPE

        # Share each kv head across n_rep query heads.
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        # Causal mask: a position may only attend to itself and the past.
        causal = torch.triu(torch.full((t, t), float("-inf"), device=x.device), 1)
        attn = F.softmax(scores + causal, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.nh * self.hd)
        return self.o(out)


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm Qwen3 block: x + attn(norm(x)); x + ffn(norm(x))."""

    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = GQA(d_model, num_heads, num_kv_heads)
        self.n2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        return x


# ---------------------------------------------------------------------------
# 2. TinyCodec: waveform <-> discrete tokens
#    A small Residual Vector Quantizer (RVQ) VQ-VAE. Each audio frame becomes
#    NQ codebook indices: codebook 0 is coarse, later ones add detail. This is
#    the "speech vocabulary" the language model predicts.
# ---------------------------------------------------------------------------
class TinyCodec(nn.Module):
    """Strided-conv encoder + RVQ + transposed-conv decoder."""

    def __init__(self, num_codebooks=4, codebook_size=256, latent_dim=64, strides=(4, 4, 4, 4)):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.hop = math.prod(strides)  # audio samples per frame

        # Encoder: downsample the waveform to one latent vector per frame.
        enc, ch = [], 1
        for s in strides:
            out = min(32 * (2 ** len(enc)), latent_dim)
            enc += [nn.Conv1d(ch, out, 2 * s, stride=s, padding=s // 2), nn.ELU()]
            ch = out
        enc += [nn.Conv1d(ch, latent_dim, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        # RVQ: a stack of codebooks, each quantizing the residual of the last.
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, latent_dim) for _ in range(num_codebooks)]
        )
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # Decoder: mirror the encoder to upsample latents back to a waveform.
        dec, ch = [], latent_dim
        for s in reversed(strides):
            out = max(ch // 2, 16)
            dec += [nn.ConvTranspose1d(ch, out, 2 * s, stride=s, padding=s // 2), nn.ELU()]
            ch = out
        dec += [nn.Conv1d(ch, 1, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def _quantize(self, z):
        """z: (b, frames, dim) -> (codes, quantized_sum, vq_loss)."""
        residual, quantized_sum, codes, loss = z, torch.zeros_like(z), [], 0.0
        for cb in self.codebooks:
            flat = residual.reshape(-1, self.latent_dim)
            dist = (
                flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ cb.weight.t()
                + cb.weight.pow(2).sum(1)
            )
            idx = dist.argmin(1).view(z.shape[:-1])          # nearest code
            q = cb(idx)
            loss = loss + F.mse_loss(q, residual.detach()) + 0.25 * F.mse_loss(residual, q.detach())
            q = residual + (q - residual).detach()           # straight-through
            residual = residual - q
            quantized_sum = quantized_sum + q
            codes.append(idx)
        return torch.stack(codes, -1), quantized_sum, loss / self.num_codebooks

    def encode(self, wav):
        """wav: (b, samples) -> codes (b, frames, num_codebooks)."""
        z = self.encoder(wav.unsqueeze(1)).transpose(1, 2)
        codes, _, _ = self._quantize(z)
        return codes

    def decode(self, codes):
        """codes: (b, frames, num_codebooks) -> wav (b, samples)."""
        z = sum(self.codebooks[i](codes[..., i]) for i in range(self.num_codebooks))
        return torch.tanh(self.decoder(z.transpose(1, 2))).squeeze(1)

    def forward(self, wav):
        """Reconstruction path used to *train the codec*: returns (recon, codes, vq_loss)."""
        z = self.encoder(wav.unsqueeze(1)).transpose(1, 2)
        codes, quantized, vq_loss = self._quantize(z)
        recon = torch.tanh(self.decoder(quantized.transpose(1, 2))).squeeze(1)
        return recon, codes, vq_loss


# ---------------------------------------------------------------------------
# 3. Backbone: the dual-track autoregressive transformer
#    Sequence = [speaker vector][text tokens][BOS] then predict speech tokens.
#    Text and speech share one stack and one position line ("dual-track").
# ---------------------------------------------------------------------------
class SpeakerEncoder(nn.Module):
    """Reference waveform -> one speaker embedding (the voice-cloning hook)."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=4, padding=3), nn.ELU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.ELU(),
        )
        self.proj = nn.Linear(32, dim)

    def forward(self, wav):
        h = self.net(wav.unsqueeze(1)).mean(-1)  # mean-pool over time
        return self.proj(h)


class Backbone(nn.Module):
    """Qwen3-style trunk over [speaker][text][BOS][speech...]."""

    def __init__(self, text_vocab, speech_vocab, d_model, num_layers, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.text_emb = nn.Embedding(text_vocab, d_model)
        self.speech_emb = nn.Embedding(speech_vocab, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))          # start-of-speech
        self.speaker_proj = nn.Linear(d_model, d_model)              # speaker vec -> token
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, num_kv_heads, d_ff) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model)
        nn.init.normal_(self.text_emb.weight, std=0.02)
        nn.init.normal_(self.speech_emb.weight, std=0.02)
        nn.init.normal_(self.bos, std=0.02)
        self.head_dim = d_model // num_heads
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, text_ids, speech_ids, speaker_vec):
        """Returns (hidden, prefix_len). Speech predictions start at prefix_len-1."""
        b = text_ids.shape[0]
        parts = []
        if speaker_vec is not None:
            parts.append(self.speaker_proj(speaker_vec).unsqueeze(1))  # (b,1,d)
        parts.append(self.text_emb(text_ids))                          # (b,T,d)
        parts.append(self.bos.expand(b, -1, -1))                       # (b,1,d)
        prefix = torch.cat(parts, dim=1)
        prefix_len = prefix.shape[1]

        if speech_ids is not None and speech_ids.shape[1] > 0:
            x = torch.cat([prefix, self.speech_emb(speech_ids)], dim=1)
        else:
            x = prefix

        cos, sin = self.rope(x)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x), prefix_len


# ---------------------------------------------------------------------------
# 4. MTP head: one whole audio frame per backbone step
#    The backbone predicts codebook 0. MTP fills codebooks 1..NQ-1 by running a
#    tiny transformer *along the codebook axis*: position k sees the backbone
#    state plus the codes already chosen for codebooks < k.
# ---------------------------------------------------------------------------
class MTPHead(nn.Module):
    def __init__(self, d_model, num_codebooks, codebook_size):
        super().__init__()
        self.nq = num_codebooks
        # Embed a chosen code so it can inform the next codebook's prediction.
        self.code_emb = nn.ModuleList(
            [nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks - 1)]
        )
        self.block = Block(d_model, num_heads=4, num_kv_heads=2, d_ff=2 * d_model)
        # One head per residual codebook 1..NQ-1; codebook 0 is the backbone's job.
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size, bias=False) for _ in range(num_codebooks - 1)]
        )
        self.head_dim = d_model // 4
        self.rope = RotaryEmbedding(self.head_dim)

    def _run(self, seq):
        cos, sin = self.rope(seq)
        return self.block(seq, cos, sin)

    def forward(self, hidden, codes):
        """Teacher-forced. hidden: (N, d); codes: (N, NQ).

        Returns NQ-1 logits, one per residual codebook 1..NQ-1. Codebook-axis
        position k has seen the backbone hidden plus codes 0..k-1, so its head
        predicts codebook k; codebook 0 comes from the backbone, not here.
        """
        seq = [hidden.unsqueeze(1)] + [
            self.code_emb[k](codes[:, k]).unsqueeze(1) for k in range(self.nq - 1)
        ]
        out = self._run(torch.cat(seq, dim=1))            # (N, NQ, d)
        return [self.heads[k](out[:, k + 1]) for k in range(self.nq - 1)]

    @torch.no_grad()
    def generate(self, hidden, code0, temperature=1.0):
        """Given codebook-0 indices, sample codebooks 1..NQ-1. Returns (N, NQ)."""
        chosen = [code0]
        seq = torch.cat([hidden.unsqueeze(1), self.code_emb[0](code0).unsqueeze(1)], dim=1)
        for k in range(1, self.nq):
            logits = self.heads[k - 1](self._run(seq)[:, -1]) / temperature
            idx = torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)
            chosen.append(idx)
            if k < self.nq - 1:
                seq = torch.cat([seq, self.code_emb[k](idx).unsqueeze(1)], dim=1)
        return torch.stack(chosen, -1)


# ---------------------------------------------------------------------------
# 5. Qwen3TTS: the assembled model
# ---------------------------------------------------------------------------
class Qwen3TTS(nn.Module):
    def __init__(
        self,
        text_vocab,
        num_codebooks=4,
        codebook_size=256,
        d_model=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
        sample_rate=16000,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.text_vocab = text_vocab
        self.d_model = d_model
        self.EOS = codebook_size  # extra id on the codebook-0 track for end-of-speech
        code0_vocab = codebook_size + 1

        self.speaker = SpeakerEncoder(d_model)
        self.codec = TinyCodec(num_codebooks, codebook_size)
        self.backbone = Backbone(
            text_vocab, code0_vocab, d_model, num_layers, num_heads, num_kv_heads, d_ff
        )
        self.code0_head = nn.Linear(d_model, code0_vocab, bias=False)
        self.mtp = MTPHead(d_model, num_codebooks, codebook_size)

    def forward(self, text_ids, codes, speaker_wav=None):
        """Default path = the LM training objective. TTS has no single
        tensor-in/tensor-out forward (codec and LM train separately, and
        synthesis is an autoregressive loop), so this just aliases
        compute_loss; use codec_loss / generate for the other paths."""
        return self.compute_loss(text_ids, codes, speaker_wav)

    # ---- training (language-model side; codec is trained separately) ----
    def compute_loss(self, text_ids, codes, speaker_wav=None):
        """codes: (b, frames, NQ) from the codec. Returns (loss, parts)."""
        b, frames, _ = codes.shape
        device = codes.device
        speaker_vec = self.speaker(speaker_wav) if speaker_wav is not None else None

        code0 = codes[..., 0]                                        # (b, frames)
        eos = torch.full((b, 1), self.EOS, dtype=torch.long, device=device)
        target0 = torch.cat([code0, eos], dim=1)                     # predict next code0, end with EOS

        hidden, prefix_len = self.backbone(text_ids, code0, speaker_vec)
        speech_hidden = hidden[:, prefix_len - 1:]                   # (b, frames+1, d)

        loss_code0 = F.cross_entropy(
            self.code0_head(speech_hidden).reshape(-1, self.EOS + 1), target0.reshape(-1)
        )

        # MTP: fill residual codebooks 1..NQ-1 of each real frame (codebook 0 is loss_code0).
        frame_hidden = speech_hidden[:, :frames].reshape(-1, self.d_model)
        flat = codes.reshape(-1, self.num_codebooks)
        mtp_logits = self.mtp(frame_hidden, flat)                    # logits for codebooks 1..NQ-1
        loss_mtp = sum(
            F.cross_entropy(mtp_logits[k], flat[:, k + 1]) for k in range(self.num_codebooks - 1)
        ) / (self.num_codebooks - 1)

        total = loss_code0 + loss_mtp
        return total, {"code0": loss_code0.item(), "mtp": loss_mtp.item()}

    def codec_loss(self, wav):
        """Reconstruction + VQ loss to train the codec. Returns (loss, recon)."""
        recon, _, vq_loss = self.codec(wav)
        # Conv padding can make recon a few samples shorter/longer; align both.
        n = min(recon.shape[-1], wav.shape[-1])
        return F.mse_loss(recon[..., :n], wav[..., :n]) + vq_loss, recon[..., :n]

    # ---- inference ----
    @torch.no_grad()
    def generate(self, text_ids, speaker_wav=None, max_frames=200, temperature=1.0):
        """Autoregress codebook 0, expand each frame with MTP, decode to audio.

        Returns (wav, codes). wav: (b, samples); codes: (b, frames, NQ).
        """
        self.eval()
        b, device = text_ids.shape[0], text_ids.device
        speaker_vec = self.speaker(speaker_wav) if speaker_wav is not None else None

        code0_seq = torch.empty(b, 0, dtype=torch.long, device=device)
        frames, finished = [], torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_frames):
            hidden, _ = self.backbone(text_ids, code0_seq, speaker_vec)
            last = hidden[:, -1]                                     # predicts next code0
            logits = self.code0_head(last) / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)

            finished = finished | (nxt == self.EOS)
            if finished.all():
                break
            code0 = torch.clamp(nxt, max=self.codebook_size - 1)     # keep EOS out of the codec
            frames.append(self.mtp.generate(last, code0, temperature))
            code0_seq = torch.cat([code0_seq, code0.unsqueeze(1)], dim=1)

        if not frames:
            return torch.zeros(b, 1, device=device), torch.zeros(b, 0, self.num_codebooks, device=device)
        codes = torch.stack(frames, dim=1)                           # (b, frames, NQ)
        return self.codec.decode(codes), codes

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):
        cfg = {"text_vocab": self.text_vocab, "num_codebooks": self.num_codebooks,
               "codebook_size": self.codebook_size, "d_model": self.d_model}
        torch.save({"config": cfg, "state_dict": self.state_dict()}, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        print(f"Model loaded from {path}")
        return model


# ---------------------------------------------------------------------------
# 6. Smoke test:  python3 qwen3_tts.py
# ---------------------------------------------------------------------------
def _smoke_test():
    torch.manual_seed(0)
    text_vocab, b, sr = 64, 2, 16000
    text_ids = torch.randint(0, text_vocab, (b, 10))
    wav = torch.randn(b, sr) * 0.1  # 1 second of toy audio

    model = Qwen3TTS(text_vocab)
    print(f"params: {model.num_parameters():,}")

    # (a) codec reconstruction step
    codec_loss, recon = model.codec_loss(wav)
    print(f"codec loss: {codec_loss.item():.4f}  recon: {tuple(recon.shape)}")

    # (b) language-model step on the codec's tokens
    codes = model.codec.encode(wav)
    print(f"codes: {tuple(codes.shape)}  (batch, frames, num_codebooks)")
    loss, parts = model.compute_loss(text_ids, codes, speaker_wav=wav)
    loss.backward()
    print(f"LM loss: {loss.item():.4f}  parts: {parts}")

    # (c) generation
    gen_wav, gen_codes = model.generate(text_ids, speaker_wav=wav, max_frames=15)
    print(f"generated wav: {tuple(gen_wav.shape)}  codes: {tuple(gen_codes.shape)}")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    _smoke_test()
