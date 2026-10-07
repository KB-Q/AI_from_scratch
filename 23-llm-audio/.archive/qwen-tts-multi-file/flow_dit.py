"""
Flow-Matching Diffusion Transformer (DiT) + vocoder for the 25Hz path.

The 25Hz backbone predicts a sequence of single-codebook *semantic* tokens.
Those tokens say roughly *what* to say but not the fine acoustic detail, so a
generative decoder turns them into a mel-spectrogram, and a small vocoder turns
the mel into a waveform. The paper uses a block-wise DiT trained with Flow
Matching plus a BigVGAN vocoder; we implement the same shape at toy scale.

Flow Matching in one paragraph:
  We want to sample mel `x1` given the token features `c`. Define a straight
  path from noise `x0 ~ N(0, I)` to data `x1`:  x_t = (1 - t) * x0 + t * x1,
  for t in [0, 1]. The path's velocity is constant: dx_t/dt = x1 - x0. We train
  a network v_theta(x_t, t, c) to regress that velocity (MSE). To sample, start
  from noise at t=0 and integrate dx = v_theta * dt with a few Euler steps up to
  t=1. This is simpler and faster than score-based diffusion and is what F5-TTS
  / E2-TTS / CosyVoice-style systems use.

The DiT conditions on the timestep via adaLN (adaptive LayerNorm: the timestep
embedding produces per-block scale/shift/gate) and on the token features via
cross-attention.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MultiHeadCrossAttention, RMSNorm, RotaryEmbedding, GroupedQueryAttention


def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of a scalar timestep t in [0, 1]. t: (batch,)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device).float() / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb  # (batch, dim)


class DiTBlock(nn.Module):
    """A DiT block: adaLN-conditioned self-attn + cross-attn + MLP.

    The timestep embedding `c` produces six modulation vectors (scale/shift/gate
    for the attention path and the MLP path), the standard adaLN-Zero recipe.
    """

    def __init__(self, d_model, num_heads, d_ctx, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = GroupedQueryAttention(d_model, num_heads, num_heads)
        self.norm_ctx = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, d_ctx)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        # adaLN: map timestep embedding -> 6 modulation vectors.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cos, sin, cond, ctx, ctx_mask=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)
        shift1, scale1, gate1 = shift1[:, None], scale1[:, None], gate1[:, None]
        shift2, scale2, gate2 = shift2[:, None], scale2[:, None], gate2[:, None]

        # Self-attention over mel frames (adaLN-modulated), then cross-attention
        # to the token features, then the MLP (adaLN-modulated).
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.self_attn(h, cos, sin, mask=None)
        x = x + self.cross_attn(self.norm_ctx(x), ctx, ctx_mask)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


class FlowMatchingDiT(nn.Module):
    """Predicts the flow velocity for mel generation, conditioned on tokens.

    Token features `c` come from the 25Hz backbone (per semantic token). They
    are length T_tokens; the mel is length T_mel. We do not assume the two
    lengths match - cross-attention handles the alignment - but in practice the
    caller upsamples/repeats token features to roughly mel length first.
    """

    def __init__(self, n_mels=80, d_model=192, num_layers=4, num_heads=6, d_ctx=256, d_ff=512):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.in_proj = nn.Linear(n_mels, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        head_dim = d_model // num_heads
        self.rotary = RotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, num_heads, d_ctx, d_ff) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_proj = nn.Linear(d_model, n_mels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t, t, ctx, ctx_mask=None):
        """Predict velocity. x_t: (b, T_mel, n_mels); t: (b,); ctx: (b, T_tok, d_ctx)."""
        cond = self.time_mlp(timestep_embedding(t, self.d_model))
        h = self.in_proj(x_t)
        cos, sin = self.rotary(h.shape[1], h.device)
        for block in self.blocks:
            h = block(h, cos, sin, cond, ctx, ctx_mask)
        return self.out_proj(self.norm_out(h))

    def flow_loss(self, mel, ctx, ctx_mask=None):
        """Flow-matching training loss for a batch of mels.

        mel: (b, T_mel, n_mels) target; ctx: token features (b, T_tok, d_ctx).
        Samples t and noise, forms x_t on the straight path, regresses velocity.
        """
        b = mel.shape[0]
        t = torch.rand(b, device=mel.device)
        x0 = torch.randn_like(mel)
        x_t = (1 - t)[:, None, None] * x0 + t[:, None, None] * mel
        target_v = mel - x0                       # constant velocity of the path
        pred_v = self.forward(x_t, t, ctx, ctx_mask)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, ctx, mel_len, ctx_mask=None, steps=16):
        """Generate a mel by Euler-integrating the flow from noise to data.

        ctx: (b, T_tok, d_ctx); returns mel (b, mel_len, n_mels).
        """
        b = ctx.shape[0]
        x = torch.randn(b, mel_len, self.n_mels, device=ctx.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), i * dt, device=ctx.device)
            v = self.forward(x, t, ctx, ctx_mask)
            x = x + v * dt
        return x


class MelVocoder(nn.Module):
    """Tiny mel-spectrogram -> waveform ConvNet (stands in for BigVGAN).

    Upsamples the mel frames back to audio-rate samples with transposed convs.
    `total_upsample` must equal the mel hop length so output length matches the
    original waveform.
    """

    def __init__(self, n_mels=80, base_channels=64, upsample_factors=(4, 4, 4, 4)):
        super().__init__()
        self.total_upsample = 1
        for f in upsample_factors:
            self.total_upsample *= f

        ch = base_channels
        layers = [nn.Conv1d(n_mels, ch, kernel_size=7, padding=3), nn.ELU()]
        for f in upsample_factors:
            out_ch = max(ch // 2, 16)
            layers += [
                nn.ConvTranspose1d(ch, out_ch, kernel_size=2 * f, stride=f, padding=f // 2),
                nn.ELU(),
            ]
            ch = out_ch
        layers += [nn.Conv1d(ch, 1, kernel_size=7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, mel):
        """mel: (b, T_mel, n_mels) -> wav (b, 1, T_mel * total_upsample)."""
        x = mel.transpose(1, 2)  # (b, n_mels, T_mel)
        return torch.tanh(self.net(x))
