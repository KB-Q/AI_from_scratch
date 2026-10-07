"""
Qwen3-style transformer primitives shared across the TTS model.

These are the modern building blocks the Qwen3 backbone uses, and they differ
from the older `gpt_torch.py` blocks on purpose (that file uses LayerNorm +
sinusoidal position encoding + GELU). Here we implement the pieces that make a
Qwen3 transformer what it is:

    * RMSNorm            - cheaper, bias-free normalization
    * RotaryEmbedding    - relative position via rotating Q/K (RoPE)
    * GroupedQueryAttention with QK-norm - fewer KV heads than Q heads, plus
      an RMSNorm on the per-head query/key vectors (a Qwen3 stability trick)
    * SwiGLU             - gated feed-forward, replaces the plain GELU MLP
    * Qwen3Block         - pre-norm block: x + attn(norm(x)); x + ffn(norm(x))

Everything is deliberately small and readable rather than optimized.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize over the last dimension by its RMS, then rescale.
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputes the cos/sin tables used by rotary position embeddings.

    Call `forward(seq_len, device)` to get (cos, sin) each of shape
    (seq_len, head_dim). `head_dim` must be even.
    """

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))  # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (seq_len, head_dim)
        return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotate the two halves of the last dim: [a, b] -> [-b, a]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """Apply RoPE to x of shape (batch, num_heads, seq_len, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep):
    """Repeat KV heads so a group of Q heads can share them (GQA)."""
    if n_rep == 1:
        return x
    b, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(b, kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(b, kv_heads * n_rep, seq_len, head_dim)


def causal_mask(seq_len, device):
    """Additive (seq_len, seq_len) mask: 0 on/below diagonal, -inf above."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class GroupedQueryAttention(nn.Module):
    """Multi-head self-attention with grouped KV heads, QK-norm and RoPE."""

    def __init__(self, d_model, num_heads, num_kv_heads, head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = head_dim or (d_model // num_heads)

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        # QK-norm: RMSNorm applied to each head's query/key vector (Qwen3).
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, cos, sin, mask=None):
        """x: (batch, seq_len, d_model); mask: (seq_len, seq_len) additive."""
        b, seq_len, _ = x.shape

        q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm then rotary position embedding.
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Share KV heads across query groups.
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)

        out = attn @ v  # (b, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MultiHeadCrossAttention(nn.Module):
    """Plain multi-head cross-attention: queries from x, keys/values from ctx.

    Used by the diffusion decoder to attend from mel frames to the sequence of
    semantic-token features. No RoPE (the two streams live in different index
    spaces), no causal mask.
    """

    def __init__(self, d_model, num_heads, d_ctx=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        d_ctx = d_ctx or d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_ctx, d_model, bias=False)
        self.v_proj = nn.Linear(d_ctx, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, ctx, ctx_mask=None):
        """x: (b, Tq, d_model); ctx: (b, Tk, d_ctx); ctx_mask: (b, Tk)."""
        b, tq, _ = x.shape
        tk = ctx.shape[1]
        q = self.q_proj(x).view(b, tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if ctx_mask is not None:
            # (b, Tk) -> (b, 1, 1, Tk); masked positions get -inf.
            add = (1.0 - ctx_mask.float())[:, None, None, :] * float("-inf")
            add = torch.nan_to_num(add, nan=0.0)  # 0*-inf -> keep 0 where valid
            scores = scores + add
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, tq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        for layer in [self.gate, self.up, self.down]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Qwen3Block(nn.Module):
    """Pre-norm transformer block with GQA self-attention and a SwiGLU FFN."""

    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, head_dim=None):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def count_parameters(module):
    """Total number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
