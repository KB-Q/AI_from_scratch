"""
Multi-Token Prediction (MTP) head for the 12Hz path.

The 12Hz tokenizer emits NQ codebooks per frame (RVQ). The backbone predicts
only codebook 0 - the coarse *semantic* token - one frame at a time. The MTP
module then fills in the remaining residual codebooks (1..NQ-1) for that same
frame, so a whole frame is produced per backbone step ("single-frame instant
generation" in the paper).

The trick: codebook k is predicted from the backbone hidden state *plus* the
embeddings of codebooks 0..k-1 already chosen for this frame. A tiny transformer
runs across the codebook axis (length NQ) with a causal mask, so each codebook
attends to the earlier ones. This mirrors how the residuals depend on each
other in RVQ.
"""
import torch
import torch.nn as nn

from layers import Qwen3Block, RMSNorm, RotaryEmbedding, causal_mask


class MultiTokenPredictor(nn.Module):
    """Predicts all NQ codebooks of a frame from the backbone hidden state.

    Input embeddings along the codebook axis are:
        position 0: backbone hidden state (the frame's context)
        position k: embedding of codebook (k-1)'s chosen index
    A small causal transformer over this length-NQ axis outputs, at each
    position k, the logits for codebook k.
    """

    def __init__(self, d_model, num_codebooks, codebook_size, num_layers=2, num_heads=4, d_ff=None):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.d_model = d_model
        d_ff = d_ff or (2 * d_model)

        # One embedding table per residual codebook (0..NQ-2 feed the next step).
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks - 1)]
        )
        for emb in self.code_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

        head_dim = d_model // num_heads
        self.rotary = RotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            [Qwen3Block(d_model, num_heads, max(1, num_heads // 2), d_ff) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model)
        # A separate output head per codebook (each has its own distribution).
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size, bias=False) for _ in range(num_codebooks)]
        )

    def _sequence(self, hidden, codes=None):
        """Build the codebook-axis input sequence.

        hidden: (N, d_model) backbone states for N frames.
        codes:  (N, num_codebooks) ground-truth indices (teacher forcing) or
                None (inference builds the sequence step by step elsewhere).
        Returns (N, num_codebooks, d_model).
        """
        n = hidden.shape[0]
        seq = [hidden.unsqueeze(1)]  # position 0
        for k in range(self.num_codebooks - 1):
            seq.append(self.code_embeddings[k](codes[:, k]).unsqueeze(1))
        return torch.cat(seq, dim=1)  # (N, num_codebooks, d_model)

    def _run(self, seq):
        """Run the small transformer across the codebook axis."""
        length = seq.shape[1]
        cos, sin = self.rotary(length, seq.device)
        mask = causal_mask(length, seq.device)
        x = seq
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.norm(x)

    def forward(self, hidden, codes):
        """Teacher-forced training over all codebooks.

        hidden: (N, d_model), codes: (N, num_codebooks).
        Returns list of logits, one (N, codebook_size) tensor per codebook.
        """
        seq = self._sequence(hidden, codes)          # (N, NQ, d_model)
        out = self._run(seq)                          # (N, NQ, d_model)
        return [self.heads[k](out[:, k]) for k in range(self.num_codebooks)]

    @torch.no_grad()
    def generate(self, hidden, temperature=1.0):
        """Autoregressively sample all codebooks for a batch of frames.

        hidden: (N, d_model) -> codes (N, num_codebooks).
        Codebook 0 is also sampled here (the caller may instead supply it from
        the backbone; see qwen_tts.py, which passes the backbone's codebook-0
        choice and only uses MTP for codebooks >= 1).
        """
        n = hidden.shape[0]
        chosen = []
        # Position 0 input is the hidden state; predict codebook 0.
        step_input = hidden.unsqueeze(1)             # (N, 1, d_model)
        for k in range(self.num_codebooks):
            out = self._run(step_input)              # (N, k+1, d_model)
            logits = self.heads[k](out[:, -1]) / temperature
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)  # (N,)
            chosen.append(idx)
            if k < self.num_codebooks - 1:
                emb = self.code_embeddings[k](idx).unsqueeze(1)
                step_input = torch.cat([step_input, emb], dim=1)
        return torch.stack(chosen, dim=-1)           # (N, num_codebooks)

    @torch.no_grad()
    def generate_residuals(self, hidden, code0, temperature=1.0):
        """Given codebook-0 indices from the backbone, sample codebooks 1..NQ-1.

        hidden: (N, d_model); code0: (N,) -> codes (N, num_codebooks).
        """
        chosen = [code0]
        step_input = torch.cat(
            [hidden.unsqueeze(1), self.code_embeddings[0](code0).unsqueeze(1)], dim=1
        )
        for k in range(1, self.num_codebooks):
            out = self._run(step_input)
            logits = self.heads[k](out[:, -1]) / temperature
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
            chosen.append(idx)
            if k < self.num_codebooks - 1:
                emb = self.code_embeddings[k](idx).unsqueeze(1)
                step_input = torch.cat([step_input, emb], dim=1)
        return torch.stack(chosen, dim=-1)
