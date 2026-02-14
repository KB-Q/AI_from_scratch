"""
Tiny, fully-numeric demonstrations of LoRA and common PEFT variants.

Run:
  python transformers/lora_examples.py
"""

from __future__ import annotations

import numpy as np


def _print_matrix(name: str, x: np.ndarray) -> None:
    print(f"{name} (shape={x.shape}):")
    print(x)
    print()


def lora_delta(B: np.ndarray, A: np.ndarray, alpha: float, r: int, use_sqrt_scaling: bool = False) -> np.ndarray:
    """
    LoRA update:
      ΔW = s * (B @ A)
    where:
      s = alpha/r  (standard)
      s = alpha/sqrt(r) (rsLoRA-style alternative scaling)
    """
    if use_sqrt_scaling:
        scale = alpha / np.sqrt(r)
    else:
        scale = alpha / r
    return scale * (B @ A)


def demo_lora_and_rslora() -> None:
    print("=== LoRA vs rsLoRA scaling (toy example) ===\n")

    # We'll use y = x @ W (row-vector convention), with:
    # x: (1, d_in), W: (d_in, d_out)
    W0 = np.array([[1.0, -1.0],
                   [0.5,  2.0],
                   [-1.5, 0.0]])  # (3, 2)
    x = np.array([[2.0, -1.0, 0.5]])  # (1, 3)

    # Rank-1 LoRA: A: (r, d_in) = (1, 3), B: (d_out, r) = (2, 1)
    r = 1
    alpha = 2.0
    A = np.array([[0.2, -0.1, 0.4]])      # (1, 3)
    B = np.array([[1.0],
                  [-0.5]])                # (2, 1)

    dW_lora = lora_delta(B=B, A=A, alpha=alpha, r=r, use_sqrt_scaling=False)  # (2,3) in this convention
    # Our W is (d_in, d_out). B@A is (d_out, d_in), so transpose to match W shape.
    dW_lora = dW_lora.T  # (3,2)

    dW_rslora = lora_delta(B=B, A=A, alpha=alpha, r=r, use_sqrt_scaling=True).T  # (3,2)

    W_lora = W0 + dW_lora
    W_rslora = W0 + dW_rslora

    _print_matrix("W0", W0)
    _print_matrix("x", x)
    _print_matrix("ΔW (LoRA, alpha/r)", dW_lora)
    _print_matrix("W_eff (LoRA)", W_lora)
    _print_matrix("y (LoRA) = x @ W_eff", x @ W_lora)

    _print_matrix("ΔW (rsLoRA-style, alpha/sqrt(r))", dW_rslora)
    _print_matrix("W_eff (rsLoRA-style)", W_rslora)
    _print_matrix("y (rsLoRA-style) = x @ W_eff", x @ W_rslora)


def quantize_symmetric(W: np.ndarray, bits: int = 4) -> tuple[np.ndarray, float]:
    """
    Very simple symmetric per-tensor quantization:
      Q = 2^(bits-1)-1
      delta = max(|W|)/Q
      W_q = clip(round(W/delta), -Q, Q)   (stored as int)
      W_hat = delta * W_q
    Returns (W_hat, delta).
    """
    Q = 2 ** (bits - 1) - 1
    max_abs = float(np.max(np.abs(W)))
    if max_abs == 0.0:
        delta = 1.0
    else:
        delta = max_abs / Q
    W_q = np.clip(np.round(W / delta), -Q, Q).astype(np.int32)
    W_hat = (delta * W_q).astype(np.float64)
    return W_hat, delta


def demo_qlora_style() -> None:
    print("=== QLoRA-style idea (toy quantization + LoRA) ===\n")

    W0 = np.array([[1.25, -0.75],
                   [0.10,  2.30],
                   [-1.10, 0.20]])  # (3,2)
    x = np.array([[1.0, -2.0, 0.5]])  # (1,3)

    # Quantize base (toy) and dequantize to W_hat
    W_hat, delta = quantize_symmetric(W0, bits=4)

    # Add a tiny LoRA update (still in full precision in practice)
    r = 1
    alpha = 1.0
    A = np.array([[0.3, 0.0, -0.2]])      # (1,3)
    B = np.array([[0.2],
                  [0.1]])                 # (2,1)
    dW = lora_delta(B=B, A=A, alpha=alpha, r=r).T  # (3,2)

    y_full = x @ (W0 + dW)
    y_quant = x @ (W_hat + dW)

    _print_matrix("W0 (full precision)", W0)
    _print_matrix("W_hat (dequantized 4-bit toy)", W_hat)
    print(f"quant step delta = {delta}\n")
    _print_matrix("ΔW (LoRA)", dW)
    _print_matrix("y_full = x @ (W0 + ΔW)", y_full)
    _print_matrix("y_quant = x @ (W_hat + ΔW)", y_quant)
    _print_matrix("difference (y_full - y_quant)", y_full - y_quant)


def demo_adalora_budget_allocation() -> None:
    print("=== AdaLoRA-style intuition: allocate rank under a budget ===\n")

    # Two layers with different "importance" (toy proxy: ||ΔW||_F).
    # We'll pretend we can pick ranks r1 and r2, with r1 + r2 <= R_budget.
    R_budget = 3

    # Toy "candidate updates" as if produced by training (higher norm => more impact).
    dW_layer1 = np.array([[0.8, -0.2],
                          [0.1,  0.0]])
    dW_layer2 = np.array([[0.05, -0.02],
                          [0.01,  0.00]])

    imp1 = float(np.linalg.norm(dW_layer1, ord="fro"))
    imp2 = float(np.linalg.norm(dW_layer2, ord="fro"))

    print(f"budget R_budget = {R_budget}")
    print(f"importance(layer1) = ||ΔW1||_F = {imp1:.4f}")
    print(f"importance(layer2) = ||ΔW2||_F = {imp2:.4f}\n")

    # Simplified greedy allocation:
    # give 1 rank to each, then allocate remaining ranks to the most important layer
    r1, r2 = 1, 1
    remaining = R_budget - (r1 + r2)
    if remaining > 0:
        if imp1 >= imp2:
            r1 += remaining
        else:
            r2 += remaining

    print(f"allocated ranks (toy): r1={r1}, r2={r2} (r1+r2={r1+r2})\n")
    print("In real AdaLoRA, rank is adjusted during training using more granular importance signals\n"
          "and pruning schedules; this is just the core budgeting intuition.\n")


def dora_rowwise(W0: np.ndarray, dV: np.ndarray, m: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    """
    DoRA-style row-wise magnitude+direction update.

    Treat each output channel as one row (row-wise decomposition):
      m_i = ||W0_i||2
      V_i = W0_i / m_i
      W_i = m_i * (V_i + dV_i) / ||V_i + dV_i||2

    Inputs:
      W0: (d_out, d_in) if you use row-vector outputs (common in many writeups)
      dV: (d_out, d_in) low-rank direction update
      m: optional explicit magnitude vector (d_out,)
    """
    if W0.shape != dV.shape:
        raise ValueError("W0 and dV must have the same shape")

    row_norm = np.linalg.norm(W0, axis=1, keepdims=True) + eps
    V = W0 / row_norm

    if m is None:
        m = row_norm[:, 0]  # (d_out,)
    else:
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (W0.shape[0],):
            raise ValueError("m must have shape (d_out,)")

    Vp = V + dV
    Vp_norm = np.linalg.norm(Vp, axis=1, keepdims=True) + eps
    Vp_unit = Vp / Vp_norm

    return (m[:, None] * Vp_unit).astype(np.float64)


def demo_dora() -> None:
    print("=== DoRA (toy row-wise magnitude+direction update) ===\n")

    # We'll use y = W @ x (column-vector convention) for this demo to match row-wise output channels.
    # W0: (d_out, d_in), x: (d_in,)
    W0 = np.array([[2.0, 0.0],
                   [0.0, 1.0]])  # (2,2)
    x = np.array([1.0, 3.0])     # (2,)

    # Build a low-rank dV using LoRA factors (rank-1):
    r = 1
    alpha = 1.0
    A = np.array([[0.25, -0.10]])   # (1, d_in=2)
    B = np.array([[0.20],
                  [-0.40]])         # (d_out=2, 1)
    dV = lora_delta(B=B, A=A, alpha=alpha, r=r)  # (d_out, d_in) == (2,2)

    W_lora_like = W0 + dV
    W_dora = dora_rowwise(W0=W0, dV=dV)

    _print_matrix("W0", W0)
    _print_matrix("x", x.reshape(-1, 1))
    _print_matrix("dV (low-rank direction update)", dV)
    _print_matrix("W (plain add: W0 + dV)", W_lora_like)
    _print_matrix("W (DoRA row-wise normalized)", W_dora)

    y_plain = W_lora_like @ x
    y_dora = W_dora @ x
    _print_matrix("y_plain = (W0 + dV) @ x", y_plain.reshape(-1, 1))
    _print_matrix("y_dora = W_dora @ x", y_dora.reshape(-1, 1))


def demo_ia3() -> None:
    print("=== IA³ (toy multiplicative scaling) ===\n")

    # y = x @ W (row-vector convention)
    W0 = np.array([[1.0, 0.0],
                   [0.0, 2.0],
                   [1.0, -1.0]])  # (3,2)
    x = np.array([[2.0, -1.0, 0.5]])  # (1,3)

    s_in = np.array([[1.0, 0.5, 2.0]])     # input scaling (1,3)
    s_out = np.array([[2.0, 0.25]])        # output scaling (1,2)

    y_base = x @ W0
    y_in_scaled = (x * s_in) @ W0
    y_out_scaled = (x @ W0) * s_out

    _print_matrix("W0", W0)
    _print_matrix("x", x)
    _print_matrix("y_base = x @ W0", y_base)
    _print_matrix("s_in", s_in)
    _print_matrix("y_in_scaled = (x ⊙ s_in) @ W0", y_in_scaled)
    _print_matrix("s_out", s_out)
    _print_matrix("y_out_scaled = (x @ W0) ⊙ s_out", y_out_scaled)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    demo_lora_and_rslora()
    demo_qlora_style()
    demo_adalora_budget_allocation()
    demo_dora()
    demo_ia3()


if __name__ == "__main__":
    main()


