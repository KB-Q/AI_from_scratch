"""
Compare all retriever and ranker configurations.

Usage:
    python3 compare_models.py --data-dir data
    python3 compare_models.py --data-dir data --include-neural  # includes text-emb and two-tower retrievers
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RANKERS = [
    "heuristic",
    "pointwise_gbdt",
    "pairwise_logreg",
    "xgboost_lambdarank",
    "lightgbm_lambdarank",
    # "cross_encoder",  # slow, uncomment if desired
]


def parse_metrics(output: str) -> Dict[str, float]:
    """Extract metrics from pipeline output."""
    metrics = {}
    for line in output.strip().split("\n"):
        if "retrieval_recall" in line and "eval_users" in line:
            val = float(line.split(":")[-1].strip())
            metrics["retrieval_recall"] = val
        elif "hit_rate@" in line:
            val = float(line.split(":")[-1].strip())
            metrics["hit_rate"] = val
        elif "ndcg@" in line:
            val = float(line.split(":")[-1].strip())
            metrics["ndcg"] = val
        elif "mrr@" in line:
            val = float(line.split(":")[-1].strip())
            metrics["mrr"] = val
    return metrics


def run_pipeline(
    ranker: str,
    data_dir: str,
    enable_text_emb: bool = False,
    enable_two_tower: bool = False,
    retrieval_k: int = 400,
    rank_k: int = 20,
    seed: int = 7,
) -> Tuple[Dict[str, float], bool]:
    """Run pipeline with given config and return metrics."""
    cmd = [
        sys.executable, "pipeline.py",
        "--data-dir", data_dir,
        "--ranker", ranker,
        "--retrieval-k", str(retrieval_k),
        "--rank-k", str(rank_k),
        "--seed", str(seed),
    ]
    if enable_text_emb:
        cmd.append("--enable-text-emb-retriever")
    if enable_two_tower:
        cmd.append("--enable-two-tower-retriever")

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  [FAILED] {result.stderr[:200]}")
            return {}, False
        return parse_metrics(result.stdout), True
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT]")
        return {}, False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {}, False


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare ranker and retriever configurations")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--include-neural", action="store_true", help="Include text-emb and two-tower retrievers")
    p.add_argument("--include-cross-encoder", action="store_true", help="Include cross-encoder ranker (slow)")
    p.add_argument("--retrieval-k", type=int, default=400)
    p.add_argument("--rank-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    rankers = RANKERS.copy()
    if args.include_cross_encoder:
        rankers.append("cross_encoder")

    retriever_configs = [
        ("baseline", False, False),
    ]
    if args.include_neural:
        retriever_configs.extend([
            ("+text_emb", True, False),
            ("+two_tower", False, True),
            ("+both_neural", True, True),
        ])

    results: List[Dict] = []

    print("=" * 70)
    print("Running comparison experiments...")
    print("=" * 70)

    for retriever_name, text_emb, two_tower in retriever_configs:
        print(f"\n[Retriever config: {retriever_name}]")
        for ranker in rankers:
            print(f"  Running {ranker}...", end=" ", flush=True)
            metrics, success = run_pipeline(
                ranker=ranker,
                data_dir=args.data_dir,
                enable_text_emb=text_emb,
                enable_two_tower=two_tower,
                retrieval_k=args.retrieval_k,
                rank_k=args.rank_k,
                seed=args.seed,
            )
            if success:
                print(f"hit={metrics.get('hit_rate', 0):.4f} ndcg={metrics.get('ndcg', 0):.4f} mrr={metrics.get('mrr', 0):.4f}")
                results.append({
                    "retriever": retriever_name,
                    "ranker": ranker,
                    **metrics,
                })

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Retriever':<15} {'Ranker':<22} {'Recall':<10} {'Hit@k':<10} {'NDCG@k':<10} {'MRR@k':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['retriever']:<15} {r['ranker']:<22} "
            f"{r.get('retrieval_recall', 0):<10.4f} {r.get('hit_rate', 0):<10.4f} "
            f"{r.get('ndcg', 0):<10.4f} {r.get('mrr', 0):<10.4f}"
        )

    if results:
        best = max(results, key=lambda x: x.get("ndcg", 0))
        print("-" * 70)
        print(f"Best by NDCG: {best['retriever']} + {best['ranker']} (ndcg={best.get('ndcg', 0):.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
