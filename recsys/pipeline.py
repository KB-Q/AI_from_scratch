"""
Recommendation Retrieval → Ranking demo on MovieLens-100k.

Files:
- retrieval.py: dataset loading/prep + retrieval metrics/utilities.
- retrieval_models.py: candidate generators (popularity, co-visit, TF-IDF, BM25, SVD, dense text embeddings + ANN, two-tower + ANN, hybrid).
- ranking.py: feature builder + ranking metrics + cross-encoder ranking option.
- ranking_models.py: trainable rankers (heuristic, pointwise GBDT, pairwise logreg, XGBoost/LightGBM LambdaRank).

Run:
python3 recsys/pipeline.py --data-dir data --retrieval-k 400 --rank-k 20 --ranker xgboost_lambdarank
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ranking import (
    build_feature_artifacts, 
    build_ranking_dataset, 
    evaluate_ranking, 
    rank_candidates_for_users,
    rank_candidates_with_cross_encoder,
)
from ranking_models import get_ranker

from retrieval import (
    download_movielens_100k,
    load_movielens_100k,
    pivot_retrieval_scores,
    prepare_data,
    recall_at_k,
    set_seed,
    split_users,
)
from retrieval_models import (
    BM25Retriever,
    CoVisitRetriever,
    HybridRetriever,
    ItemKNNCosineRetriever,
    PopularityRetriever,
    RandomRetriever,
    SVDRetriever,
    TfidfCosineRetriever,
    TextEmbeddingANNRetriever,
    TwoTowerANNRetriever,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(Path("data").resolve()), help="Directory to download/cache datasets")
    p.add_argument("--min-positive-rating", type=int, default=4)
    p.add_argument("--min-user-positives", type=int, default=5)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--retrieval-k", type=int, default=400)
    p.add_argument("--rank-k", type=int, default=20)
    p.add_argument("--train-negatives", type=int, default=200)
    p.add_argument("--enable-text-emb-retriever", action="store_true")
    p.add_argument("--text-emb-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--text-emb-device", type=str, default="cpu")
    p.add_argument("--text-emb-index", type=str, default="faiss_hnsw", choices=["faiss_hnsw", "faiss_flat"])
    p.add_argument("--enable-two-tower-retriever", action="store_true")
    p.add_argument("--two-tower-device", type=str, default="cpu")
    p.add_argument("--two-tower-dim", type=int, default=64)
    p.add_argument("--two-tower-epochs", type=int, default=2)
    p.add_argument("--two-tower-lr", type=float, default=1e-2)
    p.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--cross-encoder-device", type=str, default="cpu")
    p.add_argument("--cross-encoder-batch-size", type=int, default=64)
    p.add_argument("--cross-encoder-max-length", type=int, default=128)
    p.add_argument("--cross-encoder-context-n", type=int, default=10)
    p.add_argument(
        "--ranker",
        type=str,
        default="xgboost_lambdarank",
        choices=[
            "heuristic",
            "pointwise_gbdt",
            "pairwise_logreg",
            "xgboost_lambdarank",
            "lightgbm_lambdarank",
            "cross_encoder",
        ],
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    ml_dir = download_movielens_100k(data_dir)
    ratings, items = load_movielens_100k(ml_dir)
    data = prepare_data(
        ratings_all=ratings,
        items=items,
        min_positive_rating=int(args.min_positive_rating),
        min_user_positives=int(args.min_user_positives),
    )

    users = sorted(data.positives_test["user_id"].unique().tolist())
    train_users, eval_users = split_users(users, eval_fraction=float(args.eval_fraction), seed=int(args.seed))

    retrievers = [
        PopularityRetriever(),
        CoVisitRetriever(recent_n=10, max_neighbors=200),
        ItemKNNCosineRetriever(max_neighbors=200),
        TfidfCosineRetriever(profile_recent_n=30),
        BM25Retriever(profile_recent_n=30),
        SVDRetriever(n_components=64, random_state=args.seed),
        RandomRetriever(seed=args.seed),
    ]
    if bool(args.enable_text_emb_retriever):
        retrievers.append(
            TextEmbeddingANNRetriever(
                model_name=str(args.text_emb_model),
                device=str(args.text_emb_device),
                index_type=str(args.text_emb_index),
            )
        )
    if bool(args.enable_two_tower_retriever):
        retrievers.append(
            TwoTowerANNRetriever(
                device=str(args.two_tower_device),
                embedding_dim=int(args.two_tower_dim),
                epochs=int(args.two_tower_epochs),
                lr=float(args.two_tower_lr),
                seed=int(args.seed),
            )
        )
    per_k = {
        "popularity": max(100, int(args.retrieval_k // 4)),
        "covisit": max(150, int(args.retrieval_k // 3)),
        "itemknn": max(150, int(args.retrieval_k // 3)),
        "tfidf": max(150, int(args.retrieval_k // 3)),
        "bm25": max(150, int(args.retrieval_k // 3)),
        "svd": max(150, int(args.retrieval_k // 3)),
        "random": max(50, int(args.retrieval_k // 6)),
        "text_emb_ann": max(150, int(args.retrieval_k // 3)),
        "two_tower_ann": max(150, int(args.retrieval_k // 3)),
    }
    hybrid = HybridRetriever(retrievers=retrievers, per_retriever_k=per_k).fit(data)

    retrieval_df_train = hybrid.retrieve_batch(data, train_users, k=int(args.retrieval_k))
    retrieval_df_eval = hybrid.retrieve_batch(data, eval_users, k=int(args.retrieval_k))

    retrieval_wide_train = pivot_retrieval_scores(retrieval_df_train)
    retrieval_wide_eval = pivot_retrieval_scores(retrieval_df_eval)

    test_item_by_user = data.positives_test.set_index("user_id")["item_id"].to_dict()
    candidates_train = retrieval_wide_train.groupby("user_id")["item_id"].apply(list).to_dict()
    candidates_eval = retrieval_wide_eval.groupby("user_id")["item_id"].apply(list).to_dict()
    retrieval_recall_train = recall_at_k(candidates_train, {int(u): int(test_item_by_user[int(u)]) for u in train_users})
    retrieval_recall_eval = recall_at_k(candidates_eval, {int(u): int(test_item_by_user[int(u)]) for u in eval_users})

    if str(args.ranker) == "cross_encoder":
        eval_pairs = retrieval_wide_eval[["user_id", "item_id"]].copy()
        forced = [{"user_id": int(u), "item_id": int(test_item_by_user[int(u)])} for u in eval_users]
        eval_pairs = pd.concat([eval_pairs, pd.DataFrame(forced)], ignore_index=True).drop_duplicates(
            ["user_id", "item_id"], keep="first"
        )
        ranked = rank_candidates_with_cross_encoder(
            eval_pairs,
            data,
            top_k=int(args.rank_k),
            model_name=str(args.cross_encoder_model),
            device=str(args.cross_encoder_device),
            batch_size=int(args.cross_encoder_batch_size),
            max_length=int(args.cross_encoder_max_length),
            user_context_recent_n=int(args.cross_encoder_context_n),
        )
        metrics = evaluate_ranking(
            ranked,
            {int(u): int(test_item_by_user[int(u)]) for u in eval_users},
            k=int(args.rank_k),
        )
        print(
            "\n".join(
                [
                    "=== Recommendation Retrieval→Ranking Demo (MovieLens-100k) ===",
                    f"users: {len(users)} | train_users: {len(train_users)} | eval_users: {len(eval_users)}",
                    f"retrieval_k: {int(args.retrieval_k)} | rank_k: {int(args.rank_k)} | ranker: cross_encoder",
                    f"retrieval_recall@{int(args.retrieval_k)} (train_users): {retrieval_recall_train:.4f}",
                    f"retrieval_recall@{int(args.retrieval_k)} (eval_users):  {retrieval_recall_eval:.4f}",
                    f"hit_rate@{int(args.rank_k)}: {metrics['hit_rate']:.4f}",
                    f"ndcg@{int(args.rank_k)}:     {metrics['ndcg']:.4f}",
                    f"mrr@{int(args.rank_k)}:      {metrics['mrr']:.4f}",
                ]
            )
        )
        return 0

    feat_art = build_feature_artifacts(
        data,
        positive_min_rating=int(args.min_positive_rating),
        recent_days=(1, 7, 14, 30, 60),
        recent_lastn=(3, 5, 10, 20),
    )

    train_df, X_train, y_train, g_train, feature_cols = build_ranking_dataset(
        data=data,
        retrieval_scores_wide=retrieval_wide_train,
        users=train_users,
        num_random_negatives=int(args.train_negatives),
        seed=int(args.seed),
        feature_artifacts=feat_art,
    )
    eval_df, X_eval, y_eval, g_eval, _ = build_ranking_dataset(
        data=data,
        retrieval_scores_wide=retrieval_wide_eval,
        users=eval_users,
        num_random_negatives=0,
        seed=int(args.seed),
        feature_artifacts=feat_art,
    )
    ranker = get_ranker(str(args.ranker), seed=int(args.seed)).fit(X_train, y_train, g_train)

    eval_df_scored = eval_df.copy()
    eval_df_scored[feature_cols] = eval_df_scored[feature_cols].fillna(0.0)
    ranked = rank_candidates_for_users(ranker, eval_df_scored, feature_cols=feature_cols, top_k=int(args.rank_k))
    metrics = evaluate_ranking(
        ranked,
        {int(u): int(test_item_by_user[int(u)]) for u in eval_users},
        k=int(args.rank_k),
    )

    print(
        "\n".join(
            [
                "=== Recommendation Retrieval→Ranking Demo (MovieLens-100k) ===",
                f"users: {len(users)} | train_users: {len(train_users)} | eval_users: {len(eval_users)}",
                f"retrieval_k: {int(args.retrieval_k)} | rank_k: {int(args.rank_k)} | ranker: {ranker.name}",
                f"retrieval_recall@{int(args.retrieval_k)} (train_users): {retrieval_recall_train:.4f}",
                f"retrieval_recall@{int(args.retrieval_k)} (eval_users):  {retrieval_recall_eval:.4f}",
                f"hit_rate@{int(args.rank_k)}: {metrics['hit_rate']:.4f}",
                f"ndcg@{int(args.rank_k)}:     {metrics['ndcg']:.4f}",
                f"mrr@{int(args.rank_k)}:      {metrics['mrr']:.4f}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


