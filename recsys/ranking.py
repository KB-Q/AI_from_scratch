from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ranking_models import FeatureArtifacts, Ranker
from retrieval_models import PreparedData

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


def _rank_metrics_single_positive(ranked_items: Sequence[int], positive_item: int, k: int) -> Tuple[float, float, float]:
    k = int(k)
    top = list(ranked_items[:k])
    if int(positive_item) not in top:
        return 0.0, 0.0, 0.0
    rank = top.index(int(positive_item)) + 1
    hit = 1.0
    mrr = 1.0 / float(rank)
    ndcg = 1.0 / math.log2(float(rank) + 1.0)
    return hit, ndcg, mrr


def evaluate_ranking(
    ranked: Mapping[int, Sequence[int]],
    test_item_by_user: Mapping[int, int],
    k: int,
) -> Dict[str, float]:
    hits, ndcgs, mrrs = [], [], []
    for u, pos in test_item_by_user.items():
        hit, ndcg, mrr = _rank_metrics_single_positive(ranked.get(int(u), []), int(pos), k=k)
        hits.append(hit)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
    return {
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def _safe_reindex(series: pd.Series, index: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return series.reindex(index).fillna(fill).to_numpy()


def _build_user_event_history(ratings_all: pd.DataFrame, test_pos: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    test_ts = test_pos.set_index("user_id")["timestamp"].to_dict()
    out: Dict[int, pd.DataFrame] = {}
    for u, df in ratings_all.groupby("user_id"):
        t = test_ts.get(int(u))
        if t is None:
            continue
        hist = df[df["timestamp"] < t].sort_values("timestamp", kind="mergesort")
        out[int(u)] = hist
    return out


def build_feature_artifacts(
    data: PreparedData,
    positive_min_rating: int,
    recent_days: Sequence[int] = (1, 7, 14, 30, 60),
    recent_lastn: Sequence[int] = (3, 5, 10, 20),
    bad_rating_threshold: int = 2,
) -> FeatureArtifacts:
    item_ids = data.items["item_id"].to_numpy(dtype=np.int64)
    item_id_to_row = {int(it): i for i, it in enumerate(item_ids)}

    user_train_count = data.positives_train.groupby("user_id")["item_id"].size()
    user_mean_rating = data.ratings_all.groupby("user_id")["rating"].mean()

    genre_cols = [c for c in data.items.columns if c.startswith("genre_")]
    genre_names = [c.replace("genre_", "") for c in genre_cols]
    item_genres = data.items[genre_cols].astype(np.float32).to_numpy()

    histories = _build_user_event_history(data.ratings_all, data.positives_test)
    test_ts_by_user = data.positives_test.set_index("user_id")["timestamp"].to_dict()
    user_genre_pref: Dict[int, np.ndarray] = {}
    user_recent_by_days: Dict[int, Dict[int, np.ndarray]] = {}
    user_recent_by_lastn: Dict[int, Dict[int, np.ndarray]] = {}
    user_bad_by_days: Dict[int, Dict[int, np.ndarray]] = {}

    for u, hist in histories.items():
        hist_items = hist["item_id"].to_numpy(dtype=np.int64)
        hist_ratings = hist["rating"].to_numpy(dtype=np.int64)
        hist_ts = hist["timestamp"].to_numpy(dtype=np.int64)

        idx = np.asarray([item_id_to_row.get(int(it), -1) for it in hist_items], dtype=np.int64)
        valid = idx >= 0
        idx = idx[valid]
        hist_ratings = hist_ratings[valid]
        hist_ts = hist_ts[valid]

        if idx.size == 0:
            user_genre_pref[int(u)] = np.zeros(item_genres.shape[1], dtype=np.float32)
            user_recent_by_days[int(u)] = {int(d): np.zeros(item_genres.shape[1], dtype=np.float32) for d in recent_days}
            user_recent_by_lastn[int(u)] = {int(n): np.zeros(item_genres.shape[1], dtype=np.float32) for n in recent_lastn}
            user_bad_by_days[int(u)] = {int(d): np.zeros(item_genres.shape[1], dtype=np.float32) for d in recent_days}
            continue

        genres_seq = item_genres[idx]
        pos_mask = hist_ratings >= positive_min_rating
        if pos_mask.any():
            user_genre_pref[int(u)] = genres_seq[pos_mask].mean(axis=0).astype(np.float32)
        else:
            user_genre_pref[int(u)] = genres_seq.mean(axis=0).astype(np.float32)

        now = int(test_ts_by_user[int(u)])

        by_days: Dict[int, np.ndarray] = {}
        bad_by_days: Dict[int, np.ndarray] = {}
        for d in recent_days:
            window_start = now - int(d) * 24 * 3600
            m = hist_ts >= window_start
            by_days[int(d)] = genres_seq[m].sum(axis=0).astype(np.float32) if m.any() else np.zeros(item_genres.shape[1], dtype=np.float32)
            bad_m = m & (hist_ratings <= bad_rating_threshold)
            bad_by_days[int(d)] = (
                genres_seq[bad_m].sum(axis=0).astype(np.float32) if bad_m.any() else np.zeros(item_genres.shape[1], dtype=np.float32)
            )
        user_recent_by_days[int(u)] = by_days
        user_bad_by_days[int(u)] = bad_by_days

        by_lastn: Dict[int, np.ndarray] = {}
        for n in recent_lastn:
            last = genres_seq[-int(n) :] if genres_seq.shape[0] >= int(n) else genres_seq
            by_lastn[int(n)] = last.sum(axis=0).astype(np.float32) if last.size else np.zeros(item_genres.shape[1], dtype=np.float32)
        user_recent_by_lastn[int(u)] = by_lastn

    item_pop = _safe_reindex(data.item_popularity, item_ids, fill=0.0).astype(np.float32)
    item_mean = _safe_reindex(data.item_mean_rating, item_ids, fill=float(np.nanmean(data.ratings_all["rating"]))).astype(np.float32)
    item_n = _safe_reindex(data.item_num_ratings, item_ids, fill=0.0).astype(np.float32)

    return FeatureArtifacts(
        item_id_to_row=item_id_to_row,
        user_train_count=user_train_count,
        user_mean_rating=user_mean_rating,
        genre_names=genre_names,
        user_genre_pref=user_genre_pref,
        user_genre_recent_counts_by_days=user_recent_by_days,
        user_genre_recent_counts_by_lastn=user_recent_by_lastn,
        user_genre_bad_recent_counts_by_days=user_bad_by_days,
        item_pop=item_pop,
        item_mean_rating=item_mean,
        item_num_ratings=item_n,
    )


def build_ranking_dataset(
    data: PreparedData,
    retrieval_scores_wide: pd.DataFrame,
    users: Sequence[int],
    num_random_negatives: int,
    seed: int,
    feature_artifacts: FeatureArtifacts,
    recent_days: Sequence[int] = (1, 7, 14, 30, 60),
    recent_lastn: Sequence[int] = (3, 5, 10, 20),
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    items_all = data.items["item_id"].to_numpy(dtype=np.int64)
    test_item = data.positives_test.set_index("user_id")["item_id"].to_dict()

    base = retrieval_scores_wide[retrieval_scores_wide["user_id"].isin(users)].copy()
    base["label"] = 0
    base["group"] = base["user_id"]

    forced_rows = []
    for u in users:
        pos = test_item.get(int(u))
        if pos is None:
            continue
        forced_rows.append({"user_id": int(u), "item_id": int(pos), "label": 1, "group": int(u)})
    forced = pd.DataFrame(forced_rows)

    merged = pd.concat([base, forced], ignore_index=True, sort=False) if not forced.empty else base
    merged.drop_duplicates(["user_id", "item_id"], keep="first", inplace=True)

    negatives = []
    for u in users:
        u = int(u)
        interacted = data.user_train_item_set.get(u, set())
        pos = int(test_item.get(u, -1))
        avoid = set(interacted)
        if pos != -1:
            avoid.add(pos)
        eligible = np.setdiff1d(items_all, np.fromiter(avoid, dtype=np.int64), assume_unique=False)
        if eligible.size == 0:
            continue
        pick = min(int(num_random_negatives), int(eligible.size))
        chosen = rng.choice(eligible, size=pick, replace=False)
        for it in chosen:
            negatives.append({"user_id": u, "item_id": int(it), "label": 0, "group": u})
    negatives_df = pd.DataFrame(negatives)
    if not negatives_df.empty:
        merged = pd.concat([merged, negatives_df], ignore_index=True, sort=False)
        merged.drop_duplicates(["user_id", "item_id"], keep="first", inplace=True)

    merged = merged.merge(retrieval_scores_wide, on=["user_id", "item_id"], how="left")
    source_cols = [c for c in retrieval_scores_wide.columns if c not in {"user_id", "item_id"}]
    for c in source_cols:
        if c not in merged.columns:
            merged[c] = 0.0
    merged[source_cols] = merged[source_cols].fillna(0.0)

    item_id_to_row = feature_artifacts.item_id_to_row
    item_rows = merged["item_id"].map(item_id_to_row).to_numpy(dtype=np.int64)
    merged["item_popularity"] = feature_artifacts.item_pop[item_rows]
    merged["item_mean_rating"] = feature_artifacts.item_mean_rating[item_rows]
    merged["item_num_ratings"] = feature_artifacts.item_num_ratings[item_rows]

    merged["user_train_positives"] = merged["user_id"].map(feature_artifacts.user_train_count).fillna(0.0).astype(np.float32)
    merged["user_mean_rating"] = merged["user_id"].map(feature_artifacts.user_mean_rating).fillna(0.0).astype(np.float32)

    item_genres = data.item_genre_matrix
    user_pref = np.vstack(
        [
            feature_artifacts.user_genre_pref.get(int(u), np.zeros(item_genres.shape[1], dtype=np.float32))
            for u in merged["user_id"]
        ]
    )
    merged["user_item_genre_affinity"] = np.sum(user_pref * item_genres[item_rows], axis=1).astype(np.float32)

    for d in recent_days:
        recent = np.vstack(
            [
                feature_artifacts.user_genre_recent_counts_by_days.get(int(u), {}).get(int(d), np.zeros(item_genres.shape[1], dtype=np.float32))
                for u in merged["user_id"]
            ]
        )
        merged[f"user_recent_genre_count_{d}d"] = np.sum(recent * item_genres[item_rows], axis=1).astype(np.float32)
        bad_recent = np.vstack(
            [
                feature_artifacts.user_genre_bad_recent_counts_by_days.get(int(u), {}).get(int(d), np.zeros(item_genres.shape[1], dtype=np.float32))
                for u in merged["user_id"]
            ]
        )
        merged[f"user_bad_recent_genre_count_{d}d"] = np.sum(bad_recent * item_genres[item_rows], axis=1).astype(np.float32)

    for n in recent_lastn:
        recent = np.vstack(
            [
                feature_artifacts.user_genre_recent_counts_by_lastn.get(int(u), {}).get(int(n), np.zeros(item_genres.shape[1], dtype=np.float32))
                for u in merged["user_id"]
            ]
        )
        merged[f"user_recent_genre_count_last{n}"] = np.sum(recent * item_genres[item_rows], axis=1).astype(np.float32)

    feature_cols = (
        source_cols
        + [
            "item_popularity",
            "item_mean_rating",
            "item_num_ratings",
            "user_train_positives",
            "user_mean_rating",
            "user_item_genre_affinity",
        ]
        + [f"user_recent_genre_count_{d}d" for d in recent_days]
        + [f"user_bad_recent_genre_count_{d}d" for d in recent_days]
        + [f"user_recent_genre_count_last{n}" for n in recent_lastn]
    )

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=np.int32)
    group = merged["group"].to_numpy(dtype=np.int64)
    return merged, X, y, group, list(feature_cols)


def rank_candidates_for_users(
    ranker: Ranker,
    candidates_df: pd.DataFrame,
    feature_cols: Sequence[str],
    top_k: int,
) -> Dict[int, List[int]]:
    if candidates_df.empty:
        return {}
    X = candidates_df[list(feature_cols)].to_numpy(dtype=np.float32)
    scores = ranker.predict(X)
    tmp = candidates_df[["user_id", "item_id"]].copy()
    tmp["score"] = scores
    ranked: Dict[int, List[int]] = {}
    for u, df in tmp.groupby("user_id"):
        s = df.sort_values("score", ascending=False, kind="mergesort")
        ranked[int(u)] = s["item_id"].head(int(top_k)).astype(int).tolist()
    return ranked


def build_user_context_text(data: PreparedData, recent_n: int = 10) -> Dict[int, str]:
    item_id_to_text = dict(zip(data.items["item_id"].to_numpy(dtype=np.int64), data.items["item_text"].to_numpy()))
    out: Dict[int, str] = {}
    for u, hist in data.user_train_items.items():
        recent = hist[-int(recent_n) :]
        out[int(u)] = " ".join(item_id_to_text.get(int(it), "") for it in recent).strip()
    return out


def score_pairs_cross_encoder(
    left_text: Sequence[str],
    right_text: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ImportError("torch + transformers are required for cross-encoder ranking")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    scores: List[np.ndarray] = []
    left = list(map(str, left_text))
    right = list(map(str, right_text))
    with torch.no_grad():
        for i in range(0, len(left), int(batch_size)):
            l = left[i : i + int(batch_size)]
            r = right[i : i + int(batch_size)]
            enc = tok(l, r, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            s = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, 0]
            scores.append(s.detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32) if scores else np.asarray([], dtype=np.float32)


def rank_candidates_with_cross_encoder(
    candidates_df: pd.DataFrame,
    data: PreparedData,
    top_k: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "cpu",
    batch_size: int = 64,
    max_length: int = 128,
    user_context_recent_n: int = 10,
) -> Dict[int, List[int]]:
    if candidates_df.empty:
        return {}

    df = candidates_df[["user_id", "item_id"]].copy()
    user_ctx = build_user_context_text(data, recent_n=int(user_context_recent_n))
    item_id_to_text = dict(zip(data.items["item_id"].to_numpy(dtype=np.int64), data.items["item_text"].to_numpy()))
    df["left_text"] = df["user_id"].map(user_ctx).fillna("").astype(str)
    df["right_text"] = df["item_id"].map(item_id_to_text).fillna("").astype(str)
    df["score"] = score_pairs_cross_encoder(
        df["left_text"].tolist(),
        df["right_text"].tolist(),
        model_name=str(model_name),
        device=str(device),
        batch_size=int(batch_size),
        max_length=int(max_length),
    )

    ranked: Dict[int, List[int]] = {}
    for u, g in df.groupby("user_id"):
        s = g.sort_values("score", ascending=False, kind="mergesort")
        ranked[int(u)] = s["item_id"].head(int(top_k)).astype(int).tolist()
    return ranked


