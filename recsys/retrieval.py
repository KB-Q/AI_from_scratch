from __future__ import annotations

import os
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from retrieval_models import IdMaps, PreparedData


ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def download_movielens_100k(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    extract_dir = data_dir / "ml-100k"
    if extract_dir.exists() and (extract_dir / "u.data").exists():
        return extract_dir

    import urllib.request

    tmp_zip = zip_path.with_suffix(".zip.tmp")
    with urllib.request.urlopen(ML_100K_URL) as resp, open(tmp_zip, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp_zip, zip_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    if not (extract_dir / "u.data").exists():
        raise RuntimeError("MovieLens-100k download/extract failed: missing u.data")
    return extract_dir


def _parse_year_from_title(title: str) -> Optional[int]:
    m = re.search(r"\((\d{4})\)\s*$", title)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_movielens_100k(ml100k_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = ml100k_dir / "u.data"
    items_path = ml100k_dir / "u.item"
    if not ratings_path.exists() or not items_path.exists():
        raise FileNotFoundError("Expected MovieLens-100k files u.data and u.item")

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    genre_names = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    item_cols = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ] + [f"genre_{g}" for g in genre_names]

    items = pd.read_csv(
        items_path,
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="python",
    )

    genre_flag_cols = [f"genre_{g}" for g in genre_names]
    items["year"] = items["title"].map(_parse_year_from_title).astype("Int64")
    items["genres"] = items[genre_flag_cols].apply(
        lambda r: [g.replace("genre_", "") for g, v in r.items() if int(v) == 1],
        axis=1,
    )
    items["item_text"] = (
        items["title"].fillna("").astype(str) + " " + items["genres"].map(lambda xs: " ".join(xs)).fillna("").astype(str)
    ).str.strip()

    return ratings, items[["item_id", "title", "year", "genres", "item_text"] + genre_flag_cols]


def make_implicit_positives(ratings: pd.DataFrame, min_rating: int = 4) -> pd.DataFrame:
    positives = ratings.loc[ratings["rating"] >= min_rating, ["user_id", "item_id", "rating", "timestamp"]].copy()
    positives.sort_values(["user_id", "timestamp"], inplace=True, kind="mergesort")
    return positives


def leave_last_out(positives: pd.DataFrame, min_user_positives: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_idx = positives.groupby("user_id")["timestamp"].idxmax()
    test = positives.loc[last_idx].copy()
    train = positives.drop(index=last_idx).copy()

    train_counts = train.groupby("user_id")["item_id"].size()
    keep_users = train_counts[train_counts >= (min_user_positives - 1)].index
    train = train[train["user_id"].isin(keep_users)].copy()
    test = test[test["user_id"].isin(keep_users)].copy()
    train.sort_values(["user_id", "timestamp"], inplace=True, kind="mergesort")
    test.sort_values(["user_id", "timestamp"], inplace=True, kind="mergesort")
    return train, test


def split_users(user_ids: Sequence[int], eval_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    user_ids = np.asarray(list(user_ids), dtype=np.int64)
    rng.shuffle(user_ids)
    cut = int(round(len(user_ids) * (1.0 - eval_fraction)))
    cut = max(1, min(cut, len(user_ids) - 1))
    return user_ids[:cut], user_ids[cut:]


def prepare_data(
    ratings_all: pd.DataFrame,
    items: pd.DataFrame,
    min_positive_rating: int,
    min_user_positives: int,
) -> PreparedData:
    positives = make_implicit_positives(ratings_all, min_rating=min_positive_rating)
    train_pos, test_pos = leave_last_out(positives, min_user_positives=min_user_positives)
    user_ids = train_pos["user_id"].unique()
    item_ids = items["item_id"].unique()
    id_maps = IdMaps.from_ids(user_ids=user_ids, item_ids=item_ids)

    user_train_items = train_pos.groupby("user_id")["item_id"].apply(lambda s: np.asarray(s, dtype=np.int64)).to_dict()
    user_train_item_set = {int(u): set(map(int, xs)) for u, xs in user_train_items.items()}

    item_popularity = train_pos.groupby("item_id")["user_id"].size()
    item_stats = ratings_all.groupby("item_id")["rating"].agg(["mean", "count"])
    item_mean_rating = item_stats["mean"]
    item_num_ratings = item_stats["count"]

    genre_flag_cols = [c for c in items.columns if c.startswith("genre_")]
    items_sorted = items.sort_values("item_id", kind="mergesort").reset_index(drop=True)
    item_genre_matrix = items_sorted[genre_flag_cols].astype(np.float32).to_numpy()
    item_texts = items_sorted["item_text"].fillna("").astype(str).to_numpy()

    return PreparedData(
        ratings_all=ratings_all,
        items=items_sorted,
        positives_train=train_pos,
        positives_test=test_pos,
        id_maps=id_maps,
        user_train_items=user_train_items,
        user_train_item_set=user_train_item_set,
        item_popularity=item_popularity,
        item_mean_rating=item_mean_rating,
        item_num_ratings=item_num_ratings,
        item_genre_matrix=item_genre_matrix,
        item_texts=item_texts,
    )


def pivot_retrieval_scores(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    if retrieval_df.empty:
        return pd.DataFrame(columns=["user_id", "item_id"])
    wide = retrieval_df.pivot_table(
        index=["user_id", "item_id"],
        columns="source",
        values="score",
        aggfunc="max",
        fill_value=0.0,
    ).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def recall_at_k(candidates: Mapping[int, Sequence[int]], test_item_by_user: Mapping[int, int]) -> float:
    hits = []
    for u, test_item in test_item_by_user.items():
        cand = candidates.get(int(u), [])
        hits.append(1.0 if int(test_item) in set(map(int, cand)) else 0.0)
    return float(np.mean(hits)) if hits else 0.0


