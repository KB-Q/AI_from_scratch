from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureArtifacts:
    item_id_to_row: Dict[int, int]
    user_train_count: pd.Series
    user_mean_rating: pd.Series
    genre_names: List[str]
    user_genre_pref: Dict[int, np.ndarray]
    user_genre_recent_counts_by_days: Dict[int, Dict[int, np.ndarray]]
    user_genre_recent_counts_by_lastn: Dict[int, Dict[int, np.ndarray]]
    user_genre_bad_recent_counts_by_days: Dict[int, Dict[int, np.ndarray]]
    item_pop: np.ndarray
    item_mean_rating: np.ndarray
    item_num_ratings: np.ndarray


class Ranker:
    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "Ranker":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HeuristicRanker(Ranker):
    name = "heuristic"

    def __init__(self, weights: Optional[Mapping[str, float]] = None) -> None:
        self.weights = dict(weights or {})

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "HeuristicRanker":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.weights:
            return X.sum(axis=1)
        w = np.asarray(list(self.weights.values()), dtype=np.float32)
        return X[:, : w.size] @ w


class PointwiseGBDTRanker(Ranker):
    name = "pointwise_gbdt"

    def __init__(self, random_state: int = 7) -> None:
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "PointwiseGBDTRanker":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=400,
            random_state=self.random_state,
        )
        clf.fit(X, y)
        self._clf = clf
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1].astype(np.float32)


class PairwiseLogRegRanker(Ranker):
    name = "pairwise_logreg"

    def __init__(self, random_state: int = 7, max_pairs_per_group: int = 40) -> None:
        self.random_state = int(random_state)
        self.max_pairs_per_group = int(max_pairs_per_group)

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "PairwiseLogRegRanker":
        rng = np.random.default_rng(self.random_state)
        pairs_X = []
        pairs_y = []
        for g in np.unique(group):
            idx = np.flatnonzero(group == g)
            if idx.size == 0:
                continue
            pos = idx[y[idx] == 1]
            neg = idx[y[idx] == 0]
            if pos.size == 0 or neg.size == 0:
                continue
            p = int(pos[0])
            n_pick = min(self.max_pairs_per_group, int(neg.size))
            chosen = rng.choice(neg, size=n_pick, replace=False)
            diffs = X[p] - X[chosen]
            pairs_X.append(diffs)
            pairs_y.append(np.ones(n_pick, dtype=np.int32))
        if not pairs_X:
            raise RuntimeError("Pairwise training failed: no (pos, neg) pairs available")
        PX = np.vstack(pairs_X).astype(np.float32)
        Py = np.concatenate(pairs_y).astype(np.int32)
        scaler = StandardScaler(with_mean=True, with_std=True)
        PXs = scaler.fit_transform(PX)
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, random_state=self.random_state)
        clf.fit(PXs, Py)
        self._scaler = scaler
        self._clf = clf
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(X)
        return self._clf.decision_function(Xs).astype(np.float32)


def _sort_by_group_and_sizes(X: np.ndarray, y: np.ndarray, group: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(group, kind="mergesort")
    Xs = X[order]
    ys = y[order]
    gs = group[order]
    if gs.size == 0:
        return Xs, ys, np.asarray([], dtype=np.int32)
    sizes: List[int] = []
    prev = int(gs[0])
    cnt = 1
    for g in gs[1:]:
        gi = int(g)
        if gi == prev:
            cnt += 1
        else:
            sizes.append(cnt)
            prev = gi
            cnt = 1
    sizes.append(cnt)
    return Xs, ys, np.asarray(sizes, dtype=np.int32)


class XGBoostLambdaRanker(Ranker):
    name = "xgboost_lambdarank"

    def __init__(self, random_state: int = 7) -> None:
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "XGBoostLambdaRanker":
        import xgboost as xgb

        Xs, ys, group_sizes = _sort_by_group_and_sizes(X, y, group)
        model = xgb.XGBRanker(
            tree_method="hist",
            objective="rank:ndcg",
            learning_rate=0.08,
            max_depth=7,
            n_estimators=500,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
        )
        model.fit(Xs, ys, group=group_sizes)
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(np.float32)


class LightGBMLambdaRanker(Ranker):
    name = "lightgbm_lambdarank"

    def __init__(self, random_state: int = 7) -> None:
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "LightGBMLambdaRanker":
        import lightgbm as lgb

        Xs, ys, group_sizes = _sort_by_group_and_sizes(X, y, group)
        model = lgb.LGBMRanker(
            objective="lambdarank",
            learning_rate=0.08,
            max_depth=-1,
            num_leaves=63,
            n_estimators=700,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
        )
        model.fit(Xs, ys, group=group_sizes)
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(np.float32)


def get_ranker(name: str, seed: int) -> Ranker:
    name = str(name).lower().strip()
    if name == HeuristicRanker.name:
        return HeuristicRanker()
    if name == PointwiseGBDTRanker.name:
        return PointwiseGBDTRanker(random_state=seed)
    if name == PairwiseLogRegRanker.name:
        return PairwiseLogRegRanker(random_state=seed)
    if name == XGBoostLambdaRanker.name:
        return XGBoostLambdaRanker(random_state=seed)
    if name == LightGBMLambdaRanker.name:
        return LightGBMLambdaRanker(random_state=seed)
    raise ValueError(f"Unknown ranker: {name}")


