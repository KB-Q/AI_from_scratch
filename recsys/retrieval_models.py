from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModel = None
    AutoTokenizer = None


@dataclass(frozen=True)
class IdMaps:
    user_to_index: Dict[int, int]
    index_to_user: np.ndarray
    item_to_index: Dict[int, int]
    index_to_item: np.ndarray

    @staticmethod
    def from_ids(user_ids: Sequence[int], item_ids: Sequence[int]) -> "IdMaps":
        unique_users = np.unique(np.asarray(user_ids, dtype=np.int64))
        unique_items = np.unique(np.asarray(item_ids, dtype=np.int64))
        user_to_index = {int(u): int(i) for i, u in enumerate(unique_users)}
        item_to_index = {int(it): int(i) for i, it in enumerate(unique_items)}
        return IdMaps(
            user_to_index=user_to_index,
            index_to_user=unique_users,
            item_to_index=item_to_index,
            index_to_item=unique_items,
        )


@dataclass
class PreparedData:
    ratings_all: pd.DataFrame
    items: pd.DataFrame
    positives_train: pd.DataFrame
    positives_test: pd.DataFrame
    id_maps: IdMaps
    user_train_items: Dict[int, np.ndarray]
    user_train_item_set: Dict[int, set]
    item_popularity: pd.Series
    item_mean_rating: pd.Series
    item_num_ratings: pd.Series
    item_genre_matrix: np.ndarray
    item_texts: np.ndarray


def eligible_item_mask_for_user(data: PreparedData, user_id: int) -> np.ndarray:
    interacted = data.user_train_item_set.get(int(user_id), set())
    item_ids = data.items["item_id"].to_numpy(dtype=np.int64)
    if not interacted:
        return np.ones_like(item_ids, dtype=bool)
    return ~np.isin(item_ids, np.fromiter(interacted, dtype=np.int64))


@dataclass(frozen=True)
class RetrievalRow:
    user_id: int
    item_id: int
    source: str
    score: float


class Retriever:
    name: str = "base"

    def fit(self, data: PreparedData) -> "Retriever":
        return self

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        raise NotImplementedError

    def retrieve_batch(self, data: PreparedData, user_ids: Sequence[int], k: int) -> pd.DataFrame:
        rows: List[RetrievalRow] = []
        for u in user_ids:
            rows.extend(self.retrieve(data, int(u), k))
        if not rows:
            return pd.DataFrame(columns=["user_id", "item_id", "source", "score"])
        return pd.DataFrame(dataclasses.asdict(r) for r in rows)


class PopularityRetriever(Retriever):
    name = "popularity"

    def fit(self, data: PreparedData) -> "PopularityRetriever":
        pop = data.item_popularity.reindex(data.items["item_id"]).fillna(0.0).astype(np.float32).to_numpy()
        self._pop_scores = pop
        return self

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        eligible = eligible_item_mask_for_user(data, user_id)
        scores = self._pop_scores.copy()
        scores[~eligible] = -1.0
        top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)[top_idx]
        return [RetrievalRow(user_id, int(it), self.name, float(scores[i])) for it, i in zip(item_ids, top_idx)]


class RandomRetriever(Retriever):
    name = "random"

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        eligible = eligible_item_mask_for_user(data, user_id)
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size == 0:
            return []
        pick = min(k, eligible_idx.size)
        chosen = self._rng.choice(eligible_idx, size=pick, replace=False)
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)[chosen]
        scores = self._rng.random(size=pick).astype(np.float32)
        order = np.argsort(-scores)
        return [RetrievalRow(user_id, int(item_ids[i]), self.name, float(scores[i])) for i in order]


class CoVisitRetriever(Retriever):
    name = "covisit"

    def __init__(self, recent_n: int = 10, max_neighbors: int = 200) -> None:
        self.recent_n = int(recent_n)
        self.max_neighbors = int(max_neighbors)

    def fit(self, data: PreparedData) -> "CoVisitRetriever":
        neighbors: DefaultDict[int, Counter] = defaultdict(Counter)
        by_user = data.positives_train.groupby("user_id")["item_id"].apply(list)
        for items in by_user:
            uniq = list(dict.fromkeys(items))
            for i in range(len(uniq)):
                a = int(uniq[i])
                for j in range(len(uniq)):
                    if i == j:
                        continue
                    b = int(uniq[j])
                    neighbors[a][b] += 1

        top_neighbors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for a, ctr in neighbors.items():
            if not ctr:
                continue
            most = ctr.most_common(self.max_neighbors)
            item_ids = np.fromiter((it for it, _ in most), dtype=np.int64)
            scores = np.fromiter((c for _, c in most), dtype=np.float32)
            top_neighbors[int(a)] = (item_ids, scores)

        self._top_neighbors = top_neighbors
        return self

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        history = data.user_train_items.get(int(user_id))
        if history is None or history.size == 0:
            return []
        recent = history[-self.recent_n :]
        eligible = eligible_item_mask_for_user(data, user_id)
        eligible_items = set(map(int, data.items.loc[eligible, "item_id"].to_numpy(dtype=np.int64)))

        scores: Dict[int, float] = {}
        for it in recent[::-1]:
            neigh = self._top_neighbors.get(int(it))
            if neigh is None:
                continue
            n_items, n_scores = neigh
            for cand, s in zip(n_items, n_scores):
                cand_i = int(cand)
                if cand_i not in eligible_items:
                    continue
                scores[cand_i] = scores.get(cand_i, 0.0) + float(s)

        if not scores:
            return []
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [RetrievalRow(user_id, int(it), self.name, float(sc)) for it, sc in top]


class ItemKNNCosineRetriever(Retriever):
    name = "itemknn"

    def __init__(self, max_neighbors: int = 200) -> None:
        self.max_neighbors = int(max_neighbors)

    def fit(self, data: PreparedData) -> "ItemKNNCosineRetriever":
        user_index = {int(u): i for i, u in enumerate(sorted(data.user_train_items.keys()))}
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)
        item_index = {int(it): i for i, it in enumerate(item_ids)}

        rows, cols = [], []
        for u, its in data.user_train_items.items():
            ui = user_index.get(int(u))
            if ui is None:
                continue
            for it in np.unique(its):
                ii = item_index.get(int(it))
                if ii is None:
                    continue
                rows.append(ii)
                cols.append(ui)

        X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(item_ids), len(user_index)))
        sim = cosine_similarity(X, dense_output=False)

        top_neighbors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz == 0:
                continue
            idx = row.indices
            vals = row.data
            mask = idx != i
            idx = idx[mask]
            vals = vals[mask]
            if idx.size == 0:
                continue
            topk = min(self.max_neighbors, idx.size)
            part = np.argpartition(-vals, kth=topk - 1)[:topk]
            sel_idx = idx[part]
            sel_vals = vals[part]
            order = np.argsort(-sel_vals)
            top_neighbors[int(item_ids[i])] = (
                item_ids[sel_idx[order]].astype(np.int64),
                sel_vals[order].astype(np.float32),
            )

        self._top_neighbors = top_neighbors
        return self

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        history = data.user_train_items.get(int(user_id))
        if history is None or history.size == 0:
            return []
        eligible = eligible_item_mask_for_user(data, user_id)
        eligible_items = set(map(int, data.items.loc[eligible, "item_id"].to_numpy(dtype=np.int64)))
        scores: Dict[int, float] = {}
        for it in np.unique(history):
            neigh = self._top_neighbors.get(int(it))
            if neigh is None:
                continue
            n_items, n_scores = neigh
            for cand, s in zip(n_items, n_scores):
                cand_i = int(cand)
                if cand_i not in eligible_items:
                    continue
                scores[cand_i] = scores.get(cand_i, 0.0) + float(s)
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [RetrievalRow(user_id, int(it), self.name, float(sc)) for it, sc in top]


class TfidfCosineRetriever(Retriever):
    name = "tfidf"

    def __init__(self, max_features: int = 20000, profile_recent_n: int = 30) -> None:
        self.max_features = int(max_features)
        self.profile_recent_n = int(profile_recent_n)

    def fit(self, data: PreparedData) -> "TfidfCosineRetriever":
        vect = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), stop_words="english")
        X = vect.fit_transform(data.item_texts)
        self._vect = vect
        self._item_matrix = X.tocsr()
        return self

    def _user_profile(self, data: PreparedData, user_id: int) -> sp.csr_matrix:
        history = data.user_train_items.get(int(user_id))
        if history is None or history.size == 0:
            return sp.csr_matrix((1, self._item_matrix.shape[1]), dtype=np.float32)
        recent = history[-self.profile_recent_n :]
        item_id_to_row = {int(it): i for i, it in enumerate(data.items["item_id"].to_numpy(dtype=np.int64))}
        rows = [item_id_to_row.get(int(it)) for it in recent]
        rows = [r for r in rows if r is not None]
        if not rows:
            return sp.csr_matrix((1, self._item_matrix.shape[1]), dtype=np.float32)
        prof = self._item_matrix[rows].mean(axis=0)
        return sp.csr_matrix(prof, dtype=np.float32)

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        prof = self._user_profile(data, user_id)
        eligible = eligible_item_mask_for_user(data, user_id)
        scores = (self._item_matrix @ prof.T).toarray().ravel()
        scores[~eligible] = -1.0
        if scores.size == 0:
            return []
        top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)[top_idx]
        return [RetrievalRow(user_id, int(it), self.name, float(scores[i])) for it, i in zip(item_ids, top_idx)]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, profile_recent_n: int = 30, max_features: int = 50000):
        self.k1 = float(k1)
        self.b = float(b)
        self.profile_recent_n = int(profile_recent_n)
        self.max_features = int(max_features)

    def fit(self, data: PreparedData) -> "BM25Retriever":
        vect = CountVectorizer(max_features=self.max_features, ngram_range=(1, 2), stop_words="english")
        X = vect.fit_transform(data.item_texts).tocsr()
        df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
        n_docs = float(X.shape[0])
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(dl.mean()) if dl.size else 1.0
        self._vect = vect
        self._X = X
        self._idf = idf
        self._dl = dl
        self._avgdl = avgdl
        return self

    def _user_query(self, data: PreparedData, user_id: int) -> str:
        history = data.user_train_items.get(int(user_id))
        if history is None or history.size == 0:
            return ""
        recent = history[-self.profile_recent_n :]
        item_id_to_text = dict(zip(data.items["item_id"].to_numpy(dtype=np.int64), data.items["item_text"].to_numpy()))
        return " ".join(item_id_to_text.get(int(it), "") for it in recent)

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        q = self._user_query(data, user_id)
        if not q:
            return []
        q_vec = self._vect.transform([q]).tocsr()
        if q_vec.nnz == 0:
            return []
        eligible = eligible_item_mask_for_user(data, user_id)

        rows = np.repeat(q_vec.indices, q_vec.data.astype(np.int64))
        q_terms = Counter(map(int, rows))

        scores = np.zeros(self._X.shape[0], dtype=np.float32)
        for term_idx, _qtf in q_terms.items():
            idf = float(self._idf[term_idx])
            col = self._X.getcol(term_idx)
            if col.nnz == 0:
                continue
            tf = col.data.astype(np.float32)
            doc_idx = col.indices
            dl = self._dl[doc_idx]
            denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
            s = idf * (tf * (self.k1 + 1.0) / denom)
            scores[doc_idx] += s

        scores[~eligible] = -1.0
        top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)[top_idx]
        return [RetrievalRow(user_id, int(it), self.name, float(scores[i])) for it, i in zip(item_ids, top_idx)]


class SVDRetriever(Retriever):
    name = "svd"

    def __init__(self, n_components: int = 64, random_state: int = 7) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)

    def fit(self, data: PreparedData) -> "SVDRetriever":
        users = sorted(data.user_train_items.keys())
        user_index = {int(u): i for i, u in enumerate(users)}
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)
        item_index = {int(it): i for i, it in enumerate(item_ids)}

        rows, cols = [], []
        for u, its in data.user_train_items.items():
            ui = user_index[int(u)]
            for it in np.unique(its):
                ii = item_index.get(int(it))
                if ii is None:
                    continue
                rows.append(ui)
                cols.append(ii)
        X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), len(item_ids)))
        n_comp = min(self.n_components, min(X.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        U = svd.fit_transform(X).astype(np.float32)
        V = svd.components_.T.astype(np.float32)
        self._user_index = user_index
        self._U = U
        self._V = V
        return self

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        ui = self._user_index.get(int(user_id))
        if ui is None:
            return []
        eligible = eligible_item_mask_for_user(data, user_id)
        scores = (self._U[ui] @ self._V.T).astype(np.float32, copy=False)
        scores[~eligible] = -1.0
        top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)[top_idx]
        return [RetrievalRow(user_id, int(it), self.name, float(scores[i])) for it, i in zip(item_ids, top_idx)]


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32, copy=False)


def _build_faiss_index(
    dim: int,
    index_type: str,
    metric: str,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
):
    if faiss is None:
        raise ImportError("faiss is required for ANN retrieval (install faiss-cpu)")
    index_type = str(index_type).lower().strip()
    metric = str(metric).lower().strip()
    if metric not in {"ip"}:
        raise ValueError("Only metric='ip' is supported (use normalized vectors for cosine)")

    if index_type in {"flat", "faiss_flat"}:
        return faiss.IndexFlatIP(int(dim))
    if index_type in {"hnsw", "faiss_hnsw"}:
        idx = faiss.IndexHNSWFlat(int(dim), int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = int(ef_construction)
        idx.hnsw.efSearch = int(ef_search)
        return idx
    raise ValueError(f"Unknown FAISS index_type: {index_type}")


def _mean_pool_last_hidden(last_hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_texts_hf(
    texts: Union[Sequence[str], np.ndarray],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    if torch is None or AutoTokenizer is None or AutoModel is None:
        raise ImportError("torch + transformers are required for text embedding retrieval")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    out: List[np.ndarray] = []
    texts_list = list(map(str, texts))
    with torch.no_grad():
        for i in range(0, len(texts_list), int(batch_size)):
            batch = texts_list[i : i + int(batch_size)]
            enc = tok(batch, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            pooled = _mean_pool_last_hidden(h, enc["attention_mask"])
            vec = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
            out.append(vec)
    emb = np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)
    return _l2_normalize_rows(emb) if normalize and emb.size else emb


class TextEmbeddingANNRetriever(Retriever):
    name = "text_emb_ann"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 64,
        normalize: bool = True,
        index_type: str = "faiss_hnsw",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        profile_recent_n: int = 30,
    ) -> None:
        self.model_name = str(model_name)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.index_type = str(index_type)
        self.hnsw_m = int(hnsw_m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.profile_recent_n = int(profile_recent_n)

    def fit(self, data: PreparedData) -> "TextEmbeddingANNRetriever":
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)
        emb = _encode_texts_hf(
            data.item_texts,
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=self.normalize,
        )
        idx = _build_faiss_index(
            dim=int(emb.shape[1]),
            index_type=self.index_type,
            metric="ip",
            hnsw_m=self.hnsw_m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
        )
        idx.add(np.ascontiguousarray(emb, dtype=np.float32))
        self._index = idx
        self._item_ids = item_ids
        self._item_id_to_row = {int(it): i for i, it in enumerate(item_ids)}
        self._item_emb = emb
        return self

    def _user_profile(self, data: PreparedData, user_id: int) -> Optional[np.ndarray]:
        hist = data.user_train_items.get(int(user_id))
        if hist is None or hist.size == 0:
            return None
        recent = hist[-self.profile_recent_n :]
        rows = [self._item_id_to_row.get(int(it)) for it in recent]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None
        prof = self._item_emb[np.asarray(rows, dtype=np.int64)].mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        return _l2_normalize_rows(prof) if self.normalize else prof

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        prof = self._user_profile(data, user_id)
        if prof is None:
            return []
        seen = data.user_train_item_set.get(int(user_id), set())
        search_k = min(int(k) * 5, int(self._item_ids.size))
        scores, idxs = self._index.search(np.ascontiguousarray(prof, dtype=np.float32), int(search_k))
        cand_ids = self._item_ids[idxs[0]]
        out: List[RetrievalRow] = []
        for it, sc in zip(cand_ids, scores[0]):
            it_i = int(it)
            if it_i < 0 or it_i in seen:
                continue
            out.append(RetrievalRow(int(user_id), it_i, self.name, float(sc)))
            if len(out) >= int(k):
                break
        return out


class TwoTowerANNRetriever(Retriever):
    name = "two_tower_ann"

    def __init__(
        self,
        embedding_dim: int = 64,
        epochs: int = 2,
        batch_size: int = 1024,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        device: str = "cpu",
        normalize: bool = True,
        index_type: str = "faiss_hnsw",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 7,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = str(device)
        self.normalize = bool(normalize)
        self.index_type = str(index_type)
        self.hnsw_m = int(hnsw_m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.seed = int(seed)

    def fit(self, data: PreparedData) -> "TwoTowerANNRetriever":
        if torch is None:
            raise ImportError("torch is required for two-tower retrieval")

        rng = np.random.default_rng(self.seed)
        users = np.asarray(sorted(data.user_train_items.keys()), dtype=np.int64)
        item_ids = data.items["item_id"].to_numpy(dtype=np.int64)
        user_index = {int(u): i for i, u in enumerate(users)}
        item_index = {int(it): i for i, it in enumerate(item_ids)}
        user_seen: Dict[int, set] = {}
        for u, its in data.user_train_items.items():
            ui = user_index[int(u)]
            user_seen[ui] = {item_index[int(it)] for it in np.unique(its) if int(it) in item_index}

        pairs = data.positives_train[["user_id", "item_id"]].to_numpy(dtype=np.int64)
        u_idx = np.asarray([user_index[int(u)] for u in pairs[:, 0]], dtype=np.int64)
        i_idx = np.asarray([item_index[int(it)] for it in pairs[:, 1] if int(it) in item_index], dtype=np.int64)
        if i_idx.size != u_idx.size:
            keep = np.asarray([int(it) in item_index for it in pairs[:, 1]], dtype=bool)
            u_idx = u_idx[keep]
            i_idx = np.asarray([item_index[int(it)] for it in pairs[:, 1][keep]], dtype=np.int64)

        import torch.nn as nn

        class _TwoTower(nn.Module):
            def __init__(self, n_users: int, n_items: int, dim: int) -> None:
                super().__init__()
                self.user = nn.Embedding(n_users, dim)
                self.item = nn.Embedding(n_items, dim)
                nn.init.normal_(self.user.weight, std=0.02)
                nn.init.normal_(self.item.weight, std=0.02)

            def forward(self, u: "torch.Tensor", it: "torch.Tensor") -> "torch.Tensor":
                return (self.user(u) * self.item(it)).sum(dim=1)

        model = _TwoTower(int(users.size), int(item_ids.size), int(self.embedding_dim)).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def sample_neg(batch_users: np.ndarray) -> np.ndarray:
            neg = rng.integers(0, item_ids.size, size=batch_users.size, dtype=np.int64)
            bad = np.asarray([neg[i] in user_seen.get(int(u), set()) for i, u in enumerate(batch_users)], dtype=bool)
            while bad.any():
                neg[bad] = rng.integers(0, item_ids.size, size=int(bad.sum()), dtype=np.int64)
                bad = np.asarray([neg[i] in user_seen.get(int(u), set()) for i, u in enumerate(batch_users)], dtype=bool)
            return neg

        idx = np.arange(u_idx.size, dtype=np.int64)
        for _ in range(int(self.epochs)):
            rng.shuffle(idx)
            for start in range(0, idx.size, int(self.batch_size)):
                batch = idx[start : start + int(self.batch_size)]
                bu = u_idx[batch]
                bp = i_idx[batch]
                bn = sample_neg(bu)

                tu = torch.tensor(bu, dtype=torch.long, device=self.device)
                tp = torch.tensor(bp, dtype=torch.long, device=self.device)
                tn = torch.tensor(bn, dtype=torch.long, device=self.device)

                pos = model(tu, tp)
                neg = model(tu, tn)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            item_vec = model.item.weight.detach().cpu().numpy().astype(np.float32, copy=False)
            user_vec = model.user.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        if self.normalize:
            item_vec = _l2_normalize_rows(item_vec)
            user_vec = _l2_normalize_rows(user_vec)

        ann = _build_faiss_index(
            dim=int(item_vec.shape[1]),
            index_type=self.index_type,
            metric="ip",
            hnsw_m=self.hnsw_m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
        )
        ann.add(np.ascontiguousarray(item_vec, dtype=np.float32))
        self._index = ann
        self._item_ids = item_ids
        self._user_index = user_index
        self._user_vec = user_vec
        return self

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        ui = self._user_index.get(int(user_id))
        if ui is None:
            return []
        seen = data.user_train_item_set.get(int(user_id), set())
        q = self._user_vec[int(ui)].reshape(1, -1).astype(np.float32, copy=False)
        search_k = min(int(k) * 5, int(self._item_ids.size))
        scores, idxs = self._index.search(np.ascontiguousarray(q, dtype=np.float32), int(search_k))
        cand_ids = self._item_ids[idxs[0]]
        out: List[RetrievalRow] = []
        for it, sc in zip(cand_ids, scores[0]):
            it_i = int(it)
            if it_i < 0 or it_i in seen:
                continue
            out.append(RetrievalRow(int(user_id), it_i, self.name, float(sc)))
            if len(out) >= int(k):
                break
        return out


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], per_retriever_k: Mapping[str, int]) -> None:
        self.retrievers = list(retrievers)
        self.per_retriever_k = dict(per_retriever_k)

    def fit(self, data: PreparedData) -> "HybridRetriever":
        for r in self.retrievers:
            r.fit(data)
        return self

    def retrieve(self, data: PreparedData, user_id: int, k: int) -> List[RetrievalRow]:
        union: Dict[Tuple[int, str], float] = {}
        for r in self.retrievers:
            rk = int(self.per_retriever_k.get(r.name, k))
            for row in r.retrieve(data, user_id, rk):
                key = (int(row.item_id), str(row.source))
                union[key] = max(union.get(key, float("-inf")), float(row.score))
        rows = [RetrievalRow(user_id, it, src, sc) for (it, src), sc in union.items()]
        rows.sort(key=lambda x: x.score, reverse=True)
        return rows[:k] if k > 0 else rows

    def retrieve_batch(self, data: PreparedData, user_ids: Sequence[int], k: int) -> pd.DataFrame:
        frames = []
        for r in self.retrievers:
            rk = int(self.per_retriever_k.get(r.name, k))
            frames.append(r.retrieve_batch(data, user_ids, rk))
        if not frames:
            return pd.DataFrame(columns=["user_id", "item_id", "source", "score"])
        return pd.concat(frames, ignore_index=True)


