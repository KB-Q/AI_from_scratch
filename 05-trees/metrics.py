"""
Ranking metrics.

- dcg / ndcg / mean_ndcg: take true labels and predicted scores (used by the LambdaMART objective and the examples).
- RankingMetrics: takes relevance labels already in ranked order (rel[i] = true relevance of the item ranked at position i).
"""

import numpy as np
from typing import Optional, Callable, Tuple

def dcg(relevance: np.ndarray, k: Optional[int] = None) -> float:
    """DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1), for relevance labels in ranked order (k=None uses all)."""
    if k is not None:
        relevance = relevance[:k]

    if len(relevance) == 0:
        return 0.0

    gains = np.power(2.0, relevance) - 1.0
    discounts = np.log2(np.arange(len(relevance)) + 2.0)
    return np.sum(gains / discounts)


def ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: Optional[int] = None) -> float:
    """NDCG@k = DCG@k of the ranking by y_pred / DCG@k of the ideal ranking by y_true."""
    sorted_indices = np.argsort(-y_pred)
    sorted_relevance = y_true[sorted_indices]

    dcg_value = dcg(sorted_relevance, k)

    ideal_sorted_relevance = np.sort(y_true)[::-1]
    idcg = dcg(ideal_sorted_relevance, k)

    if idcg == 0.0:
        return 0.0

    return dcg_value / idcg


def mean_ndcg(y_true: np.ndarray, y_pred: np.ndarray, query_ids: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k over query groups."""
    ndcg_scores = []
    for qid in np.unique(query_ids):
        query_mask = query_ids == qid
        ndcg_scores.append(ndcg(y_true[query_mask], y_pred[query_mask], k=k))
    return float(np.mean(ndcg_scores))


class RankingMetrics:
    """
    Stateless ranking metrics for IR and recommendations.

    Input: rel[i] = true relevance of item YOUR MODEL ranked at position i.

    Example: Model ranks items [D, A, C, B]. True relevances: A=3, B=2, C=0, D=1.
             Input: rel = [1, 3, 0, 2]  (D's rel, A's rel, C's rel, B's rel)

    Binary metrics (Precision, Recall, MRR, MAP) treat rel > threshold as relevant.
    Graded metrics (NDCG) use raw relevance scores.
    """

    @staticmethod
    def cumulative_gain(rel, k, do_exp=True):
        """CG=Σrel, DCG=Σ(gain/log2(i+1)), NDCG=DCG/IDCG. gain=2^rel-1 if do_exp else rel."""
        rel = np.asarray(rel)
        rel_k = rel[:k]
        if len(rel_k) == 0:
            return {'CG': 0.0, 'DCG': 0.0, 'IDCG': 0.0, 'NDCG': 0.0}

        positions = np.arange(1, len(rel_k) + 1)
        discounts = np.log2(positions + 1)
        gains = (np.power(2, rel_k) - 1) if do_exp else rel_k.astype(float)

        cg = float(np.sum(rel_k))
        dcg = float(np.sum(gains / discounts))

        ideal_rel = np.sort(rel)[::-1][:k]
        ideal_gains = (np.power(2, ideal_rel) - 1) if do_exp else ideal_rel.astype(float)
        ideal_discounts = np.log2(np.arange(1, len(ideal_rel) + 1) + 1)
        idcg = float(np.sum(ideal_gains / ideal_discounts))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        return {'CG': cg, 'DCG': dcg, 'IDCG': idcg, 'NDCG': ndcg}

    @staticmethod
    def precision_recall(rel, k, total_relevant=None, threshold=0.0):
        """P@K=#rel_in_k/K, R@K=#rel_in_k/total_rel, F1=2PR/(P+R), Hit=1 if any rel in k."""
        rel = np.asarray(rel)
        relevant_mask = rel > threshold
        rel_in_k = int(np.sum(relevant_mask[:k]))
        total_rel = total_relevant if total_relevant is not None else int(np.sum(relevant_mask))

        precision = rel_in_k / k if k > 0 else 0.0
        recall = rel_in_k / total_rel if total_rel > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        hit = 1.0 if rel_in_k > 0 else 0.0

        return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Hit': hit}

    @staticmethod
    def average_precision(rel, threshold=0.0):
        """RR=1/rank_of_first_rel, AP=mean(P@k for each rel position k)."""
        rel = np.asarray(rel)
        relevant_mask = rel > threshold
        relevant_positions = np.where(relevant_mask)[0]

        rr = 1.0 / (relevant_positions[0] + 1) if len(relevant_positions) > 0 else 0.0

        if len(relevant_positions) == 0:
            return {'RR': 0.0, 'AP': 0.0}

        precisions_at_rel = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
        ap = float(np.mean(precisions_at_rel))

        return {'RR': rr, 'AP': ap}

    @staticmethod
    def mrr(rel_list, threshold=0.0):
        """MRR = mean(1/rank_of_first_rel) across queries."""
        if not rel_list:
            return 0.0
        rrs = [RankingMetrics.average_precision(r, threshold)['RR'] for r in rel_list]
        return float(np.mean(rrs))

    @staticmethod
    def map_score(rel_list, threshold=0.0):
        """MAP = mean(AP) across queries."""
        if not rel_list:
            return 0.0
        aps = [RankingMetrics.average_precision(r, threshold)['AP'] for r in rel_list]
        return float(np.mean(aps))

    @staticmethod
    def compute_all(rel, k, total_relevant=None, do_exp=True):
        """All single-query metrics at K."""
        cg = RankingMetrics.cumulative_gain(rel, k, do_exp)
        pr = RankingMetrics.precision_recall(rel, k, total_relevant)
        ap = RankingMetrics.average_precision(rel)
        return {**cg, **pr, **ap}

    @staticmethod
    def compute_corpus(rel_list, k, do_exp=True):
        """Corpus-level metrics: mean of single-query metrics + MRR + MAP."""
        if not rel_list:
            return {}

        ndcgs = [RankingMetrics.cumulative_gain(r, k, do_exp)['NDCG'] for r in rel_list]
        prs = [RankingMetrics.precision_recall(r, k) for r in rel_list]

        return {
            'Mean_NDCG': float(np.mean(ndcgs)),
            'Mean_Precision': float(np.mean([p['Precision'] for p in prs])),
            'Mean_Recall': float(np.mean([p['Recall'] for p in prs])),
            'Mean_Hit': float(np.mean([p['Hit'] for p in prs])),
            'MRR': RankingMetrics.mrr(rel_list),
            'MAP': RankingMetrics.map_score(rel_list),
        }


# if __name__ == "__main__":
#     rel = np.array([3, 1, 0, 2, 0])
#     print("Single query:", RankingMetrics.compute_all(rel, k=3))

#     queries = [np.array([1, 0, 1, 0]), np.array([0, 1, 0, 0]), np.array([1, 1, 0, 1])]
#     print("Corpus:", RankingMetrics.compute_corpus(queries, k=3))

class ObjectiveFunctions:
    """Stateless objective functions for XGBoost; all methods are static."""

    @staticmethod
    def compute_lambda_gradients(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        query_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LambdaMART gradients and Hessians. For each query and each document pair with
        different labels, the pair's lambda is scaled by |ΔNDCG|, the change in NDCG if the
        two documents swapped ranks; the Hessian adds σ(1 - σ)·|ΔNDCG|, σ = sigmoid(s_i - s_j).
        """
        n_samples = len(y_true)
        gradients = np.zeros(n_samples)
        hessians = np.zeros(n_samples)

        unique_queries = np.unique(query_ids)

        for qid in unique_queries:
            query_mask = query_ids == qid
            query_indices = np.where(query_mask)[0]
            if len(query_indices) <= 1:
                continue

            query_y_true = y_true[query_mask]
            query_y_pred = y_pred[query_mask]
            sorted_order = np.argsort(-query_y_pred)
            n_docs = len(query_indices)

            ideal_sorted_relevance = np.sort(query_y_true)[::-1]
            idcg = dcg(ideal_sorted_relevance)
            if idcg == 0.0:
                continue

            # Pairwise lambdas weighted by the NDCG change of swapping documents i and j
            for i in range(n_docs):
                for j in range(n_docs):
                    if i == j or query_y_true[i] == query_y_true[j]:
                        continue

                    i_rank = np.where(sorted_order == i)[0][0]
                    j_rank = np.where(sorted_order == j)[0][0]

                    gain_i = np.power(2.0, query_y_true[i]) - 1.0
                    gain_j = np.power(2.0, query_y_true[j]) - 1.0
                    discount_i = 1.0 / np.log2(i_rank + 2.0)
                    discount_j = 1.0 / np.log2(j_rank + 2.0)
                    delta_dcg = (gain_i - gain_j) * (discount_i - discount_j)
                    delta_ndcg = delta_dcg / idcg

                    score_diff = query_y_pred[i] - query_y_pred[j]
                    sigmoid = 1.0 / (1.0 + np.exp(-score_diff))

                    lambda_ij = -sigmoid * abs(delta_ndcg)
                    if query_y_true[i] > query_y_true[j]:
                        lambda_ij = -lambda_ij

                    global_i = query_indices[i]
                    gradients[global_i] += lambda_ij
                    hessian_ij = sigmoid * (1.0 - sigmoid) * abs(delta_ndcg)
                    hessians[global_i] += hessian_ij

        hessians = np.maximum(hessians, 1e-16)

        return gradients, hessians

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over axis 1."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def get_gradient_hessian_functions(
        objective: str
    ) -> Tuple[Callable, Callable]:
        """
        (gradient, hessian) functions of the raw score ŷ for each objective:
        - reg:squarederror: g = ŷ - y, h = 1
        - binary:logistic: g = p - y, h = p(1 - p), with p = sigmoid(ŷ)
        - multi:softmax / multi:softprob: per class k, g = p_k - I(y = k), h = p_k(1 - p_k), with p = softmax(ŷ)
        - rank:ndcg: placeholders; the lambda gradients come from compute_lambda_gradients
        """
        if objective == 'reg:squarederror':
            def gradient(y_true, y_pred):
                return y_pred - y_true

            def hessian(y_true, y_pred):
                return np.ones_like(y_true)

        elif objective == 'binary:logistic':
            def gradient(y_true, y_pred):
                p = 1.0 / (1.0 + np.exp(-y_pred))
                return p - y_true

            def hessian(y_true, y_pred):
                p = 1.0 / (1.0 + np.exp(-y_pred))
                return p * (1.0 - p)

        elif objective in ['multi:softmax', 'multi:softprob']:
            def gradient(y_true, y_pred, class_idx):
                probs = ObjectiveFunctions.softmax(y_pred)
                grad = probs[:, class_idx].copy()
                grad[y_true == class_idx] -= 1
                return grad

            def hessian(y_true, y_pred, class_idx):
                probs = ObjectiveFunctions.softmax(y_pred)
                p_k = probs[:, class_idx]
                return p_k * (1.0 - p_k)

        elif objective == 'rank:ndcg':
            def gradient(y_true, y_pred):
                return np.zeros_like(y_true)

            def hessian(y_true, y_pred):
                return np.ones_like(y_true)

        else:
            raise ValueError(f"Unsupported objective: {objective}")

        return gradient, hessian

    @staticmethod
    def get_base_prediction(y: np.ndarray, objective: str, n_classes: Optional[int] = None) -> float:
        """Initial score: the mean (regression), the log-odds of the positive rate (binary), 0 otherwise."""
        if objective == 'reg:squarederror':
            return np.mean(y)
        elif objective == 'binary:logistic':
            p = np.mean(y)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.log(p / (1 - p))
        elif objective in ['multi:softmax', 'multi:softprob']:
            return 0.0
        else:
            return 0.0
