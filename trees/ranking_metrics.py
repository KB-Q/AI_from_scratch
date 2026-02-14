import numpy as np

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


if __name__ == "__main__":
    rel = np.array([3, 1, 0, 2, 0])
    print("Single query:", RankingMetrics.compute_all(rel, k=3))
    
    queries = [np.array([1, 0, 1, 0]), np.array([0, 1, 0, 0]), np.array([1, 1, 0, 1])]
    print("Corpus:", RankingMetrics.compute_corpus(queries, k=3))
