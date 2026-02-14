### broad themes:

- **retrieval** - retrieve top 1000 candidates using cosine similarity of embeddings (FAST, recall optimized)
- **ranking** - rank top 10 candidates using complex models like DNNs or LambdaMART (SLOW, precision optimized)
- **old school methods** 
    - collaborative filtering (based on user similarities), 
    - content filtering (based on item similarities), 
    - hybrid methods combining both
    - matrix factorization - given a User x Item matrix (of say reviews) of shape m x n, factorize it into a User latent matrix (m x l) and Item latent matrix (l x n)
    - from these latent matrices we can cluster users / determine similarities and do the above filtering methods

OG paper - https://minsuk.com/research/paper/www2011-kahng-contextrec.pdf

**key model milestones:**

- ranknet, lambdarank, lambdamart

### Learning to Rank (LTR)

- **pointwise** - each candidate is independent, so do simple regression / classification. ignores interaction between items
- **pairwise** - take pairs of items to predict which out of the two should be ranked higher (RankNet and RankSVM)
- **listwise** 
    - take lists of items, optimize a list level metric (such as NDCG)
    - LambdaMART does this with boosted trees, but uses a pair wise gradient approximation called a 'lambda'
    - "LambdaMART computes these 'lambdas' based on how swapping a pair changes NDCG, then trains a tree ensemble using those gradients"

### retrieval and reranking

- **Bi-encoder retrieval**  
    - Encode query and docs **independently** (fast, indexable). Good for the first pass over millions of items. 
    - Common encoders: DPR, sentence-transformers. [GitHub+1](https://github.com/facebookresearch/DPR?utm_source=chatgpt.com)
- **Re-rank top-K**  
    - Take the K candidates and score them with a **cross-encoder** (feeds query+doc _together_ through a transformer; slower but more accurate). 
    - Typical models: MS-MARCO cross-encoders from Sentence-Transformers (MiniLM variants). [SentenceTransformers+2SentenceTransformers+2](https://sbert.net/examples/cross_encoder/training/ms_marco/README.html?utm_source=chatgpt.com)
- **Late-interaction option (ColBERT)**  
    - Middle ground: encode query/docs separately, but keep token-level vectors and do a cheap late interaction at query time. 
    - Higher accuracy than pure bi-encoder at lower cost than full cross-encoder; production-proven. [arXiv+1](https://arxiv.org/abs/2004.12832?utm_source=chatgpt.com)
- **Learning-to-rank option (tree models)**  
    - Instead of a cross-encoder, you can feature-engineer (similarities, BM25, position, click priors) and train **LambdaMART/XGBoost rankers** to re-rank K. 

### NDCG:

- CG - Cumulative Gain:
    - CG_p = sum(rel_i) for i from 1 to p
    - basically sum of relevance scores (can be binary or ordinal) upto position p in the list
- DCG - Discounted Cumulative Gain:
    - highly relevant documents appearing lower should be penalized, with a logarithm term of the position index
    - DCG_p = sum(rel_i / log2(i+1)) for i from 1 to p
- DCG - alternative formula: 
    - DCG_p = sum((2^rel_i - 1) / log2(i+1)) for i from 1 to p
    - this is commonly used in industry and kaggle competitions
- Ideal DCG: 
    - IDCG_p is same as DCG_p but with documents in correct order from 1 to p
- Normalized DCG: 
    - NDCG_p = DCG_p / IDCG_p
- Python code:

```python
  def calculate_NDCG(rel_scores, p):
		import numpy as np
		CG_p = sum(rel_scores[:p])
		DCG_p = sum(rel_scores[:p] / np.log2(range(1, p + 1)))
		IDCG_p = sum(np.sort(rel_scores)[:p] / np.log2(range(1, p + 1)))
		NDCG_p = DCG_p / IDCG_p
		return CG_p, DCG_p, IDCG_p, NDCG_p
```

### Observational vs. interventional problems[](https://eugeneyan.com/writing/counterfactual-evaluation/#observational-vs-interventional-problems)

- predict product category -> observations that directly lead to outcome (True or False)
- item recommendations -> interventions that lead indirectly to outcome (increase in clicks)
- direct evaluation - A/B testing. but costly, time consuming, can't be done offline
- indirect evaluation - counterfactual
    - “what would have happened if we show users our new recommendations instead of the existing recommendations?”

### IPS - Inverse Propensity Score

- The intuition behind it is that we can estimate how customer interactions will change — by reweighting how often each interaction will occur — based on how much more (or less) each item is shown by our new recommendation model.
- Formula: ![alt text](<../IMAGES/Pasted image 20251105185107.png>)
- Explanation of terms:
    - `r` represents the reward for an observation. This is the number of clicks or purchases or whatever metric is important to you in the logged data.
    - The denominator represents our existing production recommender’s (`π0`) probability of making a recommendation (aka _action_ `a`) given the context `x`. 
    - The numerator (section 2b) represents the same probability but for our new recommender (`πe`). (`π` stands for recommendation _policy_.) 
    - For a user-to-item recommender, `x` is the user; for an item-to-item recommender, `x` is an item.
- Example:
    - we have an old model (`π0`) and new model (`πe`) that recommend iPhone on the Pixel detail page
    - _π0(recommend=iPhone|view=Pixel)_ = 0.4
    - _πe(recommend=iPhone|view=Pixel)_ = 0.6
    - In this scenario, the new model will recommend iPhone 0.6/0.4 = 1.5x as often as the old model. Thus we can reweight the logged reward (click / purchase) to be worth 1.5x as much.
- Drawbacks:
    - insufficient support 
        - when production recommender _π0_ probability is 0 (for new samples)
        - can show new samples for a small subset, or get probabilities offline
    - high variance 
        - when new model recommends very differently from old model, especially in edge cases (0.001 vs 0.1 -> 100x weight to the reward)
        - can be solved by CIPS - set a max threshold to the weight (say 10)
- SNIPS - self normalize the weighted sum by the sum of weights
    - Note: should compute weights for all rewards (zero and non-zero)
    - Formula: ![SNIPS formula](<../IMAGES/Pasted image 20251105190427.png>)

### BM 25 (Best Matching) - upgraded TFIDF:

- Formula ![[Pasted image 20251105191952.png]]