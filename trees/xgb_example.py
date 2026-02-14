# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, log_loss
from sklearn.datasets import make_classification
from xgb_manual import XGBoost
import numpy as np

print("=" * 60)
print("XGBoost Manual Implementation - Example Usage")
print("=" * 60)

# %%
def example_regression():
    # Example 1: Regression
    print("\n" + "=" * 60)
    print("Example 1: Regression Task")
    print("=" * 60)

    # Generate synthetic regression data
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Train XGBoost regressor
    xgb_reg = XGBoost(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1.0,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        lambda_reg=1.0,
        objective='reg:squarederror',
        random_state=42
    )

    print("\nTraining XGBoost Regressor...")
    xgb_reg.fit(X_train_reg, y_train_reg, verbose=True)

    # Make predictions
    y_pred_reg = xgb_reg.predict(X_test_reg)

    # Evaluate
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print(f"\nTest Set Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")

# %%
def example_classification():
    # Example 2: Binary Classification
    print("\n" + "=" * 60)
    print("Example 2: Binary Classification Task")
    print("=" * 60)

    # Generate synthetic classification data
    X_clf, y_clf = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    # Train XGBoost classifier
    xgb_clf = XGBoost(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1.0,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        lambda_reg=1.0,
        objective='binary:logistic',
        random_state=42
    )

    print("\nTraining XGBoost Classifier...")
    xgb_clf.fit(X_train_clf, y_train_clf, verbose=True)

    # Make predictions
    y_pred_proba_clf = xgb_clf.predict(X_test_clf)
    y_pred_clf = (y_pred_proba_clf > 0.5).astype(int)

    # Evaluate
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    auc = roc_auc_score(y_test_clf, y_pred_proba_clf)
    logloss = log_loss(y_test_clf, y_pred_proba_clf)

    print(f"\nTest Set Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Log Loss: {logloss:.4f}")

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)

# %%
def generate_synthetic_ranking_data(n_queries=100, docs_per_query=10, n_features=20, random_state=42):
    """
    Generate synthetic learning-to-rank data.
    
    Returns:
        X: Feature matrix (n_queries * docs_per_query, n_features)
        y: Relevance labels (n_queries * docs_per_query,)
        query_ids: Query IDs (n_queries * docs_per_query,)
    """
    np.random.seed(random_state)
    
    X_list = []
    y_list = []
    query_ids_list = []
    
    for qid in range(n_queries):
        # Generate features for documents in this query
        X_query = np.random.randn(docs_per_query, n_features)
        
        # Generate relevance labels (0-4 scale, with some noise)
        # More relevant documents tend to have higher feature values
        relevance_scores = np.dot(X_query[:, :3], [2.0, 1.5, 1.0])
        relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)
        
        # Convert to discrete relevance labels (0-4)
        y_query = (relevance_scores * 4).astype(int)
        y_query = np.clip(y_query, 0, 4)
        
        # Add some randomness
        noise = np.random.randint(-1, 2, size=docs_per_query)
        y_query = np.clip(y_query + noise, 0, 4)
        
        X_list.append(X_query)
        y_list.append(y_query)
        query_ids_list.append(np.full(docs_per_query, qid))
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    query_ids = np.hstack(query_ids_list)
    
    return X, y, query_ids


def compute_ndcg(y_true, y_pred, k=10):
    """Compute NDCG@k for a single query."""
    # Sort by predicted scores
    sorted_indices = np.argsort(-y_pred)[:k]
    sorted_relevance = y_true[sorted_indices]
    
    # DCG
    gains = np.power(2.0, sorted_relevance) - 1.0
    discounts = np.log2(np.arange(len(sorted_relevance)) + 2.0)
    dcg = np.sum(gains / discounts)
    
    # IDCG (ideal)
    ideal_sorted = np.sort(y_true)[::-1][:k]
    ideal_gains = np.power(2.0, ideal_sorted) - 1.0
    ideal_discounts = np.log2(np.arange(len(ideal_sorted)) + 2.0)
    idcg = np.sum(ideal_gains / ideal_discounts)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


# %%
def evaluate_ranking(y_true, y_pred, query_ids, k=10):
    """Evaluate ranking performance using NDCG@k."""
    unique_queries = np.unique(query_ids)
    ndcg_scores = []
    
    for qid in unique_queries:
        query_mask = query_ids == qid
        ndcg = compute_ndcg(y_true[query_mask], y_pred[query_mask], k=k)
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)


def example_lambdamart():
    print("=" * 70)
    print("LambdaMART Example: Learning to Rank with XGBoost")
    print("=" * 70)
    
    # Generate synthetic ranking data
    print("\n1. Generating synthetic ranking data...")
    X_train, y_train, qids_train = generate_synthetic_ranking_data(
        n_queries=200, docs_per_query=15, n_features=30, random_state=42
    )
    X_test, y_test, qids_test = generate_synthetic_ranking_data(
        n_queries=50, docs_per_query=15, n_features=30, random_state=123
    )
    
    print(f"   Training data: {len(X_train)} documents, {len(np.unique(qids_train))} queries")
    print(f"   Test data: {len(X_test)} documents, {len(np.unique(qids_test))} queries")
    print(f"   Relevance label distribution (train): {np.bincount(y_train.astype(int))}")
    
    # Compute baseline NDCG (random ranking)
    print("\n2. Baseline NDCG (random predictions):")
    random_preds_train = np.random.randn(len(y_train))
    random_preds_test = np.random.randn(len(y_test))
    baseline_ndcg_train = evaluate_ranking(y_train, random_preds_train, qids_train, k=10)
    baseline_ndcg_test = evaluate_ranking(y_test, random_preds_test, qids_test, k=10)
    print(f"   Train NDCG@10: {baseline_ndcg_train:.4f}")
    print(f"   Test NDCG@10: {baseline_ndcg_test:.4f}")
    
    # Train LambdaMART model
    print("\n3. Training LambdaMART model...")
    print("-" * 70)
    
    model = XGBoost(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1.0,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        lambda_reg=1.0,
        objective='rank:ndcg',
        random_state=42
    )
    
    model.fit(X_train, y_train, query_ids=qids_train, verbose=10)
    
    print("-" * 70)
    
    # Make predictions
    print("\n4. Making predictions and evaluating...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Evaluate
    train_ndcg = evaluate_ranking(y_train, train_preds, qids_train, k=10)
    test_ndcg = evaluate_ranking(y_test, test_preds, qids_test, k=10)
    
    print(f"\n5. Final Results:")
    print(f"   Training NDCG@10: {train_ndcg:.4f} (baseline: {baseline_ndcg_train:.4f})")
    print(f"   Test NDCG@10: {test_ndcg:.4f} (baseline: {baseline_ndcg_test:.4f})")
    print(f"   Improvement over baseline: {(test_ndcg - baseline_ndcg_test) / baseline_ndcg_test * 100:.1f}%")
    
    # Show example ranking for one query
    print(f"\n6. Example ranking for a test query:")
    example_qid = np.unique(qids_test)[0]
    example_mask = qids_test == example_qid
    example_y_true = y_test[example_mask]
    example_y_pred = test_preds[example_mask]
    
    # Sort by predictions
    pred_order = np.argsort(-example_y_pred)
    
    print(f"   Query ID: {example_qid}")
    print(f"   {'Rank':<6} {'Predicted Score':<18} {'True Relevance'}")
    print(f"   {'-'*6} {'-'*18} {'-'*15}")
    for rank, doc_idx in enumerate(pred_order[:10], 1):
        print(f"   {rank:<6} {example_y_pred[doc_idx]:<18.4f} {example_y_true[doc_idx]}")
    
    # Compute NDCG for this specific query
    example_ndcg = compute_ndcg(example_y_true, example_y_pred, k=10)
    print(f"\n   NDCG@10 for this query: {example_ndcg:.4f}")
    
    print("\n" + "=" * 70)
    print("LambdaMART example completed successfully!")
    print("=" * 70)