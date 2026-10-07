"""
XGBoost (Extreme Gradient Boosting) from scratch: the boosting loop.
This implementation follows the core XGBoost algorithm as described in the original paper:
"XGBoost: A Scalable Tree Boosting System" by Chen & Guestrin (2016)
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, List

from core_xgb_tree import XGBoostTree
from metrics import ObjectiveFunctions, ndcg


class XGBoost:
    """
    XGBoost classifier/regressor: gradient boosting with regularized trees.
    The model is an additive ensemble:
    ŷᵢ = Σₖ fₖ(xᵢ) = ŷᵢ⁽⁰⁾ + f₁(xᵢ) + f₂(xᵢ) + ... + fₜ(xᵢ)
    where each fₜ is a regression tree.

    Objectives: reg:squarederror, binary:logistic, multi:softmax (returns labels),
    multi:softprob (returns probabilities), rank:ndcg (LambdaMART; fit needs query_ids).
    learning_rate is the shrinkage η; subsample and colsample_bytree sample rows and columns
    per tree; lambda_reg, alpha, gamma, and min_child_weight are passed to each XGBoostTree.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        lambda_reg: float = 1.0,
        alpha: float = 0.0,
        objective: Literal['reg:squarederror', 'binary:logistic', 'multi:softmax', 'multi:softprob', 'rank:ndcg'] = 'reg:squarederror',
        base_score: Optional[float] = None,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.objective = objective
        self.base_score = base_score
        self.random_state = random_state

        self.trees: List[XGBoostTree] = []
        self.base_prediction = None
        self.n_classes = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        query_ids: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        """
        Fit the XGBoost model using the additive training strategy.

        At iteration t, we optimize:
        L⁽ᵗ⁾ = Σᵢ l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ)) + Ω(fₜ)

        Using second-order Taylor expansion:
        L⁽ᵗ⁾ ≈ Σᵢ [l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + ½hᵢfₜ²(xᵢ)] + Ω(fₜ)

        where gᵢ = ∂l/∂ŷ⁽ᵗ⁻¹⁾ and hᵢ = ∂²l/∂ŷ⁽ᵗ⁻¹⁾²

        For ranking objectives, lambda gradients are computed using pairwise comparisons
        within each query group.

        This implements the additive training in Algorithm 1 and Equations (2)-(3).
        verbose=k prints the training loss (NDCG@10 for ranking) every k rounds.
        """
        # Inputs: arrays, the class count (multiclass), and query ids (ranking)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if self.objective in ['multi:softmax', 'multi:softprob']:
            self.n_classes = len(np.unique(y))
            y_multiclass = y.astype(int)
        else:
            y_multiclass = None

        if self.objective == 'rank:ndcg':
            if query_ids is None:
                raise ValueError("query_ids must be provided for ranking objectives")
            if isinstance(query_ids, pd.Series):
                query_ids = query_ids.values
            query_ids = np.array(query_ids)
            if len(query_ids) != len(y):
                raise ValueError("query_ids must have the same length as y")

        # Base score and initial raw predictions ((n_samples, n_classes) for multiclass)
        if self.base_score is not None:
            self.base_prediction = self.base_score
        else:
            self.base_prediction = ObjectiveFunctions.get_base_prediction(y, self.objective, self.n_classes)

        if self.objective in ['multi:softmax', 'multi:softprob']:
            current_predictions = np.zeros((len(y), self.n_classes), dtype=np.float64)
        else:
            current_predictions = np.full(len(y), self.base_prediction)

        gradient_func, hessian_func = ObjectiveFunctions.get_gradient_hessian_functions(self.objective)

        for iteration in range(self.n_estimators):
            if self.objective == 'rank:ndcg':
                gradients, hessians = ObjectiveFunctions.compute_lambda_gradients(
                    y, current_predictions, query_ids
                )
            elif self.objective in ['multi:softmax', 'multi:softprob']:
                # Multiclass: one tree per class per round, each fitted to that class's softmax gradients
                for class_idx in range(self.n_classes):
                    gradients = gradient_func(y_multiclass, current_predictions, class_idx)
                    hessians = hessian_func(y_multiclass, current_predictions, class_idx)

                    if self.subsample < 1.0:
                        n_samples = int(len(X) * self.subsample)
                        indices = np.random.choice(len(X), n_samples, replace=False)
                        X_sample = X[indices]
                        grad_sample = gradients[indices]
                        hess_sample = hessians[indices]
                    else:
                        X_sample = X
                        grad_sample = gradients
                        hess_sample = hessians

                    tree = XGBoostTree(
                        max_depth=self.max_depth,
                        min_child_weight=self.min_child_weight,
                        gamma=self.gamma,
                        lambda_reg=self.lambda_reg,
                        alpha=self.alpha,
                        colsample_bytree=self.colsample_bytree
                    )
                    tree.fit(X_sample, grad_sample, hess_sample)
                    self.trees.append(tree)

                    tree_predictions = tree.predict(X)
                    current_predictions[:, class_idx] += self.learning_rate * tree_predictions

                if verbose > 0 and iteration % verbose == 0:
                    probs = ObjectiveFunctions.softmax(current_predictions)
                    probs = np.clip(probs, 1e-7, 1 - 1e-7)
                    loss = -np.mean(np.log(probs[np.arange(len(y)), y_multiclass]))
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, Log Loss: {loss:.6f}")

                continue
            else:
                gradients = gradient_func(y, current_predictions)
                hessians = hessian_func(y, current_predictions)

            # Row subsampling, fit one tree to (g, h), then update ŷ⁽ᵗ⁾ = ŷ⁽ᵗ⁻¹⁾ + η·fₜ(x)
            if self.subsample < 1.0:
                n_samples = int(len(X) * self.subsample)
                indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X[indices]
                grad_sample = gradients[indices]
                hess_sample = hessians[indices]
            else:
                X_sample = X
                grad_sample = gradients
                hess_sample = hessians

            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                lambda_reg=self.lambda_reg,
                alpha=self.alpha,
                colsample_bytree=self.colsample_bytree
            )
            tree.fit(X_sample, grad_sample, hess_sample)
            self.trees.append(tree)

            tree_predictions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predictions

            # Training MSE / log loss / NDCG@10 every `verbose` rounds
            if verbose > 0 and iteration % verbose == 0:
                if self.objective == 'reg:squarederror':
                    loss = np.mean((y - current_predictions) ** 2)
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, MSE: {loss:.6f}")
                elif self.objective == 'binary:logistic':
                    probs = 1.0 / (1.0 + np.exp(-current_predictions))
                    probs = np.clip(probs, 1e-7, 1 - 1e-7)
                    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, Log Loss: {loss:.6f}")
                elif self.objective == 'rank:ndcg':
                    unique_queries = np.unique(query_ids)
                    ndcg_scores = []
                    for qid in unique_queries:
                        query_mask = query_ids == qid
                        score = ndcg(
                            y[query_mask],
                            current_predictions[query_mask],
                            k=10
                        )
                        ndcg_scores.append(score)
                    avg_ndcg = np.mean(ndcg_scores)
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, NDCG@10: {avg_ndcg:.6f}")

        return self

    def predict(self, X: np.ndarray, output_margin: bool = False) -> np.ndarray:
        """
        Final prediction: ŷᵢ = ŷ⁽⁰⁾ + η·Σₖ fₖ(xᵢ)

        binary:logistic returns probabilities, multi:softmax class labels, multi:softprob
        class probabilities; output_margin=True returns the raw scores instead.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.array(X, dtype=np.float64)

        # Multiclass: trees are stored as n_classes per round
        if self.objective in ['multi:softmax', 'multi:softprob']:
            predictions = np.zeros((len(X), self.n_classes), dtype=np.float64)
            tree_idx = 0
            for _ in range(self.n_estimators):
                for class_idx in range(self.n_classes):
                    if tree_idx < len(self.trees):
                        tree_pred = self.trees[tree_idx].predict(X)
                        predictions[:, class_idx] += self.learning_rate * tree_pred
                        tree_idx += 1

            if output_margin:
                return predictions

            if self.objective == 'multi:softmax':
                return np.argmax(predictions, axis=1)
            else:
                return ObjectiveFunctions.softmax(predictions)

        predictions = np.full(len(X), self.base_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        if not output_margin:
            if self.objective == 'binary:logistic':
                predictions = 1.0 / (1.0 + np.exp(-predictions))

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities: (n_samples, 2) for binary, (n_samples, n_classes) for multiclass."""
        if self.objective not in ['binary:logistic', 'multi:softmax', 'multi:softprob']:
            raise ValueError("predict_proba is only available for classification")

        if self.objective in ['multi:softmax', 'multi:softprob']:
            raw_predictions = self.predict(X, output_margin=True)
            return ObjectiveFunctions.softmax(raw_predictions)
        else:
            probs_class1 = self.predict(X, output_margin=False)
            probs_class0 = 1 - probs_class1
            return np.column_stack([probs_class0, probs_class1])
