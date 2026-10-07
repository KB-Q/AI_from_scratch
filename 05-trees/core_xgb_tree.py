"""
XGBoost core: one regression tree fitted to gradients and Hessians.

Implements the optimal leaf weight (Eq. 5), split gain (Eq. 7), and the exact greedy
split-finding algorithm (Algorithm 1) from "XGBoost: A Scalable Tree Boosting System"
(Chen & Guestrin, 2016).
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Internal nodes hold feature, threshold, children, and the split gain; leaves hold the weight in `value`."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    value: Optional[float] = None
    gain: Optional[float] = None

    def is_leaf(self) -> bool:
        return self.value is not None


class XGBoostTree:
    """
    One XGBoost regression tree, grown with the exact greedy algorithm.

    Regularisation: lambda_reg is the L2 penalty λ on leaf weights, alpha the L1 penalty
    (soft-thresholds G), gamma the minimum split gain γ, and min_child_weight the minimum
    Hessian sum per child. colsample_bytree samples the features once per tree.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        lambda_reg: float = 1.0,
        alpha: float = 0.0,
        colsample_bytree: float = 1.0
    ):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.colsample_bytree = colsample_bytree
        self.root = None
        self.feature_indices = np.array([])

    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        n_features = X.shape[1]
        n_sample_features = max(1, int(n_features * self.colsample_bytree))
        self.feature_indices = np.random.choice(n_features, n_sample_features, replace=False)
        self.root = self._build_tree(X, gradients, hessians, depth=0)

    def _calculate_leaf_weight(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        """Optimal leaf weight (Eq. 5): w* = -G / (H + λ), with G = Σg, H = Σh; L1 soft-thresholds G by α."""
        G = np.sum(gradients)
        H = np.sum(hessians)

        if self.alpha > 0:
            if G > self.alpha:
                G = G - self.alpha
            elif G < -self.alpha:
                G = G + self.alpha
            else:
                return 0.0

        return -G / (H + self.lambda_reg)

    def _calculate_similarity_score(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        """Node score -G² / (H + λ), the per-node term of Eq. 7, with the same L1 soft-thresholding."""
        G = np.sum(gradients)
        H = np.sum(hessians)

        if self.alpha > 0:
            if G > self.alpha:
                G = G - self.alpha
            elif G < -self.alpha:
                G = G + self.alpha
            else:
                G = 0.0

        return -(G ** 2) / (H + self.lambda_reg)

    def _calculate_split_gain(
        self,
        grad_left: np.ndarray,
        hess_left: np.ndarray,
        grad_right: np.ndarray,
        hess_right: np.ndarray,
        grad_parent: np.ndarray,
        hess_parent: np.ndarray
    ) -> float:
        """
        Split gain (Eq. 7):
        Gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ)] - γ

        The node scores are negative (-G²/(H + λ)), so this equals
        0.5 * (score_parent - score_left - score_right) - γ.
        """
        score_left = self._calculate_similarity_score(grad_left, hess_left)
        score_right = self._calculate_similarity_score(grad_right, hess_right)
        score_parent = self._calculate_similarity_score(grad_parent, hess_parent)
        gain = 0.5 * (score_parent - score_left - score_right) - self.gamma
        return gain

    def _find_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Exact greedy split finding (Algorithm 1):
        1. For each sampled feature, take its sorted unique values
        2. Try every threshold between consecutive values
        3. Skip splits where either child's Hessian sum is below min_child_weight
        4. Return the (feature, threshold, gain) with the maximum gain
        """
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in self.feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            if len(unique_values) == 1:
                continue

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(hessians[left_mask]) < self.min_child_weight:
                    continue
                if np.sum(hessians[right_mask]) < self.min_child_weight:
                    continue

                gain = self._calculate_split_gain(
                    gradients[left_mask],
                    hessians[left_mask],
                    gradients[right_mask],
                    hessians[right_mask],
                    gradients,
                    hessians
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int
    ) -> TreeNode:
        """Recursively grow the tree; a node becomes a leaf at max depth, when empty, or when no split has positive gain."""
        node = TreeNode()

        if depth >= self.max_depth or len(X) == 0:
            node.value = self._calculate_leaf_weight(gradients, hessians)
            return node

        best_feature, best_threshold, best_gain = self._find_best_split(
            X, gradients, hessians
        )
        if best_feature is None or best_gain <= 0:
            node.value = self._calculate_leaf_weight(gradients, hessians)
            return node

        node.feature = best_feature
        node.threshold = best_threshold
        node.gain = best_gain

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        node.left = self._build_tree(
            X[left_mask],
            gradients[left_mask],
            hessians[left_mask],
            depth + 1
        )
        node.right = self._build_tree(
            X[right_mask],
            gradients[right_mask],
            hessians[right_mask],
            depth + 1
        )

        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
