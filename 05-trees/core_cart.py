"""
Decision tree classifier (CART) with Gini or entropy split criteria.
"""

import numpy as np
from collections import Counter

# %%
class TreeNode:
    """Internal nodes hold feature, threshold, and children; leaves hold a class label in `value`."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# %%
class CARTClassifier:
    """
    Decision tree classifier (CART).

    Split criteria:
    - Gini impurity: how often a randomly chosen element would be mislabeled
    - Entropy (information gain): the impurity/disorder of the labels
    """

    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes_ = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _gini(self, y):
        """Gini = 1 - Σ p_i², range [0, 1 - 1/K]; lower is purer."""
        counter = Counter(y)
        n_samples = len(y)
        gini = 1.0
        for count in counter.values():
            p = count / n_samples
            gini -= p ** 2
        return gini

    def _entropy(self, y):
        """Entropy = -Σ p_i log2(p_i), range [0, log2(K)]; lower is purer."""
        counter = Counter(y)
        n_samples = len(y)
        entropy = 0.0
        for count in counter.values():
            p = count / n_samples
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _information_gain(self, parent_y, left_y, right_y):
        """Gain = impurity(parent) - size-weighted average impurity(children)."""
        n = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)

        if self.criterion == 'gini':
            parent_impurity = self._gini(parent_y)
            left_impurity = self._gini(left_y)
            right_impurity = self._gini(right_y)
        else:
            parent_impurity = self._entropy(parent_y)
            left_impurity = self._entropy(left_y)
            right_impurity = self._entropy(right_y)

        weighted_child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        gain = parent_impurity - weighted_child_impurity
        return gain

    def _find_best_split(self, X, y):
        """Exhaustive search over every feature and every unique value as threshold; returns (feature, threshold, gain)."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]
                gain = self._information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        """
        Recursively build the tree. A node becomes a majority-class leaf when:
        1. the maximum depth is reached
        2. it has fewer than min_samples_split samples
        3. it is pure (one label)
        4. no split has positive gain
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)

        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        if best_feature is None or best_gain <= 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def predict_proba(self, X):
        """One-hot of the predicted class; leaves store labels, not class distributions."""
        predictions = self.predict(X)
        n_samples = len(X)
        proba = np.zeros((n_samples, self.n_classes_))
        proba[np.arange(n_samples), predictions] = 1.0
        return proba
