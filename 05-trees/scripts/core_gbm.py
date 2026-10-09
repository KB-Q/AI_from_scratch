"""
Gradient boosting classifier.

Trees are built sequentially; each new tree is fit to the gradient of the loss
with respect to the current prediction.
"""

import numpy as np
from core_cart import CARTClassifier

# %%
class GradientBoostingClassifier:
    """
    Gradient boosting for binary (logistic loss) and multi-class (softmax cross-entropy) classification.
    A more general framework than AdaBoost: any differentiable loss works.
    subsample < 1 gives stochastic gradient boosting.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.trees_ = []
        self.init_prediction_ = None
        self.n_classes_ = None

    def _sigmoid(self, x):
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _compute_gradients_binary(self, y, y_pred):
        """Logistic loss L = -[y log p + (1-y) log(1-p)], p = sigmoid(f); returns dL/df = p - y."""
        p = self._sigmoid(y_pred)
        return -(y - p)

    def _compute_gradients_multiclass(self, y, y_pred):
        """Softmax cross-entropy, p = softmax(f); returns dL/df_k = p_k - y_k, shape (n_samples, n_classes)."""
        proba = self._softmax(y_pred)
        y_onehot = np.zeros((len(y), self.n_classes_))
        y_onehot[np.arange(len(y)), y] = 1
        return -(y_onehot - proba)

    def fit(self, X, y, verbose=False):
        """
        1. Initialize with a constant prediction (log-odds for binary, zeros for multi-class)
        2. For m = 1 to M:
            a) Compute negative gradients (pseudo-residuals)
            b) Fit a tree to predict these residuals
            c) Update predictions: F_m = F_{m-1} + learning_rate * tree_m
        """
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))
        is_binary = self.n_classes_ == 2

        # Initial raw scores: log-odds (binary) or zeros per class (multi-class)
        if is_binary:
            pos_count = np.sum(y == 1)
            neg_count = np.sum(y == 0)
            self.init_prediction_ = np.log(pos_count / neg_count) if neg_count > 0 else 0.0
            raw_predictions = np.full(n_samples, self.init_prediction_)
        else:
            self.init_prediction_ = np.zeros(self.n_classes_)
            raw_predictions = np.tile(self.init_prediction_, (n_samples, 1))

        for i in range(self.n_estimators):
            if self.subsample < 1.0:
                n_subset = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, n_subset, replace=False)
                X_subset = X[indices]
                y_subset = y[indices]
                raw_pred_subset = raw_predictions[indices] if not is_binary else raw_predictions[indices]
            else:
                X_subset = X
                y_subset = y
                raw_pred_subset = raw_predictions

            # Approximation used here: fit a classification tree to the sign of the gradient relative to its median; each leaf adds ±learning_rate
            if is_binary:
                gradients = self._compute_gradients_binary(y_subset, raw_pred_subset)
                tree = CARTClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    criterion='gini'
                )
                residual_labels = (gradients < np.median(gradients)).astype(int)
                tree.fit(X_subset, residual_labels)
                tree_predictions = tree.predict(X)
                tree_predictions = (tree_predictions * 2 - 1) * self.learning_rate
                raw_predictions += tree_predictions

            else:
                # Multi-class: one tree per class per round
                trees_for_round = []
                for class_idx in range(self.n_classes_):
                    gradients = self._compute_gradients_multiclass(y_subset, raw_pred_subset)[:, class_idx]
                    tree = CARTClassifier(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        criterion='gini'
                    )
                    residual_labels = (gradients < np.median(gradients)).astype(int)
                    tree.fit(X_subset, residual_labels)
                    tree_predictions = tree.predict(X)
                    tree_predictions = (tree_predictions * 2 - 1) * self.learning_rate
                    raw_predictions[:, class_idx] += tree_predictions
                    trees_for_round.append(tree)

                self.trees_.append(trees_for_round)
                continue

            self.trees_.append(tree)

            if verbose and (i + 1) % 10 == 0:
                if is_binary:
                    proba = self._sigmoid(raw_predictions)
                    y_pred = (proba >= 0.5).astype(int)
                else:
                    proba = self._softmax(raw_predictions)
                    y_pred = np.argmax(proba, axis=1)

                accuracy = np.mean(y_pred == y)
                print(f"Iteration {i + 1}/{self.n_estimators}, Training Accuracy: {accuracy:.4f}")

        return self

    def _predict_raw(self, X):
        """Raw scores before sigmoid / softmax."""
        n_samples = X.shape[0]
        is_binary = self.n_classes_ == 2

        if is_binary:
            raw_predictions = np.full(n_samples, self.init_prediction_)
            for tree in self.trees_:
                tree_pred = tree.predict(X)
                tree_pred = (tree_pred * 2 - 1) * self.learning_rate
                raw_predictions += tree_pred
        else:
            raw_predictions = np.tile(self.init_prediction_, (n_samples, 1))
            for trees_for_round in self.trees_:
                for class_idx, tree in enumerate(trees_for_round):
                    tree_pred = tree.predict(X)
                    tree_pred = (tree_pred * 2 - 1) * self.learning_rate
                    raw_predictions[:, class_idx] += tree_pred

        return raw_predictions

    def predict_proba(self, X):
        raw_predictions = self._predict_raw(X)
        if self.n_classes_ == 2:
            proba_class_1 = self._sigmoid(raw_predictions)
            proba = np.column_stack([1 - proba_class_1, proba_class_1])
        else:
            proba = self._softmax(raw_predictions)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
