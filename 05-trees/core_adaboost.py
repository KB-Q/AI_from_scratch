"""
AdaBoost (Adaptive Boosting) classifier.

Weak learners (decision stumps by default) are trained sequentially; each round
increases the weights of the samples the previous learners misclassified.
"""

import numpy as np
from core_cart import CARTClassifier

# %%
class AdaBoostClassifier:
    """AdaBoost with the SAMME multi-class update; the base learner defaults to a depth-1 CART stump."""

    def __init__(self, n_estimators=50, learning_rate=1.0, base_estimator=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.n_classes_ = None

    def fit(self, X, y, verbose=False):
        """
        SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss):

        1. Initialize sample weights: w_i = 1/n for all samples
        2. For m = 1 to M (number of estimators):
            a) Train weak learner h_m on weighted samples
            b) Calculate weighted error: err_m = Σ w_i * I(y_i ≠ h_m(x_i))
            c) Calculate estimator weight: α_m = log((1 - err_m) / err_m) + log(K - 1)
            d) Update sample weights: w_i *= exp(α_m * I(y_i ≠ h_m(x_i)))
            e) Normalize weights
        3. Final prediction: argmax_k Σ α_m * I(h_m(x) = k)

        Weighted training is simulated by resampling the data according to w.
        Labels must be in {0, 1, ..., K-1}.
        """
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))
        sample_weights = np.ones(n_samples) / n_samples

        for estimator_idx in range(self.n_estimators):
            if self.base_estimator is None:
                estimator = CARTClassifier(max_depth=1, criterion='gini')
            else:
                estimator = self.base_estimator

            # Fit on a resample drawn with the sample weights, then measure the weighted error on all samples
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
            estimator.fit(X[indices], y[indices])
            y_pred = estimator.predict(X)
            incorrect = (y_pred != y)
            estimator_error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            # SAMME estimator weight and sample-weight update; a perfect learner gets a large fixed weight, a worse-than-random one stops training
            if estimator_error <= 0:
                estimator_weight = 10.0
                sample_weights = np.ones(n_samples) / n_samples
            elif estimator_error >= 1 - 1/self.n_classes_:
                if verbose:
                    print(f"Stopping at iteration {estimator_idx}: error = {estimator_error:.4f}")
                break
            else:
                estimator_weight = np.log((1 - estimator_error) / estimator_error)
                estimator_weight += np.log(self.n_classes_ - 1)
                sample_weights *= np.exp(estimator_weight * incorrect)
                sample_weights /= np.sum(sample_weights)

            estimator_weight *= self.learning_rate
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(estimator_error)

            if verbose and (estimator_idx + 1) % 10 == 0:
                print(f"Iteration {estimator_idx + 1}/{self.n_estimators}, "
                      f"Error: {estimator_error:.4f}, Weight: {estimator_weight:.4f}")

        return self

    def predict(self, X):
        """Weighted vote: y(x) = argmax_k Σ α_m * I(h_m(x) = k)."""
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        n_samples = X.shape[0]
        weighted_votes = np.zeros((n_samples, self.n_classes_))
        for i, (pred, weight) in enumerate(zip(predictions, self.estimator_weights_)):
            for class_idx in range(self.n_classes_):
                weighted_votes[:, class_idx] += weight * (pred == class_idx)
        return np.argmax(weighted_votes, axis=1)

    def predict_proba(self, X):
        """Weighted votes normalized to sum to 1."""
        n_samples = X.shape[0]
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        weighted_votes = np.zeros((n_samples, self.n_classes_))
        for pred, weight in zip(predictions, self.estimator_weights_):
            for class_idx in range(self.n_classes_):
                weighted_votes[:, class_idx] += weight * (pred == class_idx)
        proba = weighted_votes / np.sum(weighted_votes, axis=1, keepdims=True)
        return proba

    def feature_importances_(self):
        """Weight-averaged importances of the base estimators that define feature_importances_."""
        importances = np.zeros(self.estimators_[0].n_features_)
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            if hasattr(estimator, 'feature_importances_'):
                importances += weight * estimator.feature_importances_
        return importances / np.sum(self.estimator_weights_)
