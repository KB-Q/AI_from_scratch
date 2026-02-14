"""
AdaBoost (Adaptive Boosting) Classifier

AdaBoost is an ensemble method that combines multiple weak learners (typically decision stumps)
by training them sequentially. Each subsequent learner focuses more on the samples that previous
learners misclassified by adjusting sample weights.
"""

import numpy as np
from cart import CARTClassifier

# %%
class AdaBoostClassifier:
    """
    AdaBoost Classifier (SAMME algorithm for multi-class).
    
    AdaBoost trains weak learners sequentially, where each new learner focuses on 
    the mistakes of previous learners by increasing the weights of misclassified samples.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The maximum number of estimators (weak learners) to train
    learning_rate : float, default=1.0
        Weight applied to each classifier. Lower values need more estimators
    base_estimator : object, default=None
        The base estimator (weak learner). If None, uses a decision stump (depth=1 tree)
    """
    
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
        Build the AdaBoost classifier.
        
        Algorithm (SAMME - Stagewise Additive Modeling using a Multi-class Exponential loss):
        
        1. Initialize sample weights: w_i = 1/n for all samples
        2. For m = 1 to M (number of estimators):
            a) Train weak learner h_m on weighted samples
            b) Calculate weighted error: err_m = Σ w_i * I(y_i ≠ h_m(x_i))
            c) Calculate estimator weight: α_m = log((1 - err_m) / err_m) + log(K - 1)
            d) Update sample weights: w_i *= exp(α_m * I(y_i ≠ h_m(x_i)))
            e) Normalize weights
        3. Final prediction: argmax_k Σ α_m * I(h_m(x) = k)
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels (must be in {0, 1, ..., K-1})
        verbose : bool, default=False
            Print training progress
        """
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))
        
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        for estimator_idx in range(self.n_estimators):
            # Create base estimator (decision stump if not specified)
            if self.base_estimator is None:
                estimator = CARTClassifier(max_depth=1, criterion='gini')
            else:
                estimator = self.base_estimator
            
            # Train estimator on weighted samples
            # We simulate weighted sampling by resampling according to weights
            indices = np.random.choice(
                n_samples, 
                size=n_samples, 
                replace=True, 
                p=sample_weights
            )
            estimator.fit(X[indices], y[indices])
            
            # Predict on all training samples
            y_pred = estimator.predict(X)
            
            # Calculate misclassification mask
            incorrect = (y_pred != y)
            
            # Calculate weighted error
            estimator_error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # Handle edge cases
            if estimator_error <= 0:
                # Perfect classifier - use it with maximum weight
                estimator_weight = 10.0  # Large weight
                sample_weights = np.ones(n_samples) / n_samples  # Reset weights
            elif estimator_error >= 1 - 1/self.n_classes_:
                # Worse than random - stop early
                if verbose:
                    print(f"Stopping at iteration {estimator_idx}: error = {estimator_error:.4f}")
                break
            else:
                # Calculate estimator weight using SAMME formula
                # α = log((1-err)/err) + log(K-1)
                estimator_weight = np.log((1 - estimator_error) / estimator_error)
                estimator_weight += np.log(self.n_classes_ - 1)
                
                # Update sample weights
                # w_i = w_i * exp(α * I(incorrect))
                sample_weights *= np.exp(estimator_weight * incorrect)
                
                # Normalize weights
                sample_weights /= np.sum(sample_weights)
            
            # Apply learning rate
            estimator_weight *= self.learning_rate
            
            # Store estimator and its weight
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(estimator_error)
            
            if verbose and (estimator_idx + 1) % 10 == 0:
                print(f"Iteration {estimator_idx + 1}/{self.n_estimators}, "
                      f"Error: {estimator_error:.4f}, Weight: {estimator_weight:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        The final prediction is made by weighted voting:
        y(x) = argmax_k Σ α_m * I(h_m(x) = k)
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        # Get predictions from all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Calculate weighted votes for each class
        n_samples = X.shape[0]
        weighted_votes = np.zeros((n_samples, self.n_classes_))
        
        for i, (pred, weight) in enumerate(zip(predictions, self.estimator_weights_)):
            for class_idx in range(self.n_classes_):
                weighted_votes[:, class_idx] += weight * (pred == class_idx)
        
        # Return class with highest weighted vote
        return np.argmax(weighted_votes, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Returns normalized weighted votes as probabilities.
        """
        n_samples = X.shape[0]
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Calculate weighted votes
        weighted_votes = np.zeros((n_samples, self.n_classes_))
        
        for pred, weight in zip(predictions, self.estimator_weights_):
            for class_idx in range(self.n_classes_):
                weighted_votes[:, class_idx] += weight * (pred == class_idx)
        
        # Normalize to get probabilities
        proba = weighted_votes / np.sum(weighted_votes, axis=1, keepdims=True)
        return proba
    
    def feature_importances_(self):
        """
        Calculate feature importances as the average over all estimators.
        Note: Only works if base estimators support feature_importances_.
        """
        importances = np.zeros(self.estimators_[0].n_features_)
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            if hasattr(estimator, 'feature_importances_'):
                importances += weight * estimator.feature_importances_
        return importances / np.sum(self.estimator_weights_)

# %%
def example():
    """Example usage of AdaBoost Classifier"""
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=" * 70)
    print("AdaBoost Example 1: Binary Classification")
    print("=" * 70)
    
    # Generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train AdaBoost with decision stumps (default)
    print("\nAdaBoost with Decision Stumps (depth=1):")
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    ada.fit(X_train, y_train, verbose=True)
    y_pred = ada.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Show how error evolves
    print(f"\nFirst 5 estimator errors: {ada.estimator_errors_[:5]}")
    print(f"Last 5 estimator errors: {ada.estimator_errors_[-5:]}")
    
    print("\n" + "=" * 70)
    print("AdaBoost Example 2: Multi-class Classification (Iris)")
    print("=" * 70)
    
    # Load Iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    # Train AdaBoost
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    ada.fit(X_train, y_train, verbose=False)
    y_pred = ada.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Show probability predictions for first 5 test samples
    y_proba = ada.predict_proba(X_test[:5])
    print("\nProbability predictions for first 5 test samples:")
    for i, (true_label, proba) in enumerate(zip(y_test[:5], y_proba)):
        pred_label = np.argmax(proba)
        print(f"Sample {i}: True={iris.target_names[true_label]}, "
              f"Pred={iris.target_names[pred_label]}, "
              f"Proba={proba}")
    
    print("\n" + "=" * 70)

# %%
if __name__ == "__main__":
    example()
