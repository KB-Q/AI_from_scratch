"""
Gradient Boosting Classifier

Gradient Boosting is an ensemble method that builds trees sequentially,
where each new tree predicts the gradient (residual) of the loss function
with respect to the current prediction.
"""

import numpy as np
from cart import CARTClassifier

# %%
class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier for binary and multi-class classification.
    
    Gradient Boosting builds an ensemble of trees where each tree is trained
    to predict the negative gradient of the loss function. This is a more
    general framework than AdaBoost.
    
    For binary classification, uses logistic loss (log loss).
    For multi-class, uses softmax with cross-entropy loss.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages (trees) to perform
    learning_rate : float, default=0.1
        Shrinkage parameter to prevent overfitting (0 < lr <= 1)
    max_depth : int, default=3
        Maximum depth of individual trees
    min_samples_split : int, default=2
        Minimum samples required to split a node
    subsample : float, default=1.0
        Fraction of samples to use for fitting each tree (stochastic gradient boosting)
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
        """Numerically stable sigmoid function"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _softmax(self, x):
        """Numerically stable softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _compute_gradients_binary(self, y, y_pred):
        """
        Compute gradients for binary classification (logistic loss).
        
        Loss: L = -[y*log(p) + (1-y)*log(1-p)]
        Gradient: dL/df = p - y (where p = sigmoid(f))
        
        We use the raw prediction f (before sigmoid) and return negative gradient
        so trees predict residuals.
        """
        p = self._sigmoid(y_pred)
        return -(y - p)  # Negative gradient
    
    def _compute_gradients_multiclass(self, y, y_pred):
        """
        Compute gradients for multi-class classification (softmax + cross-entropy).
        
        For each class k:
        Gradient: dL/df_k = p_k - y_k (where p = softmax(f))
        
        Returns negative gradients shaped (n_samples, n_classes)
        """
        proba = self._softmax(y_pred)
        # Convert y to one-hot encoding
        y_onehot = np.zeros((len(y), self.n_classes_))
        y_onehot[np.arange(len(y)), y] = 1
        return -(y_onehot - proba)  # Negative gradient
    
    def fit(self, X, y, verbose=False):
        """
        Build the gradient boosting classifier.
        
        Algorithm:
        1. Initialize with constant prediction (log-odds for binary, zeros for multi-class)
        2. For m = 1 to M:
            a) Compute negative gradients (pseudo-residuals)
            b) Fit a tree to predict these residuals
            c) Update predictions: F_m = F_{m-1} + learning_rate * tree_m
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        verbose : bool, default=False
            Print training progress
        """
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))
        is_binary = self.n_classes_ == 2
        
        # Initialize predictions
        if is_binary:
            # For binary: initialize with log-odds
            pos_count = np.sum(y == 1)
            neg_count = np.sum(y == 0)
            self.init_prediction_ = np.log(pos_count / neg_count) if neg_count > 0 else 0.0
            raw_predictions = np.full(n_samples, self.init_prediction_)
        else:
            # For multi-class: initialize with zeros for each class
            self.init_prediction_ = np.zeros(self.n_classes_)
            raw_predictions = np.tile(self.init_prediction_, (n_samples, 1))
        
        # Build trees sequentially
        for i in range(self.n_estimators):
            # Subsample if requested (stochastic gradient boosting)
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
            
            if is_binary:
                # Binary classification
                gradients = self._compute_gradients_binary(y_subset, raw_pred_subset)
                
                # Fit tree to negative gradients (residuals)
                tree = CARTClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    criterion='gini'
                )
                
                # Convert gradients to binary labels for tree fitting
                # Use median as threshold
                residual_labels = (gradients < np.median(gradients)).astype(int)
                tree.fit(X_subset, residual_labels)
                
                # Update predictions for ALL samples
                tree_predictions = tree.predict(X)
                # Map {0, 1} to {-learning_rate, +learning_rate}
                tree_predictions = (tree_predictions * 2 - 1) * self.learning_rate
                raw_predictions += tree_predictions
                
            else:
                # Multi-class classification - need one tree per class
                trees_for_round = []
                for class_idx in range(self.n_classes_):
                    gradients = self._compute_gradients_multiclass(y_subset, raw_pred_subset)[:, class_idx]
                    
                    tree = CARTClassifier(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        criterion='gini'
                    )
                    
                    # Convert gradients to binary labels
                    residual_labels = (gradients < np.median(gradients)).astype(int)
                    tree.fit(X_subset, residual_labels)
                    
                    # Update predictions
                    tree_predictions = tree.predict(X)
                    tree_predictions = (tree_predictions * 2 - 1) * self.learning_rate
                    raw_predictions[:, class_idx] += tree_predictions
                    
                    trees_for_round.append(tree)
                
                self.trees_.append(trees_for_round)
                continue
            
            self.trees_.append(tree)
            
            if verbose and (i + 1) % 10 == 0:
                # Calculate training accuracy
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
        """Predict raw scores (before sigmoid/softmax)"""
        n_samples = X.shape[0]
        is_binary = self.n_classes_ == 2
        
        if is_binary:
            # Binary classification
            raw_predictions = np.full(n_samples, self.init_prediction_)
            for tree in self.trees_:
                tree_pred = tree.predict(X)
                tree_pred = (tree_pred * 2 - 1) * self.learning_rate
                raw_predictions += tree_pred
        else:
            # Multi-class classification
            raw_predictions = np.tile(self.init_prediction_, (n_samples, 1))
            for trees_for_round in self.trees_:
                for class_idx, tree in enumerate(trees_for_round):
                    tree_pred = tree.predict(X)
                    tree_pred = (tree_pred * 2 - 1) * self.learning_rate
                    raw_predictions[:, class_idx] += tree_pred
        
        return raw_predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        raw_predictions = self._predict_raw(X)
        
        if self.n_classes_ == 2:
            # Binary: apply sigmoid
            proba_class_1 = self._sigmoid(raw_predictions)
            proba = np.column_stack([1 - proba_class_1, proba_class_1])
        else:
            # Multi-class: apply softmax
            proba = self._softmax(raw_predictions)
        
        return proba
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# %%
def example():
    """Example usage of Gradient Boosting Classifier"""
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=" * 70)
    print("Gradient Boosting Example 1: Binary Classification")
    print("=" * 70)
    
    # Generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Gradient Boosting
    print("\nTraining Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8
    )
    gb.fit(X_train, y_train, verbose=True)
    
    # Predict
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)
    
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\n" + "=" * 70)
    print("Gradient Boosting Example 2: Multi-class Classification (Iris)")
    print("=" * 70)
    
    # Load Iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    # Train Gradient Boosting
    print("\nTraining Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3
    )
    gb.fit(X_train, y_train, verbose=False)
    
    # Predict
    y_pred = gb.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Show probability predictions
    y_proba = gb.predict_proba(X_test[:5])
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
