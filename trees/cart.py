"""
Decision Tree Classifier (CART Algorithm)
Supports both Gini impurity and Entropy (Information Gain) for splitting.
"""

import numpy as np
from collections import Counter

# %%
class TreeNode:
    """Node in a decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Feature index for split
        self.threshold = threshold    # Threshold value for split
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.value = value           # Class label (for leaf nodes)

# %%
class CARTClassifier:
    """
    Decision Tree Classifier using CART algorithm.
    
    Supports two split criteria:
    - Gini impurity: measures how often a randomly chosen element would be incorrectly labeled
    - Entropy (Information Gain): measures the impurity/disorder in the data
    
    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split a node
    criterion : str, default='gini'
        The function to measure split quality. Options: 'gini', 'entropy'
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes_ = None
        
    def fit(self, X, y):
        """
        Build the decision tree classifier.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        """
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self
        
    def _gini(self, y):
        """
        Calculate Gini impurity.
        
        Gini = 1 - Σ(p_i²) where p_i is the probability of class i
        
        Lower values indicate purer nodes (better).
        Range: [0, 1 - 1/n_classes]
        """
        counter = Counter(y)
        n_samples = len(y)
        gini = 1.0
        
        for count in counter.values():
            p = count / n_samples
            gini -= p ** 2
            
        return gini
    
    def _entropy(self, y):
        """
        Calculate entropy (Shannon entropy).
        
        Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
        
        Lower values indicate purer nodes (better).
        Range: [0, log2(n_classes)]
        """
        counter = Counter(y)
        n_samples = len(y)
        entropy = 0.0
        
        for count in counter.values():
            p = count / n_samples
            if p > 0:  # Avoid log(0)
                entropy -= p * np.log2(p)
                
        return entropy
    
    def _information_gain(self, parent_y, left_y, right_y):
        """
        Calculate information gain from a split.
        
        Information Gain = Entropy(parent) - Weighted_Average(Entropy(children))
        
        Higher values indicate better splits.
        """
        n = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        if self.criterion == 'gini':
            # For Gini, we calculate reduction in impurity
            parent_impurity = self._gini(parent_y)
            left_impurity = self._gini(left_y)
            right_impurity = self._gini(right_y)
        else:  # entropy
            parent_impurity = self._entropy(parent_y)
            left_impurity = self._entropy(left_y)
            right_impurity = self._entropy(right_y)
        
        # Weighted average of children impurities
        weighted_child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # Information gain is reduction in impurity
        gain = parent_impurity - weighted_child_impurity
        return gain
    
    def _find_best_split(self, X, y):
        """
        Find the best split for a node.
        
        Returns
        -------
        best_feature : int
            Index of the feature to split on
        best_threshold : float
            Threshold value for the split
        best_gain : float
            Information gain from the split
        """
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Try every feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try every unique value as a threshold
            for threshold in thresholds:
                # Split the data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Skip if split doesn't divide the data
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Calculate information gain
                gain = self._information_gain(y, left_y, right_y)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        
        Stopping criteria:
        1. Maximum depth reached
        2. Fewer than min_samples_split samples
        3. All samples have the same label (pure node)
        4. No valid split found
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Create leaf node with majority class
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no valid split found, create leaf
        if best_feature is None or best_gain <= 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Return internal node
        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample by traversing the tree"""
        # If leaf node, return the class value
        if node.value is not None:
            return node.value
        
        # Otherwise, traverse left or right based on feature value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        Note: This is a simplified version that returns 0 or 1 for each class.
        A proper implementation would require storing class distributions in leaf nodes.
        """
        predictions = self.predict(X)
        n_samples = len(X)
        proba = np.zeros((n_samples, self.n_classes_))
        proba[np.arange(n_samples), predictions] = 1.0
        return proba

# %%
def example():
    """Example usage of Decision Tree Classifier"""
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=" * 70)
    print("Decision Tree Example 1: Synthetic Data")
    print("=" * 70)
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,
                               n_redundant=2, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with Gini
    print("\nUsing Gini Impurity:")
    tree_gini = CARTClassifier(max_depth=5, criterion='gini')
    tree_gini.fit(X_train, y_train)
    y_pred_gini = tree_gini.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gini):.4f}")
    
    # Train with Entropy
    print("\nUsing Entropy (Information Gain):")
    tree_entropy = CARTClassifier(max_depth=5, criterion='entropy')
    tree_entropy.fit(X_train, y_train)
    y_pred_entropy = tree_entropy.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_entropy):.4f}")
    
    print("\n" + "=" * 70)
    print("Decision Tree Example 2: Iris Dataset")
    print("=" * 70)
    
    # Load Iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    tree = CARTClassifier(max_depth=3, criterion='gini')
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    print("\n" + "=" * 70)

# %%
if __name__ == "__main__":
    example()
