"""
XGBoost (Extreme Gradient Boosting) Implementation from Scratch

This implementation follows the core XGBoost algorithm as described in the original paper:
"XGBoost: A Scalable Tree Boosting System" by Chen & Guestrin (2016)

The implementation focuses on the exact greedy algorithm for split finding.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, Callable, List, Tuple
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Node in a decision tree"""
    feature: Optional[int] = None  # Feature index for split
    threshold: Optional[float] = None  # Threshold value for split
    left: Optional['TreeNode'] = None  # Left child
    right: Optional['TreeNode'] = None  # Right child
    value: Optional[float] = None  # Leaf value (weight)
    gain: Optional[float] = None  # Gain from this split
    
    def is_leaf(self) -> bool:
        return self.value is not None


class ObjectiveFunctions:
    """
    Stateless class containing objective function implementations for XGBoost.
    All methods are static to keep the class stateless and reusable.
    """
    
    @staticmethod
    def compute_dcg(relevance: np.ndarray, k: Optional[int] = None) -> float:
        """
        Compute Discounted Cumulative Gain (DCG).
        
        DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)
        
        Args:
            relevance: Relevance labels in ranked order
            k: Cutoff position (if None, use all positions)
            
        Returns:
            DCG score
        """
        if k is not None:
            relevance = relevance[:k]
        
        if len(relevance) == 0:
            return 0.0
        
        gains = np.power(2.0, relevance) - 1.0
        discounts = np.log2(np.arange(len(relevance)) + 2.0)
        return np.sum(gains / discounts)
    
    @staticmethod
    def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: Optional[int] = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            y_true: True relevance labels
            y_pred: Predicted scores
            k: Cutoff position (if None, use all positions)
            
        Returns:
            NDCG score
        """
        sorted_indices = np.argsort(-y_pred)
        sorted_relevance = y_true[sorted_indices]
        
        dcg = ObjectiveFunctions.compute_dcg(sorted_relevance, k)
        
        ideal_sorted_relevance = np.sort(y_true)[::-1]
        idcg = ObjectiveFunctions.compute_dcg(ideal_sorted_relevance, k)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def compute_lambda_gradients(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        query_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lambda gradients and Hessians for LambdaMART.
        
        Args:
            y_true: True relevance labels
            y_pred: Current predictions
            query_ids: Query group IDs
            
        Returns:
            Tuple of (gradients, hessians)
        """
        n_samples = len(y_true)
        gradients = np.zeros(n_samples)
        hessians = np.zeros(n_samples)
        
        unique_queries = np.unique(query_ids)
        
        for qid in unique_queries:
            query_mask = query_ids == qid
            query_indices = np.where(query_mask)[0]
            
            if len(query_indices) <= 1:
                continue
            
            query_y_true = y_true[query_mask]
            query_y_pred = y_pred[query_mask]
            
            sorted_order = np.argsort(-query_y_pred)
            n_docs = len(query_indices)
            
            ideal_sorted_relevance = np.sort(query_y_true)[::-1]
            idcg = ObjectiveFunctions.compute_dcg(ideal_sorted_relevance)
            
            if idcg == 0.0:
                continue
            
            for i in range(n_docs):
                for j in range(n_docs):
                    if i == j or query_y_true[i] == query_y_true[j]:
                        continue
                    
                    i_rank = np.where(sorted_order == i)[0][0]
                    j_rank = np.where(sorted_order == j)[0][0]
                    
                    gain_i = np.power(2.0, query_y_true[i]) - 1.0
                    gain_j = np.power(2.0, query_y_true[j]) - 1.0
                    
                    discount_i = 1.0 / np.log2(i_rank + 2.0)
                    discount_j = 1.0 / np.log2(j_rank + 2.0)
                    
                    delta_dcg = (gain_i - gain_j) * (discount_i - discount_j)
                    delta_ndcg = delta_dcg / idcg
                    
                    score_diff = query_y_pred[i] - query_y_pred[j]
                    sigmoid = 1.0 / (1.0 + np.exp(-score_diff))
                    
                    lambda_ij = -sigmoid * abs(delta_ndcg)
                    
                    if query_y_true[i] > query_y_true[j]:
                        lambda_ij = -lambda_ij
                    
                    global_i = query_indices[i]
                    gradients[global_i] += lambda_ij
                    
                    hessian_ij = sigmoid * (1.0 - sigmoid) * abs(delta_ndcg)
                    hessians[global_i] += hessian_ij
        
        hessians = np.maximum(hessians, 1e-16)
        
        return gradients, hessians
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Compute softmax in a numerically stable way.
        
        Args:
            x: Input array of shape (n_samples, n_classes)
            
        Returns:
            Softmax probabilities of same shape
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def get_gradient_hessian_functions(
        objective: str
    ) -> Tuple[Callable, Callable]:
        """
        Get gradient and hessian functions based on objective.
        
        Args:
            objective: Loss function type
            
        Returns:
            Tuple of (gradient_func, hessian_func)
        """
        if objective == 'reg:squarederror':
            def gradient(y_true, y_pred):
                return y_pred - y_true
            
            def hessian(y_true, y_pred):
                return np.ones_like(y_true)
            
        elif objective == 'binary:logistic':
            def gradient(y_true, y_pred):
                p = 1.0 / (1.0 + np.exp(-y_pred))
                return p - y_true
            
            def hessian(y_true, y_pred):
                p = 1.0 / (1.0 + np.exp(-y_pred))
                return p * (1.0 - p)
        
        elif objective in ['multi:softmax', 'multi:softprob']:
            def gradient(y_true, y_pred, class_idx):
                """Gradient for one class in multiclass."""
                # y_pred shape: (n_samples, n_classes)
                probs = ObjectiveFunctions.softmax(y_pred)
                # Gradient: p_k - I(y == k)
                grad = probs[:, class_idx].copy()
                grad[y_true == class_idx] -= 1
                return grad
            
            def hessian(y_true, y_pred, class_idx):
                """Hessian for one class in multiclass."""
                probs = ObjectiveFunctions.softmax(y_pred)
                p_k = probs[:, class_idx]
                # Hessian: p_k * (1 - p_k)
                return p_k * (1.0 - p_k)
        
        elif objective == 'rank:ndcg':
            def gradient(y_true, y_pred):
                return np.zeros_like(y_true)
            
            def hessian(y_true, y_pred):
                return np.ones_like(y_true)
        
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        
        return gradient, hessian
    
    @staticmethod
    def get_base_prediction(y: np.ndarray, objective: str, n_classes: Optional[int] = None) -> float:
        """
        Calculate initial prediction (base score).
        
        Args:
            y: Target values
            objective: Loss function type
            n_classes: Number of classes for multiclass
            
        Returns:
            Base prediction value (scalar) or 0 for multiclass
        """
        if objective == 'reg:squarederror':
            return np.mean(y)
        elif objective == 'binary:logistic':
            p = np.mean(y)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.log(p / (1 - p))
        elif objective in ['multi:softmax', 'multi:softprob']:
            # Initialize all classes to 0 (log-odds)
            return 0.0
        else:
            return 0.0


class XGBoostTree:
    """
    Single decision tree for XGBoost using exact greedy algorithm.
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
        """
        Initialize XGBoost tree parameters.
        
        Args:
            max_depth: Maximum depth of the tree
            min_child_weight: Minimum sum of instance weight (hessian) in a child
            gamma: Minimum loss reduction required to make a split (gamma in regularization)
            lambda_reg: L2 regularization term on weights (lambda)
            alpha: L1 regularization term on weights
            colsample_bytree: Subsample ratio of columns when constructing each tree
        """
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.colsample_bytree = colsample_bytree
        self.root = None
        self.feature_indices = np.array([])
        
    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        """
        Build the tree using gradients and hessians.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            gradients: First-order gradients (n_samples,)
            hessians: Second-order gradients (n_samples,)
        """
        n_features = X.shape[1]
        # Column sampling
        n_sample_features = max(1, int(n_features * self.colsample_bytree))
        self.feature_indices = np.random.choice(
            n_features, n_sample_features, replace=False
        )
        
        # Build tree recursively starting from root
        self.root = self._build_tree(X, gradients, hessians, depth=0)
        
    def _calculate_leaf_weight(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        """
        Calculate optimal leaf weight using equation:
        w* = -G / (H + λ)
        
        where G = sum of gradients, H = sum of hessians
        
        Line reference: This implements Equation (5) from XGBoost paper
        
        Args:
            gradients: Gradient values for samples in this leaf
            hessians: Hessian values for samples in this leaf
            
        Returns:
            Optimal weight for this leaf
        """
        G = np.sum(gradients)
        H = np.sum(hessians)
        
        # L1 regularization (soft thresholding)
        if self.alpha > 0:
            if G > self.alpha:
                G = G - self.alpha
            elif G < -self.alpha:
                G = G + self.alpha
            else:
                return 0.0
        
        return -G / (H + self.lambda_reg)
    
    def _calculate_similarity_score(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        """
        Calculate similarity score for a node:
        Score = -G² / (H + λ)
        
        This is used in the gain calculation.
        Line reference: This is part of Equation (7) from XGBoost paper
        
        Args:
            gradients: Gradient values
            hessians: Hessian values
            
        Returns:
            Similarity score
        """
        G = np.sum(gradients)
        H = np.sum(hessians)
        
        # L1 regularization adjustment
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
        Calculate gain from a split using equation:
        Gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ)] - γ
        
        Line reference: This implements Equation (7) from XGBoost paper
        
        Args:
            grad_left: Gradients for left child
            hess_left: Hessians for left child
            grad_right: Gradients for right child
            hess_right: Hessians for right child
            grad_parent: Gradients for parent node
            hess_parent: Hessians for parent node
            
        Returns:
            Gain value (can be negative)
        """
        score_left = self._calculate_similarity_score(grad_left, hess_left)
        score_right = self._calculate_similarity_score(grad_right, hess_right)
        score_parent = self._calculate_similarity_score(grad_parent, hess_parent)
        
        # Since similarity scores are negative (-G²/(H+λ)), we need to subtract them correctly
        # Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G_P²/(H_P+λ)] - γ
        # With negative scores: Gain = 0.5 * [(-score_L) + (-score_R) - (-score_P)] - γ
        #                            = 0.5 * (score_parent - score_left - score_right) - γ
        gain = 0.5 * (score_parent - score_left - score_right) - self.gamma
        
        return gain
    
    def _find_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split using exact greedy algorithm.
        
        Algorithm:
        1. For each feature, sort the data
        2. Scan through all possible split points
        3. Calculate gain for each split
        4. Return the split with maximum gain
        
        Line reference: This implements Algorithm 1 (Exact Greedy Algorithm) from XGBoost paper
        
        Args:
            X: Feature matrix
            gradients: Gradient values
            hessians: Hessian values
            
        Returns:
            Tuple of (best_feature, best_threshold, best_gain)
        """
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        # Try each feature
        for feature_idx in self.feature_indices:
            # Get unique values and sort them
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # If all values are the same, skip this feature
            if len(unique_values) == 1:
                continue
            
            # Try splits between consecutive unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split the data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Check minimum child weight constraint
                if np.sum(hessians[left_mask]) < self.min_child_weight:
                    continue
                if np.sum(hessians[right_mask]) < self.min_child_weight:
                    continue
                
                # Calculate gain
                gain = self._calculate_split_gain(
                    gradients[left_mask],
                    hessians[left_mask],
                    gradients[right_mask],
                    hessians[right_mask],
                    gradients,
                    hessians
                )
                
                # Update best split if this is better
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
        """
        Recursively build the tree.
        
        Args:
            X: Feature matrix for current node
            gradients: Gradients for current node
            hessians: Hessians for current node
            depth: Current depth
            
        Returns:
            TreeNode representing the current node
        """
        node = TreeNode()
        
        # Stopping criteria
        if depth >= self.max_depth or len(X) == 0:
            node.value = self._calculate_leaf_weight(gradients, hessians)
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, gradients, hessians
        )
        
        # If no valid split found or gain is not positive, make this a leaf
        if best_feature is None or best_gain <= 0:
            node.value = self._calculate_leaf_weight(gradients, hessians)
            return node
        
        # Create split
        node.feature = best_feature
        node.threshold = best_threshold
        node.gain = best_gain
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
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
        """
        Predict using the tree.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def _predict_single(self, x: np.ndarray, node: TreeNode) -> float:
        """
        Predict for a single sample by traversing the tree.
        
        Args:
            x: Single sample features
            node: Current node
            
        Returns:
            Prediction value
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


class XGBoost:
    """
    XGBoost Classifier/Regressor implementing gradient boosting with regularization.
    The model is an additive ensemble:
    ŷᵢ = Σₖ fₖ(xᵢ) = ŷᵢ⁽⁰⁾ + f₁(xᵢ) + f₂(xᵢ) + ... + fₜ(xᵢ)
    where each fₜ is a regression tree.
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
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds (trees)
            learning_rate: Step size shrinkage (η/eta in the paper)
            max_depth: Maximum depth of each tree
            min_child_weight: Minimum sum of hessian in a child
            gamma: Minimum loss reduction for split (complexity control)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            lambda_reg: L2 regularization term on weights
            alpha: L1 regularization term on weights
            objective: Loss function to use
                - 'reg:squarederror': Regression with squared error
                - 'binary:logistic': Binary classification with logistic loss
                - 'multi:softmax': Multiclass classification (returns class labels)
                - 'multi:softprob': Multiclass classification (returns probabilities)
                - 'rank:ndcg': Learning to rank with LambdaMART (NDCG-based)
            base_score: Initial prediction score (if None, uses optimal initial value)
            random_state: Random seed for reproducibility
        """
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
        self.n_classes = None  # Set during fit for multiclass
        
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
        
        Line reference: This implements the additive training in Algorithm 1 and Equation (2)-(3)
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - For ranking, these are relevance labels
            query_ids: Query group IDs (n_samples,) - Required for ranking objectives
            verbose: Whether to print training progress (int for frequency, bool for every 10)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Handle multiclass classification
        if self.objective in ['multi:softmax', 'multi:softprob']:
            self.n_classes = len(np.unique(y))
            y_multiclass = y.astype(int)  # Ensure integer class labels
        else:
            y_multiclass = None
        
        # Validate query_ids for ranking
        if self.objective == 'rank:ndcg':
            if query_ids is None:
                raise ValueError("query_ids must be provided for ranking objectives")
            if isinstance(query_ids, pd.Series):
                query_ids = query_ids.values
            query_ids = np.array(query_ids)
            if len(query_ids) != len(y):
                raise ValueError("query_ids must have the same length as y")
        
        # Initialize base prediction
        if self.base_score is not None:
            self.base_prediction = self.base_score
        else:
            self.base_prediction = ObjectiveFunctions.get_base_prediction(y, self.objective, self.n_classes)
        
        # Initialize predictions
        if self.objective in ['multi:softmax', 'multi:softprob']:
            # For multiclass: (n_samples, n_classes) predictions
            current_predictions = np.zeros((len(y), self.n_classes), dtype=np.float64)
        else:
            current_predictions = np.full(len(y), self.base_prediction)
        
        # Get gradient and hessian functions
        gradient_func, hessian_func = ObjectiveFunctions.get_gradient_hessian_functions(self.objective)
        
        # Build trees iteratively
        for iteration in range(self.n_estimators):
            # Calculate gradients and hessians
            if self.objective == 'rank:ndcg':
                # For ranking, use lambda gradients
                gradients, hessians = ObjectiveFunctions.compute_lambda_gradients(
                    y, current_predictions, query_ids
                )
            elif self.objective in ['multi:softmax', 'multi:softprob']:
                # For multiclass: build one tree per class per iteration
                for class_idx in range(self.n_classes):
                    # Calculate gradients and hessians for this class
                    gradients = gradient_func(y_multiclass, current_predictions, class_idx)
                    hessians = hessian_func(y_multiclass, current_predictions, class_idx)
                    
                    # Row subsampling
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
                    
                    # Build a new tree for this class
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
                    
                    # Update predictions for this class
                    tree_predictions = tree.predict(X)
                    current_predictions[:, class_idx] += self.learning_rate * tree_predictions
                
                # Skip the normal tree building below for multiclass
                if verbose > 0 and iteration % verbose == 0:
                    # Calculate multiclass log loss
                    probs = ObjectiveFunctions.softmax(current_predictions)
                    probs = np.clip(probs, 1e-7, 1 - 1e-7)
                    loss = -np.mean(np.log(probs[np.arange(len(y)), y_multiclass]))
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, Log Loss: {loss:.6f}")
                
                continue  # Skip normal tree building
            else:
                # For other objectives, use standard gradients
                gradients = gradient_func(y, current_predictions)
                hessians = hessian_func(y, current_predictions)
            
            # Row subsampling
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
            
            # Build a new tree
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
            
            # Update predictions
            # ŷ⁽ᵗ⁾ = ŷ⁽ᵗ⁻¹⁾ + η·fₜ(x)
            tree_predictions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predictions
            
            # if verbose and (iteration + 1) % 10 == 0:
            if verbose > 0 and iteration % verbose == 0:
                # Calculate training loss/metric
                if self.objective == 'reg:squarederror':
                    loss = np.mean((y - current_predictions) ** 2)
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, MSE: {loss:.6f}")
                elif self.objective == 'binary:logistic':
                    # Convert to probabilities
                    probs = 1.0 / (1.0 + np.exp(-current_predictions))
                    # Log loss
                    probs = np.clip(probs, 1e-7, 1 - 1e-7)
                    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, Log Loss: {loss:.6f}")
                elif self.objective == 'rank:ndcg':
                    # Calculate average NDCG across all queries
                    unique_queries = np.unique(query_ids)
                    ndcg_scores = []
                    for qid in unique_queries:
                        query_mask = query_ids == qid
                        ndcg = ObjectiveFunctions.compute_ndcg(
                            y[query_mask],
                            current_predictions[query_mask],
                            k=10
                        )
                        ndcg_scores.append(ndcg)
                    avg_ndcg = np.mean(ndcg_scores)
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, NDCG@10: {avg_ndcg:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray, output_margin: bool = False) -> np.ndarray:
        """
        Make predictions on new data.
        
        Final prediction:
        ŷᵢ = ŷ⁽⁰⁾ + η·Σₖ fₖ(xᵢ)
        
        For multiclass: Returns class labels unless output_margin=True
        
        Line reference: This implements the final prediction formula
        
        Args:
            X: Feature matrix (n_samples, n_features)
            output_margin: If True, output raw margin values (before transformation)
            
        Returns:
            Predictions (n_samples,) or (n_samples, n_classes) for multiclass with output_margin
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.array(X, dtype=np.float64)
        
        # Handle multiclass classification
        if self.objective in ['multi:softmax', 'multi:softprob']:
            # Initialize predictions (n_samples, n_classes)
            predictions = np.zeros((len(X), self.n_classes), dtype=np.float64)
            
            # Add predictions from all trees (n_classes trees per iteration)
            tree_idx = 0
            for _ in range(self.n_estimators):
                for class_idx in range(self.n_classes):
                    if tree_idx < len(self.trees):
                        tree_pred = self.trees[tree_idx].predict(X)
                        predictions[:, class_idx] += self.learning_rate * tree_pred
                        tree_idx += 1
            
            if output_margin:
                return predictions
            
            # Convert to class labels for softmax, probabilities for softprob
            if self.objective == 'multi:softmax':
                return np.argmax(predictions, axis=1)
            else:  # multi:softprob
                return ObjectiveFunctions.softmax(predictions)
        
        # Binary/regression case
        predictions = np.full(len(X), self.base_prediction)
        
        # Add predictions from all trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        # Transform output based on objective
        if not output_margin:
            if self.objective == 'binary:logistic':
                # Convert to probabilities
                predictions = 1.0 / (1.0 + np.exp(-predictions))
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classification only).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Class probabilities:
            - (n_samples, 2) for binary classification
            - (n_samples, n_classes) for multiclass classification
        """
        if self.objective not in ['binary:logistic', 'multi:softmax', 'multi:softprob']:
            raise ValueError("predict_proba is only available for classification")
        
        if self.objective in ['multi:softmax', 'multi:softprob']:
            # Get raw predictions and apply softmax
            raw_predictions = self.predict(X, output_margin=True)
            return ObjectiveFunctions.softmax(raw_predictions)
        else:
            # Binary classification
            probs_class1 = self.predict(X, output_margin=False)
            probs_class0 = 1 - probs_class1
            return np.column_stack([probs_class0, probs_class1])


