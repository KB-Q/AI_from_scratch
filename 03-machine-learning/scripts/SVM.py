import numpy as np
import pandas as pd

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000, 
                 kernel='linear', gamma=0.1, degree=3, coef0=1.0):
        """
        Initialize SVM with gradient descent.
        
        Args:
            learning_rate: Step size for gradient descent
            lambda_param: Regularization parameter (controls margin vs errors trade-off)
            n_iters: Number of training iterations
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
            degree: Degree for 'poly' kernel
            coef0: Independent term in 'poly' and 'sigmoid' kernels
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.w = None
        self.b = None
        # For kernel methods, store support vectors
        self.X_train = None
        self.y_train = None
        self.alpha = None

    def _kernel_function(self, X1, X2):
        """
        Compute kernel matrix between X1 and X2.
        
        Args:
            X1: (n1, d) matrix
            X2: (n2, d) matrix
            
        Returns:
            K: (n1, n2) kernel matrix
        """
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'rbf':
            # RBF (Gaussian) kernel: exp(-gamma * ||x1 - x2||^2)
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * distances_sq)
        
        elif self.kernel == 'poly':
            # Polynomial kernel: (gamma * <x1, x2> + coef0)^degree
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'sigmoid':
            # Sigmoid kernel: tanh(gamma * <x1, x2> + coef0)
            return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y, verbose=False):
        """
        Fit the SVM model using stochastic gradient descent.
        
        For linear kernel: Uses primal formulation
        For non-linear kernels: Uses dual formulation with kernel trick
        
        Minimizes: λ||w||² + (1/n) Σ max(0, 1 - yᵢ(w·xᵢ + b))
        
        This is the hinge loss formulation of SVM.
        """
        n_samples, n_features = X.shape
        
        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        
        if self.kernel == 'linear':
            # Linear kernel: use primal formulation
            self.w = np.zeros(n_features)
            self.b = 0
            self._fit_primal(X, y_, verbose)
        else:
            # Non-linear kernel: use dual formulation
            self.X_train = X
            self.y_train = y_
            self.alpha = np.zeros(n_samples)
            self.b = 0
            self._fit_dual(X, y_, verbose)
    
    def _fit_primal(self, X, y_, verbose):
        """
        Fit using primal formulation (for linear kernel).
        """
        n_samples = X.shape[0]
        
        # Stochastic gradient descent
        for epoch in range(self.n_iters):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_i = X[idx]
                y_i = y_[idx]
                
                # Compute margin: yᵢ(w·xᵢ + b)
                margin = y_i * (np.dot(x_i, self.w) + self.b)
                
                if margin >= 1:
                    # Sample is correctly classified with sufficient margin
                    # Only update with regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Sample violates margin (misclassified or too close)
                    # Update with both regularization and hinge loss gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_i * x_i)
                    self.b -= self.lr * (-y_i)
            
            # Optional: print progress
            if verbose and (epoch + 1) % 100 == 0:
                # Compute hinge loss
                margins = y_ * (np.dot(X, self.w) + self.b)
                hinge_loss = np.mean(np.maximum(0, 1 - margins))
                reg_loss = self.lambda_param * np.dot(self.w, self.w)
                total_loss = reg_loss + hinge_loss
                print(f"Epoch {epoch + 1}/{self.n_iters}, Loss: {total_loss:.4f}")
    
    def _fit_dual(self, X, y_, verbose):
        """
        Fit using dual formulation (for non-linear kernels).
        Uses kernel trick with alpha coefficients.
        """
        n_samples = X.shape[0]
        
        # Precompute kernel matrix
        K = self._kernel_function(X, X)
        
        # Stochastic gradient descent on dual variables
        for epoch in range(self.n_iters):
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                # Compute decision function using kernel: Σ αⱼyⱼK(xⱼ, xᵢ) + b
                decision = np.sum(self.alpha * y_ * K[:, idx]) + self.b
                margin = y_[idx] * decision
                
                if margin >= 1:
                    # Correctly classified with sufficient margin
                    self.alpha[idx] -= self.lr * (2 * self.lambda_param * self.alpha[idx])
                else:
                    # Violates margin
                    self.alpha[idx] -= self.lr * (2 * self.lambda_param * self.alpha[idx] - y_[idx])
                    self.b -= self.lr * (-y_[idx])
            
            # Optional: print progress
            if verbose and (epoch + 1) % 100 == 0:
                decisions = np.array([np.sum(self.alpha * y_ * K[:, i]) + self.b 
                                     for i in range(n_samples)])
                margins = y_ * decisions
                hinge_loss = np.mean(np.maximum(0, 1 - margins))
                reg_loss = self.lambda_param * np.sum(self.alpha ** 2)
                total_loss = reg_loss + hinge_loss
                print(f"Epoch {epoch + 1}/{self.n_iters}, Loss: {total_loss:.4f}")

    def decision_function(self, X):
        """
        Compute the decision function (distance from hyperplane).
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Decision values (n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.array(X, dtype=np.float64)
        
        if self.kernel == 'linear':
            # Linear: w·x + b
            return np.dot(X, self.w) + self.b
        else:
            # Kernel: Σ αᵢyᵢK(xᵢ, x) + b
            K = self._kernel_function(self.X_train, X)
            return np.dot(self.alpha * self.y_train, K) + self.b
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Returns class labels: 1 or -1 (converted to 1 or 0 if needed)
        """
        decision = self.decision_function(X)
        
        # Return sign (-1 or 1), treating 0 as positive
        predictions = np.sign(decision)
        predictions[predictions == 0] = 1  # Handle boundary case
        
        # Convert back to {0, 1} if original labels were binary
        return np.where(predictions == -1, 0, 1)

def example():
    from sklearn.datasets import make_classification, make_circles
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 70)
    print("Linear SVM Example")
    print("=" * 70)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000, kernel='linear')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Linear Kernel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\n" + "=" * 70)
    print("RBF SVM Example (Non-linear Data)")
    print("=" * 70)
    # Create non-linearly separable data
    X_circle, y_circle = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_circle, y_circle, test_size=0.2, random_state=42
    )
    
    # Linear kernel (should perform poorly on circular data)
    model_linear = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000, kernel='linear')
    model_linear.fit(X_train_c, y_train_c)
    y_pred_linear = model_linear.predict(X_test_c)
    print(f"Linear Kernel (on circles): {accuracy_score(y_test_c, y_pred_linear):.4f}")
    
    # RBF kernel (should perform much better)
    model_rbf = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000, 
                    kernel='rbf', gamma=1.0)
    model_rbf.fit(X_train_c, y_train_c)
    y_pred_rbf = model_rbf.predict(X_test_c)
    print(f"RBF Kernel (on circles):    {accuracy_score(y_test_c, y_pred_rbf):.4f}")
    
    # Polynomial kernel
    model_poly = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000,
                     kernel='poly', degree=2, gamma=1.0)
    model_poly.fit(X_train_c, y_train_c)
    y_pred_poly = model_poly.predict(X_test_c)
    print(f"Poly Kernel (on circles):   {accuracy_score(y_test_c, y_pred_poly):.4f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    example()