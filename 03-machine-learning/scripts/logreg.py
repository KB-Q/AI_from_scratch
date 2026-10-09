# %%
import numpy as np

class LogisticRegressionNumpy:
    def __init__(self, lr=0.1, n_iters=1000, l2=0.0, intercept=True, tol=1e-7):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.intercept = intercept
        self.tol = tol
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        # Stable sigmoid
        pos = z >= 0
        neg = ~pos
        out = np.empty_like(z, dtype=float)
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)

        prev_loss = np.inf
        for _ in range(self.n_iters):
            z = np.dot(X, self.w)
            if self.intercept: z += self.b
            p = self._sigmoid(z)

            # Binary cross-entropy with optional L2
            p2 = np.clip(p, 1e-12, 1 - 1e-12)
            loss = (-np.mean(y * np.log(p2) + (1 - y) * np.log(1 - p2)))
            loss += 0.5 * self.l2 * np.sum(self.w ** 2)

            # Gradients
            err = (p - y)  # shape (n,)
            grad_w = (np.dot(X.T, err)) / n + self.l2 * self.w
            if self.intercept: grad_b = np.mean(err)
            else: grad_b = 0.0

            # Update
            self.w -= self.lr * grad_w
            if self.intercept:
                self.b -= self.lr * grad_b

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X):
        z = np.dot(X, self.w)
        if self.intercept: z += self.b
        p = self._sigmoid(z)
        return np.c_[1 - p, p]  # columns: P(class 0), P(class 1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# %%
def example():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegressionNumpy(lr=0.1, n_iters=1000, l2=0.0, intercept=True, tol=1e-7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"AUC: {roc_auc_score(y_test, y_pred)}")

# %%
if __name__ == "__main__":
    example()