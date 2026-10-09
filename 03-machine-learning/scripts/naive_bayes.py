# Simple NaiveBayes implementation using numpy.
import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1.0, variant="multinomial", fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.variant = variant
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.class_probs = None
        self.feature_probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize probability arrays
        self.class_probs = np.zeros(n_classes)
        
        if self.variant == "multinomial":
            self.feature_probs = np.zeros((n_classes, n_features))
        elif self.variant == "gaussian":
            self.feature_means = np.zeros((n_classes, n_features))
            self.feature_vars = np.zeros((n_classes, n_features))
        
        # Calculate class probabilities
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            if self.fit_prior:
                if self.class_prior is not None:
                    self.class_probs[idx] = self.class_prior[idx]
                else:
                    self.class_probs[idx] = X_c.shape[0] / n_samples
            else:
                self.class_probs[idx] = 1.0 / n_classes
            
            # Calculate feature probabilities based on variant
            if self.variant == "multinomial":
                # Laplace smoothing
                self.feature_probs[idx, :] = (X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * n_features)
            elif self.variant == "gaussian":
                self.feature_means[idx, :] = X_c.mean(axis=0)
                self.feature_vars[idx, :] = X_c.var(axis=0)

    def predict(self, X):
        predictions = []
        
        for x in X:
            class_scores = []
            
            for idx, c in enumerate(self.classes):
                # Start with class probability
                score = np.log(self.class_probs[idx])
                
                if self.variant == "multinomial":
                    # Multinomial: sum of log probabilities
                    score += np.sum(x * np.log(self.feature_probs[idx, :]))
                elif self.variant == "gaussian":
                    # Gaussian: sum of log likelihood
                    for j in range(len(x)):
                        mean = self.feature_means[idx, j]
                        var = self.feature_vars[idx, j]
                        score += -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x[j] - mean) ** 2) / var
                
                class_scores.append(score)
            
            # Predict class with highest score
            predictions.append(self.classes[np.argmax(class_scores)])
        
        return np.array(predictions)