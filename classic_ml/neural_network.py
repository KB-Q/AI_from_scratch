# Neural network from scratch (NumPy) — corrected implementation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(
            self, input_size, output_size, 
            hidden_config:list[dict] = [
                {"dim": 16, "activation": "sigmoid", "dropout": 0.0},
                {"dim": 16, "activation": "relu", "dropout": 0.0},
            ],
            lr=0.01, optimizer='sgd',
            batch_size=32, seed=42,
            task='classification' # 'classification' or 'regression'
        ):
        np.random.seed(seed)
        # Initializations (Xavier / He where appropriate)
        self.Wi, self.bi = [], []
        for i in range(len(hidden_config) + 1):
            in_dim = input_size if i == 0 else hidden_config[i - 1]["dim"]
            out_dim = output_size if i == len(hidden_config) else hidden_config[i]["dim"]
            init_scaler = np.sqrt(2.0 if i < len(hidden_config) else 1.0) / np.sqrt(in_dim)
            W = np.random.randn(in_dim, out_dim) * init_scaler
            b = np.zeros((1, out_dim))
            self.Wi.append(W)
            self.bi.append(b)
        
        self.hidden_config = hidden_config
        self.num_layers = len(hidden_config) + 1
        self.task = task
        # Adam optimizer states
        self.mW = [np.zeros_like(W) for W in self.Wi]
        self.vW = [np.zeros_like(W) for W in self.Wi]
        self.mb = [np.zeros_like(b) for b in self.bi]
        self.vb = [np.zeros_like(b) for b in self.bi]
        self.t = 0  # Adam timestep
            
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size

    # Activations
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(activated):
        return activated * (1.0 - activated)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Loss / grads
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-15
        p = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(np.log(p[np.arange(m), y_true])) / m
        return loss

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        m = y_true.shape[0]
        grad = y_pred.copy()
        grad[np.arange(m), y_true] -= 1
        grad = grad / m
        return grad

    @staticmethod
    def mse_loss(y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred)**2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def sgd_update(W, b, dW, db, lr):
        W_new = W - lr * dW
        b_new = b - lr * db
        return W_new, b_new, {}
    
    @staticmethod
    def adam_update(
        W, b, dW, db, lr, mW, vW, mb, vb, t,
        beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        # Adam moment updates (weights)
        mW = beta1 * mW + (1 - beta1) * dW
        vW = beta2 * vW + (1 - beta2) * (dW ** 2)
        mW_hat = mW / (1 - (beta1 ** t))
        vW_hat = vW / (1 - (beta2 ** t))
        
        # Adam moment updates (biases)
        mb = beta1 * mb + (1 - beta1) * db
        vb = beta2 * vb + (1 - beta2) * (db ** 2)
        mb_hat = mb / (1 - (beta1 ** t))
        vb_hat = vb / (1 - (beta2 ** t))
        
        # Parameter updates
        W_new = W - lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
        b_new = b - lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
        
        # Return updated parameters and state
        state = {'mW': mW, 'vW': vW, 'mb': mb, 'vb': vb}
        return W_new, b_new, state
    
    @staticmethod
    def apply_optimizer(optimizer, W, b, dW, db, lr, optimizer_state=None, **kwargs):
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            return NeuralNetwork.sgd_update(W, b, dW, db, lr)
        
        elif optimizer == 'adam':
            if optimizer_state is None:
                raise ValueError("Adam optimizer requires state (mW, vW, mb, vb, t)")
            return NeuralNetwork.adam_update(
                W, b, dW, db, lr,
                optimizer_state['mW'],
                optimizer_state['vW'],
                optimizer_state['mb'],
                optimizer_state['vb'],
                optimizer_state['t'],
                beta1=kwargs.get('beta1', 0.9),
                beta2=kwargs.get('beta2', 0.999),
                epsilon=kwargs.get('epsilon', 1e-8)
            )
        
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Use 'sgd' or 'adam'.")

    # Forward / backward
    def forward(self, X, training=False):
        A = X
        # caches for backprop
        self.Z_cache = []            # pre-activations per layer (including last logits)
        self.A_cache = [A]           # post-activations per layer (A0=X, A1,..., A_{L-1})
        self.dropout_masks = []      # masks for hidden layers only (length L-1)

        activation_mapping = {"sigmoid": self.sigmoid, "relu": self.relu}
        
        for i in range(len(self.Wi)):
            Z = np.dot(A, self.Wi[i]) + self.bi[i]
            self.Z_cache.append(Z)
            if i < len(self.Wi) - 1:
                # Hidden layer activation
                A = activation_mapping[self.hidden_config[i]["activation"]](Z)
                p_drop = float(self.hidden_config[i].get("dropout", 0.0) or 0.0)
                mask = None
                if training and p_drop > 0.0:
                    # Inverted dropout
                    mask = (np.random.rand(*A.shape) > p_drop).astype(float) / (1.0 - p_drop)
                    A = A * mask
                self.dropout_masks.append(mask)
                self.A_cache.append(A)
            else:
                logits = Z
        
        if self.task == 'classification':
            P = self.softmax(logits)
        else:
            P = logits # Linear for regression
            
        return P

    def backward(self, X, y_true, y_pred, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Increment timestep for Adam
        if self.optimizer.lower() == 'adam':
            self.t += 1
        
        # Initial gradient at output layer
        if self.task == 'classification':
            dZ = self.cross_entropy_derivative(y_true, y_pred)
        else:
            dZ = self.mse_derivative(y_true, y_pred)
            
        L = len(self.Wi)
        dA_prev = None # To store gradient w.r.t input

        for i in range(L - 1, -1, -1):
            # Get activation from previous layer (or input X for first layer)
            A_prev = X if i == 0 else self.A_cache[i]
            
            # Compute gradients for current layer
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)

            # Backpropagate to previous layer
            # This calculates dX when i=0, or dA_prev for hidden layers
            dA_prev = np.dot(dZ, self.Wi[i].T)

            if i > 0:
                # Apply dropout mask if it exists
                mask = None
                if hasattr(self, 'dropout_masks') and len(self.dropout_masks) >= i:
                    mask = self.dropout_masks[i - 1]
                if mask is not None:
                    dA_prev = dA_prev * mask

                # Apply activation derivative for previous layer
                act_name = self.hidden_config[i - 1]["activation"]
                if act_name == "sigmoid":
                    A_prev_post = self.A_cache[i]
                    dZ = dA_prev * (A_prev_post * (1.0 - A_prev_post))
                elif act_name == "relu":
                    Z_prev = self.Z_cache[i - 1]
                    dZ = dA_prev * (Z_prev > 0).astype(float)
                else:
                    raise ValueError(f"Unsupported activation: {act_name}")

            # Apply optimizer to update parameters
            optimizer_state = None
            if self.optimizer.lower() == 'adam':
                optimizer_state = {
                    'mW': self.mW[i],
                    'vW': self.vW[i],
                    'mb': self.mb[i],
                    'vb': self.vb[i],
                    't': self.t
                }
            
            W_new, b_new, state = self.apply_optimizer(
                self.optimizer,
                self.Wi[i],
                self.bi[i],
                dW,
                db,
                self.lr,
                optimizer_state=optimizer_state,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon
            )
            
            # Update parameters
            self.Wi[i] = W_new
            self.bi[i] = b_new
            
            # Update Adam state if using Adam optimizer
            if self.optimizer.lower() == 'adam':
                self.mW[i] = state['mW']
                self.vW[i] = state['vW']
                self.mb[i] = state['mb']
                self.vb[i] = state['vb']
        
        return dA_prev # Return gradient w.r.t input X

    # Helpers
    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y, epochs=100, verbose=True, X_val=None, y_val=None):
        m = X.shape[0]
        history = {"loss": [], "val_loss": [], "acc": [], "auc": [], "val_acc": [], "val_auc": []}

        for epoch in range(1, epochs + 1):
            idx = np.arange(m)
            np.random.shuffle(idx)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                y_pred_batch = self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch, y_pred_batch)

            y_pred_all = self.forward(X)
            loss = self.cross_entropy_loss(y, y_pred_all)
            acc = np.mean(self.predict(X) == y)
            auc = metrics.roc_auc_score(y, y_pred_all, multi_class='ovr')
            history["loss"].append(loss)
            history["acc"].append(acc)
            history["auc"].append(auc)

            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(y_val, y_val_pred)
                val_acc = np.mean(self.predict(X_val) == y_val)
                val_auc = metrics.roc_auc_score(y_val, y_val_pred, multi_class='ovr')
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_auc"].append(val_auc)
            else:
                val_loss, val_acc, val_auc = None, None, None

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                msg = f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f} - auc: {auc:.4f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_auc: {val_auc:.4f}"
                print(msg)

        return history


def main():
    n_features, n_classes = 20, 4
    X, y = make_classification(
        n_samples=10000, n_features=n_features, n_informative=3,
        n_redundant=3, n_repeated=10, n_classes=n_classes, 
        n_clusters_per_class=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    nn = NeuralNetwork(
        input_size=n_features,
        output_size=n_classes,
        hidden_config=[
            {"dim": 16, "activation": "sigmoid", "dropout": 0.0},
            {"dim": 16, "activation": "relu", "dropout": 0.0},
        ],
        learning_rate=0.01,
        batch_size=32,
        seed=42,
        optimizer='adam'
    )

    history = nn.fit(X_train, y_train, epochs=60, verbose=True, X_val=X_val, y_val=y_val)

    test_loss = nn.cross_entropy_loss(y_test, nn.predict_proba(X_test))
    test_acc = np.mean(nn.predict(X_test) == y_test)
    test_auc = metrics.roc_auc_score(y_test, nn.predict_proba(X_test), multi_class='ovr')
    print(f"\n Test - loss: {test_loss:.4f} - acc: {test_acc:.4f} - auc: {test_auc:.4f}")

if __name__ == "__main__":
    main()