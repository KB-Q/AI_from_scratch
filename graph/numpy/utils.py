import numpy as np

def sigmoid(x):
    """σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    """ReLU(x) = max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU'(x) = 1 if x > 0 else 0"""
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    """LeakyReLU(x) = max(αx, x)"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """LeakyReLU'(x) = 1 if x > 0 else α"""
    return np.where(x > 0, 1.0, alpha)

def softmax(x, axis=-1):
    """softmax(x_i) = e^(x_i) / Σ e^(x_j)"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def xavier_init(shape):
    """Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))"""
    return np.random.randn(*shape) * np.sqrt(2.0 / sum(shape))

def normalize_adjacency(adj):
    """
    Symmetric normalization: D^(-1/2) A D^(-1/2)
    
    adj: Adjacency matrix (N, N)
    Returns: Normalized adjacency matrix
    """
    adj = adj + np.eye(adj.shape[0])
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def row_normalize_adjacency(adj):
    """
    Row normalization: D^(-1) A
    
    adj: Adjacency matrix (N, N)
    Returns: Row-normalized adjacency matrix
    """
    adj = adj + np.eye(adj.shape[0])
    degree = np.sum(adj, axis=1, keepdims=True)
    return adj / np.maximum(degree, 1e-10)

def compute_degree_matrix(adj):
    """Compute degree matrix from adjacency matrix."""
    return np.diag(np.sum(adj, axis=1))

def get_neighbors(adj, node_idx):
    """Get neighbor indices for a given node."""
    return np.where(adj[node_idx] > 0)[0]

def cross_entropy_loss(pred, target):
    """Cross-entropy loss for classification."""
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    return -np.sum(target * np.log(pred)) / target.shape[0]

def accuracy(pred, target):
    """Classification accuracy."""
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(target, axis=1) if target.ndim > 1 else target
    return np.mean(pred_labels == true_labels)
