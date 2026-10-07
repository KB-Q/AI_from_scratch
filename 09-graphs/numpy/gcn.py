import numpy as np
from utils import relu, relu_derivative, xavier_init, softmax, normalize_adjacency

class GCNLayer:
    """
    Graph Convolutional Network Layer (Kipf & Welling, 2017):
    
    H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
    
    Where:
    - Ã = A + I (adjacency with self-loops)
    - D̃ = degree matrix of Ã
    - H^(l): Node features at layer l
    - W^(l): Learnable weight matrix
    - σ: Non-linear activation (ReLU)
    
    Key insight: Symmetric normalization D̃^(-1/2) Ã D̃^(-1/2) prevents
    scale issues and enables spectral graph convolution approximation.
    
    This implements 1st-order Chebyshev polynomial approximation of
    spectral graph convolutions.
    """
    def __init__(self, input_dim, output_dim, activation=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W = xavier_init((input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        self.adj_norm = None
        
    def forward(self, X, adj_norm):
        """
        Forward pass through GCN layer.
        
        X: Node features (N, input_dim)
        adj_norm: Normalized adjacency matrix D̃^(-1/2) Ã D̃^(-1/2) (N, N)
        Returns: Updated node features (N, output_dim)
        """
        self.X = X
        self.adj_norm = adj_norm
        
        self.agg = adj_norm @ X
        self.z = self.agg @ self.W + self.b
        self.out = relu(self.z) if self.activation else self.z
        return self.out
    
    def backward(self, d_out):
        """
        Backward pass through GCN layer.
        
        d_out: Gradient w.r.t. output (N, output_dim)
        Returns: Gradient w.r.t. input, gradients dict
        """
        dz = d_out * relu_derivative(self.z) if self.activation else d_out
        
        dW = self.agg.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        
        d_agg = dz @ self.W.T
        dX = self.adj_norm.T @ d_agg
        
        grads = {'W': dW, 'b': db}
        return dX, grads


class GCN:
    """
    Multi-layer Graph Convolutional Network for node classification.
    
    Architecture:
    - Preprocessing: Normalize adjacency matrix once
    - Multiple GCN layers with spectral convolutions
    - Final softmax layer for classification
    
    Pros:
    - Efficient: O(|E|) complexity per layer
    - Principled: Spectral graph theory foundation
    - Simple: Few hyperparameters
    
    Cons:
    - Transductive: Needs full graph at training
    - Fixed receptive field per layer
    - Over-smoothing with many layers
    
    Loss: Cross-entropy
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = []
        
        if num_layers == 1:
            self.layers.append(GCNLayer(input_dim, output_dim, activation=False))
        else:
            self.layers.append(GCNLayer(input_dim, hidden_dim, activation=True))
            for _ in range(num_layers - 2):
                self.layers.append(GCNLayer(hidden_dim, hidden_dim, activation=True))
            self.layers.append(GCNLayer(hidden_dim, output_dim, activation=False))
        
        self.adj_norm = None
    
    def preprocess(self, adj):
        """Compute normalized adjacency matrix once."""
        self.adj_norm = normalize_adjacency(adj)
    
    def forward(self, X, adj=None, training=True):
        """
        Forward pass through multi-layer GCN.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N) - only needed if not preprocessed
        training: Whether in training mode (for dropout)
        Returns: Node predictions (N, output_dim)
        """
        if adj is not None:
            self.preprocess(adj)
        
        self.hidden_outputs = []
        self.dropout_masks = []
        h = X
        
        if training and self.dropout > 0:
            mask = (np.random.rand(*h.shape) > self.dropout).astype(float)
            mask /= (1 - self.dropout)
            h = h * mask
            self.input_dropout_mask = mask
        else:
            self.input_dropout_mask = None
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer.forward(h, self.adj_norm)
            self.hidden_outputs.append(h)
            
            if training and self.dropout > 0:
                mask = (np.random.rand(*h.shape) > self.dropout).astype(float)
                mask /= (1 - self.dropout)
                h = h * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
        
        h = self.layers[-1].forward(h, self.adj_norm)
        self.logits = h
        self.probs = softmax(h, axis=1)
        return self.probs
    
    def backward(self, targets, mask=None):
        """
        Backward pass with cross-entropy loss.
        
        targets: One-hot encoded targets (N, output_dim)
        mask: Boolean mask for which nodes to compute loss on (train nodes)
        Returns: Loss value
        """
        if mask is None:
            mask = np.ones(targets.shape[0], dtype=bool)
        
        d_logits = self.probs.copy()
        d_logits[mask] -= targets[mask]
        d_logits[~mask] = 0
        d_logits /= np.sum(mask)
        
        loss = -np.sum(targets[mask] * np.log(self.probs[mask] + 1e-10)) / np.sum(mask)
        
        self.grads = []
        d_h = d_logits
        
        for i in range(self.num_layers - 1, -1, -1):
            d_h, grads = self.layers[i].backward(d_h)
            self.grads.insert(0, grads)
            
            if i > 0 and self.dropout_masks[i-1] is not None:
                d_h = d_h * self.dropout_masks[i-1]
        
        return loss
    
    def update(self, learning_rate=0.01):
        """Update parameters using accumulated gradients."""
        for i, layer in enumerate(self.layers):
            for k, v in self.grads[i].items():
                np.clip(v, -5, 5, out=v)
                param = getattr(layer, k)
                param -= learning_rate * v
