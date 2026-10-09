import numpy as np
from utils import relu, relu_derivative, xavier_init, softmax

class GraphSAGELayer:
    """
    GraphSAGE Layer (Hamilton et al., 2017):
    
    Sample & Aggregate Framework:
    1. Sample: Sample fixed number of neighbors
    2. Aggregate: h_N(v) = AGGREGATE({h_u : u ∈ N(v)})  
    3. Update: h_v' = σ(W · CONCAT(h_v, h_N(v)))
    
    Aggregation methods:
    - Mean: h_N(v) = mean({h_u : u ∈ N(v)})
    - Max Pool: h_N(v) = max(σ(W_pool · h_u + b))
    - LSTM: h_N(v) = LSTM({h_u : u ∈ N(v)}) [order-invariant via permutation]
    
    Key differences from GCN:
    - Inductive: Can generalize to unseen nodes
    - Sampling: Fixed computation budget
    - Concatenation: Explicit self-representation
    
    This implementation uses Mean aggregation for simplicity.
    """
    def __init__(self, input_dim, output_dim, aggregator='mean', sample_size=10, activation=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.sample_size = sample_size
        self.activation = activation
        
        self.W = xavier_init((2 * input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        
    def sample_neighbors(self, adj, node_idx):
        """Sample fixed number of neighbors for a node."""
        neighbors = np.where(adj[node_idx] > 0)[0]
        if len(neighbors) == 0:
            return np.array([node_idx])
        if len(neighbors) <= self.sample_size:
            return neighbors
        return np.random.choice(neighbors, self.sample_size, replace=False)
    
    def aggregate_mean(self, X, adj, sampled=False):
        """
        Mean aggregation over neighbors.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        sampled: Whether to use neighbor sampling
        Returns: Aggregated neighbor features (N, input_dim)
        """
        N = X.shape[0]
        h_neigh = np.zeros((N, self.input_dim))
        
        if sampled:
            for v in range(N):
                neighbors = self.sample_neighbors(adj, v)
                if len(neighbors) > 0:
                    h_neigh[v] = np.mean(X[neighbors], axis=0)
        else:
            adj_with_self = adj + np.eye(N)
            degree = np.sum(adj_with_self, axis=1, keepdims=True)
            h_neigh = adj_with_self @ X / np.maximum(degree, 1)
        
        return h_neigh
    
    def aggregate_maxpool(self, X, adj):
        """
        Max pooling aggregation over neighbors.
        
        X: Node features (N, input_dim)  
        adj: Adjacency matrix (N, N)
        Returns: Aggregated neighbor features (N, input_dim)
        """
        N = X.shape[0]
        h_neigh = np.zeros((N, self.input_dim))
        
        for v in range(N):
            neighbors = np.where(adj[v] > 0)[0]
            if len(neighbors) > 0:
                h_neigh[v] = np.max(X[neighbors], axis=0)
            else:
                h_neigh[v] = X[v]
        
        return h_neigh
    
    def forward(self, X, adj, training=True):
        """
        Forward pass through GraphSAGE layer.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        training: Whether in training mode (affects sampling)
        Returns: Updated node features (N, output_dim)
        """
        self.X = X
        self.adj = adj
        
        if self.aggregator == 'mean':
            self.h_neigh = self.aggregate_mean(X, adj, sampled=training)
        elif self.aggregator == 'maxpool':
            self.h_neigh = self.aggregate_maxpool(X, adj)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        self.concat = np.concatenate([X, self.h_neigh], axis=1)
        self.z = self.concat @ self.W + self.b
        
        if self.activation:
            self.out = relu(self.z)
            self.out = self.out / (np.linalg.norm(self.out, axis=1, keepdims=True) + 1e-10)
        else:
            self.out = self.z
        
        return self.out
    
    def backward(self, d_out):
        """
        Backward pass through GraphSAGE layer.
        
        d_out: Gradient w.r.t. output (N, output_dim)
        Returns: Gradient w.r.t. input, gradients dict
        """
        if self.activation:
            out_norm = np.linalg.norm(self.out * (np.linalg.norm(self.out, axis=1, keepdims=True) + 1e-10), axis=1, keepdims=True) + 1e-10
            d_out_unnorm = d_out / (out_norm + 1e-10)
            dz = d_out_unnorm * relu_derivative(self.z)
        else:
            dz = d_out
        
        dW = self.concat.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        
        d_concat = dz @ self.W.T
        dX_self = d_concat[:, :self.input_dim]
        d_h_neigh = d_concat[:, self.input_dim:]
        
        N = self.X.shape[0]
        dX_neigh = np.zeros_like(self.X)
        
        adj_with_self = self.adj + np.eye(N)
        degree = np.sum(adj_with_self, axis=1, keepdims=True)
        dX_neigh = adj_with_self.T @ (d_h_neigh / np.maximum(degree, 1))
        
        dX = dX_self + dX_neigh
        
        grads = {'W': dW, 'b': db}
        return dX, grads


class GraphSAGE:
    """
    Multi-layer GraphSAGE for node classification.
    
    Architecture:
    - Multiple GraphSAGE layers with neighbor sampling
    - L2 normalization after each layer
    - Final softmax layer for classification
    
    Pros:
    - Inductive: Works on unseen nodes/graphs
    - Scalable: Fixed computation via sampling
    - Flexible: Multiple aggregation strategies
    
    Cons:
    - Sampling variance in training
    - Information loss from sampling
    - Hyperparameter: sample size selection
    
    Loss: Cross-entropy
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 aggregator='mean', sample_size=10, dropout=0.0):
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = []
        
        if num_layers == 1:
            self.layers.append(GraphSAGELayer(input_dim, output_dim, aggregator, sample_size, activation=False))
        else:
            self.layers.append(GraphSAGELayer(input_dim, hidden_dim, aggregator, sample_size, activation=True))
            for _ in range(num_layers - 2):
                self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator, sample_size, activation=True))
            self.layers.append(GraphSAGELayer(hidden_dim, output_dim, aggregator, sample_size, activation=False))
    
    def forward(self, X, adj, training=True):
        """
        Forward pass through multi-layer GraphSAGE.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        training: Whether in training mode
        Returns: Node predictions (N, output_dim)
        """
        self.hidden_outputs = []
        self.dropout_masks = []
        h = X
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer.forward(h, adj, training=training)
            self.hidden_outputs.append(h)
            
            if training and self.dropout > 0:
                mask = (np.random.rand(*h.shape) > self.dropout).astype(float)
                mask /= (1 - self.dropout)
                h = h * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
        
        h = self.layers[-1].forward(h, adj, training=training)
        self.logits = h
        self.probs = softmax(h, axis=1)
        return self.probs
    
    def backward(self, targets, mask=None):
        """
        Backward pass with cross-entropy loss.
        
        targets: One-hot encoded targets (N, output_dim)
        mask: Boolean mask for which nodes to compute loss on
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
