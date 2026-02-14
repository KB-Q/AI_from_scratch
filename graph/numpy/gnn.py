import numpy as np
from utils import relu, relu_derivative, xavier_init, softmax

class GNNLayer:
    """
    Basic Graph Neural Network Layer (Message Passing):
    
    Message Passing:
    - m_v = Σ_{u∈N(v)} W_msg · h_u          (aggregate neighbor messages)
    - h_v' = σ(W_self · h_v + m_v + b)      (update node representation)
    
    h_v: Node feature for node v
    N(v): Neighbors of node v  
    W_msg: Message transformation weights
    W_self: Self-loop transformation weights
    b: Bias
    σ: Non-linear activation (ReLU)
    
    This is the simplest form of GNN that learns to aggregate neighbor information.
    """
    def __init__(self, input_dim, output_dim, activation=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W_msg = xavier_init((input_dim, output_dim))
        self.W_self = xavier_init((input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        
    def forward(self, X, adj):
        """
        Forward pass through GNN layer.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        Returns: Updated node features (N, output_dim)
        """
        self.X = X
        self.adj = adj
        
        self.msg_agg = adj @ X @ self.W_msg
        self.self_transform = X @ self.W_self
        self.z = self.msg_agg + self.self_transform + self.b
        self.out = relu(self.z) if self.activation else self.z
        return self.out
    
    def backward(self, d_out):
        """
        Backward pass through GNN layer.
        
        d_out: Gradient w.r.t. output (N, output_dim)
        Returns: Gradient w.r.t. input, gradients dict
        """
        dz = d_out * relu_derivative(self.z) if self.activation else d_out
        
        dW_msg = self.X.T @ (self.adj.T @ dz)
        dW_self = self.X.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        
        dX_msg = self.adj.T @ dz @ self.W_msg.T
        dX_self = dz @ self.W_self.T
        dX = dX_msg + dX_self
        
        grads = {'W_msg': dW_msg, 'W_self': dW_self, 'b': db}
        return dX, grads


class GNN:
    """
    Multi-layer Graph Neural Network for node classification.
    
    Architecture:
    - Multiple GNN layers with message passing
    - Final softmax layer for classification
    
    Loss: Cross-entropy
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = []
        
        if num_layers == 1:
            self.layers.append(GNNLayer(input_dim, output_dim, activation=False))
        else:
            self.layers.append(GNNLayer(input_dim, hidden_dim, activation=True))
            for _ in range(num_layers - 2):
                self.layers.append(GNNLayer(hidden_dim, hidden_dim, activation=True))
            self.layers.append(GNNLayer(hidden_dim, output_dim, activation=False))
    
    def forward(self, X, adj, training=True):
        """
        Forward pass through multi-layer GNN.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        training: Whether in training mode (for dropout)
        Returns: Node predictions (N, output_dim)
        """
        self.hidden_outputs = []
        self.dropout_masks = []
        h = X
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer.forward(h, adj)
            self.hidden_outputs.append(h)
            
            if training and self.dropout > 0:
                mask = (np.random.rand(*h.shape) > self.dropout).astype(float)
                mask /= (1 - self.dropout)
                h = h * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
        
        h = self.layers[-1].forward(h, adj)
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
