import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W_msg = nn.Linear(input_dim, output_dim, bias=False)  # Message transformation
        self.W_self = nn.Linear(input_dim, output_dim, bias=False)  # Self-loop transformation
        self.b = nn.Parameter(torch.zeros(output_dim))  # Bias
        
    def forward(self, X, adj):
        """
        Forward pass through GNN layer.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        Returns: Updated node features (N, output_dim)
        """
        msg_agg = adj @ self.W_msg(X)
        self_transform = self.W_self(X)
        z = msg_agg + self_transform + self.b
        out = F.relu(z) if self.activation else z
        return out


class GNN(nn.Module):
    """
    Multi-layer Graph Neural Network for node classification.
    
    Architecture:
    - Multiple GNN layers with message passing
    - Final softmax layer for classification
    
    Loss: Cross-entropy
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(GNNLayer(input_dim, output_dim, activation=False))
        else:
            self.layers.append(GNNLayer(input_dim, hidden_dim, activation=True))
            for _ in range(num_layers - 2):
                self.layers.append(GNNLayer(hidden_dim, hidden_dim, activation=True))
            self.layers.append(GNNLayer(hidden_dim, output_dim, activation=False))
    
    def forward(self, X, adj):
        """
        Forward pass through multi-layer GNN.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        Returns: Node predictions (N, output_dim)
        """
        h = X
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, adj)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.layers[-1](h, adj)
        return F.log_softmax(h, dim=1)
