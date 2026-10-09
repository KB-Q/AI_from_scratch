import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalize_adjacency

class GCNLayer(nn.Module):
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W = nn.Linear(input_dim, output_dim, bias=True)
        
    def forward(self, X, adj_norm):
        """
        Forward pass through GCN layer.
        
        X: Node features (N, input_dim)
        adj_norm: Normalized adjacency matrix D̃^(-1/2) Ã D̃^(-1/2) (N, N)
        Returns: Updated node features (N, output_dim)
        """
        agg = adj_norm @ X
        z = self.W(agg)
        out = F.relu(z) if self.activation else z
        return out


class GCN(nn.Module):
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
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
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
    
    def forward(self, X, adj=None):
        """
        Forward pass through multi-layer GCN.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N) - only needed if not preprocessed
        Returns: Node predictions (N, output_dim)
        """
        if adj is not None:
            self.preprocess(adj)
        
        h = X
        
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, self.adj_norm)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.layers[-1](h, self.adj_norm)
        return F.log_softmax(h, dim=1)
