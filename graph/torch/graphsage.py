import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.sample_size = sample_size
        self.activation = activation
        
        self.W = nn.Linear(2 * input_dim, output_dim, bias=True)
        
    def sample_neighbors(self, adj, node_idx):
        """Sample fixed number of neighbors for a node."""
        neighbors = torch.where(adj[node_idx] > 0)[0]
        if len(neighbors) == 0:
            return torch.tensor([node_idx], device=adj.device)
        if len(neighbors) <= self.sample_size:
            return neighbors
        perm = torch.randperm(len(neighbors), device=adj.device)[:self.sample_size]
        return neighbors[perm]
    
    def aggregate_mean(self, X, adj, sampled=False):
        """
        Mean aggregation over neighbors.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        sampled: Whether to use neighbor sampling
        Returns: Aggregated neighbor features (N, input_dim)
        """
        N = X.shape[0]
        device = X.device
        
        if sampled and self.training:
            h_neigh = torch.zeros(N, self.input_dim, device=device)
            for v in range(N):
                neighbors = self.sample_neighbors(adj, v)
                if len(neighbors) > 0:
                    h_neigh[v] = X[neighbors].mean(dim=0)
        else:
            adj_with_self = adj + torch.eye(N, device=device)
            degree = adj_with_self.sum(dim=1, keepdim=True).clamp(min=1)
            h_neigh = adj_with_self @ X / degree
        
        return h_neigh
    
    def aggregate_maxpool(self, X, adj):
        """
        Max pooling aggregation over neighbors.
        
        X: Node features (N, input_dim)  
        adj: Adjacency matrix (N, N)
        Returns: Aggregated neighbor features (N, input_dim)
        """
        N = X.shape[0]
        device = X.device
        h_neigh = torch.zeros(N, self.input_dim, device=device)
        
        for v in range(N):
            neighbors = torch.where(adj[v] > 0)[0]
            if len(neighbors) > 0:
                h_neigh[v] = X[neighbors].max(dim=0)[0]
            else:
                h_neigh[v] = X[v]
        
        return h_neigh
    
    def forward(self, X, adj):
        """
        Forward pass through GraphSAGE layer.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        Returns: Updated node features (N, output_dim)
        """
        if self.aggregator == 'mean':
            h_neigh = self.aggregate_mean(X, adj, sampled=self.training)
        elif self.aggregator == 'maxpool':
            h_neigh = self.aggregate_maxpool(X, adj)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        concat = torch.cat([X, h_neigh], dim=1)
        z = self.W(concat)
        
        if self.activation:
            out = F.relu(z)
            out = F.normalize(out, p=2, dim=1)
        else:
            out = z
        
        return out


class GraphSAGE(nn.Module):
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
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(GraphSAGELayer(input_dim, output_dim, aggregator, sample_size, activation=False))
        else:
            self.layers.append(GraphSAGELayer(input_dim, hidden_dim, aggregator, sample_size, activation=True))
            for _ in range(num_layers - 2):
                self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator, sample_size, activation=True))
            self.layers.append(GraphSAGELayer(hidden_dim, output_dim, aggregator, sample_size, activation=False))
    
    def forward(self, X, adj):
        """
        Forward pass through multi-layer GraphSAGE.
        
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
