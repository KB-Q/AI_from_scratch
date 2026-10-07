import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
    Graph Attention Network Layer (Veličković et al., 2018):
    
    Attention mechanism:
    - e_ij = LeakyReLU(a^T · [Wh_i || Wh_j])     (attention coefficients)
    - α_ij = softmax_j(e_ij)                      (normalized attention)
    - h_i' = σ(Σ_{j∈N(i)} α_ij · Wh_j)           (weighted aggregation)
    
    Where:
    - W: Shared linear transformation
    - a: Attention mechanism parameters
    - ||: Concatenation
    - N(i): Neighbors of node i (including self)
    
    Key innovations:
    - Learned attention weights (not pre-defined like GCN)
    - Node-pair specific weighting
    - Supports multi-head attention
    
    Multi-head attention:
    - h_i' = ||_{k=1}^K σ(Σ_j α_ij^k · W^k h_j)  (concatenate heads)
    - h_i' = σ(1/K Σ_k Σ_j α_ij^k · W^k h_j)     (average heads, final layer)
    """
    def __init__(self, input_dim, output_dim, num_heads=1, concat_heads=True, 
                 alpha=0.2, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.alpha = alpha
        self.dropout = dropout
        
        self.W = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_heads)
        ])
        
        self.a_src = nn.ParameterList([
            nn.Parameter(torch.zeros(output_dim, 1)) for _ in range(num_heads)
        ])
        self.a_dst = nn.ParameterList([
            nn.Parameter(torch.zeros(output_dim, 1)) for _ in range(num_heads)
        ])
        
        for k in range(num_heads):
            nn.init.xavier_uniform_(self.a_src[k])
            nn.init.xavier_uniform_(self.a_dst[k])
        
    def forward(self, X, adj):
        """
        Forward pass through GAT layer.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N), non-zero where edges exist
        Returns: Updated node features (N, output_dim * num_heads) if concat, else (N, output_dim)
        """
        N = X.shape[0]
        
        mask = adj + torch.eye(N, device=X.device)
        mask = (mask > 0).float()
        
        head_outputs = []
        
        for k in range(self.num_heads):
            Wh = self.W[k](X)
            
            attn_src = Wh @ self.a_src[k]
            attn_dst = Wh @ self.a_dst[k]
            
            e = attn_src + attn_dst.t()
            e = F.leaky_relu(e, negative_slope=self.alpha)
            
            e_masked = e.masked_fill(mask == 0, float('-inf'))
            
            attn = F.softmax(e_masked, dim=1)
            
            if self.dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout, training=self.training)
            
            h_prime = attn @ Wh
            head_outputs.append(h_prime)
        
        if self.concat_heads:
            out = torch.cat(head_outputs, dim=1)
        else:
            out = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        return out


class GAT(nn.Module):
    """
    Multi-layer Graph Attention Network for node classification.
    
    Architecture:
    - Multiple GAT layers with multi-head attention
    - Concatenate heads in hidden layers, average in output
    - ELU activation between layers
    - Final softmax layer for classification
    
    Pros:
    - Learned attention: Adapts to graph structure
    - Node-specific weights: Different importance per neighbor
    - Multi-head: Stabilizes learning, captures different aspects
    - Inductive: Attention generalizes to new graphs
    
    Cons:
    - O(N²) attention computation (full graph)
    - More parameters than GCN
    - Attention can be unstable without regularization
    
    Loss: Cross-entropy
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2,
                 num_heads=4, dropout=0.6, attn_dropout=0.6):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(GATLayer(input_dim, output_dim, num_heads=1, 
                                       concat_heads=False, dropout=attn_dropout))
        else:
            self.layers.append(GATLayer(input_dim, hidden_dim, num_heads=num_heads,
                                       concat_heads=True, dropout=attn_dropout))
            for _ in range(num_layers - 2):
                self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, 
                                           num_heads=num_heads, concat_heads=True,
                                           dropout=attn_dropout))
            self.layers.append(GATLayer(hidden_dim * num_heads, output_dim,
                                       num_heads=1, concat_heads=False, 
                                       dropout=attn_dropout))
    
    def forward(self, X, adj):
        """
        Forward pass through multi-layer GAT.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        Returns: Node predictions (N, output_dim)
        """
        h = X
        
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, adj)
            h = F.elu(h)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.layers[-1](h, adj)
        return F.log_softmax(h, dim=1)
