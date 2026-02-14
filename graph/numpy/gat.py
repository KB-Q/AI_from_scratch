import numpy as np
from utils import relu, relu_derivative, leaky_relu, leaky_relu_derivative, xavier_init, softmax

class GATLayer:
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.alpha = alpha
        self.dropout = dropout
        
        self.W = [xavier_init((input_dim, output_dim)) for _ in range(num_heads)]
        
        self.a_src = [xavier_init((output_dim, 1)) for _ in range(num_heads)]
        self.a_dst = [xavier_init((output_dim, 1)) for _ in range(num_heads)]
        
    def forward(self, X, adj, training=True):
        """
        Forward pass through GAT layer.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N), non-zero where edges exist
        training: Whether in training mode
        Returns: Updated node features (N, output_dim * num_heads) if concat, else (N, output_dim)
        """
        self.X = X
        self.adj = adj
        N = X.shape[0]
        
        mask = adj + np.eye(N)
        mask = (mask > 0).astype(float)
        
        self.head_outputs = []
        self.attention_weights = []
        self.Wh_list = []
        self.e_list = []
        
        for k in range(self.num_heads):
            Wh = X @ self.W[k]
            self.Wh_list.append(Wh)
            
            attn_src = Wh @ self.a_src[k]
            attn_dst = Wh @ self.a_dst[k]
            
            e = attn_src + attn_dst.T
            e = leaky_relu(e, self.alpha)
            self.e_list.append(e)
            
            e_masked = np.where(mask > 0, e, -1e10)
            
            attn = softmax(e_masked, axis=1)
            self.attention_weights.append(attn)
            
            if training and self.dropout > 0:
                drop_mask = (np.random.rand(*attn.shape) > self.dropout).astype(float)
                attn = attn * drop_mask
                row_sum = np.sum(attn, axis=1, keepdims=True)
                attn = attn / (row_sum + 1e-10)
            
            h_prime = attn @ Wh
            self.head_outputs.append(h_prime)
        
        if self.concat_heads:
            self.out = np.concatenate(self.head_outputs, axis=1)
        else:
            self.out = np.mean(self.head_outputs, axis=0)
        
        return self.out
    
    def backward(self, d_out):
        """
        Backward pass through GAT layer.
        
        d_out: Gradient w.r.t. output
        Returns: Gradient w.r.t. input, gradients dict
        """
        N = self.X.shape[0]
        
        if self.concat_heads:
            d_heads = np.split(d_out, self.num_heads, axis=1)
        else:
            d_heads = [d_out / self.num_heads] * self.num_heads
        
        dX = np.zeros_like(self.X)
        grads = {
            'W': [np.zeros_like(w) for w in self.W],
            'a_src': [np.zeros_like(a) for a in self.a_src],
            'a_dst': [np.zeros_like(a) for a in self.a_dst]
        }
        
        for k in range(self.num_heads):
            d_h_prime = d_heads[k]
            attn = self.attention_weights[k]
            Wh = self.Wh_list[k]
            e = self.e_list[k]
            
            d_attn = d_h_prime @ Wh.T
            d_Wh_from_agg = attn.T @ d_h_prime
            
            d_e = attn * (d_attn - np.sum(d_attn * attn, axis=1, keepdims=True))
            
            d_e = d_e * leaky_relu_derivative(e, self.alpha)
            
            d_attn_src = np.sum(d_e, axis=1, keepdims=True)
            d_attn_dst = np.sum(d_e, axis=0, keepdims=True).T
            
            grads['a_src'][k] = Wh.T @ d_attn_src
            grads['a_dst'][k] = Wh.T @ d_attn_dst
            
            d_Wh_from_attn = d_attn_src @ self.a_src[k].T + d_attn_dst @ self.a_dst[k].T
            d_Wh = d_Wh_from_agg + d_Wh_from_attn
            
            grads['W'][k] = self.X.T @ d_Wh
            
            dX += d_Wh @ self.W[k].T
        
        return dX, grads


class GAT:
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
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = []
        
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
    
    def elu(self, x, alpha=1.0):
        """ELU activation: x if x > 0, else α(e^x - 1)"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def elu_derivative(self, x, alpha=1.0):
        """ELU derivative: 1 if x > 0, else α*e^x"""
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    def forward(self, X, adj, training=True):
        """
        Forward pass through multi-layer GAT.
        
        X: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        training: Whether in training mode
        Returns: Node predictions (N, output_dim)
        """
        self.hidden_outputs = []
        self.pre_activations = []
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
            h_pre = layer.forward(h, adj, training=training)
            self.pre_activations.append(h_pre)
            h = self.elu(h_pre)
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
            
            if i > 0:
                if self.dropout_masks[i-1] is not None:
                    d_h = d_h * self.dropout_masks[i-1]
                d_h = d_h * self.elu_derivative(self.pre_activations[i-1])
        
        return loss
    
    def update(self, learning_rate=0.005):
        """Update parameters using accumulated gradients."""
        for i, layer in enumerate(self.layers):
            for k in range(layer.num_heads):
                np.clip(self.grads[i]['W'][k], -1, 1, out=self.grads[i]['W'][k])
                layer.W[k] -= learning_rate * self.grads[i]['W'][k]
                
                np.clip(self.grads[i]['a_src'][k], -1, 1, out=self.grads[i]['a_src'][k])
                layer.a_src[k] -= learning_rate * self.grads[i]['a_src'][k]
                
                np.clip(self.grads[i]['a_dst'][k], -1, 1, out=self.grads[i]['a_dst'][k])
                layer.a_dst[k] -= learning_rate * self.grads[i]['a_dst'][k]
