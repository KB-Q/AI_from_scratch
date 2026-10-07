import torch
import torch.nn.functional as F

def get_device():
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def normalize_adjacency(adj):
    """
    Symmetric normalization: D^(-1/2) A D^(-1/2)
    
    adj: Adjacency matrix (N, N) - torch.Tensor
    Returns: Normalized adjacency matrix
    """
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def row_normalize_adjacency(adj):
    """
    Row normalization: D^(-1) A
    
    adj: Adjacency matrix (N, N) - torch.Tensor
    Returns: Row-normalized adjacency matrix
    """
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    degree = adj.sum(dim=1, keepdim=True)
    return adj / degree.clamp(min=1e-10)

def sparse_normalize_adjacency(edge_index, num_nodes):
    """
    Symmetric normalization for sparse adjacency (edge_index format).
    
    edge_index: (2, E) tensor of edge indices
    num_nodes: Number of nodes
    Returns: Normalized edge weights
    """
    row, col = edge_index
    
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    edge_index = torch.cat([edge_index, torch.stack([self_loops, self_loops])], dim=1)
    row, col = edge_index
    
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(row.shape[0], device=edge_index.device))
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm

def accuracy(pred, target):
    """Classification accuracy."""
    pred_labels = pred.argmax(dim=1)
    true_labels = target.argmax(dim=1) if target.dim() > 1 else target
    return (pred_labels == true_labels).float().mean().item()

def edge_index_to_adj(edge_index, num_nodes):
    """Convert edge_index to dense adjacency matrix."""
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def adj_to_edge_index(adj):
    """Convert dense adjacency matrix to edge_index."""
    return adj.nonzero().t()
