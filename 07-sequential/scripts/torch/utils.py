import torch
import torch.nn.functional as F

def xavier_init(shape, device='cpu'):
    """Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))"""
    tensor = torch.empty(shape, device=device)
    torch.nn.init.xavier_uniform_(tensor)
    return tensor

def get_device():
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
