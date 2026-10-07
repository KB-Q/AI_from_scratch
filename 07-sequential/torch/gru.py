import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn import SimpleFFN

class GRUCell(nn.Module):
    """
    GRU Cell:
    - z_t = σ(W_z·[h_{t-1}, x_t] + b_z)
    - r_t = σ(W_r·[h_{t-1}, x_t] + b_r)
    - h̃_t = tanh(W_h·[r_t ⊙ h_{t-1}, x_t] + b_h)
    - h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

    x_t: Input at time t
    [h_{t-1}, x_t]: Concat of hidden state and input
    z_t: Update gate at time t
    r_t: Reset gate at time t
    h̃_t: Hidden candidate at time t
    h_t: Hidden state at time t

    W_z: Hidden-to-update gate weights
    W_r: Hidden-to-reset gate weights
    W_h: Hidden-to-candidate weights
    b_z: Update gate bias
    b_r: Reset gate bias
    b_h: Hidden candidate bias
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        concat_len = input_size + hidden_size
        self.W_z = nn.Linear(concat_len, hidden_size)  # Update gate
        self.W_r = nn.Linear(concat_len, hidden_size)  # Reset gate
        self.W_h = nn.Linear(concat_len, hidden_size)  # Candidate hidden state

    def forward(self, x, h_prev):
        """Forward pass through GRU cell."""
        concat = torch.cat([h_prev, x], dim=-1)
        z = torch.sigmoid(self.W_z(concat))
        r = torch.sigmoid(self.W_r(concat))
        h_reset_concat = torch.cat([r * h_prev, x], dim=-1)
        h_tilde = torch.tanh(self.W_h(h_reset_concat))
        h = (1 - z) * h_prev + z * h_tilde
        return h


class GRU(nn.Module):
    """
    GRU with optional FFN output layer.
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_ffn = use_ffn
        self.cell = GRUCell(input_size, hidden_size)
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)
        else:
            self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_init=None):
        """
        Forward pass through GRU sequence.
        
        x: Input sequence (batch_size, seq_len, input_size)
        h_init: Initial hidden state (batch_size, hidden_size)
        Returns: outputs (batch_size, seq_len, output_size), final hidden state
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if h_init is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h = h_init
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.cell(x_t, h)
            
            if self.use_ffn:
                out = self.ffn(h)
            else:
                out = self.W_hy(h)
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h


class BidirectionalGRU(nn.Module):
    """
    Bidirectional GRU: y_t = FFN([h_fwd_t, h_bwd_t])
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fwd_gru = GRU(input_size, hidden_size, hidden_size, use_ffn=False)
        self.bwd_gru = GRU(input_size, hidden_size, hidden_size, use_ffn=False)
        
        self.ffn = SimpleFFN(hidden_size * 2, hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through bidirectional GRU.
        
        x: Input sequence (batch_size, seq_len, input_size)
        Returns: outputs (batch_size, seq_len, output_size)
        """
        fwd_out, _ = self.fwd_gru(x)
        bwd_out, _ = self.bwd_gru(torch.flip(x, dims=[1]))
        bwd_out = torch.flip(bwd_out, dims=[1])
        
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        
        batch_size, seq_len, _ = combined.shape
        combined_flat = combined.view(batch_size * seq_len, -1)
        outputs = self.ffn(combined_flat)
        outputs = outputs.view(batch_size, seq_len, -1)
        
        return outputs


class NativeGRU(nn.Module):
    """
    GRU using PyTorch's native nn.GRU for fast training.
    Same API as the scratch GRU class but uses optimized CUDA/MPS kernels.
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_ffn = use_ffn
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)
        else:
            self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_init=None):
        """
        Forward pass through GRU sequence.
        
        x: Input sequence (batch_size, seq_len, input_size)
        h_init: Initial hidden state (num_layers, batch_size, hidden_size)
        Returns: outputs (batch_size, seq_len, output_size), final hidden state
        """
        batch_size = x.shape[0]
        device = x.device
        
        if h_init is None:
            h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        gru_out, h_n = self.gru(x, h_init)
        
        if self.use_ffn:
            batch_size, seq_len, _ = gru_out.shape
            gru_out_flat = gru_out.reshape(batch_size * seq_len, -1)
            outputs = self.ffn(gru_out_flat).view(batch_size, seq_len, -1)
        else:
            outputs = self.W_hy(gru_out)
        
        return outputs, h_n[-1]


class NativeBidirectionalGRU(nn.Module):
    """
    Bidirectional GRU using PyTorch's native nn.GRU for fast training.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                         batch_first=True, bidirectional=True,
                         dropout=dropout if num_layers > 1 else 0.0)
        
        self.ffn = SimpleFFN(hidden_size * 2, hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through bidirectional GRU.
        
        x: Input sequence (batch_size, seq_len, input_size)
        Returns: outputs (batch_size, seq_len, output_size)
        """
        gru_out, _ = self.gru(x)
        
        batch_size, seq_len, _ = gru_out.shape
        gru_out_flat = gru_out.reshape(batch_size * seq_len, -1)
        outputs = self.ffn(gru_out_flat).view(batch_size, seq_len, -1)
        
        return outputs
