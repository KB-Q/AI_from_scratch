import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFFN(nn.Module):
    """Simple Feed-Forward Network: y = W2·ReLU(W1·x + b1) + b2"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class RNNCell(nn.Module):
    """
    RNN Cell:
    - h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)
    
    x_t: Input at time t
    h_{t-1}: Hidden state at previous timestep
    h_t: Hidden state at time t
    
    W_xh: Input-to-hidden weights
    W_hh: Hidden-to-hidden weights
    b_h: Hidden bias
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size  # Size of hidden state vector
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)  # Input-to-hidden weights
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)  # Hidden-to-hidden weights
        self.b_h = nn.Parameter(torch.zeros(hidden_size))  # Hidden bias
    
    def forward(self, x, h_prev):
        """Forward pass through RNN cell."""
        h = torch.tanh(self.W_xh(x) + self.W_hh(h_prev) + self.b_h)
        return h


class RNN(nn.Module):
    """
    Simple RNN with optional FFN output layer.
    
    h_t: Hidden state at time t
    y_t: Output at time t
    
    W_hy: Hidden-to-output weights (linear mode)
    b_y: Output bias (linear mode)
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False):
        super().__init__()
        self.input_size = input_size  # Dimensionality of input features
        self.hidden_size = hidden_size  # Size of RNN hidden state
        self.output_size = output_size  # Dimensionality of output
        self.use_ffn = use_ffn  # Whether to use FFN for output layer
        self.cell = RNNCell(input_size, hidden_size)  # RNN cell for sequence processing
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)  # FFN output layer
        else:
            self.W_hy = nn.Linear(hidden_size, output_size)  # Hidden-to-output projection

    def forward(self, x, h_init=None):
        """
        Forward pass through RNN sequence.
        
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
                y = self.ffn(h)
            else:
                y = self.W_hy(h)
            
            outputs.append(y)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h


class NativeRNN(nn.Module):
    """
    RNN using PyTorch's native nn.RNN for fast training.
    Same API as the scratch RNN class but uses optimized CUDA/MPS kernels.
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_ffn = use_ffn
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)
        else:
            self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_init=None):
        """
        Forward pass through RNN sequence.
        
        x: Input sequence (batch_size, seq_len, input_size)
        h_init: Initial hidden state (num_layers, batch_size, hidden_size)
        Returns: outputs (batch_size, seq_len, output_size), final hidden state
        """
        batch_size = x.shape[0]
        device = x.device
        
        if h_init is None:
            h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        rnn_out, h_n = self.rnn(x, h_init)
        
        if self.use_ffn:
            batch_size, seq_len, _ = rnn_out.shape
            rnn_out_flat = rnn_out.reshape(batch_size * seq_len, -1)
            outputs = self.ffn(rnn_out_flat).view(batch_size, seq_len, -1)
        else:
            outputs = self.W_hy(rnn_out)
        
        return outputs, h_n[-1]
