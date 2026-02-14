import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn import SimpleFFN

class LSTMCell(nn.Module):
    """
    LSTM Cell: 
    - f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
    - i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
    - c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)
    - o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
    - c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    - h_t = o_t ⊙ tanh(c_t)

    x_t: Input at time t
    [h_{t-1}, x_t]: Concat of hidden state and input
    f_t: Forget gate at time t
    i_t: Input gate at time t
    c̃_t: Cell candidate at time t
    o_t: Output gate at time t
    c_t: Cell state at time t
    h_t: Hidden state at time t

    W_f: Hidden-to-forget gate weights
    W_i: Hidden-to-input gate weights
    W_c: Hidden-to-cell candidate weights
    W_o: Hidden-to-output gate weights
    b_f: Forget gate bias
    b_i: Input gate bias
    b_c: Cell candidate bias
    b_o: Output gate bias
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size  # Size of hidden state and cell state vectors
        concat_len = input_size + hidden_size
        self.W_f = nn.Linear(concat_len, hidden_size)  # Forget gate weights
        self.W_i = nn.Linear(concat_len, hidden_size)  # Input gate weights
        self.W_c = nn.Linear(concat_len, hidden_size)  # Cell candidate weights
        self.W_o = nn.Linear(concat_len, hidden_size)  # Output gate weights

    def forward(self, x, h_prev, c_prev):
        """Forward pass through LSTM cell."""
        concat = torch.cat([h_prev, x], dim=-1)
        f = torch.sigmoid(self.W_f(concat))
        i = torch.sigmoid(self.W_i(concat))
        c_tilde = torch.tanh(self.W_c(concat))
        o = torch.sigmoid(self.W_o(concat))
        c = f * c_prev + i * c_tilde
        h = o * torch.tanh(c)
        return h, c


class LSTM(nn.Module):
    """
    LSTM with optional FFN output layer.
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False):
        super().__init__()
        self.input_size = input_size  # Dimensionality of input features
        self.hidden_size = hidden_size  # Size of LSTM hidden state
        self.output_size = output_size  # Dimensionality of output
        self.use_ffn = use_ffn  # Whether to use FFN for output layer
        self.cell = LSTMCell(input_size, hidden_size)  # LSTM cell for sequence processing
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)  # FFN output layer
        else:
            self.W_hy = nn.Linear(hidden_size, output_size)  # Hidden-to-output weights

    def forward(self, x, h_init=None, c_init=None):
        """
        Forward pass through LSTM sequence.
        
        x: Input sequence (batch_size, seq_len, input_size)
        h_init: Initial hidden state (batch_size, hidden_size)
        c_init: Initial cell state (batch_size, hidden_size)
        Returns: outputs (batch_size, seq_len, output_size), (h_final, c_final)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if h_init is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h = h_init
        if c_init is None:
            c = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            c = c_init
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h, c = self.cell(x_t, h, c)
            
            if self.use_ffn:
                out = self.ffn(h)
            else:
                out = self.W_hy(h)
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, c)


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM: y_t = FFN([h_fwd_t, h_bwd_t])
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size  # Dimensionality of input features
        self.hidden_size = hidden_size  # Size of each LSTM's hidden state
        self.output_size = output_size  # Dimensionality of output
        
        self.fwd_lstm = LSTM(input_size, hidden_size, hidden_size, use_ffn=False)  # Forward LSTM
        self.bwd_lstm = LSTM(input_size, hidden_size, hidden_size, use_ffn=False)  # Backward LSTM
        
        self.ffn = SimpleFFN(hidden_size * 2, hidden_size, output_size)  # FFN combining both directions
        
    def forward(self, x):
        """
        Forward pass through bidirectional LSTM.
        
        x: Input sequence (batch_size, seq_len, input_size)
        Returns: outputs (batch_size, seq_len, output_size)
        """
        fwd_out, _ = self.fwd_lstm(x)
        bwd_out, _ = self.bwd_lstm(torch.flip(x, dims=[1]))
        bwd_out = torch.flip(bwd_out, dims=[1])
        
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        
        batch_size, seq_len, _ = combined.shape
        combined_flat = combined.view(batch_size * seq_len, -1)
        outputs = self.ffn(combined_flat)
        outputs = outputs.view(batch_size, seq_len, -1)
        
        return outputs


class NativeLSTM(nn.Module):
    """
    LSTM using PyTorch's native nn.LSTM for fast training.
    Same API as the scratch LSTM class but uses optimized CUDA/MPS kernels.
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_ffn = use_ffn
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)
        else:
            self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_init=None, c_init=None):
        """
        Forward pass through LSTM sequence.
        
        x: Input sequence (batch_size, seq_len, input_size)
        h_init: Initial hidden state (num_layers, batch_size, hidden_size)
        c_init: Initial cell state (num_layers, batch_size, hidden_size)
        Returns: outputs (batch_size, seq_len, output_size), (h_final, c_final)
        """
        batch_size = x.shape[0]
        device = x.device
        
        if h_init is None:
            h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if c_init is None:
            c_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        lstm_out, (h_n, c_n) = self.lstm(x, (h_init, c_init))
        
        if self.use_ffn:
            batch_size, seq_len, _ = lstm_out.shape
            lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)
            outputs = self.ffn(lstm_out_flat).view(batch_size, seq_len, -1)
        else:
            outputs = self.W_hy(lstm_out)
        
        return outputs, (h_n[-1], c_n[-1])


class NativeBidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM using PyTorch's native nn.LSTM for fast training.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        
        self.ffn = SimpleFFN(hidden_size * 2, hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through bidirectional LSTM.
        
        x: Input sequence (batch_size, seq_len, input_size)
        Returns: outputs (batch_size, seq_len, output_size)
        """
        lstm_out, _ = self.lstm(x)
        
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)
        outputs = self.ffn(lstm_out_flat).view(batch_size, seq_len, -1)
        
        return outputs
