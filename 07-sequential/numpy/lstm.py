import numpy as np
from utils import sigmoid, tanh, xavier_init
from rnn import SimpleFFN

class LSTMCell:
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
        self.hidden_size = hidden_size  # Size of hidden state and cell state vectors
        concat_len = input_size + hidden_size
        self.W_f = xavier_init((hidden_size, concat_len))  # Forget gate weights
        self.W_i = xavier_init((hidden_size, concat_len))  # Input gate weights
        self.W_c = xavier_init((hidden_size, concat_len))  # Cell candidate weights
        self.W_o = xavier_init((hidden_size, concat_len))  # Output gate weights
        self.b_f = np.zeros((hidden_size, 1))  # Forget gate bias
        self.b_i = np.zeros((hidden_size, 1))  # Input gate bias
        self.b_c = np.zeros((hidden_size, 1))  # Cell candidate bias
        self.b_o = np.zeros((hidden_size, 1))  # Output gate bias

    def forward(self, x, h_prev, c_prev):
        """Forward pass through LSTM cell."""
        concat = np.vstack((h_prev, x))
        f = sigmoid(self.W_f @ concat + self.b_f)
        i = sigmoid(self.W_i @ concat + self.b_i)
        c_tilde = tanh(self.W_c @ concat + self.b_c)
        o = sigmoid(self.W_o @ concat + self.b_o)
        c = f * c_prev + i * c_tilde
        h = o * tanh(c)
        return h, c, (concat, f, i, c_tilde, o, c_prev, c)

    def backward(self, dh_next, dc_next, cache):
        """Backward pass: BPTT through LSTM cell."""
        concat, f, i, c_tilde, o, c_prev, c = cache
        tc = tanh(c)
        ds = dc_next + (dh_next * o) * (1 - tc**2)
        do = dh_next * tc
        
        df = ds * c_prev
        di = ds * c_tilde
        dc_tilde = ds * i
        
        df_raw = df * f * (1 - f)
        di_raw = di * i * (1 - i)
        dc_tilde_raw = dc_tilde * (1 - c_tilde**2)
        do_raw = do * o * (1 - o)
        
        grads = {
            'W_f': df_raw @ concat.T, 'b_f': df_raw,
            'W_i': di_raw @ concat.T, 'b_i': di_raw,
            'W_c': dc_tilde_raw @ concat.T, 'b_c': dc_tilde_raw,
            'W_o': do_raw @ concat.T, 'b_o': do_raw
        }
        
        dconcat = (self.W_f.T @ df_raw + self.W_i.T @ di_raw + 
                  self.W_c.T @ dc_tilde_raw + self.W_o.T @ do_raw)
        dh_prev = dconcat[:self.hidden_size]
        dc_prev = ds * f
        
        return dh_prev, dc_prev, grads


class LSTM:
    """
    LSTM with optional FFN output layer.
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False):
        self.input_size = input_size  # Dimensionality of input features
        self.hidden_size = hidden_size  # Size of LSTM hidden state
        self.output_size = output_size  # Dimensionality of output
        self.use_ffn = use_ffn  # Whether to use FFN for output layer
        self.cell = LSTMCell(input_size, hidden_size)  # LSTM cell for sequence processing
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)  # FFN output layer
        else:
            self.W_hy = xavier_init((output_size, hidden_size))  # Hidden-to-output weights
            self.b_y = np.zeros((output_size, 1))  # Output bias

    def forward(self, x):
        """Forward pass through LSTM sequence."""
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        self.caches = []
        self.hidden_states = []
        outputs = []
        
        for x_t in x:
            x_t = x_t.reshape(-1, 1)
            h, c, cache = self.cell.forward(x_t, h, c)
            self.caches.append(cache)
            self.hidden_states.append(h)
            
            if self.use_ffn:
                out = self.ffn.forward(h)
            else:
                out = self.W_hy @ h + self.b_y
            outputs.append(out)
            
        return np.array(outputs).squeeze()

    def backward(self, targets=None, upstream_grads=None, learning_rate=0.01):
        """Backward pass: BPTT with gradient clipping."""
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        
        grads_cum = {
            k: np.zeros_like(getattr(self.cell, k)) 
            for k in ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o']
        }
        
        if self.use_ffn:
            ffn_grads_cum = {k: np.zeros_like(getattr(self.ffn, k)) for k in ['W1', 'b1', 'W2', 'b2']}
        else:
            dWy, dby = np.zeros_like(self.W_hy), np.zeros_like(self.b_y)
            
        total_loss = 0

        for t in reversed(range(len(self.caches))):
            h = self.hidden_states[t]
            
            if upstream_grads is not None:
                dy = upstream_grads[t].reshape(-1, 1)
                if self.use_ffn:
                    ffn_grads, dh = self.ffn.backward(dy)
                    for k in ffn_grads:
                        ffn_grads_cum[k] += ffn_grads[k]
                else:
                    dWy += dy @ h.T
                    dby += dy
                    dh = self.W_hy.T @ dy
            else:
                target = targets[t].reshape(-1, 1)
                if self.use_ffn:
                    pred = self.ffn.forward(h)
                    dy = pred - target
                    ffn_grads, dh = self.ffn.backward(dy)
                    for k in ffn_grads:
                        ffn_grads_cum[k] += ffn_grads[k]
                    total_loss += 0.5 * np.sum(dy**2)
                else:
                    pred = self.W_hy @ h + self.b_y
                    dy = pred - target
                    dWy += dy @ h.T
                    dby += dy
                    dh = self.W_hy.T @ dy
                    total_loss += 0.5 * np.sum(dy**2)
            
            dh += dh_next
            dh_prev, dc_prev, grads = self.cell.backward(dh, dc_next, self.caches[t])
            
            for k in grads:
                grads_cum[k] += grads[k]
                
            dh_next = dh_prev
            dc_next = dc_prev

        for k, v in grads_cum.items():
            np.clip(v, -5, 5, out=v)
            param = getattr(self.cell, k)
            param -= learning_rate * v
            
        if self.use_ffn:
            for k, v in ffn_grads_cum.items():
                np.clip(v, -5, 5, out=v)
                param = getattr(self.ffn, k)
                param -= learning_rate * v
        else:
            np.clip(dWy, -5, 5, out=dWy)
            np.clip(dby, -5, 5, out=dby)
            self.W_hy -= learning_rate * dWy
            self.b_y -= learning_rate * dby
            
        return total_loss / len(self.caches)

class BidirectionalLSTM:
    """
    Bidirectional LSTM: y_t = FFN([h_fwd_t, h_bwd_t])
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size  # Dimensionality of input features
        self.hidden_size = hidden_size  # Size of each LSTM's hidden state
        self.output_size = output_size  # Dimensionality of output
        
        self.fwd_lstm = LSTM(input_size, hidden_size, hidden_size, use_ffn=False)  # Forward LSTM
        self.bwd_lstm = LSTM(input_size, hidden_size, hidden_size, use_ffn=False)  # Backward LSTM
        
        self.ffn = SimpleFFN(hidden_size * 2, hidden_size, output_size)  # FFN combining both directions
        self.output_size = output_size
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """Forward pass through bidirectional LSTM."""
        fwd_out = self.fwd_lstm.forward(x)
        bwd_out = self.bwd_lstm.forward(x[::-1])[::-1]
        
        if fwd_out.ndim == 1:
            fwd_out = fwd_out.reshape(-1, 1)
        if bwd_out.ndim == 1:
            bwd_out = bwd_out.reshape(-1, 1)
            
        self.combined = np.hstack((fwd_out, bwd_out))
        outputs = []
        for t in range(len(x)):
            out = self.ffn.forward(self.combined[t:t+1].T)
            outputs.append(out)
        return np.array(outputs).squeeze()
        
    def backward(self, targets, learning_rate=0.01):
        """Backward pass through bidirectional LSTM."""
        ffn_grads_cum = {k: np.zeros_like(getattr(self.ffn, k)) for k in ['W1', 'b1', 'W2', 'b2']}
        d_combined = np.zeros_like(self.combined)
        total_loss = 0
        
        for t in range(len(self.combined)):
            combined_t = self.combined[t:t+1].T
            target_t = targets[t].reshape(-1, 1)
            pred = self.ffn.forward(combined_t)
            dy = pred - target_t
            
            ffn_grads, d_comb_t = self.ffn.backward(dy)
            for k in ffn_grads:
                ffn_grads_cum[k] += ffn_grads[k]
            d_combined[t] = d_comb_t.flatten()
            total_loss += 0.5 * np.sum(dy**2)
        
        for k, v in ffn_grads_cum.items():
            np.clip(v, -5, 5, out=v)
            param = getattr(self.ffn, k)
            param -= learning_rate * v
        
        d_fwd = d_combined[:, :self.hidden_size]
        d_bwd = d_combined[:, self.hidden_size:]
        
        self.fwd_lstm.backward(upstream_grads=d_fwd, learning_rate=learning_rate)
        self.bwd_lstm.backward(upstream_grads=d_bwd[::-1], learning_rate=learning_rate)
        
        return total_loss / len(targets) 


 

