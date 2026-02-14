import numpy as np
from utils import sigmoid, tanh, xavier_init
from rnn import SimpleFFN

class GRUCell:
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
        self.hidden_size = hidden_size
        concat_len = input_size + hidden_size
        self.W_z = xavier_init((hidden_size, concat_len))
        self.W_r = xavier_init((hidden_size, concat_len))
        self.W_h = xavier_init((hidden_size, concat_len))
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        """Forward pass through GRU cell."""
        concat = np.vstack((h_prev, x))
        z = sigmoid(self.W_z @ concat + self.b_z)
        r = sigmoid(self.W_r @ concat + self.b_r)
        h_reset_concat = np.vstack((r * h_prev, x))
        h_tilde = tanh(self.W_h @ h_reset_concat + self.b_h)
        h = (1 - z) * h_prev + z * h_tilde
        return h, (concat, h_reset_concat, z, r, h_tilde, h_prev)

    def backward(self, dh_next, cache):
        """Backward pass: BPTT through GRU cell."""
        concat, h_reset_concat, z, r, h_tilde, h_prev = cache
        
        dh_tilde = dh_next * z
        dz = dh_next * (h_tilde - h_prev)
        dh_prev_from_h = dh_next * (1 - z)
        
        dh_tilde_raw = dh_tilde * (1 - h_tilde**2)
        dz_raw = dz * z * (1 - z)
        
        grads = {
            'W_h': dh_tilde_raw @ h_reset_concat.T,
            'b_h': dh_tilde_raw,
            'W_z': dz_raw @ concat.T,
            'b_z': dz_raw
        }
        
        dh_reset_concat = self.W_h.T @ dh_tilde_raw
        dr_h_prev = dh_reset_concat[:self.hidden_size]
        dr = dr_h_prev * h_prev
        dr_raw = dr * r * (1 - r)
        
        grads['W_r'] = dr_raw @ concat.T
        grads['b_r'] = dr_raw
        
        dconcat = self.W_z.T @ dz_raw + self.W_r.T @ dr_raw
        dh_prev = dconcat[:self.hidden_size] + dr_h_prev * r + dh_prev_from_h
        
        return dh_prev, grads


class GRU:
    """
    GRU with optional FFN output layer.
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_ffn = use_ffn
        self.cell = GRUCell(input_size, hidden_size)
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)
        else:
            self.W_hy = xavier_init((output_size, hidden_size))
            self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """Forward pass through GRU sequence."""
        h = np.zeros((self.hidden_size, 1))
        self.caches = []
        self.hidden_states = []
        outputs = []
        
        for x_t in x:
            x_t = x_t.reshape(-1, 1)
            h, cache = self.cell.forward(x_t, h)
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
        
        grads_cum = {
            k: np.zeros_like(getattr(self.cell, k)) 
            for k in ['W_z', 'W_r', 'W_h', 'b_z', 'b_r', 'b_h']
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
            dh_prev, grads = self.cell.backward(dh, self.caches[t])
            
            for k in grads:
                grads_cum[k] += grads[k]
                
            dh_next = dh_prev

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

class BidirectionalGRU:
    """
    Bidirectional GRU: y_t = FFN([h_fwd_t, h_bwd_t])
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fwd_gru = GRU(input_size, hidden_size, hidden_size, use_ffn=False)
        self.bwd_gru = GRU(input_size, hidden_size, hidden_size, use_ffn=False)
        
        self.ffn = SimpleFFN(hidden_size * 2, hidden_size, output_size)
        self.output_size = output_size
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """Forward pass through bidirectional GRU."""
        fwd_out = self.fwd_gru.forward(x)
        bwd_out = self.bwd_gru.forward(x[::-1])[::-1]
        
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
        """Backward pass through bidirectional GRU."""
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
        
        self.fwd_gru.backward(upstream_grads=d_fwd, learning_rate=learning_rate)
        self.bwd_gru.backward(upstream_grads=d_bwd[::-1], learning_rate=learning_rate)
        
        return total_loss / len(targets) 


 

