import numpy as np
from utils import tanh, xavier_init

class SimpleFFN:
    """Simple Feed-Forward Network: y = W2·ReLU(W1·x + b1) + b2"""
    def __init__(self, input_size, hidden_size, output_size):
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W1 = np.random.randn(hidden_size, input_size) * scale
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * scale
        self.b2 = np.zeros((output_size, 1))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        self.x = x
        self.z1 = self.W1 @ x + self.b1
        self.h1 = self.relu(self.z1)
        self.z2 = self.W2 @ self.h1 + self.b2
        return self.z2
    
    def backward(self, dy):
        dW2 = dy @ self.h1.T
        db2 = dy
        dh1 = self.W2.T @ dy
        dz1 = dh1 * self.relu_derivative(self.z1)
        dW1 = dz1 @ self.x.T
        db1 = dz1
        dx = self.W1.T @ dz1
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}, dx


class RNNCell:
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
        self.hidden_size = hidden_size  # Size of hidden state vector
        self.W_xh = xavier_init((hidden_size, input_size))  # Input-to-hidden weights
        self.W_hh = xavier_init((hidden_size, hidden_size))  # Hidden-to-hidden weights
        self.b_h = np.zeros((hidden_size, 1))  # Hidden bias
    
    def forward(self, x, h_prev):
        """Forward pass through RNN cell."""
        h = tanh(self.W_xh @ x + self.W_hh @ h_prev + self.b_h)
        return h, h_prev
    
    def backward(self, dh, h_t, h_prev, x):
        """Backward pass: gradients for single timestep."""
        dh_raw = (1 - h_t**2) * dh
        
        dW_xh = dh_raw @ x.T
        dW_hh = dh_raw @ h_prev.T
        db_h = dh_raw
        dh_prev = self.W_hh.T @ dh_raw
        
        return dh_prev, {'W_xh': dW_xh, 'W_hh': dW_hh, 'b_h': db_h}


class RNN:
    """
    Simple RNN with optional FFN output layer.
    
    h_t: Hidden state at time t
    y_t: Output at time t
    
    W_hy: Hidden-to-output weights (linear mode)
    b_y: Output bias (linear mode)
    """
    def __init__(self, input_size, hidden_size, output_size, use_ffn=False):
        self.input_size = input_size  # Dimensionality of input features
        self.hidden_size = hidden_size  # Size of RNN hidden state
        self.output_size = output_size  # Dimensionality of output
        self.use_ffn = use_ffn  # Whether to use FFN for output layer
        self.cell = RNNCell(input_size, hidden_size)  # RNN cell for sequence processing
        
        if use_ffn:
            self.ffn = SimpleFFN(hidden_size, hidden_size // 2, output_size)  # FFN output layer
        else:
            self.W_hy = xavier_init((output_size, hidden_size))  # Hidden-to-output weights
            self.b_y = np.zeros((output_size, 1))  # Output bias

    def forward(self, x):
        """Forward pass through RNN sequence."""
        h = np.zeros((self.hidden_size, 1))
        self.inputs = x
        self.hidden_states = []
        outputs = []
        
        for x_t in x:
            x_t = x_t.reshape(-1, 1)
            h, _ = self.cell.forward(x_t, h)
            self.hidden_states.append(h)
            
            if self.use_ffn:
                y = self.ffn.forward(h)
            else:
                y = self.W_hy @ h + self.b_y
            
            outputs.append(y)
            
        return np.array(outputs).squeeze()

    def backward(self, targets, learning_rate=0.01):
        """Backward pass: BPTT with gradient clipping."""
        loss = 0
        grads_cum = {
            'W_xh': np.zeros_like(self.cell.W_xh),
            'W_hh': np.zeros_like(self.cell.W_hh),
            'b_h': np.zeros_like(self.cell.b_h)
        }
        
        if self.use_ffn:
            ffn_grads_cum = {k: np.zeros_like(getattr(self.ffn, k)) for k in ['W1', 'b1', 'W2', 'b2']}
        else:
            dW_hy, db_y = np.zeros_like(self.W_hy), np.zeros_like(self.b_y)
        
        dh_next = np.zeros((self.hidden_size, 1))
        
        for t in reversed(range(len(self.inputs))):
            h_t = self.hidden_states[t]
            h_prev = self.hidden_states[t-1] if t > 0 else np.zeros((self.hidden_size, 1))
            x_t = self.inputs[t].reshape(-1, 1)
            target_t = targets[t].reshape(-1, 1)
            
            if self.use_ffn:
                pred = self.ffn.forward(h_t)
                dy = pred - target_t
                loss += 0.5 * np.sum(dy**2)
                
                ffn_grads, dh_from_output = self.ffn.backward(dy)
                for k in ffn_grads:
                    ffn_grads_cum[k] += ffn_grads[k]
                dh = dh_from_output + dh_next
            else:
                pred = self.W_hy @ h_t + self.b_y
                dy = pred - target_t
                loss += 0.5 * np.sum(dy**2)
                
                dW_hy += dy @ h_t.T
                db_y += dy
                dh = self.W_hy.T @ dy + dh_next
            
            dh_prev, grads = self.cell.backward(dh, h_t, h_prev, x_t)
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
            np.clip(dW_hy, -5, 5, out=dW_hy)
            np.clip(db_y, -5, 5, out=db_y)
            self.W_hy -= learning_rate * dW_hy
            self.b_y -= learning_rate * db_y
        
        return loss / len(self.inputs)
