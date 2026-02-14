"""
GPT-style Transformer implementation using only NumPy.
Educational implementation for understanding next-token prediction.
"""
import numpy as np
from utils_data import positional_encoding, softmax, gelu, layer_norm

class MultiHeadAttention:
    """
    Multi-head self-attention with causal masking.

    Args:
        d_model: Dimension of the model (d_k * num_heads)
        NOTE: d_model is also same as the embedding dimension.
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Xavier initialization
        # NOTE: We are creating single weight matrices for all heads here.
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q: np.ndarray = np.random.randn(d_model, d_model) * scale
        self.W_k: np.ndarray = np.random.randn(d_model, d_model) * scale
        self.W_v: np.ndarray = np.random.randn(d_model, d_model) * scale
        self.W_o: np.ndarray = np.random.randn(d_model, d_model) * scale
        
        # Gradients
        self.dW_q: np.ndarray = np.zeros_like(self.W_q)
        self.dW_k: np.ndarray = np.zeros_like(self.W_k)
        self.dW_v: np.ndarray = np.zeros_like(self.W_v)
        self.dW_o: np.ndarray = np.zeros_like(self.W_o)
        
        # Cache for backward pass
        self.cache = {}
        
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        x: (batch, seq_len, d_model) - input embeddings
        mask: (seq_len, seq_len) - causal mask
        """
        B, T, _ = x.shape
        d_m, H, d_k = self.d_model, self.num_heads, self.d_k
        
        # Linear projections: (B, T, d_m) @ (d_m, d_m) -> (B, T, d_m)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape for multi-head: (B, T, d_m) -> (B, T, H, d_k) -> (B, H, T, d_k)
        Q = Q.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        
        # Attention scores:
        # K^T: (B, H, T, K) -> (B, H, K, T)
        # Q @ K^T: (B, H, T, K) @ (B, H, K, T) -> (B, H, T, T)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # Apply causal mask
        if mask is not None: scores += mask[None, None, :, :]
        
        # Attention output: (B, H, T, T) @ (B, H, T, K) -> (B, H, T, K)
        attn_weights = softmax(scores, axis=-1)
        attn_out = attn_weights @ V
        
        # Concatenate heads: (B, T, C)
        # (B, H, T, K) -> transpose (B, T, H, K) -> reshape (B, T, C)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, d_m)
        
        # Output projection
        # (B, T, d_m) @ (d_m, d_m) -> (B, T, d_m)
        output = attn_out @ self.W_o
        
        # Cache for backward
        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'scores': scores, 'attn_weights': attn_weights,
            'attn_out': attn_out, 'mask': mask
        }
        
        return output

    def forward_slow(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Slower but easier-to-read multi-head attention.

        Computes attention for each head separately using simple slicing, then
        concatenates head outputs and applies the output projection.

        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len) additive causal mask (e.g., 0 or -1e9)
        """
        B, T, _ = x.shape
        d_m, H, d_k = self.d_model, self.num_heads, self.d_k

        # Linear projections (one big projection per Q/K/V, then slice per head)
        Q_full = x @ self.W_q  # (B, T, d_m)
        K_full = x @ self.W_k  # (B, T, d_m)
        V_full = x @ self.W_v  # (B, T, d_m)

        # Cache tensors in the same shapes as forward() so backward() can reuse them.
        Q = np.empty((B, H, T, d_k), dtype=Q_full.dtype)
        K = np.empty((B, H, T, d_k), dtype=K_full.dtype)
        V = np.empty((B, H, T, d_k), dtype=V_full.dtype)
        scores = np.empty((B, H, T, T), dtype=Q_full.dtype)
        attn_weights = np.empty((B, H, T, T), dtype=Q_full.dtype)

        head_outputs = []
        scale = 1.0 / np.sqrt(d_k)

        for h in range(H):
            start = h * d_k
            end = (h + 1) * d_k

            q = Q_full[:, :, start:end]  # (B, T, d_k)
            k = K_full[:, :, start:end]  # (B, T, d_k)
            v = V_full[:, :, start:end]  # (B, T, d_k)

            Q[:, h, :, :] = q
            K[:, h, :, :] = k
            V[:, h, :, :] = v

            # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
            s = (q @ k.transpose(0, 2, 1)) * scale
            if mask is not None:
                s = s + mask[None, :, :]

            a = softmax(s, axis=-1)  # (B, T, T)
            o = a @ v  # (B, T, d_k)

            scores[:, h, :, :] = s
            attn_weights[:, h, :, :] = a
            head_outputs.append(o)

        # Concatenate heads along the channel dimension: (B, T, H*d_k) == (B, T, d_m)
        attn_out = np.concatenate(head_outputs, axis=-1)

        # Output projection back to model dimension
        output = attn_out @ self.W_o

        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'scores': scores, 'attn_weights': attn_weights,
            'attn_out': attn_out, 'mask': mask
        }

        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: gradient from upstream (batch, seq_len, d_model)
        Returns: dx (batch, seq_len, d_model)
        """
        x = self.cache['x']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attn_weights = self.cache['attn_weights']
        attn_out = self.cache['attn_out']
        
        B, T, _ = x.shape
        C, H, K = self.d_model, self.num_heads, self.d_k
        
        # Gradient through output projection
        # (B*T, C)^T @ (B*T, C) -> (C, C)
        self.dW_o += attn_out.reshape(-1, C).T @ dout.reshape(-1, C)
        # (B, T, C) @ (C, C) -> (B, T, C)
        d_attn_out = dout @ self.W_o.T
        
        # Reshape back to multi-head format
        # (B, T, C) -> (B, T, H, K) -> (B, H, T, K)
        d_attn_out = d_attn_out.reshape(B, T, H, K).transpose(0, 2, 1, 3)
        
        # Gradient through attention
        # (B, H, T, T) -> (B, H, T, T)
        # (B, H, T, T) @ (B, H, T, K) -> (B, H, T, K)
        dV = attn_weights.transpose(0, 1, 3, 2) @ d_attn_out
        
        # (B, H, T, K) -> (B, H, K, T)
        # (B, H, T, K) @ (B, H, K, T) -> (B, H, T, T)
        d_attn_weights = d_attn_out @ V.transpose(0, 1, 3, 2)
        
        # Gradient through softmax
        dscores = d_attn_weights * attn_weights
        dscores -= attn_weights * np.sum(dscores, axis=-1, keepdims=True)
        dscores = dscores / np.sqrt(self.d_k)
        
        # Gradient through Q, K
        # (B, H, T, T) @ (B, H, T, K) -> (B, H, T, K)
        dQ = dscores @ K
        
        # (B, H, T, T) -> (B, H, T, T)
        # (B, H, T, T) @ (B, H, T, K) -> (B, H, T, K)
        dK = dscores.transpose(0, 1, 3, 2) @ Q
        
        # Reshape back
        # (B, H, T, K) -> (B, T, H, K) -> (B, T, C)
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, C)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, C)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Gradient through projections
        # (B*T, C)^T @ (B*T, C) -> (C, C)
        self.dW_q += x.reshape(-1, C).T @ dQ.reshape(-1, C)
        self.dW_k += x.reshape(-1, C).T @ dK.reshape(-1, C)
        self.dW_v += x.reshape(-1, C).T @ dV.reshape(-1, C)
        
        # (B, T, C) @ (C, C) -> (B, T, C)
        dx = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T
        
        return dx


class FeedForward:
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff):
        scale = np.sqrt(2.0 / d_model)
        self.W1: np.ndarray = np.random.randn(d_model, d_ff) * scale
        self.b1: np.ndarray = np.zeros(d_ff)
        self.W2: np.ndarray = np.random.randn(d_ff, d_model) * scale
        self.b2: np.ndarray = np.zeros(d_model)
        
        self.dW1: np.ndarray = np.zeros_like(self.W1)
        self.db1: np.ndarray = np.zeros_like(self.b1)
        self.dW2: np.ndarray = np.zeros_like(self.W2)
        self.db2: np.ndarray = np.zeros_like(self.b2)
        
        self.cache: dict = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq_len, d_model)"""
        h = x @ self.W1 + self.b1
        h_act = gelu(h)
        out = h_act @ self.W2 + self.b2
        
        self.cache: dict = {'x': x, 'h': h, 'h_act': h_act}
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (batch, seq_len, d_model)"""
        x = self.cache['x']
        h = self.cache['h']
        h_act = self.cache['h_act']
        
        # Gradient through second layer
        # dout_flat: (B, T, C) -> (B*T, C); h_act_flat: (B, T, d_ff) -> (B*T, d_ff)
        dout_flat = dout.reshape(-1, dout.shape[-1])
        h_act_flat = h_act.reshape(-1, h_act.shape[-1])
        
        # (B*T, d_ff)^T @ (B*T, C) -> (d_ff, C)
        self.dW2 += h_act_flat.T @ dout_flat
        self.db2 += np.sum(dout_flat, axis=0)
        
        # (B, T, C) @ (C, d_ff) -> (B, T, d_ff)
        dh_act = dout @ self.W2.T
        
        # Gradient through GELU (approximation)
        dh = dh_act * (0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (h + 0.044715 * h**3))) + 
                       0.5 * h * (1 - np.tanh(np.sqrt(2/np.pi) * (h + 0.044715 * h**3))**2) * 
                       np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * h**2))
        
        # Gradient through first layer
        # dh_flat: (B, T, d_ff) -> (B*T, d_ff); x_flat: (B, T, C) -> (B*T, C)
        dh_flat = dh.reshape(-1, dh.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        
        # (B*T, C)^T @ (B*T, d_ff) -> (C, d_ff)
        self.dW1 += x_flat.T @ dh_flat
        self.db1 += np.sum(dh_flat, axis=0)
        
        # (B, T, d_ff) @ (d_ff, C) -> (B, T, C)
        dx = dh @ self.W1.T
        
        return dx


class TransformerBlock:
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer norm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
        self.dgamma1 = np.zeros_like(self.gamma1)
        self.dbeta1 = np.zeros_like(self.beta1)
        self.dgamma2 = np.zeros_like(self.gamma2)
        self.dbeta2 = np.zeros_like(self.beta2)
        
        self.cache: dict = {}
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """x: (batch, seq_len, d_model)"""
        # Attention with residual
        attn_out = self.attn.forward(x, mask)
        x1 = x + attn_out
        x1_norm = layer_norm(x1, self.gamma1, self.beta1)
        
        # Feed-forward with residual
        ffn_out = self.ffn.forward(x1_norm)
        x2 = x1_norm + ffn_out
        x2_norm = layer_norm(x2, self.gamma2, self.beta2)
        
        self.cache = {
            'x': x, 'attn_out': attn_out, 
            'x1': x1, 'x1_norm': x1_norm,
            'ffn_out': ffn_out, 'x2': x2
        }
        return x2_norm
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (batch, seq_len, d_model)"""
        x = self.cache['x']
        x1 = self.cache['x1']
        x1_norm = self.cache['x1_norm']
        x2 = self.cache['x2']
        
        # Gradient through second layer norm
        dx2, dgamma2, dbeta2 = self._layer_norm_backward(dout, x2, self.gamma2)
        self.dgamma2 += dgamma2
        self.dbeta2 += dbeta2
        
        # Gradient through FFN residual
        dx1_norm = dx2
        dffn_out = dx2
        dx1_norm += self.ffn.backward(dffn_out)
        
        # Gradient through first layer norm
        dx1, dgamma1, dbeta1 = self._layer_norm_backward(dx1_norm, x1, self.gamma1)
        self.dgamma1 += dgamma1
        self.dbeta1 += dbeta1
        
        # Gradient through attention residual
        dx = dx1
        dattn_out = dx1
        dx += self.attn.backward(dattn_out)
        
        return dx
    
    def _layer_norm_backward(
        self,
        dout: np.ndarray,
        x: np.ndarray,
        gamma: np.ndarray,
        eps: float = 1e-5
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass for layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        x_norm = (x - mean) / std
        
        dgamma = np.sum(dout * x_norm, axis=(0, 1))
        dbeta = np.sum(dout, axis=(0, 1))
        
        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1/std, axis=-1, keepdims=True) + dvar * np.mean(-2*(x - mean), axis=-1, keepdims=True)
        
        dx = dx_norm / std + dvar * 2 * (x - mean) / x.shape[-1] + dmean / x.shape[-1]
        
        return dx, dgamma, dbeta


class GPTModel:
    """GPT-style autoregressive transformer."""
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_seq_len=64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embedding = positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
        
        # Gradients
        self.dtoken_embedding = np.zeros_like(self.token_embedding)
        self.doutput_proj = np.zeros_like(self.output_proj)
        
        # Cache
        self.cache = {}
    
    def create_causal_mask(self, seq_len):
        """Upper triangular mask to prevent attending to future tokens."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        return mask
    
    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Embed tokens and add positional encoding
        x = self.token_embedding[token_ids]
        x = x + self.pos_embedding[:seq_len]
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Project to vocabulary
        logits = x @ self.output_proj
        
        self.cache = {'token_ids': token_ids, 'x_final': x}
        
        return logits
    
    def backward(self, dlogits):
        """
        dlogits: gradient from loss (batch, seq_len, vocab_size)
        """
        token_ids = self.cache['token_ids']
        x_final = self.cache['x_final']
        
        # Gradient through output projection
        # x_final.reshape: (B, T, C) -> (B*T, C); dlogits.reshape: (B, T, V) -> (B*T, V)
        # (B*T, C)^T @ (B*T, V) -> (C, V)
        self.doutput_proj = x_final.reshape(-1, self.d_model).T @ dlogits.reshape(-1, self.vocab_size)
        # (B, T, V) @ (V, C) -> (B, T, C)
        dx = dlogits @ self.output_proj.T
        
        # Backward through blocks
        mask = self.create_causal_mask(token_ids.shape[1])
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        
        # Gradient through embeddings (positional encoding is fixed)
        batch_size, seq_len = token_ids.shape
        for b in range(batch_size):
            for t in range(seq_len):
                self.dtoken_embedding[token_ids[b, t]] += dx[b, t]
    
    def zero_grad(self):
        """Reset all gradients to zero."""
        self.dtoken_embedding.fill(0)
        self.doutput_proj.fill(0)
        
        for block in self.blocks:
            block.attn.dW_q.fill(0)
            block.attn.dW_k.fill(0)
            block.attn.dW_v.fill(0)
            block.attn.dW_o.fill(0)
            
            block.ffn.dW1.fill(0)
            block.ffn.db1.fill(0)
            block.ffn.dW2.fill(0)
            block.ffn.db2.fill(0)
            
            block.dgamma1.fill(0)
            block.dbeta1.fill(0)
            block.dgamma2.fill(0)
            block.dbeta2.fill(0)
    
    def update_params(self, lr=0.001):
        """SGD parameter update."""
        self.token_embedding -= lr * self.dtoken_embedding
        self.output_proj -= lr * self.doutput_proj
        
        for block in self.blocks:
            block.attn.W_q -= lr * block.attn.dW_q
            block.attn.W_k -= lr * block.attn.dW_k
            block.attn.W_v -= lr * block.attn.dW_v
            block.attn.W_o -= lr * block.attn.dW_o
            
            block.ffn.W1 -= lr * block.ffn.dW1
            block.ffn.b1 -= lr * block.ffn.db1
            block.ffn.W2 -= lr * block.ffn.dW2
            block.ffn.b2 -= lr * block.ffn.db2
            
            block.gamma1 -= lr * block.dgamma1
            block.beta1 -= lr * block.dbeta1
            block.gamma2 -= lr * block.dgamma2
            block.beta2 -= lr * block.dbeta2
    
    def generate(self, seed_ids, max_new_tokens=20, temperature=1.0):
        """
        Generate tokens autoregressively.
        seed_ids: (1, seq_len) initial tokens
        """
        generated = seed_ids[0].tolist()
        
        for _ in range(max_new_tokens):
            # Take last max_seq_len tokens
            context = np.array([generated[-self.max_seq_len:]])
            
            # Forward pass
            logits = self.forward(context)
            
            # Get logits for last position
            next_logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = softmax(next_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)
            
            generated.append(next_token)
        
        return np.array(generated)
    
    def save(self, filepath):
        """Save model parameters to file."""
        import pickle
        
        model_data = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len,
            'num_layers': len(self.blocks),
            'token_embedding': self.token_embedding,
            'pos_embedding': self.pos_embedding,
            'output_proj': self.output_proj,
            'blocks': []
        }
        
        # Save each block's parameters
        for block in self.blocks:
            block_data = {
                'attn': {
                    'W_q': block.attn.W_q,
                    'W_k': block.attn.W_k,
                    'W_v': block.attn.W_v,
                    'W_o': block.attn.W_o,
                },
                'ffn': {
                    'W1': block.ffn.W1,
                    'b1': block.ffn.b1,
                    'W2': block.ffn.W2,
                    'b2': block.ffn.b2,
                },
                'gamma1': block.gamma1,
                'beta1': block.beta1,
                'gamma2': block.gamma2,
                'beta2': block.beta2,
            }
            model_data['blocks'].append(block_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model parameters from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model with same architecture
        model = cls(
            vocab_size=model_data['vocab_size'],
            d_model=model_data['d_model'],
            num_layers=model_data['num_layers'],
            num_heads=4,  # Default, not critical for loading
            d_ff=512,     # Default, not critical for loading
            max_seq_len=model_data['max_seq_len']
        )
        
        # Load embeddings
        model.token_embedding = model_data['token_embedding']
        model.pos_embedding = model_data['pos_embedding']
        model.output_proj = model_data['output_proj']
        
        # Load each block's parameters
        for i, block_data in enumerate(model_data['blocks']):
            model.blocks[i].attn.W_q = block_data['attn']['W_q']
            model.blocks[i].attn.W_k = block_data['attn']['W_k']
            model.blocks[i].attn.W_v = block_data['attn']['W_v']
            model.blocks[i].attn.W_o = block_data['attn']['W_o']
            
            model.blocks[i].ffn.W1 = block_data['ffn']['W1']
            model.blocks[i].ffn.b1 = block_data['ffn']['b1']
            model.blocks[i].ffn.W2 = block_data['ffn']['W2']
            model.blocks[i].ffn.b2 = block_data['ffn']['b2']
            
            model.blocks[i].gamma1 = block_data['gamma1']
            model.blocks[i].beta1 = block_data['beta1']
            model.blocks[i].gamma2 = block_data['gamma2']
            model.blocks[i].beta2 = block_data['beta2']
        
        print(f"Model loaded from {filepath}")
        return model
