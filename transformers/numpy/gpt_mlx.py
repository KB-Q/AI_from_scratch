"""
GPT-style Transformer implementation using MLX (Apple Silicon optimized).
Much faster than NumPy version with automatic differentiation.
"""
import mlx.core as mx
import mlx.nn as nn
import pickle


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def __call__(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len) causal mask
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.d_k))
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask[None, None, :, :]
        
        # Attention weights and output
        attn_weights = mx.softmax(scores, axis=-1)
        attn_out = attn_weights @ v
        
        # Concatenate heads
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def __call__(self, x):
        return self.linear2(nn.gelu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def __call__(self, x, mask=None):
        # Attention with residual and layer norm
        x = x + self.attn(self.norm1(x), mask)
        
        # Feed-forward with residual and layer norm
        x = x + self.ffn(self.norm2(x))
        
        return x


class GPTModel(nn.Module):
    """GPT-style autoregressive transformer using MLX."""
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (fixed, not learned)
        self.register_buffer('pos_embedding', self._create_positional_encoding(max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings."""
        import numpy as np
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return mx.array(pe)
    
    def register_buffer(self, name, value):
        """Store non-trainable parameters."""
        setattr(self, name, value)
    
    def create_causal_mask(self, seq_len):
        """Upper triangular mask to prevent attending to future tokens."""
        mask = mx.triu(mx.ones((seq_len, seq_len)), k=1) * -1e9
        return mask
    
    def __call__(self, token_ids):
        """
        token_ids: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Embed tokens and add positional encoding
        x = self.token_embedding(token_ids)
        x = x + self.pos_embedding[:seq_len]
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, seed_ids, max_new_tokens=20, temperature=1.0):
        """
        Generate tokens autoregressively.
        seed_ids: (1, seq_len) initial tokens as numpy array
        """
        import numpy as np
        
        # Convert to MLX array
        generated = seed_ids[0].tolist()
        
        for _ in range(max_new_tokens):
            # Take last max_seq_len tokens
            context = mx.array([generated[-self.max_seq_len:]])
            
            # Forward pass
            logits = self(context)
            
            # Get logits for last position
            next_logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = mx.softmax(next_logits)
            probs_np = np.array(probs)
            next_token = np.random.choice(self.vocab_size, p=probs_np)
            
            generated.append(int(next_token))
        
        return np.array(generated)
    
    def save(self, filepath):
        """Save model parameters to file."""
        # Get all trainable parameters
        params = self.parameters()
        
        model_data = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len,
            'num_layers': len(self.blocks),
            'params': params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model parameters from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model with same architecture
        model = cls(
            vocab_size=model_data['vocab_size'],
            d_model=model_data['d_model'],
            num_layers=model_data['num_layers'],
            num_heads=4,
            d_ff=512,
            max_seq_len=model_data['max_seq_len']
        )
        
        # Load parameters
        model.update(model_data['params'])
        
        print(f"Model loaded from {filepath}")
        return model


def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss for language modeling.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) with token IDs
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for loss computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Cross-entropy loss
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
    
    return loss
