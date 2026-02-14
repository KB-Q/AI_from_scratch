"""
GPT-style Transformer implementation using PyTorch.
Educational implementation for understanding next-token prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def positional_encoding(max_seq_len, d_model):
    """Generate sinusoidal positional encodings."""
    pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                         (-np.log(10000.0) / d_model))
    
    pe = torch.zeros(max_seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Xavier initialization
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize with Xavier
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len) causal mask
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q: torch.Tensor = self.W_q(x)
        K: torch.Tensor = self.W_k(x)
        V: torch.Tensor = self.W_v(x)
        
        # Reshape for multi-head: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: (batch, num_heads, seq_len, seq_len)
        # scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Attention output: (batch, num_heads, seq_len, d_k)
        attn_out = attn_weights @ V
        
        # Concatenate heads: (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(attn_out)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        
        # Initialize
        nn.init.normal_(self.W1.weight, std=np.sqrt(2.0 / d_model))
        nn.init.zeros_(self.W1.bias)
        nn.init.normal_(self.W2.weight, std=np.sqrt(2.0 / d_ff))
        nn.init.zeros_(self.W2.bias)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        h = self.W1(x)
        h_act = F.gelu(h)
        out = self.W2(h_act)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """x: (batch, seq_len, d_model)"""
        # Attention with residual
        attn_out = self.attn(x, mask)
        x1 = x + attn_out
        x1_norm = self.ln1(x1)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x1_norm)
        x2 = x1_norm + ffn_out
        x2_norm = self.ln2(x2)
        
        return x2_norm


class GPTModel(nn.Module):
    """GPT-style autoregressive transformer."""
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Positional encoding (fixed, not learned)
        self.register_buffer('pos_embedding', positional_encoding(max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.output_proj.weight, std=0.02)
    
    def create_causal_mask(self, seq_len, device):
        """Upper triangular mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * -1e9
        return mask
    
    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Embed tokens and add positional encoding
        x = self.token_embedding(token_ids)
        x = x + self.pos_embedding[:seq_len]
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, seed_ids, max_new_tokens=20, temperature=1.0):
        """
        Generate tokens autoregressively.
        seed_ids: (1, seq_len) initial tokens
        """
        self.eval()
        device = seed_ids.device
        generated = seed_ids[0].tolist()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Take last max_seq_len tokens
                context = torch.tensor(
                    [generated[-self.max_seq_len:]], 
                    dtype=torch.long, 
                    device=device
                )
                
                # Forward pass
                logits = self.forward(context)
                
                # Get logits for last position
                next_logits = logits[0, -1, :] / temperature
                
                # Sample from distribution
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
        
        return torch.tensor(generated, device=device)
    
    def save(self, filepath):
        """Save model parameters to file."""
        torch.save({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len,
            'num_layers': len(self.blocks),
            'state_dict': self.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device='cpu'):
        """Load model parameters from file."""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with same architecture
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            num_layers=checkpoint['num_layers'],
            num_heads=4,  # Default, not critical for loading
            d_ff=512,     # Default, not critical for loading
            max_seq_len=checkpoint['max_seq_len']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        print(f"Model loaded from {filepath}")
        return model
