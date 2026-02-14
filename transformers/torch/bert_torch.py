"""
BERT-style Transformer implementation using PyTorch.
Encoder-only bidirectional model for masked language modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_torch import positional_encoding, TransformerBlock


class BERTModel(nn.Module):
    """BERT-style bidirectional encoder transformer."""
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        self.register_buffer('pos_embedding', positional_encoding(max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # MLM head
        self.mlm_head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.mlm_head.weight, std=0.02)
    
    def create_padding_mask(self, attention_mask):
        """Convert padding mask (batch, seq) to attention mask (batch, 1, 1, seq)."""
        if attention_mask is None:
            return None
        # Expand and convert: 0 -> -1e9, 1 -> 0
        mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -1e9
        return mask
    
    def forward(self, token_ids, attention_mask=None):
        """
        token_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) optional, 1 for valid, 0 for padding
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        x = self.token_embedding(token_ids) + self.pos_embedding[:seq_len]
        mask = self.create_padding_mask(attention_mask)
        
        for block in self.blocks:
            x = block(x, mask)
        
        return self.mlm_head(x)
    
    def save(self, filepath):
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
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            num_layers=checkpoint['num_layers'],
            max_seq_len=checkpoint['max_seq_len']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"Model loaded from {filepath}")
        return model
